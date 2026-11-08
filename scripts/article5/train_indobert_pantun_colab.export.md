# Export Notebook: train_indobert_pantun_colab.ipynb

Sumber: `/mnt/d/workspace/penelitian/Disertasi/coding/rima/scripts/article5/train_indobert_pantun_colab.ipynb`

## Cell 1 (markdown)

# IndoBERT Token Classification untuk Pantun

Notebook ini melatih model token classification berbasis IndoBERT pada file `pantun-anotated.jsonl` untuk mendeteksi token seperti singkatan, disemvowelling, dan bahasa asing sesuai label anotasi.

## Cell 2 (code)

```python
!pip -q install -U "pyarrow>=19,<24" transformers datasets seqeval accelerate evaluate scikit-learn
!git --version
```

## Cell 3 (code)

```python
import os
import json
import random
import numpy as np
import torch
import evaluate
from collections import Counter
from pathlib import Path
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print('CUDA available:', torch.cuda.is_available())
```

## Cell 4 (code)

```python
from google.colab import drive
from pathlib import Path
import subprocess

drive.mount('/content/drive')

REPO_URL = 'https://github.com/NazirArifin/pantunify.git'
WORKSPACE_DIR = Path('/content/drive/MyDrive/NERPantun')
REPO_DIR = WORKSPACE_DIR / 'pantunify'

WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

if REPO_DIR.exists() and (REPO_DIR / '.git').exists():
    print('Repo sudah ada, melakukan git pull...')
    subprocess.run(['git', '-C', str(REPO_DIR), 'pull'], check=True)
else:
    if REPO_DIR.exists():
        raise RuntimeError(f'{REPO_DIR} sudah ada tapi bukan repo git. Rename/hapus dulu lalu jalankan ulang.')
    print('Cloning repo ke folder induk NERPantun...')
    subprocess.run(['git', 'clone', REPO_URL, str(REPO_DIR)], check=True)

print('WORKSPACE_DIR =', WORKSPACE_DIR)
print('REPO_DIR =', REPO_DIR)

# --- UBAH BAGIAN INI ---
# Mengarahkan langsung ke subfolder /data/article5/ di dalam repository
data_path = REPO_DIR / 'data' / 'article5' / 'pantun-anotated.jsonl'
print('data_path =', data_path)
```

## Cell 5 (code)

```python
# 3) Validasi format dan panjang tokens/ner_tags
STRICT_VALIDATION = False  # ubah ke True jika ingin hard fail saat mismatch

# Safety guard: jika cell dijalankan tidak urut dan raw_objects belum ada, bangun ulang dari data_path
if 'raw_objects' not in globals():
    if 'data_path' not in globals():
        raise RuntimeError('data_path belum ada. Jalankan dulu cell setup path data (mount drive + load data).')

    def parse_concatenated_json_objects(path: Path):
        text = path.read_text(encoding='utf-8')
        decoder = json.JSONDecoder()
        i, n = 0, len(text)
        objects = []
        while i < n:
            while i < n and text[i].isspace():
                i += 1
            if i >= n:
                break
            obj, j = decoder.raw_decode(text, i)
            i = j
            objects.append(obj)
        return objects

    raw_objects = parse_concatenated_json_objects(Path(data_path))
    print('raw_objects dibuat ulang:', len(raw_objects))

examples = []
mismatch_records = []

for idx, obj in enumerate(raw_objects):
    tokens = obj.get('tokens')
    tags = obj.get('ner_tags')

    if not isinstance(tokens, list) or not isinstance(tags, list):
        raise ValueError(f'Object #{idx} tidak memiliki tokens/ner_tags list')

    if len(tokens) != len(tags):
        if STRICT_VALIDATION:
            raise ValueError(f'Length mismatch di object #{idx}: tokens={len(tokens)} tags={len(tags)}')

        mismatch_records.append((idx, len(tokens), len(tags)))

        if len(tags) < len(tokens):
            tags = tags + ['O'] * (len(tokens) - len(tags))
        else:
            tags = tags[:len(tokens)]

    examples.append({'id': idx, 'tokens': tokens, 'ner_tags': tags})

print('Valid examples:', len(examples))
print('Mismatched examples fixed:', len(mismatch_records))
if mismatch_records:
    print('Contoh mismatch (max 10):', mismatch_records[:10])
```

## Cell 6 (code)

```python
# Label mapping otomatis
label_set = sorted({tag for ex in examples for tag in ex['ner_tags']})
label2id = {label: i for i, label in enumerate(label_set)}
id2label = {i: label for label, i in label2id.items()}
print('Labels:', label_set)

# Stratified split berdasarkan ada/tidaknya label non-O
y_strat = [int(any(tag != 'O' for tag in ex['ner_tags'])) for ex in examples]
train_idx, val_idx = train_test_split(
    np.arange(len(examples)),
    test_size=0.2,
    random_state=SEED,
    stratify=y_strat,
)

train_data = [examples[i] for i in train_idx]
val_data = [examples[i] for i in val_idx]

dataset = DatasetDict({
    'train': Dataset.from_list(train_data),
    'validation': Dataset.from_list(val_data),
})

print(dataset)

# Hitung class weight dari distribusi label train (untuk dorong recall label minoritas)
train_label_counter = Counter()
for ex in train_data:
    train_label_counter.update(ex['ner_tags'])

total_labels = sum(train_label_counter.values())
raw_weights = {}
for lbl in label_set:
    # inverse frequency + smoothing agar stabil
    raw_weights[lbl] = total_labels / max(1, train_label_counter[lbl])

# Normalkan agar rata-rata weight ~ 1
mean_w = sum(raw_weights.values()) / len(raw_weights)
class_weights = torch.tensor([raw_weights[lbl] / mean_w for lbl in label_set], dtype=torch.float)

print('Train label counts:', dict(train_label_counter))
print('Class weights:', {lbl: round(float(class_weights[label2id[lbl]]), 4) for lbl in label_set})
```

## Cell 7 (code)

```python
model_name = 'indobenchmark/indobert-base-p1'
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_and_align_labels(batch):
    tokenized = tokenizer(
        batch['tokens'],
        truncation=True,
        is_split_into_words=True,
        max_length=128,
    )

    aligned_labels = []
    for i, labels in enumerate(batch['ner_tags']):
        word_ids = tokenized.word_ids(batch_index=i)
        prev_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != prev_word_idx:
                label_ids.append(label2id[labels[word_idx]])
            else:
                label_ids.append(-100)
            prev_word_idx = word_idx
        aligned_labels.append(label_ids)

    tokenized['labels'] = aligned_labels
    return tokenized

tokenized_ds = dataset.map(tokenize_and_align_labels, batched=True)

model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label_set),
    id2label=id2label,
    label2id=label2id,
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
seqeval = evaluate.load('seqeval')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=2)

    true_predictions = []
    true_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        pred_tags = []
        gold_tags = []
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                pred_tags.append(id2label[int(p)])
                gold_tags.append(id2label[int(l)])
        true_predictions.append(pred_tags)
        true_labels.append(gold_tags)

    metrics = seqeval.compute(predictions=true_predictions, references=true_labels, zero_division=0)
    return {
        'precision': metrics['overall_precision'],
        'recall': metrics['overall_recall'],
        'f1': metrics['overall_f1'],
        'accuracy': metrics['overall_accuracy'],
    }
```

## Cell 8 (code)

```python
output_dir = str(WORKSPACE_DIR / 'models/indobert-pantun-ner')
Path(output_dir).mkdir(parents=True, exist_ok=True)

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get('labels')
        outputs = model(**inputs)
        logits = outputs.get('logits')

        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device),
            ignore_index=-100
        )
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=15,
    weight_decay=0.01,
    eval_strategy='epoch',
    save_strategy='epoch',
    logging_strategy='steps',
    logging_steps=20,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    report_to='none',
    seed=SEED,
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds['train'],
    eval_dataset=tokenized_ds['validation'],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    class_weights=class_weights,
)

train_result = trainer.train()
print(train_result)

eval_result = trainer.evaluate()
print('Eval:', eval_result)
```

## Cell 9 (code)

```python
def predict_tokens(tokens):
    enc = tokenizer(tokens, is_split_into_words=True, return_tensors='pt', truncation=True, max_length=128)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    pred_ids = logits.argmax(dim=-1)[0].cpu().numpy().tolist()
    word_ids = tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=128).word_ids()

    results = []
    prev = None
    for i, widx in enumerate(word_ids):
        if widx is None or widx == prev:
            prev = widx
            continue
        results.append((tokens[widx], id2label[pred_ids[i]]))
        prev = widx
    return results

def evaluate_with_label_breakdown(trainer, tokenized_dataset):
    pred_output = trainer.predict(tokenized_dataset['validation'])
    logits, labels = pred_output.predictions, pred_output.label_ids
    pred_ids = np.argmax(logits, axis=2)

    true_predictions = []
    true_labels = []

    for pred_seq, label_seq in zip(pred_ids, labels):
        pred_tags = []
        gold_tags = []
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                pred_tags.append(id2label[int(p)])
                gold_tags.append(id2label[int(l)])
        true_predictions.append(pred_tags)
        true_labels.append(gold_tags)

    detailed = seqeval.compute(
        predictions=true_predictions,
        references=true_labels,
        zero_division=0
    )

    print('Overall metrics:', {
        'precision': detailed['overall_precision'],
        'recall': detailed['overall_recall'],
        'f1': detailed['overall_f1'],
        'accuracy': detailed['overall_accuracy'],
    })

    print('\nPer-label metrics:')
    for lbl in label_set:
        if lbl == 'O':
            continue
        if lbl in detailed:
            m = detailed[lbl]
            print(f"{lbl}: P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} support={m['number']}")

sample_tokens = ["yang", "pake", "nama", "samaran", "kemarin", "bajunya", "kena", "tinta", "kamu", "tuh", "jangan", "baperan", "status", "di", "like", "langsung", "di", "kira", "cinta"]
print('Sample prediction:', predict_tokens(sample_tokens))

evaluate_with_label_breakdown(trainer, tokenized_ds)
```

## Cell 10 (code)

```python
# Jalankan cell ini hanya saat ingin menyimpan model
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print('Saved model dir:', output_dir)
```

