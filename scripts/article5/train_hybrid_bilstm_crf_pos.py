import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from HybridBiLSTMCRFPOSTagger import HybridBiLSTMCRFPOSTagger


class PosJsonlDataset(Dataset):
    def __init__(
        self,
        path: str,
        disemvowel_tag_ids: Optional[Set[int]] = None,
        disemvowel_prob: float = 0.0,
        vowel_char_ids: Optional[Set[int]] = None,
        unk_char_id: Optional[int] = None,
        unk_word_id: Optional[int] = None,
        disemvowel_word_id_map: Optional[Dict[int, int]] = None,
    ):
        self.samples: List[Dict] = []
        self.disemvowel_tag_ids = set(disemvowel_tag_ids or set())
        self.disemvowel_prob = float(disemvowel_prob)
        self.vowel_char_ids = set(vowel_char_ids or set())
        self.unk_char_id = unk_char_id
        self.unk_word_id = unk_word_id
        self.disemvowel_word_id_map = dict(disemvowel_word_id_map or {})
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                for key in ("word_ids", "char_ids", "tag_ids"):
                    if key not in obj:
                        raise ValueError(f"Missing key '{key}' at {path}:{line_idx}")
                if len(obj["word_ids"]) != len(obj["char_ids"]) or len(obj["word_ids"]) != len(obj["tag_ids"]):
                    raise ValueError(f"Length mismatch at {path}:{line_idx}")
                if "oov_mask" in obj and len(obj["oov_mask"]) != len(obj["word_ids"]):
                    raise ValueError(f"oov_mask length mismatch at {path}:{line_idx}")
                self.samples.append(obj)
        if not self.samples:
            raise ValueError(f"Dataset kosong: {path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if (
            not self.disemvowel_tag_ids
            or self.disemvowel_prob <= 0.0
            or not self.vowel_char_ids
            or "tag_ids" not in sample
        ):
            return sample

        augmented = {
            "word_ids": list(sample["word_ids"]),
            "char_ids": [list(chars) for chars in sample["char_ids"]],
            "tag_ids": list(sample["tag_ids"]),
        }
        if "oov_mask" in sample:
            augmented["oov_mask"] = list(sample["oov_mask"])

        for token_idx, tag_id in enumerate(augmented["tag_ids"]):
            if int(tag_id) not in self.disemvowel_tag_ids:
                continue
            if torch.rand(1).item() >= self.disemvowel_prob:
                continue

            original_chars = augmented["char_ids"][token_idx]
            disemvowel_chars = [cid for cid in original_chars if int(cid) not in self.vowel_char_ids]
            if not disemvowel_chars:
                if self.unk_char_id is not None:
                    disemvowel_chars = [self.unk_char_id]
                elif original_chars:
                    disemvowel_chars = [original_chars[0]]
                else:
                    continue
            augmented["char_ids"][token_idx] = disemvowel_chars

            current_word_id = int(augmented["word_ids"][token_idx])
            replacement_word_id = self.disemvowel_word_id_map.get(current_word_id)
            if replacement_word_id is not None:
                augmented["word_ids"][token_idx] = replacement_word_id
                if "oov_mask" in augmented and self.unk_word_id is not None:
                    augmented["oov_mask"][token_idx] = 1 if replacement_word_id == self.unk_word_id else 0

        return augmented


def collate_fn(batch: List[Dict], pad_word_id: int, pad_char_id: int) -> Dict[str, torch.Tensor]:
    batch_size = len(batch)
    sentence_lengths = [len(item["word_ids"]) for item in batch]
    max_seq_len = max(sentence_lengths)
    max_char_len = max(max(len(chars) for chars in item["char_ids"]) for item in batch)

    sentence_word_ids = torch.full((batch_size, max_seq_len), pad_word_id, dtype=torch.long)
    sentence_char_ids = torch.full((batch_size, max_seq_len, max_char_len), pad_char_id, dtype=torch.long)
    word_lengths = torch.ones((batch_size, max_seq_len), dtype=torch.long)
    gold_tags = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    oov_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
    token_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)

    for i, item in enumerate(batch):
        token_count = len(item["word_ids"])
        sentence_word_ids[i, :token_count] = torch.tensor(item["word_ids"], dtype=torch.long)
        gold_tags[i, :token_count] = torch.tensor(item["tag_ids"], dtype=torch.long)
        token_mask[i, :token_count] = True

        if "oov_mask" in item:
            oov_mask[i, :token_count] = torch.tensor(item["oov_mask"], dtype=torch.bool)

        for j, char_list in enumerate(item["char_ids"]):
            char_len = max(1, min(len(char_list), max_char_len))
            word_lengths[i, j] = char_len
            sentence_char_ids[i, j, :char_len] = torch.tensor(char_list[:char_len], dtype=torch.long)

    return {
        "sentence_word_ids": sentence_word_ids,
        "sentence_char_ids": sentence_char_ids,
        "word_lengths": word_lengths,
        "sentence_lengths": torch.tensor(sentence_lengths, dtype=torch.long),
        "gold_tags": gold_tags,
        "token_mask": token_mask,
        "oov_mask": oov_mask,
    }


def move_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def train_one_epoch(
    model: HybridBiLSTMCRFPOSTagger,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    aux_ce_loss_weight: float = 0.0,
    class_weights: Optional[torch.Tensor] = None,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        batch = move_to_device(batch, device)
        optimizer.zero_grad()
        emissions = model(
            sentence_word_ids=batch["sentence_word_ids"],
            sentence_char_ids=batch["sentence_char_ids"],
            word_lengths=batch["word_lengths"],
            sentence_lengths=batch["sentence_lengths"],
        )
        crf_loss = model.compute_loss(emissions, batch["gold_tags"], mask=batch["token_mask"])
        loss = crf_loss
        if aux_ce_loss_weight > 0.0:
            valid = batch["token_mask"]
            logits = emissions[valid]
            targets = batch["gold_tags"][valid]
            ce_loss = F.cross_entropy(logits, targets, weight=class_weights)
            loss = crf_loss + (aux_ce_loss_weight * ce_loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(dataloader), 1)


def load_tag_vocab(path: Optional[str]) -> Dict[str, int]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if "tag_vocab" not in obj or not isinstance(obj["tag_vocab"], dict):
        raise ValueError("tag-vocab-json must contain object key 'tag_vocab'")
    out: Dict[str, int] = {}
    for key, value in obj["tag_vocab"].items():
        if isinstance(value, int):
            out[key] = value
        else:
            raise ValueError(f"tag_vocab[{key!r}] must be int, got {type(value).__name__}")
    return out


def parse_critical_tag_ids(
    critical_tags_csv: str,
    tag_vocab: Dict[str, int],
    strict: bool = True,
) -> Set[int]:
    tag_names = [x.strip() for x in critical_tags_csv.split(",") if x.strip()]
    if not tag_names:
        return set()
    if not tag_vocab:
        if strict:
            raise ValueError("critical tags requested but tag-vocab-json is not provided")
        return set()
    missing = [name for name in tag_names if name not in tag_vocab]
    if missing:
        raise ValueError(f"Unknown critical tags: {missing}")
    return {tag_vocab[name] for name in tag_names}


def load_vocab_payload(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Vocab JSON harus berupa object: {path}")
    return obj


def parse_vocab_mapping(
    payload: Dict[str, Any],
    key: str,
    value_kind: str,
) -> Dict[str, int]:
    if key not in payload:
        return {}
    raw = payload[key]
    if not isinstance(raw, dict):
        raise ValueError(f"{key} harus berupa object pada vocab JSON")
    out: Dict[str, int] = {}
    for name, idx in raw.items():
        if not isinstance(name, str):
            raise ValueError(f"{key} memiliki key non-string: {name!r}")
        if not isinstance(idx, int):
            raise ValueError(f"{key}[{name!r}] harus int, got {type(idx).__name__}")
        out[name] = idx
    if value_kind == "char":
        for ch in out:
            if len(ch) != 1 and ch not in {"<PAD>", "<UNK>"}:
                raise ValueError(f"{key} berisi token bukan karakter tunggal: {ch!r}")
    return out


def build_disemvowel_word_id_map(
    word_vocab: Dict[str, int],
    vowels: Set[str],
    unk_word_id: Optional[int],
    allow_unk_rewrite: bool,
) -> Dict[int, int]:
    if not word_vocab:
        return {}
    disemvowel_map: Dict[int, int] = {}
    for word, word_id in word_vocab.items():
        if word.startswith("<") and word.endswith(">"):
            continue
        disemvowelled = "".join(ch for ch in word if ch not in vowels)
        if disemvowelled == word:
            continue
        replacement = word_vocab.get(disemvowelled) if disemvowelled else None
        if replacement is None and allow_unk_rewrite:
            replacement = unk_word_id
        if replacement is not None:
            disemvowel_map[word_id] = replacement
    return disemvowel_map


def compute_macro_f1_from_confusion(
    confusion: torch.Tensor,
    class_ids: Sequence[int],
) -> float:
    if confusion.dim() != 2 or confusion.size(0) != confusion.size(1):
        raise ValueError("confusion must be square matrix")
    f1_values: List[float] = []
    for class_id in class_ids:
        tp = confusion[class_id, class_id].item()
        fp = confusion[:, class_id].sum().item() - tp
        fn = confusion[class_id, :].sum().item() - tp
        denom = (2 * tp) + fp + fn
        f1 = (2 * tp) / denom if denom > 0 else 0.0
        f1_values.append(f1)
    if not f1_values:
        return 0.0
    return sum(f1_values) / len(f1_values)


def compute_selection_score(
    metrics: Dict[str, float],
    weight_acc: float,
    weight_oov_acc: float,
    weight_critical_f1: float,
) -> float:
    return (
        weight_acc * metrics.get("accuracy", 0.0)
        + weight_oov_acc * metrics.get("oov_accuracy", 0.0)
        + weight_critical_f1 * metrics.get("critical_macro_f1", 0.0)
    )


def append_metrics_jsonl(path: Path, record: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_class_weights(
    num_classes: int,
    critical_tag_ids: Set[int],
    non_critical_weight: float,
    critical_weight: float,
) -> torch.Tensor:
    weights = torch.full((num_classes,), float(non_critical_weight), dtype=torch.float)
    for class_id in critical_tag_ids:
        if class_id < 0 or class_id >= num_classes:
            raise ValueError(f"critical class id out of range: {class_id} for num_classes={num_classes}")
        weights[class_id] = float(critical_weight)
    return weights


def compute_sentence_sampling_weight(
    sample: Dict,
    critical_tag_ids: Set[int],
    oov_alpha: float,
    critical_alpha: float,
) -> float:
    token_count = len(sample["word_ids"])
    if token_count <= 0:
        return 1.0
    oov_mask = sample.get("oov_mask")
    oov_ratio = 0.0
    if isinstance(oov_mask, list) and len(oov_mask) == token_count:
        oov_ratio = sum(1 for x in oov_mask if int(x) != 0) / token_count
    critical_ratio = 0.0
    if critical_tag_ids:
        tag_ids = sample.get("tag_ids", [])
        if isinstance(tag_ids, list) and len(tag_ids) == token_count:
            critical_ratio = sum(1 for x in tag_ids if int(x) in critical_tag_ids) / token_count
    return 1.0 + (oov_alpha * oov_ratio) + (critical_alpha * critical_ratio)


def build_weighted_sampler(
    dataset: PosJsonlDataset,
    critical_tag_ids: Set[int],
    oov_alpha: float,
    critical_alpha: float,
    replacement: bool,
    num_samples: int,
) -> WeightedRandomSampler:
    weights = [
        compute_sentence_sampling_weight(
            sample=sample,
            critical_tag_ids=critical_tag_ids,
            oov_alpha=oov_alpha,
            critical_alpha=critical_alpha,
        )
        for sample in dataset.samples
    ]
    weight_tensor = torch.tensor(weights, dtype=torch.double)
    if weight_tensor.numel() == 0:
        raise ValueError("Cannot build sampler for empty dataset")
    resolved_num_samples = num_samples if num_samples > 0 else len(dataset)
    return WeightedRandomSampler(
        weights=weight_tensor,
        num_samples=resolved_num_samples,
        replacement=replacement,
    )


@torch.no_grad()
def evaluate(
    model: HybridBiLSTMCRFPOSTagger,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    critical_tag_ids: Optional[Set[int]] = None,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    total_oov_correct = 0
    total_oov_tokens = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)

    for batch in dataloader:
        batch = move_to_device(batch, device)
        emissions = model(
            sentence_word_ids=batch["sentence_word_ids"],
            sentence_char_ids=batch["sentence_char_ids"],
            word_lengths=batch["word_lengths"],
            sentence_lengths=batch["sentence_lengths"],
        )
        loss = model.compute_loss(emissions, batch["gold_tags"], mask=batch["token_mask"])
        total_loss += loss.item()

        paths = model.decode(emissions, mask=batch["token_mask"])
        pred = torch.zeros_like(batch["gold_tags"])
        for i, seq in enumerate(paths):
            if seq:
                pred[i, : len(seq)] = torch.tensor(seq, dtype=pred.dtype, device=pred.device)

        valid = batch["token_mask"]
        correct = (pred == batch["gold_tags"]) & valid
        total_correct += correct.sum().item()
        total_tokens += valid.sum().item()

        valid_pred = pred[valid].view(-1).cpu()
        valid_gold = batch["gold_tags"][valid].view(-1).cpu()
        for gold_tag, pred_tag in zip(valid_gold.tolist(), valid_pred.tolist()):
            confusion[gold_tag, pred_tag] += 1

        oov_valid = batch["oov_mask"] & valid
        if oov_valid.any():
            total_oov_tokens += oov_valid.sum().item()
            total_oov_correct += ((pred == batch["gold_tags"]) & oov_valid).sum().item()

    result = {
        "loss": total_loss / max(len(dataloader), 1),
        "accuracy": total_correct / max(total_tokens, 1),
    }
    if total_oov_tokens > 0:
        result["oov_accuracy"] = total_oov_correct / total_oov_tokens
    if critical_tag_ids:
        result["critical_macro_f1"] = compute_macro_f1_from_confusion(
            confusion=confusion,
            class_ids=sorted(critical_tag_ids),
        )
    return result


def load_pretrained_embeddings(path: Optional[str]) -> Optional[torch.Tensor]:
    if path is None:
        return None
    tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Pretrained embeddings file must contain a torch.Tensor")
    return tensor


def save_checkpoint(
    path: Path,
    model: HybridBiLSTMCRFPOSTagger,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Hybrid BiLSTM-CRF POS Tagger")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--valid-jsonl", required=True)
    parser.add_argument("--save-dir", default="checkpoints_crf")

    parser.add_argument("--num-chars", type=int, required=True)
    parser.add_argument("--char-emb-dim", type=int, default=64)
    parser.add_argument("--char-hidden-dim", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--word-emb-dim", type=int, default=300)
    parser.add_argument("--word-hidden-dim", type=int, default=256)
    parser.add_argument("--num-classes", type=int, required=True)

    parser.add_argument("--pad-word-id", type=int, default=0)
    parser.add_argument("--pad-char-id", type=int, default=0)

    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pretrained-emb-path", default=None)
    parser.add_argument("--freeze-pretrained-word-embeddings", action="store_true")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-crf", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--best-metric", choices=("accuracy", "composite"), default="accuracy")
    parser.add_argument("--score-weight-acc", type=float, default=1.0)
    parser.add_argument("--score-weight-oov-acc", type=float, default=0.0)
    parser.add_argument("--score-weight-critical-f1", type=float, default=0.0)
    parser.add_argument("--critical-tags", default="ADV,PART,NOUN,VERB")
    parser.add_argument("--tag-vocab-json", default=None)
    parser.add_argument("--vocab-json", default=None, help="JSON vocab dari convert_conllu_to_jsonl.py")
    parser.add_argument("--metrics-jsonl", default=None)
    parser.add_argument("--aux-ce-loss-weight", type=float, default=0.0)
    parser.add_argument("--class-weighting", choices=("none", "critical"), default="none")
    parser.add_argument("--critical-class-weight", type=float, default=2.0)
    parser.add_argument("--non-critical-class-weight", type=float, default=1.0)
    parser.add_argument("--sampling-strategy", choices=("none", "oov_critical"), default="none")
    parser.add_argument("--sampling-oov-alpha", type=float, default=1.5)
    parser.add_argument("--sampling-critical-alpha", type=float, default=1.0)
    parser.add_argument("--sampling-num-samples", type=int, default=0)
    parser.add_argument("--sampling-no-replacement", action="store_true")
    parser.add_argument("--enable-disemvowel-augmentation", action="store_true")
    parser.add_argument("--disemvowel-tags", default="ADV,PART,PRON,CCONJ,SCONJ")
    parser.add_argument("--disemvowel-prob", type=float, default=0.5)
    parser.add_argument("--disemvowel-vowels", default="aiueoAIUEO")
    parser.add_argument(
        "--disemvowel-allow-unk-word-rewrite",
        action="store_true",
        help="Jika aktif, word_id boleh di-rewrite ke <UNK> saat bentuk disemvowel tidak ada di vocab.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    if args.disemvowel_prob < 0.0 or args.disemvowel_prob > 1.0:
        raise ValueError("--disemvowel-prob harus pada rentang [0.0, 1.0]")

    vocab_payload = load_vocab_payload(args.vocab_json or args.tag_vocab_json)
    word_vocab = parse_vocab_mapping(vocab_payload, "word_vocab", value_kind="word")
    char_vocab = parse_vocab_mapping(vocab_payload, "char_vocab", value_kind="char")
    unk_word_id = word_vocab.get("<UNK>")
    unk_char_id = char_vocab.get("<UNK>")

    tag_vocab = load_tag_vocab(args.tag_vocab_json)
    if not tag_vocab:
        tag_vocab = parse_vocab_mapping(vocab_payload, "tag_vocab", value_kind="tag")

    disemvowel_tag_ids: Set[int] = set()
    disemvowel_vowel_char_ids: Set[int] = set()
    disemvowel_word_id_map: Dict[int, int] = {}
    if args.enable_disemvowel_augmentation:
        if not tag_vocab:
            raise ValueError(
                "augmentasi disemvowel membutuhkan tag vocab. Berikan --tag-vocab-json atau --vocab-json valid."
            )
        if not char_vocab:
            raise ValueError(
                "augmentasi disemvowel membutuhkan char_vocab pada --vocab-json/--tag-vocab-json."
            )
        disemvowel_tag_ids = parse_critical_tag_ids(args.disemvowel_tags, tag_vocab, strict=True)
        if not disemvowel_tag_ids:
            raise ValueError("--disemvowel-tags kosong setelah diparse.")
        disemvowel_vowel_char_ids = {char_vocab[ch] for ch in args.disemvowel_vowels if ch in char_vocab}
        if not disemvowel_vowel_char_ids:
            raise ValueError(
                "Tidak ada karakter vokal pada char_vocab yang cocok dengan --disemvowel-vowels."
            )
        disemvowel_word_id_map = build_disemvowel_word_id_map(
            word_vocab=word_vocab,
            vowels=set(args.disemvowel_vowels),
            unk_word_id=unk_word_id,
            allow_unk_rewrite=args.disemvowel_allow_unk_word_rewrite,
        )
        print(
            "[INFO] Disemvowel augmentation aktif: "
            f"tags={args.disemvowel_tags}, prob={args.disemvowel_prob}, "
            f"vowel_chars_found={len(disemvowel_vowel_char_ids)}, "
            f"word_map_size={len(disemvowel_word_id_map)}, "
            f"allow_unk_word_rewrite={args.disemvowel_allow_unk_word_rewrite}"
        )

    train_ds = PosJsonlDataset(
        args.train_jsonl,
        disemvowel_tag_ids=disemvowel_tag_ids,
        disemvowel_prob=args.disemvowel_prob if args.enable_disemvowel_augmentation else 0.0,
        vowel_char_ids=disemvowel_vowel_char_ids,
        unk_char_id=unk_char_id,
        unk_word_id=unk_word_id,
        disemvowel_word_id_map=disemvowel_word_id_map,
    )
    valid_ds = PosJsonlDataset(args.valid_jsonl)

    pretrained = load_pretrained_embeddings(args.pretrained_emb_path)
    model = HybridBiLSTMCRFPOSTagger(
        num_chars=args.num_chars,
        char_emb_dim=args.char_emb_dim,
        char_hidden_dim=args.char_hidden_dim,
        vocab_size=args.vocab_size,
        word_emb_dim=args.word_emb_dim,
        word_hidden_dim=args.word_hidden_dim,
        num_classes=args.num_classes,
        pretrained_word_embeddings=pretrained,
        char_padding_idx=args.pad_char_id,
        word_padding_idx=args.pad_word_id,
        dropout=args.dropout,
        freeze_pretrained_word_embeddings=args.freeze_pretrained_word_embeddings,
    ).to(device)

    lr_crf = args.lr if args.lr_crf is None else args.lr_crf
    base_params = [
        param for name, param in model.named_parameters() if not name.startswith("crf.")
    ]
    crf_params = [
        param for name, param in model.named_parameters() if name.startswith("crf.")
    ]
    optimizer = AdamW(
        [
            {"params": base_params, "lr": args.lr},
            {"params": crf_params, "lr": lr_crf},
        ],
        weight_decay=args.weight_decay,
    )
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_jsonl_path = Path(args.metrics_jsonl) if args.metrics_jsonl else (save_dir / "metrics.jsonl")

    if args.critical_tags and tag_vocab:
        critical_tag_ids = parse_critical_tag_ids(args.critical_tags, tag_vocab, strict=True)
    else:
        critical_tag_ids = set()
        if args.critical_tags and not tag_vocab:
            print("[WARN] --tag-vocab-json tidak diberikan; critical metrics/weighting berbasis nama tag dinonaktifkan.")

    class_weights: Optional[torch.Tensor] = None
    if args.class_weighting == "critical":
        if not critical_tag_ids:
            raise ValueError("class-weighting=critical memerlukan --tag-vocab-json dan --critical-tags valid")
        class_weights = build_class_weights(
            num_classes=args.num_classes,
            critical_tag_ids=critical_tag_ids,
            non_critical_weight=args.non_critical_class_weight,
            critical_weight=args.critical_class_weight,
        ).to(device)
        print(
            "[INFO] Class weighting aktif: "
            f"critical_weight={args.critical_class_weight}, non_critical_weight={args.non_critical_class_weight}"
        )

    train_sampler = None
    train_shuffle = True
    if args.sampling_strategy == "oov_critical":
        if not critical_tag_ids:
            raise ValueError("sampling-strategy=oov_critical memerlukan --tag-vocab-json dan --critical-tags valid")
        train_sampler = build_weighted_sampler(
            dataset=train_ds,
            critical_tag_ids=critical_tag_ids,
            oov_alpha=args.sampling_oov_alpha,
            critical_alpha=args.sampling_critical_alpha,
            replacement=not args.sampling_no_replacement,
            num_samples=args.sampling_num_samples,
        )
        train_shuffle = False
        print(
            "[INFO] Weighted sampling aktif: "
            f"oov_alpha={args.sampling_oov_alpha}, critical_alpha={args.sampling_critical_alpha}, "
            f"replacement={not args.sampling_no_replacement}, "
            f"num_samples={args.sampling_num_samples if args.sampling_num_samples > 0 else len(train_ds)}"
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(b, args.pad_word_id, args.pad_char_id),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(b, args.pad_word_id, args.pad_char_id),
    )

    if args.best_metric == "composite":
        if args.score_weight_acc == 0.0 and args.score_weight_oov_acc == 0.0 and args.score_weight_critical_f1 == 0.0:
            raise ValueError("At least one score weight must be non-zero when --best-metric=composite")
    best_score = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
            aux_ce_loss_weight=args.aux_ce_loss_weight,
            class_weights=class_weights,
        )
        val_metrics = evaluate(
            model=model,
            dataloader=valid_loader,
            device=device,
            num_classes=args.num_classes,
            critical_tag_ids=critical_tag_ids,
        )
        val_metrics["selection_score"] = compute_selection_score(
            metrics=val_metrics,
            weight_acc=args.score_weight_acc,
            weight_oov_acc=args.score_weight_oov_acc,
            weight_critical_f1=args.score_weight_critical_f1,
        )

        log_msg = (
            f"[EPOCH {epoch:03d}] "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_acc={val_metrics['accuracy']:.6f}"
        )
        if "oov_accuracy" in val_metrics:
            log_msg += f" val_oov_acc={val_metrics['oov_accuracy']:.6f}"
        if "critical_macro_f1" in val_metrics:
            log_msg += f" val_critical_macro_f1={val_metrics['critical_macro_f1']:.6f}"
        log_msg += f" score={val_metrics['selection_score']:.6f}"
        print(log_msg)
        append_metrics_jsonl(
            metrics_jsonl_path,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_metrics": val_metrics,
            },
        )

        save_checkpoint(save_dir / "last.pt", model, optimizer, epoch, val_metrics)
        current_score = (
            val_metrics["accuracy"] if args.best_metric == "accuracy" else val_metrics["selection_score"]
        )
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(save_dir / "best.pt", model, optimizer, epoch, val_metrics)
            print(
                f"[INFO] New best checkpoint: {save_dir / 'best.pt'} "
                f"(metric={args.best_metric}, score={current_score:.6f})"
            )
        else:
            epochs_without_improvement += 1

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(
                "[INFO] Early stopping triggered at epoch "
                f"{epoch} (best_epoch={best_epoch}, best_score={best_score:.6f})."
            )
            break

    print("[INFO] Training selesai.")


if __name__ == "__main__":
    main()
