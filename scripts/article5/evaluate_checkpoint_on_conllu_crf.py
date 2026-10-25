import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from HybridBiLSTMCRFPOSTagger import HybridBiLSTMCRFPOSTagger


TokenTagSentence = List[Tuple[str, str]]


def parse_conllu(path: Path) -> List[TokenTagSentence]:
    sentences: List[TokenTagSentence] = []
    current: TokenTagSentence = []

    with path.open("r", encoding="utf-8") as f:
        for line_idx, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue
            if line.startswith("#"):
                continue

            cols = line.split("\t")
            if len(cols) < 4:
                raise ValueError(f"Invalid CoNLL-U columns at {path}:{line_idx}")

            token_id = cols[0]
            if "-" in token_id or "." in token_id:
                continue

            form = cols[1]
            upos = cols[3]
            if upos == "_":
                continue
            current.append((form, upos))

    if current:
        sentences.append(current)
    return sentences


class EncodedDataset(Dataset):
    def __init__(self, samples: Sequence[Dict]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def collate_fn(batch: List[Dict], pad_word_id: int, pad_char_id: int) -> Dict[str, torch.Tensor]:
    batch_size = len(batch)
    sentence_lengths = [len(item["word_ids"]) for item in batch]
    max_seq_len = max(sentence_lengths)
    max_char_len = max(max(len(chars) for chars in item["char_ids"]) for item in batch)

    sentence_word_ids = torch.full((batch_size, max_seq_len), pad_word_id, dtype=torch.long)
    sentence_char_ids = torch.full((batch_size, max_seq_len, max_char_len), pad_char_id, dtype=torch.long)
    word_lengths = torch.ones((batch_size, max_seq_len), dtype=torch.long)
    gold_tags = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    token_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
    oov_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)

    for i, item in enumerate(batch):
        token_count = len(item["word_ids"])
        sentence_word_ids[i, :token_count] = torch.tensor(item["word_ids"], dtype=torch.long)
        gold_tags[i, :token_count] = torch.tensor(item["tag_ids"], dtype=torch.long)
        token_mask[i, :token_count] = True
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


def encode_sentences(
    sentences: Sequence[TokenTagSentence],
    word_vocab: Dict[str, int],
    char_vocab: Dict[str, int],
    tag_vocab: Dict[str, int],
    lowercase: bool,
) -> List[Dict]:
    unk_word_id = word_vocab["<UNK>"]
    unk_char_id = char_vocab["<UNK>"]
    encoded = []

    for sentence in sentences:
        word_ids = []
        char_ids = []
        tag_ids = []
        oov_mask = []

        for token, upos in sentence:
            token_norm = token.lower() if lowercase else token
            wid = word_vocab.get(token_norm, unk_word_id)
            word_ids.append(wid)
            oov_mask.append(1 if wid == unk_word_id else 0)

            chars = [char_vocab.get(ch, unk_char_id) for ch in token_norm]
            char_ids.append(chars if chars else [unk_char_id])

            if upos not in tag_vocab:
                raise ValueError(f"UPOS '{upos}' tidak ditemukan di tag_vocab.")
            tag_ids.append(tag_vocab[upos])

        encoded.append({"word_ids": word_ids, "char_ids": char_ids, "tag_ids": tag_ids, "oov_mask": oov_mask})
    return encoded


def infer_model_hparams_from_checkpoint(checkpoint: Dict) -> Dict[str, int]:
    state = checkpoint["model_state_dict"]
    num_chars, char_emb_dim = state["char_embeddings.weight"].shape
    vocab_size, word_emb_dim = state["word_embeddings.weight"].shape
    num_classes, two_word_hidden = state["emission.weight"].shape
    word_hidden_dim = two_word_hidden // 2
    char_hidden_dim = state["char_lstm.weight_ih_l0"].shape[0] // 4
    return {
        "num_chars": int(num_chars),
        "char_emb_dim": int(char_emb_dim),
        "char_hidden_dim": int(char_hidden_dim),
        "vocab_size": int(vocab_size),
        "word_emb_dim": int(word_emb_dim),
        "word_hidden_dim": int(word_hidden_dim),
        "num_classes": int(num_classes),
    }


@torch.no_grad()
def evaluate(model: HybridBiLSTMCRFPOSTagger, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    total_oov_correct = 0
    total_oov_tokens = 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
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

        oov_valid = batch["oov_mask"] & valid
        if oov_valid.any():
            total_oov_tokens += oov_valid.sum().item()
            total_oov_correct += ((pred == batch["gold_tags"]) & oov_valid).sum().item()

    metrics = {
        "loss": total_loss / max(len(dataloader), 1),
        "accuracy": total_correct / max(total_tokens, 1),
        "tokens": float(total_tokens),
    }
    if total_oov_tokens > 0:
        metrics["oov_accuracy"] = total_oov_correct / total_oov_tokens
        metrics["oov_tokens"] = float(total_oov_tokens)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HybridBiLSTMCRFPOSTagger checkpoint on CoNLL-U split")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab-json", required=True)
    parser.add_argument("--conllu", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lowercase", action="store_true")
    args = parser.parse_args()

    with open(args.vocab_json, "r", encoding="utf-8") as f:
        vocab_payload = json.load(f)
    word_vocab = vocab_payload["word_vocab"]
    char_vocab = vocab_payload["char_vocab"]
    tag_vocab = vocab_payload["tag_vocab"]

    sentences = parse_conllu(Path(args.conllu))
    encoded = encode_sentences(sentences, word_vocab, char_vocab, tag_vocab, lowercase=args.lowercase)
    dataset = EncodedDataset(encoded)

    pad_word_id = word_vocab["<PAD>"]
    pad_char_id = char_vocab["<PAD>"]
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_word_id, pad_char_id),
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    hparams = infer_model_hparams_from_checkpoint(checkpoint)
    model = HybridBiLSTMCRFPOSTagger(
        num_chars=hparams["num_chars"],
        char_emb_dim=hparams["char_emb_dim"],
        char_hidden_dim=hparams["char_hidden_dim"],
        vocab_size=hparams["vocab_size"],
        word_emb_dim=hparams["word_emb_dim"],
        word_hidden_dim=hparams["word_hidden_dim"],
        num_classes=hparams["num_classes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    metrics = evaluate(model, dataloader, device)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
