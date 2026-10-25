import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from HybridBiLSTMPOSTagger import HybridBiLSTMPOSTagger


class PosJsonlDataset(Dataset):
    """
    Format per baris JSONL:
    {
      "word_ids": [int, ...],                     # wajib
      "char_ids": [[int, ...], [int, ...], ...],  # wajib, sejajar word_ids
      "tag_ids": [int, ...],                       # wajib untuk train/valid
      "oov_mask": [0/1, ...]                       # opsional (untuk metrik OOV)
    }
    """

    def __init__(self, path: str):
        self.samples: List[Dict] = []
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
        return self.samples[idx]


def collate_fn(batch: List[Dict], pad_word_id: int, pad_char_id: int, pad_tag_id: int) -> Dict[str, torch.Tensor]:
    batch_size = len(batch)
    sentence_lengths = [len(item["word_ids"]) for item in batch]
    max_seq_len = max(sentence_lengths)
    max_char_len = max(max(len(chars) for chars in item["char_ids"]) for item in batch)

    sentence_word_ids = torch.full((batch_size, max_seq_len), pad_word_id, dtype=torch.long)
    sentence_char_ids = torch.full((batch_size, max_seq_len, max_char_len), pad_char_id, dtype=torch.long)
    word_lengths = torch.ones((batch_size, max_seq_len), dtype=torch.long)
    gold_tags = torch.full((batch_size, max_seq_len), pad_tag_id, dtype=torch.long)
    oov_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)

    for i, item in enumerate(batch):
        token_count = len(item["word_ids"])
        sentence_word_ids[i, :token_count] = torch.tensor(item["word_ids"], dtype=torch.long)
        gold_tags[i, :token_count] = torch.tensor(item["tag_ids"], dtype=torch.long)

        if "oov_mask" in item:
            oov_mask[i, :token_count] = torch.tensor(item["oov_mask"], dtype=torch.bool)

        for j, char_list in enumerate(item["char_ids"]):
            char_len = max(1, min(len(char_list), max_char_len))
            word_lengths[i, j] = char_len
            if char_len > 0:
                sentence_char_ids[i, j, :char_len] = torch.tensor(char_list[:char_len], dtype=torch.long)

    return {
        "sentence_word_ids": sentence_word_ids,
        "sentence_char_ids": sentence_char_ids,
        "word_lengths": word_lengths,
        "sentence_lengths": torch.tensor(sentence_lengths, dtype=torch.long),
        "gold_tags": gold_tags,
        "oov_mask": oov_mask,
    }


def move_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def train_one_epoch(
    model: HybridBiLSTMPOSTagger,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    pad_tag_id: int,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        batch = move_to_device(batch, device)
        optimizer.zero_grad()

        logits = model(
            sentence_word_ids=batch["sentence_word_ids"],
            sentence_char_ids=batch["sentence_char_ids"],
            word_lengths=batch["word_lengths"],
            sentence_lengths=batch["sentence_lengths"],
        )
        loss = model.compute_loss(logits, batch["gold_tags"], pad_tag_id=pad_tag_id)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def evaluate(
    model: HybridBiLSTMPOSTagger,
    dataloader: DataLoader,
    pad_tag_id: int,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    total_oov_correct = 0
    total_oov_tokens = 0

    for batch in dataloader:
        batch = move_to_device(batch, device)
        logits = model(
            sentence_word_ids=batch["sentence_word_ids"],
            sentence_char_ids=batch["sentence_char_ids"],
            word_lengths=batch["word_lengths"],
            sentence_lengths=batch["sentence_lengths"],
        )
        loss = model.compute_loss(logits, batch["gold_tags"], pad_tag_id=pad_tag_id)
        total_loss += loss.item()

        pred_tags = logits.argmax(dim=-1)
        non_pad = batch["gold_tags"] != pad_tag_id
        correct = (pred_tags == batch["gold_tags"]) & non_pad
        total_correct += correct.sum().item()
        total_tokens += non_pad.sum().item()

        oov_valid = batch["oov_mask"] & non_pad
        if oov_valid.any():
            total_oov_tokens += oov_valid.sum().item()
            total_oov_correct += ((pred_tags == batch["gold_tags"]) & oov_valid).sum().item()

    result = {
        "loss": total_loss / max(len(dataloader), 1),
        "accuracy": total_correct / max(total_tokens, 1),
    }
    if total_oov_tokens > 0:
        result["oov_accuracy"] = total_oov_correct / total_oov_tokens
    return result


def load_pretrained_embeddings(path: Optional[str]) -> Optional[torch.Tensor]:
    if path is None:
        return None
    tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Pretrained embeddings file must contain a torch.Tensor")
    return tensor


def save_checkpoint(path: Path, model: HybridBiLSTMPOSTagger, optimizer: torch.optim.Optimizer, epoch: int, metrics: Dict[str, float]) -> None:
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
    parser = argparse.ArgumentParser(description="Train Hybrid BiLSTM POS Tagger")
    parser.add_argument("--train-jsonl", required=True, help="Path dataset train JSONL")
    parser.add_argument("--valid-jsonl", required=True, help="Path dataset valid JSONL")
    parser.add_argument("--save-dir", default="checkpoints", help="Folder output checkpoint")

    parser.add_argument("--num-chars", type=int, required=True)
    parser.add_argument("--char-emb-dim", type=int, default=64)
    parser.add_argument("--char-hidden-dim", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--word-emb-dim", type=int, default=300)
    parser.add_argument("--word-hidden-dim", type=int, default=256)
    parser.add_argument("--num-classes", type=int, required=True)

    parser.add_argument("--pad-word-id", type=int, default=0)
    parser.add_argument("--pad-char-id", type=int, default=0)
    parser.add_argument("--pad-tag-id", type=int, default=-100)

    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pretrained-emb-path", default=None, help="Path .pt tensor [vocab_size, word_emb_dim]")
    parser.add_argument("--freeze-pretrained-word-embeddings", action="store_true")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    return parser


def main():
    args = build_arg_parser().parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    train_ds = PosJsonlDataset(args.train_jsonl)
    valid_ds = PosJsonlDataset(args.valid_jsonl)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_fn(batch, args.pad_word_id, args.pad_char_id, args.pad_tag_id),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_fn(batch, args.pad_word_id, args.pad_char_id, args.pad_tag_id),
    )

    pretrained = load_pretrained_embeddings(args.pretrained_emb_path)
    model = HybridBiLSTMPOSTagger(
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

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            pad_tag_id=args.pad_tag_id,
            device=device,
            grad_clip=args.grad_clip,
        )
        val_metrics = evaluate(
            model=model,
            dataloader=valid_loader,
            pad_tag_id=args.pad_tag_id,
            device=device,
        )

        log_msg = (
            f"[EPOCH {epoch:03d}] "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_acc={val_metrics['accuracy']:.6f}"
        )
        if "oov_accuracy" in val_metrics:
            log_msg += f" val_oov_acc={val_metrics['oov_accuracy']:.6f}"
        print(log_msg)

        last_path = save_dir / "last.pt"
        save_checkpoint(last_path, model, optimizer, epoch, val_metrics)

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_path = save_dir / "best.pt"
            save_checkpoint(best_path, model, optimizer, epoch, val_metrics)
            print(f"[INFO] New best checkpoint: {best_path}")

    print("[INFO] Training selesai.")


if __name__ == "__main__":
    main()
