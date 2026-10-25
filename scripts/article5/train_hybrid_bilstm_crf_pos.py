import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from HybridBiLSTMCRFPOSTagger import HybridBiLSTMCRFPOSTagger


class PosJsonlDataset(Dataset):
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
        loss = model.compute_loss(emissions, batch["gold_tags"], mask=batch["token_mask"])
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
) -> Set[int]:
    tag_names = [x.strip() for x in critical_tags_csv.split(",") if x.strip()]
    if not tag_names:
        return set()
    if not tag_vocab:
        raise ValueError("critical tags requested but tag-vocab-json is not provided")
    missing = [name for name in tag_names if name not in tag_vocab]
    if missing:
        raise ValueError(f"Unknown critical tags: {missing}")
    return {tag_vocab[name] for name in tag_names}


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
    parser.add_argument("--metrics-jsonl", default=None)
    return parser


def main() -> None:
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
        collate_fn=lambda b: collate_fn(b, args.pad_word_id, args.pad_char_id),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(b, args.pad_word_id, args.pad_char_id),
    )

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

    tag_vocab = load_tag_vocab(args.tag_vocab_json)
    critical_tag_ids = parse_critical_tag_ids(args.critical_tags, tag_vocab) if args.critical_tags else set()

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
