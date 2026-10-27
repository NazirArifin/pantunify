from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL_DIR = Path("data/article5/models")


@dataclass(frozen=True)
class TokenLabel:
    token: str
    label: str
    score: float


def _validate_model_dir(model_dir: Path) -> None:
    required_files = ("config.json", "tokenizer.json", "tokenizer_config.json")
    missing_required = [name for name in required_files if not (model_dir / name).exists()]
    if missing_required:
        raise FileNotFoundError(
            f"Model directory tidak lengkap: {model_dir}. "
            f"File wajib belum ada: {', '.join(missing_required)}."
        )

    has_weights = any(
        (model_dir / filename).exists()
        for filename in (
            "pytorch_model.bin",
            "model.safetensors",
            "tf_model.h5",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )
    )
    if not has_weights:
        raise FileNotFoundError(
            f"Model weights tidak ditemukan di {model_dir}. "
            "Simpan salah satu file berikut: "
            "pytorch_model.bin / model.safetensors / tf_model.h5 "
            "(atau index file untuk model yang di-shard)."
        )


def _import_runtime_deps() -> tuple[Any, Any, Any]:
    try:
        import torch  # type: ignore
        from transformers import AutoModelForTokenClassification, AutoTokenizer  # type: ignore
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            "Dependensi belum lengkap. Install dulu: "
            "python -m pip install torch transformers"
        ) from err
    return torch, AutoTokenizer, AutoModelForTokenClassification


def load_layer1_model(model_dir: Path):
    _validate_model_dir(model_dir)
    _, AutoTokenizer, AutoModelForTokenClassification = _import_runtime_deps()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
    model.eval()
    return tokenizer, model


def predict_labels(text: str, tokenizer, model, max_length: int = 256) -> list[TokenLabel]:
    torch, _, _ = _import_runtime_deps()
    words = text.strip().split()
    if not words:
        return []

    encoded = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    with torch.no_grad():
        output = model(**encoded)

    probs = torch.softmax(output.logits, dim=-1)
    pred_ids = torch.argmax(output.logits, dim=-1)[0].tolist()
    word_ids = encoded.word_ids(batch_index=0)
    id2label = model.config.id2label
    if not id2label:
        raise ValueError("id2label tidak ditemukan pada config model.")

    results: list[TokenLabel] = []
    seen_word_index = set()
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx in seen_word_index:
            continue
        seen_word_index.add(word_idx)
        pred_id = pred_ids[token_idx]
        label = id2label.get(pred_id, str(pred_id))
        confidence = float(probs[0, token_idx, pred_id].item())
        results.append(TokenLabel(token=words[word_idx], label=label, score=confidence))

    return results


def _print_results(text: str, predictions: list[TokenLabel]) -> None:
    print(f"Input: {text}")
    if not predictions:
        print("Tidak ada token yang bisa diprediksi.")
        return
    for item in predictions:
        print(f"{item.token}\t{item.label}\t{item.score:.4f}")


def _to_json_record(text: str, predictions: list[TokenLabel]) -> str:
    payload = {
        "text": text,
        "tokens": [
            {"token": item.token, "label": item.label, "score": round(item.score, 6)}
            for item in predictions
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Layer-1: IndoBERT token classifier untuk label O/B-TYP/B-SKT/B-DSM/B-ASG."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Folder model lokal hasil fine-tuning IndoBERT.",
    )
    parser.add_argument("--text", type=str, help="Satu kalimat untuk inferensi.")
    parser.add_argument(
        "--input-file",
        type=Path,
        help="File teks; setiap baris non-kosong akan diprediksi.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        help="Simpan hasil inferensi per baris ke JSONL.",
    )
    parser.add_argument("--max-length", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.text and not args.input_file:
        print("Wajib isi --text atau --input-file.", file=sys.stderr)
        return 2

    tokenizer, model = load_layer1_model(args.model_dir)

    lines: list[str] = []
    if args.text:
        lines.append(args.text.strip())
    if args.input_file:
        lines.extend(
            line.strip()
            for line in args.input_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    jsonl_rows: list[str] = []
    for idx, line in enumerate(lines, start=1):
        preds = predict_labels(line, tokenizer=tokenizer, model=model, max_length=args.max_length)
        print(f"\n=== Baris {idx} ===")
        _print_results(line, preds)
        if args.output_jsonl:
            jsonl_rows.append(_to_json_record(line, preds))

    if args.output_jsonl:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.output_jsonl.write_text("\n".join(jsonl_rows) + "\n", encoding="utf-8")
        print(f"\nHasil JSONL tersimpan di: {args.output_jsonl}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, OSError, ValueError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        raise SystemExit(1)
