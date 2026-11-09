#!/usr/bin/env python3
"""Train/test CRF layer-3 for BI syllabification from phoneme unit sequences.

Input CSV is expected to contain at least:
- record_id
- token_id
- units
- manual_bi_final

Example:
python3 scripts/article5/train_crf_layer3_bi.py \
  --data data/article5/manual_bi_random_sample.csv \
  --model-out data/article5/crf_layer3_bi.joblib
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json

import joblib
import sklearn_crfsuite
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn_crfsuite import metrics

DEFAULT_DATA = Path("data/article5/manual_bi_random_sample.csv")
DEFAULT_MODEL_OUT = Path("data/article5/crf_layer3_bi.joblib")
DEFAULT_CONFIG_OUT = Path("data/article5/crf_layer3_bi_train_config.json")

IPA_VOWELS = {
    "a",
    "i",
    "u",
    "e",
    "o",
    "ə",
    "ɛ",
    "ɪ",
    "ʊ",
    "ɔ",
    "æ",
    "ɐ",
    "ɒ",
}


@dataclass(frozen=True)
class Sample:
    record_id: str
    token_id: int
    units: list[str]
    labels: list[str]


def is_vowel(phone: str) -> bool:
    return phone in IPA_VOWELS


def phone_features(seq: list[str], i: int) -> dict[str, object]:
    cur = seq[i]
    prev = seq[i - 1] if i > 0 else "<BOS>"
    nxt = seq[i + 1] if i < len(seq) - 1 else "<EOS>"
    return {
        "bias": 1.0,
        "phone": cur,
        "prev_phone": prev,
        "next_phone": nxt,
        "bigram_prev_cur": f"{prev}+{cur}",
        "bigram_cur_next": f"{cur}+{nxt}",
        "is_vowel": is_vowel(cur),
        "prev_is_vowel": False if prev == "<BOS>" else is_vowel(prev),
        "next_is_vowel": False if nxt == "<EOS>" else is_vowel(nxt),
        "pos": i,
        "rev_pos": len(seq) - 1 - i,
        "len_seq": len(seq),
        "bos": i == 0,
        "eos": i == len(seq) - 1,
    }


def parse_samples(csv_path: Path) -> list[Sample]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    samples: list[Sample] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {"record_id", "token_id", "units", "manual_bi_final"}
        missing = required_cols.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in CSV: {sorted(missing)}")

        for row_idx, row in enumerate(reader, start=2):
            units = str(row["units"]).strip().split()
            labels_raw = str(row["manual_bi_final"]).strip()
            labels = list(labels_raw)

            if not units:
                raise ValueError(f"Row {row_idx}: units empty")
            if any(ch not in {"B", "I"} for ch in labels):
                raise ValueError(f"Row {row_idx}: manual_bi_final must contain only B/I")
            if len(units) != len(labels):
                raise ValueError(
                    f"Row {row_idx}: len(units)={len(units)} != len(labels)={len(labels)}"
                )

            token_id_raw = str(row["token_id"]).strip()
            token_id = int(token_id_raw) if token_id_raw.isdigit() else row_idx

            samples.append(
                Sample(
                    record_id=str(row["record_id"]).strip(),
                    token_id=token_id,
                    units=units,
                    labels=labels,
                )
            )

    if not samples:
        raise ValueError("No valid rows found in CSV")

    samples.sort(key=lambda s: (s.record_id, s.token_id))
    return samples


def to_xyg(samples: list[Sample]) -> tuple[list[list[dict[str, object]]], list[list[str]], list[str]]:
    x_data: list[list[dict[str, object]]] = []
    y_data: list[list[str]] = []
    groups: list[str] = []

    for sample in samples:
        x_data.append([phone_features(sample.units, i) for i in range(len(sample.units))])
        y_data.append(sample.labels)
        groups.append(sample.record_id)

    return x_data, y_data, groups


def split_train_test(
    x_data: list[list[dict[str, object]]],
    y_data: list[list[str]],
    groups: list[str],
    test_size: float,
    random_state: int,
) -> tuple[list, list, list, list]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(x_data, y_data, groups=groups))
    x_train = [x_data[i] for i in train_idx]
    y_train = [y_data[i] for i in train_idx]
    x_test = [x_data[i] for i in test_idx]
    y_test = [y_data[i] for i in test_idx]
    return x_train, y_train, x_test, y_test


def build_crf(c1: float, c2: float, max_iterations: int) -> sklearn_crfsuite.CRF:
    return sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=c1,
        c2=c2,
        max_iterations=max_iterations,
        all_possible_transitions=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/test CRF layer-3 BI from phoneme units + manual BI gold labels."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help=f"Path CSV dataset (default: {DEFAULT_DATA})",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=DEFAULT_MODEL_OUT,
        help=f"Output path for trained model (default: {DEFAULT_MODEL_OUT})",
    )
    parser.add_argument(
        "--config-out",
        type=Path,
        default=DEFAULT_CONFIG_OUT,
        help=f"Output path for training config JSON (default: {DEFAULT_CONFIG_OUT})",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for split reproducibility (default: 42)",
    )
    parser.add_argument(
        "--c1",
        type=float,
        default=0.1,
        help="L1 regularization coefficient (default: 0.1)",
    )
    parser.add_argument(
        "--c2",
        type=float,
        default=0.1,
        help="L2 regularization coefficient (default: 0.1)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=200,
        help="Max optimization iterations (default: 200)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = parse_samples(args.data)
    x_data, y_data, groups = to_xyg(samples)
    x_train, y_train, x_test, y_test = split_train_test(
        x_data=x_data,
        y_data=y_data,
        groups=groups,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print(
        f"Loaded {len(samples)} samples | "
        f"train={len(x_train)} test={len(x_test)} | "
        f"unique_records={len(set(groups))}"
    )

    crf = build_crf(c1=args.c1, c2=args.c2, max_iterations=args.max_iterations)
    crf.fit(x_train, y_train)
    y_pred = crf.predict(x_test)

    accuracy = metrics.flat_accuracy_score(y_test, y_pred)
    precision_macro = metrics.flat_precision_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    recall_macro = metrics.flat_recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = metrics.flat_f1_score(y_test, y_pred, average="macro", zero_division=0)

    precision_weighted = metrics.flat_precision_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
    recall_weighted = metrics.flat_recall_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
    f1_weighted = metrics.flat_f1_score(y_test, y_pred, average="weighted", zero_division=0)

    y_true_flat = [label for seq in y_test for label in seq]
    y_pred_flat = [label for seq in y_pred for label in seq]
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=["B", "I"])

    print(f"Token-level accuracy          : {accuracy:.6f}")
    print(f"Token-level precision (macro) : {precision_macro:.6f}")
    print(f"Token-level recall (macro)    : {recall_macro:.6f}")
    print(f"Token-level F1 (macro)        : {f1_macro:.6f}")
    print(f"Token-level precision (wtd)   : {precision_weighted:.6f}")
    print(f"Token-level recall (wtd)      : {recall_weighted:.6f}")
    print(f"Token-level F1 (wtd)          : {f1_weighted:.6f}")
    print("Confusion matrix [B, I] rows=true cols=pred:")
    print(cm)
    print(metrics.flat_classification_report(y_test, y_pred, labels=["B", "I"], digits=4))

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(crf, args.model_out)
    print(f"Model saved to: {args.model_out}")

    config_payload = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(args.data),
        "model_out": str(args.model_out),
        "split": {
            "test_size": args.test_size,
            "random_state": args.random_state,
            "group_key": "record_id",
        },
        "crf_params": {
            "algorithm": "lbfgs",
            "c1": args.c1,
            "c2": args.c2,
            "max_iterations": args.max_iterations,
            "all_possible_transitions": True,
        },
        "dataset_stats": {
            "total_samples": len(samples),
            "train_samples": len(x_train),
            "test_samples": len(x_test),
            "unique_records": len(set(groups)),
            "test_tokens": len(y_true_flat),
        },
        "metrics": {
            "accuracy": float(accuracy),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "precision_weighted": float(precision_weighted),
            "recall_weighted": float(recall_weighted),
            "f1_weighted": float(f1_weighted),
            "confusion_matrix_labels": ["B", "I"],
            "confusion_matrix": cm.tolist(),
        },
    }
    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    args.config_out.write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Training config saved to: {args.config_out}")


if __name__ == "__main__":
    main()
