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
import re

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
    token: str
    annotator_notes: str
    is_synthetic: bool
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
                    token=str(row.get("token", "")).strip(),
                    annotator_notes=str(row.get("annotator_notes", "")).strip(),
                    is_synthetic=(
                        str(row.get("annotator_notes", "")).strip().lower()
                        == "synthetic_hiatus_fix"
                    ),
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
    force_train_mask: list[bool] | None = None,
) -> tuple[list, list, list, list, list[int], list[int]]:
    if force_train_mask is None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(x_data, y_data, groups=groups))
        train_idx = train_idx.tolist()
        test_idx = test_idx.tolist()
    else:
        if len(force_train_mask) != len(x_data):
            raise ValueError("force_train_mask length must match data length")
        candidate_idx = [i for i, forced in enumerate(force_train_mask) if not forced]
        forced_train_idx = [i for i, forced in enumerate(force_train_mask) if forced]
        if not candidate_idx:
            raise ValueError("No non-forced samples available for test split")

        x_candidate = [x_data[i] for i in candidate_idx]
        y_candidate = [y_data[i] for i in candidate_idx]
        g_candidate = [groups[i] for i in candidate_idx]
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_local, test_local = next(splitter.split(x_candidate, y_candidate, groups=g_candidate))
        train_idx = [candidate_idx[i] for i in train_local.tolist()] + forced_train_idx
        test_idx = [candidate_idx[i] for i in test_local.tolist()]

    x_train = [x_data[i] for i in train_idx]
    y_train = [y_data[i] for i in train_idx]
    x_test = [x_data[i] for i in test_idx]
    y_test = [y_data[i] for i in test_idx]
    return x_train, y_train, x_test, y_test, train_idx, test_idx


def count_syllables_from_bi(labels: list[str]) -> int:
    return sum(1 for lb in labels if lb == "B")


def is_reduplication_like(token: str) -> bool:
    t = "".join(ch for ch in str(token).lower().strip() if ch.isalpha())
    if len(t) < 6:
        return False
    if "-" in token:
        return True
    for stem_len in range(3, (len(t) // 2) + 1):
        stem = t[:stem_len]
        if t[stem_len : stem_len * 2] == stem:
            return True
    return False


def evaluate_subset(
    subset_name: str,
    samples: list[Sample],
    y_true: list[list[str]],
    y_pred: list[list[str]],
    predicate,
) -> dict[str, float | int]:
    idxs = [i for i, sample in enumerate(samples) if predicate(sample)]
    if not idxs:
        return {
            "subset_name": subset_name,
            "sample_count": 0,
            "token_count": 0,
            "word_exact_match": 0.0,
            "syllable_count_accuracy": 0.0,
        }

    true_subset = [y_true[i] for i in idxs]
    pred_subset = [y_pred[i] for i in idxs]
    token_count = sum(len(seq) for seq in true_subset)

    word_exact_match = sum(1 for t, p in zip(true_subset, pred_subset) if t == p) / len(idxs)
    syll_count_acc = (
        sum(
            1
            for t, p in zip(true_subset, pred_subset)
            if count_syllables_from_bi(t) == count_syllables_from_bi(p)
        )
        / len(idxs)
    )

    return {
        "subset_name": subset_name,
        "sample_count": len(idxs),
        "token_count": token_count,
        "word_exact_match": float(word_exact_match),
        "syllable_count_accuracy": float(syll_count_acc),
    }


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
    synthetic_mask = [sample.is_synthetic for sample in samples]
    x_train, y_train, x_test, y_test, _train_idx, test_idx = split_train_test(
        x_data=x_data,
        y_data=y_data,
        groups=groups,
        test_size=args.test_size,
        random_state=args.random_state,
        force_train_mask=synthetic_mask,
    )
    test_samples = [samples[i] for i in test_idx]
    synthetic_count = sum(1 for s in samples if s.is_synthetic)

    print(
        f"Loaded {len(samples)} samples | "
        f"train={len(x_train)} test={len(x_test)} | "
        f"unique_records={len(set(groups))} | "
        f"synthetic_forced_train={synthetic_count}"
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

    targeted_metrics = [
        evaluate_subset(
            subset_name="hiatus_ai_ui_oi",
            samples=test_samples,
            y_true=y_test,
            y_pred=y_pred,
            predicate=lambda s: bool(re.search(r"(ai|ui|oi)", s.token.lower())),
        ),
        evaluate_subset(
            subset_name="token_endswith_i",
            samples=test_samples,
            y_true=y_test,
            y_pred=y_pred,
            predicate=lambda s: s.token.lower().endswith("i"),
        ),
        evaluate_subset(
            subset_name="reduplication_like",
            samples=test_samples,
            y_true=y_test,
            y_pred=y_pred,
            predicate=lambda s: is_reduplication_like(s.token),
        ),
    ]
    print("Targeted subset metrics:")
    for row in targeted_metrics:
        print(
            f"- {row['subset_name']}: samples={row['sample_count']} tokens={row['token_count']} "
            f"| word_exact={row['word_exact_match']:.4f} "
            f"| syllable_count_acc={row['syllable_count_accuracy']:.4f}"
        )

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
            "synthetic_policy": "annotator_notes=synthetic_hiatus_fix forced to train",
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
            "targeted_subset_metrics": targeted_metrics,
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
