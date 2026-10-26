import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


def parse_tags_csv(text: str) -> Set[str]:
    return {x.strip().upper() for x in text.split(",") if x.strip()}


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def compute_tag_stats(y_true: List[str], y_pred: List[str], labels: List[str]) -> List[Dict]:
    stats = []
    for tag in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == tag and p == tag)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != tag and p == tag)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == tag and p != tag)
        support = sum(1 for t in y_true if t == tag)

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        stats.append(
            {
                "tag": tag,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
    return stats


def build_confusion(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Dict[str, int]]:
    conf: Dict[str, Dict[str, int]] = {true_tag: {pred_tag: 0 for pred_tag in labels} for true_tag in labels}
    for t, p in zip(y_true, y_pred):
        conf[t][p] += 1
    return conf


def format_metrics_table(tag_stats: List[Dict]) -> str:
    header = f"{'TAG':<8} {'P':>8} {'R':>8} {'F1':>8} {'SUPPORT':>8}"
    rows = [header, "-" * len(header)]
    for row in tag_stats:
        rows.append(
            f"{row['tag']:<8} {row['precision']:>8.4f} {row['recall']:>8.4f} "
            f"{row['f1']:>8.4f} {row['support']:>8}"
        )
    return "\n".join(rows)


def write_confusion_csv(path: Path, confusion: Dict[str, Dict[str, int]], labels: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gold\\pred", *labels])
        for gold_tag in labels:
            writer.writerow([gold_tag, *[confusion[gold_tag][pred_tag] for pred_tag in labels]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate token-level mini-gold POS annotations")
    parser.add_argument("--gold-csv", required=True, help="Token-level CSV with columns gold_tag and pred_tag")
    parser.add_argument("--pred-col", default="pred_tag", help="Prediction column name")
    parser.add_argument("--gold-col", default="gold_tag", help="Gold label column name")
    parser.add_argument("--critical-tags", default="ADV,PART,NOUN,VERB")
    parser.add_argument("--out-metrics-json", default=None)
    parser.add_argument("--out-confusion-csv", default=None)
    args = parser.parse_args()

    critical_tags = parse_tags_csv(args.critical_tags)
    y_true: List[str] = []
    y_pred: List[str] = []
    skipped_missing = 0

    with open(args.gold_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing_cols = {args.gold_col, args.pred_col} - set(reader.fieldnames or [])
        if missing_cols:
            raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

        for row in reader:
            gold = (row.get(args.gold_col, "") or "").strip().upper()
            pred = (row.get(args.pred_col, "") or "").strip().upper()
            if not gold or not pred:
                skipped_missing += 1
                continue
            y_true.append(gold)
            y_pred.append(pred)

    if not y_true:
        raise ValueError("No labeled rows found. Please fill gold_tag (and pred_tag if needed).")

    labels = sorted(set(y_true) | set(y_pred))
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

    tag_stats = compute_tag_stats(y_true, y_pred, labels)
    macro_f1 = safe_div(sum(x["f1"] for x in tag_stats), len(tag_stats))
    weighted_f1 = safe_div(
        sum(x["f1"] * x["support"] for x in tag_stats),
        sum(x["support"] for x in tag_stats),
    )

    critical_present = [t for t in labels if t in critical_tags]
    critical_stats = compute_tag_stats(y_true, y_pred, critical_present) if critical_present else []
    critical_macro_f1 = (
        safe_div(sum(x["f1"] for x in critical_stats), len(critical_stats)) if critical_stats else 0.0
    )

    confusion = build_confusion(y_true, y_pred, labels)

    print("[SUMMARY]")
    print(f"labeled_tokens={len(y_true)}")
    print(f"skipped_rows_missing_label={skipped_missing}")
    print(f"accuracy={accuracy:.6f}")
    print(f"macro_f1={macro_f1:.6f}")
    print(f"weighted_f1={weighted_f1:.6f}")
    print(f"critical_macro_f1={critical_macro_f1:.6f}")
    print(f"critical_tags_used={critical_present}")
    print()

    print("[PER-TAG METRICS]")
    print(format_metrics_table(tag_stats))
    if critical_stats:
        print()
        print("[CRITICAL TAG METRICS]")
        print(format_metrics_table(critical_stats))

    if args.out_confusion_csv:
        conf_path = Path(args.out_confusion_csv)
        write_confusion_csv(conf_path, confusion, labels)
        print(f"\n[OK] confusion matrix saved: {conf_path}")

    if args.out_metrics_json:
        out_path = Path(args.out_metrics_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": {
                "labeled_tokens": len(y_true),
                "skipped_rows_missing_label": skipped_missing,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "critical_macro_f1": critical_macro_f1,
                "critical_tags_used": critical_present,
            },
            "per_tag": tag_stats,
            "critical_per_tag": critical_stats,
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[OK] metrics json saved: {out_path}")


if __name__ == "__main__":
    main()
