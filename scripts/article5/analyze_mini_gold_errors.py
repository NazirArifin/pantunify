import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set


def parse_tags_csv(text: str) -> Set[str]:
    return {x.strip().upper() for x in text.split(",") if x.strip()}


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Focused error analysis for mini-gold (e.g. ADV/PART/VERB -> NOUN)"
    )
    parser.add_argument("--gold-csv", required=True, help="Token-level mini-gold CSV")
    parser.add_argument("--gold-col", default="gold_tag")
    parser.add_argument("--pred-col", default="pred_tag")
    parser.add_argument("--focus-gold-tags", default="ADV,PART,VERB")
    parser.add_argument("--target-pred-tag", default="NOUN")
    parser.add_argument("--top-k", type=int, default=25, help="Number of example rows to print")
    parser.add_argument("--out-errors-csv", default=None, help="Optional output CSV of focused errors")
    args = parser.parse_args()

    focus_gold_tags = parse_tags_csv(args.focus_gold_tags)
    target_pred_tag = args.target_pred_tag.strip().upper()

    rows: List[Dict[str, str]] = []
    source_support = Counter()
    source_errors = Counter()
    token_error_counter = Counter()
    line_error_counter = Counter()

    with open(args.gold_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        needed = {args.gold_col, args.pred_col}
        missing = needed - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        for row in reader:
            gold = (row.get(args.gold_col, "") or "").strip().upper()
            pred = (row.get(args.pred_col, "") or "").strip().upper()
            if not gold or not pred:
                continue
            if gold in focus_gold_tags:
                source_support[gold] += 1
                if pred == target_pred_tag:
                    source_errors[gold] += 1
                    token_key = (row.get("token", "") or "").strip().lower()
                    if token_key:
                        token_error_counter[token_key] += 1
                    line_key = (row.get("line_text", "") or "").strip()
                    if line_key:
                        line_error_counter[line_key] += 1
                    rows.append(row)

    total_focus_support = sum(source_support.values())
    total_focus_errors = sum(source_errors.values())

    print("[FOCUSED CONFUSION SUMMARY]")
    print(f"focus_gold_tags={sorted(focus_gold_tags)}")
    print(f"target_pred_tag={target_pred_tag}")
    print(f"focus_support={total_focus_support}")
    print(f"focus_errors_to_target={total_focus_errors}")
    print(f"focus_error_rate={safe_div(total_focus_errors, total_focus_support):.6f}")
    print()

    print("[PER-SOURCE-TAG ERROR RATE]")
    print(f"{'TAG':<8} {'SUPPORT':>8} {'ERR->TARGET':>12} {'ERR_RATE':>10}")
    print("-" * 42)
    for tag in sorted(focus_gold_tags):
        support = source_support[tag]
        errors = source_errors[tag]
        print(f"{tag:<8} {support:>8} {errors:>12} {safe_div(errors, support):>10.4f}")

    print()
    print(f"[TOP TOKENS MISCLASSIFIED AS {target_pred_tag}]")
    for token, count in token_error_counter.most_common(args.top_k):
        print(f"{token}\t{count}")

    print()
    print("[SAMPLE ERROR ROWS]")
    sample_rows = rows[: args.top_k]
    for idx, row in enumerate(sample_rows, start=1):
        print(
            f"{idx:02d}. token={row.get('token','')} "
            f"gold={row.get(args.gold_col,'')} pred={row.get(args.pred_col,'')} "
            f"line='{row.get('line_text','')}'"
        )

    if args.out_errors_csv:
        out_path = Path(args.out_errors_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys()) if rows else ["token", args.gold_col, args.pred_col, "line_text"]
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"\n[OK] focused error rows saved: {out_path}")


if __name__ == "__main__":
    main()
