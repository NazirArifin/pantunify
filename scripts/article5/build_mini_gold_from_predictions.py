import argparse
import csv
import random
from pathlib import Path
from typing import List, Set


def parse_tags_csv(text: str) -> Set[str]:
    return {x.strip().upper() for x in text.split(",") if x.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build token-level mini-gold annotation template from prediction CSV"
    )
    parser.add_argument("--pred-csv", required=True, help="CSV output from predict_pantun_csv_crf.py")
    parser.add_argument("--out-csv", required=True, help="Token-level template CSV for manual annotation")
    parser.add_argument("--max-tokens", type=int, default=400, help="Max tokens to include")
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=None,
        help="Alias for max tokens (e.g. 1000 for 800-1200 plan). Overrides --max-tokens.",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=0,
        help="Fail if selected tokens are below this threshold.",
    )
    parser.add_argument("--shuffle-candidates", action="store_true", help="Shuffle candidate tokens before cut-off")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --shuffle-candidates")
    parser.add_argument(
        "--critical-tags",
        default="ADV,PART,NOUN,VERB",
        help="Comma-separated tag set to prioritize",
    )
    parser.add_argument(
        "--critical-only",
        action="store_true",
        help="If set, only include tokens whose pred_tag is in critical-tags",
    )
    args = parser.parse_args()

    critical_tags = parse_tags_csv(args.critical_tags)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    selected_rows: List[dict] = []
    total_read_tokens = 0
    max_tokens = args.target_tokens if args.target_tokens is not None else args.max_tokens

    with open(args.pred_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tokens = (row.get("tokens", "") or "").split()
            pred_tags = (row.get("pred_tags", "") or "").split()
            if not tokens:
                continue
            if len(tokens) != len(pred_tags):
                raise ValueError(
                    f"Mismatch token/tag count at pantun_index={row.get('pantun_index')} "
                    f"line_no={row.get('line_no')}: {len(tokens)} tokens vs {len(pred_tags)} tags"
                )

            for idx, (token, pred_tag) in enumerate(zip(tokens, pred_tags), start=1):
                total_read_tokens += 1
                pred_tag_norm = pred_tag.strip().upper()
                is_critical = pred_tag_norm in critical_tags
                if args.critical_only and not is_critical:
                    continue

                selected_rows.append(
                    {
                        "id": row.get("id", ""),
                        "pantun_index": row.get("pantun_index", ""),
                        "line_no": row.get("line_no", ""),
                        "line_text": row.get("line_text", ""),
                        "token_index": idx,
                        "token": token,
                        "pred_tag": pred_tag_norm,
                        "gold_tag": "",
                        "is_critical": "1" if is_critical else "0",
                        "notes": "",
                    }
                )

    if args.shuffle_candidates:
        rng = random.Random(args.seed)
        rng.shuffle(selected_rows)

    selected_rows = selected_rows[:max_tokens]
    if len(selected_rows) < args.min_tokens:
        raise ValueError(
            f"Selected tokens ({len(selected_rows)}) below required min-tokens ({args.min_tokens}). "
            "Increase source predictions or relax filters."
        )

    fieldnames = [
        "id",
        "pantun_index",
        "line_no",
        "line_text",
        "token_index",
        "token",
        "pred_tag",
        "gold_tag",
        "is_critical",
        "notes",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)

    print(f"[OK] template: {out_path}")
    print(f"[INFO] total source tokens read: {total_read_tokens}")
    print(f"[INFO] tokens selected: {len(selected_rows)}")
    print(f"[INFO] target_tokens={max_tokens} min_tokens={args.min_tokens}")
    print(f"[INFO] critical tags: {sorted(critical_tags)}")


if __name__ == "__main__":
    main()
