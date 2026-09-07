#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

KEY_FIELDS = ["no", "id_data_asli", "model", "setting"]


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        sample = f.read(8192)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;")
        except csv.Error:
            dialect = csv.get_dialect("excel")
        reader = csv.DictReader(f, dialect=dialect)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def row_key(row):
    return tuple((row.get(k, "") or "").strip() for k in KEY_FIELDS)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Merge completed manual queue scores back into 1200 master scoring CSV."
        )
    )
    parser.add_argument(
        "--master",
        default="data/article2/100_Pantun_Eksperimen_1200_llm_scoring_master.csv",
        help="Master 1200 CSV.",
    )
    parser.add_argument(
        "--manual",
        default="data/article2/100_pantun_eksperimen_1200_manual_queue.csv",
        help="Manual queue CSV that has been filled.",
    )
    parser.add_argument(
        "--output",
        default="data/article2/100_Pantun_Eksperimen_1200_llm_scoring_master_final.csv",
        help="Output final master CSV.",
    )
    parser.add_argument(
        "--score-col",
        default="semantic_bridge_score_llm_1_5",
        help="Score column name.",
    )
    args = parser.parse_args()

    master_rows, master_fields = read_csv(Path(args.master))
    manual_rows, _ = read_csv(Path(args.manual))

    if args.score_col not in master_fields:
        raise ValueError(f"Score column not found in master: {args.score_col}")

    if "score_source" not in master_fields:
        master_fields.append("score_source")

    manual_score_map = {}
    duplicate_manual_keys = 0
    for row in manual_rows:
        key = row_key(row)
        score = (row.get(args.score_col, "") or "").strip()
        if key in manual_score_map:
            duplicate_manual_keys += 1
        manual_score_map[key] = score

    updated_from_manual = 0
    still_empty = 0
    unmatched_manual_rows = 0

    master_key_set = {row_key(r) for r in master_rows}
    for key in manual_score_map:
        if key not in master_key_set:
            unmatched_manual_rows += 1

    for row in master_rows:
        current_score = (row.get(args.score_col, "") or "").strip()
        if current_score:
            continue

        key = row_key(row)
        manual_score = (manual_score_map.get(key, "") or "").strip()
        if manual_score:
            row[args.score_col] = manual_score
            row["score_source"] = "manual_queue_filled"
            updated_from_manual += 1
        else:
            still_empty += 1

    total = len(master_rows)
    filled = sum(1 for r in master_rows if (r.get(args.score_col, "") or "").strip())

    write_csv(Path(args.output), master_rows, master_fields)

    print(f"total_rows={total}")
    print(f"filled_scores={filled}")
    print(f"updated_from_manual={updated_from_manual}")
    print(f"still_empty={still_empty}")
    print(f"manual_rows={len(manual_rows)}")
    print(f"duplicate_manual_keys={duplicate_manual_keys}")
    print(f"unmatched_manual_rows={unmatched_manual_rows}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
