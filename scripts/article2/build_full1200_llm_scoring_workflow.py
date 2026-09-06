#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


KEY_FIELDS = ["no", "id_data_asli", "model", "setting"]


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
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


def as_float(value, default=0.0):
    try:
        return float((value or "").strip())
    except Exception:
        return default


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build full 1200 LLM scoring file by merging scored sample300, "
            "auto-scoring leakage rows (CLR>0 => score=1), and producing manual queue."
        )
    )
    parser.add_argument(
        "--full-input",
        default="data/article2/100_Pantun_Eksperimen_long_scored.csv",
        help="Full 1200 CSV with structural metrics.",
    )
    parser.add_argument(
        "--sample-scored",
        default="data/article2/100_Pantun_Eksperimen_sample300_for_chat.csv",
        help="Sample300 CSV already filled with LLM score.",
    )
    parser.add_argument(
        "--output-full",
        default="data/article2/100_Pantun_Eksperimen_1200_llm_scoring_master.csv",
        help="Master output file that holds all current LLM scores for 1200 rows.",
    )
    parser.add_argument(
        "--output-manual",
        default="data/article2/100_Pantun_Eksperimen_1200_manual_queue.csv",
        help="Rows still needing manual LLM scoring.",
    )
    parser.add_argument(
        "--score-col",
        default="semantic_bridge_score_llm_1_5",
        help="LLM score column name.",
    )
    parser.add_argument(
        "--clr-threshold",
        type=float,
        default=0.0,
        help="Auto score=1 when CLR is strictly greater than this threshold.",
    )
    args = parser.parse_args()

    full_rows, full_fields = read_csv(Path(args.full_input))
    sample_rows, sample_fields = read_csv(Path(args.sample_scored))

    if args.score_col not in full_fields:
        full_fields.append(args.score_col)
        for r in full_rows:
            r[args.score_col] = ""

    if "score_source" not in full_fields:
        full_fields.append("score_source")

    # Index sample scores by key.
    sample_score_map = {}
    for r in sample_rows:
        key = row_key(r)
        score = (r.get(args.score_col, "") or "").strip()
        if score:
            sample_score_map[key] = score

    merged_from_sample = 0
    auto_from_clr = 0
    already_filled = 0

    for r in full_rows:
        key = row_key(r)
        existing = (r.get(args.score_col, "") or "").strip()

        if existing:
            r["score_source"] = r.get("score_source", "manual_existing") or "manual_existing"
            already_filled += 1
            continue

        s300_score = sample_score_map.get(key, "")
        if s300_score:
            r[args.score_col] = s300_score
            r["score_source"] = "sample300_manual"
            merged_from_sample += 1
            continue

        clr = as_float(r.get("clr", "0"), default=0.0)
        if clr > args.clr_threshold:
            r[args.score_col] = "1"
            r["score_source"] = "auto_clr_leakage"
            auto_from_clr += 1
        else:
            r["score_source"] = "manual_pending"

    # Build manual queue from rows still empty.
    manual_rows = [
        r for r in full_rows if not (r.get(args.score_col, "") or "").strip()
    ]

    # Put easy copy columns at front in manual queue.
    manual_front = [
        "isi_input_model",
        "sampiran_generated",
        args.score_col,
        "score_source",
    ]
    manual_fields = manual_front + [
        c for c in full_fields if c not in manual_front
    ]

    write_csv(Path(args.output_full), full_rows, full_fields)
    write_csv(Path(args.output_manual), manual_rows, manual_fields)

    total = len(full_rows)
    filled = sum(1 for r in full_rows if (r.get(args.score_col, "") or "").strip())

    print(f"total_rows={total}")
    print(f"filled_scores={filled}")
    print(f"merged_from_sample300={merged_from_sample}")
    print(f"auto_scored_clr_leakage={auto_from_clr}")
    print(f"manual_pending={len(manual_rows)}")
    print(f"output_full={args.output_full}")
    print(f"output_manual={args.output_manual}")


if __name__ == "__main__":
    main()
