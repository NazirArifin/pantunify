#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def normalize_s_judge(score_1_5: float) -> float:
    # Normalisasi Likert 1-5 ke [0, 1].
    return (score_1_5 - 1.0) / 4.0


def calc_xrasa(a_score: float, r_score: float, s_judge_norm: float) -> float:
    # xRASA = A * (0.5 * R + 0.5 * S_judge)
    return a_score * (0.5 * r_score + 0.5 * s_judge_norm)


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
        fields = list(r.fieldnames or [])
    return rows, fields


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def summarize_group(rows):
    grp = defaultdict(list)
    for r in rows:
        grp[(r["model"], r["setting"])].append(r)

    out = []
    for (model, setting), items in sorted(grp.items()):
        x_vals = [float(x["xrasa_final"]) for x in items]
        r_vals = [float(x["r_score"]) for x in items]
        a_vals = [float(x["a_score"]) for x in items]
        s_vals = [float(x["s_judge_norm"]) for x in items]
        clr_vals = [float(x["clr"]) for x in items]

        out.append(
            {
                "model": model,
                "setting": setting,
                "n": len(items),
                "xrasa_mean": f"{mean(x_vals):.6f}",
                "xrasa_std": f"{(stdev(x_vals) if len(x_vals) > 1 else 0.0):.6f}",
                "r_mean": f"{mean(r_vals):.6f}",
                "a_mean": f"{mean(a_vals):.6f}",
                "s_judge_mean": f"{mean(s_vals):.6f}",
                "clr_mean": f"{mean(clr_vals):.6f}",
            }
        )

    for setting in sorted({x["setting"] for x in out}):
        subset = [x for x in out if x["setting"] == setting]
        rank_x = sorted(subset, key=lambda x: float(x["xrasa_mean"]), reverse=True)
        rank_c = sorted(subset, key=lambda x: float(x["clr_mean"]))

        rank_x_map = {x["model"]: i + 1 for i, x in enumerate(rank_x)}
        rank_c_map = {x["model"]: i + 1 for i, x in enumerate(rank_c)}

        for row in subset:
            row["rank_xrasa"] = rank_x_map[row["model"]]
            row["rank_clr"] = rank_c_map[row["model"]]

    return out


def main():
    parser = argparse.ArgumentParser(description="Compute final xRASA scores from scored CSV.")
    parser.add_argument(
        "--input",
        default="data/article2/100_Pantun_Eksperimen_sample300_for_chat.csv",
        help="Input CSV with a_score, r_score, clr, and semantic_bridge_score_llm_1_5.",
    )
    parser.add_argument(
        "--output",
        default="data/article2/100_Pantun_Eksperimen_sample300_xrasa.csv",
        help="Output CSV with additional xRASA columns.",
    )
    parser.add_argument(
        "--summary",
        default="data/article2/table_main_xrasa_sample300.csv",
        help="Summary CSV by model and setting.",
    )
    parser.add_argument(
        "--score-col",
        default="semantic_bridge_score_llm_1_5",
        help="LLM score column (1..5).",
    )
    args = parser.parse_args()

    rows, fields = read_csv(Path(args.input))
    required = ["a_score", "r_score", "clr", "model", "setting", args.score_col]
    missing = [c for c in required if c not in fields]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for extra in ["s_judge_norm", "xrasa_final"]:
        if extra not in fields:
            fields.append(extra)

    valid = 0
    for row in rows:
        s_raw = (row.get(args.score_col) or "").strip()
        if not s_raw:
            row["s_judge_norm"] = ""
            row["xrasa_final"] = ""
            continue

        s15 = float(s_raw)
        if s15 < 1 or s15 > 5:
            row["s_judge_norm"] = ""
            row["xrasa_final"] = ""
            continue

        s_norm = normalize_s_judge(s15)
        a = float(row["a_score"])
        r = float(row["r_score"])
        xrasa = calc_xrasa(a, r, s_norm)

        row["s_judge_norm"] = f"{s_norm:.6f}"
        row["xrasa_final"] = f"{xrasa:.6f}"
        valid += 1

    write_csv(Path(args.output), rows, fields)

    summary_rows = summarize_group([r for r in rows if (r.get("xrasa_final") or "").strip()])
    summary_fields = [
        "model",
        "setting",
        "n",
        "xrasa_mean",
        "xrasa_std",
        "r_mean",
        "a_mean",
        "s_judge_mean",
        "clr_mean",
        "rank_xrasa",
        "rank_clr",
    ]
    write_csv(Path(args.summary), summary_rows, summary_fields)

    print(f"rows_total={len(rows)}")
    print(f"rows_with_valid_xrasa={valid}")
    print(f"output={args.output}")
    print(f"summary={args.summary}")


if __name__ == "__main__":
    main()
