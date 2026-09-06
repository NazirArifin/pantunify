#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import mean, median


def _average_ranks(values):
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)

    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1

        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank

        i = j + 1

    return ranks


def _pearson_corr(x, y):
    n = len(x)
    if n != len(y) or n == 0:
        return 0.0

    mx = mean(x)
    my = mean(y)

    num = 0.0
    den_x = 0.0
    den_y = 0.0
    for a, b in zip(x, y):
        dx = a - mx
        dy = b - my
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy

    den = math.sqrt(den_x * den_y)
    if den == 0:
        return 0.0
    return num / den


def spearman_rho(x, y):
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    return _pearson_corr(rx, ry)


def normal_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def p_value_t_approx_from_r(r, n):
    if n < 3:
        return 1.0
    if abs(r) >= 1.0:
        return 0.0
    t = r * math.sqrt((n - 2) / (1.0 - r * r))
    p = 2.0 * (1.0 - normal_cdf(abs(t)))
    return max(0.0, min(1.0, p))


def spearman_permutation_pvalue(x, y, n_perm=20000, seed=42):
    rnd = random.Random(seed)
    obs = abs(spearman_rho(x, y))

    y_copy = list(y)
    ge = 0
    for _ in range(n_perm):
        rnd.shuffle(y_copy)
        rp = abs(spearman_rho(x, y_copy))
        if rp >= obs:
            ge += 1

    return (ge + 1) / (n_perm + 1)


def load_calibration(blind_path, key_path):
    with blind_path.open("r", encoding="utf-8", newline="") as f:
        blind = list(csv.DictReader(f))

    key_map = {}
    with key_path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            key_map[r["item_id"]] = r

    pairs = []
    for r in blind:
        item_id = r.get("item_id", "")
        hs = (r.get("semantic_bridge_score_human_1_5", "") or "").strip()
        ls = (r.get("semantic_bridge_score_llm_1_5", "") or "").strip()

        try:
            human = float(hs)
            llm = float(ls)
        except ValueError:
            continue

        meta = key_map.get(item_id, {})
        pairs.append(
            {
                "item_id": item_id,
                "human": human,
                "llm": llm,
                "diff_llm_minus_human": llm - human,
                "abs_diff": abs(llm - human),
                "model": meta.get("model", ""),
                "setting": meta.get("setting", ""),
                "id_data_asli": meta.get("id_data_asli", ""),
                "no": meta.get("no", ""),
            }
        )

    return pairs


def save_pairs(path, pairs):
    fields = [
        "item_id",
        "no",
        "id_data_asli",
        "model",
        "setting",
        "human",
        "llm",
        "diff_llm_minus_human",
        "abs_diff",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p in pairs:
            w.writerow(
                {
                    "item_id": p["item_id"],
                    "no": p["no"],
                    "id_data_asli": p["id_data_asli"],
                    "model": p["model"],
                    "setting": p["setting"],
                    "human": f"{p['human']:.6f}",
                    "llm": f"{p['llm']:.6f}",
                    "diff_llm_minus_human": f"{p['diff_llm_minus_human']:.6f}",
                    "abs_diff": f"{p['abs_diff']:.6f}",
                }
            )


def save_summary(path, summary):
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Evaluate human vs LLM calibration on blind sample.")
    parser.add_argument("--blind", default="data/article2/human_calibration_50_blind.csv")
    parser.add_argument("--key", default="data/article2/human_calibration_50_key.csv")
    parser.add_argument("--outdir", default="data/article2")
    parser.add_argument("--n-perm", type=int, default=20000)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pairs = load_calibration(Path(args.blind), Path(args.key))
    if len(pairs) < 3:
        raise ValueError("Pasangan data valid kurang dari 3.")

    x = [p["human"] for p in pairs]
    y = [p["llm"] for p in pairs]

    rho = spearman_rho(x, y)
    p_t = p_value_t_approx_from_r(rho, len(pairs))
    p_perm = spearman_permutation_pvalue(x, y, n_perm=args.n_perm, seed=123)

    diffs = [p["diff_llm_minus_human"] for p in pairs]
    abs_diffs = [abs(d) for d in diffs]

    by_model = {}
    model_names = sorted({p["model"] for p in pairs if p["model"]})
    for m in model_names:
        vals = [p for p in pairs if p["model"] == m]
        by_model[m] = {
            "n": len(vals),
            "mean_human": round(mean([v["human"] for v in vals]), 6),
            "mean_llm": round(mean([v["llm"] for v in vals]), 6),
            "mean_abs_diff": round(mean([v["abs_diff"] for v in vals]), 6),
        }

    summary = {
        "n_pairs": len(pairs),
        "spearman_rho": round(rho, 6),
        "p_value_t_approx": p_t,
        "p_value_permutation": p_perm,
        "mean_human": round(mean(x), 6),
        "mean_llm": round(mean(y), 6),
        "mean_diff_llm_minus_human": round(mean(diffs), 6),
        "median_diff_llm_minus_human": round(median(diffs), 6),
        "mean_abs_diff": round(mean(abs_diffs), 6),
        "median_abs_diff": round(median(abs_diffs), 6),
        "by_model": by_model,
    }

    save_pairs(outdir / "human_llm_calibration_pairs.csv", pairs)
    save_summary(outdir / "human_llm_calibration_summary.json", summary)

    print("Selesai evaluasi kalibrasi.")
    print(f"n_pairs={summary['n_pairs']}")
    print(f"spearman_rho={summary['spearman_rho']}")
    print(f"p_value_t_approx={summary['p_value_t_approx']:.10g}")
    print(f"p_value_permutation={summary['p_value_permutation']:.10g}")
    print(f"mean_abs_diff={summary['mean_abs_diff']}")
    print(f"Output: {outdir / 'human_llm_calibration_pairs.csv'}")
    print(f"Output: {outdir / 'human_llm_calibration_summary.json'}")


if __name__ == "__main__":
    main()
