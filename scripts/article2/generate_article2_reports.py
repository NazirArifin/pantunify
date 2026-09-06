#!/usr/bin/env python3
import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


METRICS = ["r_score", "a_score", "clr"]


def _f(x):
    return float(x)


def _mean_std(values):
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def _normal_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def paired_t_test_approx(x, y):
    """
    Paired t-test with normal approximation for p-value (n besar, n=100 pada dataset ini).
    Returns (t_stat, p_value_two_tailed, mean_diff).
    """
    if len(x) != len(y):
        raise ValueError("Panjang pasangan berbeda.")
    n = len(x)
    if n < 2:
        return 0.0, 1.0, 0.0

    diffs = [a - b for a, b in zip(x, y)]
    md = mean(diffs)

    if n == 2:
        sd = abs(diffs[0] - md) * math.sqrt(2)
    else:
        sd = stdev(diffs)

    if sd == 0:
        if md == 0:
            return 0.0, 1.0, 0.0
        return float("inf") if md > 0 else float("-inf"), 0.0, md

    t_stat = md / (sd / math.sqrt(n))
    p_val = 2.0 * (1.0 - _normal_cdf(abs(t_stat)))
    return t_stat, max(0.0, min(1.0, p_val)), md


def permutation_test_mean_diff(x, y, n_perm=4000, seed=42):
    """
    Paired permutation test berbasis sign-flip untuk rata-rata selisih.
    """
    if len(x) != len(y):
        raise ValueError("Panjang pasangan berbeda.")
    diffs = [a - b for a, b in zip(x, y)]
    obs = abs(mean(diffs))
    rnd = random.Random(seed)

    ge = 0
    n = len(diffs)
    for _ in range(n_perm):
        flips = [1 if rnd.random() < 0.5 else -1 for _ in range(n)]
        perm = [d * s for d, s in zip(diffs, flips)]
        if abs(mean(perm)) >= obs:
            ge += 1

    p = (ge + 1) / (n_perm + 1)
    return p


def one_way_anova_permutation(groups, n_perm=3000, seed=42):
    """
    One-way ANOVA F + permutation p-value untuk beberapa grup independen.
    groups: dict[str, list[float]]
    """
    labels = list(groups.keys())
    data = [groups[k] for k in labels]

    flat = []
    group_sizes = []
    for g in data:
        flat.extend(g)
        group_sizes.append(len(g))

    grand = mean(flat)
    ss_between = 0.0
    ss_within = 0.0
    for g in data:
        gm = mean(g)
        ss_between += len(g) * ((gm - grand) ** 2)
        ss_within += sum((v - gm) ** 2 for v in g)

    k = len(data)
    n = len(flat)
    df_between = k - 1
    df_within = n - k
    ms_between = ss_between / df_between if df_between > 0 else 0.0
    ms_within = ss_within / df_within if df_within > 0 else 0.0
    f_obs = ms_between / ms_within if ms_within > 0 else float("inf")

    rnd = random.Random(seed)
    ge = 0
    flat_copy = list(flat)

    for _ in range(n_perm):
        rnd.shuffle(flat_copy)
        idx = 0
        perm_groups = []
        for s in group_sizes:
            perm_groups.append(flat_copy[idx:idx + s])
            idx += s

        gmean = mean(flat_copy)
        ssb = 0.0
        ssw = 0.0
        for g in perm_groups:
            gm = mean(g)
            ssb += len(g) * ((gm - gmean) ** 2)
            ssw += sum((v - gm) ** 2 for v in g)

        msb = ssb / df_between if df_between > 0 else 0.0
        msw = ssw / df_within if df_within > 0 else 0.0
        f_perm = msb / msw if msw > 0 else float("inf")

        if f_perm >= f_obs:
            ge += 1

    p_perm = (ge + 1) / (n_perm + 1)
    return f_obs, p_perm


def load_rows(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def build_main_table(rows):
    by_group = defaultdict(list)
    for r in rows:
        by_group[(r["model"], r["setting"])].append(r)

    out = []
    for (model, setting), grp in sorted(by_group.items()):
        r_vals = [_f(x["r_score"]) for x in grp]
        a_vals = [_f(x["a_score"]) for x in grp]
        clr_vals = [_f(x["clr"]) for x in grp]

        r_mean, r_std = _mean_std(r_vals)
        a_mean, a_std = _mean_std(a_vals)
        clr_mean, clr_std = _mean_std(clr_vals)

        # Proxy skor gabungan sebelum S_judge tersedia.
        x_struct = a_mean * r_mean

        out.append(
            {
                "model": model,
                "setting": setting,
                "n": len(grp),
                "r_mean": f"{r_mean:.6f}",
                "r_std": f"{r_std:.6f}",
                "a_mean": f"{a_mean:.6f}",
                "a_std": f"{a_std:.6f}",
                "clr_mean": f"{clr_mean:.6f}",
                "clr_std": f"{clr_std:.6f}",
                "x_struct_proxy": f"{x_struct:.6f}",
            }
        )

    # Ranking per setting berdasarkan x_struct_proxy (besar lebih baik) dan CLR (kecil lebih baik).
    for setting in sorted({r["setting"] for r in out}):
        subset = [r for r in out if r["setting"] == setting]

        rank_x = sorted(subset, key=lambda x: float(x["x_struct_proxy"]), reverse=True)
        rank_clr = sorted(subset, key=lambda x: float(x["clr_mean"]))

        rx_map = {r["model"]: i + 1 for i, r in enumerate(rank_x)}
        rc_map = {r["model"]: i + 1 for i, r in enumerate(rank_clr)}

        for r in subset:
            r["rank_x_struct_proxy"] = rx_map[r["model"]]
            r["rank_clr"] = rc_map[r["model"]]

    return out


def build_significance_tables(rows):
    # Pair per model by id_data_asli, compare few_shot - zero_shot.
    idx = defaultdict(dict)
    for r in rows:
        key = (r["model"], r["id_data_asli"], r["setting"])
        idx[key] = r

    paired_results = []
    models = sorted({r["model"] for r in rows})

    for model in models:
        for metric in METRICS:
            few = []
            zero = []
            ids = sorted({r["id_data_asli"] for r in rows if r["model"] == model})
            for iid in ids:
                rf = idx.get((model, iid, "few_shot"))
                rz = idx.get((model, iid, "zero_shot"))
                if not rf or not rz:
                    continue
                few.append(_f(rf[metric]))
                zero.append(_f(rz[metric]))

            t_stat, p_t_approx, md = paired_t_test_approx(few, zero)
            p_perm = permutation_test_mean_diff(few, zero, n_perm=4000, seed=123)

            paired_results.append(
                {
                    "model": model,
                    "metric": metric,
                    "n_pairs": len(few),
                    "mean_few_shot": f"{mean(few):.6f}" if few else "0.000000",
                    "mean_zero_shot": f"{mean(zero):.6f}" if zero else "0.000000",
                    "mean_diff_few_minus_zero": f"{md:.6f}",
                    "t_stat": f"{t_stat:.6f}",
                    "p_t_approx": f"{p_t_approx:.6f}",
                    "p_permutation": f"{p_perm:.6f}",
                }
            )

    # Per setting: beda antar model (one-way ANOVA permutation).
    anova_results = []
    settings = sorted({r["setting"] for r in rows})

    for setting in settings:
        subset = [r for r in rows if r["setting"] == setting]
        models_in_setting = sorted({r["model"] for r in subset})
        for metric in METRICS:
            groups = {}
            for model in models_in_setting:
                vals = [_f(r[metric]) for r in subset if r["model"] == model]
                groups[model] = vals

            f_obs, p_perm = one_way_anova_permutation(groups, n_perm=3000, seed=321)
            anova_results.append(
                {
                    "setting": setting,
                    "metric": metric,
                    "k_models": len(groups),
                    "n_total": sum(len(v) for v in groups.values()),
                    "f_stat": f"{f_obs:.6f}",
                    "p_permutation": f"{p_perm:.6f}",
                }
            )

    return paired_results, anova_results


def build_human_blind_sample(rows, out_blind, out_key, n=50, seed=777):
    rnd = random.Random(seed)

    by_group = defaultdict(list)
    for r in rows:
        by_group[(r["model"], r["setting"])].append(r)

    groups = sorted(by_group.keys())

    # Alokasi seimbang + sisa acak.
    base = n // len(groups)
    rem = n % len(groups)

    group_order = groups[:]
    rnd.shuffle(group_order)
    extra_groups = set(group_order[:rem])

    picked = []
    for g in groups:
        pool = by_group[g][:]
        rnd.shuffle(pool)
        take = base + (1 if g in extra_groups else 0)
        picked.extend(pool[:take])

    rnd.shuffle(picked)
    picked = picked[:n]

    blind_rows = []
    key_rows = []

    for i, r in enumerate(picked, start=1):
        item_id = f"HB-{i:03d}"
        blind_rows.append(
            {
                "item_id": item_id,
                "isi_input_model": r["isi_input_model"],
                "sampiran_generated": r["sampiran_generated"],
                "semantic_bridge_score_human_1_5": "",
                "catatan_penilai": "",
            }
        )

        key_rows.append(
            {
                "item_id": item_id,
                "no": r["no"],
                "id_data_asli": r["id_data_asli"],
                "model": r["model"],
                "setting": r["setting"],
            }
        )

    save_csv(
        out_blind,
        [
            "item_id",
            "isi_input_model",
            "sampiran_generated",
            "semantic_bridge_score_human_1_5",
            "catatan_penilai",
        ],
        blind_rows,
    )

    save_csv(out_key, ["item_id", "no", "id_data_asli", "model", "setting"], key_rows)


def main():
    parser = argparse.ArgumentParser(description="Generate report artifacts for article2 from scored long CSV.")
    parser.add_argument(
        "--input",
        default="data/article2/100_Pantun_Eksperimen_long_scored.csv",
        help="Input scored long CSV.",
    )
    parser.add_argument(
        "--outdir",
        default="data/article2",
        help="Output directory for report artifacts.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Jumlah sampel untuk blind human calibration.",
    )
    args = parser.parse_args()

    rows = load_rows(Path(args.input))
    outdir = Path(args.outdir)

    main_table = build_main_table(rows)
    save_csv(
        outdir / "table_main_metrics.csv",
        [
            "model",
            "setting",
            "n",
            "r_mean",
            "r_std",
            "a_mean",
            "a_std",
            "clr_mean",
            "clr_std",
            "x_struct_proxy",
            "rank_x_struct_proxy",
            "rank_clr",
        ],
        main_table,
    )

    paired, anova = build_significance_tables(rows)
    save_csv(
        outdir / "significance_zero_vs_few.csv",
        [
            "model",
            "metric",
            "n_pairs",
            "mean_few_shot",
            "mean_zero_shot",
            "mean_diff_few_minus_zero",
            "t_stat",
            "p_t_approx",
            "p_permutation",
        ],
        paired,
    )

    save_csv(
        outdir / "significance_across_models.csv",
        ["setting", "metric", "k_models", "n_total", "f_stat", "p_permutation"],
        anova,
    )

    build_human_blind_sample(
        rows,
        outdir / "human_calibration_50_blind.csv",
        outdir / "human_calibration_50_key.csv",
        n=args.sample_size,
    )

    print("Selesai menghasilkan artefak artikel2:")
    print(str(outdir / "table_main_metrics.csv"))
    print(str(outdir / "significance_zero_vs_few.csv"))
    print(str(outdir / "significance_across_models.csv"))
    print(str(outdir / "human_calibration_50_blind.csv"))
    print(str(outdir / "human_calibration_50_key.csv"))


if __name__ == "__main__":
    main()
