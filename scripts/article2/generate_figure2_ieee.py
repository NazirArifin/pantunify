#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = [
    "Claude",
    "Gemini",
    "Chat-GPT",
    "Llama 3.1: 8B",
    "DeepSeek-R1",
    "Sailor2",
]
SETTING_ORDER = ["zero_shot", "few_shot"]
SETTING_LABEL = {"zero_shot": "Zero-shot", "few_shot": "Few-shot"}


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["model", "setting"], as_index=False)
        .agg(
            n=("xrasa_final", "count"),
            xrasa_mean=("xrasa_final", "mean"),
            xrasa_std=("xrasa_final", "std"),
            clr_mean=("clr", "mean"),
            clr_std=("clr", "std"),
        )
    )
    grouped["xrasa_ci95"] = 1.96 * grouped["xrasa_std"] / np.sqrt(grouped["n"])
    grouped["clr_ci95"] = 1.96 * grouped["clr_std"] / np.sqrt(grouped["n"])
    return grouped


def _vals(summary: pd.DataFrame, metric_mean: str, metric_ci: str, model: str, setting: str):
    r = summary[(summary["model"] == model) & (summary["setting"] == setting)]
    if r.empty:
        return 0.0, 0.0
    return float(r.iloc[0][metric_mean]), float(r.iloc[0][metric_ci])


def draw_figure(summary: pd.DataFrame, out_png: Path, out_pdf: Path):
    # IEEE-friendly style: clean axes, compact sizing, high contrast, no chart junk.
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.8,
        }
    )

    x = np.arange(len(MODEL_ORDER))
    width = 0.36
    # Blue-toned palette for cleaner IEEE-style visuals on screen and print.
    colors = {"zero_shot": "#BFD7EA", "few_shot": "#1F4E79"}
    hatches = {"zero_shot": "", "few_shot": "//"}

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.75), constrained_layout=True)
    fig.patch.set_facecolor("white")

    # Panel (a): xRASA
    ax = axes[0]
    for i, setting in enumerate(SETTING_ORDER):
        means = []
        errs = []
        for model in MODEL_ORDER:
            m, e = _vals(summary, "xrasa_mean", "xrasa_ci95", model, setting)
            means.append(m)
            errs.append(e)
        ax.bar(
            x + (i - 0.5) * width,
            means,
            width,
            label=SETTING_LABEL[setting],
            color=colors[setting],
            edgecolor="black",
            linewidth=0.8,
            hatch=hatches[setting],
            yerr=errs,
            capsize=2,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
        )
    ax.set_title("(a) xRASA Mean by Model", color="#163A5F", pad=6)
    ax.set_ylabel("xRASA")
    ax.set_ylim(0, 0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, rotation=18, ha="right")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35, color="#5B7FA3")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel (b): CLR
    ax = axes[1]
    for i, setting in enumerate(SETTING_ORDER):
        means = []
        errs = []
        for model in MODEL_ORDER:
            m, e = _vals(summary, "clr_mean", "clr_ci95", model, setting)
            means.append(m)
            errs.append(e)
        ax.bar(
            x + (i - 0.5) * width,
            means,
            width,
            label=SETTING_LABEL[setting],
            color=colors[setting],
            edgecolor="black",
            linewidth=0.8,
            hatch=hatches[setting],
            yerr=errs,
            capsize=2,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
        )
    ax.set_title("(b) CLR Mean by Model", color="#163A5F", pad=6)
    ax.set_ylabel("CLR")
    ax.set_ylim(0, 0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, rotation=18, ha="right")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35, color="#5B7FA3")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.06),
        handlelength=1.8,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate IEEE-style Figure 2 (xRASA + CLR by model and setting).")
    parser.add_argument(
        "--input",
        default="data/article2/100_Pantun_Eksperimen_1200_xrasa_final.csv",
        help="Input CSV containing model, setting, xrasa_final, clr.",
    )
    parser.add_argument(
        "--out-png",
        default="figures/article2/figure2_xrasa_clr_ieee.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--out-pdf",
        default="figures/article2/figure2_xrasa_clr_ieee.pdf",
        help="Output PDF path.",
    )
    parser.add_argument(
        "--summary-out",
        default="figures/article2/figure2_xrasa_clr_summary.csv",
        help="Optional summary CSV used for plotting.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    needed = ["model", "setting", "xrasa_final", "clr"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["xrasa_final"] = pd.to_numeric(df["xrasa_final"], errors="coerce")
    df["clr"] = pd.to_numeric(df["clr"], errors="coerce")
    df = df[df["model"].isin(MODEL_ORDER) & df["setting"].isin(SETTING_ORDER)].copy()

    summary = compute_summary(df)
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_out, index=False)

    draw_figure(summary, Path(args.out_png), Path(args.out_pdf))

    print(f"input={args.input}")
    print(f"out_png={args.out_png}")
    print(f"out_pdf={args.out_pdf}")
    print(f"summary_out={args.summary_out}")


if __name__ == "__main__":
    main()
