import ast
import json
import statistics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def load_syllable_lists(df: pd.DataFrame) -> pd.Series:
    if {
        "suku_kata_baris_1",
        "suku_kata_baris_2",
        "suku_kata_baris_3",
        "suku_kata_baris_4",
    }.issubset(df.columns):
        return (
            df[
                [
                    "suku_kata_baris_1",
                    "suku_kata_baris_2",
                    "suku_kata_baris_3",
                    "suku_kata_baris_4",
                ]
            ]
            .astype(int)
            .values.tolist()
        )

    if "jumlah_suku_kata" in df.columns:
        return df["jumlah_suku_kata"].apply(ast.literal_eval).apply(lambda x: [int(v) for v in x])

    raise ValueError("Kolom suku kata tidak ditemukan di data/pantun_dataset.csv")


def main() -> None:
    df = pd.read_csv("data/pantun_dataset.csv")
    df["syll_list"] = load_syllable_lists(df)

    all_syll = [x for row in df["syll_list"] for x in row]
    line_df = pd.DataFrame(
        {
            "line_1": [x[0] for x in df["syll_list"]],
            "line_2": [x[1] for x in df["syll_list"]],
            "line_3": [x[2] for x in df["syll_list"]],
            "line_4": [x[3] for x in df["syll_list"]],
        }
    )
    line_long = line_df.melt(var_name="line_position", value_name="syllable_count")

    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df, x="skema_rima", order=["a-a-a-a", "a-b-a-b"])
    ax.set_xlabel("Rhyme schema")
    ax.set_ylabel("Count")
    ax.set_title("Rhyme schema distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "figure1_rhyme_schema.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    ax = sns.histplot(all_syll, bins=range(min(all_syll), max(all_syll) + 2), discrete=True)
    ax.set_xlabel("Syllable count")
    ax.set_ylabel("Frequency")
    ax.set_title("Syllable-count distribution (all lines)")
    plt.tight_layout()
    plt.savefig(out_dir / "figure2_syllable_hist.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 4))
    ax = sns.boxplot(data=line_long, x="line_position", y="syllable_count")
    ax.set_xlabel("Line position")
    ax.set_ylabel("Syllable count")
    ax.set_title("Syllable-count by line position")
    plt.tight_layout()
    plt.savefig(out_dir / "figure3_syllable_boxplot.png", dpi=300)
    plt.close()

    summary = {
        "dataset": "data/pantun_dataset.csv",
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1] - 1),
        "rhyme_counts": {
            "a-a-a-a": int((df["skema_rima"] == "a-a-a-a").sum()),
            "a-b-a-b": int((df["skema_rima"] == "a-b-a-b").sum()),
        },
        "line_position_syllable_profile": {
            f"line_{i}": {
                "min": int(line_df[f"line_{i}"].min()),
                "max": int(line_df[f"line_{i}"].max()),
                "mean": round(float(statistics.mean(line_df[f"line_{i}"])), 2),
            }
            for i in range(1, 5)
        },
    }
    (out_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Charts written to figures/")
    print("Summary written to figures/dataset_summary.json")


if __name__ == "__main__":
    main()