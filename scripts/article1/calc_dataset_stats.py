import argparse
from pathlib import Path

import pandas as pd


def compute_stats(df: pd.DataFrame) -> dict:
    rhyme_col = "skema_rima"
    rhyme_end_cols = [f"rima_akhir_baris_{i}" for i in range(1, 5)]
    syll_cols = [f"suku_kata_baris_{i}" for i in range(1, 5)]
    word_cols = [f"jumlah_kata_baris_{i}" for i in range(1, 5)]

    syll_values = df[syll_cols].to_numpy().ravel()
    word_values = df[word_cols].to_numpy().ravel()

    syll_series = pd.Series(syll_values).dropna()
    word_series = pd.Series(word_values).dropna()

    line_profile = {}
    for i, col in enumerate(syll_cols, start=1):
        line_vals = pd.Series(df[col]).dropna()
        line_profile[f"line_{i}"] = {
            "min": int(line_vals.min()),
            "max": int(line_vals.max()),
            "mean": float(line_vals.mean()),
        }

    rhyme_end_profile = {
        col: sorted({str(value) for value in df[col].dropna().tolist()})
        for col in rhyme_end_cols
        if col in df.columns
    }

    return {
        "total_data": int(len(df)),
        "skema_rima_a_a_a_a": int((df[rhyme_col] == "a-a-a-a").sum()),
        "skema_rima_a_b_a_b": int((df[rhyme_col] == "a-b-a-b").sum()),
        "total_observasi_suku_kata": int(len(syll_series)),
        "rentang_suku_kata_global": (int(syll_series.min()), int(syll_series.max())),
        "rata_rata_suku_kata_global": float(syll_series.mean()),
        "rentang_jumlah_kata_global": (int(word_series.min()), int(word_series.max())),
        "rata_rata_jumlah_kata_global": float(word_series.mean()),
        "line_position_syllable_profile": line_profile,
        "rima_akhir_columns": rhyme_end_profile,
    }


def print_report(stats: dict) -> None:
    print("| Statistik | Nilai |")
    print("|---|---:|")
    print(f"| Total data | {stats['total_data']} |")
    print(f"| skema_rima = a-a-a-a | {stats['skema_rima_a_a_a_a']} |")
    print(f"| skema_rima = a-b-a-b | {stats['skema_rima_a_b_a_b']} |")
    print(f"| Total observasi suku kata | {stats['total_observasi_suku_kata']} |")
    print(
        "| Rentang suku kata global | "
        f"{stats['rentang_suku_kata_global'][0]}-{stats['rentang_suku_kata_global'][1]} |"
    )
    print(f"| Rata-rata suku kata global | {stats['rata_rata_suku_kata_global']:.2f} |")
    print(
        "| Rentang jumlah kata global | "
        f"{stats['rentang_jumlah_kata_global'][0]}-{stats['rentang_jumlah_kata_global'][1]} |"
    )
    print(f"| Rata-rata jumlah kata global | {stats['rata_rata_jumlah_kata_global']:.2f} |")

    print()
    print("Line-Position Syllable Profile")
    print("| Posisi baris | Min | Max | Mean |")
    print("|---|---:|---:|---:|")
    for i in range(1, 5):
        row = stats["line_position_syllable_profile"][f"line_{i}"]
        print(f"| Line {i} | {row['min']} | {row['max']} | {row['mean']:.2f} |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hitung statistik ringkas data pantun.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/pantun_dataset.csv"),
        help="Path CSV input (default: data/pantun_dataset.csv)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    stats = compute_stats(df)
    print_report(stats)


if __name__ == "__main__":
    main()