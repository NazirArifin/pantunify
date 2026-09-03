#!/usr/bin/env python3
import argparse
import csv
from collections import Counter
from pathlib import Path


def normalize_reason(row: dict[str, str]) -> str:
    reason = (row.get("alasan") or row.get("reason") or "").strip()
    if not reason:
        return "Tidak diketahui"
    return reason.split(":", 1)[0].strip() or "Tidak diketahui"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reason_counts.csv from excluded pantun dataset"
    )
    parser.add_argument(
        "--input",
        default="data/excluded_pantun_dataset.csv",
        help="Path to excluded pantun CSV input",
    )
    parser.add_argument(
        "--output",
        default="data/reason_counts.csv",
        help="Path to output reason counts CSV",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file tidak ditemukan: {input_path}")

    with input_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"Tidak ada baris data di {input_path}")

    counts = Counter(normalize_reason(row) for row in rows)
    total = sum(counts.values())
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "reason", "count", "percentage"])
        for rank, (reason, count) in enumerate(ordered, start=1):
            percentage = (count / total * 100) if total else 0.0
            writer.writerow([rank, reason, count, f"{percentage:.2f}"])

    print(f"Berhasil menulis {len(ordered)} kategori ke: {output_path}")


if __name__ == "__main__":
    main()