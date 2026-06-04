#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

from pantunify.classifier import validate_pantun
from pantunify.utils import clean_line, count_syllables, last_syllable


def parse_blocks(text: str) -> list[list[str]]:
    """Split text by blank lines into blocks of non-empty stripped lines."""
    raw_blocks = re.split(r"\n\s*\n", text.strip())
    blocks: list[list[str]] = []
    for block in raw_blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if lines:
            blocks.append(lines)
    return blocks


def rhyme_scheme(lines: list[str]) -> str:
    """Create a rhyme label like a-b-a-b based on line endings."""
    endings = [last_syllable(line) for line in lines]
    label_map: dict[str, str] = {}
    labels: list[str] = []
    next_label_ord = ord("a")

    for end in endings:
        key = end or "_"
        if key not in label_map:
            label_map[key] = chr(next_label_ord)
            next_label_ord += 1
        labels.append(label_map[key])

    return "-".join(labels)


def to_row(row_id: int, lines: list[str]) -> dict[str, str | int]:
    is_valid, alasan = validate_pantun(lines)
    is_valid_4_lines = len(lines) == 4

    # Keep row shape compatible with pantun_dataset.csv.
    base_lines = lines[:4] if len(lines) >= 4 else lines + [""] * (4 - len(lines))
    text_pantun = "\\n".join(base_lines)
    baris_sampiran = "\\n".join(base_lines[:2])
    baris_isi = "\\n".join(base_lines[2:])

    if is_valid_4_lines:
        skema = rhyme_scheme(base_lines)
        syllables = [count_syllables(line) for line in base_lines]
        word_counts = [len(clean_line(line).split()) for line in base_lines]
    else:
        skema = ""
        syllables = ["", "", "", ""]
        word_counts = ["", "", "", ""]

    return {
        "id": row_id,
        "text_pantun": text_pantun,
        "baris_sampiran": baris_sampiran,
        "baris_isi": baris_isi,
        "skema_rima": skema,
        "suku_kata_baris_1": syllables[0],
        "suku_kata_baris_2": syllables[1],
        "suku_kata_baris_3": syllables[2],
        "suku_kata_baris_4": syllables[3],
        "jumlah_kata_baris_1": word_counts[0],
        "jumlah_kata_baris_2": word_counts[1],
        "jumlah_kata_baris_3": word_counts[2],
        "jumlah_kata_baris_4": word_counts[3],
        "alasan": "OK" if is_valid else alasan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert fail.txt into audit-friendly CSV format"
    )
    parser.add_argument("--input", default="data/fail.txt", help="Path to input txt file")
    parser.add_argument(
        "--output",
        default="data/excluded_pantun_dataset.csv",
        help="Path to output csv file",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    text = input_path.read_text(encoding="utf-8")
    blocks = parse_blocks(text)

    rows = [to_row(i, lines) for i, lines in enumerate(blocks, start=1)]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "text_pantun",
        "baris_sampiran",
        "baris_isi",
        "skema_rima",
        "suku_kata_baris_1",
        "suku_kata_baris_2",
        "suku_kata_baris_3",
        "suku_kata_baris_4",
        "jumlah_kata_baris_1",
        "jumlah_kata_baris_2",
        "jumlah_kata_baris_3",
        "jumlah_kata_baris_4",
        "alasan",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Berhasil menulis {len(rows)} baris ke: {output_path}")


if __name__ == "__main__":
    main()
