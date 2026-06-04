#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

from pantunify.utils import clean_line, count_syllables, last_syllable


def split_blocks(text: str) -> list[list[str]]:
    """Split text into pantun blocks and keep only valid 4-line blocks."""
    raw_blocks = re.split(r"\n\s*\n", text.strip())
    blocks: list[list[str]] = []
    for block in raw_blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) == 4:
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
    syllables = [count_syllables(line) for line in lines]
    word_counts = [len(clean_line(line).split()) for line in lines]
    text_pantun = "\\n".join(lines)
    return {
        "id": row_id,
        "text_pantun": text_pantun,
        "baris_sampiran": "\\n".join(lines[:2]),
        "baris_isi": "\\n".join(lines[2:]),
        "skema_rima": rhyme_scheme(lines),
        "suku_kata_baris_1": syllables[0],
        "suku_kata_baris_2": syllables[1],
        "suku_kata_baris_3": syllables[2],
        "suku_kata_baris_4": syllables[3],
        "jumlah_kata_baris_1": word_counts[0],
        "jumlah_kata_baris_2": word_counts[1],
        "jumlah_kata_baris_3": word_counts[2],
        "jumlah_kata_baris_4": word_counts[3],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert pantun txt to structured CSV")
    parser.add_argument("--input", default="data/ok.txt", help="Path to input txt file")
    parser.add_argument(
        "--output", default="data/pantun_dataset.csv", help="Path to output csv file"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of pantun records to write (default: all records)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    text = input_path.read_text(encoding="utf-8")
    blocks = split_blocks(text)

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit harus lebih dari 0 jika diisi.")

    if args.limit is None:
        selected = blocks
    else:
        if len(blocks) < args.limit:
            raise ValueError(
                f"Pantun valid hanya {len(blocks)} blok, kurang dari limit {args.limit}."
            )
        selected = blocks[: args.limit]

    rows = [to_row(i, lines) for i, lines in enumerate(selected, start=1)]
    rows.sort(key=lambda row: str(row["text_pantun"]).casefold())
    for i, row in enumerate(rows, start=1):
        row["id"] = i

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
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Berhasil menulis {len(rows)} baris ke: {output_path}")


if __name__ == "__main__":
    main()