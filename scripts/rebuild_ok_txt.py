#!/usr/bin/env python3
import argparse
import re
from datetime import datetime
from pathlib import Path

from pantunify.classifier import deduplicate_pantuns, validate_pantun


def split_blocks(text: str) -> list[list[str]]:
    raw_blocks = re.split(r"\n\s*\n+", text.strip())
    blocks: list[list[str]] = []
    for block in raw_blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) == 4:
            blocks.append(lines)
    return blocks


def write_blocks(path: Path, blocks: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for pantun in blocks:
            for line in pantun:
                f.write(line + "\n")
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild ok.txt with exactly N valid pantun from merged source"
    )
    parser.add_argument("--input", default="data/merged.txt", help="Source txt path")
    parser.add_argument("--output", default="data/ok.txt", help="Target ok txt path")
    parser.add_argument("--target", type=int, default=5349, help="Target pantun count")
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Backup existing output before overwrite",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input tidak ditemukan: {input_path}")

    if args.backup and output_path.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = output_path.with_name(f"{output_path.name}.bak.{stamp}")
        backup_path.write_text(output_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Backup dibuat: {backup_path}")

    merged_blocks = split_blocks(input_path.read_text(encoding="utf-8"))
    unique_blocks, duplicate_blocks = deduplicate_pantuns(merged_blocks)

    valid_blocks: list[list[str]] = []
    invalid_count = 0
    for block in unique_blocks:
        is_valid, _ = validate_pantun(block)
        if is_valid:
            valid_blocks.append(block)
        else:
            invalid_count += 1

    if len(valid_blocks) < args.target:
        raise ValueError(
            f"Pantun valid hanya {len(valid_blocks)}, kurang dari target {args.target}."
        )

    selected = valid_blocks[: args.target]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_blocks(output_path, selected)

    # Verify output
    written_blocks = split_blocks(output_path.read_text(encoding="utf-8"))
    still_invalid = 0
    for block in written_blocks:
        is_valid, _ = validate_pantun(block)
        if not is_valid:
            still_invalid += 1

    print(f"Sumber blok mentah: {len(merged_blocks)}")
    print(f"Unik: {len(unique_blocks)} | Duplikat: {len(duplicate_blocks)}")
    print(f"Valid: {len(valid_blocks)} | Invalid: {invalid_count}")
    print(f"Ditulis ke {output_path}: {len(written_blocks)}")
    print(f"Invalid setelah tulis: {still_invalid}")


if __name__ == "__main__":
    main()
