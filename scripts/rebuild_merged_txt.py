#!/usr/bin/env python3
import argparse
import re
from datetime import datetime
from pathlib import Path


def normalize_line(text: str) -> str:
    s = text.strip()
    # Normalize trailing punctuation so mixed sources (e.g. 4.txt) stay consistent.
    s = re.sub(r"[\.,;:!\?，。、]+$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def split_plain_blocks(text: str) -> list[list[str]]:
    raw_blocks = re.split(r"\n\s*\n+", text.strip())
    blocks: list[list[str]] = []
    for raw in raw_blocks:
        lines = [normalize_line(x) for x in raw.splitlines() if x.strip()]
        if len(lines) == 4:
            blocks.append(lines)
    return blocks


def parse_tagged_line(line: str) -> list[str] | None:
    s = " ".join(line.strip().split())
    pattern = (
        r"^<BOS>\s*(.*?)\s*<CLS>\s*(.*?)\s*<CONTENT>\s*(.*?)\s*<CLS>\s*(.*?)\s*<EOS>$"
    )
    m = re.match(pattern, s)
    if not m:
        return None
    return [normalize_line(m.group(i)) for i in range(1, 5)]


def parse_tagged_file(path: Path) -> list[list[str]]:
    blocks: list[list[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        block = parse_tagged_line(line)
        if block and len(block) == 4:
            blocks.append(block)
    return blocks


def write_blocks(path: Path, blocks: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for block in blocks:
            for line in block:
                f.write(line + "\n")
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild data/merged.txt from multiple source files")
    parser.add_argument("--output", default="data/merged.txt", help="Output merged txt")
    parser.add_argument("--backup", action="store_true", help="Backup output before overwrite")
    parser.add_argument(
        "--anggi",
        default="data/anggi/sampiran.txt",
        help="Tagged source file from anggi",
    )
    parser.add_argument(
        "--plain",
        nargs="+",
        default=["data/1.txt", "data/2.txt", "data/3.txt", "data/4.txt", "data/5.txt"],
        help="Plain 4-line-block source files",
    )
    args = parser.parse_args()

    output = Path(args.output)
    anggi = Path(args.anggi)
    plain_files = [Path(p) for p in args.plain]

    if not anggi.exists():
        raise FileNotFoundError(f"Anggi source not found: {anggi}")

    for p in plain_files:
        if not p.exists():
            raise FileNotFoundError(f"Plain source not found: {p}")

    if args.backup and output.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = output.with_name(f"{output.name}.bak.{stamp}")
        backup_path.write_text(output.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Backup created: {backup_path}")

    all_blocks: list[list[str]] = []

    anggi_blocks = parse_tagged_file(anggi)
    all_blocks.extend(anggi_blocks)

    plain_counts: list[tuple[str, int]] = []
    for p in plain_files:
        blocks = split_plain_blocks(p.read_text(encoding="utf-8"))
        plain_counts.append((str(p), len(blocks)))
        all_blocks.extend(blocks)

    output.parent.mkdir(parents=True, exist_ok=True)
    write_blocks(output, all_blocks)

    print(f"anggi_blocks={len(anggi_blocks)}")
    for name, count in plain_counts:
        print(f"plain_blocks[{name}]={count}")
    print(f"total_blocks={len(all_blocks)}")
    print(f"written={output}")


if __name__ == "__main__":
    main()
