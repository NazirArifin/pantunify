#!/usr/bin/env python3
"""Clean tags (<BOS>, <CONTENT>, <CLS>, <EOS>) from a source file and append pantun
in groups of 4 lines separated by a blank line to a destination file.

Usage:
  python3 scripts/clean_merge.py --src data/anggi/sampiran.txt --dst data/merged.txt [--backup] [--dry-run]
"""
import argparse
import re
from pathlib import Path
import shutil
import time

PATTERN = re.compile(r'<\s*BOS\s*>(.*?)<\s*CLS\s*>(.*?)<\s*CONTENT\s*>(.*?)<\s*CLS\s*>(.*?)<\s*EOS\s*>', re.S)
TAG_RE = re.compile(r'<\s*(BOS|CONTENT|CLS|EOS)\s*>', re.I)


def parse_line(line: str):
    """Return list of 4 strings for a pantun parsed from a line.
    If explicit tags present, extract groups; otherwise split cleaned text into 4 chunks.
    """
    m = PATTERN.search(line)
    if m:
        parts = [re.sub(r"\s+", " ", g).strip() for g in m.groups()]
        # ensure exactly 4
        if len(parts) >= 4:
            return parts[:4]
        while len(parts) < 4:
            parts.append("")
        return parts
    # fallback: remove any tags and chunk words roughly equally into 4 lines
    cleaned = TAG_RE.sub('', line)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return []
    words = cleaned.split()
    k = len(words)
    base = k // 4
    sizes = [base] * 4
    for i in range(k % 4):
        sizes[i] += 1
    parts = []
    idx = 0
    for s in sizes:
        parts.append(' '.join(words[idx:idx + s]).strip())
        idx += s
    return parts


def build_blocks(src_path: Path):
    lines = src_path.read_text(encoding='utf-8').splitlines()
    blocks = []  # each block is list of 4 lines
    for raw in lines:
        if not raw.strip():
            continue
        parts = parse_line(raw)
        if not parts:
            continue
        # ensure 4 elements
        if len(parts) < 4:
            parts += [""] * (4 - len(parts))
        blocks.append(parts[:4])
    return blocks


def append_blocks(dst_path: Path, blocks, backup=False):
    if backup and dst_path.exists():
        stamp = time.strftime('%Y%m%d-%H%M%S')
        bak = dst_path.with_name(dst_path.name + f'.bak.{stamp}')
        shutil.copy2(dst_path, bak)
        print('Backup created:', bak)
    with dst_path.open('a', encoding='utf-8') as f:
        for b in blocks:
            for line in b:
                f.write(line + '\n')
            f.write('\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', '-s', default='data/anggi/sampiran.txt')
    p.add_argument('--dst', '-d', default='data/merged.txt')
    p.add_argument('--backup', action='store_true', help='Make a backup of destination before appending')
    p.add_argument('--dry-run', action='store_true', help='Do not write; just show summary')
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists():
        print('Source not found:', src)
        raise SystemExit(2)

    blocks = build_blocks(src)
    print(f'Parsed {len(blocks)} pantun blocks from', src)

    if args.dry_run:
        # print first 3 blocks as sample
        print('Dry-run: showing up to first 3 pantun blocks:')
        for i, b in enumerate(blocks[:3], 1):
            print('--- pantun', i, '---')
            for line in b:
                print(line)
            print()
        print('No changes written (dry-run).')
        return

    # ensure dst parent exists
    dst.parent.mkdir(parents=True, exist_ok=True)
    append_blocks(dst, blocks, backup=args.backup)
    print('Appended', len(blocks), 'pantun blocks to', dst)


if __name__ == '__main__':
    main()
