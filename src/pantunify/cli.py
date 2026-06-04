import re
import argparse
from pathlib import Path
from .classifier import validate_pantun, deduplicate_pantuns

def main():
    parser = argparse.ArgumentParser(description="Pantunify CLI - Alat pembersih dan validasi pantun.")
    parser.add_argument("--input", "-i", type=Path, default=Path("data/merged.txt"), help="File input dataset (default: data/merged.txt)")
    parser.add_argument("--ok", "-o", type=Path, default=Path("data/ok.txt"), help="File output pantun valid (default: data/ok.txt)")
    parser.add_argument("--fail", "-f", type=Path, default=Path("data/fail.txt"), help="File output pantun gagal (default: data/fail.txt)")
    parser.add_argument("--min_syllables", type=int, default=8)
    parser.add_argument("--max_syllables", type=int, default=12)
    parser.add_argument("--min_words", type=int, default=4)
    parser.add_argument("--max_words", type=int, default=6)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"File {args.input} tidak ditemukan.")
        return

    text = args.input.read_text(encoding='utf-8')
    raw_blocks = re.split(r'\n\s*\n+', text)
    
    # Kumpulkan blok yang bukan kosong
    blocks = []
    for b in raw_blocks:
        lines = [l.strip() for l in b.splitlines() if l.strip()]
        if len(lines) == 4:
            blocks.append(lines)

    # Langkah 1: Deduplikasi
    unique_blocks, duplicates = deduplicate_pantuns(blocks)
    print(f"Total pantun unik: {len(unique_blocks)} (Terbuang {len(duplicates)} duplikat)")

    # Langkah 2: Validasi
    ok = []
    fail = []

    for lines in unique_blocks:
        is_valid, reason = validate_pantun(
            lines, 
            min_syllables=args.min_syllables, 
            max_syllables=args.max_syllables,
            min_words=args.min_words,
            max_words=args.max_words
        )
        if is_valid:
            ok.append(lines)
        else:
            fail.append(lines)

    # Langkah 3: Tulis ke file
    def write_blocks(path, blocks):
        with path.open('w', encoding='utf-8') as f:
            for p in blocks:
                for l in p:
                    f.write(l + '\n')
                f.write('\n')

    write_blocks(args.ok, ok)
    write_blocks(args.fail, fail)

    print(f"Hasil pemrosesan dipindahkan ke:")
    print(f" - {args.ok}: {len(ok)} pantun.")
    print(f" - {args.fail}: {len(fail)} pantun.")

if __name__ == '__main__':
    main()
