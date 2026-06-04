#!/usr/bin/env python3
import sys
import re
from pathlib import Path

def count_pantuns(file_path):
    """
    Menghitung jumlah pantun dalam sebuah file.
    Pantun dianggap valid jika terdiri dari tepat 4 baris teks 
    yang dipisahkan oleh satu atau lebih baris kosong.
    """
    p = Path(file_path)
    if not p.exists():
        return None
    
    content = p.read_text(encoding='utf-8')
    # Pisahkan berdasarkan baris kosong
    blocks = re.split(r'\n\s*\n', content.strip())
    
    count = 0
    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if len(lines) == 4:
            count += 1
    return count

def main():
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        # Default ke data/ok.txt jika tidak ada argumen
        files = ['data/ok.txt']
    
    for f_path in files:
        count = count_pantuns(f_path)
        if count is not None:
            print(f"{f_path}: {count} pantun")
        else:
            print(f"Error: File '{f_path}' tidak ditemukan.")

if __name__ == "__main__":
    main()
