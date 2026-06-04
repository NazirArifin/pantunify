import re
from pathlib import Path

def merge_datasets(input_files, output_file):
    pantun_set = set()
    
    for file_path in input_files:
        p = Path(file_path)
        if not p.exists():
            print(f"Skipping {file_path}: File not found.")
            continue
            
        with p.open(encoding='utf-8') as f:
            content = f.read()
            # Pisahkan pantun dengan 2 baris kosong atau 1 baris kosong
            pantun_blocks = re.split(r'\n{2,}|\r\n{2,}', content)
            for block in pantun_blocks:
                lines = [l.strip() for l in block.splitlines() if l.strip()]
                if len(lines) == 4:
                    clean_lines = []
                    for l in lines:
                        l = re.sub(r'^\d+\.\s*', '', l)  # hapus angka di awal
                        l = re.sub(r'[.,;:!?]+$', '', l)    # hapus tanda baca di akhir
                        clean_lines.append(l)
                    pantun_tuple = tuple(clean_lines)
                    pantun_set.add(pantun_tuple)

    with Path(output_file).open('w', encoding='utf-8') as out:
        for pantun in sorted(pantun_set):
            for line in pantun:
                out.write(line + '\n')
            out.write('\n')
    
    print(f"Berhasil menggabungkan dataset ke {output_file}. Total pantun unik: {len(pantun_set)}")

if __name__ == "__main__":
    files = [f'data/{i}.txt' for i in range(1, 6)]
    merge_datasets(files, 'data/merged.txt')
