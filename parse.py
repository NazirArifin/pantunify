import re
from pathlib import Path

# --- Gabung dan bersihkan pantun dari 1-5.txt ---
files = [
	'dataset/1.txt',
	'dataset/2.txt',
	'dataset/3.txt',
	'dataset/4.txt',
	'dataset/5.txt',
]

pantun_set = set()

for file in files:
	with open(file, encoding='utf-8') as f:
		content = f.read()
		# Pisahkan pantun dengan 2 baris kosong atau 1 baris kosong
		pantun_blocks = re.split(r'\n{2,}|\r\n{2,}', content)
		for block in pantun_blocks:
			# Ambil hanya baris yang tidak kosong
			lines = [l.strip() for l in block.splitlines() if l.strip()]
			if len(lines) == 4:
				# Bersihkan setiap baris: hapus angka di awal, hapus tanda baca di akhir
				clean_lines = []
				for l in lines:
					l = re.sub(r'^\d+\.\s*', '', l)  # hapus angka di awal
					l = re.sub(r'[.,;:!?]+$', '', l)    # hapus tanda baca di akhir
					clean_lines.append(l)
				pantun_tuple = tuple(clean_lines)
				pantun_set.add(pantun_tuple)

# Simpan ke file baru
with open('dataset/merged.txt', 'w', encoding='utf-8') as out:
	for pantun in sorted(pantun_set):
		for line in pantun:
			out.write(line + '\n')
		out.write('\n')

print(f"Total pantun unik: {len(pantun_set)}")