# Pantunify (Pantun Classifier)

Alat otomatis berbasis Python untuk membersihkan, memvalidasi, dan mengklasifikasikan pantun Bahasa Indonesia secara massal. Proyek ini menggunakan pendekatan heuristik untuk menghitung suku kata, kata, dan rima secara akurat.

## Fitur Utama

- **Syllable Counter (Heuristik)**: Penghitung suku kata yang dioptimalkan untuk Bahasa Indonesia (menangani diftong, gugus vokal, dan pengecualian kata).
- **Word Count Filter**: Memastikan tiap baris memiliki panjang kata yang ideal (4-6 kata).
- **Rhyme Scheme Validator**: Mendeteksi pola rima $a-b-a-b$ atau $a-a-a-a$ dengan cerdas.
- **Fuzzy Deduplication**: Menghapus duplikasi pantun yang mirip menggunakan algoritma `difflib`.
- **CLI Interface**: Mudahkan pemrosesan data dengan command line yang dapat dikustomisasi.

## Struktur Proyek

- `src/pantunify/`: Paket utama Python.
  - `utils.py`: Logika dasar (suku kata, rima, pembersihan teks).
  - `classifier.py`: Logika validasi dan deduplikasi.
  - `cli.py`: Antarmuka Command Line.
- `data/`: Direktori untuk menyimpan file teks input (`merged.txt`) dan output.
- `scripts/`: Script utilitas tambahan untuk pembersihan dataset.

## Instalasi

Gunakan `pip` untuk menginstal paket ini dalam mode pengembangan:

```bash
pip install -e .
```

Setelah instalasi, perintah `pantunify` akan tersedia secara global di terminal Anda.

## Cara Penggunaan

### 1. Persiapkan Data
Letakkan dataset pantun Anda di `data/merged.txt` (setiap pantun dipisahkan oleh minimal satu baris kosong).

### 2. Jalankan Perintah
Gunakan CLI bawaan untuk memproses data:

```bash
pantunify
```

Secara default, ini akan membaca `data/merged.txt` dan menghasilkan `data/ok.txt` serta `data/fail.txt`.

### 3. Kustomisasi Parameter
Anda bisa mengatur batas suku kata atau jumlah kata melalui argumen:

```bash
pantunify --min_syllables 7 --max_syllables 13 --input data/my_pantun.txt
```

Atau gunakan `--help` untuk bantuan:
```bash
pantunify --help
```

## Kriteria Validasi Pantun

Agar pantun masuk ke kategori `OK`, harus memenuhi syarat berikut:
1. **Jumlah Suku Kata**: 8 - 12 suku kata per baris.
2. **Jumlah Kata**: 4 - 6 kata per baris.
3. **Pola Rima**: Akhiran baris 1 = baris 3, dan baris 2 = baris 4 (atau semuanya sama).

## Pengembangan

Jika ingin menggunakan fungsi di dalam kode Python Anda sendiri:

```python
from pantunify import validate_pantun

lines = [
    "Masak air biar matang",
    "Air dimasukkan ke dalam gelas",
    "Niat belajar jadi terhalang",
    "Karena hati sedang malas"
]

is_valid, reason = validate_pantun(lines)
print(f"Valid: {is_valid}, Alasan: {reason}")
```

## Kontribusi

Kontribusi dipersilakan! Silakan buka *Issue* atau kirimkan *Pull Request*.

---
Dibuat untuk keperluan pengolahan data rima dan sastra Indonesia oleh Nazir Arifin.
