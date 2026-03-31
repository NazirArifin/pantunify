# Pantunify (Pantun Classifier)

Alat otomatis berbasis Python untuk membersihkan, memvalidasi, dan mengklasifikasikan pantun Bahasa Indonesia secara massal. Proyek ini menggunakan pendekatan heuristik untuk menghitung suku kata, kata, dan rima secara akurat.

## Fitur Utama

- **Syllable Counter (Heuristik)**: Penghitung suku kata yang dioptimalkan untuk Bahasa Indonesia (menangani diftong, gugus vokal, dan pengecualian kata).
- **Word Count Filter**: Memastikan tiap baris memiliki panjang kata yang ideal (4-6 kata).
- **Rhyme Scheme Validator**: Mendeteksi pola rima $a-b-a-b$ atau $a-a-a-a$ dengan cerdas (membedakan diftong seperti 'au' dengan vokal tunggal 'u').
- **Fuzzy Deduplication**: Menghapus duplikasi pantun yang mirip menggunakan algoritma `difflib`.
- **Text Cleaning**: Membersihkan nomor urut, tanda baca, dan whitespace secara otomatis dari dataset mentah.

## Struktur Proyek

- `classify_merged.py`: Script utama untuk memproses dataset `merged.txt` dan memisahkan hasil ke `ok.txt` dan `fail.txt`.
- `pantun_utils.py`: Modul inti yang berisi logika perhitungan suku kata, kata, dan rima.
- `dataset/`: Direktori tempat menyimpan file teks input dan output.
- `test.py`: Script sederhana untuk menguji fungsi utilitas.

## Cara Penggunaan

1. Pastikan Anda memiliki Python 3 terinstall.
2. Letakkan dataset pantun Anda di `dataset/merged.txt` (setiap pantun dipisahkan oleh satu baris kosong).
3. Jalankan perintah:
   ```bash
   python3 classify_merged.py
   ```
4. Hasil validasi akan tersedia di:
   - `dataset/ok.txt`: Pantun yang memenuhi semua syarat.
   - `dataset/fail.txt`: Pantun yang tidak lulusa validasi.

## Kriteria Validasi Pantun

Agar pantun masuk ke kategori `OK`, harus memenuhi syarat berikut:
1. **Jumlah Suku Kata**: 8 - 12 suku kata per baris.
2. **Jumlah Kata**: 4 - 6 kata per baris.
3. **Pola Rima**: Akhiran baris 1 = baris 3, dan baris 2 = baris 4 (atau semuanya sama).

## Kontribusi

Kontribusi dipersilakan! Silakan buka *Issue* atau kirimkan *Pull Request*.

---
Dibuat untuk keperluan pengolahan data rima dan sastra Indonesia.
