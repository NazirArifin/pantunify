# Pantunify: A Rule-Based Toolkit for Indonesian Pantun Preprocessing and Filtering

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Pantunify merupakan toolkit berbasis Python untuk mendukung kurasi korpus pantun Indonesia secara sistematis. Sistem ini mengintegrasikan pembersihan teks, validasi formal pantun, deduplikasi exact-fuzzy, serialisasi dataset terstruktur, serta pembuatan visualisasi deskriptif. Pendekatan yang digunakan bersifat rule-based agar keputusan filtering dapat ditelusuri dan direproduksi.

## Research-Oriented Objectives

Pengembangan toolkit ini diarahkan untuk memenuhi kebutuhan pemrosesan data pada studi komputasional sastra lisan, dengan sasaran berikut:

1. Menstandarkan korpus pantun mentah menjadi unit empat baris yang dapat dianalisis.
2. Menegakkan batasan formal pantun melalui validasi suku kata dan pola rima.
3. Memisahkan data lolos dan data eksklusi dalam format yang audit-friendly.
4. Menyediakan dataset tabular dengan fitur per baris untuk analisis kuantitatif lanjutan.
5. Menghasilkan artefak visual dan ringkasan statistik untuk kebutuhan pelaporan ilmiah.

## Validation Criteria

Setiap kandidat pantun pada jalur utama dievaluasi menggunakan kriteria berikut:

- jumlah baris tepat empat,
- jumlah suku kata per baris berada pada rentang 8-12,
- pola rima valid (a-b-a-b atau a-a-a-a),
- filtering tambahan berbasis penanda dialektal,
- deduplikasi exact dan fuzzy untuk mengendalikan redundansi korpus.

## Data Processing Workflow

Alur proses yang direkomendasikan adalah sebagai berikut:

1. Menyiapkan korpus gabungan pada file data/merged.txt.
2. Menjalankan proses deduplikasi dan validasi untuk menghasilkan:
   - data/ok.txt (pantun lolos),
   - data/fail.txt (pantun tidak lolos).
3. Mengonversi data/ok.txt menjadi dataset utama data/pantun_dataset.csv.
4. Mengonversi data/fail.txt menjadi dataset audit data/excluded_pantun_dataset.csv.
5. Menghasilkan visualisasi serta ringkasan statistik dataset.

Jika ingin menghitung ulang statistik dataset saja, jalankan:

```bash
python scripts/calc_dataset_stats.py
```

Skrip ini membaca data/pantun_dataset.csv dan mencetak ringkasan total data, distribusi skema rima, observasi suku kata, rentang dan rata-rata global, serta profil suku kata per posisi baris.

## Installation

```bash
pip install -e .
```

## Core Commands

Filtering utama korpus:

```bash
pantunify --input data/merged.txt --ok data/ok.txt --fail data/fail.txt
```

Konversi data valid ke dataset utama:

```bash
python scripts/ok_txt_to_csv.py --input data/ok.txt --output data/pantun_dataset.csv
```

Konversi data tidak lolos ke dataset audit:

```bash
python scripts/fail_txt_to_csv.py --input data/fail.txt --output data/excluded_pantun_dataset.csv
```

Pembuatan grafik deskriptif dan ringkasan statistik:

```bash
python scripts/chart_generate.py
```

Keluaran visual ditulis pada folder figures dengan artefak berikut:

- figure1_rhyme_schema.png
- figure2_syllable_hist.png
- figure3_syllable_boxplot.png
- dataset_summary.json

## Primary Dataset Schema

Dataset utama berada pada data/pantun_dataset.csv dan terdiri dari 17 variabel:

1. id
2. text_pantun
3. baris_sampiran
4. baris_isi
5. skema_rima
6. rima_akhir_baris_1
7. rima_akhir_baris_2
8. rima_akhir_baris_3
9. rima_akhir_baris_4
10. suku_kata_baris_1
11. suku_kata_baris_2
12. suku_kata_baris_3
13. suku_kata_baris_4
14. jumlah_kata_baris_1
15. jumlah_kata_baris_2
16. jumlah_kata_baris_3
17. jumlah_kata_baris_4

Kolom rima_akhir_baris_1 sampai rima_akhir_baris_4 berisi hasil ekstraksi rima akhir per baris dari fungsi last_syllable.

## Notes for Scientific Reporting

- Untuk deskripsi dataset inti, gunakan data/pantun_dataset.csv.
- Untuk audit terhadap entri yang tidak lolos validasi, gunakan data/excluded_pantun_dataset.csv.
- Folder data/ai_sampiran merepresentasikan material riset berbeda dan tidak wajib dimasukkan dalam deskripsi dataset utama.

## Contribution

Kontribusi untuk peningkatan metodologi validasi linguistik, deduplikasi, dan dokumentasi ilmiah sangat terbuka.
