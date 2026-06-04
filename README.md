# Pantunify: A Rule-Based Toolkit for Indonesian Pantun Preprocessing and Filtering

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Pantunify merupakan toolkit berbasis Python untuk mendukung kurasi korpus pantun Indonesia secara sistematis. Sistem ini mengintegrasikan pembersihan teks, validasi formal pantun, deduplikasi exact-fuzzy, serialisasi dataset terstruktur, serta pembuatan visualisasi deskriptif. Pendekatan yang digunakan bersifat rule-based agar keputusan filtering dapat ditelusuri dan direproduksi.

## Research-Oriented Objectives

Pengembangan toolkit ini diarahkan untuk memenuhi kebutuhan pemrosesan data pada studi komputasional sastra lisan, dengan sasaran berikut:

1. Menstandarkan korpus pantun mentah menjadi unit empat baris yang dapat dianalisis.
2. Menegakkan batasan formal pantun melalui validasi suku kata, jumlah kata, dan pola rima.
3. Memisahkan data lolos dan data eksklusi dalam format yang audit-friendly.
4. Menyediakan dataset tabular dengan fitur per baris untuk analisis kuantitatif lanjutan.
5. Menghasilkan artefak visual dan ringkasan statistik untuk kebutuhan pelaporan ilmiah.

## Validation Criteria

Setiap kandidat pantun pada jalur utama dievaluasi menggunakan kriteria berikut:

- jumlah baris tepat empat,
- jumlah suku kata per baris berada pada rentang 8-12,
- jumlah kata per baris berada pada rentang 4-6,
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

Dataset utama berada pada data/pantun_dataset.csv dan terdiri dari 13 variabel:

1. id
2. text_pantun
3. baris_sampiran
4. baris_isi
5. skema_rima
6. suku_kata_baris_1
7. suku_kata_baris_2
8. suku_kata_baris_3
9. suku_kata_baris_4
10. jumlah_kata_baris_1
11. jumlah_kata_baris_2
12. jumlah_kata_baris_3
13. jumlah_kata_baris_4

## Current Dataset Snapshot

Berdasarkan versi terkini dataset utama:

- jumlah observasi: 5636,
- jumlah variabel: 13,
- distribusi rima a-b-a-b: 4252,
- distribusi rima a-a-a-a: 1384.

## Notes for Scientific Reporting

- Untuk deskripsi dataset inti, gunakan data/pantun_dataset.csv.
- Untuk audit terhadap entri yang tidak lolos validasi, gunakan data/excluded_pantun_dataset.csv.
- Folder data/ai_sampiran merepresentasikan material riset berbeda dan tidak wajib dimasukkan dalam deskripsi dataset utama.

## Contribution

Kontribusi untuk peningkatan metodologi validasi linguistik, deduplikasi, dan dokumentasi ilmiah sangat terbuka.
