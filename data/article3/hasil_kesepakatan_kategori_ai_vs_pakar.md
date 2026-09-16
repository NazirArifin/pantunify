# Hasil Kesepakatan Kategori AI vs Kategori Pakar

Tanggal perhitungan: 2026-09-16

## Data
- Sumber data: data/article3/Dataset_Final_Klasifikasi.csv
- Jumlah baris terbaca: 6640
- Jumlah pasangan label valid: 6640
- Jumlah baris dengan label kosong/tidak valid: 0

## Metrik Kesepakatan
- Persentase kesepakatan langsung (exact agreement): 0.877108 (87.71%)
- Cohen's Kappa: 0.805307
- Interpretasi Kappa (Landis & Koch): Substansial (substantial)

## Confusion Matrix
Baris = label pakar, kolom = label AI.

| Kategori Pakar \ Kategori AI | Pantun Agama | Pantun Dukacita | Pantun Hiburan | Pantun Kasih Sayang | Pantun Kiasan | Pantun Nasihat | Pantun Sosial Budaya | Pantun Sukacita |
|---|---|---|---|---|---|---|---|---|
| Pantun Agama | 648 | 1 | 1 | 0 | 2 | 32 | 0 | 0 |
| Pantun Dukacita | 0 | 133 | 0 | 1 | 2 | 30 | 0 | 1 |
| Pantun Hiburan | 0 | 4 | 194 | 8 | 3 | 209 | 1 | 3 |
| Pantun Kasih Sayang | 2 | 2 | 8 | 787 | 16 | 157 | 3 | 15 |
| Pantun Kiasan | 0 | 0 | 2 | 0 | 228 | 119 | 1 | 0 |
| Pantun Nasihat | 0 | 6 | 1 | 10 | 13 | 3473 | 0 | 53 |
| Pantun Sosial Budaya | 0 | 0 | 0 | 0 | 0 | 28 | 80 | 0 |
| Pantun Sukacita | 0 | 0 | 0 | 0 | 2 | 79 | 1 | 281 |
