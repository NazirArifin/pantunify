# Dataset Construction: Ringkasan Statistik

Tanggal analisis: 2026-09-06

## 1) Distribusi skema rima di corpus utama (N = 6640)

Sumber: `data/pantun_dataset.csv`

- Total pantun: 6640
- `a-b-a-b`: 4945 (74.47%)
- `a-a-a-a`: 1695 (25.53%)

## 2) Distribusi skema rima di subset eksperimen (N = 100)

Sumber: `data/article2/100_Pantun_Eksperimen.csv` (100 ID unik, semuanya cocok dengan corpus utama)

- Total sampel: 100
- `a-b-a-b`: 100 (100.00%)
- `a-a-a-a`: 0 (0.00%)

## 3) Cek konsistensi klaim "stratified random"

Jika sampel 100 benar-benar mengikuti proporsi corpus (tanpa replacement), peluang mendapatkan 0 item `a-a-a-a` adalah:

- Hypergeometric exact: 1.223293397739e-13
- Binomial approximation: 1.584242488034e-13

Interpretasi: peluang ini sangat kecil, sehingga subset 100 saat ini tidak konsisten dengan klaim stratified proporsional dua skema.

## 4) Justifikasi metodologis ukuran sampel 100 dari 6640

Ukuran 100 masih masuk akal untuk benchmark komparatif awal (pilot benchmark), karena:

1. Unit evaluasi bukan hanya 100 pantun asli, tetapi 100 × jumlah konfigurasi model/prompt.
2. Dalam eksperimen ini, 100 input menghasilkan 1200 output model yang sudah cukup untuk uji beda antar model/setting.
3. Beban anotasi semantik (LLM judge/human calibration) meningkat linear terhadap jumlah output; N=100 menyeimbangkan biaya, waktu, dan replikasi eksperimen.
4. Untuk estimasi proporsi murni, n=100 memiliki margin of error 95% sekitar ±9.8 poin persentase (worst-case), sehingga cocok untuk tahap benchmark komparatif, bukan klaim prevalensi populasi yang sangat presisi.

## 5) Usulan teks pengganti paragraf Dataset Construction

"For this benchmark study, we extracted N = 100 target stanzas from a corpus of 6,640 pantun entries. The resulting benchmark subset contains 100 `a-b-a-b` stanzas (100.0%) and no `a-a-a-a` stanzas. Therefore, the current experiment should be interpreted as a controlled benchmark on the dominant cross-rhyme form (`a-b-a-b`) rather than a proportionally stratified two-scheme sample. We selected N = 100 to balance annotation cost and statistical comparability across model settings, yielding 1,200 generated outputs for downstream xRASA and leakage analyses."