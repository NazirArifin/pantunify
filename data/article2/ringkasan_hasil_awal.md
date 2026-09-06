# Ringkasan Hasil Awal Eksperimen Pantun (Structural + Leakage)

Sumber data:
- data/article2/100_Pantun_Eksperimen_long_scored.csv
- n = 1200 (100 data x 6 model x 2 setting)

## 1) Temuan Deskriptif Utama

- Pada metrik struktural gabungan sementara (x_struct_proxy = r_mean x a_mean), Claude berada di peringkat 1 pada zero_shot (0.987076) dan few_shot (0.950726).
- Pada CLR (semakin kecil semakin baik), Gemini berada di peringkat 1 pada zero_shot (0.016119) dan few_shot (0.007579).
- Few-shot cenderung menurunkan CLR pada hampir semua model, paling besar pada Sailor2 dan Llama 3.1: 8B.

## 2) Uji Zero-shot vs Few-shot (per model, berpasangan per id)

Interpretasi p-value utama mengacu pada p_permutation (lebih robust terhadap non-normality).

- Chat-GPT: penurunan CLR signifikan (p = 0.000500), perubahan R dan A tidak signifikan.
- Claude: A dan CLR membaik signifikan pada few-shot (masing-masing p = 0.010997 dan p = 0.016746), R tidak signifikan.
- DeepSeek-R1: few-shot menurunkan CLR signifikan (p = 0.000250), tetapi R justru turun signifikan (p = 0.000500); A naik signifikan (p = 0.005749).
- Gemini: tidak ada perubahan signifikan antara zero-shot dan few-shot pada R, A, CLR.
- Llama 3.1: 8B: few-shot menurunkan CLR signifikan (p = 0.000500), namun R juga turun signifikan (p = 0.002000); A naik signifikan (p = 0.001000).
- Sailor2: few-shot menurunkan CLR sangat signifikan (p = 0.000250), tetapi R turun signifikan (p = 0.000250); A tidak signifikan.

## 3) Perbedaan Antar Model (per setting)

- One-way ANOVA permutation menunjukkan perbedaan antar model yang signifikan untuk semua metrik (R, A, CLR) pada zero_shot dan few_shot (semua p = 0.000333).
- Implikasi: pemilihan arsitektur model berpengaruh kuat terhadap kualitas struktural dan kebocoran konten.

## 4) Catatan Metodologis untuk Artikel

- Hasil ini baru mencakup komponen deterministik dan diagnostik (R, A, CLR).
- xRASA penuh belum dihitung karena komponen S_judge (semantic bridge) belum diisi.
- Gunakan data/article2/human_calibration_50_blind.csv untuk studi kalibrasi manusia 50 sampel.
- Setelah skor manusia terkumpul, lanjutkan Spearman rho (Human vs LLM Judge), lalu hitung xRASA final sesuai panduan.

## 5) Hasil Kalibrasi Human vs LLM Judge (50 Sampel Blind)

Sumber data:
- data/article2/human_calibration_50_blind.csv
- data/article2/human_llm_calibration_summary.json

Ringkasan hasil:
- Jumlah pasangan skor valid: 50.
- Spearman rho = 0.902490.
- p-value permutation = 0.00009999.
- Rata-rata skor manusia = 2.42.
- Rata-rata skor LLM = 1.80.
- Mean absolute difference = 0.62 poin.

Interpretasi:
- Korelasi peringkat Human vs LLM sangat kuat dan signifikan.
- Dengan hasil ini, penggunaan LLM-as-a-Judge untuk skala penuh dapat dibenarkan secara metodologis, sambil tetap melaporkan bahwa LLM cenderung memberi skor lebih ketat (lebih rendah) dibanding manusia pada sampel ini.
