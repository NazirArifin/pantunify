# Laporan Komparatif IndoBERT: Full Pantun vs Isi Saja

## 1. Tujuan
Laporan ini membandingkan dua skenario input pada model IndoBERT untuk menguji hipotesis ablation:
- Skenario full: seluruh baris pantun digunakan.
- Skenario isi_only: hanya bagian isi (dua baris akhir) yang digunakan.

Pertanyaan utama: apakah sampiran menambah sinyal atau justru menjadi noise untuk klasifikasi kategori pantun.

## 2. Sumber Data Hasil
Analisis ini disusun dari output pada:
- data/article3/indobert-full
- data/article3/indobert-isi-only

File yang digunakan:
- test_metrics.json
- classification_report.json
- confusion_matrix.csv
- split_metadata.json
- run_config.json
- training_time.json

## 3. Validasi Kesetaraan Setup Eksperimen
Kedua run menggunakan setup yang sebanding:
- Dataset total: 6640
- Split: train 5312, valid 664, test 664
- Seed: 42
- Model: indobenchmark/indobert-base-p1
- Max length: 256
- Learning rate: 2e-5
- Batch size train/eval: 16/32
- Epoch: 4
- Weight decay: 0.01
- Class weights: identik

Dengan demikian, perbedaan hasil merefleksikan perbedaan representasi input (full vs isi_only), bukan perubahan hyperparameter utama.

## 4. Perbandingan Metrik Utama
| Metrik | Full | Isi Only | Delta (Isi - Full) |
|---|---:|---:|---:|
| Accuracy | 0.7907 | 0.8193 | +0.0286 |
| F1 Macro | 0.7226 | 0.7601 | +0.0375 |
| F1 Weighted | 0.7965 | 0.8241 | +0.0276 |
| Precision Macro | 0.7019 | 0.7335 | +0.0316 |
| Recall Macro | 0.7577 | 0.8032 | +0.0455 |
| Training Time (menit) | 2.3169 | 2.3753 | +0.0584 |

Temuan utama:
- Skenario isi_only unggul pada semua metrik kualitas klasifikasi.
- Kenaikan paling menonjol terjadi pada Recall Macro (+0.0455), yang berarti lebih banyak kelas berhasil dikenali.
- Waktu training sedikit lebih lama (+3.51 detik) namun selisih sangat kecil secara praktis.

## 5. Perbandingan Per Kelas (F1-Score)
| Kelas | F1 Full | F1 Isi Only | Delta |
|---|---:|---:|---:|
| Pantun Agama | 0.9315 | 0.9139 | -0.0176 |
| Pantun Dukacita | 0.6667 | 0.6486 | -0.0180 |
| Pantun Hiburan | 0.7083 | 0.7253 | +0.0169 |
| Pantun Kasih Sayang | 0.7773 | 0.8122 | +0.0349 |
| Pantun Kiasan | 0.5287 | 0.6190 | +0.0903 |
| Pantun Nasihat | 0.8354 | 0.8632 | +0.0278 |
| Pantun Sosial Budaya | 0.6667 | 0.8000 | +0.1333 |
| Pantun Sukacita | 0.6667 | 0.6988 | +0.0321 |

Interpretasi:
- 6 dari 8 kelas meningkat ketika hanya isi digunakan.
- Peningkatan terbesar ada pada kelas minoritas/lebih sulit:
  - Pantun Sosial Budaya (+0.1333)
  - Pantun Kiasan (+0.0903)
- Penurunan kecil terjadi pada:
  - Pantun Agama (-0.0176)
  - Pantun Dukacita (-0.0180)

## 6. Analisis Confusion Matrix
Pola pada skenario full:
- Kecenderungan prediksi bergeser ke Pantun Nasihat masih cukup kuat.
- Mis-klasifikasi besar terlihat pada aliran ke Nasihat dari beberapa kelas (misalnya Kiasan, Kasih Sayang, Sukacita).

Pola pada skenario isi_only:
- Kelas Pantun Agama menjadi sangat bersih (recall 1.0).
- Kelas minoritas seperti Sosial Budaya dan Kiasan membaik nyata.
- Bias ke kelas mayoritas (Nasihat) masih ada, tetapi intensitas kesalahan lintas kelas berkurang pada beberapa pasangan penting.

Ringkasan interpretasi:
- Sampiran kemungkinan membawa variasi semantik/stilistika yang tidak konsisten terhadap label kategori, sehingga pada banyak kasus berfungsi sebagai noise.
- Menggunakan isi saja membuat sinyal label lebih fokus, terutama untuk kelas-kelas yang sebelumnya rapuh.

## 7. Jawaban atas Hipotesis Ablation
Hipotesis: sampiran menjadi noise.

Kesimpulan empiris dari run saat ini:
- Didukung secara moderat hingga kuat.
- Bukti utama: seluruh metrik global naik pada skenario isi_only, termasuk F1 Macro yang paling relevan untuk data tidak seimbang.

## 8. Implikasi untuk Penelitian
- Untuk objective klasifikasi kategori tematik pantun, representasi isi_only layak dijadikan konfigurasi utama.
- Skenario full tetap bernilai jika target riset bergeser ke analisis gaya/struktur pantun secara holistik.

## 9. Keterbatasan dan Uji Lanjutan yang Disarankan
- Hasil saat ini berasal dari satu split (seed 42).
- Untuk kesimpulan statistik lebih kuat, disarankan:
  - Repeated stratified split multi-seed (misalnya 5-10 seed)
  - Uji signifikansi berpasangan pada F1 per run
  - Pelaporan interval kepercayaan (bootstrap)

## 10. Kesimpulan Akhir
Berdasarkan output yang tersedia, skenario isi_only memberikan kinerja lebih baik dibanding full pada eksperimen IndoBERT ini. Dengan kontrol setup yang setara, temuan ini menunjukkan bahwa penghilangan sampiran cenderung meningkatkan kualitas sinyal klasifikasi kategori pantun.
