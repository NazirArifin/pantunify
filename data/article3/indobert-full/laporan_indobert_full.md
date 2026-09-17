# Laporan Hasil Eksperimen IndoBERT (Klasifikasi Pantun)

## 1. Ringkasan Eksekutif
Eksperimen fine-tuning IndoBERT untuk klasifikasi 8 kategori pantun menunjukkan performa keseluruhan yang baik pada data uji:

- Accuracy: 0.8343
- Macro F1: 0.7554
- Weighted F1: 0.8345
- Macro Precision: 0.7424
- Macro Recall: 0.7717

Interpretasi cepat:
- Model sudah kuat secara agregat (accuracy dan weighted F1 tinggi).
- Masih ada gap performa antar kelas (macro F1 lebih rendah dari weighted F1), menandakan kelas minoritas masih lebih sulit dikenali.

## 2. Konfigurasi Eksperimen
- Model: indobenchmark/indobert-base-p1
- Jumlah kelas: 8
- Max sequence length: 256
- Learning rate: 2e-5
- Batch size train/eval: 16 / 32
- Epoch: 4
- Weight decay: 0.01
- Seed: 42
- Penanganan imbalance: class-weighted loss

## 3. Ringkasan Data dan Split
- Total data: 6640
- Train: 5312 (80%)
- Valid: 664 (10%)
- Test: 664 (10%)

Label yang digunakan:
1. Pantun Agama
2. Pantun Dukacita
3. Pantun Hiburan
4. Pantun Kasih Sayang
5. Pantun Kiasan
6. Pantun Nasihat
7. Pantun Sosial Budaya
8. Pantun Sukacita

## 4. Waktu Komputasi
- Waktu training: 139.68 detik (2.33 menit)
- Resume checkpoint: tidak (null)

Catatan: waktu ini efisien untuk ukuran data dan jumlah epoch yang digunakan.

## 5. Metrik Keseluruhan (Test Set)
| Metrik | Nilai |
|---|---:|
| Accuracy | 0.8343 |
| F1 Macro | 0.7554 |
| F1 Weighted | 0.8345 |
| Precision Macro | 0.7424 |
| Recall Macro | 0.7717 |
| Support | 664 |

## 6. Kinerja Per Kelas
| Kelas | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Pantun Agama | 0.8816 | 0.9710 | 0.9241 | 69 |
| Pantun Dukacita | 0.6842 | 0.8125 | 0.7429 | 16 |
| Pantun Hiburan | 0.7234 | 0.8095 | 0.7640 | 42 |
| Pantun Kasih Sayang | 0.7980 | 0.7980 | 0.7980 | 99 |
| Pantun Kiasan | 0.6364 | 0.6000 | 0.6176 | 35 |
| Pantun Nasihat | 0.8974 | 0.8596 | 0.8780 | 356 |
| Pantun Sosial Budaya | 0.6000 | 0.5455 | 0.5714 | 11 |
| Pantun Sukacita | 0.7179 | 0.7778 | 0.7467 | 36 |

Temuan utama per kelas:
- Kelas terkuat: Pantun Agama (F1 0.9241) dan Pantun Nasihat (F1 0.8780).
- Kelas menengah: Kasih Sayang, Hiburan, Sukacita, Dukacita.
- Kelas paling menantang: Sosial Budaya (F1 0.5714) dan Kiasan (F1 0.6176).

## 7. Analisis Confusion Matrix
Pola kekeliruan yang paling menonjol:
- Banyak kelas bergeser ke Pantun Nasihat, misalnya:
  - Kasih Sayang -> Nasihat: 12 kasus
  - Kiasan -> Nasihat: 8 kasus
  - Hiburan -> Nasihat: 5 kasus
  - Sukacita -> Nasihat: 5 kasus
  - Sosial Budaya -> Nasihat: 3 kasus
- Kelas Pantun Nasihat sendiri juga sering tertukar ke:
  - Hiburan: 10
  - Kasih Sayang: 10
  - Kiasan: 9
  - Sukacita: 9

Interpretasi:
- Pantun Nasihat adalah kelas mayoritas (support sangat besar), sehingga menjadi pusat prediksi model.
- Terjadi bias ke kelas dominan, walaupun class-weight sudah membantu menjaga performa global.

## 8. Implikasi Akademik
- Secara umum, model layak dijadikan baseline kuat untuk klasifikasi pantun multi-kelas.
- Untuk kesimpulan ilmiah yang adil antar kelas, Macro F1 lebih representatif dibanding accuracy saja.
- Masalah utama bukan pada performa total, melainkan pada separabilitas semantik kelas-kelas minoritas yang berdekatan makna.

## 9. Rekomendasi Peningkatan
1. Data-level:
- Tambah data untuk kelas minoritas (terutama Sosial Budaya, Dukacita, Kiasan).
- Lakukan audit kualitas anotasi pada pasangan kelas yang sering tertukar (Nasihat vs Kasih Sayang/Kiasan/Hiburan).

2. Training-level:
- Uji focal loss atau class-balanced focal loss untuk menekan bias ke kelas mayoritas.
- Eksperimen thresholding atau calibration (temperature scaling) agar keputusan kelas lebih stabil.

3. Evaluation-level:
- Tambahkan k-fold cross-validation stratified untuk estimasi robust.
- Laporkan confidence interval metrik (misalnya bootstrap untuk Macro F1).

## 10. Kesimpulan
Eksperimen IndoBERT ini menghasilkan performa kuat secara agregat (Accuracy 0.8343; Weighted F1 0.8345), dengan kemampuan klasifikasi antarkelas yang cukup baik (Macro F1 0.7554). Tantangan utama terletak pada kelas minoritas dan tumpang tindih semantik dengan kelas Pantun Nasihat. Dengan penguatan data minoritas dan strategi loss/evaluasi lanjutan, performa antarkelas berpotensi ditingkatkan lebih jauh.

## 11. Daftar Berkas yang Menjadi Dasar Laporan
- test_metrics.json
- test_metrics.csv
- classification_report.json
- confusion_matrix.csv
- training_time.json
- split_metadata.json
- run_config.json
- figures/confusion_matrix.png
