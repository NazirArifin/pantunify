# Laporan Eksperimen POS Tagging (BiLSTM → BiLSTM-CRF)

## 1. Tujuan

Membangun pipeline POS tagging bahasa Indonesia yang lebih tangguh untuk teks kasual/pantun gaul (banyak OOV dan disemvowelling), lalu membandingkan:

1. Hybrid BiLSTM (word + char embedding)
2. Hybrid BiLSTM-CRF
3. BiLSTM-CRF dengan hyperparameter tuning (sweep)

---

## 2. Data dan Sumber

Dataset utama:

- `data/article5/UD_Indonesian-GSD/id_gsd-ud-train.conllu`
- `data/article5/UD_Indonesian-GSD/id_gsd-ud-dev.conllu`
- `data/article5/UD_Indonesian-GSD/id_gsd-ud-test.conllu`
- `data/article5/id_csui-ud-train.conllu`

Data inferensi domain:

- `data/article5/pantun_gaul.csv`

---

## 3. Pekerjaan yang Sudah Dilakukan

### 3.1 Perbaikan model baseline (Hybrid BiLSTM)

File:

- `scripts/article5/HybridBiLSTMPOSTagger.py`

Perbaikan penting:

- Char-BiLSTM benar-benar aktif dengan packing (`pack_padded_sequence`)
- Penanganan panjang token/kalimat lebih konsisten
- Validasi shape input dan pretrained embedding
- `compute_loss` dengan masking (`ignore_index`)
- `predict()` dengan `eval()` + `torch.no_grad()`

### 3.2 Pipeline training/evaluasi/prediksi baseline

File:

- `scripts/article5/train_hybrid_bilstm_pos.py`
- `scripts/article5/evaluate_checkpoint_on_conllu.py`
- `scripts/article5/predict_pantun_csv.py`

### 3.3 Konversi data CoNLL-U → JSONL

File:

- `scripts/article5/convert_conllu_to_jsonl.py`

Output:

- `data/article5/processed/train.jsonl`
- `data/article5/processed/dev.jsonl`
- `data/article5/processed/vocab.json`

### 3.4 Implementasi BiLSTM-CRF

File:

- `scripts/article5/HybridBiLSTMCRFPOSTagger.py`
- `scripts/article5/train_hybrid_bilstm_crf_pos.py`
- `scripts/article5/evaluate_checkpoint_on_conllu_crf.py`
- `scripts/article5/predict_pantun_csv_crf.py`
- `scripts/article5/colab_train_hybrid_bilstm_crf_pos.ipynb`

### 3.5 Tuning sweep (6 run) untuk CRF

File:

- `scripts/article5/run_crf_tuning_sweep.py`

Peningkatan di trainer CRF:

- LR terpisah untuk CRF (`--lr-crf`)
- Early stopping (`--early-stopping-patience`)
- Pemilihan best model dengan metric `composite`
- Komponen score: accuracy, OOV accuracy, dan critical-tag macro F1
- Logging per epoch ke `metrics.jsonl`

### 3.6 Tooling evaluasi mini-gold pantun

File baru:

- `scripts/article5/build_mini_gold_from_predictions.py`
- `scripts/article5/evaluate_mini_gold_pos.py`
- `scripts/article5/analyze_mini_gold_errors.py`

Fungsi:

1. Mengubah output prediksi pantun (`pantun_gaul_pred_*.csv`) menjadi template anotasi token-level mini-gold.
   - Mendukung target anotasi lebih besar (mis. 800–1200 token) dengan `--target-tokens`.
2. Menghitung metrik evaluasi mini-gold:
   - Accuracy
   - Macro F1
   - Weighted F1
   - Critical-tag Macro F1 (`ADV, PART, NOUN, VERB`)
   - Per-tag precision/recall/F1
   - Confusion matrix (opsional disimpan ke CSV/JSON)
3. Error analysis terfokus untuk confusion penting (mis. `ADV/PART/VERB -> NOUN`) beserta contoh token/kalimat.

---

## 4. Hasil Utama

## 4.1 Baseline BiLSTM (sebelumnya)

- Dev best epoch: **13**
- Dev best:
  - `val_acc ≈ 0.9314`
  - `val_oov_acc ≈ 0.8633`
- Test GSD (indikatif):
  - `accuracy ≈ 0.9343`
  - `oov_accuracy ≈ 0.8729`

## 4.2 CRF awal (sebelum tuning)

- Test GSD:
  - `accuracy ≈ 0.9347`
  - `oov_accuracy ≈ 0.8650`

Catatan:

- `--lowercase` menyebabkan penurunan besar (mismatch dengan vocab/checkpoint), sehingga konfigurasi final memakai **tanpa lowercase**.

## 4.3 CRF setelah tuning sweep

Best run:

- `run_05_lr1e3_crf3e3_do05_wd1e4_gc5`

Best checkpoint:

- `scripts/article5/tuning_runs_crf_colab/run_05_lr1e3_crf3e3_do05_wd1e4_gc5/best.pt`

Test set metrics:

```json
{
  "loss": 4.750152428944905,
  "accuracy": 0.9390098673018034,
  "tokens": 11756.0,
  "oov_accuracy": 0.880184331797235,
  "oov_tokens": 1519.0
}
```

### 4.4 Ringkasan perbandingan

| Model | Accuracy (GSD test) | OOV Accuracy (GSD test) |
|---|---:|---:|
| BiLSTM baseline | ~0.9343 | ~0.8729 |
| BiLSTM-CRF (awal) | ~0.9347 | ~0.8650 |
| BiLSTM-CRF (tuned, run_05) | **0.9390** | **0.8802** |

Kesimpulan: **BiLSTM-CRF tuned memberikan perbaikan nyata** dibanding baseline dan CRF awal.

---

## 5. Hasil Inferensi Pantun

- Prediksi 10 pantun:
  - `data/article5/processed/pantun_gaul_pred_10_crf.csv`
- Prediksi 100 pantun:
  - `data/article5/processed_crf/pantun_gaul_pred_100_crf.csv`

Temuan:

- Ada perbaikan pada beberapa token penting (mis. `jangan/PART` kembali benar pada run terbaik)
- Namun beberapa token domain gaul masih menantang (mis. variasi tag pada kata konteks slang)

---

## 6. Notebook Colab (End-to-End)

Notebook:

- `scripts/article5/colab_train_hybrid_bilstm_crf_pos.ipynb`

Sekarang mendukung:

1. Training single-run CRF
2. Optional tuning sweep 6-run (`RUN_SWEEP=True/False`)
3. Auto pilih best run
4. Evaluasi ke `id_gsd-ud-test.conllu`
5. Prediksi 100 pantun awal
6. Preview output langsung di notebook

---

## 7. Rekomendasi Backup dari Google Drive

Minimum:

1. `best.pt` run terbaik
2. `vocab.json` yang cocok
3. notebook `.ipynb`

Sangat disarankan:

4. `leaderboard.json`
5. `metrics.jsonl` run terbaik
6. hasil prediksi `pantun_gaul_pred_100_crf.csv`

---

## 8. Next Step yang Direkomendasikan

1. Perbesar mini-gold ke **800–1200 token** (disarankan `--target-tokens 1000`, `--min-tokens 800`).
2. Jalankan error analysis spesifik confusion `ADV/PART/VERB -> NOUN`.
3. Retrain ringan CRF dengan:
   - class/loss weighting untuk tag kritis, dan/atau
   - weighted sampling berbasis OOV + critical-tag density (proxy domain gaul).
4. Bandingkan ulang dengan metrik mini-gold + confusion fokus sebelum memutuskan iterasi berikutnya.
