# Benchmarking Large Language Models for Reverse Pantun Generation: An xRASA Score and Content Leakage Analysis


### 1. Kontribusi Utama Paper (*Novelty*)

* **Kontribusi Utama (*Novelty*):**
1. Evaluasi komparatif 6 LLM (Closed vs Open vs Reasoning vs Regional) pada tugas *reverse poetry generation* (menghasilkan sampiran dari isi).


2. Menguji dampak *Zero-Shot* vs *Few-Shot* prompting pada kepatuhan struktur puitis Melayu/Indonesia.


3. Memperkenalkan **xRASA Score**, metrik komposit otomatis yang mengombinasikan aturan prosodi formal dengan *calibrated LLM-as-a-Judge*.


4. Memasukkan **Content Leakage Rate (CLR)** sebagai metrik diagnostik untuk mengukur kecenderungan model "mencuri" kata kunci masukan.





---

### 2. Kerangka Metrik Evaluasi

Eksperimen menggunakan dua tingkat penilaian:

#### A. Metrik Evaluasi Utama (xRASA Score)

Mengukur kualitas akhir pantun dalam skala komposit $[0,00 – 1,00]$:


$$\text{RASA} = A \times (0,5 \cdot R + 0,5 \cdot S_{\text{judge}})$$

* **$R$ (Rhyme Compliance):** Skor biner fonetis berbasis skrip Python untuk memeriksa rima silang ($a-b-a-b$) atau bersambung ($a-a-a-a$).


* **$A$ (Accentuation Penalty):** Penalti halus berbasis deviasi suku kata dari metrum ideal (8–12 suku kata per baris):



$$A = 0,5 + \left(0,5 \times \exp\left(-\lambda \cdot \sum_{k=1}^{4} \text{dev}(S_k)\right)\right)$$


* **$S_{\text{judge}}$ (Semantic Bridge):** Skor kualitas jembatan semantis/suasana dari sampiran ke isi (skala Likert 1–5 dari LLM Judge yang dinormalisasi ke $[0,0 – 1,0]$).



#### B. Protokol Validasi (Human-AI Calibration)

Untuk menjamin validitas ilmiah *LLM-as-a-Judge* di mata reviewer IEEE:

1. Ambil **50 sampel pantun teracak** dari berbagai model.


2. Lakukan penilaian *blind* oleh penilai manusia/pakar pada 50 sampel tersebut menggunakan kriteria *Semantic Bridge* yang sama.


3. Hitung **Spearman Rank Correlation ($\rho$)** antara skor LLM Judge dan skor Manusia.


4. Jika $\rho$ signifikan ($p < 0,05$), gunakan hal tersebut sebagai justifikasi metodologis untuk mengevaluasi seluruh dataset menggunakan LLM Judge.



#### C. Metrik Diagnostik (Content Leakage Rate - CLR)

Mengukur persentase kata konten (*non-stopword*) dari baris *isi* yang muncul kembali di baris *sampiran* yang tergenerasi:


$$\text{CLR} = \frac{\vert{}\text{Words}(\text{Sampiran}) \cap \text{Words}(\text{Isi})\vert{}}{\vert{}\text{Words}(\text{Isi})\vert{}}$$

---

### 3. Struktur Main Sections Paper IEEE

```
┌────────────────────────────────────────────────────────────────────────┐
│                        I. INTRODUCTION                                 │
│ - Latar belakang Pantun & tantangan inverse generation                 │
│ - Rumusan masalah & kontribusi utama paper                             │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                       II. RELATED WORK                                 │
│ - Computational Poetry Generation & LLM Benchmarking                   │
│ - Metrik Evaluasi Puisi (Metrik Otomatis vs LLM-as-a-Judge)            │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                 III. EXPERIMENTAL METHODOLOGY                          │
│ - Dataset: Struktur CSV (Isi -> Sampiran)                              │
│ - Model Benchmark: Closed vs Open vs Reasoning vs Regional             │
│ - Prompting Setup: Zero-Shot vs Few-Shot                               │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                   IV. EVALUATION FRAMEWORK                             │
│ - Structural Metrics (R & A)                                           │
│ - Empirical RASA Score & Human-AI Calibration (Spearman ρ)             │
│ - Content Leakage Rate (CLR)                                           │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                   V. RESULTS AND DISCUSSION                            │
│ - Tabel Komparasi Utama (Model vs Setting vs Metrics)                  │
│ - Statistical Significance (Paired t-test & ANOVA)                     │
│ - Qualitative Error Analysis & Content Leakage Patterns                │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                 VI. CONCLUSION & FUTURE WORK                           │
└────────────────────────────────────────────────────────────────────────┘

```

---

### 4. Rencana Analisis Statistik

| Hipotesis / Pertanyaan Riset | Uji Statistik yang Digunakan |
| --- | --- |
| Apakah *Few-Shot* secara signifikan meningkatkan RASA Score dibanding *Zero-Shot*? | **Paired t-test** atau **Wilcoxon Signed-Rank Test**.

 |
| Apakah terdapat perbedaan performa yang signifikan antar ke-6 arsitektur model? | **Two-Way ANOVA** atau **Kruskal-Wallis Test**.

 |
| Seberapa selaras LLM Judge dengan persepsi penilaian manusia? | **Spearman Rank Correlation ($\rho$)**.

 |

---

### 5. Checklist Kesiapan Sebelum Submission

* [ ] Eksekusi kode Python untuk menghitung $R$, $A$, dan $CLR$ pada seluruh output CSV.
* [ ] Eksekusi prompt LLM Judge (misal: GPT-4o) untuk menilai aspek *Semantic Bridge* ($S$).
* [ ] Jalankan *pilot study* 50 sampel (penilaian manusia + uji Spearman $\rho$).
* [ ] Hitung nilai komposit *Empirical RASA Score*.
* [ ] Jalankan uji signifikansi statistik ($p$-value).
* [ ] Susun draf paper menggunakan *template* IEEE 2-column format.



### 2. Penjelasan Eksplisit di Sub-bab Methodology (Section IV)

Di bagian metodologi penentuan metrik, buat sub-bab atau paragraf khusus yang menjelaskan posisi xRASA terhadap kerangka RASA yang utuh.

* **Contoh Kalimat di Section IV:**
> ***IV. The xRASA Evaluation Framework***
> *"The **xRASA score** presented in this study is a preliminary, empirical prototype of the broader RASA (Rhyme, Accentuation, and Semantic Alignment) framework. It is specifically designed to provide an immediate, data-driven baseline for benchmarking current LLMs. In this experimental iteration, formal prosodic rules ($R$ and $A$) are computed deterministically, while the complex semantic bridge ($S_{\text{judge}}$) is proxied via a human-calibrated LLM judge, laying the groundwork for future latent-space implementations."*



---

### 3. Penutup yang Tegas di Section VI (Conclusion & Future Work)

Bagian *Future Work* adalah tempat paling krusial untuk memperjelas *roadmap* pengembangan dari prototipe xRASA menuju **Canonical RASA Score** (versi utuh disertasimu).

* **Contoh Kalimat di Conclusion / Future Work:**
> *"While **xRASA** successfully demonstrates the viability of a unified prosodic-semantic metric using calibrated LLM judges, it remains an experimental stepping-stone. Future work will extend xRASA into the full **RASA Score**, replacing the external judge proxy with direct internal latent-space alignment and knowledge-graph-guided semantic representations."*



## Outline Detail Artikel IEEE

### Abstract

* **Context:** Tantangan generasi puitis berbasis aturan terikat (*constrained poetry generation*) pada pantun Indonesia, khususnya *reverse generation* (membuat sampiran dari isi).

* **Methodology:** Evaluasi komparatif pada 6 LLM (ChatGPT, Gemini, Claude, Llama 3.1: 8B, Sailor2, DeepSeek-R1) menggunakan strategi *Zero-Shot* dan *Few-Shot* prompting.

* **Proposed Metric:** Memperkenalkan **xRASA Score** (metrik komposit prosodi dan semantik terkalibrasi) serta **Content Leakage Rate (CLR)**.

* **Key Findings:** Ringkasan singkat model mana yang meraih xRASA tertinggi dan efek *few-shot* terhadap penekanan *content leakage*.



---

### I. INTRODUCTION

* **Latar Belakang:** Hakikat pantun Indonesia (struktur 4 baris, rima $a-b-a-b$, metrum 8–12 suku kata, serta pemisahan fungsi antara sampiran dan isi).


* **Rumusan Masalah:** Generasi sampiran (*sampiran generation*) memiliki tantangan unik: sampiran harus mandiri secara leksikal/topik tetapi wajib menyediakan jembatan rima dan suasana (*semantic bridge*) menuju isi. LLM sering kali mengalami *content leakage* (mencuri kata dari isi).


* **Kontribusi Penelitian:**
1. *Benchmarking* komprehensif 6 arsitektur LLM beragam (Closed, Open-weight, Regional, dan Reasoning) pada tugas *reverse pantun*.


2. Evaluasi dampak *Zero-Shot* vs *Few-Shot* prompting.


3. Formulasi **xRASA Score** sebagai *experimental evaluation framework* komposit.


4. Analisis kuantitatif *Content Leakage Rate* (CLR) pada luaran LLM.





---

### II. RELATED WORK

* **Automatic Poetry Generation:** Review perkembangan generasi puisi berbasis NLP dan LLM.


* **Indonesian Pantun Research:** Evaluasi penelitian terdahulu (misal: SeqGAN vs GPT-2, Pemuisi, dan Pantun Bermukun).


* **Poetry Evaluation Metrics:** Keterbatasan metrik standar (BLEU/ROUGE) dan urgensi metrik khusus prosodi/semantik.



---

### III. EXPERIMENTAL SETUP & DATASET

* **Dataset Description:** Penjelasan dataset CSV (terdiri dari `Pantun Asli Lengkap`, `Sampiran Asli`, dan `Isi untuk Input Model`).


* **Model Benchmark:**
* *Closed-Source:* ChatGPT, Gemini, Claude.


* *Open-Weight:* Llama 3.1: 8B.


* *Regional Model:* Sailor2 (Asia Tenggara).
* *Reasoning Model:* DeepSeek-R1.




* **Prompting Strategies:** Rancangan eksperimen *Zero-Shot* (instruksi langsung) vs *Few-Shot* (dengan contoh pasangan isi-sampiran).



---

### IV. THE xRASA EVALUATION FRAMEWORK

* **Overview:** Kerangka kerja xRASA menggabungkan evaluasi formal deterministik dengan evaluasi semantik terkalibrasi.


* **1. Formal Constraints Parsing ($R$ and $A$):**
* *Rhyme Compliance ($R$):* Pengecekan biner rima akhir $a-b-a-b$ atau $a-a-a-a$ via fonem Python.


* *Syllable Accentuation Penalty ($A$):* Penalti eksponensial untuk deviasi metrum suku kata (8–12 suku kata).




* **2. Semantic Bridge Evaluation ($S_{\text{judge}}$) & Calibration:**
* Evaluasi jembatan semantis/suasana menggunakan LLM-as-a-Judge (skala 1–5).


* *Human-AI Calibration:* Uji *pilot study* pada 50 sampel pantun teracak oleh penilai manusia dan penghitungan **Spearman Rank Correlation ($\rho$)**.




* **3. Formulasi Global xRASA Score:**

$$\text{xRASA} = A \times (0,5 \cdot R + 0,5 \cdot S_{\text{judge}})$$



* **4. Diagnostic Metric: Content Leakage Rate (CLR):**
* Pengukuran *Exact Keyword Leakage Rate* dan *Jaccard Lexical Overlap* antara baris sampiran hasil generasi dan baris isi masukan.





---

### V. RESULTS AND DISCUSSION

* **Main Benchmark Results:** Tabel komparasi utama (6 Model $\times$ 2 Prompt Settings $\times$ Metrik $R, A, S_{\text{judge}}, \text{CLR}, \text{xRASA}$).


* **Effect of Prompting Strategy:** Analisis komparatif kinerja *Zero-Shot* vs *Few-Shot* (apakah *few-shot* menekan CLR dan meningkatkan xRASA?).


* **Model Capability Comparison:**
* Performa model *Closed* vs *Open-Weight*.


* Apakah *Reasoning capability* (DeepSeek-R1) membantu kepatuhan rima?


* Apakah *Regional adaptation* (Sailor2) menghasilkan diksi sampiran yang lebih alami?


* **Statistical Significance:** Hasil uji paired t-test / Wilcoxon dan ANOVA untuk membuktikan signifikansi perbedaan performa.


* **Qualitative Error Analysis:** Klasifikasi tipologi kesalahan (rima dipaksa, deviasi suku kata, dan pola kebocoran kata).



---

### VI. CONCLUSION AND FUTURE WORK

* **Conclusion:** Rangkuman temuan model terbaik serta efektivitas kerangka xRASA.


* **Future Work:** Rencana pengembangan xRASA prototipe ini menjadi versi **RASA Score utuh** berbasis *fused hidden space* dan *Knowledge Graph* pada riset lanjutan.


