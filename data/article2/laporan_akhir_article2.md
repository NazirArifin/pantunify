# Laporan Akhir Eksperimen Pantun (xRASA Final + Structural + Leakage)

Sumber data utama:
- data/article2/100_Pantun_Eksperimen_1200_llm_scoring_master_final.csv
- data/article2/100_Pantun_Eksperimen_1200_xrasa_final.csv
- data/article2/table_main_xrasa_1200_final.csv

Ukuran evaluasi:
- n = 1200 (100 data x 6 model x 2 setting)

## 1) Temuan Deskriptif Utama (Final)

- Cakupan skor Semantic Bridge sudah penuh: 1200/1200.
- Rata-rata weighted xRASA keseluruhan: 0.365024.
- Rata-rata komponen global: R = 0.537500, A = 0.939134, S_norm = 0.201250, CLR = 0.256485.
- Pada xRASA, Claude peringkat 1 di zero_shot (0.700470) dan few_shot (0.717086).
- Pada CLR (lebih kecil lebih baik), Gemini peringkat 1 di zero_shot (0.016119) dan few_shot (0.007579).
- Secara agregat lintas model, few_shot menurunkan CLR (0.1562 vs 0.3568), tetapi juga menurunkan R (0.4867 vs 0.5883).

Ringkasan rata-rata per setting:

| Setting | xRASA | R | A | S_norm | CLR |
|---|---:|---:|---:|---:|---:|
| few_shot | 0.3558 | 0.4867 | 0.9457 | 0.2333 | 0.1562 |
| zero_shot | 0.3742 | 0.5883 | 0.9326 | 0.1692 | 0.3568 |

## 2) Hasil Utama per Model dan Setting (Final)

| Model | Setting | n | xRASA Mean | xRASA Std | R Mean | A Mean | S Mean | CLR Mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Chat-GPT | few_shot | 100 | 0.478515 | 0.239606 | 0.790000 | 0.994530 | 0.172500 | 0.168179 |
| Chat-GPT | zero_shot | 100 | 0.430000 | 0.219273 | 0.730000 | 1.000000 | 0.130000 | 0.240429 |
| Claude | few_shot | 100 | 0.717086 | 0.161616 | 0.970000 | 0.980130 | 0.492500 | 0.011290 |
| Claude | zero_shot | 100 | 0.700470 | 0.137107 | 0.990000 | 0.997047 | 0.415000 | 0.028254 |
| DeepSeek-R1 | few_shot | 100 | 0.072382 | 0.119324 | 0.040000 | 0.932372 | 0.107500 | 0.176778 |
| DeepSeek-R1 | zero_shot | 100 | 0.121818 | 0.211298 | 0.250000 | 0.886271 | 0.002500 | 0.468877 |
| Gemini | few_shot | 100 | 0.683381 | 0.213587 | 0.880000 | 0.998523 | 0.487500 | 0.007579 |
| Gemini | zero_shot | 100 | 0.673750 | 0.233614 | 0.880000 | 1.000000 | 0.467500 | 0.016119 |
| Llama 3.1: 8B | few_shot | 100 | 0.163641 | 0.203191 | 0.230000 | 0.950262 | 0.102500 | 0.386067 |
| Llama 3.1: 8B | zero_shot | 100 | 0.249551 | 0.247444 | 0.510000 | 0.866765 | 0.000000 | 0.642266 |
| Sailor2 | few_shot | 100 | 0.019837 | 0.064149 | 0.010000 | 0.818328 | 0.037500 | 0.187472 |
| Sailor2 | zero_shot | 100 | 0.069852 | 0.158509 | 0.170000 | 0.845378 | 0.000000 | 0.744718 |

## 3) Effect of Prompting Strategy (Zero-shot vs Few-shot)

Jawaban ringkas terhadap pertanyaan utama:
- Few-shot konsisten menekan CLR secara agregat (0.1562 vs 0.3568), tetapi tidak otomatis meningkatkan xRASA.
- Secara agregat lintas model, xRASA few-shot justru sedikit lebih rendah (0.3558) dibanding zero-shot (0.3742) karena penurunan R pada beberapa model.

Uji berpasangan xRASA per model (n = 100 pasangan/model):
- Chat-GPT: few-shot > zero-shot secara deskriptif (0.4785 vs 0.4300), namun belum signifikan (Wilcoxon p = 0.1108).
- Claude: few-shot sedikit lebih tinggi (0.7171 vs 0.7005), belum signifikan (p = 0.2284).
- DeepSeek-R1: few-shot lebih rendah (0.0724 vs 0.1218), signifikan (p = 0.0125).
- Gemini: perbedaan sangat kecil (0.6834 vs 0.6738), tidak signifikan (p = 0.9279).
- Llama 3.1: 8B: few-shot lebih rendah (0.1636 vs 0.2496), signifikan (p = 0.0232).
- Sailor2: few-shot lebih rendah (0.0198 vs 0.0699), signifikan (p = 0.0047).

Uji berpasangan metrik komponen (R, A, CLR) menunjukkan pola yang konsisten:
- Penurunan CLR signifikan pada Chat-GPT, Claude, DeepSeek-R1, Llama 3.1: 8B, dan Sailor2.
- Namun pada DeepSeek-R1, Llama 3.1: 8B, dan Sailor2, penurunan CLR disertai penurunan R yang signifikan.

Sumber statistik:
- data/article2/significance_zero_vs_few.csv
- data/article2/significance_zero_vs_few_wilcoxon.csv
- data/article2/significance_xrasa_zero_vs_few_1200.csv

## 4) Model Capability Comparison

Perbandingan kelompok model (rerata lintas setting):

| Group | xRASA | R | A | S_norm | CLR |
|---|---:|---:|---:|---:|---:|
| Closed (Chat-GPT, Gemini, Claude) | 0.613867 | 0.873333 | 0.995038 | 0.360833 | 0.078642 |
| Open-Weight (Llama 3.1: 8B, Sailor2) | 0.125720 | 0.230000 | 0.870183 | 0.035000 | 0.490131 |
| Reasoning (DeepSeek-R1) | 0.097100 | 0.145000 | 0.909321 | 0.055000 | 0.322827 |

Interpretasi:
- Closed models unggul jelas pada kualitas komposit, kepatuhan rima, dan kontrol leakage.
- Open-Weight masih tertinggal terutama pada R dan S_norm, meski memberi ruang intervensi rekayasa prompt/decoding.

Apakah reasoning capability DeepSeek-R1 membantu kepatuhan rima?
- Tidak pada konfigurasi saat ini. Nilai R DeepSeek-R1 berada di bawah rerata Closed maupun Open-Weight pada kedua setting.
- Few-shot: DeepSeek R = 0.0400, Closed R = 0.8800, Open-Weight R = 0.1200.
- Zero-shot: DeepSeek R = 0.2500, Closed R = 0.8667, Open-Weight R = 0.3400.

Apakah regional adaptation Sailor2 menghasilkan diksi sampiran lebih alami?
- Indikasi alami belum kuat pada data final ini, jika diukur dengan proxy yang tersedia.
- Sailor2 memiliki panjang keluaran stabil (gen_line_count mean = 2.0) tetapi skor semantik tetap rendah (proporsi skor >= 3 = 0.0000 pada kedua setting).
- Few-shot memang menurunkan leakage pada Sailor2 (proporsi CLR = 0 naik dari 0.0100 ke 0.3200), tetapi diikuti penurunan tajam kepatuhan rima (R 0.1700 ke 0.0100).
- Deviasi suku kata Sailor2 juga masih tinggi (dev_total mean: zero-shot 1.3900, few-shot 1.7000), sehingga naturalness prosodik belum tercapai secara konsisten.

## 5) Statistical Significance (Paired t-test / Wilcoxon / ANOVA)

Paired comparison (zero-shot vs few-shot per model, berbasis pasangan id yang sama):
- Paired t-test dan Wilcoxon sudah dihitung untuk xRASA serta komponen R, A, CLR.
- Hasil xRASA menunjukkan only-model-specific significance (signifikan turun pada DeepSeek-R1, Llama 3.1: 8B, Sailor2; tidak signifikan pada Chat-GPT, Claude, Gemini).

Across-model comparison per setting:
- xRASA berbeda signifikan antar model pada few-shot dan zero-shot.
- ANOVA: few-shot p = 5.21e-162, zero-shot p = 6.92e-115.
- Kruskal-Wallis: few-shot p = 2.81e-89, zero-shot p = 1.88e-80.

Sumber statistik:
- data/article2/significance_xrasa_zero_vs_few_1200.csv
- data/article2/significance_xrasa_across_models_1200.csv
- data/article2/significance_zero_vs_few.csv
- data/article2/significance_zero_vs_few_wilcoxon.csv
- data/article2/significance_across_models.csv

## 6) Kalibrasi Human vs LLM Judge

Sumber data:
- data/article2/human_calibration_50_blind.csv
- data/article2/human_llm_calibration_summary.json

Ringkasan:
- Jumlah pasangan valid: 50.
- Spearman rho = 0.902490.
- p-value permutation = 0.00009999.
- Rata-rata skor manusia = 2.42.
- Rata-rata skor LLM = 1.80.
- Mean absolute difference = 0.62.

Interpretasi:
- Korelasi peringkat Human vs LLM sangat kuat dan signifikan.
- Penggunaan LLM-as-a-Judge pada skala penuh dapat dipertahankan secara metodologis, dengan catatan LLM cenderung lebih ketat dibanding manusia pada sampel kalibrasi ini.

## 7) Catatan Dataset Construction (Konsistensi Narasi)

Berdasarkan audit distribusi skema rima:
- Corpus utama (N = 6640): a-b-a-b = 4945 (74.47%), a-a-a-a = 1695 (25.53%).
- Subset benchmark 100: a-b-a-b = 100 (100%), a-a-a-a = 0.

Implikasi penulisan artikel:
- Dataset eksperimen lebih tepat disebut benchmark terkontrol pada bentuk dominan a-b-a-b, bukan stratified proporsional dua skema.
- Justifikasi N = 100 tetap kuat untuk studi komparatif karena menghasilkan 1200 output lintas model/setting dengan biaya anotasi yang realistis.

## 8) Qualitative Error Analysis

Tipologi kesalahan utama yang teramati pada output final:

1. Rima dipaksa atau tidak patuh pola akhir (r_score = 0) meskipun leakage rendah.
2. Deviasi suku kata besar (dev_total tinggi), sering muncul saat model menjawab terlalu naratif/prosaik.
3. Kebocoran kata isi ke sampiran (CLR tinggi), termasuk kasus penyalinan hampir literal.

Contoh representatif:

| Tipe | Model | Setting | ID | R | A | CLR | dev_total | Cuplikan Sampiran |
|---|---|---|---:|---:|---:|---:|---:|---|
| Rima dipaksa | Llama 3.1: 8B | few_shot | 1399 | 0 | 1.000000 | 0.000000 | 0 | Bunga melati tumbuh di taman / Harumnya semerbak sepanjang hari |
| Rima dipaksa | Sailor2 | few_shot | 1399 | 0 | 1.000000 | 0.000000 | 0 | Di halaman luas rumput hijau, / Harum bunga melati semerbak pagi. |
| Deviasi suku kata | Llama 3.1: 8B | zero_shot | 1660 | 0 | 0.500226 | 0.250000 | 22 | Rasa rindu sebenarnya bukanlah perselisihan atau sengketa / Mungkin... |
| Deviasi suku kata | Llama 3.1: 8B | zero_shot | 1599 | 0 | 0.500918 | 0.375000 | 18 | Ketika rasa rindu datang, mengapa kamu tidak menjawabnya? / Mungkin... |
| Leakage tinggi | DeepSeek-R1 | zero_shot | 5357 | 1 | 1.000000 | 1.000000 | 0 | Jika hidup ingin penuh budi / Rendah hati jauhi angkuh dan marah |
| Leakage tinggi | DeepSeek-R1 | zero_shot | 5040 | 1 | 1.000000 | 1.000000 | 0 | Jika hidup ingin penuh budi / Rendah hati jangan dilalai |

Interpretasi ringkas:
- Kesalahan paling kritis untuk kualitas pantun adalah ketidakseimbangan antara kontrol leakage dan kepatuhan rima.
- Pada beberapa model, upaya menekan kebocoran kata cenderung menggeser keluaran ke bentuk prosaik dan melemahkan rima akhir.
- Error profile ini memperkuat kebutuhan evaluasi berbasis metrik komposit (xRASA), bukan satu metrik tunggal.

## 9) Discussion Siap-Tempel

### Structural Excellence
Across both prompting settings, Claude demonstrates the strongest structural reliability, with near-ceiling rhyme compliance (R = 0.99 in zero-shot and 0.97 in few-shot) and consistently high accentuation conformity (A > 0.98). This profile indicates that Claude most reliably captures formal pantun constraints, making it the top performer on xRASA in both settings. Gemini remains competitive in xRASA, but its structural compliance is slightly lower than Claude, especially on R.

### Semantic Bridge Limitation
Although Claude leads in total xRASA, its final score remains around 0.70 because the semantic component is not yet saturated. The mean semantic bridge score for Claude corresponds to S_norm values of 0.415 (zero-shot) and 0.4925 (few-shot), which mathematically limits the composite score under xRASA = A x (0.5R + 0.5S). In other words, structural quality is already strong, but semantic bridging from sampiran to isi is still the primary bottleneck preventing scores from approaching the upper bound.

### Implications for Future Work
The current benchmark suggests that future gains should target semantic alignment more directly, rather than only improving formal structure. Prompt design and decoding strategies should explicitly optimize semantic bridging while preserving low leakage, since leakage control alone does not guarantee high semantic relevance. A natural next step is to complement this evaluation with richer semantic supervision signals and expanded human calibration so that improvements in S are measurable, reproducible, and statistically attributable across model families.

### Sailor2 Trade-off Analysis
Sailor2 exhibits a pronounced structure-leakage trade-off across prompting settings. Under zero-shot, leakage is very high (CLR = 0.744718) and almost all rows trigger overlap with isi, but rhyme compliance is still higher than few-shot (R = 0.170000 vs 0.010000). Under few-shot, leakage drops substantially (CLR = 0.187472), yet rhyme compliance collapses, and this structural degradation dominates the composite score despite a small gain in semantic bridge score (S = 0.037500 vs 0.000000).

This behavior is consistent with the xRASA formulation, where the inner term T = 0.5R + 0.5S becomes much smaller in few-shot for Sailor2. Empirically, T decreases from 0.085000 (zero-shot) to 0.023750 (few-shot), while A changes only modestly (0.845378 to 0.818328), producing lower xRASA in few-shot (0.019837) than zero-shot (0.069852). Therefore, the primary optimization target for Sailor2 is preserving rhyme compliance while maintaining the observed leakage reduction.

From a dissertation perspective, this makes Sailor2 a strong intervention candidate rather than a dead-end baseline. Its error profile is clear and actionable: reduce lexical overlap without sacrificing end-line rhyme structure. Methodologically, this supports targeted prompt and decoding experiments on Sailor2, evaluated with paired tests on R, CLR, and xRASA to verify whether structural recovery can be achieved without reintroducing leakage.

## 10) Lokasi File Final

- Master final skor 1200:
  - data/article2/100_Pantun_Eksperimen_1200_llm_scoring_master_final.csv
- Hasil xRASA final per baris:
  - data/article2/100_Pantun_Eksperimen_1200_xrasa_final.csv
- Tabel utama final untuk artikel:
  - data/article2/table_main_xrasa_1200_final.csv
- Uji paired zero-vs-few (xRASA final):
  - data/article2/significance_xrasa_zero_vs_few_1200.csv
- Uji antar model untuk xRASA final (ANOVA + Kruskal):
  - data/article2/significance_xrasa_across_models_1200.csv
- Uji Wilcoxon untuk R, A, CLR:
  - data/article2/significance_zero_vs_few_wilcoxon.csv
