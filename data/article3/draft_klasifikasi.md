# Assessing the Computational Understanding of Poetic Creativity: AI Behavior on the Semantic Decoupling of Indonesian Pantun

# Computational Understanding of Traditional Poetic Creativity: Benchmarking Language Model Behavior on Non-Linear Structures of Indonesian Pantun


## PANDUAN EKSPERIMEN RISET
Topik: Studi Perilaku Model AI (PLM vs LLM) terhadap Klasifikasi Semantik Non-Linear pada Teks Budaya (Pantun Indonesia)
Infrastruktur: Google Colab Pro
------------------------------
## 📌 Rangkuman Riset (Untuk Mahasiswa)

"Riset ini bertujuan untuk menguji sejauh mana model kecerdasan buatan modern memahami puisi tradisional Indonesia (Pantun). Pantun memiliki keunikan non-linear, di mana baris 1-2 (Sampiran) sering kali tidak memiliki hubungan makna dengan baris 3-4 (Isi). Kita akan menguji dan membandingkan perilaku model Machine Learning konvensional, model bahasa lokal (IndoBERT), dan model raksasa (LLaMA-3) dalam menghadapi tantangan linguistik ini menggunakan total 6.640 data pantun."

------------------------------
## 🛠️ Langkah-Langkah Eksperimen (Action Plan)
Minta mahasiswa Anda untuk membagi pekerjaan ke dalam 4 Fase Utama berikut:
## FASE 1: Data Preparation & Semi-Automated Labeling
Karena dataset 6.640 pantun di Mendeley Data belum memiliki label, kita akan menggunakan metode Zero-Shot/Few-Shot Prompting menggunakan LLM sebagai annotator awal, dikombinasikan dengan validasi manusia.

* Langkah 1.1: Tarik dataset .csv langsung dari Mendeley Data ke dalam Google Colab menggunakan perintah wget atau requests.
* Langkah 1.2: Gunakan API LLM (atau muat model instruksi lokal seperti Llama-3-8B-Instruct di GPU A100 Colab Pro) untuk melabeli 6.640 pantun secara otomatis.
* Tugas Mahasiswa: Buat skrip looping (batch) dan rancang prompt ketat agar model hanya mengeluarkan satu label dari 4 kategori: [Nasihat/Agama, Jenaka, Cinta, Teka-teki].
* Langkah 1.3 (Human-in-the-Loop): Minta mahasiswa mengambil 10% sampel acak (sekitar 660 pantun) menggunakan fungsi pandas.sample(), lalu ekspor ke Excel.
* Langkah 1.4: Mahasiswa wajib memvalidasi 660 data tersebut secara manual untuk memastikan ketepatan label AI.
* Langkah 1.5: Hitung nilai Cohen's Kappa Score di Python antara label AI dan label manusia. Nilai harus > 0.7 sebagai bukti ilmiah bahwa sisa data 90% yang dilabeli AI layak digunakan.

------------------------------
## FASE 2: Membangun Model Benchmark (Tiga Kasta Model)
Mahasiswa akan melatih tiga jenis tingkatan model untuk melihat peta performa klasifikasi teks.

* Kasta 1: Traditional Baseline (Machine Learning)
* Tugas: Ekstrak fitur teks menggunakan TF-IDF, kemudian latih menggunakan algoritma SVM dan Naive Bayes.
   * Fungsi: Menjadi batas bawah standar performa (baseline).
* Kasta 2: Pre-trained Language Model / PLM (Deep Learning Lokal)
* Tugas: Lakukan full fine-tuning pada model IndoBERT (menggunakan pustaka Hugging Face Transformers).
   * Catatan Teknis: Mahasiswa harus mengaktifkan parameter penyimpanan checkpoint otomatis ke Google Drive (save_steps) untuk mengantisipasi session timeout di Colab.
* Kasta 3: Large Language Model / LLM (Generative AI)
* Tugas: Lakukan Fine-Tuning pada LLaMA-3-8B memanfaatkan GPU premium (A100) di Colab Pro.
   * Catatan Teknis: Mahasiswa wajib menggunakan teknik LoRA / QLoRA via pustaka PEFT dan TRL (SFTTrainer) agar tidak terjadi eror Out of Memory (OOM).

------------------------------
## FASE 3: Eksperimen Ablasi (Ablation Study)
Bagian ini adalah kunci bobot ilmiah Q1/Q2 untuk menganalisis perilaku model terhadap struktur non-linear.

* Langkah 3.1: Minta mahasiswa membuat satu subset dataset baru bernama "Dataset Hanya Isi", di mana baris 1 & 2 (Sampiran) dihapus dari teks pantun menggunakan manipulasi string di Python.
* Langkah 3.2: Latih ulang model IndoBERT dan LLaMA-3 menggunakan dataset yang hanya berisi baris 3 & 4 tersebut.
* Langkah 3.3: Catat metrik performanya (Accuracy, F1-Score).

------------------------------
## FASE 4: Evaluasi, Analisis Perilaku, dan Plotting
Mahasiswa mengumpulkan semua hasil data eksperimen untuk bahan penulisan draf jurnal Anda.

* Langkah 4.1: Buat Tabel Performa Utama yang membandingkan Accuracy, Precision, Recall, dan F1-Score dari: SVM vs Naive Bayes vs IndoBERT (Full Pantun) vs LLaMA-3 (Full Pantun).
* Langkah 4.2: Buat Tabel Komparasi Ablasi untuk membandingkan performa model: Full Pantun vs Hanya Isi.
* Poin Analisis: Jika performa "Hanya Isi" lebih tinggi, minta mahasiswa menyusun argumen bahwa Sampiran mendistorsi perhatian (attention weight) model AI karena tidak memiliki keterikatan semantik dengan isi.
* Langkah 4.3: Catat dan bandingkan waktu komputasi (training time) serta jumlah parameter aktif antara IndoBERT dan LLaMA-3 untuk menganalisis efisiensi model (efisiensi komputasi vs akurasi).
* Langkah 4.4: Menambahkan metrik analisis kesalahan (Error Analysis) di akhir eksperimen untuk memetakan jenis pantun apa saja yang paling sering membuat model AI terkeco


## 1. Pilar 1: Karakteristik Struktur Teks & Puisi (Linguistik Komputasi)
Kata kunci ini wajib dicari untuk menyusun argumen mengapa klasifikasi pantun itu menantang (mematahkan anggapan "gampang").

* "Computational linguistics for poetry" (Melihat bagaimana AI secara umum memproses puisi).
* "Semantic decoupling in NLP" atau "Semantic discontinuity" (Mencari paper tentang bagaimana AI menangani teks yang hubungan maknanya terputus).
* "Non-linear text classification Transformer" (Mempelajari perilaku arsitektur Transformer pada teks yang tidak berurutan secara naratif linear).
* "Computational analysis of rhyme and meter" (Memahami batasan model AI dalam mengenali rima akhir dan suku kata).

## 2. Pilar 2: Evaluasi Perilaku Model (AI Behavior & Benchmarking)
Kata kunci ini digunakan mahasiswa untuk meniru metodologi eksperimen, penulisan tabel komparasi, dan analisis perilaku model.

* "Pre trained Language Models vs Large Language Models benchmark" (Mencari standar paper yang membandingkan BERT-style vs LLaMA-style).
* "Ablation study for text classification" (Sangat krusial! Ini untuk mencari panduan cara melakukan eksperimen potong sampiran vs isi yang kita diskusikan).
* "Parameter efficient fine tuning LoRA text classification" (Panduan teknis agar mahasiswa tahu cara coding LLaMA-3 di Colab Pro tanpa kena eror Out of Memory).
* "Probing Transformers for cultural context" (Melihat bagaimana peneliti dunia menguji sensitivitas model AI terhadap budaya lokal negara lain).

## 3. Pilar 3: Ranah Jurnal Target (Computational Creativity)
Kata kunci ini untuk mencari paper referensi utama dari jurnal Discover AI atau jurnal sejenis, agar gaya penulisan mahasiswa Anda sesuai dengan selera editor Computational Creativity Collection.

* "Computational Creativity in NLP" (Memahami definisi kreativitas dari sudut pandang komputer).
* "AI understanding of figurative language" atau "AI metaphor detection" (Pantun kaya akan metafora, kata kunci ini membantu mahasiswa memahami bagaimana AI membaca makna tersirat).
* "Evaluating cultural heritage using language models" (Melihat contoh paper dunia yang mendigitalisasi atau menguji warisan budaya lokal menggunakan LLM).



------------------------------
## 📝 BLUEPRINT OUTLINE MANUSKRIP JURNAL
Judul: Computational Understanding of Traditional Poetic Creativity: Benchmarking Language Model Behavior on Non-Linear Structures of Indonesian Pantun
------------------------------
## ABSTRACT (Maksimal 150–250 kata)

* Background: Mengapa memahami puisi tradisional berstruktur non-linear seperti pantun menantang bagi AI (bukan hal yang gampang).
* Objective: Membandingkan perilaku PLM lokal (IndoBERT) vs LLM global (LLaMA-3) dan model tradisional dalam memproses keterputusan semantik sampiran-isi.
* Method: Menggunakan dataset baru berlabel berisi 6.640 pantun, eksperimen dilakukan lewat fine-tuning (LoRA) dan uji ablasi (potong sampiran).
* Results: Tuliskan ringkasan metrik performa terbaik (F1-score) dan model mana yang paling efisien/akurat.
* Conclusion: Implikasi temuan terhadap pemahaman kreativitas budaya lokal oleh teknologi AI masa kini.

------------------------------
## 1. INTRODUCTION (Pendahuluan)

* 1.1. Context & Motivation: Pentingnya digitalisasi warisan budaya non-benda (puisi tradisional) menggunakan AI, dan mengapa Computational Creativity menjadi domain penting saat ini.
* 1.2. The Core Problem (Tantangan Utama): Menjelaskan keunikan Pantun (struktur non-linear & semantic decoupling antara sampiran dan isi). Patahkan asumsi bahwa klasifikasi pantun itu mudah dengan argumen batasan tokenizer AI terhadap rima dan struktur puitis.
* 1.3. Research Gap: Belum banyak riset yang membandingkan secara komprehensif perilaku model bahasa terlatih lokal (monolingual small PLM) dengan model raksasa (multilingual global LLM) dalam konteks puisi tradisional terstruktur.
* 1.4. Contributions: Sebutkan 3 kontribusi utama (1. Penyediaan gold-standard open dataset 6.640 pantun, 2. Benchmark berlapis 3 kasta model, 3. Analisis perilaku model lewat studi ablasi struktur).

------------------------------
## 2. RELATED WORK (Tinjauan Pustaka)

* 2.1. Computational Linguistics in Traditional Poetry: Rangkuman riset-riset terdahulu yang memproses puisi (soneta, haiku, pantun) menggunakan NLP.
* 2.2. Pre-trained Language Models vs Large Language Models: Teori singkat mengenai perkembangan arsitektur Transformer (BERT sebagai encoder-only vs LLaMA sebagai decoder-only) dalam tugas klasifikasi teks.

------------------------------
## 3. METHODOLOGY (Metodologi Eksperimen)

* 3.1. Dataset Acquisition and Semi-Automated Annotation:
* Proses penarikan data dari Mendeley Data.
   * Kerangka kerja pelabelan otomatis (LLM-as-an-annotator) ke dalam 4 kelas (Nasihat, Jenaka, Cinta, Teka-teki).
   * Proses validasi oleh manusia (Human-in-the-loop pada 10% sampel) dan pelaporan nilai Cohen's Kappa Score.
* 3.2. Model Architectures & Training Setups: Detail teknis parameter eksperimen di Google Colab Pro.
* Traditional Baseline: TF-IDF + SVM dan Naive Bayes.
   * PLM Fine-Tuning: Spesifikasi full fine-tuning pada IndoBERT.
   * LLM Fine-Tuning: Spesifikasi Parameter-Efficient Fine-Tuning (LoRA/QLoRA) pada LLaMA-3-8B.
   * Proprietary Baseline: Konfigurasi prompting pada model tertutup (seperti ChatGPT/Gemini).
* 3.3. Ablation Study Configuration: Penjelasan skema pemotongan teks untuk menguji perilaku model (Skenario A: Pantun Utuh vs Skenario B: Hanya Baris Isi).

------------------------------
## 4. RESULTS (Hasil Eksperimen)
(Bagian ini didominasi oleh tabel, grafik, dan angka mentah performa model)

* 4.1. Main Classification Performance: Menampilkan tabel metrik (Accuracy, Precision, Recall, F1-Score) dari seluruh model pada dataset pantun utuh.
* 4.2. Compute Efficiency Metrics: Menampilkan perbandingan waktu latih (training time), memori GPU yang dihabiskan, dan efisiensi ukuran model (IndoBERT vs LLaMA-3).
* 4.3. Ablation Experiment Outcomes: Menampilkan tabel komparasi performa model ketika diberi input "Pantun Utuh" vs "Hanya Isi".

------------------------------
## 5. DISCUSSION (Pembahasan & Analisis Perilaku)
(Bagian paling berbobot untuk mendongkrak nilai paper ke Q1/Q2)

* 5.1. Interpreting AI Behavior on Non-Linear Structures: Menganalisis mengapa performa berubah ketika sampiran dihapus. Apakah sampiran terbukti menjadi semantic noise bagi mekaniske Attention Transformer?
* 5.2. Tokenization and Cultural Nuance Limitations: Membahas bagaimana perbedaan tokenizer IndoBERT (berbasis kata Indonesia) dan LLaMA (global) memengaruhi pemahaman mereka terhadap kata puitis dan rima.
* 5.3. Error Analysis (Bonus Analisis): Menampilkan Confusion Matrix. Membedah contoh-contoh kasus nyata di mana LLM terkecoh (misal: Pantun Nasihat puitis yang salah dikira Pantun Percintaan karena bias kata tertentu).

------------------------------
## 6. CONCLUSION & FUTURE WORK (Kesimpulan)

* 6.1. Conclusion: Rangkuman apakah model AI saat ini benar-benar telah memiliki computational understanding terhadap kreativitas puisi tradisional berdasarkan hasil eksperimen.
* 6.2. Limitations: Keterbatasan riset (misal: keterbatasan variasi kelas pantun atau keterbatasan variasi model LLM yang diuji).
* 6.3. Future Work: Saran riset selanjutnya (misal: menerapkan metode multimodal atau mencoba membuat model generator pantun otomatis yang dievaluasi oleh model klasifikasi ini).


