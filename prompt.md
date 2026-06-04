🎯 Topik Utama: "Benchmarking Large Language Models on Metaphorical Constraints in Indonesian Pantun Generation"

Topik ini bisa dikerjakan dalam 4-8 minggu.

🛠️ Apa yang dikerjakan (Metodologi Cepat):

Selection (1 minggu):
Ambil sampel representatif (misal 500-1.000) dari dataset Anda sebagai referensi.
Pilih 3 LLM untuk diadu: DeepSeek-V3, Claude 3.5 Sonnet, Gemini 3.1 / GPT-5 dan IndoLlama (Lokal).

Prompting (1 minggu):
Gunakan teknik Few-Shot Prompting. Berikan LLM 5 contoh pantun dari dataset Anda.
Minta LLM membuat pantun berdasarkan "Isi" tertentu yang Anda ambil dari dataset (untuk membandingkan hasilnya dengan "Sampiran" asli buatan manusia).
Automated Evaluation (1 minggu):
Gunakan Python untuk menghitung:
Rhyme Score: Cek rima akhir (a-b-a-b).
Syllable Score: Hitung jumlah suku kata (apakah 8-12).
Semantic Similarity: Gunakan Cosine Similarity (BERTScore) untuk melihat hubungan antara Sampiran buatan AI dan Isi.
Human/Expert Evaluation (1 minggu):
Minta 2-3 rekan (atau pakar sastra) menilai "Kualitas Metafora" pada skala 1-5.

📊 Struktur Artikel untuk Q1-Q4:
Abstract: Fokus pada gap antara kemampuan teknis LLM vs pemahaman budaya metafora pada pantun.
Introduction: Pantun sebagai tantangan NLP karena struktur kaku dan metafora sampiran-isi.
Methodology: Penjelasan dataset 5.349 pantun dan skenario testing pada LLM.
Results: Tampilkan tabel perbandingan. Biasanya AI bagus di rima, tapi lemah di metafora sampiran yang orisinal.
Discussion: Mengapa AI cenderung "halusinasi" atau memberikan sampiran yang generik? Di sini poin akademik Anda.

🚀 Tips Agar Tembus Jurnal:
Q4 (SINTA 2 / Jurnal Lokal Terindeks Scopus): Cukup deskriptif. Fokus pada "LLM ini bisa buat pantun atau tidak".
Q1-Q2: Anda harus menekankan pada "Metaphorical Gap". Jelaskan secara mendalam mengapa model bahasa saat ini belum bisa menangkap hubungan simbolik antara alam (sampiran) dan manusia (isi).

🔗 Hal yang bisa Anda kerjakan segera:
Ekstrak 100-200 baris "Isi" dari dataset Anda.
Generate sampiran menggunakan ChatGPT/Llama dengan prompt: "Buatlah sampiran yang memiliki metafora alam yang cocok untuk isi pantun berikut: [Isi]".
Bandingkan hasilnya secara statistik