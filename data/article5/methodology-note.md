## Outline yang disarankan untuk bagian *Methodology*

Outline ini mengikuti pipeline yang terlihat pada `notebook pelatihan IndoBERT`, `skrip CRF`, dan data anotasi `pantun-anotated.txt` serta `pantun_gaul.csv`.

### 3.1. Desain penelitian dan rumusan tugas
Jelaskan penelitian sebagai pengembangan dan evaluasi pipeline NLP hibrida untuk menganalisis pantun modern berbahasa Indonesia informal. Uraikan tugasnya secara bertahap: klasifikasi token nonstandar, konversi token menjadi representasi bunyi, segmentasi suku kata, lalu analisis pola rima. Tegaskan bahwa evaluasi setiap tahap dilakukan terpisah agar keberhasilan satu komponen tidak menutupi kesalahan pada komponen lain.

### 3.2. Korpus dan unit analisis
Paparkan sumber korpus, kriteria inklusi/eksklusi, periode dan cara pengumpulan, serta proses deduplikasi. Data anotasi yang tersedia berisi **849 pantun**, dengan token dan label per token; CSV juga memuat teks pantun, pemisahan sampiran–isi, rima akhir, jumlah suku kata, dan jumlah kata per baris.

Tetapkan unit analisis untuk tiap eksperimen: pantun untuk analisis struktur/rima, token untuk klasifikasi IndoBERT, dan urutan fonem untuk CRF. Naskah perlu menyebutkan asal data dan cara memperoleh atau menggunakan teks tersebut—informasi itu belum dapat dipastikan dari skrip dan berkas yang diperiksa.

### 3.3. Skema anotasi dan kontrol kualitas
Jelaskan definisi operasional label yang digunakan:

- `TYP`: typo atau bentuk tidak baku;
- `ASG`: kata asing;
- `SKT`: singkatan;
- `DSM`: disemvowelling;
- `O`: token di luar kategori tersebut.

Sertakan panduan anotasi, contoh kasus ambigu, jumlah anotator, prosedur adjudikasi, dan ukuran kesepakatan antar-anotator. Hal-hal ini penting untuk artikel jurnal, tetapi belum terdokumentasi dalam materi yang tersedia.

Di korpus, label `O` berjumlah 13.482 dari 14.574 token; label non-`O` relatif jarang. Label berawalan `I-` juga sangat sedikit. Karena itu, jelaskan bagaimana rentang label dibentuk dan laporkan metrik per kelas, bukan hanya akurasi agregat.

### 3.4. Pra-pemrosesan teks
Terangkan normalisasi dan tokenisasi, termasuk penanganan huruf besar-kecil, tanda baca, token kosong, dan bentuk tanpa vokal. Jelaskan pula bagaimana label token diselaraskan dengan subword tokenizer IndoBERT serta bagaimana truncation ditangani. Notebook menggunakan panjang maksimum 128 token; laporkan juga apakah ada sampel yang terpotong.

### 3.5. Pipeline pemodelan

#### 3.5.1. Klasifikasi token dengan IndoBERT
Jelaskan fine-tuning `indobenchmark/indobert-base-p1` untuk klasifikasi token. Notebook mencatat pembagian 80:20 dengan seed 42, *class weighting*, learning rate `2e-5`, batch size 16, hingga 15 epoch, dan pemilihan checkpoint berdasarkan F1 validasi.

Jabarkan label yang diprediksi serta cara mengukur kinerja per label. Hasil yang tercatat pada notebook F1 adalah precision 0,832, recall 0,720, F1 0,772, dan accuracy 0,973. Sajikan ini sebagai hasil eksperimen yang tercatat, lalu verifikasi ulang terhadap checkpoint dan konfigurasi final sebelum publikasi.

#### 3.5.2. Konversi grafem ke fonem dan penanganan OOV
Terangkan *switchboard* fonologis yang mengarahkan token menurut label: kata Indonesia ke G2P Indonesia, kata asing ke G2P asing, singkatan ke ejaan nama huruf, dan disemvowelling ke aturan fonotaktik/heuristik panjang token. Typo juga diarahkan ke G2P Indonesia dalam skrip.

#### 3.5.3. Segmentasi suku kata dengan CRF
Jelaskan CRF sebagai pemodel suku kata pada urutan unit bunyi, menggunakan label `B` dan `I` serta fitur unit saat ini, tetangga, kategori vokal/konsonan, dan posisi awal/akhir. Skrip membangun satu urutan untuk setiap token dan menggunakan threshold disemvowelling 3 pada hasil eksperimen yang tersimpan.

#### 3.5.4. Ekstraksi rima
Uraikan aturan identifikasi rima akhir dari fonem/suku kata terakhir dan pemeriksaan skema pantun, misalnya A-B-A-B. Namun, tempatkan ini sebagai tahap analisis yang perlu diuji: `blueprint-ekstrak-rima.py` berisi contoh logika, tetapi materi yang ditinjau belum menunjukkan evaluasi kuantitatif komponen rima.

### 3.6. Protokol eksperimen dan evaluasi
Laporkan metrik per tugas:

- **IndoBERT:** precision, recall, F1 per kelas, macro-F1, serta confusion matrix; sertakan accuracy sebagai metrik tambahan.
- **CRF:** precision, recall, dan F1 per label, exact match per urutan, serta metrik segmentasi pada data uji yang berlabel gold.
- **Rima:** akurasi skema rima dan kesesuaian rima akhir terhadap anotasi acuan, jika anotasi CSV telah diverifikasi sebagai *ground truth*.

Untuk komponen CRF, `metrik yang tersimpan` melaporkan token accuracy 0,982 dan sequence exact match 0,907. Akan tetapi, label evaluasinya dinyatakan sebagai *silver labels* yang dibuat otomatis dari fonem, bukan anotasi segmentasi manual. Karena itu, angka tersebut **belum membuktikan akurasi terhadap pelafalan atau segmentasi suku kata yang benar**.

### 3.7. Baseline, ablation, dan analisis kesalahan
Bandingkan pendekatan hibrida dengan baseline sederhana yang relevan, misalnya aturan heuristik tanpa IndoBERT atau tanpa CRF. Lakukan ablation dengan menghilangkan satu komponen pada satu waktu—IndoBERT, aturan OOV, atau CRF—dan ukur dampaknya terhadap hasil akhir. Analisis kesalahan berdasarkan kategori token, jenis rima, dan kondisi ambiguitas fonologis.

### 3.8. Reprodusibilitas dan pertimbangan etis
Cantumkan versi model dan pustaka, seed, perangkat, konfigurasi pelatihan, serta cara memperoleh data. Jelaskan anonimisasi atau pertimbangan privasi bila korpus bersumber dari teks pengguna.

## Hal yang perlu dibereskan sebelum naskah final

- **Validasi fonem:** hasil CRF menyatakan G2P Indonesia dan Inggris tidak tersedia, sehingga sistem memakai fallback grafem. Hindari mengklaim bahwa hasil saat ini telah memvalidasi fonem atau rima berbasis pelafalan.
- **Pemisahan data:** skrip CRF membagi urutan **per token**, bukan per pantun. Token dari pantun yang sama berpotensi masuk ke data latih dan uji; lakukan pembagian berdasarkan pantun sebelum membentuk urutan untuk mengurangi risiko *data leakage*.
- **Konsistensi model:** konfigurasi model tersimpan mencantumkan jumlah label yang tidak selaras dengan peta label. Cocokkan konfigurasi, label, checkpoint, dan metrik yang dilaporkan.
- **Validitas evaluasi:** buat atau gunakan anotasi gold untuk fonem/suku kata dan rima; evaluasi terhadap label otomatis sebaiknya dilaporkan sebagai evaluasi terhadap *silver labels*, bukan sebagai validasi linguistik.

Jadi, susunan di atas dapat menjadi kerangka metodologi, tetapi klaim utama tentang akurasi fonologis dan rima sebaiknya menunggu validasi independen dengan data gold dan pemisahan uji pada tingkat pantun.