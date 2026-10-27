📝 Catatan Teknis Pipeline NLP: Pengkondisian Fonem Terpusat untuk Pantun Modern


1. Karakteristik Masalah (Singkatan vs Disemvowelling)

Dalam teks kasual/pantun modern, kata tanpa vokal terbagi menjadi dua struktur formal yang berbeda, namun sering kali ditulis dengan huruf kecil semua (lowercase):
• Singkatan Huruf Kecil (dpr, ktp): Berasal dari frasa (banyak kata), ditranskripsikan menjadi fonem pelafalan huruf alfabetnya.
• Disemvowelling (brgkt, blm, bgmn): Berasal dari satu kata tunggal yang dihilangkan huruf vokalnya, dipetakan ke kerangka fonotaktik bunyi aslinya.

2. Alur Kerja & Pohon Keputusan (Decision Tree) IndoBERT

                 [Kata Input Pantun Gaul]
                            │
                            ▼
            [IndoBERT Token Classifier (NER)] 
     (Dilatih dengan 800 data anotasi pantun Anda)
                            │
    ┌───────────────┬───────┴───────┬───────────────┐
    ▼               ▼               ▼               ▼
[SINGKATAN]      [TYPO]       [DISEMVOWEL]       [ASING]
    │               │               │               │
[RUMUS A]           └───────┬───────┘               │
(Transkripsi                ▼                       ▼
 Fonem Ejaan)        [Kombinasi Rumus B & C]   [Failsafe G2P]
                            │                  (CMUdict/IPA)
                            ▼                       │
                     (Deret Fonem)                  │
                            │                       │
                            └───────┬───────────────┘
                                    ▼
                      [Cascaded CRF 2: Syllabifier]
                       (Pelabelan Sekuensial B-I)

• JIKA Label = singkatan (Contoh: spbu, dpr):
	• Eksekusi Rumus A: Ubah kata menjadi deret fonem berdasarkan cara eja huruf alfabet Indonesia.
• JIKA Label = disemvowelling (Contoh: skrng, brgkt, yg, lgsg):
	• Eksekusi Rumus B & C: Petakan ke dalam kerangka kluster konsonan atau aturan panjang untuk menghasilkan deret fonem estimasi tanpa membutuhkan POS Tag lagi.
• JIKA Label = asing (Contoh: makeup, now):
	• Eksekusi Failsafe: Belokkan langsung ke G2P asing untuk mendapatkan deret fonem standar internasional (IPA).
• JIKA Label = O (Kata Normal/Baku):
	• Eksekusi Cascaded CRF 1 (G2P): Konversi grafem ke fonem secara langsung termasuk penanganan ambiguitas vokal pepet (/ə/).

3. Aturan Transkripsi Fonem Berbasis Label IndoBERT

Karena pantun berbasis pada bunyian yang diucapkan, seluruh output dari Switch Board harus berupa fonem, bukan angka silabel langsung.

RUMUS A: Fonemisasi Singkatan Alfabetis

Mengubah string singkatan menjadi deret fonem pelafalan alfabet Indonesia secara literal.
• Contoh:
	• dpr → /d-e-p-e-e-r/
	• spbu → /e-s-p-e-b-e-u/
• Catatan: Berikan pengecualian string khusus untuk huruf 'w' (/w-e/) dan 'x' (/e-k-s/).

RUMUS B & C: Fonemisasi Disemvowelling (Clamping & Kluster)

Alih-alih menebak kata asli (vowel restoration), sistem menghasilkan kerangka fonem tiruan berdasarkan visual huruf untuk dilempar ke CRF Pemotong Suku Kata.
• Rumus C (Clamping Aturan Panjang untuk Kata Tugas):
	• Jika panjang kata ≤ 3 huruf (Contoh: yg, gk) → Dikonversi menjadi struktur 1 Suku Kata Bunyi.
	• Jika panjang kata >3 huruf (Contoh: lgsg, ntar) → Dikonversi menjadi struktur 2 Suku Kata Bunyi.
• Rumus B (Syllable Weight Estimation via Phonotactic Mapping):
Untuk kata konten (VERB/NOUN tanpa vokal), deret fonem dihitung dengan penyederhanaan digraf:
	1. Penyederhanaan Digraf: Gabungan konsonan rapat (ng, ny, kh, br, tr) dilebur menjadi 1 fonem tunggal.
	2. Kondisi Akhiran Mati: Konsonan penutup di akhir kata diisolasi sebagai fonem koda (konsonan mati).
	3. Hasil penyederhanaan ini dipetakan menjadi pola fonem Vokal-Konsonan tiruan yang proporsional di rentang rentang min 2 dan max 4 silabel.

4. Arsitektur Refactored: Hybrid Contextual-Phonotactic Pipeline


Layer 1: Contextual-Morphological Encoder (IndoBERT Token Classifier)

• Teks dari LLM dibedah menggunakan model IndoBERT yang telah di-fine-tune menggunakan 800 data pantun gaul.
• Output: Label token (singkatan, disemvowell, typo, asing, O) yang akurat tanpa bergantung pada standardisasi teks manual.

Layer 2a: Deterministic Acoustic Switch Board

• Menerima token OOV beserta label dari IndoBERT.
• Memproses token menggunakan Rumus A, B, atau C untuk mentransformasikannya menjadi bentuk deret fonem mentah.

Layer 2b: Cascaded CRF 1 (Grapheme-to-Phoneme / G2P)

• Tugas: Khusus mengubah kata-kata baku (Label O) menjadi urutan fonem berbasis bunyi (IPA).
• Urgensi: Menyelesaikan ambiguitas huruf 'e' (pepet /ə/ vs taling /e/) menggunakan fitur biner lokal berdasarkan huruf di kiri-kanannya. Kata asing otomatis dialihkan ke G2P eksternal.

Layer 3: Cascaded CRF 2 (Phoneme Syllabifier)

• Tugas Terpusat: Menjadi muara akhir yang menerima urutan fonem bersih dari semua jalur (Jalur G2P Kata Baku, Jalur Ejaan Singkatan Rumus A, maupun Jalur Fonotaktik Disemvowelling Rumus B/C).
• Mekanisme: Memotong urutan fonem tersebut menjadi suku kata (syllable) secara deterministik menggunakan skema pelabelan sekuensial B (Begin) dan I (Inside) berbasis fitur fonem (is_vowel, is_consonant).
• Contoh: /b-e-r-a-ng-k-a-t/ → B-I B-I-I B-I-I (Menghasilkan skor metrum final xRASA secara presisi).

5. Nilai Novelty untuk Jurnal Q1 / Disertasi

• Acoustic-First Pipeline: Menghindari kegagalan vowel restoration (pemulihan kata baku) yang sering merusak struktur rima pantun. Pipeline langsung melompati tahap konversi teks baku dan fokus memetakan struktur visual kata gaul ke bentuk akustik (fonem).
• Unified Syllabification Layer: Cascaded CRF 2 bertindak sebagai satu-satunya gatekeeper penghitung silabel. Semua anomali teks diselesaikan di level fonem, sehingga mengurangi risiko kesalahan berantai (cascading error) pada kalkulasi metrum.
Apakah Anda memerlukan contoh visual pelabelan B-I pada Cascaded CRF 2 untuk representasi kata disemvowelling (brgkt), atau ada aturan penanganan rima akhir pantun yang ingin disisipkan ke dalam pipeline ini?
