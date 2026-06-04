import os
import sys

# Cek dependencies
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    print("MENGINSTALL MODULE YANG DIBUTUHKAN...")
    os.system(f"{sys.executable} -m pip install sentence-transformers torch --break-system-packages")
    print("Instalasi selesai. Silakan jalankan ulang script ini.")
    sys.exit(0)

def main():
    input_file = 'data/ok.txt'
    if not os.path.exists(input_file):
        input_file = '../data/ok.txt'
        
    if not os.path.exists(input_file):
        print(f"File {input_file} tidak ditemukan!")
        return
        
    print("Memuat model Multilingual AI (MiniLM)...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    print(f"\nMembaca dan mengevaluasi seluruh pantun di {input_file}...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error membaca file: {str(e)}")
        return

    # Memisahkan blok pantun berdasarkan baris kosong
    blocks = content.strip().split('\n\n')
    
    # Hanya ambil blok yang benar-benar memiliki 4 baris
    valid_blocks = [b.split('\n') for b in blocks if len(b.strip().split('\n')) == 4]
    
    total = 0
    sim_sum = 0
    
    print(f"Ditemukan {len(valid_blocks)} pantun valid. Memulai kalkulasi...")
    print("Proses ini mungkin memakan waktu beberapa menit tergantung CPU Anda.\n")
    
    # Proses dalam batch kecil agar tidak memakan RAM berlebih
    # Namun karena iterasi biasa sudah cukup aman untuk CPU:
    for i, pantun in enumerate(valid_blocks):
        sampiran_text = pantun[0].strip() + " " + pantun[1].strip()
        isi_text = pantun[2].strip() + " " + pantun[3].strip()
        
        # Hitung Cosine Similarity
        emb_sampiran = model.encode(sampiran_text)
        emb_isi = model.encode(isi_text)
        sim = util.cos_sim(emb_sampiran, emb_isi).item()
        
        sim_sum += max(0, sim) # Hindari nilai negatif
        total += 1
        
        # Tampilkan progress setiap 1000 pantun
        if total % 1000 == 0:
            print(f"Telah memproses {total} / {len(valid_blocks)} pantun...")

    if total > 0:
        avg_sim = (sim_sum / total) * 100
        print("\n" + "="*50)
        print("HASIL AKHIR SEMANTIC SIMILARITY (BASELINE MANUSIA)")
        print("="*50)
        print(f"Total Pantun Dievaluasi : {total:,}")
        print(f"Rata-rata Semantic Score: {avg_sim:.2f}%")
        print("="*50)
        print("Skor ini adalah 'Baseline' mutlak untuk paper Anda!")
        print("Gunakan skor ini untuk menggantikan skor 'Manusia (Pantun Asli)' di draft.")
    else:
        print("Tidak ada pantun valid yang ditemukan.")

if __name__ == '__main__':
    main()
