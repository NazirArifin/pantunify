import csv
import random
import os

def main():
    input_file = 'data/ok.txt'
    output_file = 'data/dataset_eksperimen.csv'
    
    # Jika dijalankan dari dalam folder scripts
    if not os.path.exists(input_file):
        input_file = '../data/ok.txt'
        output_file = '../data/dataset_eksperimen.csv'

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"File {input_file} tidak ditemukan!")
        return

    # Memisahkan blok pantun berdasarkan baris kosong ganda
    blocks = content.strip().split('\n\n')
    
    # Hanya ambil blok yang benar-benar memiliki 4 baris
    valid_blocks = [b.split('\n') for b in blocks if len(b.strip().split('\n')) == 4]
    
    # Ambil sampel acak 100 pantun
    sample_size = min(100, len(valid_blocks))
    
    # Set seed agar sampel bisa direproduksi jika perlu (opsional)
    random.seed(42) 
    samples = random.sample(valid_blocks, sample_size)
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'sampiran_asli_1', 'sampiran_asli_2', 'isi_1', 'isi_2'])
        
        for i, pantun in enumerate(samples, 1):
            sampiran_1 = pantun[0].strip()
            sampiran_2 = pantun[1].strip()
            isi_1 = pantun[2].strip()
            isi_2 = pantun[3].strip()
            writer.writerow([i, sampiran_1, sampiran_2, isi_1, isi_2])

    print(f"Berhasil mengekstrak {sample_size} pantun ke {output_file}")
    print("Kolom yang diekstrak: id, sampiran_asli_1, sampiran_asli_2, isi_1, isi_2")

if __name__ == '__main__':
    main()
