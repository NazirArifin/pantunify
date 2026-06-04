import os
import csv
import sys

# Menambahkan root project ke sys.path agar bisa import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pantunify.utils import check_rhyme, count_syllables

def evaluate_file(filepath):
    # Mendeteksi pemisah (delimiter) CSV (; atau ,)
    with open(filepath, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        delimiter = ';' if ';' in first_line else ','
        
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        
        total = 0
        valid_rhyme = 0
        valid_syllables = 0
        valid_both = 0
        
        for row in reader:
            s1 = row.get('sampiran_ai_1', '').strip()
            s2 = row.get('sampiran_ai_2', '').strip()
            i1 = row.get('isi_1', '').strip()
            i2 = row.get('isi_2', '').strip()
            
            if not s1 or not s2 or not i1 or not i2:
                continue
                
            total += 1
            
            # 1. Evaluasi Rima (Rhyme Score) a-b-a-b
            is_rhyme = check_rhyme(s1, s2, i1, i2)
            if is_rhyme:
                valid_rhyme += 1
                
            # 2. Evaluasi Suku Kata (Syllable Score) 8-12 per baris
            syl_s1 = count_syllables(s1)
            syl_s2 = count_syllables(s2)
            is_syl = (8 <= syl_s1 <= 12) and (8 <= syl_s2 <= 12)
            if is_syl:
                valid_syllables += 1
                
            # 3. Keduanya Valid
            if is_rhyme and is_syl:
                valid_both += 1
                
    if total == 0:
        return None
        
    return {
        'total': total,
        'rhyme_acc': valid_rhyme / total * 100,
        'syl_acc': valid_syllables / total * 100,
        'both_acc': valid_both / total * 100
    }

def main():
    ai_dir = 'data/ai_sampiran'
    if not os.path.exists(ai_dir):
        ai_dir = '../data/ai_sampiran'
        
    if not os.path.exists(ai_dir):
        print(f"Directory {ai_dir} tidak ditemukan!")
        return
        
    print(f"\n--- HASIL EVALUASI OTOMATIS (Automated Evaluation) ---")
    print(f"{'Model AI':<25} | {'N':<3} | {'Rhyme Score':<12} | {'Syllable Score':<15} | {'Perfect (Both)':<15}")
    print("-" * 80)
    
    for filename in sorted(os.listdir(ai_dir)):
        if filename.endswith('.csv'):
            filepath = os.path.join(ai_dir, filename)
            result = evaluate_file(filepath)
            if result:
                model_name = filename.replace('.csv', '')
                print(f"{model_name:<25} | {result['total']:<3} | {result['rhyme_acc']:>11.2f}% | {result['syl_acc']:>14.2f}% | {result['both_acc']:>14.2f}%")
    print("-" * 80)
    print("Catatan:")
    print("- Rhyme Score: Persentase sampiran AI yang memiliki rima akhir a-b-a-b sesuai Isi pantun.")
    print("- Syllable Score: Persentase sampiran AI yang memiliki 8-12 suku kata per baris.")
    print("- Perfect (Both): Persentase sampiran AI yang memenuhi aturan rima DAN jumlah suku kata.\n")

if __name__ == '__main__':
    main()
