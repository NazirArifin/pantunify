import os
import csv

def main():
    ai_dir = 'data/ai_sampiran'
    if not os.path.exists(ai_dir):
        ai_dir = '../data/ai_sampiran'
        
    if not os.path.exists(ai_dir):
        print(f"Directory {ai_dir} tidak ditemukan!")
        return
        
    print(f"\n--- HASIL EVALUASI KUALITATIF (LLM-as-a-Judge) ---")
    print(f"{'Model AI':<25} | {'N':<3} | {'Rata-rata Skor (1-5)':<20}")
    print("-" * 60)
    
    results = []
    
    for filename in sorted(os.listdir(ai_dir)):
        if filename.endswith('.csv'):
            filepath = os.path.join(ai_dir, filename)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                delimiter = ';' if ';' in first_line else ','
                
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                
                total = 0
                score_sum = 0
                
                for row in reader:
                    gemini_score = row.get('gemini', '').strip()
                    
                    # Cek apakah skor bisa diubah menjadi float/int
                    try:
                        score = float(gemini_score)
                        score_sum += score
                        total += 1
                    except ValueError:
                        continue
                        
            if total > 0:
                avg_score = score_sum / total
                results.append((filename.replace('.csv', ''), total, avg_score))
                
    for r in results:
        print(f"{r[0]:<25} | {r[1]:<3} | {r[2]:>4.2f} / 5.00")
        
    print("-" * 60)

if __name__ == '__main__':
    main()
