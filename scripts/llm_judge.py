import os
import csv
import sys
import time
import json

# Cek dependencies
try:
    import google.generativeai as genai
except ImportError:
    print("MENGINSTALL MODULE YANG DIBUTUHKAN...")
    os.system(f"{sys.executable} -m pip install google-generativeai --break-system-packages")
    import google.generativeai as genai

def evaluate_with_gemini(s1, s2):
    prompt = f"""Kamu adalah seorang Pakar Sastra Melayu dan Indonesia bergelar Profesor. 
Tugasmu adalah menilai KUALITAS METAFORA ALAM pada sebuah "Sampiran" pantun secara objektif dan kritis.

Sebuah sampiran yang baik harus:
1. Memiliki citraan alam yang masuk akal dan senatural mungkin (contoh baik: "Anak rusa minum di telaga", contoh buruk/maksa: "Kucing terbang makan pizza").
2. Estetis dan memiliki nilai puitis sastra lama.
3. Terasa seperti pantun tradisional asli Nusantara, bukan sekadar terjemahan mesin kaku.

Berikan nilai skala 1 sampai 5 (1 = Sangat Buruk/Maksa/Halusinasi, 5 = Sangat Indah/Natural/Tradisional).

Evaluasi Sampiran berikut:
Sampiran 1: {s1}
Sampiran 2: {s2}

Keluarkan jawabanmu HANYA DALAM FORMAT JSON murni (tanpa markdown ```json) seperti ini:
{{"skor": 4, "alasan": "Penjelasan evaluasi estetika singkat..."}}
"""
    try:
        # Gunakan Gemini 1.5 Flash karena cepat, murah, dan sangat cerdas untuk tugas klasifikasi/penjurian
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        text = response.text.replace('```json', '').replace('```', '').strip()
        data = json.loads(text)
        return float(data.get('skor', 0)), data.get('alasan', 'Tidak ada alasan')
    except Exception as e:
        return 0.0, f"Error API/JSON: {str(e)}"

def main():
    # 1. Cek API Key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\n[ERROR] GEMINI_API_KEY tidak ditemukan di environment variables!")
        print("Silakan setel API key Anda di terminal sebelum menjalankan script ini.")
        print("Perintah Linux/Mac: export GEMINI_API_KEY='Maukkan_Key_Anda_Disini'")
        print("Dapatkan API key gratis di: https://aistudio.google.com/app/apikey\n")
        sys.exit(1)
        
    genai.configure(api_key=api_key)
    
    # 2. Tentukan target file (bisa diubah manual atau via argumen terminal)
    target_file = sys.argv[1] if len(sys.argv) > 1 else 'data/ai_sampiran/chatgpt-5-5-thinking.csv'
    
    if not os.path.exists(target_file):
        target_file = '../' + target_file
        if not os.path.exists(target_file):
            print(f"File {target_file} tidak ditemukan!")
            return
            
    print(f"Mengevaluasi {target_file} menggunakan LLM-as-a-Judge (Gemini)...")
    print("Memproses baris satu per satu, harap bersabar...")
    print("-" * 70)
    
    with open(target_file, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        delimiter = ';' if ';' in first_line else ','
        
    total_score = 0
    count = 0
    
    with open(target_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        
        # BATAS DEMO: Hanya memproses 10 data pertama agar tidak menghabiskan limit API.
        # Jika Anda ingin mengevaluasi ke-50 datanya, ubah MAX_EVAL menjadi 50.
        MAX_EVAL = 10 
        
        for idx, row in enumerate(reader):
            if count >= MAX_EVAL:
                break
                
            s1_ai = row.get('sampiran_ai_1', '').strip()
            s2_ai = row.get('sampiran_ai_2', '').strip()
            
            if not s1_ai or not s2_ai:
                continue
                
            print(f"[{count+1}/{MAX_EVAL}] Menilai: '{s1_ai}' | '{s2_ai}'")
            skor, alasan = evaluate_with_gemini(s1_ai, s2_ai)
            print(f"   => Skor : {skor}/5.0")
            print(f"   => Pakar: {alasan}\n")
            
            total_score += skor
            count += 1
            
            # Beri jeda 2 detik per panggilan agar tidak terkena Rate Limit / Spam detection dari Google
            time.sleep(2)
            
    if count > 0:
        avg = total_score / count
        print("-" * 70)
        print(f"KESIMPULAN LLM-AS-A-JUDGE UNTUK {target_file}")
        print(f"N (Sampel Uji)             : {count} pantun")
        print(f"Rata-rata Kualitas Metafora: {avg:.2f} / 5.00")
        print("-" * 70)

if __name__ == '__main__':
    main()
