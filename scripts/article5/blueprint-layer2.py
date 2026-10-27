from g2p_id import G2P  # Library g2p-id pilihan Anda
import re

# Inisialisasi g2p-id
g2p = G2P()

def rumus_a_singkatan(kata):
    """Mengubah singkatan huruf per huruf menjadi deret fonem ejaan alfabet"""
    ejaan_map = {
        'a': 'a', 'b': 'be', 'c': 'ce', 'd': 'de', 'e': 'e', 'f': 'ef', 'g': 'ge',
        'h': 'ha', 'i': 'i', 'j': 'je', 'k': 'ka', 'l': 'el', 'm': 'em', 'n': 'en',
        'o': 'o', 'p': 'pe', 'q': 'ki', 'r': 'er', 's': 'es', 't': 'te', 'u': 'u',
        'v': 've', 'w': 'we', 'x': 'eks', 'y': 'ye', 'z': 'zet'
    }
    # Pecah per huruf dan gabungkan bunyi ejaannya
    bunyi_ejaan = "".join([ejaan_map.get(char, char) for char in kata.lower()])
    # Gunakan g2p-id untuk mendapatkan representasi fonem IPA dari bunyi ejaan tersebut
    phonemes, _ = g2p.to_phoneme(bunyi_ejaan)
    return phonemes

def rumus_bc_disemvowelling(kata):
    """Menghasilkan representasi pola fonem tiruan berbasis aturan fonotaktik"""
    panjang = len(kata)
    
    # Aturan Clamping Panjang untuk Kata Tugas
    if panjang <= 3:
        # Diasumsikan berdurasi 1 silabel, kita buat struktur fonem KVK sederhana
        return "v o n"  
    else:
        # Diasumsikan berdurasi 2 silabel, kita buat struktur KVKV
        return "v o n e m"

def deterministic_acoustic_switch_board(kata, label):
    """
    LAYER 2: Pengkondisian Fonem Terpusat
    Menerima token dan label dari IndoBERT, menghasilkan output deret fonem.
    """
    label = label.upper()
    
    if label == "O":
        # Jalur Kata Baku menggunakan g2p-id
        phonemes, _ = g2p.to_phoneme(kata)
        return phonemes
        
    elif "TYP" in label: # Menangkap B-TYP atau I-TYP
        # JALUR TYPO: Dilempar langsung ke g2p-id untuk ditebak berdasarkan pendekatan neural OOV
        phonemes, _ = g2p.to_phoneme(kata)
        return phonemes
        
    elif "SINGK" in label:
        # JALUR SINGKATAN: Rumus A
        return rumus_a_singkatan(kata)
        
    elif "DISEM" in label:
        # JALUR DISEMVOWELLING: Rumus B & C
        return rumus_bc_disemvowelling(kata)
        
    elif "ASING" in label:
        # JALUR ASING: Failsafe (Idealnya ke CMUdict/g2p-en, di sini fallback aman)
        phonemes, _ = g2p.to_phoneme(kata) 
        return phonemes
        
    else:
        # Fallback default
        phonemes, _ = g2p.to_phoneme(kata)
        return phonemes

# --- CONTOH SIMULASI HASIL PIPELINE ---
output_indobert = [
    ("hari", "O"),
    ("ini", "O"),
    ("brgkt", "B-DISEM"),
    ("ke", "O"),
    ("spbu", "B-SINGK"),
    ("mksn", "B-TYP") # Contoh kata Typo
]

print("=== HASIL TRANSKRIPSI FONEM LAYER 2 ===")
for kata, label in output_indobert:
    fonem_hasil = deterministic_acoustic_switch_board(kata, label)
    print(f"Kata: {kata:8} | Label: {label:8} ➔ Fonem: {fonem_hasil}")
