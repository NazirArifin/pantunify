import re


def clean_line(s):
    s = s.strip()
    s = re.sub(r'^\s*\d+\.\s*', '', s)            # hapus angka di awal
    s = re.sub(r'[\.,;:!\?，。、]+$', '', s)         # hapus tanda baca di akhir termasuk unicode
    return s.strip()

def tail_letters(s, n=2):
    # Ambil kata terakhir, hilangkan non-huruf, kembalikan n huruf terakhir
    s = s.strip().lower()
    if not s:
        return ''
    parts = s.split()
    last = parts[-1]
    last = re.sub(r'[^a-z]', '', last)
    return last[-n:]

def last_syllable(s):
    # Kembalikan substring yang menggambarkan rima akhir:
    # 1. Jika berakhir dengan diftong (ai, au, oi, ei), gunakan diftong tersebut.
    # 2. Jika tidak, ambil vokal terakhir dan semua konsonan yang mengikutinya.
    s = s.strip().lower()
    if not s:
        return ''
    parts = s.split()
    last = parts[-1]
    last = re.sub(r'[^a-z]', '', last)
    
    if not last:
        return ''
        
    # Cek diftong di bagian paling akhir
    diphthongs = ['ai', 'au', 'oi', 'ei']
    for d in diphthongs:
        if last.endswith(d):
            return d
            
    # Cari indeks vokal terakhir
    m = list(re.finditer(r'[aiueo]', last))
    if m:
        # Ambil vokal terakhir sampai akhir kata
        start_idx = m[-1].start()
        return last[start_idx:]
        
    # fallback: gunakan huruf terakhir
    return last[-1:]


def check_rhyme(p1, p2, p3, p4):
    a = last_syllable(p1)
    b = last_syllable(p2)
    c = last_syllable(p3)
    d = last_syllable(p4)
    # Accept a-b-a-b OR all-equal a-a-a-a as valid based on suku kata akhir
    return ((a == c) and (b == d)) or (a == b == c == d)


# def count_syllables(line):
#     """
#     Penghitung suku kata heuristic khusus untuk Bahasa Indonesia.
#     Cocok untuk validasi Pantun (target 8-12 suku kata per baris).
#     """
#     if not isinstance(line, str):
#         return 0
    
#     line = line.strip().lower()
#     if not line:
#         return 0
    
#     # Bersihkan tanda baca, angka, simbol (pertahankan huruf a-z dan spasi)
#     # Catatan: Ini akan menghapus karakter non-ASCII (seperti é), tapi jarang ada di Pantun standar.
#     line = re.sub(r'[^a-z\s]', '', line)
    
#     total_syllables = 0
    
#     for word in line.split():
#         if not word:
#             continue
            
#         # 1. Hitung total vokal
#         vowels = re.findall(r'[aiueo]', word)
#         count = len(vowels)
        
#         # 2. Kurangi untuk diftong (ai, au, oi)
#         # Karena diftong dihitung 2 vokal tapi hanya 1 suku kata
#         diphthongs = re.findall(r'(ai|au|oi)', word)
#         count -= len(diphthongs)
        
#         # Pastikan minimal 1 suku kata per kata (untuk kata seperti 'kh', 'ng' yg tidak punya vokal)
#         # Meskipun dalam BI hampir semua kata punya vokal.
#         if count < 1:
#             count = 1
            
#         total_syllables += count
        
#     return total_syllables

# Kamus pengecualian untuk kata-kata dengan pola suku kata tidak beraturan dalam BI
_KAMUS_PENGECUALIAN = {
    'syukur': 2, 'syarat': 2, 'syair': 2,
    'kualitas': 4, 'kualitatif': 5, 'kuantitas': 4,
    'taat': 2, 'saat': 2, 'buat': 2, 'buat': 2,
    'jumat': 2, 'rabu': 2,
    'radio': 3, 'audio': 3, 'studio': 4,
    'idea': 3, 'ideal': 3,
}

def count_syllables(kata):
    """
    Menghitung suku kata sebuah kata Bahasa Indonesia dengan heuristik yang lebih reliabel.
    Menangani: diftong lengkap, gugus vokal, tanda hubung, karakter beraksent, dan kata pengecualian.
    """
    # Ubah ke huruf kecil dan hapus spasi tambahan
    kata = kata.lower().strip()
    if not kata:
        return 0

    # Hapus tanda hubung agar kata majemuk dihitung sebagai satu token
    kata = kata.replace('-', '')

    # Normalisasi karakter beraksen umum → ASCII terdekat
    import unicodedata
    kata = unicodedata.normalize('NFD', kata)
    kata = kata.encode('ascii', 'ignore').decode('utf-8')

    # Bersihkan karakter non-huruf
    kata = re.sub(r'[^a-z]', '', kata)
    if not kata:
        return 0

    # Cek kamus pengecualian terlebih dahulu
    if kata in _KAMUS_PENGECUALIAN:
        return _KAMUS_PENGECUALIAN[kata]

    # Ganti gugus diftong dan vokal rangkap dengan satu placeholder 'X'
    # Urutan penting: diftong spesifik dulu, lalu pasangan vokal umum
    _pola_diftong = [
        # Diftong sejati BI
        r'ai', r'au', r'oi', r'ei',
        # Gugus vokal yang biasanya 1 suku kata dalam BI lisan
        r'ia', r'io', r'iu',
        r'ua', r'ui', r'uo',
        r'ea', r'eu',
    ]
    for pola in _pola_diftong:
        kata = re.sub(pola, 'X', kata)

    # Hitung semua vokal tersisa + placeholder X (masing-masing = 1 suku kata)
    vokal_ditemukan = re.findall(r'[aiueoX]', kata)
    jumlah = len(vokal_ditemukan)

    # Minimal 1 suku kata per kata
    return max(jumlah, 1)