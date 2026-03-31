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
    'taat': 2, 'saat': 2, 'buat': 2,
    'jumat': 2, 'rabu': 2,
    'radio': 3, 'audio': 3, 'studio': 3,
    'idea': 3, 'ideal': 3,
}

def count_syllables(kata):
    """
    Menghitung suku kata sebuah kata Bahasa Indonesia dengan heuristik yang lebih reliabel.
    Menangani: diftong lengkap, gugus vokal, tanda hubung, karakter beraksent, dan kata pengecualian.
    """
    # Fungsi sekarang menerima sebuah baris/teks dan menjumlahkan suku kata per kata.
    import unicodedata

    if not isinstance(kata, str):
        return 0

    teks = kata.lower().strip()
    if not teks:
        return 0

    total = 0
    for word in re.split(r'\s+', teks):
        if not word:
            continue

        # Hapus tanda hubung dari kata majemuk
        w = word.replace('-', '')

        # Normalisasi aksen → ASCII
        w = unicodedata.normalize('NFD', w)
        w = w.encode('ascii', 'ignore').decode('utf-8')

        # Bersihkan non-huruf
        w = re.sub(r'[^a-z]', '', w)
        if not w:
            continue

        # Cek kamus pengecualian per kata
        if w in _KAMUS_PENGECUALIAN:
            total += _KAMUS_PENGECUALIAN[w]
            continue

        # Hitung vokal tersisa per kata
        vowels = re.findall(r'[aiueo]', w)
        count = len(vowels)

        # Kurangi hanya untuk diftong sejati yang umumnya dihitung 1 suku kata
        diphthongs = re.findall(r'(ai|au|oi|ei)', w)
        count -= len(diphthongs)

        if count < 1:
            count = 1

        total += count

    return total