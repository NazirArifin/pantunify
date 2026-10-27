def ekstrak_rima_akhir(suku_kata_terakhir):
    """
    Mengambil komponen Vokal (Nukleus) dan Konsonan Mati (Koda) 
    dari suku kata terakhir untuk menentukan rima pantun.
    """
    # Daftar fonem vokal dalam IPA Bahasa Indonesia (termasuk schwa/pepet)
    vowels = {'a', 'i', 'u', 'e', 'o', 'ə', 'ɔ', 'ɛ'}
    
    # Split suku kata terakhir menjadi list fonem
    fonems = suku_kata_terakhir.split('-')
    
    # Cari indeks vokal pertama dari belakang
    for i in range(len(fonems) - 1, -1, -1):
        if fonems[i] in vowels:
            # Rima adalah semua fonem dari vokal tersebut sampai akhir kata
            rima = "-".join(fonems[i:])
            return rima
            
    return suku_kata_terakhir # Fallback jika tidak ditemukan vokal

# --- SIMULASI OUTPUT HASIL LAYER 3 ---
# Contoh jika suku kata terakhir dari baris 1 s.d 4 sudah dipotong oleh CRF 2
baris_1 = "l-a-k"   # dari kata 'budak' -> /b-u/ + /d-a-k/
baris_2 = "t-a-j"   # dari kata 'pantai' -> /p-a-n/ + /t-a-j/
baris_3 = "b-a-k"   # dari kata 'tebak' -> /t-ə/ + /b-a-k/
baris_4 = "s-a-j"   # dari kata 'santai' -> /s-a-n/ + /s-a-j/

rima_1 = ekstrak_rima_akhir(baris_1) # Hasil: "a-k"
rima_2 = ekstrak_rima_akhir(baris_2) # Hasil: "a-j"
rima_3 = ekstrak_rima_akhir(baris_3) # Hasil: "a-k"
rima_4 = ekstrak_rima_akhir(baris_4) # Hasil: "a-j"

# --- VALIDASI ATURAN PANTUN (A-B-A-B) ---
print(f"Rima Baris 1: /{rima_1}/")
print(f"Rima Baris 2: /{rima_2}/")
print(f"Rima Baris 3: /{rima_3}/")
print(f"Rima Baris 4: /{rima_4}/")

if rima_1 == rima_3 and rima_2 == rima_4:
    print("\n🎉 Validasi Sukses: Struktur Rima Pantun Sah (A-B-A-B)!")
else:
    print("\n❌ Validasi Gagal: Struktur Rima Tidak Sesuai Aturan Pakem Pantun.")
