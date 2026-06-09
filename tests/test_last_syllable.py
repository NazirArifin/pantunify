from pantunify.utils import last_syllable


def test_last_syllable_regression_cases():
    cases = {
        # 1. KATA REGULER (Vokal Akhir & Konsonan Akhir)
        "makan": "an",     # Vokal + Konsonan
        "buku": "u",       # Vokal murni di akhir
        "pergi": "i",      # Vokal murni di akhir
        "melihat": "at",   # Suku kata tertutup
        "maghrib": "ib",     # Suku kata tertutup dengan konsonan ganda

        # 2. HIATUS (Dua vokal berurutan yang terpisah suku katanya)
        "daun": "un",      # Terpisah menjadi da-un
        "koin": "in",      # Terpisah menjadi ko-in
        "dia": "a",        # Terpisah menjadi di-a (Rima akhir dominan 'a')
        "ia": "a",         # Terpisah menjadi i-a
        "setia": "a",      # Terpisah menjadi se-ti-a
        "buaya": "a",      # Berakhiran huruf vokal murni

        # 3. DIFTONG BAKU (Dua vokal yang melebur jadi satu bunyi rima)
        "pantai": "ai",    # Diftong -ai
        "pulau": "au",     # Diftong -au
        "amboi": "oi",     # Diftong -oi
        "survei": "ei",    # Diftong -ei

        # 4. SINGKATAN & AKRONIM (Uji pelafalan lisan)
        "tv": "e",         # Dibaca te-ve -> vokal akhir 'e' (atau 've')
        "ac": "e",         # Dibaca a-se -> vokal akhir 'e' (atau 'se')
        "hp": "e",         # Dibaca ha-pe -> vokal akhir 'e' (atau 'pe')
        "wa": "a",         # Dibaca we-a -> vokal akhir 'a'
        "bumn": "en",      # Dibaca be-u-em-en -> berakhiran 'en'

        # 5. KONSISTENSI UKURAN HURUF & TANDA BACA (Robustness Test)
        "makan.": "an",    # Ada titik di akhir kata
        "BUKU": "u",       # Huruf kapital semua
        "pergi!": "i",     # Ada tanda seru
        "  daun  ": "un",  # Ada spasi berlebih (whitespace)
        "padi,": "i"       # Ada tanda koma
    }

    for word, expected in cases.items():
        result = last_syllable(word)
        print(f"{word!r}: {result!r} (Expected: {expected!r})")
        assert result == expected


if __name__ == "__main__":
    test_last_syllable_regression_cases()
    print("All tests passed!")