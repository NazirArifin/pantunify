from pantunify.utils import count_syllables
from pantunify.classifier import validate_pantun

def _assert_counts(tests):
    for text, expected in tests:
        res = count_syllables(text)
        print(f"'{text}': {res} (Expected: {expected})")
        assert res == expected


def test_diphthong_and_hiatus_balance():
    tests = [
        ('menunaikan', 4),
        ('tunaikan', 3),
        ('abaikan', 3),
        ('selesaikan', 4),
        ('kedaulatan', 4),
        ('persaudaraan', 5),
        ('saudara', 3),
        ('bagaimana', 4),
        ('terbaik', 3),
        ('berkilauan', 4),
        ('tauladan', 3),
        ('naik', 2),
        ('daun', 2),
        ('mau', 2),
        ('bau', 2),
        ('kemauan', 4),
        ('cobaan', 3),
        ('jauh', 2),
        ('jauhilah', 4),
        ('kesemuanya', 5),
        ('saudaraku', 4),
        ('pakaiannya', 4),
        ('tangkainya', 3),
        ('pakaian', 3),
        ('kalaulah', 3),
        ('kaulah', 2),
        ('bagaikan', 3),
        ('modernisasi', 5),
        ('praktisi', 3),
        ('reboisasi', 5),
        ('perbauan', 4),
        ('kejauhan', 4),
        ('jikalau', 3),
    ]
    _assert_counts(tests)


def test_line_level_accumulation():
    tests = [
        ('untuk yang menunaikan ibadah haji', 12),
        ('wahai saudara seiman senegara', 12),
        ('tunaikan segera adzan berkumandang', 12),
        ('kaulah sahabat terbaik sejak kecil', 12),
        ('agar engkau menjadi suri tauladan', 12),
        ('Jika ingin hidup penuh kedamaian', 12),
        ('lihatlah dulu bagaimana dirimu', 12),
        ('Cantik bersinar cahaya berkilauan', 12),
        ('bagaimana hati ini tidak rindu', 12),
        ('sekolah yang benar jangan kau abaikan', 12),
        ('Bagaimana mungkin dapat dimartabat', 12),
        ('Selesaikan amanah tanpa berpaling', 12),
        ('Jaga persaudaraan jangan terguncang', 12),
        ('daun hijau jadi sekarat', 9),
    ]
    _assert_counts(tests)


def test_regression_pack_known_cases():
    # Stable single-word regression pack for previously risky patterns.
    tests = [
        ('kiai', 2),
        ('via', 2),
        ('biaya', 3),
        ('kuota', 3),
        ('video', 3),
        ('radio', 3),
        ('teori', 3),
        ('koordinasi', 5),
        ('riwayat', 3),
        ('keyakinan', 4),
        ('bayi', 2),
        ('syariah', 3),
        ('quran', 2),
        ('penguasa', 4),
        ('kebauan', 4),
        ('penciuman', 4),
        ('puasa', 3),
        ('suara', 3),
        ('doa', 2),
        ('muazin', 3),
        ('taat', 2),
        ('ikhtiar', 3),
        ('ziarah', 3),
        ('diam', 2),
        ('piutang', 3),
        # 1. Kasus Diftong (Vokal rangkap satu bunyi)
        ('aula', 3),
        ('audio', 3),
        ('pantai', 2),
        ('landai', 2),
        ('boikot', 3),
        ('amboi', 2),
        ('survei', 2),
        ('kiai', 2),
        ('tv', 2),
        ('AC', 2),
        ('KTP', 3),
        ('CD', 2),
        ('wa', 2),

        # 2. Kasus Vokal Hiatus (Vokal berdampingan beda suku kata)
        ('aulia', 3),
        ('dia', 2),
        ('tua', 2),
        ('kuali', 3),
        ('puisi', 3),
        ('duit', 2),
        ('main', 2),
        ('kait', 2),
        ('bau', 2),
        ('saut', 2),
        ('saat', 2),
        ('beo', 2),

        # 3. Kasus Digraf (Dua huruf satu konsonan)
        ('bangun', 2),
        ('dangau', 2),
        ('nyanyi', 2),
        ('hancur', 2),
        ('syarat', 2),
        ('masyarakat', 4),
        ('khusus', 2),
        ('akhir', 2),

        # 4. Kasus Gugus Konsonan (Kluster) & Serapan
        ('strategi', 3),
        ('psikologi', 4),
        ('kredit', 2),
        ('april', 2),
        ('ekstra', 2),
        ('instrumen', 3),
        ('bentrok', 2),

        # 5. Kasus Pengaruh Imbuhan (Morfologis)
        ('makanan', 3),
        ('lompati', 3),
        ('rapatkan', 3),
        ('gelembung', 3),

        # 6. Kasus Kata Satu Suku Kata (Monosilabel)
        ('bom', 1),
        ('teks', 1),
        ('skors', 1),
        ('cat', 1)
    ]
    _assert_counts(tests)


def test_regression_pack_watchlist_snapshot():
    # Watchlist only: keep visibility for potentially ambiguous words
    # without turning all of them into strict assertions yet.
    watchlist = [
        'seia',
        'aorta',
        'syiar',
        'insyaallah',
        'alhamdulillah',
        'jalan-jalan',
        'berulang-ulang',
        'gimana',
        'nggak',
        'udah',
        'satnitean',
        'Yogyakarta',
        'Sukabumi',
        'aulia',
        'ria',
        'tiang',
        'liar',
        'siul',
    ]

    snapshot = {word: count_syllables(word) for word in watchlist}
    for word, count in snapshot.items():
        print(f"watchlist '{word}': {count}")
        assert count >= 1


def test_validate_pantun_accepts_indonesian_pantun():
    lines = [
        "Ada angin menghempas tomat",
        "Buah semangka di dalam goa",
        "Jika ingin hidup selamat",
        "Jangan lupa sering berdoa",
    ]

    is_valid, reason = validate_pantun(lines)

    assert is_valid is True
    assert reason == "OK"


def test_validate_pantun_rejects_low_lexical_validity_ratio():
    lines = [
        "Keep spirit dan berpikir positif",
        "Langkah kecil tetap konsisten",
        "Jangan lelah menata hati",
        "Agar esok jadi lebih baik",
    ]

    is_valid, reason = validate_pantun(lines, min_valid_word_ratio=0.95)

    assert is_valid is False
    assert reason.startswith("Rasio kosakata dasar Indonesia rendah:")

if __name__ == "__main__":
    test_diphthong_and_hiatus_balance()
    test_line_level_accumulation()
    test_regression_pack_known_cases()
    test_regression_pack_watchlist_snapshot()
    test_validate_pantun_accepts_indonesian_pantun()
    test_validate_pantun_rejects_low_lexical_validity_ratio()
    print("All tests passed!")
