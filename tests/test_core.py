from pantunify.utils import count_syllables
from pantunify.classifier import find_distinct_javanese_markers, validate_pantun

def _assert_counts(tests):
    for text, expected in tests:
        res = count_syllables(text)
        print(f"'{text}': {res} (Expected: {expected})")
        assert res == expected


def test_morphology_prefix_suffix_cases():
    tests = [
        ('menunaikan', 4),
        ('tunaikan', 3),
        ('abaikan', 3),
        ('selesaikan', 4),
        ('kedaulatan', 4),
        ('persaudaraan', 5),
    ]
    _assert_counts(tests)


def test_diphthong_and_hiatus_balance():
    tests = [
        ('saudara', 3),
        ('bagaimana', 4),
        ('terbaik', 3),
        ('berkilauan', 4),
        ('tauladan', 3),
        ('naik', 2),
        ('daun', 2),
        ('mau', 2),
        ('bau', 2),
    ]
    _assert_counts(tests)


def test_clitic_and_particle_suffixes():
    tests = [
        ('saudaraku', 4),
        ('pakaiannya', 4),
        ('tangkainya', 3),
        ('pakaian', 3),
        ('kalaulah', 3),
        ('kaulah', 2),
        ('bagaikan', 3),
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
    ]
    _assert_counts(tests)


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


def test_validate_pantun_rejects_strong_javanese_markers():
    lines = [
        "Numpak andhong ning Kotagedhe",
        "tuku ali ali kagem eyang putri",
        "Bilih aku nduwe luput gedhe",
        "nyuwun kawelasan sampean ampuni",
    ]

    markers = find_distinct_javanese_markers(lines)
    is_valid, reason = validate_pantun(lines)

    assert len(markers) >= 2
    assert is_valid is False
    assert reason.startswith("Terdeteksi penanda bahasa Jawa:")

if __name__ == "__main__":
    test_morphology_prefix_suffix_cases()
    test_diphthong_and_hiatus_balance()
    test_clitic_and_particle_suffixes()
    test_line_level_accumulation()
    test_validate_pantun_accepts_indonesian_pantun()
    test_validate_pantun_rejects_strong_javanese_markers()
    print("All tests passed!")
