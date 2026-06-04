from pantunify.utils import count_syllables
from pantunify.classifier import find_distinct_javanese_markers, validate_pantun

def test_counts():
    tests = [
        ('Air jernih mengalir ke muara', 11),
        ('anak kecil naik sepeda', 9),
        ('sukses di masa depan akan diraih', 12),
        ('Bingung mencium bau menyengat', 10),
        ('mau jalan dompet dah menipis', 10),
    ]
    
    for text, expected in tests:
        res = count_syllables(text)
        print(f"'{text}': {res} (Expected: {expected})")
        assert res == expected


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
    test_counts()
    test_validate_pantun_accepts_indonesian_pantun()
    test_validate_pantun_rejects_strong_javanese_markers()
    print("All tests passed!")
