from pantunify.utils import count_syllables

def test_counts():
    tests = [
        ("belajar di perpustakaan", 9),
        ("perpustakaan", 5),
        ("pantai", 2),
        ("harimau", 3),
        ("dia", 2),
    ]
    
    for text, expected in tests:
        res = count_syllables(text)
        print(f"'{text}': {res} (Expected: {expected})")
        assert res == expected

if __name__ == "__main__":
    test_counts()
    print("All tests passed!")
