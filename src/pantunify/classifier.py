import re
import difflib
from functools import lru_cache
from pathlib import Path
from .utils import clean_line, check_rhyme, count_syllables, candidate_base_forms


@lru_cache(maxsize=1)
def _load_kata_dasar():
    base_path = Path(__file__).resolve().parents[2] / 'data' / 'kata-dasar.txt'
    if not base_path.exists():
        return set()

    words = set()
    with base_path.open('r', encoding='utf-8') as f:
        for line in f:
            w = re.sub(r'[^a-z]', '', line.strip().lower())
            if w:
                words.add(w)
    return words


def _normalize_token(token):
    return re.sub(r'[^a-z]', '', token.lower())


def _is_valid_indonesian_word(token, kata_dasar):
    if token in kata_dasar:
        return True

    for cand in candidate_base_forms(token):
        if cand in kata_dasar:
            return True

    return False


def lexical_validity_ratio(lines):
    text = ' '.join(clean_line(line).lower() for line in lines)
    tokens = [_normalize_token(t) for t in re.findall(r'[a-zA-Z]+', text)]
    tokens = [t for t in tokens if t]

    if not tokens:
        return 0.0, 0, 0

    kata_dasar = _load_kata_dasar()
    if not kata_dasar:
        return 1.0, len(tokens), len(tokens)

    valid = 0
    for token in tokens:
        if _is_valid_indonesian_word(token, kata_dasar):
            valid += 1

    total = len(tokens)
    return valid / total, valid, total

def validate_pantun(
    lines,
    min_syllables=8,
    max_syllables=12,
    min_valid_word_ratio=0.8,
):
    """
    Validasi sebuah blok pantun (4 baris).
    Mengembalikan tuple (is_valid, reason).
    """
    if len(lines) != 4:
        return False, "Bukan 4 baris"

    # Bersihkan baris
    cleaned = [clean_line(l) for l in lines]

    valid_ratio, valid_count, total_count = lexical_validity_ratio(cleaned)
    if valid_ratio < min_valid_word_ratio:
        return False, (
            f"Rasio kosakata dasar Indonesia rendah: "
            f"{valid_count}/{total_count} ({valid_ratio:.1%})"
        )
    
    # Hitung suku kata
    sylls = [count_syllables(l) for l in cleaned]
    
    # Validasi jumlah suku kata per baris
    if any(s < min_syllables or s > max_syllables for s in sylls):
        return False, f"Suku kata tidak ideal: {sylls}"
        
    # Validasi rima
    if not check_rhyme(*cleaned):
        return False, "Rima tidak valid"
        
    return True, "OK"

def deduplicate_pantuns(blocks, threshold=0.92):
    """
    Menghapus duplikasi pantun dari list blok pantun.
    Setiap blok adalah list string (baris).
    """
    seen = set()
    seen_norms = []
    unique_blocks = []
    duplicate_blocks = []

    for lines in blocks:
        if len(lines) != 4:
            continue
            
        cleaned = [clean_line(l) for l in lines]
        key = tuple(cleaned)
        
        if key in seen:
            duplicate_blocks.append(lines)
            continue
            
        # Normalisasi untuk fuzzy dedupe
        norm = ' '.join(cleaned).lower()
        norm = re.sub(r'[^a-z0-9\s]', '', norm)
        
        dup = False
        la = len(norm)
        s1 = set(norm.split())
        
        for ex in seen_norms:
            ex_text = ex['text']
            lb = ex['len']
            
            # 1) Filter rasio panjang
            if abs(la - lb) / max(la, lb) > 0.25:
                continue
                
            # 2) Filter Jaccard token
            s2 = ex['tokens']
            if s1 and s2:
                inter = s1 & s2
                jaccard = len(inter) / len(s1 | s2)
                if jaccard < 0.5:
                    continue
            
            # 3) Cek substring
            if norm in ex_text or ex_text in norm:
                dup = True
                break
                
            # 4) Perbandingan presisi
            if difflib.SequenceMatcher(None, norm, ex_text).ratio() >= threshold:
                dup = True
                break
        
        if dup:
            duplicate_blocks.append(lines)
            continue
            
        seen.add(key)
        seen_norms.append({'text': norm, 'len': la, 'tokens': s1})
        unique_blocks.append(lines)
        
    return unique_blocks, duplicate_blocks
