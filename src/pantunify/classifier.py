import re
import difflib
from .utils import clean_line, check_rhyme, count_syllables

def validate_pantun(lines, min_syllables=8, max_syllables=12, min_words=4, max_words=6):
    """
    Validasi sebuah blok pantun (4 baris).
    Mengembalikan tuple (is_valid, reason).
    """
    if len(lines) != 4:
        return False, "Bukan 4 baris"

    # Bersihkan baris
    cleaned = [clean_line(l) for l in lines]
    
    # Hitung suku kata dan kata
    sylls = [count_syllables(l) for l in cleaned]
    word_counts = [len(l.split()) for l in cleaned]
    
    # Validasi jumlah suku kata per baris
    if any(s < min_syllables or s > max_syllables for s in sylls):
        return False, f"Suku kata tidak ideal: {sylls}"
        
    # Validasi jumlah kata per baris
    if any(w < min_words or w > max_words for w in word_counts):
        return False, f"Jumlah kata tidak ideal: {word_counts}"
        
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
