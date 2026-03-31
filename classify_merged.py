import re
from pathlib import Path
import difflib
from pantun_utils import clean_line, check_rhyme, count_syllables

def main():
    merged = Path('dataset/merged.txt')
    ok_path = Path('dataset/ok.txt')
    fail_path = Path('dataset/fail.txt')

    if not merged.exists():
        print('File dataset/merged.txt tidak ditemukan.')
        return

    text = merged.read_text(encoding='utf-8')
    blocks = re.split(r'\n\s*\n+', text)

    seen = set()
    # store normalized entries with cached metadata to avoid repeated work
    seen_norms = []  # list of dicts: {'text': str, 'len': int, 'tokens': set}
    ok = []
    fail = []

    for b in blocks:
        lines = [clean_line(l) for l in b.splitlines() if l.strip()]
        if len(lines) != 4:
            continue
        key = tuple(lines)
        if key in seen:
            continue
        # normalized form for fuzzy dedupe
        norm = ' '.join(lines).lower()
        norm = re.sub(r'[^a-z0-9\s]', '', norm)
        dup = False
        # precompute lightweight metadata for this norm
        la = len(norm)
        s1 = set(norm.split())
        for ex in seen_norms:
            ex_text = ex['text']
            if not ex_text:
                continue
            # 1) length ratio filter: if lengths differ too much, skip
            lb = ex['len']
            if abs(la - lb) / max(la, lb) > 0.25:
                continue
            # 2) quick token Jaccard similarity: require reasonable overlap
            s2 = ex['tokens']
            if s1 and s2:
                inter = s1 & s2
                jaccard = len(inter) / len(s1 | s2)
                if jaccard < 0.5:
                    continue
            # 3) substring containment is a strong duplicate signal
            if norm in ex_text or ex_text in norm:
                dup = True
                break
            # 4) expensive but precise check (only for candidates that pass filters)
            if difflib.SequenceMatcher(None, norm, ex_text).ratio() >= 0.92:
                dup = True
                break
        if dup:
            continue
        seen.add(key)
        seen_norms.append({'text': norm, 'len': la, 'tokens': s1})
        # count syllables and words per line
        sylls = [count_syllables(l) for l in lines]
        word_counts = [len(l.split()) for l in lines]
        
        # Validation: 8-12 syllables AND 4-6 words per line
        if any(s < 8 or s > 12 for s in sylls) or any(w < 4 or w > 6 for w in word_counts):
            fail.append(lines)
        else:
            if check_rhyme(*lines):
                ok.append(lines)
            else:
                fail.append(lines)

    with ok_path.open('w', encoding='utf-8') as f:
        for p in ok:
            for l in p:
                f.write(l + '\n')
            f.write('\n')

    with fail_path.open('w', encoding='utf-8') as f:
        for p in fail:
            for l in p:
                f.write(l + '\n')
            f.write('\n')

    print(f'Jumlah pantun OK: {len(ok)}, FAIL: {len(fail)}')

if __name__ == '__main__':
    main()
