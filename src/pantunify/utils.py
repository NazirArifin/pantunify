import re
import unicodedata


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

    # 1. HANDLING SINGKATAN UMUM (Konversi ke fonem lisan)
    abbreviation_map = {
        'tv': 've',   # te-ve (vokal e)
        'ac': 'se',   # a-se (vokal e)
        'hp': 'pe',    # ha-pe (vokal e)
        'bumn': 'en',  # be-u-em-en -> berakhiran 'en'
        'hp': 'pe',    # ha-pe (vokal e)
        'wa': 'a',     # we-a (vokal a)
    }
    if last in abbreviation_map:
        last = abbreviation_map[last]
        
    # Cek diftong di bagian paling akhir
    diphthongs = ['ai', 'au', 'oi', 'ei']
    for d in diphthongs:
        if last.endswith(d):
            return d
        
    # Jika kata berakhiran huruf vokal (seperti: dia, setia)
    # Rima pantun tradisional hanya mengambil 1 vokal terakhirnya saja.
    if last[-1] in 'aiueo':
        return last[-1]
            
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

# Kamus pengecualian untuk kata-kata dengan pola suku kata tidak beraturan dalam BI
_KAMUS_PENGECUALIAN = {
    'audio': 3,
}

# Singkatan lazim yang sering ditulis huruf kecil dan dibaca per huruf.
# Token lain yang ambigu sebaiknya ditambahkan bertahap berbasis data nyata.
_LOWER_ABBREVIATION_SYLLABLES = {
    'wa': 2,   # WhatsApp -> wa
    'ac': 2,   # Air Conditioner -> ac
}

# Pasangan vokal ini dibaca terpisah (hiatus), jadi tidak boleh dipotong sebagai diftong akhir.
_HIATUS_TERMINAL = {'mau', 'bau'}
_DIPHTHONG_RE = r'(ai|au|oi|ei)'
_INTERNAL_DIPHTHONG_RE = r'(ai|au)'

_INFLECTION_SUFFIXES = ['lah', 'kah', 'nya', 'ku', 'mu']
_DERIVATIONAL_SUFFIXES = ['kan', 'an']

_PREFIX_RULES = [
    ('meng', ['', 'k']),
    ('meny', ['s']),
    ('men', ['', 't']),
    ('mem', ['', 'p']),
    ('me', ['']),
    ('peng', ['', 'k']),
    ('peny', ['s']),
    ('pen', ['', 't']),
    ('pem', ['', 'p']),
    ('pe', ['']),
    ('ber', ['']),
    ('bel', ['']),
    ('be', ['']),
    ('per', ['']),
    ('pel', ['']),
    ('ter', ['']),
    ('te', ['']),
    ('se', ['']),
    ('ke', ['']),
    ('di', ['']),
]


def _count_core_word(w):
    vowels = re.findall(r'[aiueo]', w)
    count = len(vowels)

    # Diftong akhir kata.
    if re.search(_DIPHTHONG_RE + r'$', w) and w not in _HIATUS_TERMINAL:
        count -= 1

    # Diftong internal sebelum konsonan cenderung relevan pada kata panjang
    # turunan (mis. bagai, saudara, tauladan, kedaulatan), tetapi tidak untuk
    # pola akhir seperti "aik" pada "terbaik" (ter-ba-ik).
    if len(w) >= 5:
        diph_internal = re.findall(
            _INTERNAL_DIPHTHONG_RE + r'(?=[bcdfgjklmnpqrstvwxyz][a-z]*[aiueo])',
            w,
        )
        count -= len(diph_internal)

    if count < 1:
        count = 1

    return count


def _strip_inflection_suffixes(w):
    stem = w
    suffixes = []

    changed = True
    while changed:
        changed = False
        for suf in _INFLECTION_SUFFIXES:
            if stem.endswith(suf) and (len(stem) - len(suf) >= 3):
                stem = stem[:-len(suf)]
                suffixes.append(suf)
                changed = True
                break

    return stem, suffixes


def _strip_derivational_suffix_once(w):
    for suf in _DERIVATIONAL_SUFFIXES:
        if w.endswith(suf) and (len(w) - len(suf) >= 3):
            return w[:-len(suf)], suf
    return w, ''


def _strip_prefix_with_recoding_once(w):
    out = []
    for pref, recodes in _PREFIX_RULES:
        if w.startswith(pref) and (len(w) - len(pref) >= 3):
            tail = w[len(pref):]
            for rec in recodes:
                cand = rec + tail
                if len(cand) >= 3:
                    out.append((pref, cand))
    return out


def _build_stem_candidates(w):
    # Keluaran: list tuple (stem, affix_count)
    stem1, inf_sufs = _strip_inflection_suffixes(w)
    stem2, der_suf = _strip_derivational_suffix_once(stem1)

    suffix_count = len(inf_sufs) + (1 if der_suf else 0)

    # Kandidat tanpa prefiks.
    cands = [(stem2, suffix_count)]

    # Kandidat 1 prefiks.
    for _, p1_stem in _strip_prefix_with_recoding_once(stem2):
        cands.append((p1_stem, suffix_count + 1))

        # Kandidat 2 prefiks.
        for _, p2_stem in _strip_prefix_with_recoding_once(p1_stem):
            cands.append((p2_stem, suffix_count + 2))

    # Deduplikasi sederhana.
    uniq = {}
    for stem, n_aff in cands:
        key = (stem, n_aff)
        uniq[key] = True

    return list(uniq.keys())


def candidate_base_forms(token):
    """Return possible base-form candidates for a token after light affix stripping."""
    w = _normalize_word(token)
    if not w:
        return set()

    out = {w}
    for stem, _ in _build_stem_candidates(w):
        if stem:
            out.add(stem)

    return out


def _count_word_hybrid(w):
    # Baseline tetap dipakai sebagai fallback aman.
    baseline = _count_core_word(w)

    candidates = _build_stem_candidates(w)
    if not candidates:
        return baseline

    best = baseline
    for stem, affix_count in candidates:
        segmented = _count_core_word(stem) + affix_count

        # Hindari under-count pada pola seperti ke-ma-u-an / per-ba-u-an.
        if (
            w.endswith(('mauan', 'bauan'))
            and stem.endswith(('mau', 'bau'))
            and segmented < baseline
        ):
            continue

        # Guard-rail: hanya izinkan koreksi turun maksimal 1 suku kata.
        if segmented <= baseline and (baseline - segmented) <= 1:
            if segmented < best:
                best = segmented

    return best


def _normalize_word(raw_word):
    w = raw_word.replace('-', '')
    w = unicodedata.normalize('NFD', w)
    w = w.encode('ascii', 'ignore').decode('utf-8')
    w = w.lower()
    w = re.sub(r'[^a-z]', '', w)
    return w


def _is_initialism_like(w):
    # Singkatan huruf (mis. tv, ktp, cd, sms) biasanya dilafalkan per huruf.
    # Heuristik: token alfabetis tanpa vokal, panjang 2-6 karakter.
    return 2 <= len(w) <= 6 and re.fullmatch(r'[bcdfghjklmnpqrstvwxyz]+', w) is not None


def _is_upper_initialism(raw_word, normalized_word):
    raw_letters = re.sub(r'[^A-Za-z]', '', raw_word)
    if not raw_letters:
        return False
    if not (2 <= len(raw_letters) <= 6):
        return False
    if not raw_letters.isupper():
        return False
    return raw_letters.lower() == normalized_word


def _iter_normalized_words(text):
    for raw_word in re.split(r'\s+', text):
        if not raw_word:
            continue
        w = _normalize_word(raw_word)
        if w:
            yield raw_word, w

def count_syllables(kata):
    """
    Menghitung suku kata sebuah kata Bahasa Indonesia dengan heuristik yang lebih reliabel.
    Menangani: diftong lengkap, gugus vokal, tanda hubung, karakter beraksent, dan kata pengecualian.
    """
    # Fungsi menerima sebuah baris/teks dan menjumlahkan suku kata per kata.

    if not isinstance(kata, str):
        return 0

    teks = kata.strip()
    if not teks:
        return 0

    total = 0
    for raw_word, w in _iter_normalized_words(teks):
        if w in _LOWER_ABBREVIATION_SYLLABLES:
            total += _LOWER_ABBREVIATION_SYLLABLES[w]
            continue

        if _is_upper_initialism(raw_word, w):
            total += len(w)
            continue

        if _is_initialism_like(w):
            total += len(w)
            continue

        # Cek kamus pengecualian per kata
        if w in _KAMUS_PENGECUALIAN:
            total += _KAMUS_PENGECUALIAN[w]
            continue

        total += _count_word_hybrid(w)

    return total