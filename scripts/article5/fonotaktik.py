from typing import List, Dict, Any

IPA_VOWELS = {
    'i', 'ɪ', 'e', 'ɛ', 'æ', 'ɑ', 'ɒ', 'ɔ', 'o', 'ʊ', 'u', 'ʌ', 'ə', 'ɜ',
    'ɝ', 'ɚ', 'eɪ', 'aɪ', 'ɔɪ', 'aʊ', 'oʊ', 'əʊ', 'ɪə', 'eə', 'ʊə', 'ju',
    'iː', 'ɑː', 'ɔː', 'uː', 'ɜː'
}

# Pasangan vokal yang harus digabung jika terpecah oleh tokenizer IPA
SPLIT_DIPHTHONGS = {
    ('e', 'ɪ'), ('a', 'ɪ'), ('ɔ', 'ɪ'), ('a', 'ʊ'),
    ('o', 'ʊ'), ('ə', 'ʊ'), ('ɪ', 'ə'), ('e', 'ə'), ('ʊ', 'ə')
}

VALID_ONSETS = {
    'p', 'b', 't', 'd', 'k', 'g', 'f', 'v', 'θ', 'ð', 's', 'z', 'ʃ', 'ʒ', 'h',
    'm', 'n', 'l', 'r', 'j', 'w', 'tʃ', 'dʒ',
    'pl', 'pr', 'bl', 'br', 'tr', 'dr', 'kl', 'kr', 'gl', 'gr',
    'fl', 'fr', 'θr', 'ʃr', 'sl', 'sm', 'sn', 'sp', 'st', 'sk', 'sw',
    'tw', 'dw', 'kw', 'gw', 'θw', 'pj', 'bj', 'tj', 'dj', 'kj', 'gj',
    'fj', 'vj', 'θj', 'sj', 'zj', 'hj', 'mj', 'nj', 'lj',
    'spl', 'spr', 'str', 'skr', 'skw', 'spj', 'stj', 'skj'
}

SYLLABIC_CONSONANTS = {'l', 'n', 'm', 'ŋ', 'r'}
OBSTRUENTS = {'p', 'b', 't', 'd', 'k', 'g', 'f', 'v', 'θ', 'ð', 's', 'z', 'ʃ', 'ʒ', 'tʃ', 'dʒ'}


def _normalize_ipa_units(raw_units: List[str]) -> List[str]:
    """Membersihkan tanda stres dan menggabungkan diftong yang terpecah."""
    cleaned = [u.replace('ˈ', '').replace('ˌ', '') for u in raw_units if u not in {'ˈ', 'ˌ', ''}]
    merged = []
    i = 0
    while i < len(cleaned):
        if i + 1 < len(cleaned) and (cleaned[i], cleaned[i + 1]) in SPLIT_DIPHTHONGS:
            merged.append(cleaned[i] + cleaned[i + 1])
            i += 2
        else:
            merged.append(cleaned[i])
            i += 1
    return merged


def _find_nuclei_indices(units: List[str]) -> List[int]:
    """Mencari posisi inti silabel (vokal atau konsonan silabik akhir)."""
    nuclei = []
    n = len(units)
    for i, u in enumerate(units):
        # Mendukung tanda silabik eksplisit seperti 'l̩' atau 'n̩'
        if u in IPA_VOWELS or u.endswith('̩'):
            nuclei.append(i)
        elif i == n - 1 and u in SYLLABIC_CONSONANTS and i > 0:
            if units[i - 1] in OBSTRUENTS and (not nuclei or nuclei[-1] < i - 1):
                nuclei.append(i)
    return nuclei


def syllabify_en_ipa_robust(raw_units: List[str]) -> Dict[str, Any]:
    """
    Silabifier fonotaktik IPA bahasa Inggris yang tahan banting.
    """
    if not raw_units:
        return {"syllables": [], "syllable_count": 0, "bi": "", "bi_valid": False}

    units = _normalize_ipa_units(raw_units)
    nuclei = _find_nuclei_indices(units)

    if len(nuclei) <= 1:
        bi_str = "B" + "I" * (len(units) - 1)
        return {
            "syllables": ["".join(units)],
            "syllable_count": 1,
            "bi": bi_str,
            "bi_valid": True
        }

    split_points = [0]

    for idx in range(len(nuclei) - 1):
        v1 = nuclei[idx]
        v2 = nuclei[idx + 1]
        consonants = units[v1 + 1 : v2]
        num_c = len(consonants)

        if num_c == 0:
            split_points.append(v2)
        elif num_c == 1:
            if consonants[0] == 'ŋ':
                split_points.append(v2)
            else:
                split_points.append(v1 + 1)
        else:
            best_split = v2
            for c_idx in range(num_c):
                candidate_onset = "".join(consonants[c_idx:])
                if candidate_onset in VALID_ONSETS:
                    best_split = v1 + 1 + c_idx
                    break
            split_points.append(best_split)

    split_points.append(len(units))

    syllables = []
    bi_tags = []

    for i in range(len(split_points) - 1):
        start = split_points[i]
        end = split_points[i + 1]
        chunk = units[start:end]
        if chunk:
            syllables.append("".join(chunk))
            bi_tags.append("B" + "I" * (len(chunk) - 1))

    bi_final = "".join(bi_tags)
    bi_valid = (len(bi_final) == len(units)) and bi_final.startswith("B")

    return {
        "syllables": syllables,
        "syllable_count": len(syllables),
        "bi": bi_final,
        "bi_valid": bi_valid
    }
