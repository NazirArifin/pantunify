# Export Notebook: indobert_token_classifier_colab.ipynb

Sumber: `/mnt/d/workspace/penelitian/Disertasi/coding/rima/scripts/article5/indobert_token_classifier_colab.ipynb`

## Cell 1 (markdown)

# Layer-1 IndoBERT Token Classifier (Colab)

Notebook ini adalah versi Colab dari `layer1_indobert_token_classifier.py`.

Urutan pakai:
1. Jalankan sel **Install dependencies**.
2. Atur `MODEL_DIR` dan input inferensi.
3. Jalankan sel inferensi Layer 1 (single atau batch).
4. Jalankan sel Layer 2a/2b untuk konversi fonem.
5. Jalankan Layer 3 (CRF) untuk syllabification berbasis tag B/I.

Catatan: notebook ini mengharuskan runtime GPU Colab (CUDA) dan akan otomatis memakai GPU.

## Cell 2 (code)

```python
# Jika runtime Colab baru, jalankan dulu.
!python -m pip install -q torch transformers g2p-id g2p-en nltk python-crfsuite
```

## Cell 3 (code)

```python
from functools import lru_cache
import re
from pathlib import Path
from typing import NamedTuple, Any


# Definisikan struktur TokenLabel
class TokenLabel(NamedTuple):
    token: str
    label: str
    score: float


def _import_layer2_deps():
    try:
        import nltk  # type: ignore
        from g2p_id import G2P as G2P_ID_CLASS  # KEMBALI KE HURUFP BESAR (Sesuai dengan library Colab Anda)
        from g2p_en import G2p as G2P_EN_CLASS  # type: ignore
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            "Dependensi Layer 2 belum lengkap. Install dulu: python -m pip install g2p-id g2p-en nltk"
        ) from err
    return nltk, G2P_ID_CLASS, G2P_EN_CLASS


def _ensure_g2p_en_resources(nltk) -> None:
    packages = {
        "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
        "cmudict": "corpora/cmudict",
        "punkt": "tokenizers/punkt",
    }
    for pkg, path in packages.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


nltk, G2P_ID_CLASS, G2P_EN_CLASS = _import_layer2_deps()
_ensure_g2p_en_resources(nltk)

# Inisialisasi model
g2p_id_model = G2P_ID_CLASS()
g2p_en_model = G2P_EN_CLASS()

ALPHABET_EJA_MAP = {
    "a": "a", "b": "be", "c": "ce", "d": "de", "e": "e", "f": "ef", "g": "ge",
    "h": "ha", "i": "i", "j": "je", "k": "ka", "l": "el", "m": "em", "n": "en",
    "o": "o", "p": "pe", "q": "ki", "r": "er", "s": "es", "t": "te", "u": "u",
    "v": "ve", "w": "we", "x": "eks", "y": "ye", "z": "zet"
}

DISMV_SHORT_PATTERN = "v o n"
DISMV_LONG_PATTERN = "v o n e m"

ARPABET_TO_IPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "EH": "ɛ", "ER": "ɝ",
    "EY": "eɪ", "F": "f", "G": "ɡ", "HH": "h", "IH": "ɪ", "IY": "i",
    "JH": "dʒ", "K": "k", "L": "l", "M": "m", "NG": "ŋ",
    "OW": "oʊ", "OY": "ɔɪ", "P": "p", "R": "r", "S": "s", "SH": "ʃ",
    "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u", "V": "v", "W": "w",
    "Y": "j", "Z": "z", "ZH": "ʒ"
}


def arpabet_to_ipa_sequence(tokens: list[str]) -> str:
    ipa_tokens: list[str] = []
    for token in tokens:
        base = re.sub(r"\d", "", token).upper()
        if not base:
            continue
        ipa_tokens.append(ARPABET_TO_IPA.get(base, base.lower()))
    # KUNCI PERBAIKAN 1: Rapatkan karakter fonem di dalam SATU kata (Tanpa Spasi)
    return "".join(ipa_tokens)


def _coerce_g2p_output_to_str(raw) -> str:
    # PERBAIKAN 1: Ekstraksi tuple Wikidepia (phonemes, syllables)
    if isinstance(raw, tuple):
        return _coerce_g2p_output_to_str(raw[0])

    if isinstance(raw, list):
        if raw and isinstance(raw[0], list):
            flat_list = [ph for sublist in raw for ph in sublist if ph]
            return " ".join(flat_list)
        return " ".join(str(part) for part in raw if part)

    return str(raw).strip()



def _g2p_id_convert(word: str) -> str:
    # MEMBAGI KALIMAT PER KATA JIKA INPUT MENGANDUNG SPASI (Hasil dari rumus_a_singkatan)
    words = word.split()
    if len(words) > 1:
        # KUNCI PERBAIKAN: Gunakan "" (tanpa spasi) untuk menyambung hasil ejaan singkatan
        return "".join(_g2p_id_convert(w) for w in words)

    try:
        res = g2p_id_model.to_phoneme(word)
        return _coerce_g2p_output_to_str(res)
    except AttributeError:
        try:
            return _coerce_g2p_output_to_str(g2p_id_model(word))
        except Exception:
            try:
                return _coerce_g2p_output_to_str(g2p_id_model.convert(word))
            except Exception as e:
                raise AttributeError(f"Gagal memanggil API g2p_id: {str(e)}")



@lru_cache(maxsize=8192)
def g2p_id_phoneme(word: str) -> str:
    return _g2p_id_convert(word)


@lru_cache(maxsize=8192)
def g2p_foreign_phoneme(word: str) -> str:
    raw = g2p_en_model(word)
    phones = [p for p in raw if p and p != " " and p != "<unk>"]
    ipa = arpabet_to_ipa_sequence(phones)
    return ipa if ipa else g2p_id_phoneme(word)


def rumus_a_singkatan(word: str) -> str:
    # PERBAIKAN 2: Gunakan pemisah spasi " " agar menjadi "es pe be u", bukan "espebeu"
    ejaan = " ".join(ALPHABET_EJA_MAP.get(ch, ch) for ch in word.lower() if ch.isalpha())
    if not ejaan:
        return g2p_id_phoneme(word)
    return g2p_id_phoneme(ejaan)


def rumus_c_clamping(word: str) -> str:
    """
    Rumus C: Clamping Aturan Panjang untuk Kata Tugas (Tanpa Spasi)
    """
    if len(word) <= 3:
        return "von"  # UBAH: dari "v o n" menjadi "von"
    return "vonem"    # UBAH: dari "v o n e m" menjadi "vonem"


def rumus_b_phonotactic_mapping(word: str) -> str:
    """
    Rumus B: Syllable Weight Estimation via Phonotactic Mapping (Tanpa Spasi)
    """
    digraf_patterns = ["ng", "ny", "kh", "br", "tr", "bl", "cl", "kr", "pr"]
    processed_word = word
    for digraf in digraf_patterns:
        processed_word = processed_word.replace(digraf, "X")

    has_koda = word[-1] in "bcdfgjklmnpqrstvwxyz"

    base_count = len(processed_word)
    if has_koda and base_count > 1:
        core_phonemes = base_count - 1
    else:
        core_phonemes = base_count

    if core_phonemes <= 2:
        syllables = 2
    elif core_phonemes == 3:
        syllables = 3
    else:
        syllables = 4

    # KUNCI PERBAIKAN: Hapus semua spasi pada string pola suku kata tiruan
    pola_silabel = {
        2: "vonem",
        3: "vonemis",        # UBAH: dari "v o n e m i s" menjadi "vonemis"
        4: "vonemisasi"
    }

    return pola_silabel.get(syllables, "vonem")


def rumus_bc_disemvowelling(word: str) -> str:
    alpha_only = "".join(ch for ch in word.lower() if ch.isalpha())
    if not alpha_only:
        return DISMV_SHORT_PATTERN
    if len(alpha_only) <= 3:
        return DISMV_SHORT_PATTERN
    return DISMV_LONG_PATTERN


def _normalize_label(label: str) -> str:
    return label.upper().strip()


LABELS_ASING = {"B-ASG", "I-ASG"}
LABELS_SINGKATAN = {"B-SKT", "I-SKT"}
LABELS_DISEMVOWEL = {"B-DSM"}
LABELS_TYPO = {"B-TYP"}
LABELS_VALID = {"O", *LABELS_ASING, *LABELS_SINGKATAN, *LABELS_DISEMVOWEL, *LABELS_TYPO}


def _use_rumus_c_for_disemvowel(alpha_only: str) -> bool:
    # Tanpa kamus/hard-code kata: gunakan heuristik bentuk token.
    # Bentuk pendek (<=3) cenderung kata tugas/slang pendek -> Rumus C.
    return len(alpha_only) <= 3


def _estimate_disemvowel_pattern(alpha_only: str) -> str:
    length = len(alpha_only)
    if length <= 2:
        return "von"
    if length <= 4:
        return "vonem"
    if length <= 6:
        return "vonemis"
    return "vonemisasi"


def layer2a_2b_switchboard(token: str, label: str) -> str:
    norm_label = _normalize_label(label)
    token_lower = token.lower()

    alpha_only = "".join(ch for ch in token_lower if ch.isalpha())
    has_vowel = any(vowel in alpha_only for vowel in "aiueo")

    # --- KOREKSI ATURAN LINGUISTIK SPESIFIK ---
    if norm_label in {"O", "B-TYP"} and not has_vowel and len(alpha_only) >= 2:
        norm_label = "B-DSM"
    # ------------------------------------------

    if norm_label == "O":
        return g2p_id_phoneme(token)

    if norm_label in LABELS_ASING:
        res_asg = g2p_foreign_phoneme(token)
        # KUNCI PERBAIKAN 2: Pastikan hasil fonem bahasa asing bersih dari spasi internal
        return res_asg.replace(" ", "")

    if norm_label in LABELS_SINGKATAN:
        return rumus_a_singkatan(token)

    if norm_label in LABELS_DISEMVOWEL:
        if alpha_only and not has_vowel:
            return _estimate_disemvowel_pattern(alpha_only)
        if _use_rumus_c_for_disemvowel(alpha_only):
            res_dsm = rumus_c_clamping(alpha_only)
        else:
            res_dsm = rumus_b_phonotactic_mapping(alpha_only)
        return res_dsm.replace(" ", "")

    if norm_label in LABELS_TYPO:
        return g2p_id_phoneme(token)

    raise ValueError(f"Label tidak dikenali: {label!r}")


def rumus_b_phonotactic_mapping(word: str) -> str:
    """
    Rumus B: Syllable Weight Estimation via Phonotactic Mapping
    """
    digraf_patterns = ["ng", "ny", "kh", "br", "tr", "bl", "cl", "kr", "pr"]
    processed_word = word
    for digraf in digraf_patterns:
        processed_word = processed_word.replace(digraf, "X")

    has_koda = word[-1] in "bcdfgjklmnpqrstvwxyz"

    base_count = len(processed_word)
    if has_koda and base_count > 1:
        core_phonemes = base_count - 1
    else:
        core_phonemes = base_count

    # Penyesuaian rentang silabel agar kata sekelas "brgkt" (core = 2) pas di 2 silabel
    if core_phonemes <= 2:
        syllables = 2
    elif core_phonemes == 3:
        syllables = 3
    else:
        syllables = 4

    pola_silabel = {
        2: "v o n e m",
        3: "v o n e m i s",
        4: "v o n e m i s a s i"
    }

    return pola_silabel.get(syllables, "v o n e m")

def run_layer2_from_predictions(predictions: list[TokenLabel]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in predictions:
        phoneme = layer2a_2b_switchboard(item.token, item.label)
        rows.append({
            "token": item.token,
            "label": item.label,
            "score": round(item.score, 6),
            "phoneme": phoneme,
        })
    return rows


def run_layer2_batch(batch_predictions: list[list[TokenLabel]]) -> list[list[dict[str, object]]]:
    return [run_layer2_from_predictions(preds) for preds in batch_predictions]


def _import_layer3_deps():
    try:
        import pycrfsuite  # type: ignore
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            "Dependensi Layer 3 belum lengkap. Install dulu: python -m pip install python-crfsuite"
        ) from err
    return pycrfsuite


def load_layer3_crf_model(model_path: str | Path):
    pycrfsuite = _import_layer3_deps()
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model CRF Layer 3 tidak ditemukan: {model_file}")
    tagger = pycrfsuite.Tagger()
    tagger.open(str(model_file))
    return tagger


VOWEL_UNITS = {
    "a", "i", "u", "e", "o", "ə", "ɛ", "ɔ", "ɪ", "ʊ", "ɑ", "æ", "ʌ", "ɝ", "ɚ",
    "eɪ", "aɪ", "aʊ", "ɔɪ", "oʊ",
    "ai", "au", "oi",
}
HIATUS_TERMINAL = {"mau", "bau"}


def _phoneme_units(phoneme: str) -> list[str]:
    # 1. Bersihkan spasi dan tanda hubung
    cleaned = "".join(ch for ch in phoneme if not ch.isspace() and ch != "-")

    # 2. Pecah per-karakter, tetapi pertahankan diftong sebagai satu unit vokal.
    #    - IPA asing (aɪ/aʊ/eɪ/oʊ/ɔɪ) selalu digabung.
    #    - Diftong latin (ai/au/oi) hanya digabung jika berada di akhir kata.
    #      Contoh: "pantai" -> pan-tai (ai akhir), tetapi "kain" -> ka-in (ai + konsonan akhir).
    units = []
    i = 0
    while i < len(cleaned):
        if i + 1 < len(cleaned):
            pair = cleaned[i:i+2]
            if pair in {"eɪ", "aɪ", "aʊ", "ɔɪ", "oʊ"}:
                units.append(pair)
                i += 2
                continue
            if pair in {"ai", "au", "oi"} and i + 2 == len(cleaned):
                units.append(pair)
                i += 2
                continue
        units.append(cleaned[i])
        i += 1
    return units


def _normalize_grapheme_word(token: str) -> str:
    return "".join(ch for ch in token.lower() if ch.isalpha())


def _count_syllables_grapheme_word(token: str) -> int:
    w = _normalize_grapheme_word(token)
    if not w:
        return 0

    count = sum(1 for ch in w if ch in "aiueo")
    if re.search(r"(ai|au|oi|ei)$", w) and w not in HIATUS_TERMINAL:
        count -= 1
    return max(1, count)


def _layer3_features(units: list[str], idx: int) -> dict[str, object]:
    cur = units[idx]
    prev = units[idx - 1] if idx > 0 else "<BOS>"
    nxt = units[idx + 1] if idx + 1 < len(units) else "<EOS>"
    feats: dict[str, object] = {
        "bias": 1.0,
        "unit": cur,
        "is_vowel": cur in VOWEL_UNITS,
        "is_consonant": cur not in VOWEL_UNITS,
        "prev_unit": prev,
        "next_unit": nxt,
        "prev_is_vowel": prev in VOWEL_UNITS,
        "next_is_vowel": nxt in VOWEL_UNITS,
        "is_start": idx == 0,
        "is_end": idx == len(units) - 1,
    }
    return feats


def _decode_syllables_from_bi(units: list[str], tags: list[str]) -> list[str]:
    if len(units) != len(tags):
        raise ValueError("Panjang fonem dan tag BI tidak sama.")
    if not units:
        return []

    syllables: list[str] = []
    current = ""
    for idx, (unit, tag) in enumerate(zip(units, tags)):
        if tag not in {"B", "I"}:
            raise ValueError(f"Tag CRF Layer 3 tidak valid: {tag!r}. Harus B atau I.")
        if tag == "B" or idx == 0:
            if current:
                syllables.append(current)
            current = unit
        else:
            current += unit
    if current:
        syllables.append(current)
    return syllables


def run_layer3_from_layer2_rows(layer2_rows: list[dict[str, object]], crf_tagger) -> list[dict[str, object]]:
    rows_out: list[dict[str, object]] = []
    for row in layer2_rows:
        phoneme = str(row.get("phoneme", ""))

        units = _phoneme_units(phoneme)
        if not units:
            rows_out.append({
                **row,
                "layer3_bi": [],
                "layer3_syllables": [],
                "layer3_syllable_count": 0,
            })
            continue

        if crf_tagger is not None:
            features = [_layer3_features(units, i) for i in range(len(units))]
            bi_tags = crf_tagger.tag(features)
        else:
            bi_tags = _heuristic_bi_tags(units)

        syllables = _decode_syllables_from_bi(units, bi_tags)

        # Hitung final silabel berbasis fonem (target utama pipeline).
        # Grafem dipertahankan hanya sebagai sinyal audit/debug.
        phoneme_syllable_count = sum(1 for u in units if u in VOWEL_UNITS)
        grapheme_syllable_count = _count_syllables_grapheme_word(str(row.get("token", "")))
        syllable_count = phoneme_syllable_count
        chunk_count = bi_tags.count("B")

        rows_out.append({
            **row,
            "layer3_units": units,
            "layer3_bi": bi_tags,
            "layer3_syllables": syllables,
            "layer3_chunk_count": max(1, chunk_count),
            "layer3_syllable_count_phoneme": max(1, phoneme_syllable_count),
            "layer3_syllable_count_grapheme": max(1, grapheme_syllable_count),
            "layer3_syllable_count": max(1, syllable_count),
        })
    return rows_out


RHYME_VOWELS = VOWEL_UNITS


def _extract_rhyme_key_from_syllable(last_syllable: str) -> str:
    if not last_syllable:
        return ""
    chars = [ch for ch in last_syllable if not ch.isspace() and ch != "-"]
    if not chars:
        return ""

    for idx in range(len(chars) - 1, -1, -1):
        if chars[idx] in RHYME_VOWELS:
            return "".join(chars[idx:])
    return "".join(chars)


def summarize_pantun_lines(all_results: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for line in all_results:
        layer3_rows = line.get("layer3", [])
        if not isinstance(layer3_rows, list):
            raise TypeError("Field 'layer3' harus berupa list.")

        syllable_count = 0
        last_syllable = ""

        # PERBAIKAN UTAMA: Lakukan akumulasi total nilai silabel dari SETIAP KATA di baris tersebut
        for row in layer3_rows:
            count_value = row.get("layer3_syllable_count", 0)
            if not isinstance(count_value, int):
                raise TypeError("Field 'layer3_syllable_count' harus integer.")

            # Tambahkan nilai silabel kata saat ini ke total baris
            syllable_count += count_value

            syllables = row.get("layer3_syllables", [])
            if not isinstance(syllables, list):
                raise TypeError("Field 'layer3_syllables' harus berupa list.")
            if syllables:
                last_syllable = str(syllables[-1])

        rhyme_key = _extract_rhyme_key_from_syllable(last_syllable)
        summary.append({
            "bait_index": line.get("bait_index"),
            "text": line.get("text", ""),
            "syllable_count": syllable_count, # Nilai akumulasi yang sudah benar
            "last_syllable": last_syllable,
            "rhyme_key": rhyme_key,
        })
    return summary
```

## Cell 4 (code)

```python
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TokenLabel:
    token: str
    label: str
    score: float


def _validate_model_dir(model_dir: Path) -> None:
    required_files = ("config.json", "tokenizer.json", "tokenizer_config.json")
    missing_required = [name for name in required_files if not (model_dir / name).exists()]
    if missing_required:
        raise FileNotFoundError(
            f"Model directory tidak lengkap: {model_dir}. "
            f"File wajib belum ada: {', '.join(missing_required)}."
        )

    has_weights = any(
        (model_dir / filename).exists()
        for filename in (
            "pytorch_model.bin",
            "model.safetensors",
            "tf_model.h5",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )
    )
    if not has_weights:
        raise FileNotFoundError(
            f"Model weights tidak ditemukan di {model_dir}. "
            "Simpan salah satu file berikut: "
            "pytorch_model.bin / model.safetensors / tf_model.h5 "
            "(atau index file untuk model yang di-shard)."
        )


def _import_runtime_deps() -> tuple[Any, Any, Any]:
    try:
        import torch  # type: ignore
        from transformers import AutoModelForTokenClassification, AutoTokenizer  # type: ignore
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            "Dependensi belum lengkap. Install dulu: python -m pip install torch transformers"
        ) from err
    return torch, AutoTokenizer, AutoModelForTokenClassification


def load_layer1_model(model_dir: Path):
    _validate_model_dir(model_dir)
    torch, AutoTokenizer, AutoModelForTokenClassification = _import_runtime_deps()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
    if not torch.cuda.is_available():
        raise RuntimeError("GPU CUDA tidak terdeteksi. Pastikan Colab runtime memakai GPU.")
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    return tokenizer, model, torch, device


def _normalize_text_for_token_labeling(text: str) -> str:
    normalized = text.replace(",", " ")
    normalized = re.sub(r"(?<=\\w)-(?=\\w)", " ", normalized)
    normalized = re.sub(r"\\s+", " ", normalized).strip()
    return normalized


def _split_pantun_baits(text: str) -> list[str]:
    baits: list[str] = []
    # Memotong otomatis setiap ada baris baru (enter)
    for part in text.splitlines():
        cleaned = _normalize_text_for_token_labeling(part)
        if cleaned:
            baits.append(cleaned)
    return baits


def predict_labels(
    text: str,
    tokenizer,
    model,
    torch,
    max_length: int = 256,
) -> list[TokenLabel]:
    cleaned_text = _normalize_text_for_token_labeling(text)
    words = cleaned_text.split()
    if not words:
        return []

    encoded = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    model_inputs = {key: value.to(model.device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = model(**model_inputs)

    probs = torch.softmax(output.logits, dim=-1)
    pred_ids = torch.argmax(output.logits, dim=-1)[0].tolist()
    word_ids = encoded.word_ids(batch_index=0)
    id2label = model.config.id2label
    if not id2label:
        raise ValueError("id2label tidak ditemukan pada config model.")

    results: list[TokenLabel] = []
    seen_word_index = set()
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx in seen_word_index:
            continue
        seen_word_index.add(word_idx)
        pred_id = pred_ids[token_idx]
        label = id2label.get(pred_id, str(pred_id))
        confidence = float(probs[0, token_idx, pred_id].item())
        results.append(TokenLabel(token=words[word_idx], label=label, score=confidence))

    return results


def predict_labels_batch(
    texts: list[str],
    tokenizer,
    model,
    torch,
    max_length: int = 256,
    batch_size: int = 16,
) -> list[list[TokenLabel]]:
    if not texts:
        return []

    cleaned_texts = [_normalize_text_for_token_labeling(text) for text in texts]
    words_per_text = [text.split() for text in cleaned_texts]
    all_results: list[list[TokenLabel]] = [[] for _ in cleaned_texts]
    id2label = model.config.id2label
    if not id2label:
        raise ValueError("id2label tidak ditemukan pada config model.")

    for start in range(0, len(cleaned_texts), batch_size):
        end = min(start + batch_size, len(cleaned_texts))
        batch_words = words_per_text[start:end]
        encoded = tokenizer(
            batch_words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        model_inputs = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model(**model_inputs)

        probs = torch.softmax(output.logits, dim=-1)
        pred_ids = torch.argmax(output.logits, dim=-1).tolist()

        for local_idx, words in enumerate(batch_words):
            if not words:
                continue
            word_ids = encoded.word_ids(batch_index=local_idx)
            seen_word_index = set()
            row_results: list[TokenLabel] = []
            for token_idx, word_idx in enumerate(word_ids):
                if word_idx is None or word_idx in seen_word_index:
                    continue
                seen_word_index.add(word_idx)
                pred_id = pred_ids[local_idx][token_idx]
                label = id2label.get(pred_id, str(pred_id))
                confidence = float(probs[local_idx, token_idx, pred_id].item())
                row_results.append(TokenLabel(token=words[word_idx], label=label, score=confidence))
            all_results[start + local_idx] = row_results

    return all_results


def _print_results(text: str, predictions: list[TokenLabel]) -> None:
    print(f"Input: {text}")
    if not predictions:
        print("Tidak ada token yang bisa diprediksi.")
        return
    for item in predictions:
        print(f"{item.token}\t{item.label}\t{item.score:.4f}")


def _to_json_record(text: str, predictions: list[TokenLabel]) -> str:
    payload = {
        "text": text,
        "tokens": [
            {"token": item.token, "label": item.label, "score": round(item.score, 6)}
            for item in predictions
        ],
    }
    return json.dumps(payload, ensure_ascii=False)
```

## Cell 5 (code)

```python
import os
from google.colab import drive

# 1. Mount Google Drive jika belum
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

# 2. Definisikan Ulang Fungsi dengan Perbaikan Path Absolut
def load_layer1_model(model_dir: Path):
    _validate_model_dir(model_dir)
    torch, AutoTokenizer, AutoModelForTokenClassification = _import_runtime_deps()

    # KUNCI PERBAIKAN: Paksa path menjadi string absolut murni
    model_path_str = os.path.abspath(str(model_dir))

    # Masukkan string path absolut dan paksa pencarian berkas lokal
    tokenizer = AutoTokenizer.from_pretrained(model_path_str, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(model_path_str, local_files_only=True)

    if not torch.cuda.is_available():
        raise RuntimeError("GPU CUDA tidak terdeteksi. Pastikan Colab runtime memakai GPU.")
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    return tokenizer, model, torch, device

# 3. Pengaturan Path dan Konfigurasi Parameter Global (KUNCI UTAMA YANG HILANG)
MODEL_DIR = Path("/content/drive/MyDrive/Models/indobert-pantun-ner")
LAYER3_CRF_PATH = Path("/content/drive/MyDrive/Models/layer3_syllabifier.crfsuite")  # Sesuaikan dengan lokasi penyimpanan sukses Anda sebelumnya
MAX_LENGTH = 64
BATCH_SIZE = 32

# 4. Inisialisasi Eksekusi Pemuatan Model ke Memori Global
tokenizer, model, torch, device = load_layer1_model(MODEL_DIR)
layer3_tagger = load_layer3_crf_model(LAYER3_CRF_PATH)

print("\n[SUKSES] Model IndoBERT siap dipakai dari:", MODEL_DIR)
print("[SUKSES] Layer 3 CRF siap dipakai dari:", LAYER3_CRF_PATH)
print("Device Aktif:", device)
```

## Cell 6 (code)

```python
text = "pergi ke pasar nyari kemeja\nmalah dapet kemeja yang longgar\njgn sedih diputusin dia\nmending skrng kita chill ntar"

baits = _split_pantun_baits(text)
print(f"Total bait: {len(baits)}")
for idx, bait in enumerate(baits, start=1):
    print(f"{idx}. {bait}")
```

## Cell 7 (code)

```python
# Layer 1 + Layer 2a/2b + Layer 3 (CRF) per-bait.
all_results = []

for idx, bait in enumerate(baits, start=1):
    preds = predict_labels(
        bait,
        tokenizer=tokenizer,
        model=model,
        torch=torch,
        max_length=MAX_LENGTH,
    )
    layer2_rows = run_layer2_from_predictions(preds)
    layer3_rows = run_layer3_from_layer2_rows(layer2_rows, layer3_tagger)
    all_results.append({
        "bait_index": idx,
        "text": bait,
        "predictions": preds,
        "layer2": layer2_rows,
        "layer3": layer3_rows,
    })

    print(f"\n=== Bait {idx} ===")
    _print_results(bait, preds)
    for row in layer3_rows:
        # KUNCI PERBAIKAN 1: Tampilkan representasi potongan suku kata visual dengan pemisah tanda hubung
        syllables_visual = "-".join(row["layer3_syllables"])

        # KUNCI PERBAIKAN 2: Pastikan nilai hitungan silabel memanggil actual count dari total huruf 'B' hasil CRF
        actual_count = row["layer3_syllable_count"]
        phoneme_count = row.get("layer3_syllable_count_phoneme", actual_count)
        grapheme_count = row.get("layer3_syllable_count_grapheme", actual_count)

        print(
            f"{row['token']}\t{row['label']}\t{row['phoneme']}\t"
            f"{actual_count}\t{syllables_visual}\t"
            f"(fonem={phoneme_count}, grafem={grapheme_count})"
        )

# Hitung ringkasan komparatif akhir baris pantun
line_summary = summarize_pantun_lines(all_results)
print("\n=== Ringkasan Tiap Baris Pantun ===")
for item in line_summary:
    print(
        f"Baris {item['bait_index']}: silabel={item['syllable_count']}, "
        f"rima_akhir={item['rhyme_key']}, suku_akhir={item['last_syllable']}"
    )

if len(line_summary) >= 4:
    r1 = line_summary[0]["rhyme_key"]
    r2 = line_summary[1]["rhyme_key"]
    r3 = line_summary[2]["rhyme_key"]
    r4 = line_summary[3]["rhyme_key"]
    abab_valid = bool(r1 and r2 and r1 == r3 and r2 == r4)
    print(f"\nPola rima 1-2-3-4: {r1}-{r2}-{r3}-{r4}")
    print("Valid ABAB:", abab_valid)
```

## Cell 8 (code)

```python
# # Inferensi batch dari file (1 baris = 1 input), lalu simpan JSONL.
# INPUT_FILE = Path("/content/drive/MyDrive/Models/pantun_gaul_for_llm_batch_01.txt")
# OUTPUT_JSONL = Path("/content/drive/MyDrive/Models/layer1_predictions_batch01.jsonl")

# lines = [
#     line.strip()
#     for line in INPUT_FILE.read_text(encoding="utf-8").splitlines()
#     if line.strip()
# ]

# batched_predictions = predict_labels_batch(
#     lines,
#     tokenizer=tokenizer,
#     model=model,
#     torch=torch,
#     max_length=MAX_LENGTH,
#     batch_size=max(1, BATCH_SIZE),
# )

# jsonl_rows = []
# for idx, (line, preds) in enumerate(zip(lines, batched_predictions), start=1):
#     print(f"\n=== Baris {idx} ===")
#     _print_results(line, preds)
#     jsonl_rows.append(_to_json_record(line, preds))

# OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
# OUTPUT_JSONL.write_text("\n".join(jsonl_rows) + "\n", encoding="utf-8")
# print(f"\nHasil JSONL tersimpan di: {OUTPUT_JSONL}")
```

## Cell 9 (code)

```python
# # Layer 2a/2b untuk hasil batch; output JSONL berisi token, label, score, fonem.
# OUTPUT_LAYER2_JSONL = Path("/content/drive/MyDrive/Models/layer2a2b_phonemes_batch01.jsonl")

# layer2_batch = run_layer2_batch(batched_predictions)
# layer2_jsonl_rows = []
# for line, layer2_rows in zip(lines, layer2_batch):
#     layer2_jsonl_rows.append(json.dumps({
#         "text": line,
#         "tokens": layer2_rows,
#     }, ensure_ascii=False))

# OUTPUT_LAYER2_JSONL.parent.mkdir(parents=True, exist_ok=True)
# OUTPUT_LAYER2_JSONL.write_text("\n".join(layer2_jsonl_rows) + "\n", encoding="utf-8")
# print("Hasil Layer 2a/2b tersimpan di:", OUTPUT_LAYER2_JSONL)
```

