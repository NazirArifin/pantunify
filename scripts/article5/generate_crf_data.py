from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VOWEL_UNITS = {
    "a",
    "i",
    "u",
    "e",
    "o",
    "ə",
    "ɛ",
    "ɔ",
    "ɪ",
    "ʊ",
    "ɑ",
    "æ",
    "ʌ",
    "ɝ",
    "ɚ",
    "eɪ",
    "aɪ",
    "aʊ",
    "ɔɪ",
    "oʊ",
}

LABELS_ASING = {"B-ASG", "I-ASG"}
LABELS_SINGKATAN = {"B-SKT", "I-SKT"}
LABELS_DISEMVOWEL = {"B-DSM", "I-DSM"}
LABELS_TYPO = {"B-TYP", "I-TYP"}

ALPHABET_EJA_MAP = {
    "a": "a",
    "b": "be",
    "c": "ce",
    "d": "de",
    "e": "e",
    "f": "ef",
    "g": "ge",
    "h": "ha",
    "i": "i",
    "j": "je",
    "k": "ka",
    "l": "el",
    "m": "em",
    "n": "en",
    "o": "o",
    "p": "pe",
    "q": "ki",
    "r": "er",
    "s": "es",
    "t": "te",
    "u": "u",
    "v": "ve",
    "w": "we",
    "x": "eks",
    "y": "ye",
    "z": "zet",
}

ARPABET_TO_IPA = {
    "AA": "ɑ",
    "AE": "æ",
    "AH": "ʌ",
    "AO": "ɔ",
    "AW": "aʊ",
    "AY": "aɪ",
    "B": "b",
    "CH": "tʃ",
    "D": "d",
    "DH": "ð",
    "EH": "ɛ",
    "ER": "ɝ",
    "EY": "eɪ",
    "F": "f",
    "G": "ɡ",
    "HH": "h",
    "IH": "ɪ",
    "IY": "i",
    "JH": "dʒ",
    "K": "k",
    "L": "l",
    "M": "m",
    "NG": "ŋ",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "P": "p",
    "R": "r",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "UH": "ʊ",
    "UW": "u",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ",
}


@dataclass
class Record:
    tokens: list[str]
    ner_tags: list[str]


@dataclass
class G2PAdapters:
    g2p_id: Any | None
    g2p_en: Any | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train & evaluate Layer-3 CRF syllabifier from pantun annotations."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/article5/pantun-anotated.jsonl"),
        help="Path ke file anotasi. Format didukung: JSONL standar atau multi-JSON bertumpuk.",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("data/article5/layer3_syllabifier.crfsuite"),
        help="Path output model CRFSuite.",
    )
    parser.add_argument(
        "--output-metrics",
        type=Path,
        default=Path("data/article5/layer3_eval_metrics.json"),
        help="Path output metrik evaluasi (JSON).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Proporsi data test untuk evaluasi.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed untuk split train/test.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=150,
        help="Maksimum iterasi training CRF.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Hanya cek parsing + ekstraksi fitur (tanpa train model).",
    )
    return parser.parse_args()


def resolve_input_path(path: Path) -> Path:
    if path.exists():
        return path
    alt_name = path.name.replace("_", "-") if "_" in path.name else path.name.replace("-", "_")
    alt_path = path.with_name(alt_name)
    if alt_path.exists():
        return alt_path
    raise FileNotFoundError(f"File input tidak ditemukan: {path}")


def load_records(path: Path) -> list[Record]:
    resolved = resolve_input_path(path)
    raw = resolved.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    decoder = json.JSONDecoder()
    idx = 0
    records: list[Record] = []
    while idx < len(raw):
        while idx < len(raw) and raw[idx].isspace():
            idx += 1
        if idx >= len(raw):
            break
        obj, end = decoder.raw_decode(raw, idx)
        idx = end

        tokens = obj.get("tokens")
        ner_tags = obj.get("ner_tags")
        if not isinstance(tokens, list) or not isinstance(ner_tags, list):
            raise ValueError("Setiap objek harus punya field list: tokens dan ner_tags.")
        if len(tokens) != len(ner_tags):
            raise ValueError("Panjang tokens dan ner_tags harus sama.")
        records.append(
            Record(tokens=[str(t) for t in tokens], ner_tags=[str(t) for t in ner_tags])
        )
    return records


def import_g2p_adapters() -> G2PAdapters:
    try:
        from g2p_id import G2P as G2PID  # type: ignore
    except ModuleNotFoundError:
        G2PID = None

    try:
        from g2p_en import G2p as G2PEN  # type: ignore
    except ModuleNotFoundError:
        G2PEN = None

    g2p_id = G2PID() if G2PID else None
    g2p_en = G2PEN() if G2PEN else None
    return G2PAdapters(g2p_id=g2p_id, g2p_en=g2p_en)


def _coerce_g2p_output_to_str(raw: Any) -> str:
    if isinstance(raw, tuple) and raw:
        return _coerce_g2p_output_to_str(raw[0])
    if isinstance(raw, list):
        if raw and isinstance(raw[0], list):
            flat = [part for sublist in raw for part in sublist if part]
            return " ".join(str(part) for part in flat)
        return " ".join(str(part) for part in raw if part)
    return str(raw).strip()


def g2p_id_phoneme(word: str, adapters: G2PAdapters) -> str:
    if adapters.g2p_id is None:
        return word.lower()

    model = adapters.g2p_id
    try:
        res = model.to_phoneme(word)
        return _coerce_g2p_output_to_str(res)
    except AttributeError:
        try:
            return _coerce_g2p_output_to_str(model(word))
        except Exception:
            return _coerce_g2p_output_to_str(model.convert(word))


def arpabet_to_ipa_sequence(tokens: list[str]) -> str:
    ipa_tokens: list[str] = []
    for token in tokens:
        base = re.sub(r"\d", "", token).upper()
        if not base:
            continue
        ipa_tokens.append(ARPABET_TO_IPA.get(base, base.lower()))
    return "".join(ipa_tokens)


def g2p_foreign_phoneme(word: str, adapters: G2PAdapters) -> str:
    if adapters.g2p_en is None:
        return word.lower()
    raw = adapters.g2p_en(word)
    phones = [p for p in raw if p and p != " " and p != "<unk>"]
    ipa = arpabet_to_ipa_sequence(phones)
    return ipa if ipa else g2p_id_phoneme(word, adapters)


def rumus_a_singkatan(word: str, adapters: G2PAdapters) -> str:
    ejaan = " ".join(ALPHABET_EJA_MAP.get(ch, ch) for ch in word.lower() if ch.isalpha())
    if not ejaan:
        return g2p_id_phoneme(word, adapters)
    return g2p_id_phoneme(ejaan, adapters)


def rumus_b_phonotactic_mapping(word: str) -> str:
    digraf_patterns = ["ng", "ny", "kh", "br", "tr", "bl", "cl", "kr", "pr"]
    processed_word = word
    for digraf in digraf_patterns:
        processed_word = processed_word.replace(digraf, "X")
    has_koda = bool(word) and word[-1] in "bcdfgjklmnpqrstvwxyz"
    base_count = len(processed_word)
    core_phonemes = base_count - 1 if has_koda and base_count > 1 else base_count
    if core_phonemes <= 2:
        return "vonem"
    if core_phonemes == 3:
        return "vonemis"
    return "vonemisasi"


def rumus_c_clamping(word: str) -> str:
    return "von" if len(word) <= 3 else "vonem"


def normalize_label(label: str) -> str:
    return label.upper().strip()


def layer2_switchboard(token: str, label: str, adapters: G2PAdapters) -> str:
    norm_label = normalize_label(label)
    alpha_only = "".join(ch for ch in token.lower() if ch.isalpha())
    has_vowel = any(v in alpha_only for v in "aiueo")

    if norm_label in {"O", "B-TYP", "I-TYP"} and not has_vowel and len(alpha_only) >= 2:
        norm_label = "B-DSM"

    if norm_label == "O":
        return g2p_id_phoneme(token, adapters).replace(" ", "")
    if norm_label in LABELS_ASING:
        return g2p_foreign_phoneme(token, adapters).replace(" ", "")
    if norm_label in LABELS_SINGKATAN:
        return rumus_a_singkatan(token, adapters).replace(" ", "")
    if norm_label in LABELS_DISEMVOWEL:
        if len(alpha_only) <= 3:
            return rumus_c_clamping(alpha_only)
        return rumus_b_phonotactic_mapping(alpha_only)
    if norm_label in LABELS_TYPO:
        return g2p_id_phoneme(token, adapters).replace(" ", "")
    return g2p_id_phoneme(token, adapters).replace(" ", "")


def clean_and_tokenize_phoneme(phoneme_str: str) -> list[str]:
    cleaned = "".join(ch for ch in phoneme_str if not ch.isspace() and ch != "-")
    units: list[str] = []
    i = 0
    while i < len(cleaned):
        if i + 1 < len(cleaned) and cleaned[i : i + 2] in VOWEL_UNITS:
            units.append(cleaned[i : i + 2])
            i += 2
            continue
        units.append(cleaned[i])
        i += 1
    return units


def auto_label_bi(units: list[str]) -> list[str]:
    tags: list[str] = []
    for idx, unit in enumerate(units):
        is_vowel = unit in VOWEL_UNITS
        prev_is_vowel = units[idx - 1] in VOWEL_UNITS if idx > 0 else False
        next_is_vowel = units[idx + 1] in VOWEL_UNITS if idx + 1 < len(units) else False

        start_new = False
        if idx == 0:
            start_new = True
        elif is_vowel and not prev_is_vowel:
            start_new = True
        elif not is_vowel and prev_is_vowel and next_is_vowel:
            start_new = True
        elif idx >= 2 and (not is_vowel) and (not prev_is_vowel) and units[idx - 2] in VOWEL_UNITS and next_is_vowel:
            start_new = True

        tags.append("B" if start_new else "I")
    return tags


def layer3_features(units: list[str], idx: int) -> dict[str, object]:
    cur = units[idx]
    prev = units[idx - 1] if idx > 0 else "<BOS>"
    nxt = units[idx + 1] if idx + 1 < len(units) else "<EOS>"
    return {
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


def build_sequences(records: list[Record], adapters: G2PAdapters) -> list[tuple[list[dict[str, object]], list[str]]]:
    sequences: list[tuple[list[dict[str, object]], list[str]]] = []
    for rec in records:
        for token, label in zip(rec.tokens, rec.ner_tags):
            phoneme = layer2_switchboard(token, label, adapters)
            units = clean_and_tokenize_phoneme(phoneme)
            if not units:
                continue
            tags = auto_label_bi(units)
            features = [layer3_features(units, i) for i in range(len(units))]
            sequences.append((features, tags))
    return sequences


def train_test_split_sequences(
    sequences: list[tuple[list[dict[str, object]], list[str]]],
    test_ratio: float,
    seed: int,
) -> tuple[list[tuple[list[dict[str, object]], list[str]]], list[tuple[list[dict[str, object]], list[str]]]]:
    if not sequences:
        return [], []
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio harus di antara 0 dan 1.")
    rng = random.Random(seed)
    shuffled = list(sequences)
    rng.shuffle(shuffled)
    test_size = max(1, int(len(shuffled) * test_ratio))
    test_set = shuffled[:test_size]
    train_set = shuffled[test_size:]
    if not train_set:
        raise ValueError("Data train kosong. Kurangi test_ratio atau tambah data.")
    return train_set, test_set


def compute_label_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(sum(1 for t in y_true if t == label)),
        }
    return metrics


def evaluate(
    tagger: Any,
    test_set: list[tuple[list[dict[str, object]], list[str]]],
) -> dict[str, Any]:
    if not test_set:
        return {"warning": "Test set kosong, evaluasi dilewati."}

    y_true_flat: list[str] = []
    y_pred_flat: list[str] = []
    exact_match = 0

    for features, true_tags in test_set:
        pred_tags = tagger.tag(features)
        if pred_tags == true_tags:
            exact_match += 1
        y_true_flat.extend(true_tags)
        y_pred_flat.extend(pred_tags)

    total = len(y_true_flat)
    correct = sum(1 for t, p in zip(y_true_flat, y_pred_flat) if t == p)
    accuracy = correct / total if total else 0.0
    per_label = compute_label_metrics(y_true_flat, y_pred_flat, labels=["B", "I"])
    macro_precision = sum(m["precision"] for m in per_label.values()) / len(per_label)
    macro_recall = sum(m["recall"] for m in per_label.values()) / len(per_label)
    macro_f1 = sum(m["f1"] for m in per_label.values()) / len(per_label)

    return {
        "token_accuracy": accuracy,
        "sequence_exact_match": exact_match / len(test_set),
        "per_label": per_label,
        "macro_avg": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
        },
        "sample_count": len(test_set),
        "token_count": total,
        "note": "Evaluasi ini menggunakan silver labels auto-BI dari fonem Layer-2, bukan anotasi BI manual.",
    }


def main() -> None:
    args = parse_args()
    input_path = resolve_input_path(args.input)
    records = load_records(input_path)
    if not records:
        raise ValueError("Tidak ada data yang bisa diproses dari file input.")

    adapters = import_g2p_adapters()
    using_true_ipa = adapters.g2p_id is not None
    sequences = build_sequences(records, adapters)
    if not sequences:
        raise ValueError("Tidak ada sequence valid yang terbentuk untuk training CRF.")

    train_set, test_set = train_test_split_sequences(sequences, args.test_ratio, args.seed)

    summary = {
        "input_path": str(args.input),
        "resolved_input_path": str(input_path),
        "record_count": len(records),
        "sequence_count": len(sequences),
        "train_count": len(train_set),
        "test_count": len(test_set),
        "g2p_id_available": adapters.g2p_id is not None,
        "g2p_en_available": adapters.g2p_en is not None,
        "ipa_mode": "g2p" if using_true_ipa else "fallback-grapheme",
    }

    if args.dry_run:
        print(json.dumps({"summary": summary, "dry_run": True}, ensure_ascii=False, indent=2))
        return

    try:
        import pycrfsuite  # type: ignore
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            "pycrfsuite belum terpasang. Install: python -m pip install python-crfsuite"
        ) from err

    trainer = pycrfsuite.Trainer(verbose=False)
    for features, tags in train_set:
        trainer.append(features, tags)
    trainer.set_params(
        {
            "c1": 0.1,
            "c2": 0.01,
            "max_iterations": args.max_iterations,
            "feature.possible_transitions": True,
        }
    )

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    trainer.train(str(args.output_model))

    tagger = pycrfsuite.Tagger()
    tagger.open(str(args.output_model))
    metrics = evaluate(tagger, test_set)
    result = {"summary": summary, "metrics": metrics}

    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Model CRF tersimpan: {args.output_model}")
    print(f"[OK] Metrik evaluasi tersimpan: {args.output_metrics}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
