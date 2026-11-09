#!/usr/bin/env python3
"""Regenerate manual BI seed file (non-O priority) with updated layer-2 phoneme units.

This script rewrites `layer2_phoneme`, `units`, `unit_count`, and `silver_bi_auto`
while preserving other annotation columns.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from g2p_en import G2p as G2P_EN
from g2p_id import G2P as G2P_ID

DEFAULT_INPUT = Path("data/article5/manual_bi_non_o_priority.csv")
DEFAULT_OUTPUT = Path("data/article5/manual_bi_non_o_priority_v2.csv")

LABELS_ASING = {"B-ASG", "I-ASG"}
LABELS_SINGKATAN = {"B-SKT", "I-SKT"}
VOWELS = set("aiueo")

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

ARPABET_TO_ASCII = {
    "AA": "a",
    "AE": "e",
    "AH": "a",
    "AO": "o",
    "AW": "au",
    "AY": "ai",
    "B": "b",
    "CH": "c",
    "D": "d",
    "DH": "d",
    "EH": "e",
    "ER": "er",
    "EY": "ei",
    "F": "f",
    "G": "g",
    "HH": "h",
    "IH": "i",
    "IY": "i",
    "JH": "j",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ng",
    "OW": "ou",
    "OY": "oi",
    "P": "p",
    "R": "r",
    "S": "s",
    "SH": "sy",
    "T": "t",
    "TH": "t",
    "UH": "u",
    "UW": "u",
    "V": "v",
    "W": "w",
    "Y": "y",
    "Z": "z",
    "ZH": "z",
}


def normalize_token(token: str) -> str:
    t = token.lower().strip()
    t = "".join(ch for ch in t if ch.isalpha())
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    return t


def split_units(text: str) -> list[str]:
    plain = "".join(ch for ch in str(text).strip() if not ch.isspace())
    return [ch for ch in plain if ch]


def build_silver_bi(units: list[str]) -> str:
    if not units:
        return ""
    labels: list[str] = []
    prev_vowel = False
    for idx, u in enumerate(units):
        cur_vowel = u in VOWELS
        if idx == 0:
            labels.append("B")
        elif cur_vowel and not prev_vowel:
            labels.append("B")
        else:
            labels.append("I")
        prev_vowel = cur_vowel
    return "".join(labels)


def singkatan_to_ejaan(word: str) -> str:
    clean = "".join(ch for ch in word.lower() if ch.isalpha())
    return " ".join(ALPHABET_EJA_MAP.get(ch, ch) for ch in clean)


def coerce_g2p_output_to_str(raw) -> str:
    if isinstance(raw, tuple):
        return coerce_g2p_output_to_str(raw[0])
    if isinstance(raw, list):
        if raw and isinstance(raw[0], list):
            raw = [ph for sub in raw for ph in sub if ph]
        return "".join(str(part) for part in raw if part and str(part) != " ")
    return str(raw).strip().replace(" ", "")


def arpabet_to_ascii(arpabet_tokens: list[str]) -> str:
    parts: list[str] = []
    for tok in arpabet_tokens:
        base = re.sub(r"\d", "", tok).upper().strip()
        if not base:
            continue
        parts.append(ARPABET_TO_ASCII.get(base, base.lower()))
    return "".join(parts)


def regenerate_row_phoneme(token: str, label: str, g2p_id_model: G2P_ID, g2p_en_model: G2P_EN) -> str:
    norm_label = (label or "").upper().strip()
    norm_token = normalize_token(token)
    if not norm_token:
        return ""

    if norm_label in LABELS_ASING:
        raw = g2p_en_model(norm_token)
        phones = [p for p in raw if p and p != " " and p != "<unk>"]
        phon = arpabet_to_ascii(phones)
        return phon or norm_token

    if norm_label in LABELS_SINGKATAN:
        ejaan = singkatan_to_ejaan(norm_token)
        try:
            res = g2p_id_model.to_phoneme(ejaan)
        except AttributeError:
            try:
                res = g2p_id_model(ejaan)
            except Exception:
                res = g2p_id_model.convert(ejaan)
        phon = coerce_g2p_output_to_str(res).lower()
        return phon or norm_token

    return norm_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate non-O BI annotation seed file with refreshed layer2 phoneme/units."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"Input CSV (default: {DEFAULT_INPUT})")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    g2p_id_model = G2P_ID()
    g2p_en_model = G2P_EN()

    with args.input.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    required = {"token", "ner_tag", "layer2_phoneme", "units", "unit_count", "silver_bi_auto"}
    missing = required.difference(fieldnames)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for row in rows:
        phon = regenerate_row_phoneme(
            token=row.get("token", ""),
            label=row.get("ner_tag", ""),
            g2p_id_model=g2p_id_model,
            g2p_en_model=g2p_en_model,
        )
        units = split_units(phon)
        row["layer2_phoneme"] = phon
        row["units"] = " ".join(units)
        row["unit_count"] = str(len(units))
        row["silver_bi_auto"] = build_silver_bi(units)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved regenerated file: {args.output}")
    print(f"Total rows: {len(rows)}")


if __name__ == "__main__":
    main()
