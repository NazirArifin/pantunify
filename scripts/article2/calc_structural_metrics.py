#!/usr/bin/env python3
import argparse
import csv
import math
import re
from pathlib import Path

from pantunify.utils import check_rhyme, count_syllables

# Stopword ringkas untuk estimasi content leakage.
STOPWORDS_ID = {
    "yang", "dan", "di", "ke", "dari", "untuk", "pada", "dengan", "atau", "juga",
    "ini", "itu", "adalah", "karena", "agar", "sebagai", "dalam", "oleh", "akan", "sudah",
    "belum", "tidak", "iya", "ya", "saja", "pun", "lah", "kah", "nya", "ku", "mu",
    "sebuah", "seorang", "para", "mereka", "kami", "kita", "aku", "saya", "engkau", "kau",
    "ia", "dia", "apa", "siapa", "mana", "kapan", "bagai", "bagai", "jadi", "lebih",
}


def _split_two_lines(text: str):
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) >= 2:
        return lines[0], lines[1], len(lines)
    if len(lines) == 1:
        return lines[0], "", 1
    return "", "", 0


def _deviation_8_12(syllables: int) -> int:
    if syllables < 8:
        return 8 - syllables
    if syllables > 12:
        return syllables - 12
    return 0


def _normalize_token(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^a-zA-Z]", "", t)
    return t


def _content_word_set(text: str):
    out = set()
    for raw in re.split(r"\s+", (text or "").strip()):
        tok = _normalize_token(raw)
        if not tok:
            continue
        if tok in STOPWORDS_ID:
            continue
        out.add(tok)
    return out


def _content_leakage_rate(sampiran: str, isi: str) -> float:
    samp_set = _content_word_set(sampiran)
    isi_set = _content_word_set(isi)
    if not isi_set:
        return 0.0
    overlap = samp_set.intersection(isi_set)
    return len(overlap) / len(isi_set)


def score_rows(input_path: Path, output_path: Path, lambda_penalty: float = 0.35) -> int:
    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            raise ValueError("Input kosong.")

    fieldnames = list(rows[0].keys()) + [
        "gen_line_count",
        "isi_line_count",
        "r_score",
        "a_score",
        "clr",
        "suku_baris_1",
        "suku_baris_2",
        "suku_baris_3",
        "suku_baris_4",
        "dev_total",
    ]

    out_rows = []
    for row in rows:
        sampiran = row.get("sampiran_generated", "")
        isi = row.get("isi_input_model", "")

        s1, s2, n_samp = _split_two_lines(sampiran)
        i1, i2, n_isi = _split_two_lines(isi)

        suku1 = count_syllables(s1)
        suku2 = count_syllables(s2)
        suku3 = count_syllables(i1)
        suku4 = count_syllables(i2)

        dev_total = (
            _deviation_8_12(suku1)
            + _deviation_8_12(suku2)
            + _deviation_8_12(suku3)
            + _deviation_8_12(suku4)
        )

        # A = 0.5 + 0.5 * exp(-lambda * sum dev)
        a_score = 0.5 + (0.5 * math.exp(-lambda_penalty * dev_total))

        if s1 and s2 and i1 and i2:
            r_score = 1 if check_rhyme(s1, s2, i1, i2) else 0
        else:
            r_score = 0

        clr = _content_leakage_rate(sampiran, isi)

        new_row = dict(row)
        new_row.update(
            {
                "gen_line_count": n_samp,
                "isi_line_count": n_isi,
                "r_score": r_score,
                "a_score": f"{a_score:.6f}",
                "clr": f"{clr:.6f}",
                "suku_baris_1": suku1,
                "suku_baris_2": suku2,
                "suku_baris_3": suku3,
                "suku_baris_4": suku4,
                "dev_total": dev_total,
            }
        )
        out_rows.append(new_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    return len(out_rows)


def main():
    parser = argparse.ArgumentParser(description="Hitung metrik struktur (R, A) dan CLR untuk hasil eksperimen pantun.")
    parser.add_argument(
        "--input",
        default="data/article2/100_Pantun_Eksperimen_long.csv",
        help="CSV long-format hasil reshape.",
    )
    parser.add_argument(
        "--output",
        default="data/article2/100_Pantun_Eksperimen_long_scored.csv",
        help="CSV keluaran berisi skor tambahan.",
    )
    parser.add_argument(
        "--lambda-penalty",
        type=float,
        default=0.35,
        help="Nilai lambda pada fungsi penalti A.",
    )

    args = parser.parse_args()
    total = score_rows(
        input_path=Path(args.input),
        output_path=Path(args.output),
        lambda_penalty=args.lambda_penalty,
    )
    print(f"Selesai. Total baris terskor: {total}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
