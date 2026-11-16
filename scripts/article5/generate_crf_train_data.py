#!/usr/bin/env python3
"""Generate synthetic CRF BI training rows and optionally merge with base CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

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

DEFAULT_BASE_CSV = Path("data/article5/manual_bi_random_sample.csv")
DEFAULT_SYNTH_OUT = Path("data/article5/synthetic_hiatus_train.csv")
DEFAULT_MERGED_OUT = Path("data/article5/manual_bi_random_sample_augmented.csv")


def _coerce_g2p_output_to_str(raw) -> str:
    if isinstance(raw, tuple):
        return _coerce_g2p_output_to_str(raw[0])
    if isinstance(raw, list):
        if raw and isinstance(raw[0], list):
            raw = [ph for sub in raw for ph in sub if ph]
        return " ".join(str(part) for part in raw if part)
    return str(raw).strip()


def run_g2p_pipeline(token: str) -> str:
    try:
        from g2p_id import G2P as G2P_ID
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            "g2p-id belum terpasang. Install dulu: pip install g2p-id"
        ) from err

    g2p_model = G2P_ID()
    try:
        result = g2p_model.to_phoneme(token)
    except AttributeError:
        try:
            result = g2p_model(token)
        except Exception:
            result = g2p_model.convert(token)

    return _coerce_g2p_output_to_str(result).replace(" ", "")


def tokenize_phonemes(phoneme_str: str) -> list[str]:
    cleaned = "".join(ch for ch in str(phoneme_str) if not ch.isspace() and ch != "-")
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


def synthetic_seed_rows() -> list[tuple[str, str, str]]:
    return [
        ("menemui", "BIBIBIB", "baru mengembalikan buku disuruh menemui ibumu"),
        ("melalui", "BIBIBIB", "jalan berliku harus dilalui dengan sabar"),
        ("mengetahui", "BIBIBIBIB", "tidak ada yang mengetahui rahasia hati"),
        ("mengakui", "BIBIBIB", "berani berbuat harus berani mengakui kesalahan"),
        ("memenuhi", "BIBIBIIB", "janji manis harus segera dipenuhi"),
        ("mendahului", "BIIBIBIBIB", "jangan suka mendahului kehendak tuhan"),
        ("menyetujui", "BIBIBIBIIB", "ayah dan ibu sudah menyetujui lamaranmu"),
        ("merestui", "BIBIIBIB", "semoga orang tua merestui hubungan kita"),
        ("dihargai", "BIBIIBIB", "tutuplah aurat agar engkau dihargai"),
        ("menghargai", "BIIBIIBIB", "sesama teman harus saling menghargai"),
        ("menyukai", "BIBIBIB", "banyak orang menyukai sifatmu yang ramah"),
        ("melukai", "BIBIBIB", "berteman jangan sampai saling melukai"),
        ("menandai", "BIBIIBIB", "senyum manis menandai awal bahagia"),
        ("memaknai", "BIBIIBIB", "kita harus bijak memaknai setiap cobaan"),
        ("menguasai", "BIBIBBIB", "ilmu agama harus kita kuasai"),
        ("mencintaimu", "BIIBIIIBIBBI", "untung jika mencintaimu itu salah aku rela di hukum gantung"),
        ("mencintaimu", "BIIBIIIBIBBI", "hiasan bunga sudah tersemat dengan peniti di dekat dada mencintaimu adalah rahmat"),
        ("cintai", "BIIIBIB", "wahai adikadik yang kami cintai selamat datang di kampus ini"),
        ("dicintai", "BIBIIIBIB", "setiap insan ingin dicintai dengan tulus dan suci"),
        ("cintai", "BIIIBIB", "di timur tengah banyak gurun kabah ada di arab saudi suami yang aku cintai"),
        ("cintai", "BIIIBIB", "jika hati tulus mencintai maka cintai dia karena ilahi"),
        ("hargai", "BIBIBI", "hidup rukun saling hargai agar damai sepanjang hari"),
        ("menghargai", "BIIBIIBIB", "kita belajar menghargai perbedaan agar persatuan abadi"),
    ]


def generate_synthetic_rows(start_record_id: int = 9001) -> list[dict[str, object]]:
    rows = []
    for idx, (token, gold_bi, ctx) in enumerate(synthetic_seed_rows(), start=start_record_id):
        phoneme_str = run_g2p_pipeline(token)
        unit_list = tokenize_phonemes(phoneme_str)
        if len(unit_list) != len(gold_bi):
            raise ValueError(
                f"Mismatch panjang pada '{token}': {len(unit_list)} vs {len(gold_bi)} "
                f"(units={unit_list}, gold={gold_bi})"
            )

        rows.append(
            {
                "task_id": f"r{idx}_t001",
                "record_id": idx,
                "token_id": 1,
                "token": token,
                "ner_tag": "O",
                "layer2_phoneme": phoneme_str,
                "units": " ".join(unit_list),
                "unit_count": len(unit_list),
                "silver_bi_auto": "",
                "manual_bi_annotator1": "",
                "manual_bi_annotator2": "",
                "manual_bi_final": gold_bi,
                "qa_status": "verified",
                "annotator_notes": "synthetic_hiatus_fix",
                "context_text": ctx,
            }
        )
    return rows


def _read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise ValueError(f"CSV tanpa header: {path}")
        rows = [dict(r) for r in reader]
    return rows, list(fieldnames)


def _write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def merge_with_base(
    base_csv: Path, synthetic_rows: list[dict[str, object]], merged_out: Path
) -> tuple[int, int, int]:
    if not base_csv.exists():
        raise FileNotFoundError(f"Base CSV tidak ditemukan: {base_csv}")

    base_rows, base_cols = _read_csv_rows(base_csv)
    missing = {"record_id", "token_id", "units", "manual_bi_final"}.difference(base_cols)
    if missing:
        raise ValueError(f"Base CSV missing kolom wajib: {sorted(missing)}")

    aligned_synth_rows = []
    for row in synthetic_rows:
        aligned_synth_rows.append({col: row.get(col, "") for col in base_cols})

    merged_rows = base_rows + aligned_synth_rows
    before = len(merged_rows)
    dedup_cols = ["record_id", "token_id", "token", "manual_bi_final", "context_text"]
    dedup_cols = [c for c in dedup_cols if c in base_cols]

    seen = set()
    deduped_rows: list[dict[str, object]] = []
    for row in merged_rows:
        key = tuple(str(row.get(c, "")) for c in dedup_cols) if dedup_cols else tuple(row.items())
        if key in seen:
            continue
        seen.add(key)
        deduped_rows.append(row)
    after = len(deduped_rows)
    removed = before - after

    _write_csv_rows(merged_out, base_cols, deduped_rows)
    return len(base_rows), len(aligned_synth_rows), removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic hiatus rows and optionally merge with manual CRF BI dataset."
    )
    parser.add_argument("--base-csv", type=Path, default=DEFAULT_BASE_CSV)
    parser.add_argument("--synthetic-out", type=Path, default=DEFAULT_SYNTH_OUT)
    parser.add_argument("--merged-out", type=Path, default=DEFAULT_MERGED_OUT)
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Jika aktif, gabungkan synthetic dengan base CSV ke merged-out.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    synthetic_rows = generate_synthetic_rows()
    synthetic_cols = [
        "task_id",
        "record_id",
        "token_id",
        "token",
        "ner_tag",
        "layer2_phoneme",
        "units",
        "unit_count",
        "silver_bi_auto",
        "manual_bi_annotator1",
        "manual_bi_annotator2",
        "manual_bi_final",
        "qa_status",
        "annotator_notes",
        "context_text",
    ]
    _write_csv_rows(args.synthetic_out, synthetic_cols, synthetic_rows)
    print(f"Synthetic rows tersimpan: {len(synthetic_rows)} -> {args.synthetic_out}")

    if not args.merge:
        print("Merge OFF. Aktifkan --merge untuk membuat dataset gabungan siap train.")
        return

    base_n, synth_n, removed = merge_with_base(
        base_csv=args.base_csv,
        synthetic_rows=synthetic_rows,
        merged_out=args.merged_out,
    )
    print(
        f"Merged dataset tersimpan: {args.merged_out} "
        f"(base={base_n}, synthetic={synth_n}, dedup_removed={removed})"
    )
    print(
        "Contoh train:\n"
        f"  ./.venv/bin/python scripts/article5/train_crf_layer3_bi.py --data {args.merged_out}"
    )


if __name__ == "__main__":
    main()
