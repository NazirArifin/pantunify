from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

from generate_crf_data import (
    build_token_normalizer,
    clean_and_tokenize_phoneme,
    import_g2p_adapters,
    layer2_switchboard,
    load_records,
    resolve_input_path,
    VOWEL_UNITS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate manual BI annotation materials for Layer-3 syllabifier."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/article5/pantun-anotated.jsonl"),
        help="Path data anotasi token-level.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/article5/manual_bi_annotation"),
        help="Folder output material anotasi manual.",
    )
    parser.add_argument(
        "--dsm-threshold",
        type=int,
        default=3,
        help="Threshold disemvowelling yang dipakai Layer-2 switchboard.",
    )
    parser.add_argument(
        "--record-sample-ratio",
        type=float,
        default=0.15,
        help="Rasio record pantun untuk sampel acak utama.",
    )
    parser.add_argument(
        "--min-records",
        type=int,
        default=120,
        help="Batas minimum jumlah record sampel acak.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed pemilihan sampel acak.",
    )
    parser.add_argument(
        "--prefill-manual-final-with-silver",
        action="store_true",
        help="Isi kolom manual_bi_final dengan silver labels agar langsung bisa diaudit.",
    )
    parser.add_argument(
        "--enable-dsm-lookup",
        action="store_true",
        help="Aktifkan lookup disemvowelling ke kamus sebelum fallback Rumus B/C.",
    )
    parser.add_argument(
        "--enable-typo-normalization",
        action="store_true",
        help="Aktifkan koreksi typo berbasis kamus + Levenshtein untuk label B/I-TYP.",
    )
    parser.add_argument(
        "--strict-dsm-label",
        action="store_true",
        help="Hanya label B/I-DSM yang boleh masuk jalur Rumus B/C (tanpa auto-coercion dari O/TYP).",
    )
    parser.add_argument(
        "--normalization-lexicon",
        type=Path,
        default=Path("data/kata-dasar.txt"),
        help="Path kamus kata dasar untuk lookup DSM/typo.",
    )
    parser.add_argument(
        "--normalization-overrides",
        type=Path,
        default=Path("data/article5/token_normalization_overrides.tsv"),
        help="Path override token->kata_baku (TSV: token<TAB>normalisasi).",
    )
    parser.add_argument(
        "--typo-max-edit-distance",
        type=int,
        default=1,
        help="Maksimum edit distance untuk koreksi typo.",
    )
    parser.add_argument(
        "--dsm-max-skeleton-distance",
        type=int,
        default=1,
        help="Maksimum edit distance pada skeleton disemvowelling.",
    )
    return parser.parse_args()


def build_row(
    *,
    record_id: int,
    token_id: int,
    token: str,
    ner_tag: str,
    layer2_phoneme: str,
    units: list[str],
    context_tokens: list[str],
    prefill_manual_final_with_silver: bool,
) -> dict[str, Any]:
    silver_bi = "".join(auto_label_bi_for_audit(units))
    return {
        "task_id": f"r{record_id:04d}_t{token_id:03d}",
        "record_id": record_id,
        "token_id": token_id,
        "token": token,
        "ner_tag": ner_tag,
        "layer2_phoneme": layer2_phoneme,
        "units": " ".join(units),
        "unit_count": len(units),
        "silver_bi_auto": silver_bi,
        "manual_bi_annotator1": "",
        "manual_bi_annotator2": "",
        "manual_bi_final": silver_bi if prefill_manual_final_with_silver else "",
        "qa_status": "",
        "annotator_notes": "",
        "context_text": " ".join(context_tokens),
    }


def auto_label_bi_for_audit(units: list[str]) -> list[str]:
    tags: list[str] = []
    for idx, unit in enumerate(units):
        is_vowel = unit in VOWEL_UNITS
        prev_is_vowel = units[idx - 1] in VOWEL_UNITS if idx > 0 else False
        next_is_vowel = units[idx + 1] in VOWEL_UNITS if idx + 1 < len(units) else False

        start_new = False
        if idx == 0:
            start_new = True
        elif is_vowel and prev_is_vowel:
            start_new = True
        elif (not is_vowel) and prev_is_vowel and next_is_vowel:
            start_new = True
        elif (
            idx >= 2
            and (not is_vowel)
            and (not prev_is_vowel)
            and (units[idx - 2] in VOWEL_UNITS)
            and next_is_vowel
        ):
            start_new = True

        tags.append("B" if start_new else "I")
    return tags


def export_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.record_sample_ratio <= 1.0:
        raise ValueError("record-sample-ratio harus di rentang (0, 1].")
    if args.min_records < 1:
        raise ValueError("min-records harus >= 1.")

    input_path = resolve_input_path(args.input)
    records = load_records(input_path)
    adapters = import_g2p_adapters()
    use_normalizer = args.enable_dsm_lookup or args.enable_typo_normalization
    normalizer = (
        build_token_normalizer(
            records=records,
            lexicon_path=args.normalization_lexicon,
            overrides_path=args.normalization_overrides,
            typo_max_distance=args.typo_max_edit_distance,
            dsm_max_skeleton_distance=args.dsm_max_skeleton_distance,
        )
        if use_normalizer
        else None
    )

    all_rows: list[dict[str, Any]] = []
    non_o_rows: list[dict[str, Any]] = []
    token_rows_by_record: dict[int, list[dict[str, Any]]] = {}

    for record_idx, rec in enumerate(records, start=1):
        current_rows: list[dict[str, Any]] = []
        for token_idx, (token, label) in enumerate(zip(rec.tokens, rec.ner_tags), start=1):
            phoneme = layer2_switchboard(
                token=token,
                label=label,
                adapters=adapters,
                dsm_threshold=args.dsm_threshold,
                normalizer=normalizer if use_normalizer else None,
                strict_dsm_label=args.strict_dsm_label,
            )
            units = clean_and_tokenize_phoneme(phoneme)
            if not units:
                continue
            row = build_row(
                record_id=record_idx,
                token_id=token_idx,
                token=token,
                ner_tag=label,
                layer2_phoneme=phoneme,
                units=units,
                context_tokens=rec.tokens,
                prefill_manual_final_with_silver=args.prefill_manual_final_with_silver,
            )
            all_rows.append(row)
            current_rows.append(row)
            if label != "O":
                non_o_rows.append(row)
        token_rows_by_record[record_idx] = current_rows

    total_records = len(records)
    target_records = max(args.min_records, int(round(total_records * args.record_sample_ratio)))
    target_records = min(target_records, total_records)

    rng = random.Random(args.seed)
    all_record_ids = list(range(1, total_records + 1))
    rng.shuffle(all_record_ids)
    sampled_record_ids = sorted(all_record_ids[:target_records])

    sampled_rows: list[dict[str, Any]] = []
    for record_id in sampled_record_ids:
        sampled_rows.extend(token_rows_by_record[record_id])

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    random_sample_path = output_dir / "manual_bi_random_sample.csv"
    priority_non_o_path = output_dir / "manual_bi_non_o_priority.csv"
    summary_path = output_dir / "manual_bi_summary.json"
    guide_path = output_dir / "manual_bi_annotation_guide.txt"

    export_csv(random_sample_path, sampled_rows)
    export_csv(priority_non_o_path, non_o_rows)

    summary = {
        "input_path": str(args.input),
        "resolved_input_path": str(input_path),
        "record_count": total_records,
        "sequence_count_all": len(all_rows),
        "sequence_count_non_o": len(non_o_rows),
        "sampled_record_count": len(sampled_record_ids),
        "sampled_sequence_count": len(sampled_rows),
        "sampled_record_ids": sampled_record_ids,
        "dsm_threshold": args.dsm_threshold,
        "seed": args.seed,
        "g2p_id_available": adapters.g2p_id is not None,
        "g2p_en_available": adapters.g2p_en is not None,
        "prefill_manual_final_with_silver": args.prefill_manual_final_with_silver,
        "strict_dsm_label": args.strict_dsm_label,
        "enable_dsm_lookup": args.enable_dsm_lookup,
        "enable_typo_normalization": args.enable_typo_normalization,
        "normalization_lexicon_exists": args.normalization_lexicon.exists(),
        "normalization_overrides_exists": args.normalization_overrides.exists(),
        "normalizer_active": normalizer is not None,
        "files": {
            "random_sample": str(random_sample_path),
            "priority_non_o": str(priority_non_o_path),
            "guide": str(guide_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    guide_text = (
        "PANDUAN SINGKAT ANOTASI MANUAL BI (LAYER-3)\n"
        "=============================================\n\n"
        "Tujuan\n"
        "- Mengisi label suku kata manual (B/I) per unit fonem untuk evaluasi gold standard.\n\n"
        "File yang diisi\n"
        "- manual_bi_random_sample.csv: sampel utama untuk evaluasi representatif.\n"
        "- manual_bi_non_o_priority.csv: subset fokus token non-O (kasus sulit/OOV).\n\n"
        "Arti kolom utama\n"
        "- units: urutan unit fonem yang harus Anda labeli.\n"
        "- silver_bi_auto: baseline label B/I otomatis dari pipeline.\n"
        "- manual_bi_annotator1 / manual_bi_annotator2: isi dengan urutan label TANPA spasi (mis. BIBIBII).\n"
        "- manual_bi_final: versi final setelah adjudikasi.\n"
        "- qa_status: opsional (mis. done, need_review, conflict).\n\n"
        "Aturan pengisian label\n"
        "1) Jumlah label harus sama persis dengan jumlah unit pada kolom units.\n"
        "2) Label hanya boleh B atau I.\n"
        "3) Token pertama suku kata harus B.\n"
        "4) Unit lanjutan dalam suku kata yang sama diberi I.\n\n"
        "Contoh\n"
        "- units            : b e r a ng k a t\n"
        "- manual_bi_final  : BIIBIBII\n\n"
        "Checklist QA sebelum lock\n"
        "- Tidak ada jumlah label yang mismatch dengan unit_count.\n"
        "- Tidak ada label selain B/I.\n"
        "- Baris konflik antaranotator sudah diputuskan di manual_bi_final.\n"
    )
    guide_path.write_text(guide_text, encoding="utf-8")

    print(f"[OK] random sample: {random_sample_path}")
    print(f"[OK] non-O priority: {priority_non_o_path}")
    print(f"[OK] summary: {summary_path}")
    print(f"[OK] guide: {guide_path}")


if __name__ == "__main__":
    main()
