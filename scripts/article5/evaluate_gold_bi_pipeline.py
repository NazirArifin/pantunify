from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from generate_crf_data import (
    auto_label_bi,
    build_token_normalizer,
    clean_and_tokenize_phoneme,
    compute_label_metrics,
    import_g2p_adapters,
    layer2_switchboard,
    layer3_features,
    load_records,
    resolve_input_path,
)


@dataclass
class GoldRow:
    task_id: str
    record_id: int
    token_id: int
    units: list[str]
    unit_count: int
    silver_bi_auto: list[str]
    annotator1: list[str]
    manual_bi_final: list[str]
    annotator1_is_blank: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync manual gold labels, evaluate silver/CRF vs gold, and train gold-aware CRF."
    )
    parser.add_argument(
        "--manual-csv",
        type=Path,
        default=Path("data/article5/manual_bi_annotation/manual_bi_random_sample.csv"),
        help="CSV anotasi manual BI.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/article5/pantun-anotated.jsonl"),
        help="Data anotasi token-level untuk membangun sequence train.",
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path("data/article5/layer3_syllabifier.crfsuite"),
        help="Model CRF baseline untuk evaluasi vs gold.",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("data/article5/layer3_syllabifier.goldaware.crfsuite"),
        help="Model CRF hasil retrain/fine-tune gold-aware.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("data/article5/manual_bi_annotation/manual_bi_random_sample.eval.json"),
        help="Output ringkasan evaluasi.",
    )
    parser.add_argument(
        "--output-locked-csv",
        type=Path,
        default=Path("data/article5/manual_bi_annotation/manual_bi_random_sample.gold_locked.csv"),
        help="Snapshot CSV gold yang sudah terkunci (manual_bi_final tersinkron).",
    )
    parser.add_argument(
        "--dsm-threshold",
        type=int,
        default=3,
        help="Threshold disemvowelling yang dipakai saat build sequence train.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=150,
        help="Maksimum iterasi training CRF.",
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


def normalize_label_string(value: str) -> str:
    return "".join(ch for ch in (value or "").upper() if not ch.isspace())


def parse_bi_sequence(value: str) -> list[str]:
    cleaned = normalize_label_string(value)
    return [ch for ch in cleaned]


def validate_bi(task_id: str, field_name: str, value: str, unit_count: int) -> list[str]:
    issues: list[str] = []
    cleaned = normalize_label_string(value)
    bad_chars = sorted({ch for ch in cleaned if ch not in {"B", "I"}})
    if bad_chars:
        issues.append(
            f"{task_id}: {field_name} mengandung char invalid {bad_chars} (value='{cleaned}')"
        )
    if len(cleaned) != unit_count:
        issues.append(
            f"{task_id}: panjang {field_name}={len(cleaned)} tidak sama unit_count={unit_count}"
        )
    return issues


def aggregate_metrics(y_true: list[str], y_pred: list[str], sample_count: int) -> dict[str, Any]:
    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    token_accuracy = correct / total if total else 0.0
    per_label = compute_label_metrics(y_true, y_pred, labels=["B", "I"])
    macro_precision = sum(m["precision"] for m in per_label.values()) / len(per_label)
    macro_recall = sum(m["recall"] for m in per_label.values()) / len(per_label)
    macro_f1 = sum(m["f1"] for m in per_label.values()) / len(per_label)
    return {
        "token_accuracy": token_accuracy,
        "per_label": per_label,
        "macro_avg": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
        },
        "sample_count": sample_count,
        "token_count": total,
    }


def eval_dataset(pred_by_task: dict[str, list[str]], gold_rows: list[GoldRow]) -> dict[str, Any]:
    y_true: list[str] = []
    y_pred: list[str] = []
    exact = 0
    missing_tasks: list[str] = []
    for row in gold_rows:
        pred = pred_by_task.get(row.task_id)
        if pred is None:
            missing_tasks.append(row.task_id)
            continue
        if pred == row.manual_bi_final:
            exact += 1
        y_true.extend(row.manual_bi_final)
        y_pred.extend(pred)

    base_metrics = aggregate_metrics(y_true, y_pred, sample_count=len(gold_rows) - len(missing_tasks))
    base_metrics["sequence_exact_match"] = (
        exact / (len(gold_rows) - len(missing_tasks)) if (len(gold_rows) - len(missing_tasks)) else 0.0
    )
    base_metrics["missing_task_ids"] = missing_tasks
    return base_metrics


def predict_with_model(model_path: Path, gold_rows: list[GoldRow]) -> dict[str, list[str]]:
    try:
        import pycrfsuite  # type: ignore
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            "pycrfsuite belum terpasang. Install: python -m pip install python-crfsuite"
        ) from err

    tagger = pycrfsuite.Tagger()
    tagger.open(str(model_path))

    predictions: dict[str, list[str]] = {}
    for row in gold_rows:
        feats = [layer3_features(row.units, i) for i in range(len(row.units))]
        pred = tagger.tag(feats)
        predictions[row.task_id] = pred
    return predictions


def train_goldaware_model(
    *,
    sequences_with_task: list[tuple[str, list[dict[str, object]], list[str]]],
    gold_map: dict[str, list[str]],
    output_model_path: Path,
    max_iterations: int,
) -> None:
    try:
        import pycrfsuite  # type: ignore
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            "pycrfsuite belum terpasang. Install: python -m pip install python-crfsuite"
        ) from err

    trainer = pycrfsuite.Trainer(verbose=False)
    for task_id, features, silver_tags in sequences_with_task:
        tags = gold_map.get(task_id, silver_tags)
        trainer.append(features, tags)
    trainer.set_params(
        {
            "c1": 0.1,
            "c2": 0.01,
            "max_iterations": max_iterations,
            "feature.possible_transitions": True,
        }
    )
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.train(str(output_model_path))


def main() -> None:
    args = parse_args()

    rows_raw: list[dict[str, str]] = []
    with args.manual_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows_raw = list(reader)

    issues: list[str] = []
    gold_rows: list[GoldRow] = []

    for row in rows_raw:
        task_id = (row.get("task_id") or "").strip()
        if not task_id:
            issues.append("Baris tanpa task_id.")
            continue

        unit_count = int((row.get("unit_count") or "0").strip())
        units = [u for u in (row.get("units") or "").split(" ") if u]
        if len(units) != unit_count:
            issues.append(
                f"{task_id}: jumlah units parsed={len(units)} tidak sama unit_count={unit_count}"
            )

        ann1_raw = row.get("manual_bi_annotator1", "")
        ann1_clean = normalize_label_string(ann1_raw)
        annotator1_is_blank = ann1_clean == ""
        silver_clean = normalize_label_string(row.get("silver_bi_auto", ""))

        chosen_final = ann1_clean if ann1_clean else silver_clean
        row["manual_bi_final"] = chosen_final

        issues.extend(validate_bi(task_id, "silver_bi_auto", silver_clean, unit_count))
        if ann1_clean:
            issues.extend(validate_bi(task_id, "manual_bi_annotator1", ann1_clean, unit_count))
        issues.extend(validate_bi(task_id, "manual_bi_final", chosen_final, unit_count))

        gold_rows.append(
            GoldRow(
                task_id=task_id,
                record_id=int((row.get("record_id") or "0").strip()),
                token_id=int((row.get("token_id") or "0").strip()),
                units=units,
                unit_count=unit_count,
                silver_bi_auto=parse_bi_sequence(silver_clean),
                annotator1=parse_bi_sequence(ann1_clean),
                manual_bi_final=parse_bi_sequence(chosen_final),
                annotator1_is_blank=annotator1_is_blank,
            )
        )

    if issues:
        raise ValueError("Validasi gagal:\n- " + "\n- ".join(issues[:50]))

    # overwrite synced final back to source CSV
    with args.manual_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_raw)

    # snapshot locked csv
    args.output_locked_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_locked_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_raw)

    audited_rows = [r for r in gold_rows if not r.annotator1_is_blank]
    all_rows = list(gold_rows)

    silver_pred_all = {r.task_id: r.silver_bi_auto for r in all_rows}
    silver_vs_gold_all = eval_dataset(silver_pred_all, all_rows)
    silver_vs_gold_audited = eval_dataset(silver_pred_all, audited_rows)

    baseline_pred_all = predict_with_model(args.base_model, all_rows)
    baseline_vs_gold_all = eval_dataset(baseline_pred_all, all_rows)
    baseline_vs_gold_audited = eval_dataset(baseline_pred_all, audited_rows)

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
    sequences_with_task: list[tuple[str, list[dict[str, object]], list[str]]] = []
    for record_idx, rec in enumerate(records, start=1):
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
            feats = [layer3_features(units, i) for i in range(len(units))]
            tags = auto_label_bi(units)
            task_id = f"r{record_idx:04d}_t{token_idx:03d}"
            sequences_with_task.append((task_id, feats, tags))

    gold_map = {r.task_id: r.manual_bi_final for r in all_rows}
    train_goldaware_model(
        sequences_with_task=sequences_with_task,
        gold_map=gold_map,
        output_model_path=args.output_model,
        max_iterations=args.max_iterations,
    )

    goldaware_pred_all = predict_with_model(args.output_model, all_rows)
    goldaware_vs_gold_all = eval_dataset(goldaware_pred_all, all_rows)
    goldaware_vs_gold_audited = eval_dataset(goldaware_pred_all, audited_rows)

    report = {
        "input": {
            "manual_csv": str(args.manual_csv),
            "locked_csv": str(args.output_locked_csv),
            "input_path": str(input_path),
            "record_count": len(records),
            "sequence_count": len(sequences_with_task),
        },
        "annotation_sync": {
            "total_rows": len(all_rows),
            "audited_rows_with_manual_annotator1": len(audited_rows),
            "rows_fallback_to_silver": len(all_rows) - len(audited_rows),
            "rule": "manual_bi_final = manual_bi_annotator1 jika terisi; jika kosong fallback silver_bi_auto",
        },
        "evaluation": {
            "silver_vs_gold_all_locked_rows": silver_vs_gold_all,
            "silver_vs_gold_audited_rows_only": silver_vs_gold_audited,
            "baseline_crf_vs_gold_all_locked_rows": baseline_vs_gold_all,
            "baseline_crf_vs_gold_audited_rows_only": baseline_vs_gold_audited,
            "goldaware_crf_vs_gold_all_locked_rows": goldaware_vs_gold_all,
            "goldaware_crf_vs_gold_audited_rows_only": goldaware_vs_gold_audited,
        },
        "models": {
            "base_model": str(args.base_model),
            "goldaware_model": str(args.output_model),
            "max_iterations": args.max_iterations,
            "dsm_threshold": args.dsm_threshold,
            "g2p_id_available": adapters.g2p_id is not None,
            "g2p_en_available": adapters.g2p_en is not None,
            "strict_dsm_label": args.strict_dsm_label,
            "enable_dsm_lookup": args.enable_dsm_lookup,
            "enable_typo_normalization": args.enable_typo_normalization,
            "normalization_lexicon_exists": args.normalization_lexicon.exists(),
            "normalization_overrides_exists": args.normalization_overrides.exists(),
            "normalizer_active": normalizer is not None,
        },
    }

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Locked CSV: {args.output_locked_csv}")
    print(f"[OK] Gold-aware model: {args.output_model}")
    print(f"[OK] Evaluation report: {args.output_report}")
    print(json.dumps(report["evaluation"]["goldaware_crf_vs_gold_audited_rows_only"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
