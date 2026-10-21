#!/usr/bin/env python3
"""Stage-1 normalization for slang/typo tokens with fastText-centric hybrid ranking.

Method (minimal, publication-friendly):
1) fastText top-K retrieval over lexicon candidates.
2) Hybrid reranking using cosine + edit similarity (with length penalty).
3) Two linguistic rules:
   - consonant-abbreviation expansion (e.g. jgn -> jangan),
   - reduplication normalization (e.g. cobacoba -> coba-coba, benerbener -> benar-benar).

The script also supports:
- top-N diagnostics report for error analysis and threshold tuning,
- reproducibility manifest (JSON) with hashes, parameters, and environment metadata.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import platform
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Optional

import fasttext
import numpy as np

DEFAULT_LEXICON = Path("data/kata-dasar.txt")
DEFAULT_INPUT = Path("data/kata_tidak_di_kamus_pantun_gaul_slang_min3_fonem.csv")
DEFAULT_OUTPUT = Path("data/kata_tidak_di_kamus_pantun_gaul_slang_min3_fasttext.csv")
KATA_COL = "kata"
VOWELS = set("aiueo")


def normalize_word(word: str) -> str:
    return re.sub(r"[^a-z]", "", word.lower())


def consonant_skeleton(word: str) -> str:
    return "".join(ch for ch in word if ch not in VOWELS)


def is_abbreviation_like(word: str) -> bool:
    if not (2 <= len(word) <= 6):
        return False
    vowel_count = sum(1 for ch in word if ch in VOWELS)
    return vowel_count <= 1


def is_subsequence(short: str, long: str) -> bool:
    if len(short) > len(long):
        return False
    i = 0
    for ch in long:
        if i < len(short) and short[i] == ch:
            i += 1
            if i == len(short):
                return True
    return i == len(short)


def skeleton_subsequence_similarity(abbrev_skeleton: str, candidate_skeleton: str) -> float:
    if not abbrev_skeleton or not candidate_skeleton:
        return 0.0
    if is_subsequence(abbrev_skeleton, candidate_skeleton):
        return len(abbrev_skeleton) / len(candidate_skeleton)
    return 0.0


def char_edit_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_commit(cwd: Path) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() or None
    except Exception:
        return None


def package_version(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def read_lexicon(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Lexicon file not found: {path}")
    words: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            w = normalize_word(raw.strip())
            if not w or w in seen:
                continue
            seen.add(w)
            words.append(w)
    if not words:
        raise ValueError(f"Lexicon is empty after normalization: {path}")
    return words


def build_lexicon_by_length(lexicon_words: list[str]) -> dict[int, list[str]]:
    out: dict[int, list[str]] = defaultdict(list)
    for w in lexicon_words:
        out[len(w)].append(w)
    return dict(out)


def build_lexicon_first_char_index(lexicon_words: list[str]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for i, w in enumerate(lexicon_words):
        out[w[0]].append(i)
    return dict(out)


def build_lexicon_matrix(model: fasttext.FastText._FastText, words: list[str]) -> np.ndarray:
    matrix = np.stack([model.get_word_vector(w) for w in words]).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def cosine_scores(
    model: fasttext.FastText._FastText,
    lexicon_matrix: np.ndarray,
    query_word: str,
) -> np.ndarray:
    qvec = model.get_word_vector(query_word).astype(np.float32)
    qnorm = np.linalg.norm(qvec)
    if qnorm == 0.0:
        return np.zeros(lexicon_matrix.shape[0], dtype=np.float32)
    return lexicon_matrix @ (qvec / qnorm)


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if scores.size == 0 or k <= 0:
        return np.array([], dtype=np.int64)
    k = min(k, scores.size)
    if k == scores.size:
        return np.argsort(scores)[::-1]
    idx = np.argpartition(scores, -k)[-k:]
    return idx[np.argsort(scores[idx])[::-1]]


def best_lexicon_match_by_edit(
    token: str,
    lexicon_by_length: dict[int, list[str]],
    *,
    min_similarity: float = 0.75,
) -> Optional[tuple[str, float]]:
    best_word = ""
    best_sim = 0.0
    token_skeleton = consonant_skeleton(token)
    best_rank = -1.0
    for L in (len(token) - 2, len(token) - 1, len(token), len(token) + 1, len(token) + 2):
        if L <= 0:
            continue
        for cand in lexicon_by_length.get(L, []):
            sim = char_edit_similarity(token, cand)
            skel = skeleton_subsequence_similarity(token_skeleton, consonant_skeleton(cand))
            rank = sim + (0.25 * skel)
            if rank > best_rank:
                best_rank = rank
                best_sim = sim
                best_word = cand
    if best_word and best_sim >= min_similarity:
        return best_word, best_sim
    return None


def detect_reduplication_candidate(
    word: str,
    lexicon_set: set[str],
    lexicon_by_length: dict[int, list[str]],
) -> Optional[tuple[str, str, float, str]]:
    """Return (candidate_word, base_word, confidence, type)."""
    if len(word) < 4 or len(word) % 2 != 0:
        return None
    half = len(word) // 2
    left = word[:half]
    right = word[half:]

    if left == right and left in lexicon_set:
        return f"{left}-{right}", left, 1.0, "exact_reduplication"

    half_sim = char_edit_similarity(left, right)
    if half_sim < 0.80:
        return None

    left_match = best_lexicon_match_by_edit(left, lexicon_by_length, min_similarity=0.75)
    right_match = best_lexicon_match_by_edit(right, lexicon_by_length, min_similarity=0.75)
    if left_match is None or right_match is None:
        return None
    left_word, left_sim = left_match
    right_word, right_sim = right_match
    if left_word != right_word:
        return None
    conf = (left_sim + right_sim) / 2.0
    return f"{left_word}-{right_word}", left_word, conf, "noisy_reduplication"


def find_abbreviation_candidate(
    word: str,
    *,
    lexicon_words: list[str],
    lexicon_skeletons: list[str],
    first_char_index: dict[str, list[int]],
    scores: np.ndarray,
) -> Optional[tuple[str, float, float, float]]:
    if not is_abbreviation_like(word):
        return None

    src_skel = consonant_skeleton(word)
    if not src_skel:
        return None

    candidates_idx = first_char_index.get(word[0], [])
    best_word = ""
    best_cos = 0.0
    best_edit = 0.0
    best_rank = -1e9

    for idx in candidates_idx:
        cand_word = lexicon_words[idx]
        cand_skel = lexicon_skeletons[idx]
        if abs(len(cand_skel) - len(src_skel)) > 2:
            continue
        skel_sim = skeleton_subsequence_similarity(src_skel, cand_skel)
        if skel_sim < 0.74:
            continue
        cos = float(scores[idx])
        edit = char_edit_similarity(word, cand_word)
        rank = (0.65 * cos) + (0.20 * edit) + (0.15 * skel_sim)
        if rank > best_rank:
            best_rank = rank
            best_word = cand_word
            best_cos = cos
            best_edit = edit

    if not best_word:
        return None
    return best_word, best_cos, best_edit, best_rank


def choose_correction(
    source_word: str,
    *,
    lexicon_words: list[str],
    lexicon_skeletons: list[str],
    lexicon_set: set[str],
    lexicon_by_length: dict[int, list[str]],
    lexicon_word_to_idx: dict[str, int],
    first_char_index: dict[str, list[int]],
    scores: np.ndarray,
    top_k: int,
    cosine_floor: float,
    min_edit_similarity: float,
    max_len_diff: int,
    hybrid_threshold: float,
    w_cosine: float,
    w_edit: float,
    w_length_penalty: float,
    abbreviation_min_edit: float,
    abbreviation_hybrid_threshold: float,
    redup_hybrid_threshold: float,
) -> tuple[str, float, float, float, str, list[dict[str, float | str]]]:
    if scores.size == 0:
        return source_word, 0.0, 0.0, 0.0, "no_candidate", []

    if source_word in lexicon_set:
        return source_word, 1.0, 1.0, 1.0, "already_lexicon", []

    candidates: dict[str, tuple[float, str, float]] = {}
    # value tuple: (cosine, source_type, source_confidence)

    for idx in top_k_indices(scores, top_k):
        cand = lexicon_words[int(idx)]
        candidates[cand] = (float(scores[int(idx)]), "fasttext", 1.0)

    redup = detect_reduplication_candidate(source_word, lexicon_set, lexicon_by_length)
    if redup is not None:
        redup_word, base_word, conf, redup_type = redup
        base_idx = lexicon_word_to_idx.get(base_word)
        cos = float(scores[base_idx]) if base_idx is not None else 0.0
        prev = candidates.get(redup_word)
        if prev is None or cos > prev[0]:
            candidates[redup_word] = (cos, redup_type, conf)

    abbr = find_abbreviation_candidate(
        source_word,
        lexicon_words=lexicon_words,
        lexicon_skeletons=lexicon_skeletons,
        first_char_index=first_char_index,
        scores=scores,
    )
    if abbr is not None:
        abbr_word, abbr_cos, _, abbr_rank = abbr
        prev = candidates.get(abbr_word)
        if prev is None or prev[1] == "fasttext":
            candidates[abbr_word] = (abbr_cos, "abbreviation_rule", abbr_rank)

    if not candidates:
        return source_word, 0.0, 0.0, 0.0, "no_candidate", []

    candidate_details: list[dict[str, float | str]] = []
    best_word = source_word
    best_cos = 0.0
    best_edit = 0.0
    best_hybrid = -1e9
    best_type = "fasttext"

    for cand_word, (cand_cos, cand_type, cand_conf) in candidates.items():
        cand_norm = cand_word.replace("-", "")
        edit = char_edit_similarity(source_word, cand_norm)
        len_penalty = abs(len(source_word) - len(cand_norm)) / max(len(source_word), len(cand_norm))
        hybrid = (w_cosine * cand_cos) + (w_edit * edit) - (w_length_penalty * len_penalty)
        if cand_type != "fasttext":
            hybrid += 0.05 * cand_conf

        candidate_details.append(
            {
                "candidate": cand_word,
                "candidate_type": cand_type,
                "hybrid_score": hybrid,
                "cosine_similarity": cand_cos,
                "edit_similarity": edit,
                "length_penalty": len_penalty,
                "source_confidence": cand_conf,
            }
        )

        if hybrid > best_hybrid:
            best_hybrid = hybrid
            best_word = cand_word
            best_cos = cand_cos
            best_edit = edit
            best_type = cand_type

    candidate_details.sort(key=lambda d: float(d["hybrid_score"]), reverse=True)

    if best_word == source_word:
        return source_word, best_cos, best_edit, best_hybrid, "no_change_best_candidate", candidate_details

    best_norm = best_word.replace("-", "")
    len_diff = abs(len(source_word) - len(best_norm))

    if best_type == "abbreviation_rule":
        if best_cos < cosine_floor:
            return source_word, best_cos, best_edit, best_hybrid, "below_cosine_floor", candidate_details
        if best_edit < abbreviation_min_edit:
            return source_word, best_cos, best_edit, best_hybrid, "below_edit_threshold", candidate_details
        if best_hybrid < abbreviation_hybrid_threshold:
            return source_word, best_cos, best_edit, best_hybrid, "below_hybrid_threshold", candidate_details
        return best_word, best_cos, best_edit, best_hybrid, "corrected_abbreviation_rule", candidate_details

    if best_type in {"exact_reduplication", "noisy_reduplication"}:
        if best_hybrid < redup_hybrid_threshold:
            return source_word, best_cos, best_edit, best_hybrid, "below_hybrid_threshold", candidate_details
        return best_word, best_cos, best_edit, best_hybrid, "corrected_reduplication_rule", candidate_details

    if (
        is_abbreviation_like(source_word)
        and best_word
        and best_word[0] == source_word[0]
        and best_cos >= cosine_floor
        and best_edit >= abbreviation_min_edit
        and best_hybrid >= abbreviation_hybrid_threshold
    ):
        return best_word, best_cos, best_edit, best_hybrid, "corrected_abbreviation_rule", candidate_details

    if best_cos < cosine_floor:
        return source_word, best_cos, best_edit, best_hybrid, "below_cosine_floor", candidate_details
    if best_edit < min_edit_similarity:
        return source_word, best_cos, best_edit, best_hybrid, "below_edit_threshold", candidate_details
    if len_diff > max_len_diff:
        return source_word, best_cos, best_edit, best_hybrid, "length_diff_too_large", candidate_details
    if best_hybrid < hybrid_threshold:
        return source_word, best_cos, best_edit, best_hybrid, "below_hybrid_threshold", candidate_details
    return best_word, best_cos, best_edit, best_hybrid, "corrected_fasttext", candidate_details


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize slang/typo words with hybrid fastText ranking")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input CSV path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument("--model", type=Path, required=True, help="Path to fastText .bin model")
    parser.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON, help="Lexicon file path")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K fastText candidates (default: 50)")
    parser.add_argument("--cosine-floor", type=float, default=0.45, help="Cosine floor (default: 0.45)")
    parser.add_argument(
        "--min-edit-similarity",
        type=float,
        default=0.60,
        help="Minimum edit similarity for fastText candidate (default: 0.60)",
    )
    parser.add_argument("--max-len-diff", type=int, default=4, help="Maximum length difference (default: 4)")
    parser.add_argument("--hybrid-threshold", type=float, default=0.74, help="FastText hybrid threshold")
    parser.add_argument("--w-cosine", type=float, default=0.55, help="Weight for cosine score")
    parser.add_argument("--w-edit", type=float, default=0.30, help="Weight for edit similarity")
    parser.add_argument("--w-length-penalty", type=float, default=0.12, help="Length penalty weight")
    parser.add_argument("--abbreviation-min-edit", type=float, default=0.65, help="Abbrev min edit similarity")
    parser.add_argument(
        "--abbreviation-hybrid-threshold",
        type=float,
        default=0.46,
        help="Abbrev rule threshold",
    )
    parser.add_argument(
        "--redup-hybrid-threshold",
        type=float,
        default=0.52,
        help="Reduplication rule threshold",
    )
    parser.add_argument("--round", type=int, default=6, help="Rounding precision for scores")
    parser.add_argument(
        "--report-topn-path",
        type=Path,
        default=None,
        help="Optional CSV path for top-N candidate diagnostics per token",
    )
    parser.add_argument("--report-topn-size", type=int, default=5, help="Top-N report size")
    parser.add_argument(
        "--report-status-filter",
        type=str,
        default="all",
        help="Status filter for report rows: all or comma-separated statuses",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional JSON path for reproducibility manifest (default: <output>.manifest.json)",
    )
    parser.add_argument("--no-manifest", action="store_true", help="Disable reproducibility manifest generation")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    run_started_at = datetime.now(timezone.utc)

    if not args.input.exists():
        print(f"[error] Input CSV not found: {args.input}")
        return 1
    if not args.model.exists():
        print(f"[error] fastText model not found: {args.model}")
        return 1
    if args.top_k <= 0:
        print("[error] --top-k must be > 0")
        return 1
    if args.max_len_diff < 0:
        print("[error] --max-len-diff must be >= 0")
        return 1
    if args.round < 0:
        print("[error] --round must be >= 0")
        return 1
    if args.report_topn_size <= 0:
        print("[error] --report-topn-size must be > 0")
        return 1

    try:
        print(f"[info] Loading model: {args.model}")
        model = fasttext.load_model(str(args.model))

        print(f"[info] Loading lexicon: {args.lexicon}")
        lexicon_words = read_lexicon(args.lexicon)
        lexicon_set = set(lexicon_words)
        lexicon_by_length = build_lexicon_by_length(lexicon_words)
        lexicon_word_to_idx = {w: i for i, w in enumerate(lexicon_words)}
        lexicon_skeletons = [consonant_skeleton(w) for w in lexicon_words]
        first_char_index = build_lexicon_first_char_index(lexicon_words)
        lexicon_matrix = build_lexicon_matrix(model, lexicon_words)
        print(f"[info] Lexicon size: {len(lexicon_words):,}")

        args.output.parent.mkdir(parents=True, exist_ok=True)

        cache: dict[str, tuple[str, float, float, float, str, list[dict[str, float | str]]]] = {}
        status_counter: Counter[str] = Counter()
        report_rows: list[dict[str, str | float]] = []

        report_status_allow = None
        if args.report_status_filter != "all":
            report_status_allow = {s.strip() for s in args.report_status_filter.split(",") if s.strip()}

        with args.input.open("r", encoding="utf-8", newline="") as fin:
            reader = csv.DictReader(fin)
            if not reader.fieldnames:
                raise ValueError("Input CSV has no header")
            if KATA_COL not in reader.fieldnames:
                raise ValueError(f"Input CSV must contain column '{KATA_COL}'")

            out_fields = list(reader.fieldnames) + [
                "kata_normalized",
                "kata_koreksi_fasttext",
                "cosine_similarity",
                "edit_similarity",
                "hybrid_score",
                "correction_status",
            ]

            with args.output.open("w", encoding="utf-8", newline="") as fout:
                writer = csv.DictWriter(fout, fieldnames=out_fields)
                writer.writeheader()

                total = 0
                corrected = 0

                for row in reader:
                    total += 1
                    raw = (row.get(KATA_COL) or "").strip()
                    normalized = normalize_word(raw)

                    if not normalized:
                        result = ("", 0.0, 0.0, 0.0, "empty_token", [])
                    elif normalized in cache:
                        result = cache[normalized]
                    else:
                        scores = cosine_scores(model, lexicon_matrix, normalized)
                        result = choose_correction(
                            normalized,
                            lexicon_words=lexicon_words,
                            lexicon_skeletons=lexicon_skeletons,
                            lexicon_set=lexicon_set,
                            lexicon_by_length=lexicon_by_length,
                            lexicon_word_to_idx=lexicon_word_to_idx,
                            first_char_index=first_char_index,
                            scores=scores,
                            top_k=args.top_k,
                            cosine_floor=args.cosine_floor,
                            min_edit_similarity=args.min_edit_similarity,
                            max_len_diff=args.max_len_diff,
                            hybrid_threshold=args.hybrid_threshold,
                            w_cosine=args.w_cosine,
                            w_edit=args.w_edit,
                            w_length_penalty=args.w_length_penalty,
                            abbreviation_min_edit=args.abbreviation_min_edit,
                            abbreviation_hybrid_threshold=args.abbreviation_hybrid_threshold,
                            redup_hybrid_threshold=args.redup_hybrid_threshold,
                        )
                        cache[normalized] = result

                    fixed_word, cos_sim, edit_sim, hybrid_score, status, candidate_details = result
                    status_counter[status] += 1
                    if status.startswith("corrected_"):
                        corrected += 1

                    row["kata_normalized"] = normalized
                    row["kata_koreksi_fasttext"] = fixed_word
                    row["cosine_similarity"] = round(cos_sim, args.round)
                    row["edit_similarity"] = round(edit_sim, args.round)
                    row["hybrid_score"] = round(hybrid_score, args.round)
                    row["correction_status"] = status
                    writer.writerow(row)

                    if args.report_topn_path is not None:
                        if report_status_allow is None or status in report_status_allow:
                            base_row: dict[str, str | float] = {
                                "kata": raw,
                                "kata_normalized": normalized,
                                "kata_koreksi_fasttext": fixed_word,
                                "correction_status": status,
                                "selected_hybrid_score": round(hybrid_score, args.round),
                                "selected_cosine_similarity": round(cos_sim, args.round),
                                "selected_edit_similarity": round(edit_sim, args.round),
                            }
                            for i, cand in enumerate(candidate_details[: args.report_topn_size], start=1):
                                base_row[f"candidate_{i}"] = str(cand["candidate"])
                                base_row[f"candidate_{i}_type"] = str(cand["candidate_type"])
                                base_row[f"candidate_{i}_hybrid_score"] = round(float(cand["hybrid_score"]), args.round)
                                base_row[f"candidate_{i}_cosine"] = round(float(cand["cosine_similarity"]), args.round)
                                base_row[f"candidate_{i}_edit"] = round(float(cand["edit_similarity"]), args.round)
                                base_row[f"candidate_{i}_len_penalty"] = round(float(cand["length_penalty"]), args.round)
                                base_row[f"candidate_{i}_source_confidence"] = round(
                                    float(cand["source_confidence"]),
                                    args.round,
                                )
                            report_rows.append(base_row)

        print(f"[done] Wrote {total:,} rows to {args.output}")
        print(f"[done] Corrected rows: {corrected:,}")

        if args.report_topn_path is not None:
            args.report_topn_path.parent.mkdir(parents=True, exist_ok=True)
            report_fields = [
                "kata",
                "kata_normalized",
                "kata_koreksi_fasttext",
                "correction_status",
                "selected_hybrid_score",
                "selected_cosine_similarity",
                "selected_edit_similarity",
            ]
            for i in range(1, args.report_topn_size + 1):
                report_fields.extend(
                    [
                        f"candidate_{i}",
                        f"candidate_{i}_type",
                        f"candidate_{i}_hybrid_score",
                        f"candidate_{i}_cosine",
                        f"candidate_{i}_edit",
                        f"candidate_{i}_len_penalty",
                        f"candidate_{i}_source_confidence",
                    ]
                )
            with args.report_topn_path.open("w", encoding="utf-8", newline="") as fout:
                writer = csv.DictWriter(fout, fieldnames=report_fields)
                writer.writeheader()
                writer.writerows(report_rows)
            print(f"[done] Wrote top-N report: {args.report_topn_path} ({len(report_rows):,} rows)")

        if not args.no_manifest:
            manifest_path = args.manifest_path or Path(f"{args.output}.manifest.json")
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            run_completed_at = datetime.now(timezone.utc)

            manifest = {
                "schema_version": 1,
                "pipeline": "fasttext_slang_typo_correction",
                "method_core": {
                    "retrieval": "fastText top-K over lexicon",
                    "reranking": "cosine + edit similarity + length penalty",
                    "linguistic_rules": ["abbreviation expansion", "reduplication normalization"],
                },
                "run_started_at_utc": run_started_at.isoformat(),
                "run_completed_at_utc": run_completed_at.isoformat(),
                "runtime": {
                    "python_version": platform.python_version(),
                    "platform": platform.platform(),
                    "packages": {
                        "fasttext": package_version("fasttext"),
                        "numpy": package_version("numpy"),
                    },
                },
                "git": {"commit": get_git_commit(Path.cwd())},
                "artifacts": {
                    "input_csv": str(args.input.resolve()),
                    "lexicon_file": str(args.lexicon.resolve()),
                    "model_file": str(args.model.resolve()),
                    "output_csv": str(args.output.resolve()),
                    "topn_report_csv": str(args.report_topn_path.resolve()) if args.report_topn_path else None,
                },
                "checksums_sha256": {
                    "input_csv": sha256_file(args.input),
                    "lexicon_file": sha256_file(args.lexicon),
                    "model_file": sha256_file(args.model),
                    "output_csv": sha256_file(args.output),
                    "topn_report_csv": sha256_file(args.report_topn_path) if args.report_topn_path else None,
                },
                "parameters": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                "summary": {
                    "total_rows": total,
                    "corrected_rows": corrected,
                    "status_counts": dict(status_counter),
                    "unique_tokens_cached": len(cache),
                },
            }
            with manifest_path.open("w", encoding="utf-8") as mf:
                json.dump(manifest, mf, ensure_ascii=False, indent=2)
            print(f"[done] Wrote reproducibility manifest: {manifest_path}")

        return 0
    except KeyboardInterrupt:
        print("\n[info] Cancelled by user.")
        return 130
    except Exception as exc:
        print(f"[error] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
