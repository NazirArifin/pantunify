#!/usr/bin/env python3
"""Build frequency dictionary from Indonesian Wikipedia JSONL and score collocations.

Input format:
- JSONL file where each line is a JSON object containing a "text" field.

Outputs:
- SQLite database with unigram frequencies, bigram frequencies, and metadata.

Examples:
  /path/to/python scripts/article4/build_frequency_dictionary.py build
  /path/to/python scripts/article4/build_frequency_dictionary.py build --max-docs 10000
  /path/to/python scripts/article4/build_frequency_dictionary.py score --w1 pisang --w2 emas
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_INPUT = Path("data/article4/wikipedia_20231101.id.jsonl")
DEFAULT_DB = Path("data/article4/frequency_dictionary.sqlite")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")


def tokenize(text: str, lowercase: bool = True) -> list[str]:
    tokens = TOKEN_RE.findall(text)
    if lowercase:
        tokens = [t.lower() for t in tokens]
    return tokens


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;

        CREATE TABLE IF NOT EXISTS unigrams (
            word TEXT PRIMARY KEY,
            freq INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS bigrams (
            w1 TEXT NOT NULL,
            w2 TEXT NOT NULL,
            freq INTEGER NOT NULL,
            PRIMARY KEY (w1, w2)
        );

        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_bigrams_w1 ON bigrams(w1);
        """
    )
    conn.commit()


def upsert_unigrams(conn: sqlite3.Connection, counts: Counter[str]) -> None:
    if not counts:
        return
    conn.executemany(
        """
        INSERT INTO unigrams(word, freq) VALUES(?, ?)
        ON CONFLICT(word) DO UPDATE SET freq = freq + excluded.freq
        """,
        counts.items(),
    )


def upsert_bigrams(conn: sqlite3.Connection, counts: Counter[tuple[str, str]]) -> None:
    if not counts:
        return
    conn.executemany(
        """
        INSERT INTO bigrams(w1, w2, freq) VALUES(?, ?, ?)
        ON CONFLICT(w1, w2) DO UPDATE SET freq = freq + excluded.freq
        """,
        ((w1, w2, freq) for (w1, w2), freq in counts.items()),
    )


def set_metadata(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO metadata(key, value) VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )


def get_metadata(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    return str(row[0]) if row else None


def save_checkpoint(
    conn: sqlite3.Connection,
    *,
    total_docs: int,
    total_tokens: int,
    tokenizer_pattern: str,
    lowercase: bool,
    input_path: Path,
) -> None:
    set_metadata(conn, "total_tokens", str(total_tokens))
    set_metadata(conn, "total_docs", str(total_docs))
    set_metadata(conn, "tokenizer", tokenizer_pattern)
    set_metadata(conn, "lowercase", str(lowercase))
    set_metadata(conn, "input_path", str(input_path))


def flush_counts(
    conn: sqlite3.Connection,
    uni_counts: Counter[str],
    bi_counts: Counter[tuple[str, str]],
) -> None:
    upsert_unigrams(conn, uni_counts)
    upsert_bigrams(conn, bi_counts)
    conn.commit()
    uni_counts.clear()
    bi_counts.clear()


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at line {line_no}: {exc}") from exc


def build_dictionary(
    input_path: Path,
    db_path: Path,
    lowercase: bool,
    flush_every_docs: int,
    fresh: bool,
    max_docs: Optional[int],
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")


    if fresh and db_path.exists():
        os.remove(db_path)
        print(f"[info] Removed existing database: {db_path}")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        ensure_schema(conn)

        start_docs = 0
        start_tokens = 0
        if not fresh:
            prev_docs = get_metadata(conn, "total_docs")
            prev_tokens = get_metadata(conn, "total_tokens")
            prev_input = get_metadata(conn, "input_path")
            prev_tokenizer = get_metadata(conn, "tokenizer")
            prev_lowercase = get_metadata(conn, "lowercase")

            if prev_input and prev_input != str(input_path):
                raise RuntimeError(
                    "Input path differs from existing checkpoint. "
                    "Use --fresh to rebuild."
                )
            if prev_tokenizer and prev_tokenizer != TOKEN_RE.pattern:
                raise RuntimeError(
                    "Tokenizer pattern differs from existing checkpoint. "
                    "Use --fresh to rebuild."
                )
            if prev_lowercase and prev_lowercase != str(lowercase):
                raise RuntimeError(
                    "Lowercase setting differs from existing checkpoint. "
                    "Use --fresh to rebuild."
                )

            start_docs = int(prev_docs) if prev_docs else 0
            start_tokens = int(prev_tokens) if prev_tokens else 0

            if start_docs > 0:
                print(
                    f"[resume] Continuing from checkpoint: "
                    f"{start_docs:,} docs, {start_tokens:,} tokens"
                )

        uni_counts: Counter[str] = Counter()
        bi_counts: Counter[tuple[str, str]] = Counter()
        total_tokens = start_tokens
        total_docs = start_docs
        processed_new_docs = 0
        seen_docs = 0
        try:
            for row in iter_jsonl(input_path):
                seen_docs += 1
                if seen_docs <= start_docs:
                    if seen_docs % 50000 == 0:
                        print(
                            f"[resume] Skipped {seen_docs:,}/{start_docs:,} docs...",
                            flush=True,
                        )
                    continue

                text = row.get("text", "")
                total_docs = seen_docs
                processed_new_docs += 1

                if not isinstance(text, str):
                    if processed_new_docs % flush_every_docs == 0:
                        flush_counts(conn, uni_counts, bi_counts)
                        save_checkpoint(
                            conn,
                            total_docs=total_docs,
                            total_tokens=total_tokens,
                            tokenizer_pattern=TOKEN_RE.pattern,
                            lowercase=lowercase,
                            input_path=input_path,
                        )
                        conn.commit()
                    continue

                tokens = tokenize(text, lowercase=lowercase)
                if tokens:
                    total_tokens += len(tokens)
                    uni_counts.update(tokens)
                    bi_counts.update(zip(tokens, tokens[1:]))

                if processed_new_docs % flush_every_docs == 0:
                    flush_counts(conn, uni_counts, bi_counts)
                    save_checkpoint(
                        conn,
                        total_docs=total_docs,
                        total_tokens=total_tokens,
                        tokenizer_pattern=TOKEN_RE.pattern,
                        lowercase=lowercase,
                        input_path=input_path,
                    )
                    conn.commit()
                    print(
                        f"Processed {total_docs:,} docs, "
                        f"{total_tokens:,} tokens...",
                        flush=True,
                    )

                if max_docs is not None and processed_new_docs >= max_docs:
                    break
        except KeyboardInterrupt:
            flush_counts(conn, uni_counts, bi_counts)
            save_checkpoint(
                conn,
                total_docs=total_docs,
                total_tokens=total_tokens,
                tokenizer_pattern=TOKEN_RE.pattern,
                lowercase=lowercase,
                input_path=input_path,
            )
            conn.commit()
            print("\n[info] Interrupted. Progress checkpoint saved.")
            print("       Run the same build command again to resume.")
            return

        flush_counts(conn, uni_counts, bi_counts)
        save_checkpoint(
            conn,
            total_docs=total_docs,
            total_tokens=total_tokens,
            tokenizer_pattern=TOKEN_RE.pattern,
            lowercase=lowercase,
            input_path=input_path,
        )
        conn.commit()

        print("[done] Frequency dictionary built.")
        print(f"Database:     {db_path}")
        print(f"Total docs:   {total_docs:,}")
        print(f"Total tokens: {total_tokens:,}")

    finally:
        conn.close()


def get_freq_unigram(conn: sqlite3.Connection, word: str) -> int:
    row = conn.execute("SELECT freq FROM unigrams WHERE word = ?", (word,)).fetchone()
    return int(row[0]) if row else 0


def get_freq_bigram(conn: sqlite3.Connection, w1: str, w2: str) -> int:
    row = conn.execute(
        "SELECT freq FROM bigrams WHERE w1 = ? AND w2 = ?",
        (w1, w2),
    ).fetchone()
    return int(row[0]) if row else 0


def get_total_tokens(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT value FROM metadata WHERE key = 'total_tokens'"
    ).fetchone()
    if not row:
        raise RuntimeError("metadata.total_tokens not found in database")
    return int(row[0])


def compute_pmi(f1: int, f2: int, f12: int, n: int) -> Optional[float]:
    if f1 <= 0 or f2 <= 0 or f12 <= 0 or n <= 0:
        return None
    return math.log2((f12 * n) / (f1 * f2))


def compute_t_score(f1: int, f2: int, f12: int, n: int) -> Optional[float]:
    if f1 <= 0 or f2 <= 0 or f12 <= 0 or n <= 0:
        return None
    expected = (f1 * f2) / n
    return (f12 - expected) / math.sqrt(f12)


def score_pair(db_path: Path, w1: str, w2: str, lowercase: bool) -> int:
    if lowercase:
        w1 = w1.lower()
        w2 = w2.lower()

    if not db_path.exists():
        print(f"[error] Database not found: {db_path}")
        return 1

    conn = sqlite3.connect(db_path)
    try:
        f1 = get_freq_unigram(conn, w1)
        f2 = get_freq_unigram(conn, w2)
        f12 = get_freq_bigram(conn, w1, w2)
        n = get_total_tokens(conn)

        pmi = compute_pmi(f1, f2, f12, n)
        t_score = compute_t_score(f1, f2, f12, n)

        print(f"Pair: ({w1}, {w2})")
        print(f"f(w1): {f1}")
        print(f"f(w2): {f2}")
        print(f"f(w1,w2): {f12}")
        print(f"N: {n}")
        print(f"PMI: {pmi if pmi is not None else 'undefined'}")
        print(f"T-Score: {t_score if t_score is not None else 'undefined'}")

        return 0
    finally:
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build frequency dictionary and score word pairs (PMI, T-Score)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_build = subparsers.add_parser("build", help="Build frequency dictionary database")
    p_build.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input JSONL corpus (default: data/article4/wikipedia_20231101.id.jsonl)",
    )
    p_build.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help="Output SQLite DB path (default: data/article4/frequency_dictionary.sqlite)",
    )
    p_build.add_argument(
        "--flush-every-docs",
        type=int,
        default=1000,
        help="Flush counters to DB every N documents (default: 1000)",
    )
    p_build.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional maximum number of documents to process.",
    )
    p_build.add_argument(
        "--no-lowercase",
        action="store_true",
        help="Disable lowercasing during tokenization.",
    )
    p_build.add_argument(
        "--fresh",
        action="store_true",
        help="Rebuild from scratch by removing existing DB before processing.",
    )

    p_score = subparsers.add_parser("score", help="Score one word pair")
    p_score.add_argument("--w1", type=str, required=True, help="First word")
    p_score.add_argument("--w2", type=str, required=True, help="Second word")
    p_score.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help="SQLite DB path (default: data/article4/frequency_dictionary.sqlite)",
    )
    p_score.add_argument(
        "--no-lowercase",
        action="store_true",
        help="Disable lowercasing before lookup.",
    )

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "build":
        if args.flush_every_docs <= 0:
            print("[error] --flush-every-docs must be a positive integer")
            return 1
        if args.max_docs is not None and args.max_docs <= 0:
            print("[error] --max-docs must be a positive integer")
            return 1

        try:
            build_dictionary(
                input_path=args.input,
                db_path=args.db,
                lowercase=not args.no_lowercase,
                flush_every_docs=args.flush_every_docs,
                max_docs=args.max_docs,
                fresh=args.fresh,
            )
            return 0
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[error] {exc}")
            return 1

    if args.command == "score":
        return score_pair(
            db_path=args.db,
            w1=args.w1,
            w2=args.w2,
            lowercase=not args.no_lowercase,
        )

    print("[error] Unknown command")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
