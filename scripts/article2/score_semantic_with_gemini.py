#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

from google import genai


SYSTEM_PROMPT = """You are an expert judge for Indonesian pantun reverse-generation quality.
Task: score ONLY the semantic bridge quality between generated sampiran (2 lines)
and given isi (2 lines).

Scoring rubric (integer 1-5 only):
1 = No bridge at all, irrelevant mood/imagery.
2 = Weak bridge, mostly disconnected.
3 = Moderate bridge, partial mood or thematic continuity.
4 = Strong bridge, clear continuity and suitable transition.
5 = Excellent bridge, natural and elegant semantic transition.

Some models often employ local idioms or Malay phrasing that may sound unfamiliar yet authentic. Do not rush to assign a score of 3 (generic) if the wording feels unique; instead, appreciate it as an aesthetic asset (worthy of a score of 4 or 5). If you encounter a "word leak" (even a single keyword), you are prohibited from assigning a score higher than 1, regardless of how beautiful the resulting rhyme may be.

Output format rules:
- Return JSON only.
- Use exactly key: score.
- score must be integer 1..5.
"""


def build_user_prompt(isi_text, sampiran_text):
    return (
        "Evaluate semantic bridge quality for this pair.\n\n"
        f"ISI:\n{isi_text}\n\n"
        f"SAMPIRAN GENERATED:\n{sampiran_text}\n"
    )


def call_gemini(client, model, user_prompt):
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config={
            "system_instruction": SYSTEM_PROMPT,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "max_output_tokens": 120,
            "response_mime_type": "application/json",
        },
    )

    text_out = (getattr(response, "text", "") or "").strip()
    if not text_out:
        raise ValueError("Empty text output from Gemini API")

    return text_out


def parse_score(text):
    text = text.strip()

    # Preferred: strict JSON.
    try:
        obj = json.loads(text)
        score = int(obj.get("score"))
        if score < 1 or score > 5:
            raise ValueError("score out of range")
        return score
    except Exception:
        pass

    # Fallback: try extracting score from messy output.
    m = re.search(r"\b([1-5])\b", text)
    if not m:
        raise ValueError(f"Cannot parse score from output: {text[:200]}")

    score = int(m.group(1))
    return score


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Automatic semantic bridge scoring with Gemini API."
    )
    parser.add_argument(
        "--input",
        default="data/article2/100_Pantun_Eksperimen_long_scored.csv",
        help="CSV input path.",
    )
    parser.add_argument(
        "--output",
        default="data/article2/100_Pantun_Eksperimen_long_scored.csv",
        help="CSV output path. You can overwrite input safely.",
    )
    parser.add_argument(
        "--model",
        default="gemini-3.6-flash",
        help="Gemini model name.",
    )
    parser.add_argument(
        "--api-key-env",
        default="GEMINI_API_KEY",
        help="Environment variable name that stores Gemini API key.",
    )
    parser.add_argument(
        "--score-col",
        default="semantic_bridge_score_llm_1_5",
        help="Column name for LLM score.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only first N pending rows (0 means all pending).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Sleep seconds between requests.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries per row on API errors.",
    )

    args = parser.parse_args()

    api_key = os.getenv(args.api_key_env, "").strip()
    client = genai.Client(api_key=api_key) if api_key else genai.Client()

    in_path = Path(args.input)
    out_path = Path(args.output)

    rows, fieldnames = read_csv(in_path)
    if not rows:
        raise ValueError("Input CSV is empty")

    if args.score_col not in fieldnames:
        fieldnames.append(args.score_col)
        for r in rows:
            r[args.score_col] = ""

    pending_idx = []
    for i, r in enumerate(rows):
        score_val = (r.get(args.score_col) or "").strip()
        if score_val:
            continue

        isi_text = (r.get("isi_input_model") or "").strip()
        sampiran_text = (r.get("sampiran_generated") or "").strip()
        if not isi_text or not sampiran_text:
            continue

        pending_idx.append(i)

    if args.limit > 0:
        pending_idx = pending_idx[: args.limit]

    total = len(pending_idx)
    print(f"Pending rows to process: {total}")
    if total == 0:
        write_csv(out_path, rows, fieldnames)
        print("No pending rows. Nothing to do.")
        return

    processed = 0
    for n, idx in enumerate(pending_idx, start=1):
        row = rows[idx]
        isi_text = (row.get("isi_input_model") or "").strip()
        sampiran_text = (row.get("sampiran_generated") or "").strip()
        prompt = build_user_prompt(isi_text, sampiran_text)

        ok = False
        last_error = ""
        for attempt in range(1, args.max_retries + 1):
            try:
                text_out = call_gemini(client, args.model, prompt)
                score = parse_score(text_out)
                row[args.score_col] = str(score)
                ok = True
                break
            except Exception as e:
                last_error = str(e)

            backoff = min(30.0, args.sleep * (2 ** (attempt - 1)))
            time.sleep(backoff)

        if not ok:
            print(f"WARN row_index={idx} failed: {last_error[:220]}")

        processed += 1
        if n % 20 == 0 or n == total:
            # Save checkpoint regularly.
            write_csv(out_path, rows, fieldnames)
            print(f"Progress: {n}/{total}")

        time.sleep(args.sleep)

    # Final save.
    write_csv(out_path, rows, fieldnames)
    print(f"Done. Processed rows: {processed}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
