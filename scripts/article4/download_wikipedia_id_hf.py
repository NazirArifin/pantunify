#!/usr/bin/env python3
"""Download cleaned Indonesian Wikipedia from Hugging Face and save to data/article4.

Default behavior:
- Source dataset: wikimedia/wikipedia
- Auto-pick latest Indonesian config (e.g. 20231101.id)
- Stream rows and write JSONL to avoid loading all data in memory

Examples:
  /path/to/python scripts/article4/download_wikipedia_id_hf.py
  /path/to/python scripts/article4/download_wikipedia_id_hf.py --config 20231101.id
  /path/to/python scripts/article4/download_wikipedia_id_hf.py --limit 10000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

from datasets import get_dataset_config_names, load_dataset

DATASET_NAME = "wikimedia/wikipedia"
DEFAULT_OUT_DIR = Path("data/article4")
CONFIG_RE = re.compile(r"^(\d{8})\.id$")


def pick_latest_id_config(config_names: Iterable[str]) -> str:
    candidates: list[tuple[str, str]] = []
    for name in config_names:
        match = CONFIG_RE.match(name)
        if match:
            candidates.append((match.group(1), name))

    if not candidates:
        raise RuntimeError(
            "No Indonesian config found for wikimedia/wikipedia "
            "(expected pattern YYYYMMDD.id)."
        )

    # Sort by date string (YYYYMMDD) and pick latest.
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download cleaned Indonesian Wikipedia from Hugging Face"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Dataset config, e.g. 20231101.id. Default: auto latest Indonesian config.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory (default: data/article4)",
    )
    parser.add_argument(
        "--out-file",
        type=str,
        default=None,
        help="Output filename. Default: wikipedia_<config>.jsonl",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of rows to write.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10000,
        help="Print progress every N rows (default: 10000).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if args.limit is not None and args.limit <= 0:
        print("[error] --limit must be a positive integer")
        return 1
    if args.progress_every <= 0:
        print("[error] --progress-every must be a positive integer")
        return 1

    try:
        if args.config:
            config_name = args.config
        else:
            print("Fetching dataset configs from Hugging Face...")
            config_names = get_dataset_config_names(DATASET_NAME)
            config_name = pick_latest_id_config(config_names)

        out_file = args.out_file or f"wikipedia_{config_name}.jsonl"
        out_path = args.out_dir / out_file
        out_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Dataset:  {DATASET_NAME}")
        print(f"Config:   {config_name}")
        print(f"Output:   {out_path}")
        if args.limit:
            print(f"Limit:    {args.limit} rows")

        ds = load_dataset(
            DATASET_NAME,
            config_name,
            split="train",
            streaming=True,
        )

        count = 0
        with out_path.open("w", encoding="utf-8") as f:
            for row in ds:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1

                if count % args.progress_every == 0:
                    print(f"Written {count:,} rows...", flush=True)

                if args.limit is not None and count >= args.limit:
                    break

        print(f"[done] Wrote {count:,} rows to {out_path}")
        return 0

    except KeyboardInterrupt:
        print("\n[info] Cancelled by user.")
        return 130
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[error] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
