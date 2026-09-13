# Reproducibility Protocol

## Article
A structured dataset of 6,640 Indonesian pantun for text generation and computational linguistics

## Scope
This document describes how to reproduce the main dataset artifact and the audit artifact using the article1 pipeline in this repository.

## Canonical Artifacts
Use the following files as the canonical release artifacts:

- data/pantun_dataset.csv
- data/excluded_pantun_dataset.csv

SHA256 checksums at the time this protocol was prepared:

- data/pantun_dataset.csv: 66aa8277060040c61c2b519d084747db950c586125759931ffc6a917d7b203eb
- data/excluded_pantun_dataset.csv: 4e486eac6b642d90bbf4abe8a19684846109a0937cc03faad08949e19635a96e

Line counts at the time this protocol was prepared:

- data/pantun_dataset.csv: 6641 lines (header + 6640 records)
- data/excluded_pantun_dataset.csv: 2126 lines (header + 2125 records)

## Required Code
Primary scripts used for the Data in Brief pipeline:

- scripts/article1/merge_dataset.py
- scripts/article1/rebuild_merged_txt.py
- scripts/article1/ok_txt_to_csv.py
- scripts/article1/fail_txt_to_csv.py
- scripts/article1/calc_dataset_stats.py
- scripts/article1/chart_generate.py

Core modules called by these scripts:

- src/pantunify/cli.py
- src/pantunify/classifier.py
- src/pantunify/utils.py

Project-level requirements:

- pyproject.toml
- data/kata-dasar.txt

## Environment Setup
Run commands from the repository root.

```bash
python -m pip install -e .
```

## Protocol A: Verify Released CSV Artifacts (Recommended for Reviewers)
This protocol verifies that uploaded dataset files are identical to the release artifacts.

```bash
wc -l data/pantun_dataset.csv data/excluded_pantun_dataset.csv
sha256sum data/pantun_dataset.csv data/excluded_pantun_dataset.csv
```

Expected results:

- data/pantun_dataset.csv has 6641 lines
- data/excluded_pantun_dataset.csv has 2126 lines
- SHA256 values match those listed under Canonical Artifacts

## Protocol B: Regenerate Structured Dataset from Curated Text Corpus
This protocol regenerates the structured dataset from the curated text corpus.

1) Optional: rebuild merged corpus from source files

```bash
python scripts/article1/rebuild_merged_txt.py --output data/merged.txt
```

or

```bash
python scripts/article1/merge_dataset.py
```

2) Run deduplication and validation into accepted/rejected files

```bash
pantunify --input data/merged.txt --ok data/ok.txt --fail data/fail.txt
```

3) Build the 6,640-record main dataset from accepted data

```bash
python scripts/article1/ok_txt_to_csv.py --input data/ok.txt --output data/pantun_dataset.csv --limit 6640
```

4) Optional: if data/fail.txt is available, build the exclusion dataset

```bash
python scripts/article1/fail_txt_to_csv.py --input data/fail.txt --output data/excluded_pantun_dataset.csv
```

5) Generate summary statistics and descriptive figures

```bash
python scripts/article1/calc_dataset_stats.py --input data/pantun_dataset.csv
python scripts/article1/chart_generate.py
```

Expected visual and summary outputs:

- figures/figure1_rhyme_schema.png
- figures/figure2_syllable_hist.png
- figures/figure3_syllable_boxplot.png
- figures/dataset_summary.json

## Expected Structural Properties for Main Dataset
For data/pantun_dataset.csv (main dataset):

- 17 columns
- 6640 data rows (excluding header)
- valid rhyme scheme labels in skema_rima (a-a-a-a and a-b-a-b)
- per-line features available: rhyme ending, syllable count, and word count

Quick check:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv('data/pantun_dataset.csv')
assert len(df) == 6640, f'Row count mismatch: {len(df)}'
assert df.shape[1] == 17, f'Column count mismatch: {df.shape[1]}'
print('OK: rows=6640, cols=17')
print(df.columns.tolist())
PY
```

## Notes
- If your goal is release integrity checking for peer review, use Protocol A.
- If your goal is full regeneration from curated text, use Protocol B.
- If data/fail.txt is unavailable in a specific snapshot, skip excluded_pantun_dataset generation and use the released exclusion CSV artifact.