# Pantunify: A Rule-Based Toolkit for Indonesian Pantun Preprocessing and Filtering

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Pantunify is a Python toolkit for systematic curation of Indonesian pantun corpora. It integrates text cleaning, formal pantun validation, exact-fuzzy deduplication, structured tabular serialization, and descriptive statistical visualization. The pipeline is rule-based so filtering decisions remain transparent and reproducible.

## Research Objectives

The toolkit is designed for computational studies of oral literature with the following goals:

1. Standardize raw pantun corpora into analyzable four-line units.
2. Enforce formal constraints through syllable and rhyme validation.
3. Separate accepted and excluded records in an audit-friendly format.
4. Provide a structured dataset with per-line features for downstream quantitative analysis.
5. Produce descriptive figures and summary statistics for scientific reporting.

## Validation Criteria

Each pantun candidate in the main pipeline is evaluated with these criteria:

- exactly four lines,
- syllable count per line within 8-12,
- valid rhyme scheme (a-b-a-b or a-a-a-a),
- additional filtering with lexical constraints,
- exact and fuzzy deduplication to control redundancy.

## Data Processing Workflow

Recommended end-to-end workflow:

1. Prepare the merged corpus as data/merged.txt.
2. Run deduplication and validation to produce:
- data/ok.txt (accepted pantun),
- data/fail.txt (rejected pantun).
3. Convert data/ok.txt into the main dataset data/pantun_dataset.csv.
4. Convert data/fail.txt into the audit dataset data/excluded_pantun_dataset.csv.
5. Generate descriptive figures and statistical summaries.

To recalculate summary statistics only:

```bash
python scripts/article1/calc_dataset_stats.py --input data/pantun_dataset.csv
```

## Installation

```bash
python -m pip install -e .
```

## Core Commands

Main corpus filtering:

```bash
pantunify --input data/merged.txt --ok data/ok.txt --fail data/fail.txt
```

Convert accepted data to the main dataset:

```bash
python scripts/article1/ok_txt_to_csv.py --input data/ok.txt --output data/pantun_dataset.csv
```

Convert rejected data to the audit dataset:

```bash
python scripts/article1/fail_txt_to_csv.py --input data/fail.txt --output data/excluded_pantun_dataset.csv
```

Generate figures and JSON summary:

```bash
python scripts/article1/chart_generate.py
```

Visual outputs are written to figures:

- figure1_rhyme_schema.png
- figure2_syllable_hist.png
- figure3_syllable_boxplot.png
- dataset_summary.json

## Primary Dataset Schema

The main dataset is data/pantun_dataset.csv with 17 variables:

1. id
2. text_pantun
3. baris_sampiran
4. baris_isi
5. skema_rima
6. rima_akhir_baris_1
7. rima_akhir_baris_2
8. rima_akhir_baris_3
9. rima_akhir_baris_4
10. suku_kata_baris_1
11. suku_kata_baris_2
12. suku_kata_baris_3
13. suku_kata_baris_4
14. jumlah_kata_baris_1
15. jumlah_kata_baris_2
16. jumlah_kata_baris_3
17. jumlah_kata_baris_4

Columns rima_akhir_baris_1 through rima_akhir_baris_4 store line-ending rhyme extraction from last_syllable.

## Notes for Scientific Reporting

- Use data/pantun_dataset.csv for core dataset reporting.
- Use data/excluded_pantun_dataset.csv for exclusion and audit reporting.
- The data/ai_sampiran folder belongs to a separate research stream and is not required for the main dataset description.

## Contribution

Contributions on linguistic validation rules, deduplication strategy, and scientific documentation are welcome.
