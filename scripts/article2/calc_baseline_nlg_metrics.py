#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer


def _norm_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _bleu_score(reference: str, candidate: str) -> float:
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    if not ref_tokens or not cand_tokens:
        return 0.0
    smoothie = SmoothingFunction().method1
    return float(sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothie))


def compute_metrics(df: pd.DataFrame, ref_col: str, cand_col: str):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

    refs = [_norm_text(x) for x in df[ref_col].fillna("").tolist()]
    cands = [_norm_text(x) for x in df[cand_col].fillna("").tolist()]

    bleu_vals = []
    rouge1_vals = []
    rouge2_vals = []
    rougel_vals = []

    for ref, cand in zip(refs, cands):
        bleu_vals.append(_bleu_score(ref, cand))
        r = scorer.score(ref, cand)
        rouge1_vals.append(float(r["rouge1"].fmeasure))
        rouge2_vals.append(float(r["rouge2"].fmeasure))
        rougel_vals.append(float(r["rougeL"].fmeasure))

    out = df.copy()
    out["bleu"] = bleu_vals
    out["rouge1_f"] = rouge1_vals
    out["rouge2_f"] = rouge2_vals
    out["rougeL_f"] = rougel_vals

    return out


def merge_bertscore(scored_df: pd.DataFrame, bert_df: pd.DataFrame, merge_keys: list[str]) -> pd.DataFrame:
    required_bert_cols = {"bertscore_p", "bertscore_r", "bertscore_f1"}
    missing_bert_cols = [c for c in required_bert_cols if c not in bert_df.columns]
    if missing_bert_cols:
        raise ValueError(f"BERTScore file missing required columns: {missing_bert_cols}")

    missing_keys = [k for k in merge_keys if k not in scored_df.columns or k not in bert_df.columns]
    if missing_keys:
        raise ValueError(
            "Merge keys not found in both files: "
            f"{missing_keys}. Provide matching keys via --merge-keys."
        )

    right_cols = merge_keys + ["bertscore_p", "bertscore_r", "bertscore_f1"]
    dedup_right = bert_df[right_cols].drop_duplicates(subset=merge_keys)

    merged = scored_df.merge(dedup_right, on=merge_keys, how="left", validate="m:1")
    return merged


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["bleu", "rouge1_f", "rouge2_f", "rougeL_f"]
    optional_metrics = ["bertscore_p", "bertscore_r", "bertscore_f1"]
    metrics += [m for m in optional_metrics if m in df.columns]

    grouped = (
        df.groupby(["model", "setting"], dropna=False)[metrics]
        .agg(["count", "mean", "std"])
        .reset_index()
    )

    grouped.columns = [
        "_".join([c for c in col if c]).rstrip("_") if isinstance(col, tuple) else col
        for col in grouped.columns
    ]
    return grouped


def main():
    parser = argparse.ArgumentParser(
        description="Compute BLEU/ROUGE baselines for generated sampiran; optionally merge BERTScore from external file."
    )
    parser.add_argument(
        "--input",
        default="data/article2/100_Pantun_Eksperimen_1200_xrasa_final.csv",
        help="Input CSV with sampiran_asli and sampiran_generated.",
    )
    parser.add_argument(
        "--output",
        default="data/article2/100_Pantun_Eksperimen_1200_with_bleu_rouge.csv",
        help="Output CSV with row-level BLEU and ROUGE.",
    )
    parser.add_argument(
        "--summary",
        default="data/article2/table_main_baselines_1200.csv",
        help="Output CSV summary by model and setting.",
    )
    parser.add_argument(
        "--reference-col",
        default="sampiran_asli",
        help="Reference text column.",
    )
    parser.add_argument(
        "--candidate-col",
        default="sampiran_generated",
        help="Candidate/generated text column.",
    )
    parser.add_argument(
        "--bertscore-input",
        default="",
        help="Optional CSV from Colab containing merge keys and bertscore_p/r/f1 columns.",
    )
    parser.add_argument(
        "--output-merged",
        default="data/article2/100_Pantun_Eksperimen_1200_with_baselines.csv",
        help="Output CSV after merging BERTScore. Written only when --bertscore-input is provided.",
    )
    parser.add_argument(
        "--summary-merged",
        default="data/article2/table_main_baselines_1200_with_bertscore.csv",
        help="Summary CSV after merging BERTScore. Written only when --bertscore-input is provided.",
    )
    parser.add_argument(
        "--merge-keys",
        default="no,id_data_asli,model,setting",
        help="Comma-separated key columns for merging BERTScore output.",
    )
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    summary_out = Path(args.summary)

    df = pd.read_csv(inp)
    required = ["model", "setting", args.reference_col, args.candidate_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    scored = compute_metrics(df, ref_col=args.reference_col, cand_col=args.candidate_col)

    out.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out, index=False)

    summary = summarize(scored)
    summary.to_csv(summary_out, index=False)

    print(f"rows_total={len(df)}")
    print(f"output={out}")
    print(f"summary={summary_out}")

    if args.bertscore_input:
        bert_path = Path(args.bertscore_input)
        bert_df = pd.read_csv(bert_path)
        merge_keys = [x.strip() for x in args.merge_keys.split(",") if x.strip()]
        merged = merge_bertscore(scored, bert_df, merge_keys=merge_keys)

        output_merged = Path(args.output_merged)
        output_merged.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_merged, index=False)

        summary_merged = summarize(merged)
        summary_merged_path = Path(args.summary_merged)
        summary_merged_path.parent.mkdir(parents=True, exist_ok=True)
        summary_merged.to_csv(summary_merged_path, index=False)

        print(f"bertscore_input={bert_path}")
        print(f"output_merged={output_merged}")
        print(f"summary_merged={summary_merged_path}")


if __name__ == "__main__":
    main()
