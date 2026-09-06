# Draft Results and Discussion (Sample 300)

## A. Human-AI Calibration for LLM-as-a-Judge

Before large-scale scoring, we validated the judge reliability on a blind set of 50 samples. The correlation between human ratings and LLM ratings is strong and statistically significant (Spearman $\rho = 0.90249$, permutation $p = 9.999\times10^{-5}$). This indicates that LLM-based semantic scoring can be used as a scalable proxy for semantic bridge evaluation in the benchmark.

The calibration also shows a conservative tendency of the LLM judge, where the average LLM score is lower than the average human score (mean difference $=-0.62$ on a 1-5 Likert scale). Therefore, all final score interpretations should be read as slightly strict estimates.

## B. Final xRASA Performance on Stratified Sample (n=300)

Using a stratified subset (25 items for each model-setting cell), we computed final xRASA as:

$$
\text{xRASA} = A \times \left(0.5\cdot R + 0.5\cdot S_{\text{judge}}\right)
$$

with $S_{\text{judge}}$ normalized from Likert 1-5 to $[0,1]$.

### Main ranking by xRASA (higher is better)

Few-shot:
- Claude: 0.7841 (Rank 1)
- Gemini: 0.7550 (Rank 2)
- Chat-GPT: 0.4420 (Rank 3)
- Llama 3.1: 8B: 0.1400 (Rank 4)
- DeepSeek-R1: 0.0600 (Rank 5)
- Sailor2: 0.0190 (Rank 6)

Zero-shot:
- Claude: 0.7500 (Rank 1)
- Gemini: 0.7300 (Rank 2)
- Chat-GPT: 0.4200 (Rank 3)
- DeepSeek-R1: 0.1570 (Rank 4)
- Llama 3.1: 8B: 0.1570 (Rank 5, tie-level mean rounded)
- Sailor2: 0.1144 (Rank 6)

These results indicate that Claude and Gemini are consistently dominant under both prompting settings, while open/regional models remain substantially lower on semantic-structural composition quality.

## C. Content Leakage Pattern

For CLR (lower is better), Gemini remains the best model in both settings (few-shot mean CLR = 0.0000; zero-shot mean CLR = 0.0157), followed by Claude. This suggests that top xRASA performance is accompanied by stronger lexical separation between sampiran and isi, which is consistent with pantun constraints.

## D. Significance Analysis

### Across-model differences (within each setting)

Permutation one-way ANOVA shows statistically significant between-model differences for xRASA in both settings:
- Few-shot: $F=84.11$, $p=0.00019996$
- Zero-shot: $F=41.66$, $p=0.00019996$

Hence, architecture choice has a strong effect on reverse-pantun generation quality.

### Few-shot vs zero-shot (within each model)

Because the stratified sample was independently drawn per cell, zero-shot and few-shot subsets are not perfectly paired by original id. Therefore, we use unpaired permutation tests (difference in means) as the primary inference.

- Chat-GPT: not significant ($p=0.7932$)
- Claude: not significant ($p=0.5343$)
- DeepSeek-R1: not significant at $\alpha=0.05$ ($p=0.0866$)
- Gemini: not significant ($p=0.7588$)
- Llama 3.1: 8B: not significant ($p=0.7828$)
- Sailor2: significant decrease under few-shot ($p=0.0136$)

Overall, few-shot prompting does not universally improve final xRASA across models in this sample, but it can strongly affect specific model families.

## E. Practical Interpretation for the Paper

1. The benchmark evidence supports Claude and Gemini as leading baselines for reverse pantun generation quality under xRASA.
2. Model selection contributes more variance than prompting strategy in this sample.
3. The strong human-LLM calibration justifies scalable LLM judging, with explicit disclosure that the judge is conservative.
4. Future full-run analysis on the complete dataset (n=1200) is expected to narrow confidence intervals and stabilize per-model few-shot effect estimation.
