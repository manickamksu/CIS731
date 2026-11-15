# MP 9: Retrieval-Augmented Classification & Convolutions (NARMS *E. coli* tetracycline project track)

*Track note:* this continues my MP 7 project-track adaptation. Gold objects are 2020+ isolates; the demonstration and retrieval pool is 2013–2019; the cohort axis is collection year; and the label space is binary. Every number below comes from `results.json`, which the executed notebook (`mp9_notebook.ipynb`) writes from the raw files. *(At most 4 pages: keep each section to the tables plus 3–6 sentences.)*

## Codebook and Attestation
- Summarize the label definitions, the breakpoint (S ≤4 / I 8 / R ≥16), rules R1–R6 and the boundary cases. Point to `codebook.md`.
- Report the number of gold rows and the gate pre-check (`gold.gate`).
- Report the annotation effort from `attestation.md` (total minutes ≤ 120).
- Report label reliability: agreement of the human labels with the lab's `TET SIR` (`gold.lab_agreement`). List every disagreement and explain it, for example a censored value or a breakpoint revision.

## Inter-Annotator Agreement and Adjudication
- Give Cohen's κ with its 95% bootstrap CI, the percent agreement and n = 20 (`iaa`). Include the confusion matrix and the number of bootstrap resamples where κ was undefined.
- The CI is wide because n = 20. A κ close to 1 is expected because the rules are close to mechanical; say that this shows the codebook is unambiguous.
- Describe the adjudication rules A1–A4, and list each disagreement with the rule that settled it.

## Weak-Supervision Agreement Analysis
- **Labeling functions:** FIS_R, AMP_R, CHL_R, COT_R, MDR3, pan_S, host_prior and mp7_classifier (MP 7 kNN, k = 8 per class). Give each one's coverage, overlap, conflict and accuracy on the dev split (`lf_summary_dev`).
- **Vote aggregation:** the weights are the log-odds of each function's accuracy on the 20% dev split, floored at 0. Labels come from a weighted vote, and a tie or no vote means abstain. The silver pool has n = 2,000, every row marked `source=silver`.
- **Silver-vs-gold agreement:** report accuracy with its CI, κ and the confusion matrix. Add the bonus check of silver against the lab labels on the whole pool.
- **Failure modes:** use the error rate by host and the direction of the errors. Silver labels are advisory only and never enter the evaluation.

## Retrieval-Augmented Classification Results
- **Setup:** the same Qwen2.5-0.5B, template, verbalizers and constrained scoring as MP 7. The demonstration count is matched: k demonstrations **in total** for both the fixed and the retrieved prompts, with k ∈ {3, 5, 10}, over 3 seeds, evaluated on the adjudicated gold set.
- **Table:** macro-F1 and resistant-class recall for fixed_llm, retrieval_llm and retrieval_llm_balanced, each with a 95% bootstrap CI.
- **Comparison:** the paired bootstrap difference against fixed_llm at each k, and whether its CI excludes 0.
- **Retrieval quality:** P@k, Recall@k (pooled) and Recall@k (exact) with CIs, against the §6 relevance rule. Note that pooled Recall@10 equals 1 by construction, and that exact recall is small because thousands of pool isolates are relevant.
- **Diagnostics:** the label share of the retrieved demonstrations (majority-label bias) and the maximum predicted-class share.

## Cohort Convolution Analysis
- **Axis and vectors:** the axis is collection year. F is a years × drugs matrix of resistance rates, tetracycline included, using drugs tested in every year. Report the number of isolates per year.
- **Kernels:** smoothing [0.25, 0.5, 0.25] and first difference [-1, 1], applied in 'valid' mode with the cross-correlation convention, so a positive value means resistance went up.
- **Drift:** the drift magnitude per transition (mean |Δ|, in percentage points) with basic-bootstrap CIs, plus the signed change in tetracycline per transition. Include the figure `fig_tet_drift.png`.
- **2016→2017 (GFI #213):** report the excess drift and its CI, as an association only.
- **CIS 731:** the learned Conv1d kernel against persistence, the smoothing-kernel forecast and difference extrapolation, using rolling-origin forecasts of the last 2 years. Give MAE in pp with CIs, the learned kernel weights, the closed-form check, and the limitation that the series has only about 10 years.

## GenAI Audit
I used the Generative AI Kit (Claude) to plan the assignment, to draft the wording of the codebook, and to write the analysis code (`mp9_narms.py`, `mp9_notebook.ipynb`). It did not assign any gold label. I reviewed and ran every AI-generated analysis step, and each one reproduces from the raw files through the notebook. The full log is in `GenAI.pdf`.
