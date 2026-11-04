# MP 7 Report: Few-shot prediction of tetracycline resistance in NARMS *E. coli*
Manickam Dhandapani · https://github.com/manickamksu/CIS731/tree/main/mp7-submission · Git tag `mp7-submission`

## Part 1A. Protocol and label space
`protocol.md` was committed in `{{PROTOCOL_COMMIT}}`, before the first results commit `{{RESULTS_COMMIT}}`.

**Task and labels.** Predict the tetracycline phenotype of a food-animal *E. coli* isolate. Gold labels are laboratory results (`TET SIR`): MIC ≤ 4 µg/mL = susceptible, ≥ 16 = resistant. Intermediates (MIC 8; n = 341) and missing results (n = 66) are excluded. Labels are direct lab measurements, not inferred labels.

**Data.** FDA/USDA-FSIS NARMS cecal file `CVM-NARMS-Cecal.xlsx` (74,701 × 201), SHA-256 `{{RAW_SHA}}`, kept unmodified. *E. coli* (`GENUS_NAME = Escherichia`): 18,272 isolates (2013–2023; cattle 9,350, swine 4,972, chickens 2,293, turkeys 1,657). After cleaning: 17,865 (R 8,106 / S 9,759). Key = `SAMPLE_ID`.

**Isolate records ("chunks").** Each isolate is serialized as text, e.g. *"Animal source: swine. Production type: market hogs. Year: 2020. Resistant to: ampicillin, sulfisoxazole. Susceptible to all other tested drugs."* Features are host species, production type, year, and the R/S/I result of 11 other drugs (AMC, AMP, AXO, AZI, CHL, CIP, COT, FIS, FOX, GEN, NAL); missing results are written as "not tested". Changes from the proposal:
- Region is dropped: `STATE` is empty for every row; production type replaces it.
- STR is dropped (> 30% missing) and MER is dropped (not tested 2013–2015).
- SUF and CIP2 are dropped as duplicates of FIS and CIP.

**Split.** Demonstration/exemplar pool = 2013–2019 (12,338; R 44.4%). Test = 2020–2023 (5,527; R 47.5%). To bound CPU runtime, a fixed, label-stratified sample of N = {{N_EVAL}} test isolates (seed 20261103; ids in `test_ids.json`) is used for every classifier, k and seed.

**Sweep.**
- k ∈ {0, 1, 3, 5, 8} demonstrations **per class** (2k per run), class-balanced and nested within a seed.
- Seeds {0, 1, 2, 3, 4}. A seed fixes which demos are drawn, their order, and the order of labels in the instruction.
- All classifiers share the same demos for a given (k, seed).

**Uncertainty.** Macro-F1 is reported as mean ± sample std over seeds. The 95% CI is a percentile bootstrap (B = 1000) that resamples test isolates, paired across seeds, of the seed-mean macro-F1 (RNG seed 20261103). Per-class precision and recall are reported at the best configuration.

## Part 1B. Prompt specification and failure-mode controls
```
[system] You are a veterinary microbiology assistant. You will see an antimicrobial-susceptibility
         surveillance record for one Escherichia coli isolate from a food-producing animal. The
         tetracycline result is hidden. Predict the tetracycline phenotype and answer with exactly ONE
         label from the list below. Output only the label word, with no explanation and no other text.
         Allowed labels:                                   (order permuted per seed)
         - Resistant (Tetracycline-resistant): the isolate is resistant to tetracycline (...)
         - Susceptible (Tetracycline-susceptible): the isolate is susceptible to tetracycline (...)
         Never output any other word.
[user]   Examples:
         Isolate: <record>
         Tetracycline: <Resistant|Susceptible>        × 2k, shuffled per (seed, k)
         Now classify this isolate.
         Isolate: <test record>
         Tetracycline:
[assistant] ▌
```
- **Verbalizers:** "Resistant" and "Susceptible". `check_verbalizers` confirms that their first tokens differ.
- **Constrained decoding:** one forward pass per isolate (the shared demo prefix is KV-cached). The next-token log-probabilities are restricted to the two verbalizer first tokens and renormalised. The model therefore cannot produce an out-of-space label, and `predict_proba` is a proper distribution.
- **Output parser:** `parse_label` is used only for free-text output. It takes the first line, strips quotes and punctuation, and does a case-insensitive exact match; anything else becomes `None`, which is logged and never coerced.

| Failure mode | Control | Verification |
|---|---|---|
| Majority-label bias (R/S imbalance, LM prior) | exactly k demos per class; neutral label descriptions | per-run max predicted share and KL(pred‖gold); majority baseline |
| Demonstration-position (recency) bias | demo order and label-list order re-permuted for every (seed, k); demos interleaved | share of predictions equal to the last demo's label vs. that label's gold share; seed std |
| Label leakage | TET and same-class drugs (DOX, CTC, TGC) excluded; WGS gene columns (e.g. `Tetracyclines` = tet genes) never used; temporal split | `assert_no_target_leak` (regex over all records) and `assert_disjoint` (pool ∩ test ids = ∅) pass |

## Part 2A. Classifiers and contracts
All three models implement one `Classifier` ABC (`fit`, `predict_proba`, `predict`). Subclasses only implement `_fit` / `_proba`, so the contracts can't be bypassed.
- **Contracts:** `deal.ensure` on `predict_proba` requires finite values ≥ 0, row sums of 1 ± 1e-6, and shape (n, |labels|). `deal.ensure` on `predict` requires every label to be in the declared space. A deliberately malformed classifier raises `ContractError` (notebook §3).
- **(i) Prototype:** class means of L2-normalised `all-MiniLM-L6-v2` embeddings of the demos, scored by softmax(cos / 0.05). At k = 0, the label descriptions serve as the prototypes.
- **(ii) kNN:** FAISS `IndexFlatIP` (exact cosine) over the same demo embeddings; a similarity-weighted vote of 5 neighbours.
- **(iii) Few-shot LLM:** `Qwen2.5-0.5B-Instruct`, a fixed demo block per run, constrained scoring as above.

## Part 2B. Shot-sweep results
Macro-F1, mean ± std over 5 seeds [95% bootstrap CI], N = {{N_EVAL}} test isolates (2020–2023).

{{RESULTS_TABLE}}

Majority-class baseline (always "{{MAJ_LABEL}}"): macro-F1 = {{MAJ_F1}}. **Best configuration: {{BEST}}.**

Per-class precision and recall at the best configuration (mean over seeds):

{{PER_CLASS_TABLE}}

{{DISCUSSION}}

## Part 2C. Reproducibility and quota
- **Environment:** pinned CPU DevContainer `{{CONTAINER}}`; Python {{PYTHON}}; key packages {{PACKAGES}}; `verify_env.py` passes. CUDA disabled (`CUDA_VISIBLE_DEVICES=""`, `torch.cuda.is_available() = {{CUDA}}`).
- **Models:** LLM `{{LLM}}`; embedder `{{EMB}}`.
- **Runtime and quota:** sweep wall time {{WALL}} min. Hosted-API requests: 0 (local inference only, so the INV-18 killswitch was never triggered). Local LLM usage: {{REQS}} forward passes, {{TOKENS}} prompt tokens (`quota_log.jsonl`).
- **Re-derivation:** every number in this report is regenerated from `predictions.jsonl` by `summarize()` and `fill_report.py`.

## GenAI audit
Generative AI (Claude) was used to design the prompt harness and the code scaffolding, and to draft this report's fixed text. All prompts, raw outputs, and my verification/changes are logged in `GenAI.pdf`. All numeric results were produced by the executed notebook, not by the AI.
