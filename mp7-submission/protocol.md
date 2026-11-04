# MP 7 Protocol: few-shot prediction of tetracycline resistance 

## 1. Task, data and label space
- **Project:** ML-based prediction of tetracycline resistance in food-animal *E. coli* isolates (CIS 731, solo; PS 4 proposal).
- **Data:** FDA NARMS isolate-level cecal data (USDA-FSIS), file `CVM-NARMS-Cecal.xlsx` (sheet "Cecal", 74,701 rows × 201 columns), downloaded 03 Nov 2026 from `<URL>`, SHA-256 `116831649876a47623040f76d50f806d2ac9940fde3a9983187159452696d7af`, kept unmodified.
- **Rows used:** `GENUS_NAME == "Escherichia"` → 18,272 isolates (2013–2023; cattle 9,350, swine 4,972, chickens 2,293, turkeys 1,657). After label cleaning: **17,865** (resistant 8,106, susceptible 9,759). Isolate key = `SAMPLE_ID` (unique; `ISOLATE_ID` is missing for 264 rows).
- **Label space (L = 2):** `resistant` (verbalizer "Resistant") and `susceptible` (verbalizer "Susceptible"), from the file's `TET SIR` interpretation. In this file that corresponds to MIC ≤ 4 µg/mL = S, 8 = I, ≥ 16 = R (CLSI tetracycline breakpoints for Enterobacterales; verify against the NARMS documentation). A single breakpoint is used across all years. Intermediate isolates (341) and isolates without a result (66) are excluded.
- **Deviation from the spec (declared up front):** MP 7's 8–12 label space is defined for the trope ontology / ASR label set (VDB and ASR tracks). This project's target, fixed in the approved PS 4 proposal, is binary laboratory resistance. Turning it into an artificial 8–12 class problem would change the research question, so the label space stays at L = 2. Every other protocol element (shot sweep, ≥ 5 seeds, three classifiers, contracts, bootstrap CIs) is implemented as specified.
- **Labels are gold:** they're direct laboratory results, not LLM or weak labels (INV-17 doesn't apply).

## 2. Isolate representation (the "chunks")
Each isolate becomes one text record:
`Animal source: swine. Production type: market hogs. Year: 2020. Resistant to: ampicillin, sulfisoxazole. [Intermediate: ...] [Not tested: ...] Susceptible to all other tested drugs.`
- Features: animal source (`HOST_SPECIES`), production type (from `SOURCE`: beef/dairy/market hogs/sows), year, and the R/S/I result of **11 other drugs**: AMC, AMP, AXO, AZI, CHL, CIP, COT, FIS, FOX, GEN, NAL. Missing results are written as "not tested", not imputed.
- **Changes from the proposal, and why:** *region* is dropped because `STATE` is empty for every row; production type is added in its place. STR is dropped (>30% missing). MER is dropped (not tested 2013–2015, a panel change). SUF is dropped (identical to FIS) and CIP2 is dropped (same MIC as CIP, alternative breakpoint).
- **Excluded for leakage:** TET (target), and DOX, CTC and TGC (same drug class). Also every gene/class summary column, above all `Tetracyclines` (tet genes from WGS), which would reveal the label.
- Embedding model for prototype/kNN: `sentence-transformers/all-MiniLM-L6-v2` @ `<revision sha>`, L2-normalized (inner product = cosine), FAISS `IndexFlatIP`.
- LLM: `Qwen/Qwen2.5-0.5B-Instruct` @ `<revision sha>`, local CPU, float32, no sampling.

## 3. Split and evaluation set
- **Demonstration / exemplar pool:** isolates collected 2013–2019 (**12,338**: R 5,483 / S 6,855).
- **Test set:** isolates collected 2020–2023 (**5,527**: R 2,623 / S 2,904). To bound CPU runtime, a fixed, label-stratified random sample of **N = 400** test isolates (R 190 / S 210) (seed 20261103; ids in `test_ids.json`) is used for **all** classifiers, k values and seeds.
- **Evaluation source (INV-12):** per the instructor's clarification (Nov 2026), `eval_mp7.jsonl` / `compute_graded_metric` apply only to the standard ASR (WWI-poem) testbed. NARMS defines no benchmark metric, so the MP 7 measure is used. Records carry `grading_eligible=False` because they weren't produced by `compute_graded_metric`.

## 4. Shot sweep and seeds
- k ∈ {0, 1, 3, 5, 8} **demonstrations per class** (2k demos per run), class-balanced by construction and nested within a seed.
- k = 0: the LLM sees the instruction only; prototype and kNN use the embedded label descriptions as their only exemplars.
- Seeds {0, 1, 2, 3, 4}. A seed fixes which pool isolates are drawn, the order of the demo block, and the order of labels in the instruction. All three classifiers share the same demos for a given (k, seed).
- Fixed hyperparameters (never tuned on the test set): softmax temperature 0.05 (prototype), n_neighbors = 5 with weights exp(sim/0.05) (kNN), and isolate text untruncated (≤ 96/256 token caps for demo/query).

## 5. Failure-mode controls
1. **Majority-label bias:** exactly k demos per class whatever the R/S prevalence. Per run, the predicted-label distribution, max predicted share and KL(pred‖gold) are logged. A majority-class baseline is reported.
2. **Demonstration-position bias:** demo order and label-list order are re-permuted for every (seed, k), with demos interleaved. Per run, the share of predictions equal to the last demo's label is logged.
3. **Label leakage:** the target and all tetracycline-class columns are excluded, and an assertion scans every record for those drug names. The temporal split puts no future isolates in the pool, and an assertion checks that pool and test isolate ids are disjoint. Label descriptions are neutral (no co-resistance hints).

## 6. Metrics and uncertainty
- Headline: **macro-F1** over {resistant, susceptible} (scikit-learn `f1_score(average="macro")`).
- Per (classifier, k): mean ± sample std (ddof = 1) over the 5 seeds.
- **95% CI:** percentile bootstrap, B = 1000 resamples of test isolates with replacement, paired across seeds; statistic = seed-mean macro-F1; RNG seed 20261103.
- At the best configuration: per-class precision and recall. **Resistant-class recall** is highlighted, as in the proposal.
- Deferred: calibration, significance tests, error taxonomy (MP 10); retrieval-selected demos (MP 9).

## 7. Compute and quota
Pinned CPU DevContainer `<image@sha256:...>`, `CUDA_VISIBLE_DEVICES=""`. Local inference only ⇒ 0 hosted-API requests; tokens logged in `quota_log.jsonl`.
