# MP 7 Report: Few-shot prediction of tetracycline resistance in NARMS *E. coli*
Manickam Dhandapani · CIS 731 · tag `mp7-submission` · ≤ 3 pages

> **Evaluation source (INV-12).** Per the instructor's clarification, `eval_mp7.jsonl` / `compute_graded_metric` apply only to the standard ASR (WWI-poem) testbed. This project uses NARMS isolate data, which defines no benchmark metric, so all results use the MP 7 measure: macro-F1 with 95% bootstrap CIs on held-out 2020+ isolates.
> **Label space.** Binary (resistant/susceptible) as in the approved proposal. This is a declared deviation from the 8–12 trope labels; see protocol §1.

## Part 1A. Protocol and label space (15 pts)
`protocol.md` was committed in `<sha>` on `<date>`, before the first results commit `<sha>`. <Condensed copy of protocol §1–4 and §6, with every value unchanged.>

## Part 1B. Prompt specification and failure-mode controls (15 pts)
```
[system] You are a veterinary microbiology assistant ... The tetracycline result is hidden ...
         Allowed labels:
         - Resistant (Tetracycline-resistant): ...      (order permuted per seed)
         - Susceptible (Tetracycline-susceptible): ...
[user]   Examples:
         Isolate: Animal source: swine. Production type: sows. Year: 2014. Resistant to: ampicillin, sulfisoxazole. ...
         Tetracycline: Resistant
         ... (2k demos, shuffled per (seed, k))
         Now classify this isolate.
         Isolate: <test record>
         Tetracycline:
[assistant] ▌
```
**Verbalizers:** Resistant / Susceptible, each starting with a distinct token (checked automatically).
**Constrained decoding:** one forward pass. The next-token log-probs are restricted to the two verbalizer first tokens and renormalised, so the output is always a valid label and `predict_proba` is a true distribution.
**Parser:** `parse_label` (only for free-text generation): first line, strip punctuation, case-insensitive exact match, otherwise `None` (logged).

| Failure mode | Control | Verification |
|---|---|---|
| Majority-label bias (R/S imbalance; LM prior) | k demos per class | max predicted share, KL(pred‖gold), majority baseline |
| Position (recency) bias | demo and label order permuted per (seed, k) | share pred = last-demo label vs. gold share; seed std |
| Label leakage | TET/tetracycline-class columns excluded; temporal split; disjoint ids; neutral descriptions | `assert_no_target_leak`, `assert_disjoint` pass |

## Part 2A. Classifiers and contracts (25 pts)
One `Classifier` ABC. `deal.ensure` on `predict_proba` checks: finite, ≥ 0, row sums 1 ± 1e-6, shape (n, L). `deal.ensure` on `predict` checks that labels are in the label space. A deliberately broken classifier raises `ContractError` (notebook §3). (i) Prototype: class means of MiniLM embeddings of the isolate records. (ii) kNN: FAISS `IndexFlatIP` over the same demos. (iii) LLM: fixed demo block per run with constrained scoring.

## Part 2B. Shot-sweep results (25 pts)
| classifier | k=0 | k=1 | k=3 | k=5 | k=8 |
|---|---|---|---|---|---|
| prototype | .xx ± .xx [.xx, .xx] | | | | |
| kNN | | | | | |
| few-shot LLM | | | | | |

Macro-F1, mean ± std over 5 seeds [95% bootstrap CI], N = `<400>` test isolates (2020+). Majority baseline: `<.xx>`. Best configuration: `<clf, k>`. Per-class precision/recall at the best configuration: `<table>`. Resistant recall = `<.xx>`.
<3–4 sentences: trend with k, overlapping CIs, bias diagnostics, the temporal-shift caveat. No significance claims.>

## Part 2C. Reproducibility and quota (15 pts)
Container `<image@sha256>`; lockfile `<file>`; `verify_env.py` passes. Raw-file SHA-256 `<...>`. LLM `Qwen/Qwen2.5-0.5B-Instruct@<sha>`; embedder `all-MiniLM-L6-v2@<sha>`. CPU only (`torch.cuda.is_available()=False`), wall time `<min>`. 0 hosted-API requests; `<n>` local prompt tokens (`quota_log.jsonl`). All numbers are regenerated from `predictions.jsonl` by `summarize()`.

## GenAI audit (5 pts)
See `GenAI.pdf`.
