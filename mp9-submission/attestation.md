# Annotation attestation: MP 9

**Annotator:** Manickam Dhandapani
**Assigned partner (20-object overlap):** solo project
**Gold-set pathway:** **A — Third-party / historical ground-truth audit.** The labels being audited are the NARMS laboratory tetracycline MIC results and their `TET SIR` interpretation, which the scripted Obtain stage (MP 7 `load_raw`) loads from `CVM-NARMS-Cecal.xlsx`, a file kept unmodified (SHA-256 is in `results.json`).
**Codebook version used:** `codebook.md` at commit https://github.com/manickamksu/CIS731/tree/main/mp9-submission

## Annotation sessions (local time, America/Chicago)

| Session | Start (YYYY-MM-DD HH:MM) | Stop (YYYY-MM-DD HH:MM) | Minutes | Rows labeled |
|---|---|---|---|---|
| 1 | 2026-11-16 14:10| 2026-11-16 14:55|45 | 30|
| 2 | 2026-11-16 15:20| 2026-11-16 15:45|25 | 20|
| **Total** | |                         |70 | 50|

## Statement

I audited every historical label in `gold_raw.csv` myself against `codebook.md`. I re-derived each label blind from the raw MIC and only then compared it with the lab interpretation. No AI system assigned, suggested or verified any gold label. AI-generated labels appear only in `silver_pool.jsonl`, each marked `source=silver`.

The annotation took 70 minutes in total, which is within the 2-hour limit, and `gold_raw.csv` has 50 data rows, which is within the 50-row cap.

While labeling, I did not look at the lab's `TET SIR` interpretation. I compared it with my labels only afterwards, in notebook 3.

Signature / date: Manickam Dhandapani / 11/16/2026
