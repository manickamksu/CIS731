# Codebook: tetracycline phenotype of NARMS cecal *E. coli* isolates (MP 9 gold standard)

*Extends the PS 6 relevance protocol. Wording drafted with the Generative AI Kit; every gold label is assigned by the annotator. Exemplars come from my labeled objects.*

## 1. Object and fields

One **object** is one isolate (`item_id` = NARMS `SAMPLE_ID`) from the 2020+ test period. The annotator sees the following fields:

- `tet_mic_raw`: the raw tetracycline MIC, in µg/mL
- `profile`: the other drugs' results, host species and year

The annotator does **not** see the lab's `TET SIR` interpretation. These are the fields the annotator fills in:

| Field | Values |
|---|---|
| `label` | `resistant`, `susceptible` or `exclude` |
| `borderline` | `yes` or `no` |
| `quality_flag` | `none`, `panel_change`, `missing` or `other` |
| `notes` | Free text. Required whenever a boundary rule (B1–B5) was used. |

## 2. Label definitions (single fixed breakpoint, proposal §7)

| Label | Definition |
|---|---|
| `resistant` | Tetracycline MIC ≥ 16 µg/mL |
| `susceptible` | Tetracycline MIC ≤ 4 µg/mL |
| `exclude` | MIC = 8 (intermediate, excluded by the proposal), or the MIC cannot be read or placed on one side of the breakpoint |

## 3. Decision rules (apply in order; stop at the first that fires)

- **R1.** If `tet_mic_raw` is blank, unreadable or not a number, the label is `exclude` and `quality_flag` is `missing`.
- **R2.** If the MIC is uncensored, apply §2 directly. For example, `16` is `resistant`, `4` is `susceptible` and `8` is `exclude`.
- **R3.** A censored value ("≤x" or "<x", or ">x" or "≥x") is labeled only if every value it allows falls on one side of the breakpoint:
  - `<=4` and `<=2` are `susceptible`.
  - `>16`, `>32` and `>=16` are `resistant`.
  - A value that straddles the breakpoint is `exclude`, with a note. Examples are `<=8`, `<=16` and `>4`.
- **R4.** `borderline` is `yes` only for an **uncensored** MIC of exactly 4 or 16, one two-fold dilution from the I band. Everything else is `no`, including `<=4`.
- **R5.** `quality_flag`:
  - `panel_change` if a drug that is normally on the panel shows "Not tested" in `profile`.
  - `missing` if three or more drugs are "Not tested".
  - `none` otherwise.
- **R6.** Never use the other drugs' results, the host species or the year to decide `label`. They are model inputs and must not leak into the gold standard. They are used only for R5.

## 4. Boundary cases

- **B1.** The MIC is given with a unit or text, such as "16 ug/ml" or "16*". Read the number and apply R2/R3, and copy the raw text into `notes`.
- **B2.** The isolate looks unusual, for example resistant to many drugs or susceptible to everything. Label it by MIC only (R6). An unusual profile is not a reason to exclude.
- **B3.** The MIC is uncensored but not a standard two-fold dilution, such as 12. Label it `exclude` with `quality_flag` = `other` and a note. Do not round.
- **B4.** The same `SAMPLE_ID` appears twice. Label the first occurrence; the second is `exclude` with `quality_flag` = `other`.
- **B5.** You are unsure after applying R1–R6. Use `exclude` and explain in `notes`. Do not guess.

## 5. Adjudication rules (20-object overlap)

- **A1.** Both annotators agree: keep the label.
- **A2.** They disagree: both re-read the raw MIC and apply R1–R6 in order. The first rule that decides wins, and its rule id is recorded.
- **A3.** The disagreement comes from ambiguous wording in this codebook: fix the wording, add a boundary case, record it in the change log, and apply the fixed rule.
- **A4.** It is still unresolved: `exclude`, with a note. Every decision is stored in `gold_adjudicated.jsonl`, in the `adjudication`, `rule` and `note` fields.

## 6. Retrieval relevance rule (for P@k / Recall@k)

A retrieved 2013–2019 exemplar is **relevant** to a gold isolate when all three of these hold:

1. **Same phenotype.** The exemplar's lab tetracycline phenotype equals the gold isolate's adjudicated label.
2. **Same host species.** Both isolates come from the same host species.
3. **Similar resistance profile.** The Jaccard similarity of their sets of resistant (non-tetracycline) drugs is at least 0.5. Two isolates with no resistances count as similarity 1.

*Boundary:* two pan-susceptible isolates of the same phenotype and host are relevant.

## 7. Exemplars (fill in from your labeled objects)

| Rule | item_id | tet_mic_raw | label | why |
|---|---|---|---|---|
| R2 resistant | | | | |
| R2 susceptible | | | | |
| R3 censored, labeled | | | | |
| R3 censored, excluded (if any) | | | | |
| R4 borderline | | | | |
| R5 quality flag (if any) | | | | |

## 8. Change log

| Date | Change | Triggered by (item_id / rule) |
|---|---|---|
| | | |
