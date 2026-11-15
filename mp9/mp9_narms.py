"""
mp9_narms.py: MP 9 for the NARMS tetracycline project, built on mp7_core / mp7_narms.

  1. Gold standard   : pick <= 50 test-period isolates, write a BLANK annotation sheet, pre-check the
                       filled file against the 50-row gate, compare the human labels with the lab labels.
  2. IAA             : Cohen's kappa (95% bootstrap CI), percent agreement, confusion matrix, adjudication.
  3. Weak supervision: labeling functions over the other drugs' S/I/R results + the MP 7 classifier,
                       weighted vote, silver records marked source="silver", silver-vs-gold agreement.
  4. Retrieval-augmented classification: per-query top-k exemplars from a FAISS index over the
                       2013-2019 pool, vs. the MP 7 fixed-demonstration baseline at the SAME number of
                       demonstrations; P@k / Recall@k; paired bootstrap CIs.
  5. Cohort convolution: year x drug resistance-rate matrix, smoothing [0.25, 0.5, 0.25] and
                       first-difference [-1, 1] kernels, drift with within-year bootstrap CIs.
  6. CIS 731         : learned 1-D convolutional kernel (PyTorch, CPU) vs. fixed-kernel forecasts.

Nothing in this file assigns a gold label. Gold labels come only from the annotator's file.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from typing import Callable, Sequence

import numpy as np
import pandas as pd

import mp7_core as M
import mp7_narms as N

GOLD_CAP = 50                      # CI/CD gate: more than 50 data rows -> zero for the assignment
GOLD_LABELS = ("resistant", "susceptible", "exclude")   # "exclude" = annotator judged the row unusable
SEED = 20261103

# Drug code -> antimicrobial class (used by the multidrug-resistance labeling function and reports)
DRUG_CLASS = {
    "AMC": "beta-lactam", "AMP": "beta-lactam", "AXO": "beta-lactam", "FOX": "beta-lactam",
    "TIO": "beta-lactam", "CEP": "beta-lactam", "MER": "beta-lactam",
    "AZI": "macrolide", "ERY": "macrolide", "CHL": "phenicol",
    "CIP": "quinolone", "NAL": "quinolone",
    "COT": "folate-pathway", "FIS": "folate-pathway", "SMX": "folate-pathway", "SUF": "folate-pathway",
    "GEN": "aminoglycoside", "KAN": "aminoglycoside", "STR": "aminoglycoside", "COL": "polymyxin",
}


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


# =====================================================================================
# 1. Gold standard: candidate selection, blank annotation sheet, pre-check, lab audit
# =====================================================================================
def parse_mic(v) -> float:
    """'<=4' -> 4.0, '>32' -> 32.0, 16 -> 16.0, blank -> nan. (The qualifier is kept by `mic_qualifier`.)"""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return np.nan
    m = re.search(r"(\d+(?:\.\d+)?)", str(v))
    return float(m.group(1)) if m else np.nan


def mic_qualifier(v) -> str:
    m = re.match(r"\s*([<>]=?)", str(v))
    return m.group(1) if m else ""


def is_borderline(mic_values: pd.Series, sign_values: pd.Series | None = None,
                  borderline_mics=(4.0, 16.0)) -> pd.Series:
    """MIC exactly one two-fold dilution from the I band (4 or 16 for S<=4 / I=8 / R>=16).
    A censored value such as '<=4' or '>16' is NOT borderline (its true MIC may be far away).
    If your file stores the qualifier in its own column (check with inspect_columns), pass it."""
    q = (sign_values.astype(str).str.strip().replace({"nan": "", "=": ""}) if sign_values is not None
         else mic_values.map(mic_qualifier))
    return mic_values.map(parse_mic).isin(borderline_mics) & q.isin(["", "None"])


def select_gold_candidates(test_df: pd.DataFrame, cfg: N.NarmsConfig, n: int = GOLD_CAP,
                           mic_col: str = "TET", sign_col: str | None = None,
                           borderline_frac: float = 0.3, seed: int = SEED) -> list[str]:
    """Deterministic sample of n test-period isolate ids, proportional by host species, with about
    `borderline_frac` drawn from borderline-MIC isolates (see is_borderline). Stratifies on host and
    MIC, never on the label."""
    assert n <= GOLD_CAP, f"never select more than {GOLD_CAP} gold objects"
    d = test_df.sort_values(cfg.id_col).copy()
    d["_border"] = is_borderline(d[mic_col], d[sign_col] if sign_col else None)
    rng = np.random.default_rng(seed)
    hosts = d[cfg.source_col].astype(str)
    share = hosts.value_counts(normalize=True)
    quota = {h: max(1, int(round(n * s))) for h, s in share.items()}
    while sum(quota.values()) > n:                       # rounding fix-up, largest stratum gives way
        quota[max(quota, key=quota.get)] -= 1
    while sum(quota.values()) < n:
        quota[max(share.index, key=lambda h: share[h] * n - quota[h])] += 1
    chosen = []
    for h, q in sorted(quota.items()):
        g = d[hosts == h]
        nb = min(int(round(q * borderline_frac)), int(g["_border"].sum()))
        b_ids = g[g["_border"]][cfg.id_col].tolist()
        o_ids = g[~g["_border"]][cfg.id_col].tolist()
        bperm, operm = rng.permutation(len(b_ids)), rng.permutation(len(o_ids))
        n_o = min(q - nb, len(o_ids))
        nb = min(q - n_o, len(b_ids))           # top up from borderline if a stratum runs short
        chosen += [str(b_ids[i]) for i in bperm[:nb]] + [str(o_ids[i]) for i in operm[:n_o]]
    if len(chosen) < n:
        print(f"WARNING: only {len(chosen)} candidates available (asked for {n})")
    return sorted(chosen)[:n]


def make_annotation_sheet(df: pd.DataFrame, ids: Sequence[str], cfg: N.NarmsConfig, drugs: Sequence[str],
                          mic_col: str = "TET") -> pd.DataFrame:
    """BLANK sheet for the human annotator. Shows the raw tetracycline MIC and the other drugs'
    results; the lab's own interpretation (`TET SIR`) and the label column are NOT filled in."""
    d = df.set_index(df[cfg.id_col].astype(str)).loc[list(ids)]
    return pd.DataFrame({
        "item_id": list(ids),
        "host_species": d[cfg.source_col].astype(str).values,
        "production": d["_prod"].values if "_prod" in d else "",
        "year": d[cfg.year_col].astype(int).values,
        "tet_mic_raw": d[mic_col].astype(str).values,
        "profile": [N.serialize(r, cfg, list(drugs)) for _, r in d.iterrows()],
        "label": "",              # resistant / susceptible / exclude   (filled by YOU)
        "borderline": "",         # yes / no                            (filled by YOU)
        "quality_flag": "",       # none / panel_change / missing / other (filled by YOU)
        "notes": "",
        "annotator": "",
        "source": "gold",
    })


def read_labels_file(path: str) -> pd.DataFrame:
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True, dtype={"item_id": str})
    return pd.read_csv(path, dtype={"item_id": str}, keep_default_na=False)


def precheck_gold_raw(path: str, cap: int = GOLD_CAP) -> dict:
    """Local pre-check (run the course's validate_annotation_submission as well). Raises on failure."""
    g = read_labels_file(path)
    errs = []
    if len(g) > cap:
        errs.append(f"{len(g)} data rows > cap {cap}: the CI gate would give a zero")
    for c in ("item_id", "label", "borderline", "source"):
        if c not in g:
            errs.append(f"missing column {c!r}")
    if not errs:
        lab = g["label"].astype(str).str.strip().str.lower()
        if (lab == "").any():
            errs.append(f"{int((lab == '').sum())} rows have no label")
        bad = sorted(set(lab) - set(GOLD_LABELS) - {""})
        if bad:
            errs.append(f"labels outside {GOLD_LABELS}: {bad}")
        if g["item_id"].duplicated().any():
            errs.append("duplicate item_id")
        if (g["source"].astype(str) != "gold").any():
            errs.append("every gold row must have source=gold")
    if errs:
        raise ValueError("gold_raw pre-check FAILED:\n  - " + "\n  - ".join(errs))
    return {"rows": len(g), "cap": cap, "label_counts": g["label"].str.lower().value_counts().to_dict(),
            "sha256": M.sha256_file(path)}


def audit_vs_lab(gold: pd.DataFrame, df: pd.DataFrame, cfg: N.NarmsConfig) -> pd.DataFrame:
    """Human label vs. the lab's TET SIR interpretation (run AFTER annotation). Disagreements are
    findings for the codebook / label-reliability section, not errors to overwrite."""
    lab = df.set_index(df[cfg.id_col].astype(str))["label"]
    g = gold[gold["label"].str.lower() != "exclude"].copy()
    g["lab_label"] = g["item_id"].map(lab)
    g["agree"] = g["label"].str.lower() == g["lab_label"]
    return g[["item_id", "label", "lab_label", "agree", "borderline"]]


def gold_examples(df: pd.DataFrame, cfg: N.NarmsConfig, drugs, adjudicated_path: str) -> list[M.Example]:
    """Adjudicated gold isolates as MP 7 Example records (label = adjudicated HUMAN label)."""
    adj = pd.read_json(adjudicated_path, lines=True, dtype={"item_id": str})
    rows = df.set_index(df[cfg.id_col].astype(str))
    return [M.Example(item_id=r.item_id, text=N.serialize(rows.loc[r.item_id], cfg, list(drugs)),
                      label=r.label, doc_id=r.item_id) for r in adj.itertuples()]


# =====================================================================================
# 2. Inter-annotator agreement and adjudication
# =====================================================================================
def kappa_report(a: dict, b: dict, ids: Sequence[str], n_boot: int = 10000, seed: int = SEED) -> dict:
    """Cohen's kappa with a percentile bootstrap CI over the overlap items. Resamples where kappa is
    undefined (only one label present in both raters) are skipped and counted."""
    from sklearn.metrics import cohen_kappa_score, confusion_matrix
    ids = [str(i) for i in ids]
    missing = [i for i in ids if i not in a or i not in b]
    assert not missing, f"overlap ids without labels from both annotators: {missing[:5]}"
    ya = np.array([str(a[i]).lower() for i in ids])
    yb = np.array([str(b[i]).lower() for i in ids])
    labels = [l for l in GOLD_LABELS if l in set(ya) | set(yb)]

    def kap(x, y):
        if len(set(x) | set(y)) < 2:
            return np.nan
        return float(cohen_kappa_score(x, y, labels=labels))

    k = kap(ya, yb)
    rng = np.random.default_rng(seed)
    n = len(ids)
    boots = np.array([kap(ya[ix], yb[ix]) for ix in (rng.integers(0, n, n) for _ in range(n_boot))])
    ok = boots[~np.isnan(boots)]
    lo, hi = (np.percentile(ok, [2.5, 97.5]) if len(ok) else (np.nan, np.nan))
    cm = pd.DataFrame(confusion_matrix(ya, yb, labels=labels),
                      index=[f"A:{l}" for l in labels], columns=[f"B:{l}" for l in labels])
    return {"n": n, "kappa": k, "kappa_ci95": [float(lo), float(hi)],
            "percent_agreement": float(np.mean(ya == yb)),
            "n_boot": n_boot, "n_boot_undefined": int(np.isnan(boots).sum()),
            "labels": labels, "confusion_matrix": cm,
            "disagreements": [i for i, x, y in zip(ids, ya, yb) if x != y]}


def adjudicate(mine: pd.DataFrame, partner: pd.DataFrame, overlap_ids: Sequence[str],
               decisions: dict[str, dict], out_path: str = "gold_adjudicated.jsonl") -> pd.DataFrame:
    """Writes gold_adjudicated.jsonl. Every overlap disagreement MUST have a human decision:
    decisions[item_id] = {"label": ..., "rule": "codebook rule id", "note": "..."}.
    Rows labeled 'exclude' are dropped from the adjudicated set (and counted)."""
    a = mine.set_index("item_id")["label"].str.lower().to_dict()
    b = partner.set_index("item_id")["label"].str.lower().to_dict()
    ov = {str(i) for i in overlap_ids}
    rows, todo = [], []
    for r in mine.itertuples():
        i = str(r.item_id)
        rec = {"item_id": i, "label_a": a[i], "label_b": b.get(i) if i in ov else None,
               "borderline": str(r.borderline).lower(), "source": "gold"}
        if i in ov and a[i] != b.get(i):
            if i not in decisions:
                todo.append(i)
                continue
            rec.update(label=decisions[i]["label"].lower(), adjudication="discussed",
                       rule=decisions[i].get("rule", ""), note=decisions[i].get("note", ""))
        else:
            rec.update(label=a[i], adjudication="agreement" if i in ov else "single-annotator",
                       rule="", note="")
        rows.append(rec)
    if todo:
        raise ValueError(f"{len(todo)} disagreements still need a decision: {todo}")
    out = pd.DataFrame(rows)
    n_ex = int((out["label"] == "exclude").sum())
    out = out[out["label"] != "exclude"]
    assert set(out["label"]) <= {"resistant", "susceptible"}
    out.to_json(out_path, orient="records", lines=True)
    print(f"wrote {len(out)} adjudicated rows to {out_path} ({n_ex} excluded)")
    return out


# =====================================================================================
# 3. Weak supervision
# =====================================================================================
ABSTAIN = None


def lf_frame(df: pd.DataFrame, cfg: N.NarmsConfig, drugs: Sequence[str]) -> pd.DataFrame:
    """One row per isolate: drug code -> 'R'/'S'/'I'/None, plus host. Tetracycline is NOT included."""
    f = pd.DataFrame({N._drug_code(d): df[d].map(N._rsi) for d in drugs})
    f = f.astype(object).where(f.notna(), None)
    f.index = df[cfg.id_col].astype(str).values
    f["host"] = df[cfg.source_col].astype(str).str.strip().str.lower().values
    return f


def make_default_lfs(dev_frame: pd.DataFrame, dev_labels: pd.Series, mp7_preds: dict | None = None,
                     prior_hi: float = 0.65, prior_lo: float = 0.35) -> dict[str, Callable]:
    """Labeling functions. Each takes a row of `lf_frame` and returns a label or ABSTAIN.
    The co-resistance rules are hypotheses; their weights are estimated on the dev split, so a weak
    rule ends up with a small weight. Host priors are estimated on the dev split only."""
    prior = dev_labels.eq("resistant").groupby(dev_frame["host"]).mean().to_dict()
    codes = [c for c in dev_frame.columns if c != "host"]

    def R(row, c):
        return row.get(c) == "R"

    lfs = {}
    for c in ("FIS", "AMP", "CHL", "COT"):
        if c in codes:
            lfs[f"{c}_R"] = (lambda c: lambda row: "resistant" if R(row, c) else ABSTAIN)(c)

    def mdr3(row):
        classes = {DRUG_CLASS.get(c, c) for c in codes if R(row, c)}
        return "resistant" if len(classes) >= 3 else ABSTAIN

    def pan_s(row, min_tested=5):
        vals = [row.get(c) for c in codes if row.get(c) is not None]
        return "susceptible" if len(vals) >= min_tested and all(v == "S" for v in vals) else ABSTAIN

    def host_prior(row):
        p = prior.get(row["host"])
        if p is None:
            return ABSTAIN
        return "resistant" if p >= prior_hi else "susceptible" if p <= prior_lo else ABSTAIN

    lfs.update({"MDR3": mdr3, "pan_S": pan_s, "host_prior": host_prior})
    if mp7_preds is not None:
        lfs["mp7_classifier"] = lambda row: mp7_preds.get(row.name, ABSTAIN)
    return lfs


def apply_lfs(frame: pd.DataFrame, lfs: dict[str, Callable]) -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series([fn(r) for _, r in frame.iterrows()], dtype=object)
                         for name, fn in lfs.items()}).set_axis(frame.index)


def lf_summary(L: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
    fired = L.notna()
    n_fired = fired.sum(axis=1)
    rows = []
    for c in L.columns:
        f = fired[c]
        others = L.drop(columns=c)
        conflict = [any(pd.notna(v) and v != L.at[i, c] for v in others.loc[i]) for i in L.index[f]]
        r = {"lf": c, "coverage": f.mean(), "overlap": (f & (n_fired > 1)).mean(),
             "conflict": float(np.sum(conflict)) / len(L)}
        if y is not None:
            r["accuracy"] = float((L.loc[f, c] == y[f]).mean()) if f.any() else np.nan
            r["n_fired"] = int(f.sum())
        rows.append(r)
    return pd.DataFrame(rows).round(4)


def fit_lf_weights(L_dev: pd.DataFrame, y_dev: pd.Series, smoothing: float = 1.0) -> dict[str, float]:
    """Weight = log-odds of the LF's accuracy on the dev split (Laplace-smoothed), floored at 0, so
    an LF no better than chance gets no vote. Never fit on the gold set."""
    w = {}
    for c in L_dev.columns:
        f = L_dev[c].notna()
        correct = float((L_dev.loc[f, c] == y_dev[f]).sum())
        acc = (correct + smoothing) / (f.sum() + 2 * smoothing)
        w[c] = max(0.0, float(np.log(acc / (1 - acc))))
    return w


def weighted_vote(L: pd.DataFrame, weights: dict[str, float],
                  labels=M.LABELS) -> pd.DataFrame:
    """Score per label = sum of weights of LFs voting for it. No vote or an exact tie -> abstain."""
    S = pd.DataFrame({l: sum((L[c] == l).astype(float) * weights.get(c, 0.0) for c in L.columns)
                      for l in labels}, index=L.index)
    top = S.max(axis=1)
    n_top = S.eq(top, axis=0).sum(axis=1)
    lab = S.idxmax(axis=1).where((top > 0) & (n_top == 1), None)
    tot = S.sum(axis=1).replace(0, np.nan)
    return pd.DataFrame({"silver_label": lab, "confidence": (top / tot).round(4),
                         "n_votes": L.notna().sum(axis=1)}, index=L.index)


def silver_records(votes: pd.DataFrame, L: pd.DataFrame, path: str = "silver_pool.jsonl") -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for i, r in votes.iterrows():
            fh.write(json.dumps({"item_id": i, "label": r.silver_label, "source": "silver",
                                 "confidence": None if pd.isna(r.confidence) else float(r.confidence),
                                 "lf_votes": {c: L.at[i, c] for c in L.columns}}) + "\n")
            n += 1
    return n


def agreement(y_true: pd.Series, y_pred: pd.Series, n_boot: int = 2000, seed: int = SEED) -> dict:
    """Silver-vs-gold (or silver-vs-lab) agreement on items where silver did not abstain."""
    from sklearn.metrics import cohen_kappa_score, confusion_matrix
    m = y_pred.notna()
    yt, yp = y_true[m].astype(str).values, y_pred[m].astype(str).values
    out = {"n": int(len(y_true)), "coverage": float(m.mean()), "n_covered": int(m.sum())}
    if m.sum() == 0:
        return out
    acc = float(np.mean(yt == yp))
    kap = float(cohen_kappa_score(yt, yp, labels=list(M.LABELS))) if len(set(yt) | set(yp)) > 1 else np.nan
    rng = np.random.default_rng(seed)
    n = len(yt)
    bs = [np.mean(yt[ix] == yp[ix]) for ix in (rng.integers(0, n, n) for _ in range(n_boot))]
    out.update(accuracy=acc, accuracy_ci95=[float(x) for x in np.percentile(bs, [2.5, 97.5])], kappa=kap,
               confusion_matrix=pd.DataFrame(confusion_matrix(yt, yp, labels=list(M.LABELS)),
                                             index=[f"true:{l}" for l in M.LABELS],
                                             columns=[f"silver:{l}" for l in M.LABELS]))
    return out


# =====================================================================================
# 4. Retrieval-augmented classification
# =====================================================================================
def sample_demos_total(pool: Sequence[M.Example], k: int, seed: int, labels=M.LABELS) -> list[M.Example]:
    """Fixed demonstrations with k IN TOTAL (to match top-k retrieval), as class-balanced as possible:
    floor(k/L) per label, the remainder going to labels chosen by the seed. Shuffled by (seed, k)."""
    rng = np.random.default_rng([seed, 31337])
    per = {l: k // len(labels) for l in labels}
    for i in rng.permutation(len(labels))[: k % len(labels)]:
        per[labels[i]] += 1
    out = []
    for li, l in enumerate(labels):
        cands = sorted([e for e in pool if e.label == l], key=lambda e: e.item_id)
        perm = np.random.default_rng([seed, 101 + li]).permutation(len(cands))
        out += [cands[j] for j in perm[: per[l]]]
    order = np.random.default_rng([seed, k]).permutation(len(out))
    return [out[i] for i in order]


class ExemplarIndex:
    """FAISS IndexFlatIP over the demonstration pool (same embedder as MP 7). Items are added in
    item_id order; results are re-sorted by (similarity desc, item_id asc) so ties between the many
    identical resistance profiles are broken deterministically."""

    def __init__(self, embedder, items: Sequence[M.Example], by_label: bool = False, margin: int = 20):
        import faiss
        self.embedder, self.margin = embedder, margin
        items = sorted(items, key=lambda e: e.item_id)
        groups = {l: [e for e in items if e.label == l] for l in M.LABELS} if by_label else {None: items}
        self.groups = {}
        for g, es in groups.items():
            X = np.ascontiguousarray(M._l2norm(embedder.encode([e.text for e in es])))
            idx = faiss.IndexFlatIP(X.shape[1])
            idx.add(X)
            self.groups[g] = (idx, es)

    def search(self, text: str, n: int, group=None, exclude: set | None = None):
        idx, es = self.groups[group]
        q = np.ascontiguousarray(M._l2norm(self.embedder.encode([text])))
        m = min(idx.ntotal, n * self.margin + len(exclude or ()))
        sims, ix = idx.search(q, m)
        hits = [(es[i], float(s)) for s, i in zip(sims[0], ix[0])
                if i >= 0 and not (exclude and es[i].item_id in exclude)]
        hits.sort(key=lambda h: (-round(h[1], 6), h[0].item_id))
        return hits[:n]


class RetrievalFewShotLLMClassifier(M.FewShotLLMClassifier):
    """MP 7 few-shot LLM, but the demonstrations are retrieved PER QUERY (top-k from the FAISS index)
    instead of fixed. Same template, verbalizers, constrained scoring and label-list permutation, so
    retrieval is the only change. `balanced=True` takes the top ceil/floor(k/2) per label instead."""
    name = "retrieval_llm"

    def __init__(self, backend, index: ExemplarIndex, k: int, balanced: bool = False,
                 most_similar_last: bool = True, **kw):
        kw["contextual_calibration"] = False      # would cost 3 extra passes per query
        super().__init__(backend, **kw)
        self.index, self.k, self.balanced, self.most_similar_last = index, k, balanced, most_similar_last
        self.name = "retrieval_llm_balanced" if balanced else "retrieval_llm"
        self.retrieved: list[list[tuple[str, float]]] = []

    def retrieve(self, text: str, exclude: set | None = None):
        if not self.balanced:
            hits = self.index.search(text, self.k, None, exclude)
        else:
            per = {l: self.index.search(text, self.k, l, exclude) for l in self.labels}
            base = {l: self.k // len(self.labels) for l in self.labels}
            # remainder goes to the label whose best exemplar is most similar to the query
            for l in sorted(self.labels, key=lambda l: -(per[l][0][1] if per[l] else -9))[: self.k % len(self.labels)]:
                base[l] += 1
            hits = sorted([h for l in self.labels for h in per[l][: base[l]]],
                          key=lambda h: (-round(h[1], 6), h[0].item_id))
        return hits

    def _fit(self, demos):
        self.retrieved = []                        # demos are ignored: retrieval happens per query

    def _proba(self, texts):
        rows = []
        for t in texts:
            hits = self.retrieve(t)
            ordered = hits[::-1] if self.most_similar_last else hits
            self.demos = [h[0] for h in ordered]
            self.prefix, self.suffix_tmpl = self.build_prompt_parts()
            rest = self.backend.truncate(t, self.max_query_tokens) + self.suffix_tmpl
            lp = self.backend.label_logprobs(self.prefix, rest, self.verbalizers)
            rows.append(M._softmax(lp))
            self.retrieved.append([(h[0].item_id, round(h[1], 6)) for h in hits])
        return np.vstack(rows)


def resistant_recall(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    m = y_true == "resistant"
    return float(np.mean(y_pred[m] == "resistant")) if m.any() else np.nan


def run_rac_sweep(backend, index: ExemplarIndex, index_bal: ExemplarIndex | None, pool, ev,
                  ks=(3, 5, 10), seeds=(0, 1, 2), out_path="predictions_mp9.jsonl", eval_sha256="",
                  log=print) -> list[dict]:
    """Writes one line per (classifier, k, seed, item) in the MP 7 predictions.jsonl format,
    plus the demonstration ids used for that item. Classifiers: fixed_llm (MP 7 baseline with k demos
    in total), retrieval_llm, and retrieval_llm_balanced (if index_bal is given)."""
    texts, gold = [e.text for e in ev], [e.label for e in ev]
    label_of = {e.item_id: e.label for e in pool}
    runs = []
    with open(out_path, "w", encoding="utf-8") as fh:
        for k in ks:
            for seed in seeds:
                cands = [("fixed_llm", None)] + [("retrieval_llm", index)] + \
                        ([("retrieval_llm_balanced", index_bal)] if index_bal is not None else [])
                for cname, idx in cands:
                    t0 = time_now()
                    if idx is None:
                        demos = sample_demos_total(pool, k, seed)
                        clf = M.FewShotLLMClassifier(backend, seed=seed).fit(demos)
                        P = clf.predict_proba(texts)
                        demo_ids = [[d.item_id for d in demos]] * len(ev)
                    else:
                        clf = RetrievalFewShotLLMClassifier(backend, idx, k, balanced=(cname.endswith("balanced")),
                                                            seed=seed).fit([])
                        P = clf.predict_proba(texts)
                        demo_ids = [[i for i, _ in r] for r in clf.retrieved]
                    preds = [M.LABELS[i] for i in P.argmax(1)]
                    for e, p, row, dids in zip(ev, preds, P, demo_ids):
                        fh.write(json.dumps({"run_id": f"{cname}|k={k}|seed={seed}", "classifier": cname,
                                             "k": k, "seed": seed, "item_id": e.item_id, "gold": e.label,
                                             "pred": p, "proba": {l: round(float(v), 8) for l, v in zip(M.LABELS, row)},
                                             "demo_ids": dids, "model_id": backend.model_id,
                                             "revision": backend.revision, "eval_sha256": eval_sha256,
                                             "grading_eligible": False}) + "\n")
                    demo_labels = Counter(lbl for dids in demo_ids for lbl in
                                          [label_of.get(i) for i in dids] if lbl)
                    diag = {"classifier": cname, "k": k, "seed": seed, "seconds": round(time_now() - t0, 2),
                            "macro_f1": M.dev_macro_f1(gold, preds), "resistant_recall": resistant_recall(gold, preds),
                            "demo_label_share_resistant": demo_labels["resistant"] / max(1, sum(demo_labels.values())),
                            **M.bias_diagnostics(preds, gold, [])}
                    runs.append(diag)
                    log(f"{cname:24s} k={k:2d} seed={seed} macroF1={diag['macro_f1']:.3f} "
                        f"R-recall={diag['resistant_recall']:.3f} ({diag['seconds']:.0f}s)")
    return runs


def time_now():
    import time
    return time.time()


def metric_table(pred_path: str, n_boot: int = 2000) -> pd.DataFrame:
    """Macro-F1 and resistant-class recall per (classifier, k): seed mean +/- sd and a 95% bootstrap CI
    over gold items (paired across seeds), all re-derived from the predictions file."""
    df = pd.read_json(pred_path, lines=True, dtype={"item_id": str})
    rows = []
    for (c, k), g in df.groupby(["classifier", "k"]):
        seeds = sorted(g.seed.unique())
        items = sorted(g.item_id.unique())
        y = g[g.seed == seeds[0]].set_index("item_id").loc[items, "gold"].tolist()
        P = [g[g.seed == s].set_index("item_id").loc[items, "pred"].tolist() for s in seeds]
        for mname, fn in (("macro_f1", M.dev_macro_f1), ("resistant_recall", resistant_recall)):
            vals = [fn(y, p) for p in P]
            lo, hi = M.bootstrap_ci(y, P, fn, n_boot=n_boot)
            rows.append({"classifier": c, "k": int(k), "metric": mname, "mean": np.mean(vals),
                         "sd": np.std(vals, ddof=1) if len(vals) > 1 else 0.0, "ci95_lo": lo, "ci95_hi": hi,
                         "n_items": len(y), "n_seeds": len(seeds)})
    return pd.DataFrame(rows)


def paired_bootstrap_diff(pred_path: str, a: str, b: str, k: int, metric=M.dev_macro_f1,
                          n_boot: int = 5000, seed: int = SEED) -> dict:
    """metric(a) - metric(b) at the same k, same items, same seeds. Each replicate resamples gold
    items once and scores both systems (averaged over seeds) on that same resample."""
    df = pd.read_json(pred_path, lines=True, dtype={"item_id": str})
    df = df[df.k == k]
    items = sorted(df.item_id.unique())
    seeds = sorted(set(df[df.classifier == a].seed) & set(df[df.classifier == b].seed))

    def mat(c):
        g = df[df.classifier == c]
        return [np.array(g[g.seed == s].set_index("item_id").loc[items, "pred"]) for s in seeds]

    y = np.array(df[(df.classifier == a) & (df.seed == seeds[0])].set_index("item_id").loc[items, "gold"])
    A, B = mat(a), mat(b)

    def stat(ix):
        return (np.mean([metric(y[ix].tolist(), p[ix].tolist()) for p in A])
                - np.mean([metric(y[ix].tolist(), p[ix].tolist()) for p in B]))

    rng = np.random.default_rng(seed)
    n = len(items)
    full = stat(np.arange(n))
    bs = np.array([stat(rng.integers(0, n, n)) for _ in range(n_boot)])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return {"a": a, "b": b, "k": k, "diff": float(full), "ci95": [float(lo), float(hi)],
            "share_boot_diff_gt0": float(np.mean(bs > 0)), "n_items": n, "seeds": seeds}


# ---- retrieval quality: P@k and Recall@k --------------------------------------------
def ranked_neighbors(index: ExemplarIndex, queries: Sequence[M.Example], kmax: int = 10) -> dict[str, list[str]]:
    return {q.item_id: [h[0].item_id for h in index.search(q.text, kmax)] for q in queries}


def jaccard(a: set, b: set) -> float:
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def rule_relevance(frame_q: pd.DataFrame, frame_pool: pd.DataFrame, gold_labels: dict, pool_labels: dict,
                   min_jaccard: float = 0.5):
    """Codebook relevance rule (edit to match your PS 6 protocol): a pool isolate is relevant to a
    query isolate if it has the SAME tetracycline phenotype (query: adjudicated gold label; pool: lab
    label), the SAME host species, and resistant-drug sets with Jaccard >= min_jaccard."""
    codes = [c for c in frame_q.columns if c != "host"]
    Rq = {i: {c for c in codes if frame_q.at[i, c] == "R"} for i in frame_q.index}
    pool_R = [{c for c, v in zip(codes, row) if v == "R"} for row in frame_pool[codes].itertuples(index=False)]
    pool_ids = list(frame_pool.index)
    pool_lab = np.array([pool_labels[i] for i in pool_ids])
    pool_host = frame_pool["host"].values

    def is_rel(q, c):
        return (gold_labels[q] == pool_labels[c] and frame_q.at[q, "host"] == frame_pool.at[c, "host"]
                and jaccard(Rq[q], {x for x in codes if frame_pool.at[c, x] == "R"}) >= min_jaccard)

    def n_relevant_total(q):
        m = (pool_lab == gold_labels[q]) & (pool_host == frame_q.at[q, "host"])
        return int(sum(jaccard(Rq[q], pool_R[j]) >= min_jaccard for j in np.flatnonzero(m)))

    return is_rel, n_relevant_total


def retrieval_metrics(ranked: dict[str, list[str]], rel: dict[str, dict[str, int]], ks=(3, 5, 10),
                      n_relevant_total: dict[str, int] | None = None, n_boot: int = 5000,
                      seed: int = SEED) -> pd.DataFrame:
    """P@k = relevant in top-k / k.  Recall@k (pooled, TREC-style) = relevant in top-k / relevant in the
    judged top-kmax pool; queries with no relevant item in the pool are undefined and counted.
    Recall@k (exact) uses the total number of relevant pool items, when given.
    95% CIs: percentile bootstrap over queries."""
    kmax = max(ks)
    rng = np.random.default_rng(seed)
    qs = sorted(ranked)
    rows = []
    for k in ks:
        p = np.array([sum(rel[q].get(c, 0) for c in ranked[q][:k]) / k for q in qs])
        pool_rel = np.array([sum(rel[q].get(c, 0) for c in ranked[q][:kmax]) for q in qs], float)
        hit = np.array([sum(rel[q].get(c, 0) for c in ranked[q][:k]) for q in qs], float)
        rp = np.where(pool_rel > 0, hit / np.where(pool_rel > 0, pool_rel, 1), np.nan)
        series = {"P@k": p, "Recall@k (pooled)": rp}
        if n_relevant_total is not None:
            tot = np.array([n_relevant_total[q] for q in qs], float)
            series["Recall@k (exact)"] = np.where(tot > 0, hit / np.where(tot > 0, tot, 1), np.nan)
        for name, v in series.items():
            ok = v[~np.isnan(v)]
            bs = [np.mean(ok[rng.integers(0, len(ok), len(ok))]) for _ in range(n_boot)] if len(ok) else [np.nan]
            rows.append({"metric": name, "k": k, "mean": float(np.mean(ok)) if len(ok) else np.nan,
                         "ci95_lo": float(np.percentile(bs, 2.5)), "ci95_hi": float(np.percentile(bs, 97.5)),
                         "n_queries": int(len(ok)), "n_undefined": int(np.isnan(v).sum())})
    return pd.DataFrame(rows)


def relevance_sheet(ranked: dict[str, list[str]], frame_pool: pd.DataFrame, pool_text: dict[str, str],
                    query_text: dict[str, str], kmax: int = 10) -> pd.DataFrame:
    """BLANK sheet for human relevance judgments of the top-kmax neighbours (relevant = 1/0)."""
    return pd.DataFrame([{"query_id": q, "rank": r + 1, "cand_id": c, "query_profile": query_text[q],
                          "cand_profile": pool_text[c], "relevant": ""}
                         for q, cs in ranked.items() for r, c in enumerate(cs[:kmax])])


def read_relevance(path: str) -> dict[str, dict[str, int]]:
    d = pd.read_csv(path, dtype={"query_id": str, "cand_id": str})
    out: dict[str, dict[str, int]] = {}
    for r in d.itertuples():
        out.setdefault(r.query_id, {})[r.cand_id] = int(r.relevant)
    return out


# =====================================================================================
# 5. Cohort convolution along collection year
# =====================================================================================
SMOOTH = np.array([0.25, 0.5, 0.25])
DIFF = np.array([-1.0, 1.0])


def apply_kernel(F: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """'valid' kernel application along axis 0 (the cohort axis), CROSS-CORRELATION convention
    (the one PyTorch Conv1d uses): out[t] = sum_j kernel[j] * F[t + j].
    With [-1, 1] this gives out[t] = F[t+1] - F[t] (positive = resistance went UP).
    Note: np.convolve flips the kernel and would give F[t] - F[t+1] instead."""
    F = np.asarray(F, float)
    k = np.asarray(kernel, float)
    T = F.shape[0] - len(k) + 1
    return np.stack([sum(k[j] * F[t + j] for j in range(len(k))) for t in range(T)])


def year_value_arrays(df: pd.DataFrame, cfg: N.NarmsConfig, drug_cols: Sequence[str]):
    """year -> (n_isolates x n_drugs) array of 1 (R), 0 (S or I), nan (not tested)."""
    V = np.column_stack([df[c].map(N._rsi).map({"R": 1.0, "S": 0.0, "I": 0.0}).astype(float).values
                         for c in drug_cols])
    years = df[cfg.year_col].astype(int).values
    return {int(y): V[years == y] for y in sorted(set(years))}


def rate_matrix(arrays: dict[int, np.ndarray]) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.stack([np.nanmean(arrays[y], axis=0) for y in sorted(arrays)])


def drift_analysis(df: pd.DataFrame, cfg: N.NarmsConfig, drug_cols: Sequence[str], n_boot: int = 1000,
                   seed: int = SEED, policy_transition=(2016, 2017)) -> dict:
    """Per-year resistance-rate vectors F (years x drugs; tetracycline may be included, this part is
    descriptive). Smoothed S = F * [0.25, 0.5, 0.25] (valid: loses the first and last year);
    drift D = S * [-1, 1]; raw drift = F * [-1, 1]. Drift magnitude per transition = mean absolute
    change across drugs (percentage points). CIs: resample isolates with replacement WITHIN each year."""
    arrays = year_value_arrays(df, cfg, drug_cols)
    years = sorted(arrays)
    F = rate_matrix(arrays)
    keep = ~np.isnan(F).any(axis=0)                     # a drug must be tested in every year
    dropped = [d for d, k in zip(drug_cols, keep) if not k]
    drugs = [d for d, k in zip(drug_cols, keep) if k]
    arrays = {y: a[:, keep] for y, a in arrays.items()}
    F = F[:, keep]
    assert np.allclose(apply_kernel(F, DIFF), np.diff(F, axis=0)), "kernel sign convention check"

    def stats(F):
        S = apply_kernel(F, SMOOTH)
        Ds, Dr = apply_kernel(S, DIFF), apply_kernel(F, DIFF)
        return S, Ds, Dr, np.abs(Ds).mean(axis=1), np.abs(Dr).mean(axis=1)

    S, Ds, Dr, mag_s, mag_r = stats(F)
    rng = np.random.default_rng(seed)
    B = {"F": [], "Ds": [], "Dr": [], "mag_s": [], "mag_r": []}
    for _ in range(n_boot):
        Fb = rate_matrix({y: a[rng.integers(0, len(a), len(a))] for y, a in arrays.items()})
        _, ds, dr, ms, mr = stats(Fb)
        for k, v in zip(B, (Fb, ds, dr, ms, mr)):
            B[k].append(v)
    B = {k: np.stack(v) for k, v in B.items()}
    ci = lambda a: (np.nanpercentile(a, 2.5, axis=0), np.nanpercentile(a, 97.5, axis=0))

    def basic_ci(est, boot):
        """Basic (reverse-percentile) bootstrap interval. Used for drift MAGNITUDES: a mean absolute
        change is biased upward by sampling noise (|noise| > 0 even when nothing changed), so plain
        percentile intervals sit above the estimate. bias = mean(boot) - estimate is reported too."""
        lo, hi = ci(boot)
        return 2 * est - hi, 2 * est - lo, np.nanmean(boot, axis=0) - est

    raw_tr = [f"{a}->{b}" for a, b in zip(years[:-1], years[1:])]
    sm_years = years[1:-1]                                 # centres of the smoothed series
    sm_tr = [f"{a}->{b}" for a, b in zip(sm_years[:-1], sm_years[1:])]
    lo_r, hi_r, bias_r = basic_ci(mag_r, B["mag_r"]); lo_s, hi_s, bias_s = basic_ci(mag_s, B["mag_s"])
    magnitude = pd.concat([
        pd.DataFrame({"series": "raw", "transition": raw_tr, "drift_pp": 100 * mag_r,
                      "ci95_lo": 100 * lo_r, "ci95_hi": 100 * hi_r, "boot_bias_pp": 100 * bias_r}),
        pd.DataFrame({"series": "smoothed", "transition": sm_tr, "drift_pp": 100 * mag_s,
                      "ci95_lo": 100 * lo_s, "ci95_hi": 100 * hi_s, "boot_bias_pp": 100 * bias_s})],
        ignore_index=True)
    magnitude["ci_method"] = "basic bootstrap (magnitudes are biased upward by noise)"
    lo, hi = ci(B["Dr"])
    per_drug = pd.DataFrame([{"transition": raw_tr[t], "drug": drugs[d], "change_pp": 100 * Dr[t, d],
                              "ci95_lo": 100 * lo[t, d], "ci95_hi": 100 * hi[t, d]}
                             for t in range(len(raw_tr)) for d in range(len(drugs))])
    flo, fhi = ci(B["F"])
    rates = pd.DataFrame([{"year": y, "drug": drugs[d], "rate": F[i, d], "ci95_lo": flo[i, d],
                           "ci95_hi": fhi[i, d], "n": len(arrays[y])}
                          for i, y in enumerate(years) for d in range(len(drugs))])
    out = {"years": years, "drugs": drugs, "dropped_drugs_not_tested_every_year": dropped,
           "F": F, "S": S, "magnitude": magnitude, "per_drug": per_drug, "rates": rates,
           "n_per_year": {y: len(a) for y, a in arrays.items()}, "n_boot": n_boot}
    key = f"{policy_transition[0]}->{policy_transition[1]}"
    if key in raw_tr:   # drift at the policy transition minus the median drift of all other transitions
        t = raw_tr.index(key)
        others = [i for i in range(len(raw_tr)) if i != t]
        ex = B["mag_r"][:, t] - np.median(B["mag_r"][:, others], axis=1)
        est = mag_r[t] - np.median(mag_r[others])
        q_lo, q_hi = np.percentile(ex, [2.5, 97.5])
        out["policy_excess_drift_pp"] = {"transition": key, "value": float(100 * est),
                                         "ci95": [float(100 * (2 * est - q_hi)), float(100 * (2 * est - q_lo))],
                                         "ci_method": "basic bootstrap",
                                         "note": "association only, not a causal effect of GFI #213"}
    return out


# =====================================================================================
# 6. CIS 731: learned 1-D convolutional kernel vs. fixed kernels (one-step-ahead forecasts)
# =====================================================================================
def fixed_forecasts(F: np.ndarray, t: int) -> dict[str, np.ndarray]:
    """Forecasts of F[t] from F[:t] using only the fixed kernels."""
    out = {"persistence": F[t - 1],
           "diff_extrapolation": F[t - 1] + (F[t - 1] - F[t - 2])}   # last value + last [-1,1] change
    if t >= 3:
        out["smoothing_kernel"] = SMOOTH @ F[t - 3:t]                 # [0.25,0.5,0.25] over the last 3 years
    return out


def fit_kernel_torch(F_train: np.ndarray, w: int = 3, epochs: int = 3000, lr: float = 0.03,
                     weight_decay: float = 1e-3, seed: int = 0) -> tuple[np.ndarray, float]:
    """One Conv1d(1, 1, kernel_size=w) shared by all drugs: F[t] ~ sum_j k[j] F[t-w+j] + b.
    Each drug is one series in the batch. Full-batch Adam on CPU, fixed seed."""
    import torch
    torch.manual_seed(seed)
    x = torch.tensor(F_train[:-1].T[:, None, :], dtype=torch.float64)   # (drugs, 1, T-1)
    y = torch.tensor(F_train[w:].T[:, None, :], dtype=torch.float64)    # (drugs, 1, T-w)
    conv = torch.nn.Conv1d(1, 1, kernel_size=w, bias=True).double()
    with torch.no_grad():                       # start from persistence: k = [0, ..., 0, 1], b = 0
        conv.weight.zero_(); conv.weight[0, 0, -1] = 1.0; conv.bias.zero_()
    opt = torch.optim.Adam(conv.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.mean((conv(x) - y) ** 2) + weight_decay * torch.sum((conv.weight - _persist(w, torch)) ** 2)
        loss.backward()
        opt.step()
    return conv.weight.detach().numpy()[0, 0].copy(), float(conv.bias.detach().numpy()[0])


def _persist(w, torch):
    p = torch.zeros(1, 1, w, dtype=torch.float64); p[0, 0, -1] = 1.0
    return p


def fit_kernel_lstsq(F_train: np.ndarray, w: int = 3, weight_decay: float = 1e-3) -> tuple[np.ndarray, float]:
    """Closed-form version of the same objective (ridge towards persistence): a check on the torch fit."""
    X = np.concatenate([np.stack([F_train[t - w:t, d] for t in range(w, len(F_train))])
                        for d in range(F_train.shape[1])])
    y = np.concatenate([F_train[w:, d] for d in range(F_train.shape[1])])
    n = len(y)
    A = np.hstack([X, np.ones((n, 1))])
    prior = np.r_[np.zeros(w - 1), 1.0, 0.0]
    R = np.diag(np.r_[np.full(w, weight_decay * n), 0.0])
    theta = np.linalg.solve(A.T @ A + R, A.T @ y + R @ prior)
    return theta[:w], float(theta[w])


def forecast_comparison(F: np.ndarray, years: Sequence[int], w: int = 3, n_test_years: int = 2,
                        use_torch: bool = True, n_boot: int = 5000, seed: int = SEED) -> dict:
    """Rolling origin: for each of the last n_test_years years t, fit the learned kernel on F[:t] only
    and forecast F[t] from F[t-w:t]; fixed kernels forecast from the same history. Error = absolute
    error in percentage points over drugs x test years. CIs: bootstrap over drugs (paired)."""
    T = len(years)
    assert T - n_test_years >= w + 2, "too few years to train the learned kernel"
    preds, kernels = {}, []
    for t in range(T - n_test_years, T):
        k, b = (fit_kernel_torch if use_torch else fit_kernel_lstsq)(F[:t], w)
        kernels.append({"test_year": years[t], "kernel": k.round(4).tolist(), "bias": round(b, 5)})
        fc = fixed_forecasts(F, t)
        fc["learned_conv1d"] = k @ F[t - w:t] + b
        for name, v in fc.items():
            preds.setdefault(name, []).append(np.abs(v - F[t]) * 100)   # (drugs,) per test year
    err = {n: np.stack(v) for n, v in preds.items()}                     # (test years, drugs)
    rng = np.random.default_rng(seed)
    D = F.shape[1]
    rows = []
    for n, e in err.items():
        bs = [e[:, rng.integers(0, D, D)].mean() for _ in range(n_boot)]
        d = e - err["persistence"]
        bd = [d[:, ix].mean() for ix in (rng.integers(0, D, D) for _ in range(n_boot))]
        rows.append({"method": n, "MAE_pp": e.mean(), "ci95_lo": np.percentile(bs, 2.5),
                     "ci95_hi": np.percentile(bs, 97.5), "diff_vs_persistence_pp": d.mean(),
                     "diff_ci95_lo": np.percentile(bd, 2.5), "diff_ci95_hi": np.percentile(bd, 97.5)})
    return {"table": pd.DataFrame(rows).sort_values("MAE_pp").reset_index(drop=True),
            "kernels": kernels, "test_years": list(years[T - n_test_years:]), "w": w}


# =====================================================================================
# 7. results.json
# =====================================================================================
def to_jsonable(o):
    """DataFrames -> records, numpy -> Python, NaN -> null (so results.json is valid JSON)."""
    if isinstance(o, pd.DataFrame):
        o = o.reset_index() if not isinstance(o.index, pd.RangeIndex) else o
        return to_jsonable(o.to_dict(orient="records"))
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if math.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    return o


def write_results(path: str, results: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(results), fh, indent=2, default=str, allow_nan=False)