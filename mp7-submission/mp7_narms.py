"""
mp7_narms.py: turn the NARMS isolate table into MP 7 `Example` records.

Pipeline (matches the PS 4 proposal):
  raw CSV (unmodified) -> E. coli only -> tetracycline label (R=resistant, S=susceptible, I dropped)
  -> features: animal source, year, Census region, R/S of every OTHER drug (<=30% missing)
  -> one short text record per isolate -> temporal split (train <= 2019, test >= 2020)

TODO(student): set the column names in NarmsConfig after running `inspect_columns(df)`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mp7_core import Example

# Standard NARMS drug abbreviations -> readable names (unknown codes are kept as-is).
DRUG_NAMES = {
    "AMC": "amoxicillin-clavulanic acid", "AMP": "ampicillin", "AXO": "ceftriaxone",
    "AZI": "azithromycin", "CEP": "cephalothin", "CHL": "chloramphenicol",
    "CIP": "ciprofloxacin", "COT": "trimethoprim-sulfamethoxazole", "FIS": "sulfisoxazole",
    "FOX": "cefoxitin", "GEN": "gentamicin", "KAN": "kanamycin", "MER": "meropenem",
    "NAL": "nalidixic acid", "STR": "streptomycin", "TIO": "ceftiofur", "TET": "tetracycline",
    "SMX": "sulfamethoxazole", "COL": "colistin", "TGC": "tigecycline", "SUF": "sulfisoxazole",
    "CTC": "chlortetracycline", "DOX": "doxycycline", "STR": "streptomycin", "ERY": "erythromycin",
}
# Target + same-class drugs are never features (label-leakage control).
# Tigecycline is a glycylcycline (tetracycline derivative); it's banned by default. Remove it here
# only if you can justify keeping it, and say so in protocol.md.
BANNED_TOKENS = ("tet", "tetracycline", "doxycycline", "dox", "minocycline", "min", "ctc", "chlortetracycline",
                 "tgc", "tigecycline", "tetracyclines")

CENSUS_REGION = {  # state abbreviation -> U.S. Census region
    **{s: "Northeast" for s in "CT ME MA NH RI VT NJ NY PA".split()},
    **{s: "Midwest" for s in "IL IN MI OH WI IA KS MN MO NE ND SD".split()},
    **{s: "South" for s in "DE FL GA MD NC SC VA DC WV AL KY MS TN AR LA OK TX".split()},
    **{s: "West" for s in "AZ CO ID MT NV NM UT WY AK CA HI OR WA".split()},
}


@dataclass
class NarmsConfig:
    """Defaults match the FDA/USDA-FSIS file CVM-NARMS-Cecal.xlsx (sheet 'Cecal', 201 columns)."""
    id_col: str = "SAMPLE_ID"              # ISOLATE_ID is missing for 264 E. coli rows; SAMPLE_ID is unique
    species_col: str = "GENUS_NAME"
    species_value: str = "Escherichia"
    source_col: str = "HOST_SPECIES"       # Cattle / Swine / Chickens / Turkeys
    production_col: str | None = "SOURCE"  # Cecal (Beef) / (Dairy) / (Market Hogs) / (Sows); substitutes for region
    year_col: str = "Year"
    state_col: str | None = None           # STATE is 100% missing in this file -> no region feature
    region_col: str | None = None
    target_col: str = "TET SIR"            # S (MIC <=4) / I (MIC 8) / R (MIC >=16)
    # 'X SIR' interpretation columns. Excluded on purpose: SUF SIR (identical to FIS SIR), CIP2 SIR (same
    # MIC as CIP, alternative breakpoint), MER SIR (not tested 2013-2015: panel change). Never used:
    # gene/class columns such as 'Tetracyclines' (WGS tet genes = direct target leakage).
    drug_cols: list[str] = field(default_factory=lambda: [
        "AMC SIR", "AMP SIR", "AXO SIR", "AZI SIR", "CHL SIR", "CIP SIR", "COT SIR",
        "FIS SIR", "FOX SIR", "GEN SIR", "NAL SIR", "STR SIR"])
    max_missing: float = 0.30              # STR is dropped by this rule (>30% missing)


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=0) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path, low_memory=False)


def inspect_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Shows each column's dtype, missing share and a few values, to help fill NarmsConfig."""
    return pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().mean().round(3),
                         "examples": [df[c].dropna().astype(str).unique()[:5].tolist() for c in df]})


def _rsi(v):
    if pd.isna(v):
        return np.nan
    v = str(v).strip().upper()
    return {"R": "R", "S": "S", "I": "I", "RESISTANT": "R", "SUSCEPTIBLE": "S", "INTERMEDIATE": "I"}.get(v, np.nan)


def detect_rsi_columns(df: pd.DataFrame, min_frac: float = 0.5) -> list[str]:
    """Columns where at least min_frac of the non-missing values are R/S/I interpretations."""
    out = []
    for c in df.columns:
        vals = df[c].dropna()
        if len(vals) and vals.map(_rsi).notna().mean() >= min_frac:
            out.append(c)
    return out


def _drug_code(col: str) -> str:
    return re.split(r"[\s_]", str(col).strip())[0].upper()


def is_banned(col: str) -> bool:
    c = str(col).lower()
    return _drug_code(col).lower() in BANNED_TOKENS or any(t in c for t in BANNED_TOKENS if len(t) > 3)


def prepare(df_raw: pd.DataFrame, cfg: NarmsConfig) -> tuple[pd.DataFrame, list[str], dict]:
    """Returns the cleaned table, the feature drug columns used, and a cleaning report."""
    rep = {"rows_raw": len(df_raw)}
    df = df_raw[df_raw[cfg.species_col].astype(str).str.contains(cfg.species_value, case=False, na=False)].copy()
    rep["rows_ecoli"] = len(df)
    df["_y"] = df[cfg.target_col].map(_rsi)
    rep["intermediate_dropped"] = int((df["_y"] == "I").sum())
    rep["target_missing_dropped"] = int(df["_y"].isna().sum())
    df = df[df["_y"].isin(["R", "S"])].copy()
    df["label"] = df["_y"].map({"R": "resistant", "S": "susceptible"})
    drugs = cfg.drug_cols or detect_rsi_columns(df_raw)
    banned = [d for d in drugs if is_banned(d) or d == cfg.target_col]
    drugs = [d for d in drugs if d not in banned]
    miss = {d: float(df[d].isna().mean()) for d in drugs}
    kept = [d for d in drugs if miss[d] <= cfg.max_missing]
    rep.update({"banned_drug_cols": banned, "dropped_for_missingness": sorted(set(drugs) - set(kept)),
                "feature_drugs": kept, "rows_final": len(df),
                "label_counts": df["label"].value_counts().to_dict()})
    if cfg.region_col:
        df["_region"] = df[cfg.region_col].astype(str)
    elif cfg.state_col:
        df["_region"] = df[cfg.state_col].astype(str).str.strip().str.upper().map(CENSUS_REGION).fillna("Unknown")
    else:
        df["_region"] = None
    df["_prod"] = (df[cfg.production_col].astype(str).str.extract(r"\((.*)\)")[0].fillna("unspecified")
                   if cfg.production_col else None)
    assert df[cfg.id_col].notna().all() and not df[cfg.id_col].duplicated().any(), "id column must be unique"
    return df, kept, rep


def serialize(row, cfg: NarmsConfig, drugs: list[str]) -> str:
    """Isolate -> text. Lists resistances explicitly so that embeddings and the LLM see the
    discriminative content. Missing drug results are stated as 'not tested', never imputed."""
    res = [DRUG_NAMES.get(_drug_code(d), d) for d in drugs if _rsi(row[d]) == "R"]
    inter = [DRUG_NAMES.get(_drug_code(d), d) for d in drugs if _rsi(row[d]) == "I"]
    nt = [DRUG_NAMES.get(_drug_code(d), d) for d in drugs if pd.isna(_rsi(row[d]))]
    parts = [f"Animal source: {str(row[cfg.source_col]).strip().lower()}."]
    if row.get("_prod") is not None:
        parts.append(f"Production type: {str(row['_prod']).lower()}.")
    parts.append(f"Year: {int(row[cfg.year_col])}.")
    if row.get("_region") is not None:
        parts.append(f"Region: {row['_region']}.")
    parts += [
              "Resistant to: " + (", ".join(res) if res else "none of the other tested drugs") + "."]
    if inter:
        parts.append("Intermediate: " + ", ".join(inter) + ".")
    if nt:
        parts.append("Not tested: " + ", ".join(nt) + ".")
    parts.append("Susceptible to all other tested drugs.")
    return " ".join(parts)


def to_examples(df: pd.DataFrame, cfg: NarmsConfig, drugs: list[str]) -> list[Example]:
    return [Example(item_id=str(r[cfg.id_col]), text=serialize(r, cfg, drugs), label=r["label"],
                    doc_id=str(r[cfg.id_col]), source_hash="")
            for _, r in df.iterrows()]


def temporal_split(df: pd.DataFrame, cfg: NarmsConfig, train_max_year: int = 2019,
                   test_min_year: int = 2020) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = df[cfg.year_col].astype(int)
    return df[y <= train_max_year].copy(), df[y >= test_min_year].copy()


def stratified_subsample(items: list[Example], n: int, seed: int = 20261103) -> list[Example]:
    """Fixed, label-stratified sample of the test period (keeps CPU LLM runtime bounded).
    The same sample is used for every classifier, k and seed. Save its ids."""
    if n >= len(items):
        return list(items)
    rng = np.random.default_rng(seed)
    labs = sorted({e.label for e in items})
    out = []
    for l in labs:
        grp = sorted([e for e in items if e.label == l], key=lambda e: e.item_id)
        m = int(round(n * len(grp) / len(items)))
        out += [grp[i] for i in rng.choice(len(grp), size=min(m, len(grp)), replace=False)]
    return sorted(out, key=lambda e: e.item_id)


def assert_no_target_leak(examples: list[Example]) -> None:
    """Fails if any serialized record mentions tetracycline (or a same-class drug)."""
    bad = [e.item_id for e in examples
           if re.search(r"\b(tetracycline|doxycycline|minocycline|tigecycline|chlortetracycline|tet\([A-Z]\))", e.text, re.I)]
    assert not bad, f"target leakage in {len(bad)} records, e.g. {bad[:3]}"


def assert_disjoint(pool: list[Example], ev: list[Example]) -> None:
    overlap = {p.item_id for p in pool} & {e.item_id for e in ev}
    assert not overlap, f"{len(overlap)} isolates appear in both pool and test set"
