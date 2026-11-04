"""
mp7_core.py: MP 7 few-shot classification library, adapted to the NARMS tetracycline project.

Three classifiers behind ONE `Classifier` interface:
  (i)   PrototypeClassifier: cosine similarity to per-class mean embeddings
  (ii)  KNNClassifier: similarity-weighted vote over a FAISS IndexFlatIP exemplar index
  (iii) FewShotLLMClassifier: fixed, class-balanced demonstrations + constrained label scoring

Each isolate is serialized to a short text record (see mp7_narms.py), embedded with
all-MiniLM-L6-v2 for (i)/(ii), and shown verbatim to the LLM for (iii).

`predict_proba` is the only public scoring entry point and carries `deal` contracts:
  * every probability >= 0
  * every row sums to 1.0 within 1e-6
  * columns map 1:1 onto the declared label space, so every argmax label is in that space
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

import deal
import faiss

# ---------------------------------------------------------------------------
# 0. Label space (NARMS tetracycline phenotype)
# ---------------------------------------------------------------------------
# Keys = canonical label ids exactly as they appear in eval_mp7.jsonl.
# DESCRIPTIONS come from the ontology definitions (NOT from eval items -> no leakage).
# VERBALIZERS are the words the LLM must emit; each must start with a distinct token
# for the model's tokenizer (checked automatically in FewShotLLMClassifier).
# Project label space: tetracycline phenotype of a NARMS food-animal E. coli isolate.
# Ground truth = laboratory MIC interpreted with a single fixed breakpoint (proposal §7);
# intermediate isolates are excluded. NOTE: binary (L = 2), a documented deviation from the
# spec's 8-12 trope labels, which belong to the VDB/ASR trope tracks.
LABELS: tuple[str, ...] = ("resistant", "susceptible")
CANONICAL: dict[str, str] = {"resistant": "Tetracycline-resistant", "susceptible": "Tetracycline-susceptible"}
# Neutral descriptions on purpose: no co-resistance "rules" are written into the prompt.
DESCRIPTIONS: dict[str, str] = {
    "resistant": "the isolate is resistant to tetracycline (MIC at or above the resistance breakpoint).",
    "susceptible": "the isolate is susceptible to tetracycline (MIC at or below the susceptible breakpoint).",
}
VERBALIZERS: dict[str, str] = {"resistant": "Resistant", "susceptible": "Susceptible"}

ATOL = 1e-6  # simplex tolerance required by the spec


# ---------------------------------------------------------------------------
# 1. Data records + I/O
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Example:
    item_id: str           # chunk_id (pool) or eval id
    text: str
    label: str | None      # gold label; None if unknown
    doc_id: str = ""
    source_hash: str = ""


def load_jsonl(path: str, id_key="chunk_id", text_key="text", label_key="label",
               doc_key="doc_id", hash_key="source_hash") -> list[Example]:
    """TODO(student): set the keys to match eval_mp7.jsonl / your MP 5 chunk records.
    If your chunk records store `trope_ids: [..]`, pass label_key="trope_ids" — the first
    id is used as the single gold label (state this rule in protocol.md)."""
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            lab = r.get(label_key)
            if isinstance(lab, list):
                lab = lab[0] if lab else None
            lab = normalize_label(lab)
            out.append(Example(str(r[id_key]), r[text_key], lab,
                               str(r.get(doc_key, "")), str(r.get(hash_key, ""))))
    return out


def normalize_label(lab):
    """Map 'Summer Romance' / 'summer romance' / 'TR-03' -> 'TR-03'; unknown -> as-is."""
    if lab is None:
        return None
    lab = str(lab).strip()
    if lab in LABELS:
        return lab
    inv = {v.lower(): k for k, v in CANONICAL.items()}
    return inv.get(lab.lower(), lab)


def make_dev_split(items, eval_frac: float = 0.4, seed: int = 20261103, labels=LABELS):
    """FALLBACK when eval_mp7.jsonl is unavailable: split YOUR labeled chunks into a
    demonstration pool and a held-out dev-eval set. Grouped by doc_id (no document on both
    sides -> no leakage) and stratified per label. Metrics on it are ADVISORY (INV-12)."""
    rng = np.random.default_rng(seed)
    by_doc = {}
    for e in items:
        if e.label in labels:
            by_doc.setdefault(e.doc_id or e.item_id, []).append(e)
    # stratify on each document's majority label
    groups = {}
    for d, es in sorted(by_doc.items()):
        maj = max(labels, key=lambda l: sum(x.label == l for x in es))
        groups.setdefault(maj, []).append(d)
    pool, ev = [], []
    for l in labels:
        docs = groups.get(l, [])
        docs = [docs[i] for i in rng.permutation(len(docs))]
        n_ev = int(round(eval_frac * len(docs)))
        for j, d in enumerate(docs):
            (ev if j < n_ev else pool).extend(by_doc[d])
    return pool, ev


def check_counts(pool, ev, k_max: int = 8, labels=LABELS) -> dict:
    """Pool needs >= k_max items per label for the k=8 runs; eval needs >=1 per label."""
    rep = {l: {"pool": sum(p.label == l for p in pool), "eval": sum(e.label == l for e in ev)} for l in labels}
    short = [l for l, c in rep.items() if c["pool"] < k_max or c["eval"] < 1]
    if short:
        print("WARNING: too few items for", short, "-> reduce k_max in protocol.md or add labeled chunks")
    return rep


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 2. Embedder (sentence-transformers, normalized -> inner product == cosine)
# ---------------------------------------------------------------------------
class Embedder:
    """Wraps a sentence-transformers model pinned to a revision hash.
    Use the SAME embedding model you used for your MP 5 / PS 6 index."""

    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
                 revision: str | None = None, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer
        self.model_id, self.revision = model_id, revision
        self.model = SentenceTransformer(model_id, revision=revision, device="cpu")
        self.batch_size = batch_size
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if todo:
            vecs = self.model.encode(list(todo), batch_size=self.batch_size,
                                     normalize_embeddings=True, convert_to_numpy=True)
            for t, v in zip(todo, vecs):
                self._cache[t] = v.astype("float32")
        return np.stack([self._cache[t] for t in texts]).astype("float32")


def _l2norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float32")
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def _softmax(z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def label_description_texts(labels: Sequence[str]) -> list[str]:
    """k = 0 'exemplars' for the embedding classifiers: ontology descriptions only."""
    return [f"{CANONICAL.get(l, l)}: {DESCRIPTIONS[l]}" for l in labels]


# ---------------------------------------------------------------------------
# 3. Unified Classifier interface + deal contracts
# ---------------------------------------------------------------------------
def _simplex_ok(self, texts, result) -> bool:
    r = np.asarray(result)
    return (r.ndim == 2
            and r.shape == (len(texts), len(self.labels))
            and bool(np.all(np.isfinite(r)))
            and bool(np.all(r >= 0.0))
            and bool(np.all(np.abs(r.sum(axis=1) - 1.0) <= ATOL)))


def _labels_ok(self, texts, result) -> bool:
    return all(lbl in self.labels for lbl in result)


class Classifier(ABC):
    """Shared interface reused unchanged in MP 9.

    Subclasses implement `_fit` and `_proba`; the public, contracted methods are
    `fit`, `predict_proba`, and `predict` (template-method pattern, so the contracts
    apply to every subclass — deal decorators on an overridden method would be lost)."""

    name: str = "base"

    def __init__(self, labels: Sequence[str] = LABELS):
        assert len(labels) >= 2
        if not 8 <= len(labels) <= 12:
            import warnings
            warnings.warn(f"label space has {len(labels)} labels; the MP 7 spec's 8-12 applies to the "
                          "trope tracks. Document this deviation in protocol.md.", stacklevel=2)
        assert len(set(labels)) == len(labels)
        self.labels: tuple[str, ...] = tuple(labels)
        self.demos: list[Example] = []

    # --- public API -------------------------------------------------------
    @deal.pre(lambda self, demos: all(d.label in self.labels for d in demos),
              message="demonstration label outside declared label space")
    def fit(self, demos: Sequence[Example]) -> "Classifier":
        self.demos = list(demos)
        self._fit(self.demos)
        return self

    @deal.pre(lambda self, texts: len(texts) > 0)
    @deal.ensure(_simplex_ok,
                 message="predict_proba must be non-negative, rows sum to 1 (1e-6), shape (n, |labels|)")
    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        p = np.asarray(self._proba(list(texts)), dtype=np.float64)
        p = np.clip(p, 0.0, None)
        return p / p.sum(axis=1, keepdims=True)

    @deal.ensure(_labels_ok, message="predicted label outside declared label space")
    def predict(self, texts: Sequence[str]) -> list[str]:
        p = self.predict_proba(texts)
        return [self.labels[i] for i in p.argmax(axis=1)]

    # --- subclass hooks ----------------------------------------------------
    @abstractmethod
    def _fit(self, demos: list[Example]) -> None: ...

    @abstractmethod
    def _proba(self, texts: list[str]) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# 3(i). Embedding-prototype classifier
# ---------------------------------------------------------------------------
class PrototypeClassifier(Classifier):
    name = "prototype"

    def __init__(self, embedder, labels=LABELS, temperature: float = 0.05):
        super().__init__(labels)
        self.embedder, self.temperature = embedder, temperature
        self.prototypes: np.ndarray | None = None

    def _fit(self, demos):
        if not demos:  # k = 0: prototypes = ontology descriptions
            self.prototypes = _l2norm(self.embedder.encode(label_description_texts(self.labels)))
            return
        X = self.embedder.encode([d.text for d in demos])
        y = np.array([d.label for d in demos])
        protos = []
        for lbl in self.labels:
            m = y == lbl
            assert m.any(), f"no demonstrations for {lbl} — demos must be class-balanced"
            protos.append(X[m].mean(axis=0))
        self.prototypes = _l2norm(np.stack(protos))

    def _proba(self, texts):
        Q = _l2norm(self.embedder.encode(texts))
        sims = Q @ self.prototypes.T            # cosine, since both are unit-norm
        return _softmax(sims, self.temperature)  # T fixed in protocol.md, never tuned on eval


# ---------------------------------------------------------------------------
# 3(ii). kNN over a FAISS IndexFlatIP exemplar index
# ---------------------------------------------------------------------------
class KNNClassifier(Classifier):
    name = "knn"

    def __init__(self, embedder, labels=LABELS, n_neighbors: int = 5, temperature: float = 0.05):
        super().__init__(labels)
        self.embedder, self.n_neighbors, self.temperature = embedder, n_neighbors, temperature
        self.index: faiss.Index | None = None
        self.index_labels: np.ndarray | None = None

    def _fit(self, demos):
        if demos:
            X = _l2norm(self.embedder.encode([d.text for d in demos]))
            lab = [d.label for d in demos]
        else:  # k = 0: index holds one description vector per label
            X = _l2norm(self.embedder.encode(label_description_texts(self.labels)))
            lab = list(self.labels)
        self.index = faiss.IndexFlatIP(X.shape[1])   # exact inner-product search
        self.index.add(np.ascontiguousarray(X))
        lab_to_i = {l: i for i, l in enumerate(self.labels)}
        self.index_labels = np.array([lab_to_i[l] for l in lab])

    def _proba(self, texts):
        Q = np.ascontiguousarray(_l2norm(self.embedder.encode(texts)))
        nn = min(self.n_neighbors, self.index.ntotal)
        sims, idx = self.index.search(Q, nn)
        w = np.exp((sims - sims.max(axis=1, keepdims=True)) / self.temperature)
        P = np.zeros((len(texts), len(self.labels)))
        for r in range(len(texts)):
            np.add.at(P[r], self.index_labels[idx[r]], w[r])
        return P / P.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# 3(iii). Few-shot LLM with fixed, class-balanced demonstrations
# ---------------------------------------------------------------------------
SYSTEM_TEMPLATE = (
    "You are a veterinary microbiology assistant. You will see an antimicrobial-susceptibility "
    "surveillance record for one Escherichia coli isolate from a food-producing animal. The "
    "tetracycline result is hidden. Predict the tetracycline phenotype and answer with exactly ONE "
    "label from the list below. Output only the label word, with no explanation and no other text.\n"
    "Allowed labels:\n{label_block}\n"
    "Never output any other word."
)
DEMO_TEMPLATE = "Isolate: {text}\nTetracycline: {verbalizer}"
QUERY_TEMPLATE = "Isolate: {text}\nTetracycline:"
QUERY_SLOT = "<<<QUERY_ISOLATE>>>"


class LLMBackend(ABC):
    """Scores the first token of each verbalizer after `prefix + rest` (constrained decoding)."""
    model_id: str = ""
    revision: str = ""

    @abstractmethod
    def check_verbalizers(self, verbalizers: Sequence[str]) -> None: ...

    @abstractmethod
    def truncate(self, text: str, max_tokens: int) -> str: ...

    @abstractmethod
    def render(self, system: str, user: str) -> str:
        """Return the full chat-formatted prompt string ending where the answer begins."""

    @abstractmethod
    def label_logprobs(self, prefix: str, rest: str, verbalizers: Sequence[str]) -> np.ndarray: ...


class HFBackend(LLMBackend):
    """Local CPU inference with Hugging Face transformers. Shared-prefix KV caching makes
    the fixed demonstration block cost one forward pass per run instead of one per item."""

    def __init__(self, model_id="Qwen/Qwen2.5-0.5B-Instruct", revision: str | None = None,
                 use_prefix_cache: bool = True, quota_log: "QuotaLog | None" = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import HfApi
        self.torch = torch
        torch.set_grad_enabled(False)
        self.revision = revision or HfApi().model_info(model_id).sha  # pin & record
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=self.revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=self.revision, torch_dtype=torch.float32).to("cpu").eval()
        self.use_prefix_cache = use_prefix_cache
        self.quota_log = quota_log
        self._prefix_key, self._prefix_cache, self._prefix_len = None, None, 0
        self._first_ids: dict[tuple, list[int]] = {}

    def _ids(self, s: str) -> list[int]:
        return self.tok(s, add_special_tokens=False).input_ids

    def first_token_ids(self, verbalizers):
        key = tuple(verbalizers)
        if key not in self._first_ids:
            # tokenize as they appear after the generation prompt (no leading space)
            self._first_ids[key] = [self._ids(v)[0] for v in verbalizers]
        return self._first_ids[key]

    def check_verbalizers(self, verbalizers):
        ids = self.first_token_ids(verbalizers)
        if len(set(ids)) != len(ids):
            clash = [(v, self.tok.decode([i])) for v, i in zip(verbalizers, ids)]
            raise ValueError(f"verbalizers share a first token; rename them: {clash}")

    def truncate(self, text, max_tokens):
        ids = self._ids(text)
        return text if len(ids) <= max_tokens else self.tok.decode(ids[:max_tokens]) + " …"

    def render(self, system, user):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def label_logprobs(self, prefix, rest, verbalizers):
        torch = self.torch
        cand = self.first_token_ids(verbalizers)
        pre_ids, rest_ids = self._ids(prefix), self._ids(rest)
        if self.use_prefix_cache:
            if self._prefix_key != prefix:
                out = self.model(torch.tensor([pre_ids]), use_cache=True)
                self._prefix_key, self._prefix_cache = prefix, out.past_key_values
                self._prefix_len = len(pre_ids)
                if self.quota_log:
                    self.quota_log.add(prompt_tokens=len(pre_ids), requests=1)
            cache = copy.deepcopy(self._prefix_cache)
            out = self.model(torch.tensor([rest_ids]), past_key_values=cache, use_cache=True)
            n_tok = len(rest_ids)
        else:
            out = self.model(torch.tensor([pre_ids + rest_ids]))
            n_tok = len(pre_ids) + len(rest_ids)
        if self.quota_log:
            self.quota_log.add(prompt_tokens=n_tok, requests=1)
        logits = out.logits[0, -1].double()
        logp = torch.log_softmax(logits, dim=-1)       # over the FULL vocabulary
        return logp[cand].cpu().numpy()                 # constrained to label tokens


class FewShotLLMClassifier(Classifier):
    """Fixed demonstrations: the SAME demo block (chosen + ordered by `seed`) is used for every
    eval item in a run — no retrieval conditioning (that is MP 9)."""
    name = "fewshot_llm"

    def __init__(self, backend: LLMBackend, labels=LABELS, seed: int = 0,
                 max_demo_tokens: int = 96, max_query_tokens: int = 256,
                 permute_label_list: bool = True, contextual_calibration: bool = False):
        super().__init__(labels)
        self.backend, self.seed = backend, seed
        self.max_demo_tokens, self.max_query_tokens = max_demo_tokens, max_query_tokens
        self.permute_label_list = permute_label_list
        self.contextual_calibration = contextual_calibration
        self.verbalizers = [VERBALIZERS[l] for l in self.labels]
        backend.check_verbalizers(self.verbalizers)
        assert len({v.lower() for v in self.verbalizers}) == len(self.verbalizers)
        self.prefix = self.suffix_tmpl = ""
        self.calib_bias = np.zeros(len(self.labels))

    def build_prompt_parts(self) -> tuple[str, str]:
        rng = np.random.default_rng([self.seed, 7919])
        order = rng.permutation(len(self.labels)) if self.permute_label_list else np.arange(len(self.labels))
        label_block = "\n".join(f"- {self.verbalizers[i]} ({CANONICAL.get(self.labels[i], self.labels[i])}): "
                                f"{DESCRIPTIONS[self.labels[i]]}" for i in order)
        system = SYSTEM_TEMPLATE.format(label_block=label_block)
        demo_block = "\n\n".join(
            DEMO_TEMPLATE.format(text=self.backend.truncate(d.text, self.max_demo_tokens),
                                 verbalizer=VERBALIZERS[d.label])
            for d in self.demos)  # self.demos is ALREADY in the seed-permuted order
        user = (f"Examples:\n\n{demo_block}\n\nNow classify this isolate.\n\n" if self.demos
                else "Classify this isolate.\n\n") + QUERY_TEMPLATE.format(text=QUERY_SLOT)
        full = self.backend.render(system, user)
        prefix, suffix = full.split(QUERY_SLOT)
        return prefix, suffix

    def _fit(self, demos):
        self.prefix, self.suffix_tmpl = self.build_prompt_parts()
        if self.contextual_calibration:  # Zhao et al. 2021 content-free calibration
            lp = np.mean([self.backend.label_logprobs(self.prefix, cf + self.suffix_tmpl, self.verbalizers)
                          for cf in ("N/A", "[MASK]", "")], axis=0)
            self.calib_bias = lp - np.log(np.exp(lp).sum())
        else:
            self.calib_bias = np.zeros(len(self.labels))

    def prompt_sha256(self) -> str:
        return hashlib.sha256((self.prefix + "|" + self.suffix_tmpl).encode()).hexdigest()

    def render_example(self, text: str) -> str:
        return self.prefix + self.backend.truncate(text, self.max_query_tokens) + self.suffix_tmpl

    def _proba(self, texts):
        rows = []
        for t in texts:
            rest = self.backend.truncate(t, self.max_query_tokens) + self.suffix_tmpl
            lp = self.backend.label_logprobs(self.prefix, rest, self.verbalizers)
            rows.append(_softmax(lp - self.calib_bias))  # renormalize over the label set only
        return np.vstack(rows)


def parse_label(raw: str, labels: Sequence[str] = LABELS) -> str | None:
    """Output parser for FREE-TEXT generation (e.g. a hosted API without logprobs).
    Not needed for constrained scoring, but document it. Rules: strip, take first line,
    drop punctuation/quotes, case-insensitive exact match on verbalizer, then on label id.
    Anything else -> None (counted as a parse failure and logged, never silently mapped)."""
    s = raw.strip().splitlines()[0] if raw.strip() else ""
    s = s.strip(" \t\"'`.:;,!*").lower()
    for l in labels:
        if s == VERBALIZERS[l].lower() or s == l.lower():
            return l
    return None


# ---------------------------------------------------------------------------
# 4. Demonstration sampling + failure-mode controls
# ---------------------------------------------------------------------------
def sample_demos(pool: Sequence[Example], k: int, seed: int,
                 labels: Sequence[str] = LABELS) -> list[Example]:
    """Class-balanced: exactly k demos PER LABEL (majority-label-bias control).
    Nested across k for a fixed seed (k=3 demos ⊂ k=5 demos), then the whole block is
    shuffled with a (seed, k)-specific permutation (demonstration-order / position-bias control)."""
    if k == 0:
        return []
    rng = np.random.default_rng(seed)
    chosen = []
    for lbl in labels:
        cands = sorted([e for e in pool if e.label == lbl], key=lambda e: e.item_id)
        if len(cands) < k:
            raise ValueError(f"label {lbl!r} has only {len(cands)} pool items; need k={k}")
        perm = rng.permutation(len(cands))       # one permutation per label per seed
        chosen.extend(cands[i] for i in perm[:k])  # prefix -> nested sets
    order = np.random.default_rng([seed, k]).permutation(len(chosen))
    return [chosen[i] for i in order]


def leakage_filter(pool: Sequence[Example], eval_items: Sequence[Example], embedder=None,
                   near_dup_threshold: float = 0.95) -> tuple[list[Example], dict]:
    """Label-leakage control: no eval item (or its document) may serve as a demonstration.
    Removes pool items sharing chunk_id, doc_id or source_hash with any eval item, then
    (if an embedder is given) any pool item with cosine >= threshold to an eval item."""
    ev_ids = {e.item_id for e in eval_items}
    ev_docs = {e.doc_id for e in eval_items if e.doc_id}
    ev_hash = {e.source_hash for e in eval_items if e.source_hash}
    kept, report = [], {"id": 0, "doc": 0, "hash": 0, "near_dup": 0}
    for p in pool:
        if p.item_id in ev_ids:
            report["id"] += 1
        elif p.doc_id and p.doc_id in ev_docs:
            report["doc"] += 1
        elif p.source_hash and p.source_hash in ev_hash:
            report["hash"] += 1
        else:
            kept.append(p)
    if embedder is not None and kept:
        P = _l2norm(embedder.encode([p.text for p in kept]))
        E = _l2norm(embedder.encode([e.text for e in eval_items]))
        idx = faiss.IndexFlatIP(E.shape[1]); idx.add(np.ascontiguousarray(E))
        sims, _ = idx.search(np.ascontiguousarray(P), 1)
        keep_mask = sims[:, 0] < near_dup_threshold
        report["near_dup"] = int((~keep_mask).sum())
        kept = [p for p, m in zip(kept, keep_mask) if m]
    report["pool_after"] = len(kept)
    return kept, report


def label_mention_rate(eval_items: Sequence[Example], labels=LABELS) -> float:
    """Diagnostic: share of eval passages that literally contain a label word/verbalizer
    (surface-cue leakage). Report it; do not 'fix' it by editing eval text."""
    words = {w.lower() for l in labels for w in (l.replace("_", " "), VERBALIZERS[l])}
    return float(np.mean([any(w in e.text.lower() for w in words) for e in eval_items]))


def bias_diagnostics(preds: Sequence[str], gold: Sequence[str], demos: Sequence[Example],
                     labels=LABELS) -> dict:
    """Verifiable checks for majority-label bias and recency (position) bias."""
    L = len(labels)
    pc = np.array([sum(p == l for p in preds) for l in labels], float) / len(preds)
    gc = np.array([sum(g == l for g in gold) for l in labels], float) / len(gold)
    eps = 1e-9
    out = {
        "pred_dist": dict(zip(labels, pc.round(4).tolist())),
        "max_pred_share": float(pc.max()),
        "kl_pred_vs_gold": float(np.sum(pc * np.log((pc + eps) / (gc + eps)))),
        "uniform_share": 1.0 / L,
    }
    if demos:
        last = demos[-1].label
        out["last_demo_label"] = last
        out["share_pred_eq_last_demo"] = float(np.mean([p == last for p in preds]))
        out["gold_share_last_demo_label"] = float(np.mean([g == last for g in gold]))
    return out


# ---------------------------------------------------------------------------
# 5. Metrics — graded vs. dev (INV-12)
# ---------------------------------------------------------------------------
GRADED_METRIC_FN: Callable | None = None   # TODO(student): set to the instructor's compute_graded_metric


def graded_macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels=LABELS) -> float:
    """Graded macro-F1 — ONLY via the instructor's compute_graded_metric.
    TODO(student): adapt the call below to the real signature you were given, e.g.
        from course_eval import compute_graded_metric
        mp7_core.GRADED_METRIC_FN = compute_graded_metric
    """
    if GRADED_METRIC_FN is None:
        raise RuntimeError("compute_graded_metric not wired in — refusing to produce a graded number. "
                           "Use compute_dev_metric for advisory numbers.")
    res = GRADED_METRIC_FN(y_true=list(y_true), y_pred=list(y_pred), metric="macro_f1", labels=list(labels))
    return float(res["value"] if isinstance(res, dict) else res)


def compute_dev_metric(y_true, y_pred, labels=LABELS) -> dict:
    """Advisory metric on your own splits (grading_eligible=False)."""
    from sklearn.metrics import f1_score
    return {"metric": "macro_f1",
            "value": float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)),
            "grading_eligible": False}


def bootstrap_ci(y_true: Sequence[str], preds_by_seed: Sequence[Sequence[str]],
                 metric: Callable[[Sequence[str], Sequence[str]], float],
                 n_boot: int = 1000, alpha: float = 0.05, rng_seed: int = 20261103) -> tuple[float, float]:
    """Percentile bootstrap over EVAL ITEMS (paired across seeds): each replicate resamples
    item indices with replacement, computes macro-F1 for every seed on the same indices,
    and averages over seeds. Returns the (2.5%, 97.5%) percentiles."""
    y = np.asarray(y_true)
    P = [np.asarray(p) for p in preds_by_seed]
    n = len(y)
    rng = np.random.default_rng(rng_seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        ix = rng.integers(0, n, n)
        stats[b] = np.mean([metric(y[ix].tolist(), p[ix].tolist()) for p in P])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# 6. Sweep driver -> predictions.jsonl
# ---------------------------------------------------------------------------
class QuotaLog:
    def __init__(self, path="quota_log.jsonl", request_cap: int | None = None):
        self.path, self.cap = path, request_cap
        self.prompt_tokens = self.requests = 0

    def add(self, prompt_tokens=0, requests=0):
        self.prompt_tokens += prompt_tokens
        self.requests += requests
        if self.cap is not None and self.requests > self.cap:
            raise RuntimeError("quota killswitch tripped (INV-18) — switch to the local CPU backend")

    def flush(self, **extra):
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": time.time(), "requests": self.requests,
                                 "prompt_tokens": self.prompt_tokens, **extra}) + "\n")


def run_sweep(make_classifier: dict[str, Callable[[int], Classifier]],
              pool: Sequence[Example], eval_items: Sequence[Example],
              ks: Iterable[int] = (0, 1, 3, 5, 8), seeds: Iterable[int] = (0, 1, 2, 3, 4),
              out_path: str = "predictions.jsonl", eval_sha256: str = "",
              grading_eligible: bool = True, labels=LABELS, log=print) -> list[dict]:
    """make_classifier: {name: factory(seed) -> Classifier}. Writes one JSON line per
    (classifier, k, seed, eval item) and returns per-run diagnostics."""
    texts = [e.text for e in eval_items]
    gold = [e.label for e in eval_items]
    assert all(g in labels for g in gold), "eval gold label outside declared label space"
    runs = []
    with open(out_path, "w", encoding="utf-8") as fh:
        for cname, factory in make_classifier.items():
            for k in ks:
                for seed in seeds:
                    t0 = time.time()
                    demos = sample_demos(pool, k, seed, labels)
                    clf = factory(seed).fit(demos)
                    P = clf.predict_proba(texts)
                    preds = [clf.labels[i] for i in P.argmax(1)]
                    run_id = f"{cname}|k={k}|seed={seed}"
                    meta = {"run_id": run_id, "classifier": cname, "k": k, "seed": seed,
                            "demo_ids": [d.item_id for d in demos],
                            "prompt_sha256": clf.prompt_sha256() if hasattr(clf, "prompt_sha256") else None,
                            "model_id": getattr(getattr(clf, "backend", None), "model_id", None)
                                        or getattr(getattr(clf, "embedder", None), "model_id", None),
                            "revision": getattr(getattr(clf, "backend", None), "revision", None)
                                        or getattr(getattr(clf, "embedder", None), "revision", None),
                            "eval_sha256": eval_sha256, "grading_eligible": grading_eligible}
                    for e, p, row in zip(eval_items, preds, P):
                        fh.write(json.dumps({**{x: meta[x] for x in ("run_id", "classifier", "k", "seed")},
                                             "item_id": e.item_id, "gold": e.label, "pred": p,
                                             "proba": {l: round(float(v), 8) for l, v in zip(clf.labels, row)},
                                             "prompt_sha256": meta["prompt_sha256"],
                                             "model_id": meta["model_id"], "revision": meta["revision"],
                                             "eval_sha256": eval_sha256,
                                             "grading_eligible": grading_eligible}) + "\n")
                    diag = bias_diagnostics(preds, gold, demos, labels)
                    runs.append({**meta, "seconds": round(time.time() - t0, 2), **diag})
                    log(f"{run_id:32s} dev-macroF1={compute_dev_metric(gold, preds, labels)['value']:.3f} "
                        f"max_pred_share={diag['max_pred_share']:.2f} ({time.time()-t0:.1f}s)")
    return runs


# ---------------------------------------------------------------------------
# 7. Results audit: every reported number is re-derived from predictions.jsonl
# ---------------------------------------------------------------------------
def dev_macro_f1(y_true, y_pred, labels=LABELS) -> float:
    return compute_dev_metric(y_true, y_pred, labels)["value"]


def summarize(pred_path: str, metric: Callable = None, labels=LABELS, n_boot: int = 1000):
    """Returns (table rows, per-class P/R at best config). `metric` defaults to graded_macro_f1."""
    import pandas as pd
    from sklearn.metrics import precision_recall_fscore_support
    metric = metric or (lambda yt, yp: graded_macro_f1(yt, yp, labels))
    df = pd.read_json(pred_path, lines=True)
    rows = []
    for (c, k), g in df.groupby(["classifier", "k"], sort=True):
        g = g.sort_values(["seed", "item_id"])
        seeds = sorted(g.seed.unique())
        items = g[g.seed == seeds[0]].item_id.tolist()
        y = g[g.seed == seeds[0]].gold.tolist()
        P = [g[g.seed == s].set_index("item_id").loc[items, "pred"].tolist() for s in seeds]
        f1s = [metric(y, p) for p in P]
        lo, hi = bootstrap_ci(y, P, metric, n_boot=n_boot)
        rows.append({"classifier": c, "k": int(k), "n_seeds": len(seeds), "n_items": len(y),
                     "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
                     "ci95_lo": lo, "ci95_hi": hi})
    table = pd.DataFrame(rows).sort_values(["classifier", "k"]).reset_index(drop=True)
    best = table.loc[table.macro_f1_mean.idxmax()]
    g = df[(df.classifier == best.classifier) & (df.k == best.k)]
    per = []
    for s, gs in g.groupby("seed"):
        p, r, f, sup = precision_recall_fscore_support(gs.gold, gs.pred, labels=list(labels), zero_division=0)
        per.append(np.vstack([p, r, f, sup]))
    per = np.mean(per, axis=0)
    per_class = pd.DataFrame({"label": labels, "precision": per[0], "recall": per[1],
                              "f1": per[2], "support": per[3].astype(int)})
    return table, best, per_class


# ---------------------------------------------------------------------------
# 8. Environment / reproducibility record
# ---------------------------------------------------------------------------
def environment_report(extra_packages=("numpy", "faiss", "deal", "sklearn", "torch",
                                       "transformers", "sentence_transformers", "huggingface_hub")) -> dict:
    import importlib
    rep = {"python": sys.version.split()[0], "platform": platform.platform(),
           "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "packages": {}}
    for p in extra_packages:
        try:
            m = importlib.import_module(p)
            rep["packages"][p] = getattr(m, "__version__", "?")
        except Exception:
            rep["packages"][p] = None
    try:
        import torch
        rep["torch_cuda_available"] = bool(torch.cuda.is_available())
        rep["torch_threads"] = torch.get_num_threads()
    except Exception:
        pass
    return rep
