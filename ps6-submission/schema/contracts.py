"""PS 6 contracts: admit_document and vdb_upsert (lyrics corpus, VDB track).

Invariants, in plain language
-----------------------------
EmbeddingCompleteness
    Every vector stored in the index belongs to a schema-valid record that was
    produced by the store's pinned embedding model, has exactly embedding_dim
    finite components, and has unit L2 norm (so inner product == cosine).

NoOrphanEmbeddings
    Every vector in the index belongs to exactly one chunk record, and every
    chunk record belongs to a document that was admitted through
    admit_document with provenance_verified == True. Deleting a document
    deletes all of its vectors in the same call.

Contract-level guarantees
    admit_document  pre:  license_class in {public_domain, instructor_provided} (INV-9)
                    pre:  SHA-256 of the text equals the declared source_hash
                    post: the stored document has provenance_verified == True
    vdb_upsert      pre:  EmbeddingCompleteness and NoOrphanEmbeddings hold for the batch
                    post: exactly one vector per chunk_id (re-upserting replaces, never duplicates)

faiss.IndexIDMap2 does NOT reject duplicate ids, so "one vector per chunk_id"
must be enforced here rather than trusted to the index.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import deal
import faiss
import numpy as np
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

SCHEMA_DIR = Path(__file__).resolve().parent
ALLOWED_LICENSES = frozenset({"public_domain", "instructor_provided"})
UNIT_NORM_TOL = 1e-3
INT63_MASK = (1 << 63) - 1


# --------------------------------------------------------------------------- schemas
def _load(name: str) -> dict:
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


_SCHEMAS = {
    name: _load(name)
    for name in (
        "mp5_corpus_chunk.schema.json",
        "vector_record.schema.json",
        "document_record.schema.json",
    )
}
_REGISTRY = Registry().with_resources(
    (s["$id"], Resource.from_contents(s)) for s in _SCHEMAS.values()
)
for _s in _SCHEMAS.values():
    Draft202012Validator.check_schema(_s)

RECORD_VALIDATOR = Draft202012Validator(
    _SCHEMAS["vector_record.schema.json"], registry=_REGISTRY, format_checker=FormatChecker()
)
DOCUMENT_VALIDATOR = Draft202012Validator(
    _SCHEMAS["document_record.schema.json"], registry=_REGISTRY, format_checker=FormatChecker()
)


# --------------------------------------------------------------------------- helpers
def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def faiss_id_for(chunk_id: str) -> int:
    """Deterministic int64 key for IndexIDMap2: first 63 bits of SHA-256(chunk_id)."""
    digest = hashlib.sha256(chunk_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & INT63_MASK


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class VectorStore:
    """FAISS IndexFlatIP wrapped in IndexIDMap2, plus the metadata needed for provenance."""

    def __init__(self, embedding_model: str, dim: int = 384) -> None:
        self.embedding_model = embedding_model
        self.dim = dim
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self.documents: dict[str, dict] = {}   # doc_id   -> admitted document
        self.records: dict[str, dict] = {}     # chunk_id -> record (vector kept in FAISS)
        self.chunk_of: dict[int, str] = {}     # faiss_id -> chunk_id


# --------------------------------------------------------------------------- predicates
# Each predicate is total (never raises) so a bad input yields a contract
# violation rather than a stray KeyError.

def _license_ok(store: VectorStore, doc: Any) -> bool:
    return isinstance(doc, dict) and doc.get("license_class") in ALLOWED_LICENSES


def _doc_schema_ok(store: VectorStore, doc: Any) -> bool:
    return isinstance(doc, dict) and DOCUMENT_VALIDATOR.is_valid(doc)


def _hash_ok(store: VectorStore, doc: Any) -> bool:
    return (
        isinstance(doc, dict)
        and isinstance(doc.get("text"), str)
        and sha256_hex(doc["text"]) == doc.get("source_hash")
    )


def _not_conflicting(store: VectorStore, doc: Any) -> bool:
    prior = store.documents.get(doc.get("doc_id")) if isinstance(doc, dict) else None
    return prior is None or prior["source_hash"] == doc.get("source_hash")


def provenance_verified(store: VectorStore, doc_id: str) -> bool:
    doc = store.documents.get(doc_id)
    return (
        doc is not None
        and doc.get("provenance_verified") is True
        and doc.get("license_class") in ALLOWED_LICENSES
        and sha256_hex(doc["text"]) == doc["source_hash"]
    )


def embedding_complete(store: VectorStore, rec: Any) -> bool:
    """EmbeddingCompleteness for one record."""
    if not isinstance(rec, dict) or not RECORD_VALIDATOR.is_valid(rec):
        return False
    vec = rec["vector"]
    if rec["embedding_model"] != store.embedding_model or rec["embedding_dim"] != store.dim:
        return False
    if len(vec) != store.dim or not all(math.isfinite(x) for x in vec):
        return False
    return abs(math.sqrt(sum(x * x for x in vec)) - 1.0) <= UNIT_NORM_TOL


def has_parent(store: VectorStore, rec: Any) -> bool:
    """NoOrphanEmbeddings for one record: it points at an admitted, verified document
    and its identity/provenance fields agree with that document."""
    if not isinstance(rec, dict):
        return False
    doc_id, chunk_id = rec.get("doc_id"), rec.get("chunk_id")
    if not isinstance(chunk_id, str) or not provenance_verified(store, doc_id):
        return False
    doc = store.documents[doc_id]
    return (
        chunk_id.startswith(f"{doc_id}#c")
        and rec.get("record_id") == chunk_id          # MP 5 identity carried forward
        and rec.get("source_id") == doc_id
        and rec.get("source_hash") == doc["source_hash"]
        and rec.get("license_class") == doc["license_class"]
        and rec.get("release_year") == doc["release_year"]
    )


def _batch_ids_ok(store: VectorStore, recs: Any) -> bool:
    if not isinstance(recs, list) or not recs:
        return False
    ids = [r.get("chunk_id") for r in recs if isinstance(r, dict)]
    if len(ids) != len(recs) or not all(isinstance(i, str) for i in ids):
        return False
    if len(set(ids)) != len(ids):
        return False  # duplicate chunk_id inside one batch
    for r in recs:
        fid = faiss_id_for(r["chunk_id"])
        if r.get("faiss_id") != fid:
            return False
        owner = store.chunk_of.get(fid)
        if owner is not None and owner != r["chunk_id"]:
            return False  # 63-bit hash collision with a different chunk
    fids = [faiss_id_for(i) for i in ids]
    return len(set(fids)) == len(fids)


def store_consistent(store: VectorStore) -> bool:
    """Store-wide NoOrphanEmbeddings + one-vector-per-chunk check."""
    n = store.index.ntotal
    if not (n == len(store.records) == len(store.chunk_of)):
        return False
    for fid, cid in store.chunk_of.items():
        rec = store.records.get(cid)
        if rec is None or rec["faiss_id"] != fid or not provenance_verified(store, rec["doc_id"]):
            return False
    return True


def _one_vector_each(store: VectorStore, recs: list[dict]) -> bool:
    for r in recs:
        cid, fid = r["chunk_id"], r["faiss_id"]
        if store.chunk_of.get(fid) != cid or cid not in store.records:
            return False
        stored = store.index.reconstruct(fid)
        if not np.allclose(stored, np.asarray(r["vector"], dtype="float32"), atol=1e-6):
            return False
    return store_consistent(store)


# --------------------------------------------------------------------------- contracts
@deal.pre(_license_ok, message="INV-9: license_class must be public_domain or instructor_provided")
@deal.pre(_doc_schema_ok, message="document does not validate against document_record.schema.json")
@deal.pre(_hash_ok, message="SHA-256 of text does not match source_hash")
@deal.pre(_not_conflicting, message="doc_id already admitted with different content")
@deal.ensure(
    lambda store, doc, result: result is store.documents[doc["doc_id"]]
    and provenance_verified(store, doc["doc_id"]),
    message="postcondition provenance_verified failed",
)
def admit_document(store: VectorStore, doc: dict) -> dict:
    """Admit one source document. Idempotent for identical content."""
    if doc["doc_id"] in store.documents:
        return store.documents[doc["doc_id"]]
    admitted = dict(doc)
    admitted["admitted_ts"] = _now()
    admitted["provenance_verified"] = True
    store.documents[doc["doc_id"]] = admitted
    return admitted


@deal.pre(lambda store, recs: _batch_ids_ok(store, recs),
          message="batch must be non-empty, chunk_ids unique, faiss_id = H63(chunk_id), no id collision")
@deal.pre(lambda store, recs: all(embedding_complete(store, r) for r in recs),
          message="EmbeddingCompleteness violated (schema, model, dim, finiteness, or unit norm)")
@deal.pre(lambda store, recs: all(has_parent(store, r) for r in recs),
          message="NoOrphanEmbeddings violated (unknown/unverified doc_id or provenance mismatch)")
@deal.ensure(lambda store, recs, result: result == len(recs) and _one_vector_each(store, recs),
             message="postcondition failed: exactly one vector per chunk_id")
def vdb_upsert(store: VectorStore, recs: list[dict]) -> int:
    """Insert or replace vectors. Update = remove_ids(old) then add_with_ids(new)."""
    ids = np.array([r["faiss_id"] for r in recs], dtype="int64")
    existing = np.array([i for i in ids.tolist() if i in store.chunk_of], dtype="int64")
    if existing.size:
        store.index.remove_ids(existing)
    vecs = np.ascontiguousarray([r["vector"] for r in recs], dtype="float32")
    store.index.add_with_ids(vecs, ids)
    for r in recs:
        meta = {k: v for k, v in r.items() if k != "vector"}
        store.records[r["chunk_id"]] = meta
        store.chunk_of[r["faiss_id"]] = r["chunk_id"]
    return len(recs)


@deal.pre(lambda store, doc_id: doc_id in store.documents, message="unknown doc_id")
@deal.ensure(
    lambda store, doc_id, result: doc_id not in store.documents
    and all(r["doc_id"] != doc_id for r in store.records.values())
    and store_consistent(store),
    message="delete left orphan vectors",
)
def delete_document(store: VectorStore, doc_id: str) -> int:
    """Cascade delete: a document and all of its vectors leave together."""
    chunk_ids = [cid for cid, r in store.records.items() if r["doc_id"] == doc_id]
    fids = np.array([store.records[c]["faiss_id"] for c in chunk_ids], dtype="int64")
    if fids.size:
        store.index.remove_ids(fids)
    for cid in chunk_ids:
        del store.chunk_of[store.records[cid]["faiss_id"]]
        del store.records[cid]
    del store.documents[doc_id]
    return len(chunk_ids)


def search(store: VectorStore, query_vec: np.ndarray, k: int = 50) -> list[tuple[str, float]]:
    """Exact top-k by inner product (== cosine for unit vectors); returns (chunk_id, score)."""
    q = np.ascontiguousarray(query_vec, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(q)
    scores, fids = store.index.search(q, k)
    return [(store.chunk_of[int(f)], float(s)) for s, f in zip(scores[0], fids[0]) if f != -1]
