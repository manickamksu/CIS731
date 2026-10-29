"""Tests for schema/contracts.py.  Run from the repo root:  python -m pytest -q"""
import copy
import sys
from pathlib import Path

import deal
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "schema"))
import contracts as C  # noqa: E402

MODEL = "sentence-transformers/all-MiniLM-L6-v2@0123abc"
TEXT = "Oh, the summer time is coming\nAnd the trees are sweetly blooming"


def make_doc(doc_id="song-0001", text=TEXT, license_class="public_domain"):
    return {
        "doc_id": doc_id,
        "title": "Test Song",
        "source_uri": "https://example.org/songs/1",
        "source_hash": C.sha256_hex(text),
        "license_class": license_class,
        "corpus_type": "lyrics",
        "release_year": 1909,
        "text": text,
    }


def unit_vec(seed, dim=384):
    v = np.random.default_rng(seed).normal(size=dim)
    return (v / np.linalg.norm(v)).tolist()


def make_rec(doc, ordinal=0, seed=0):
    cid = f"{doc['doc_id']}#c{ordinal:04d}"
    return {
        # MP 5 CorpusChunk fields
        "record_id": cid, "track": "VDB", "source_id": doc["doc_id"],
        "ts_start": None, "ts_end": None, "text": doc["text"],
        "ingest_ts": "2026-10-20T12:00:00+00:00", "license_class": doc["license_class"],
        # PS 6 extension
        "chunk_id": cid, "doc_id": doc["doc_id"], "faiss_id": C.faiss_id_for(cid),
        "source_hash": doc["source_hash"], "corpus_type": "lyrics", "release_year": 1909,
        "trope_ids": ["trope:longing.parted_lovers"],
        "embedding_model": MODEL, "embedding_dim": 384, "vector": unit_vec(seed),
        "created_ts": "2026-10-25T12:00:00+00:00",
        "chunking": {"strategy": "structure_aware", "unit": "stanza", "ordinal": ordinal,
                     "char_start": 0, "char_end": len(doc["text"]), "n_tokens": 20,
                     "repeat_count": 1},
        "provenance_verified": True,
    }


@pytest.fixture
def store():
    return C.VectorStore(MODEL)


def test_valid_record_passes_schema():
    assert C.RECORD_VALIDATOR.is_valid(make_rec(make_doc()))


def test_admit_sets_provenance_verified(store):
    out = C.admit_document(store, make_doc())
    assert out["provenance_verified"] is True


def test_unlicensed_document_rejected(store):
    with pytest.raises(deal.PreContractError):
        C.admit_document(store, make_doc(license_class="copyrighted"))


def test_tampered_text_rejected(store):
    doc = make_doc()
    doc["text"] += " (edited)"
    with pytest.raises(deal.PreContractError):
        C.admit_document(store, doc)


def test_orphan_vector_rejected(store):
    rec = make_rec(make_doc())  # document never admitted
    with pytest.raises(deal.PreContractError):
        C.vdb_upsert(store, [rec])


def test_wrong_dimension_rejected(store):
    doc = C.admit_document(store, make_doc())
    rec = make_rec(doc)
    rec["vector"] = rec["vector"][:300]
    with pytest.raises(deal.PreContractError):
        C.vdb_upsert(store, [rec])


def test_unnormalised_vector_rejected(store):
    doc = C.admit_document(store, make_doc())
    rec = make_rec(doc)
    rec["vector"] = [x * 2 for x in rec["vector"]]
    with pytest.raises(deal.PreContractError):
        C.vdb_upsert(store, [rec])


def test_duplicate_chunk_id_in_batch_rejected(store):
    doc = C.admit_document(store, make_doc())
    rec = make_rec(doc)
    with pytest.raises(deal.PreContractError):
        C.vdb_upsert(store, [rec, copy.deepcopy(rec)])


def test_reupsert_replaces_not_duplicates(store):
    doc = C.admit_document(store, make_doc())
    C.vdb_upsert(store, [make_rec(doc, seed=1)])
    C.vdb_upsert(store, [make_rec(doc, seed=2)])
    assert store.index.ntotal == 1 and len(store.records) == 1
    hits = C.search(store, np.array(unit_vec(2)), k=1)
    assert hits[0][0] == "song-0001#c0000" and hits[0][1] == pytest.approx(1.0, abs=1e-5)


def test_delete_cascades(store):
    doc = C.admit_document(store, make_doc())
    C.vdb_upsert(store, [make_rec(doc, 0, 1), make_rec(doc, 1, 2)])
    other = C.admit_document(store, make_doc("song-0002", TEXT + "!"))
    C.vdb_upsert(store, [make_rec(other, 0, 3)])
    assert C.delete_document(store, "song-0001") == 2
    assert store.index.ntotal == 1 and C.store_consistent(store)
