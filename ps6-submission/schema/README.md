# PS 6 schema/ (CIS 731, lyrics corpus, VDB track)

| File | Purpose |
|---|---|
| `mp5_corpus_chunk.schema.json` | MP 5 `CorpusChunk` as JSON Schema (base record, unchanged) |
| `vector_record.schema.json` | PS 6 vector record; extends the MP 5 record via `allOf` |
| `document_record.schema.json` | Input to `admit_document` |
| `contracts.py` | `deal` contracts: `admit_document`, `vdb_upsert`, `delete_document` (EmbeddingCompleteness, NoOrphanEmbeddings) |

Dependencies (pin the versions your DevContainer resolves in your lockfile): `faiss-cpu`, `deal`, `jsonschema`, `numpy`, `pytest`.

Run the tests from the repo root:

```bash
python -m pytest -q tests/
```
