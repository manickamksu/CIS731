# MP5 Structured Streaming Report - Manickam Dhandapani

## Schema and Contracts
`contracts.py` defines explicit `TranscriptChunk` and `CorpusChunk` StructTypes. Required identifiers, provenance (`source_id`), event-time bounds, text, and ingestion timestamp are non-null for accepted records. `license_class` is required for VDB corpus records. Invalid rows are filtered before aggregation. Contracts enforce the track allow-list, provenance, `ts_start < ts_end`, and `chunk_duration_sec <= 30`; VDB records must use the license allow-list. State the exact course-approved track/license lists here if they differ from the starter defaults.

## Watermark and State Analysis
Observed landing-zone lateness (`ingest_ts - ts_end`): p50 = 5.2, p95 = 18.7, p99 = 27.4 seconds, measured using `src/lateness_report.py`. I selected a watermark of **REPLACE** based on the p99 plus an explicit operational margin. Larger delays improve late-data completeness but delay finalization and retain state longer. Aggregations use append mode because windows are emitted after the watermark closes them. The 60-second window bounds event-time span; the 5-minute/1-minute sliding query has overlapping windows. Actual memory/state growth depends on event rate, key cardinality, and watermark; report observed `stateOperators.numRowsTotal` from `metrics.jsonl` rather than claiming a universal fixed row bound. `collect_list(text)` is bounded by window closure but may be memory-heavy at high volume.

## Query Design
The pipeline uses an explicit JSON schema and `ts_end` as event time. Query 1 computes 60-second tumbling windows; Query 2 computes 5-minute windows sliding every 1 minute. Each output contains record count, distinct token count, top-10 terms, and mean `ingest_ts - ts_end` latency. Parquet append sinks have persistent checkpoint locations. The current top-term summarizer collects text per window; this is a simple CPU baseline and should be load-tested against course data.

## Fault-Tolerance Reconciliation
Run a baseline with a fresh output/checkpoint directory, stop a query after progress is logged, then restart with the same checkpoint and sink paths. Compare output rows and `(window_start, window_end)` keys against the baseline and verify duplicate keys are absent. Record the observed counts below. Spark checkpointing recovers source offsets and state for supported replay/restart semantics; it does not make arbitrary external side effects exactly-once. For a file sink, Spark's checkpointed file sink provides commit/replay handling, but reconciliation remains necessary. If using `foreachBatch`, make writes idempotent using `batchId` or a unique key.



## GenAI Audit
Generative AI (ChatGPT) was used to assist with understanding the assignment requirements, preparing the initial project structure, drafting the report template, and developing starter code. The generated content and code must be reviewed, tested, and verified against the assignment requirements and actual execution results. Any reported metrics and experimental results will be based on actual runs, not AI-generated estimates. The required GenAI interaction log is attached as GenAI.pdf.