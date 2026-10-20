# MP5 Structured Streaming — CPU starter

This is a runnable CPU-only starter for the MP5 assignment. It creates deterministic synthetic JSONL input, validates contracts, runs two Spark Structured Streaming window queries, writes append-mode Parquet outputs with checkpoints, logs progress to `metrics.jsonl`, and includes a reconciliation notebook and report template.

## Important limitations / what you must personalize
- Synthetic records are for validating the pipeline only. Replace or supplement them with your course landing-zone records before reporting assignment results.
- The code measures lateness from the input data (`ingest_ts - ts_end`). Do not report the sample metrics as your course data's results.
- The AssemblyAI quota log is not fabricated. Add the real log if you used AssemblyAI; otherwise state what actually happened.
- Set `GENAI_AUDIT` honestly in `report.md`: include your actual GenAI.pdf log if required, or write exactly `No GenAI Used.` if you did not use it.
- The report should only claim CI passed after you actually run the checks.

## Environment
Use the course's pinned DevContainer if provided. The project is compatible with Python 3.10 and PySpark 3.5.x (Spark 3.5.7 recommended).

```bash
pip install pyspark==3.5.7 pandas jupyter
python src/generate_landing_zone.py
python src/run_pipeline.py --input data/landing --output data/output --checkpoint data/checkpoints --metrics metrics.jsonl --once
```

`--once` processes the files currently in the landing directory and exits. If your Spark distribution keeps the file stream active, use Ctrl+C after `inputRowsPerSecond`/progress is logged; the checkpoint is reusable for restart testing. To exercise restart behavior, run once, stop the process, then rerun with the same checkpoint and output directories.

For a clean fault-injection experiment, first stop the process, then remove only `data/checkpoints` and `data/output` before a fresh baseline run. Do not remove a checkpoint between the first run and its restart.

## Data contracts
`TranscriptChunk` is the ASR shape. `CorpusChunk` is the VDB/corpus shape, including `license_class`. All timestamps are ISO-8601 UTC strings in JSON; Spark parses them as timestamps.

## Outputs
- `data/output/tumbling/`: 60-second event-time tumbling windows
- `data/output/sliding/`: 5-minute windows sliding every 1 minute
- `metrics.jsonl`: one JSON object per query progress event
- `report.md`: short report template to fill with observed results
- `contracts.py`: schema and record-level validators
- `notebooks/reconcile.ipynb`: starter reconciliation notebook

If the environment is already pinned by the course, do not upgrade Spark or Java inside it.
