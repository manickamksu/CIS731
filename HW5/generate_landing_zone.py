"""Generate deterministic sample records to test the pipeline locally."""
import json, random
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contracts import validate_python_record

OUT = Path("data/landing")
OUT.mkdir(parents=True, exist_ok=True)
random.seed(17)
base = datetime(2026, 10, 1, 12, 0, 0, tzinfo=timezone.utc)
vocab = ["bacteria", "resistance", "sample", "culture", "antimicrobial", "gene",
         "isolate", "analysis", "sequence", "plasmid", "pathogen", "testing"]
records = []
for i in range(240):
    start = base + timedelta(seconds=i * 8)
    duration = random.randint(5, 28)
    end = start + timedelta(seconds=duration)
    # Deliberate lateness distribution from 0 to 75 seconds, deterministic.
    delay = [1, 3, 5, 8, 12, 18, 25, 35, 48, 65][i % 10]
    ingest = end + timedelta(seconds=delay)
    row = {
        "record_id": f"sample-{i:05d}",
        "track": "ASR" if i % 2 == 0 else "VDB",
        "source_id": f"source-{i % 7}",
        "ts_start": start.isoformat().replace("+00:00", "Z"),
        "ts_end": end.isoformat().replace("+00:00", "Z"),
        "text": " ".join(random.choices(vocab, k=random.randint(4, 12))),
        "ingest_ts": ingest.isoformat().replace("+00:00", "Z"),
    }
    if row["track"] == "VDB":
        row["license_class"] = "cc-by"
    errors = validate_python_record(row)
    if errors:
        raise ValueError((row["record_id"], errors))
    records.append(row)

# One JSONL file is an atomic arrival unit for Spark's file source.
path = OUT / "batch-0001.json"
path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
print(f"Wrote {len(records)} synthetic records to {path}")
