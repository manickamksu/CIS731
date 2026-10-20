"""Explicit schemas and contract checks for MP5."""
from pyspark.sql.types import (
    StructType, StructField, StringType, TimestampType, IntegerType
)

# Nullability reflects the deal: identifiers, provenance, event-time bounds, and text
# are required for valid records. The parser may still produce nulls for malformed
# input, so validate records after parsing.
TranscriptChunk = StructType([
    StructField("record_id", StringType(), False),
    StructField("track", StringType(), False),
    StructField("source_id", StringType(), False),
    StructField("ts_start", TimestampType(), False),
    StructField("ts_end", TimestampType(), False),
    StructField("text", StringType(), False),
    StructField("ingest_ts", TimestampType(), False),
])

CorpusChunk = StructType([
    StructField("record_id", StringType(), False),
    StructField("track", StringType(), False),
    StructField("source_id", StringType(), False),
    StructField("ts_start", TimestampType(), False),
    StructField("ts_end", TimestampType(), False),
    StructField("text", StringType(), False),
    StructField("ingest_ts", TimestampType(), False),
    StructField("license_class", StringType(), False),
])

MAX_CHUNK_DURATION_SEC = 30
ALLOWED_TRACKS = {"ASR", "VDB"}
ALLOWED_LICENSE_CLASSES = {"public-domain", "cc-by", "cc-by-sa", "licensed-internal"}

def validate_python_record(row: dict, *, track_whitelist=ALLOWED_TRACKS,
                           allowed_licenses=ALLOWED_LICENSE_CLASSES) -> list[str]:
    """Return a list of contract violations; empty means the record passes."""
    errors = []
    required = ["record_id", "track", "source_id", "ts_start", "ts_end", "text", "ingest_ts"]
    for key in required:
        if row.get(key) is None or row.get(key) == "":
            errors.append(f"{key} is required")
    if row.get("track") not in track_whitelist:
        errors.append("track is not whitelisted")
    if row.get("track") == "VDB" and row.get("license_class") not in allowed_licenses:
        errors.append("license_class is missing or not allowed for VDB")
    if row.get("track") == "ASR" and not row.get("source_id"):
        errors.append("ASR provenance/source_id is required")
    try:
        from datetime import datetime
        def parse(x):
            return datetime.fromisoformat(x.replace("Z", "+00:00"))
        start, end = parse(row["ts_start"]), parse(row["ts_end"])
        duration = (end - start).total_seconds()
        if not start < end:
            errors.append("ts_start must be earlier than ts_end")
        if duration > MAX_CHUNK_DURATION_SEC:
            errors.append("chunk_duration_sec must be <= 30")
    except Exception:
        errors.append("ts_start and ts_end must be ISO-8601 timestamps")
    return errors
