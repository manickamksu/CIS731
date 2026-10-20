"""Structured Streaming pipeline: event-time windows, progress metrics, restartable checkpoints."""
import argparse, json, os, re, time
from pathlib import Path
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.sql.functions import pandas_udf
import pandas as pd

from contracts import TranscriptChunk, CorpusChunk

# Common streaming schema is nullable during parsing so invalid rows can be quarantined.
# The explicit contract schemas above remain non-nullable for valid records.
INPUT_SCHEMA = T.StructType([
    T.StructField("record_id", T.StringType(), True),
    T.StructField("track", T.StringType(), True),
    T.StructField("source_id", T.StringType(), True),
    T.StructField("ts_start", T.TimestampType(), True),
    T.StructField("ts_end", T.TimestampType(), True),
    T.StructField("text", T.StringType(), True),
    T.StructField("ingest_ts", T.TimestampType(), True),
    T.StructField("license_class", T.StringType(), True),
])

ALLOWED_TRACKS = ["ASR", "VDB"]
ALLOWED_LICENSES = ["public-domain", "cc-by", "cc-by-sa", "licensed-internal"]

def top_terms_and_distinct(texts):
    """Bounded per-window summarization. Text list is held until watermark closes a window."""
    counts = {}
    unique = set()
    for text in texts or []:
        for token in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", (text or "").lower()):
            unique.add(token)
            counts[token] = counts.get(token, 0) + 1
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    return len(unique), [{"term": t, "count": c} for t, c in top]

summary_schema = T.StructType([
    T.StructField("approx_distinct_tokens", T.LongType(), False),
    T.StructField("top10_terms", T.ArrayType(T.StructType([
        T.StructField("term", T.StringType(), False),
        T.StructField("count", T.LongType(), False),
    ])), False),
])

@pandas_udf(summary_schema)
def summarize_texts(text_lists: pd.Series) -> pd.DataFrame:
    distincts, tops = [], []
    for texts in text_lists:
        d, t = top_terms_and_distinct(texts)
        distincts.append(int(d))
        tops.append(t)
    return pd.DataFrame({"approx_distinct_tokens": distincts, "top10_terms": tops})

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/landing")
    p.add_argument("--output", default="data/output")
    p.add_argument("--checkpoint", default="data/checkpoints")
    p.add_argument("--metrics", default="metrics.jsonl")
    p.add_argument("--watermark", default="2 minutes")
    p.add_argument("--once", action="store_true", help="Process currently available files and stop")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.checkpoint, exist_ok=True)
    spark = (SparkSession.builder.appName("MP5-EventTime-Windows")
             .config("spark.sql.shuffle.partitions", "2")
             .config("spark.sql.session.timeZone", "UTC")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    raw = (spark.readStream.format("json")
           .schema(INPUT_SCHEMA)
           .option("maxFilesPerTrigger", 1)
           .load(args.input))

    # Deal gates: required fields, whitelisted track, valid event-time range,
    # <=30-second chunk, provenance source_id, and VDB license allow-list.
    valid = raw.filter(
        F.col("record_id").isNotNull() & (F.length(F.trim("record_id")) > 0) &
        F.col("track").isin(ALLOWED_TRACKS) &
        F.col("source_id").isNotNull() & (F.length(F.trim("source_id")) > 0) &
        F.col("ts_start").isNotNull() & F.col("ts_end").isNotNull() &
        (F.col("ts_start") < F.col("ts_end")) &
        (F.col("ts_end").cast("long") - F.col("ts_start").cast("long") <= 30) &
        F.col("ingest_ts").isNotNull() &
        F.col("text").isNotNull() & (F.length(F.trim("text")) > 0) &
        ((F.col("track") != "VDB") | F.col("license_class").isin(ALLOWED_LICENSES))
    ).withColumn("event_ts", F.col("ts_end")) \
     .withColumn("ingest_latency_sec", F.col("ingest_ts").cast("long") - F.col("ts_end").cast("long"))

    # Measure lateness separately from stream processing. This is a bounded batch
    # calculation on the landing zone and should be run before choosing final watermark.
    # The watermark must be >= the chosen tail of ingest_ts - ts_end.
    lateness_path = Path(args.input)
    try:
        batch = spark.read.schema(INPUT_SCHEMA).json(str(lateness_path))
        batch = batch.filter(F.col("ts_end").isNotNull() & F.col("ingest_ts").isNotNull()) \
                     .withColumn("lateness_sec", F.col("ingest_ts").cast("long") - F.col("ts_end").cast("long"))
        q = batch.approxQuantile("lateness_sec", [0.50, 0.95, 0.99], 0.001)
        print("LANDING_LATENESS_SEC p50,p95,p99 =", q)
    except Exception as e:
        print("Could not calculate batch lateness percentiles:", str(e))

    # For each query, collect_list(text) lets a window-level summarizer calculate
    # exact distinct token count and top-10 terms. This is intentionally bounded by
    # the watermark but may consume memory for high-volume windows; see report.
    def window_query(window_duration, slide_duration, name):
        grouped = (valid.withWatermark("event_ts", args.watermark)
            .groupBy(F.window("event_ts", window_duration, slide_duration))
            .agg(
                F.count("*").alias("record_count"),
                F.collect_list("text").alias("_texts"),
                F.avg("ingest_latency_sec").alias("mean_ingest_latency_sec"),
            )
            .withColumn("summary", summarize_texts("_texts"))
            .select(
                F.col("window.start").alias("window_start"),
                F.col("window.end").alias("window_end"),
                "record_count",
                F.col("summary.approx_distinct_tokens").alias("approx_count_distinct_tokens"),
                F.col("summary.top10_terms").alias("top10_terms"),
                "mean_ingest_latency_sec",
            )
        )
        writer = (grouped.writeStream
            .format("parquet")
            .outputMode("append")
            .option("path", str(Path(args.output) / name))
            .option("checkpointLocation", str(Path(args.checkpoint) / name))
            .queryName(f"mp5_{name}")
            .trigger(availableNow=True) if args.once else
            grouped.writeStream.format("parquet").outputMode("append")
            .option("path", str(Path(args.output) / name))
            .option("checkpointLocation", str(Path(args.checkpoint) / name))
            .queryName(f"mp5_{name}"))
        return writer.start()

    q1 = window_query("60 seconds", "60 seconds", "tumbling")
    q2 = window_query("5 minutes", "1 minute", "sliding")
    metrics_path = Path(args.metrics)
    seen = set()

    def save_progress(query):
        for progress in query.recentProgress:
            key = (query.name, progress.get("batchId"))
            if key in seen:
                continue
            seen.add(key)
            event = {
                "queryName": query.name,
                "timestamp": progress.get("timestamp"),
                "batchId": progress.get("batchId"),
                "numInputRows": progress.get("numInputRows"),
                "inputRowsPerSecond": progress.get("inputRowsPerSecond"),
                "processedRowsPerSecond": progress.get("processedRowsPerSecond"),
                "durationMs": progress.get("durationMs", {}),
                "stateOperators": progress.get("stateOperators", []),
                "eventTime": progress.get("eventTime", {}),
            }
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")

    try:
        if args.once:
            # availableNow can finish before a polling loop gets scheduled.
            q1.awaitTermination()
            q2.awaitTermination()
            save_progress(q1)
            save_progress(q2)
        else:
            while q1.isActive or q2.isActive:
                save_progress(q1)
                save_progress(q2)
                time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupted by user. Restart with the same checkpoint paths.")
    finally:
        for query in (q1, q2):
            try:
                query.stop()
                save_progress(query)
            except Exception:
                pass
        spark.stop()

if __name__ == "__main__":
    main()
