"""Calculate p50/p95/p99 of ingest_ts - ts_end from landing JSON files."""
import argparse
from pyspark.sql import SparkSession, functions as F, types as T
schema = T.StructType([
    T.StructField("record_id", T.StringType(), True),
    T.StructField("track", T.StringType(), True),
    T.StructField("source_id", T.StringType(), True),
    T.StructField("ts_start", T.TimestampType(), True),
    T.StructField("ts_end", T.TimestampType(), True),
    T.StructField("text", T.StringType(), True),
    T.StructField("ingest_ts", T.TimestampType(), True),
    T.StructField("license_class", T.StringType(), True),
])
p=argparse.ArgumentParser(); p.add_argument("--input",default="data/landing"); a=p.parse_args()
s=SparkSession.builder.appName("MP5-lateness-calibration").config("spark.sql.session.timeZone","UTC").getOrCreate()
d=s.read.schema(schema).json(a.input).filter(F.col("ts_end").isNotNull() & F.col("ingest_ts").isNotNull())
d=d.withColumn("lateness_sec",F.col("ingest_ts").cast("long")-F.col("ts_end").cast("long"))
print("count =",d.count())
print("p50,p95,p99 =",d.approxQuantile("lateness_sec",[.5,.95,.99],.001))
s.stop()
