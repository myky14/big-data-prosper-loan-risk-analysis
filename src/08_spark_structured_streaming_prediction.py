"""
FILE 08 - Spark Structured Streaming Prediction for Prosper Loan Risk

Purpose:
    Read streaming loan records from Kafka topic prosper_loan_stream,
    load the best classification PipelineModel saved by Source 06,
    predict Good Loan / Bad Loan, and write prediction results to HDFS.

Execution:
    This source code is designed to be copied into main.ipynb and run as one cell.

Important:
    If Sources 00-07 were already run in the same notebook, restart the kernel
    before running this Source 08 cell. Spark must load the Kafka connector before
    SparkSession is created.

Before running this cell:
    1. HDFS must be running.
    2. ZooKeeper must be running in a separate terminal.
    3. Kafka server must be running in another separate terminal.
    4. Source 06 must already save the best classification model and streaming JSON sample.

After this cell prints:
    STREAMING PREDICTION STARTED

Open a new terminal at the project root and run:
    python src\\09_kafka_producer.py
    or for testing:
    python src\\09_kafka_producer.py --limit 10 --sleep 0.1
"""

import os
import sys

# ============================================================
# 0. PYSPARK PACKAGE CONFIGURATION
# ============================================================
# This must be set BEFORE importing SparkSession / any PySpark modules.
# Otherwise Spark cannot find the Kafka data source in notebook mode.

SPARK_KAFKA_PACKAGE = "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1"

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["PYSPARK_SUBMIT_ARGS"] = f"--packages {SPARK_KAFKA_PACKAGE} pyspark-shell"


from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, from_json, lit, when


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = os.path.abspath(os.getcwd())

SPARK_KAFKA_PACKAGE = "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1"

KAFKA_BOOTSTRAP_SERVER = "localhost:9092"
KAFKA_TOPIC = "prosper_loan_stream"

MODEL_PATH = (
    "hdfs://localhost:9000/bigdata/prosper_loan/models/classification/"
    "best_classification_pipeline_model"
)

SAMPLE_JSON_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "processed",
    "06_ml_classification",
    "streaming_json_sample"
)

OUTPUT_PATH = (
    "hdfs://localhost:9000/bigdata/prosper_loan/streaming/"
    "prediction_output"
)

CHECKPOINT_PATH = (
    "hdfs://localhost:9000/bigdata/prosper_loan/streaming/"
    "checkpoint_prediction"
)

TRIGGER_SECONDS = 5


# Force Spark to use the same Python executable as this notebook
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


# ============================================================
# 2. STOP OLD SPARK SESSION IF EXISTS
# ============================================================

try:
    active_spark = SparkSession.getActiveSession()
    if active_spark is not None:
        print("Stopping existing SparkSession before starting streaming session...")
        active_spark.stop()
except Exception as error:
    print(f"Warning while stopping existing SparkSession: {error}")


# ============================================================
# 3. CREATE SPARK SESSION WITH KAFKA PACKAGE
# ============================================================

print("=" * 88)
print("SOURCE 08 - SPARK STRUCTURED STREAMING WITH KAFKA")
print("=" * 88)

print("\nProject directory:")
print(PROJECT_DIR)

print("\nKafka topic:")
print(KAFKA_TOPIC)

print("\nKafka bootstrap server:")
print(KAFKA_BOOTSTRAP_SERVER)

print("\nModel path:")
print(MODEL_PATH)

print("\nSample JSON directory:")
print(SAMPLE_JSON_DIR)

print("\nSpark Kafka package:")
print(SPARK_KAFKA_PACKAGE)

if not os.path.exists(SAMPLE_JSON_DIR):
    raise FileNotFoundError(
        f"Sample JSON directory not found: {SAMPLE_JSON_DIR}\n"
        "Please run Source 06 first."
    )

spark = (
    SparkSession.builder
    .appName("Prosper_Loan_Structured_Streaming_Prediction")
    .config("spark.jars.packages", SPARK_KAFKA_PACKAGE)
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 4. INFER STREAMING INPUT SCHEMA
# ============================================================

print("\n" + "=" * 88)
print("INFER STREAMING INPUT SCHEMA")
print("=" * 88)

sample_df = spark.read.json(SAMPLE_JSON_DIR)

print("Inferred schema:")
sample_df.printSchema()

input_schema = sample_df.schema


# ============================================================
# 5. LOAD BEST CLASSIFICATION MODEL
# ============================================================

print("\n" + "=" * 88)
print("LOAD BEST CLASSIFICATION MODEL")
print("=" * 88)

model = PipelineModel.load(MODEL_PATH)

print("Best classification model loaded successfully.")


# ============================================================
# 6. READ STREAM FROM KAFKA
# ============================================================

print("\n" + "=" * 88)
print("READ STREAM FROM KAFKA")
print("=" * 88)

kafka_df = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVER)
    .option("subscribe", KAFKA_TOPIC)
    .option("startingOffsets", "earliest")
    .load()
)


# ============================================================
# 7. PARSE JSON MESSAGE FROM KAFKA
# ============================================================

stream_input_df = (
    kafka_df
    .selectExpr("CAST(value AS STRING) AS json_value")
    .select(from_json(col("json_value"), input_schema).alias("data"))
    .select("data.*")
)


# ============================================================
# 8. APPLY MODEL PREDICTION
# ============================================================

prediction_df = model.transform(stream_input_df)

result_df = (
    prediction_df
    .withColumn("probability_array", vector_to_array(col("probability")))
    .withColumn("bad_loan_probability", col("probability_array")[1])
    .withColumn("good_loan_probability", col("probability_array")[0])
    .withColumn(
        "prediction_label",
        when(col("prediction") == 1.0, lit("Bad Loan"))
        .otherwise(lit("Good Loan"))
    )
    .withColumn("prediction_time", current_timestamp())
    .drop("rawPrediction", "probability", "probability_array", "features")
)


# ============================================================
# 9. WRITE STREAMING OUTPUT WITH FOREACHBATCH
# ============================================================

def write_prediction_batch(batch_df, batch_id):
    print("\n" + "-" * 88)
    print(f"BATCH ID: {batch_id}")
    print("-" * 88)

    record_count = batch_df.count()
    print(f"Number of prediction records in this batch: {record_count}")

    if record_count > 0:
        print("\nPrediction result preview:")

        (
            batch_df
            .select(
                "prediction_time",
                "prediction",
                "prediction_label",
                "good_loan_probability",
                "bad_loan_probability"
            )
            .show(20, truncate=False)
        )

        (
            batch_df
            .write
            .mode("append")
            .parquet(OUTPUT_PATH)
        )

        print(f"\nBatch {batch_id} written to HDFS:")
        print(OUTPUT_PATH)
    else:
        print("No records in this batch.")


prediction_query = (
    result_df
    .writeStream
    .foreachBatch(write_prediction_batch)
    .outputMode("append")
    .option("checkpointLocation", CHECKPOINT_PATH)
    .trigger(processingTime=f"{TRIGGER_SECONDS} seconds")
    .start()
)


# ============================================================
# 10. KEEP STREAMING JOB RUNNING
# ============================================================

print("\n" + "=" * 88)
print("STREAMING PREDICTION STARTED")
print("=" * 88)
print(f"Prediction output path: {OUTPUT_PATH}")
print(f"Checkpoint path: {CHECKPOINT_PATH}")
print()
print("Now open a NEW terminal at the project root and run:")
print(r"python src\09_kafka_producer.py --limit 10 --sleep 0.1")
print()
print("This cell will keep running. Interrupt this cell to stop streaming.")
print("=" * 88)

try:
    prediction_query.awaitTermination()
except KeyboardInterrupt:
    print("Stopping streaming query...")
    prediction_query.stop()
    spark.stop()
    print("Streaming stopped.")