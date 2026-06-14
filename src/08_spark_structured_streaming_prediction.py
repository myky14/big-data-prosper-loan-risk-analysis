"""
FILE 09 - Spark Structured Streaming Prediction for Prosper Loan Risk

Purpose:
    Read streaming loan records from Kafka topic prosper_loan_stream,
    load the best classification PipelineModel saved by file 06,
    predict Good Loan / Bad Loan, and write prediction results to HDFS.

Flow:
    Kafka Producer
        -> Kafka topic: prosper_loan_stream
        -> Spark Structured Streaming
        -> Best Classification PipelineModel
        -> Prediction output to HDFS
"""

import argparse

from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, from_json, lit, when


DEFAULT_MODEL_PATH = (
    "hdfs://localhost:9000/bigdata/prosper_loan/models/classification/"
    "best_classification_pipeline_model"
)

DEFAULT_SAMPLE_JSON_DIR = (
    "data/processed/06_ml_classification/streaming_json_sample"
)

DEFAULT_OUTPUT_PATH = (
    "hdfs://localhost:9000/bigdata/prosper_loan/streaming/"
    "prediction_output"
)

DEFAULT_CHECKPOINT_PATH = (
    "hdfs://localhost:9000/bigdata/prosper_loan/streaming/"
    "checkpoint_prediction"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bootstrap-server",
        default="localhost:9092",
        help="Kafka bootstrap server."
    )

    parser.add_argument(
        "--topic",
        default="prosper_loan_stream",
        help="Kafka topic name."
    )

    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="HDFS path of the saved best classification PipelineModel."
    )

    parser.add_argument(
        "--sample-json-dir",
        default=DEFAULT_SAMPLE_JSON_DIR,
        help="Local folder containing sample JSON files used to infer schema."
    )

    parser.add_argument(
        "--output-path",
        default=DEFAULT_OUTPUT_PATH,
        help="HDFS output path for streaming prediction results."
    )

    parser.add_argument(
        "--checkpoint-path",
        default=DEFAULT_CHECKPOINT_PATH,
        help="HDFS checkpoint path for Spark Structured Streaming."
    )

    parser.add_argument(
        "--trigger-seconds",
        type=int,
        default=5,
        help="Micro-batch trigger interval in seconds."
    )

    return parser.parse_args()


def create_spark_session():
    spark = (
        SparkSession.builder
        .appName("Prosper_Loan_Structured_Streaming_Prediction")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    return spark


def infer_input_schema(spark, sample_json_dir):
    print("=" * 88)
    print("INFER STREAMING INPUT SCHEMA")
    print("=" * 88)
    print(f"Sample JSON directory: {sample_json_dir}")

    sample_df = spark.read.json(sample_json_dir)

    print("Inferred schema:")
    sample_df.printSchema()

    return sample_df.schema


def read_kafka_stream(spark, bootstrap_server, topic):
    print("=" * 88)
    print("READ STREAM FROM KAFKA")
    print("=" * 88)
    print(f"Bootstrap server: {bootstrap_server}")
    print(f"Topic: {topic}")

    kafka_df = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_server)
        .option("subscribe", topic)
        .option("startingOffsets", "latest")
        .load()
    )

    return kafka_df


def parse_kafka_json(kafka_df, input_schema):
    parsed_df = (
        kafka_df
        .selectExpr("CAST(value AS STRING) AS json_value")
        .select(from_json(col("json_value"), input_schema).alias("data"))
        .select("data.*")
    )

    return parsed_df


def add_prediction_columns(prediction_df):
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

    return result_df


def write_prediction_to_console(result_df, trigger_seconds):
    console_query = (
        result_df
        .select(
            "prediction_time",
            "prediction",
            "prediction_label",
            "good_loan_probability",
            "bad_loan_probability"
        )
        .writeStream
        .outputMode("append")
        .format("console")
        .option("truncate", "false")
        .trigger(processingTime=f"{trigger_seconds} seconds")
        .start()
    )

    return console_query


def write_prediction_to_hdfs(result_df, output_path, checkpoint_path, trigger_seconds):
    hdfs_query = (
        result_df
        .writeStream
        .outputMode("append")
        .format("parquet")
        .option("path", output_path)
        .option("checkpointLocation", checkpoint_path)
        .trigger(processingTime=f"{trigger_seconds} seconds")
        .start()
    )

    return hdfs_query


def main():
    args = parse_args()

    spark = create_spark_session()

    input_schema = infer_input_schema(spark, args.sample_json_dir)

    print("=" * 88)
    print("LOAD BEST CLASSIFICATION MODEL")
    print("=" * 88)
    print(f"Model path: {args.model_path}")

    model = PipelineModel.load(args.model_path)

    kafka_df = read_kafka_stream(
        spark=spark,
        bootstrap_server=args.bootstrap_server,
        topic=args.topic
    )

    stream_input_df = parse_kafka_json(kafka_df, input_schema)

    prediction_df = model.transform(stream_input_df)

    result_df = add_prediction_columns(prediction_df)

    console_query = write_prediction_to_console(
        result_df=result_df,
        trigger_seconds=args.trigger_seconds
    )

    hdfs_query = write_prediction_to_hdfs(
        result_df=result_df,
        output_path=args.output_path,
        checkpoint_path=args.checkpoint_path,
        trigger_seconds=args.trigger_seconds
    )

    print("=" * 88)
    print("STREAMING PREDICTION STARTED")
    print("=" * 88)
    print(f"Prediction output path: {args.output_path}")
    print(f"Checkpoint path: {args.checkpoint_path}")
    print("Press Ctrl + C to stop streaming.")

    try:
        hdfs_query.awaitTermination()
    except KeyboardInterrupt:
        print("Stopping streaming queries...")
        console_query.stop()
        hdfs_query.stop()
        spark.stop()
        print("Streaming stopped.")


if __name__ == "__main__":
    main()