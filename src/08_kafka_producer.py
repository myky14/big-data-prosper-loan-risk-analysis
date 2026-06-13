KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "prosper_loan_stream"

# Spark 3.5.x usually uses: org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0
SPARK_KAFKA_PACKAGE = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0"

LOCAL_PROCESSED_DATA_DIR = os.path.join("data", "processed", "06_ml_classification")
STREAM_JSON_DIR = os.path.join(
    LOCAL_PROCESSED_DATA_DIR,
    "streaming_json_sample",
)

HDFS_STREAM_SCHEMA_PATH = (
    "hdfs://localhost:9000/bigdata/prosper_loan/streaming/"
    "classification_simulation_data"
)

HDFS_BEST_CLASSIFICATION_MODEL_PATH = (
    "hdfs://localhost:9000/bigdata/prosper_loan/models/classification/"
    "best_classification_pipeline_model"
)

HDFS_STREAM_OUTPUT_PATH = (
    "hdfs://localhost:9000/bigdata/prosper_loan/streaming/"
    "classification_prediction_output"
)

HDFS_STREAM_CHECKPOINT_PATH = (
    "hdfs://localhost:9000/bigdata/prosper_loan/streaming/"
    "classification_prediction_checkpoint"
)