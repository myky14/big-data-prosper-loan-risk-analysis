# -*- coding: utf-8 -*-
"""
FILE 06 - FEATURE SELECTION + CLASSIFICATION MODELING PIPELINE
Project: Prosper Loan Risk Analysis using Hadoop + Spark

Purpose
-------
This file runs one complete Data Science flow for the classification task:

1. Read preprocessed Good Loan / Bad Loan dataset from HDFS.
2. Validate label distribution and class imbalance.
3. Create a stratified train/test split.
4. Perform supervised Feature Selection on TRAINING DATA ONLY:
   - Chi-Square Test after discretizing numeric variables.
   - Random Forest Feature Importance for nonlinear signal.
   - Combined normalized score = 50% Chi-Square + 50% RF importance.
5. Train Logistic Regression, Random Forest, and GBT models using selected features.
6. Evaluate models with Accuracy, F1, ROC-AUC, PR-AUC, Bad Loan Recall, and threshold analysis.
7. Save report-ready tables and visualizations.
8. Save the best Spark PipelineModel to HDFS.

Why feature selection is inside this file
----------------------------------------
Putting Feature Selection and model training in one script makes the workflow easier
to run and explain in the report. The feature selection step is still methodologically
valid because it is fitted only on the training set, so the test set remains unseen
until final evaluation.

Run
---
    spark-submit src/06_feature_selection_ml_pipeline.py

Expected input
--------------
    hdfs://localhost:9000/bigdata/prosper_loan/processed/prosper_loan_preprocessed_classification

Expected input columns include:
    label = 0 for Good Loan
    label = 1 for Bad Loan

Main local outputs
------------------
    outputs/tables/feature_selection_modeling/feature_selection_scores.csv
    outputs/tables/feature_selection_modeling/selected_features.csv
    outputs/tables/feature_selection_modeling/model_metrics.csv
    outputs/tables/feature_selection_modeling/threshold_analysis.csv
    outputs/tables/feature_selection_modeling/confusion_matrices.csv
    outputs/tables/feature_selection_modeling/final_feature_importance.csv

Main figure outputs
-------------------
    outputs/figures/feature_selection_modeling/01_label_distribution.png
    outputs/figures/feature_selection_modeling/02_feature_selection_scores.png
    outputs/figures/feature_selection_modeling/03_selected_vs_removed.png
    outputs/figures/feature_selection_modeling/04_model_metric_comparison.png
    outputs/figures/feature_selection_modeling/05_threshold_tradeoff_best_model.png
    outputs/figures/feature_selection_modeling/06_confusion_matrix_best_model.png
    outputs/figures/feature_selection_modeling/07_feature_importance_best_model.png
"""

import csv
import json
import math
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

from pyspark import StorageLevel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import GBTClassifier, LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import (
    OneHotEncoder,
    QuantileDiscretizer,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)
from pyspark.ml.functions import vector_to_array
from pyspark.ml.stat import ChiSquareTest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lit, monotonically_increasing_id, when
from pyspark.sql.types import BooleanType, NumericType, StringType

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42
TRAIN_RATIO = 0.80

HDFS_INPUT_PATH = (
    "hdfs://localhost:9000/bigdata/prosper_loan/processed/"
    "prosper_loan_preprocessed_classification"
)

HDFS_SELECTED_DATA_OUTPUT_PATH = (
    "hdfs://localhost:9000/bigdata/prosper_loan/processed/"
    "prosper_loan_selected_classification"
)

HDFS_BEST_MODEL_OUTPUT_PATH = (
    "hdfs://localhost:9000/bigdata/prosper_loan/models/classification/"
    "best_feature_selected_model"
)

LOCAL_TABLE_DIR = os.path.join("outputs", "tables", "feature_selection_modeling")
LOCAL_FIGURE_DIR = os.path.join("outputs", "figures", "feature_selection_modeling")

FEATURE_SELECTION_SCORES_CSV = os.path.join(LOCAL_TABLE_DIR, "feature_selection_scores.csv")
SELECTED_FEATURES_CSV = os.path.join(LOCAL_TABLE_DIR, "selected_features.csv")
MODEL_METRICS_CSV = os.path.join(LOCAL_TABLE_DIR, "model_metrics.csv")
THRESHOLD_ANALYSIS_CSV = os.path.join(LOCAL_TABLE_DIR, "threshold_analysis.csv")
CONFUSION_MATRIX_CSV = os.path.join(LOCAL_TABLE_DIR, "confusion_matrices.csv")
FINAL_FEATURE_IMPORTANCE_CSV = os.path.join(LOCAL_TABLE_DIR, "final_feature_importance.csv")
METADATA_JSON = os.path.join(LOCAL_TABLE_DIR, "run_metadata.json")

# Number of raw variables to keep after feature selection.
# With the current 20 feature columns, keeping 12 gives real selection but does not
# remove too much signal.
TOP_N_FEATURES = 12

# Feature selection model settings.
FS_RF_NUM_TREES = 80
FS_RF_MAX_DEPTH = 6
FS_CHI_NUM_BUCKETS = 5

# Final model settings.
RF_NUM_TREES = 120
RF_MAX_DEPTH = 6
GBT_MAX_ITER = 60
GBT_MAX_DEPTH = 5
LR_MAX_ITER = 80

# If True, exclude variables that may act as pricing/internal-risk proxy features.
# Set False to keep the current report direction and compare all available features.
# Set True for a stricter real-world credit approval scenario.
EXCLUDE_PROXY_RISK_FEATURES = False
PROXY_RISK_FEATURES = [
    "BorrowerAPR",
    "EstimatedLoss",
    "ProsperScore",
    "BorrowerRate",
    "LenderYield",
    "EstimatedReturn",
    "EstimatedEffectiveYield",
]

ALWAYS_DROP_COLUMNS = [
    "label",
    "LoanStatus",
    "loan_status",
    "loan_outcome",
    "features",
    "raw_features",
    "scaled_features",
    "selected_features",
    "prediction",
    "rawPrediction",
    "probability",
    "prob_bad",
    "row_id",
    "class_weight",
]

# Business thresholds for Bad Loan probability.
DECISION_THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

# Main metric for final model selection.
# Options: pr_auc, roc_auc, bad_f1_best_threshold, bad_recall_best_threshold, f1, balanced_accuracy
MODEL_SELECTION_METRIC = "pr_auc"

# Save artifacts.
SAVE_SELECTED_DATASET_TO_HDFS = True
SAVE_BEST_MODEL_TO_HDFS = True

MAX_PROBABILITY_POINTS = 8000


# =============================================================================
# BASIC UTILITIES
# =============================================================================


def create_spark_session() -> SparkSession:
    spark = (
        SparkSession.builder
        .appName("ProsperLoan_File06_FeatureSelection_Classification")
        .config("spark.sql.shuffle.partitions", "100")
        .config("spark.sql.debug.maxToStringFields", "200")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def print_header(title: str) -> None:
    line = "=" * 90
    print(f"\n{line}\n{title}\n{line}")


def print_subheader(title: str) -> None:
    print(f"\n--- {title} ---")


def ensure_output_dirs() -> None:
    os.makedirs(LOCAL_TABLE_DIR, exist_ok=True)
    os.makedirs(LOCAL_FIGURE_DIR, exist_ok=True)


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator is None or denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default


def round_float(value: float, digits: int = 6) -> float:
    return round(safe_float(value), digits)


def write_dicts_to_csv(rows: List[Dict], output_path: str) -> None:
    if not rows:
        print(f"No rows to write: {output_path}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(output_path, "w", newline="", encoding="utf-8-sig") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    print(f"Saved CSV: {output_path}")


def write_json(obj: Dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(obj, file_obj, indent=2, ensure_ascii=False)
    print(f"Saved JSON: {output_path}")


def get_existing_columns(df: DataFrame, columns: Iterable[str]) -> List[str]:
    return [column_name for column_name in columns if column_name in df.columns]


def sanitize_name(name: str, used_names: set) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "_", str(name).strip())
    safe = re.sub(r"_+", "_", safe).strip("_")
    if not safe:
        safe = "feature"
    if safe[0].isdigit():
        safe = f"f_{safe}"

    base = safe
    counter = 1
    while safe in used_names:
        counter += 1
        safe = f"{base}_{counter}"
    used_names.add(safe)
    return safe


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================


def save_current_figure(path: str) -> None:
    if not HAS_MATPLOTLIB:
        return
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {path}")


def plot_label_distribution(label_rows: List[Dict]) -> None:
    if not HAS_MATPLOTLIB or not label_rows:
        return
    labels = [row["meaning"] for row in label_rows]
    counts = [row["count"] for row in label_rows]

    plt.figure(figsize=(7, 4.5))
    plt.bar(labels, counts)
    plt.title("Label Distribution: Good Loan vs Bad Loan")
    plt.xlabel("Loan outcome")
    plt.ylabel("Number of loans")
    for idx, value in enumerate(counts):
        plt.text(idx, value, f"{value:,}", ha="center", va="bottom", fontsize=10)
    save_current_figure(os.path.join(LOCAL_FIGURE_DIR, "01_label_distribution.png"))


def plot_feature_selection_scores(score_rows: List[Dict], top_n: int = 15) -> None:
    if not HAS_MATPLOTLIB or not score_rows:
        return
    rows = sorted(score_rows, key=lambda row: row["combined_score"], reverse=True)[:top_n]
    rows = list(reversed(rows))
    features = [row["feature"] for row in rows]
    scores = [row["combined_score"] for row in rows]

    plt.figure(figsize=(10, 6))
    plt.barh(features, scores)
    plt.title("Top Feature Selection Scores")
    plt.xlabel("Combined normalized score")
    plt.ylabel("Raw feature")
    save_current_figure(os.path.join(LOCAL_FIGURE_DIR, "02_feature_selection_scores.png"))


def plot_selected_vs_removed(selected_features: List[str], removed_features: List[str]) -> None:
    if not HAS_MATPLOTLIB:
        return
    labels = ["Selected", "Removed"]
    values = [len(selected_features), len(removed_features)]

    plt.figure(figsize=(6.5, 4.5))
    plt.bar(labels, values)
    plt.title("Feature Selection Result")
    plt.xlabel("Feature group")
    plt.ylabel("Number of raw features")
    for idx, value in enumerate(values):
        plt.text(idx, value, str(value), ha="center", va="bottom", fontsize=10)
    save_current_figure(os.path.join(LOCAL_FIGURE_DIR, "03_selected_vs_removed.png"))


def plot_model_metric_comparison(model_rows: List[Dict]) -> None:
    if not HAS_MATPLOTLIB or not model_rows:
        return

    metrics = ["accuracy", "f1", "roc_auc", "pr_auc", "bad_recall_best_threshold"]
    metric_labels = ["Accuracy", "F1", "ROC-AUC", "PR-AUC", "Bad recall"]
    model_names = [row["model"] for row in model_rows]
    x = list(range(len(model_names)))
    width = 0.15

    plt.figure(figsize=(11, 6))
    for i, metric in enumerate(metrics):
        values = [safe_float(row.get(metric, 0.0)) for row in model_rows]
        positions = [pos + (i - 2) * width for pos in x]
        plt.bar(positions, values, width=width, label=metric_labels[i])

    plt.xticks(x, model_names, rotation=15, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Metric value")
    plt.title("Model Metric Comparison")
    plt.legend()
    save_current_figure(os.path.join(LOCAL_FIGURE_DIR, "04_model_metric_comparison.png"))


def plot_threshold_tradeoff(threshold_rows: List[Dict], model_name: str) -> None:
    if not HAS_MATPLOTLIB or not threshold_rows:
        return

    rows = [row for row in threshold_rows if row["model"] == model_name]
    if not rows:
        return

    thresholds = [row["threshold"] for row in rows]
    precision = [row["bad_precision"] for row in rows]
    recall = [row["bad_recall"] for row in rows]
    f1 = [row["bad_f1"] for row in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precision, marker="o", label="Bad Loan precision")
    plt.plot(thresholds, recall, marker="o", label="Bad Loan recall")
    plt.plot(thresholds, f1, marker="o", label="Bad Loan F1")
    plt.xlabel("Bad Loan probability threshold")
    plt.ylabel("Metric value")
    plt.ylim(0, 1)
    plt.title(f"Threshold Trade-off: {model_name}")
    plt.legend()
    save_current_figure(os.path.join(LOCAL_FIGURE_DIR, "05_threshold_tradeoff_best_model.png"))


def plot_confusion_matrix(confusion: Dict[str, int], model_name: str, threshold: float) -> None:
    if not HAS_MATPLOTLIB:
        return

    matrix = [
        [confusion.get("tn", 0), confusion.get("fp", 0)],
        [confusion.get("fn", 0), confusion.get("tp", 0)],
    ]

    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(matrix, interpolation="nearest")
    plt.title(f"Confusion Matrix: {model_name} at threshold {threshold:.2f}")
    plt.xticks([0, 1], ["Pred Good", "Pred Bad"])
    plt.yticks([0, 1], ["Actual Good", "Actual Bad"])
    plt.colorbar(fraction=0.046, pad=0.04)

    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{matrix[i][j]:,}", ha="center", va="center", fontsize=12)

    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    save_current_figure(os.path.join(LOCAL_FIGURE_DIR, "06_confusion_matrix_best_model.png"))


def plot_feature_importance(rows: List[Dict], model_name: str, top_n: int = 15) -> None:
    if not HAS_MATPLOTLIB or not rows:
        return

    model_rows = [row for row in rows if row.get("model") == model_name]
    if not model_rows:
        return

    top_rows = sorted(model_rows, key=lambda row: abs(safe_float(row["importance"])), reverse=True)[:top_n]
    top_rows = list(reversed(top_rows))
    features = [row["feature"] for row in top_rows]
    values = [safe_float(row["importance"]) for row in top_rows]

    plt.figure(figsize=(10, 6))
    plt.barh(features, values)
    plt.title(f"Top Feature Importance: {model_name}")
    plt.xlabel("Importance / absolute coefficient")
    plt.ylabel("Feature")
    save_current_figure(os.path.join(LOCAL_FIGURE_DIR, "07_feature_importance_best_model.png"))


# =============================================================================
# DATA LOADING, VALIDATION, AND SPLITTING
# =============================================================================


def load_dataset(spark: SparkSession) -> DataFrame:
    print_header("LOAD PREPROCESSED CLASSIFICATION DATASET")
    print(f"Input path: {HDFS_INPUT_PATH}")
    df = spark.read.parquet(HDFS_INPUT_PATH)
    df = df.persist(StorageLevel.MEMORY_AND_DISK)
    print("Dataset loaded and persisted with MEMORY_AND_DISK.")
    return df


def validate_dataset(df: DataFrame) -> None:
    print_header("DATASET VALIDATION")
    if "label" not in df.columns:
        raise ValueError("Input dataset must contain a label column.")

    print(f"Rows: {df.count()}")
    print(f"Columns: {len(df.columns)}")
    print("Columns:")
    for i, column_name in enumerate(df.columns, start=1):
        print(f"{i:02d}. {column_name}")

    print("\nSchema:")
    df.printSchema()


def get_label_distribution(df: DataFrame) -> List[Dict]:
    total = df.count()
    rows = []
    label_counts = df.groupBy("label").count().orderBy("label").collect()
    for row in label_counts:
        label_value = int(row["label"])
        count_value = int(row["count"])
        rows.append({
            "label": label_value,
            "meaning": "Good Loan" if label_value == 0 else "Bad Loan",
            "count": count_value,
            "percentage": round_float(100.0 * safe_divide(count_value, total), 2),
        })
    return rows


def print_label_distribution(df: DataFrame) -> List[Dict]:
    print_header("LABEL DISTRIBUTION")
    rows = get_label_distribution(df)
    print(f"{'label':<8}{'meaning':<15}{'count':<12}{'percentage':<12}")
    print("-" * 50)
    for row in rows:
        print(f"{row['label']:<8}{row['meaning']:<15}{row['count']:<12}{row['percentage']}%")
    plot_label_distribution(rows)
    return rows


def stratified_train_test_split(df: DataFrame) -> Tuple[DataFrame, DataFrame]:
    """
    Split each label separately so class ratio is preserved.
    """
    print_header("STRATIFIED TRAIN / TEST SPLIT")
    positive_df = df.filter(col("label") == 1)
    negative_df = df.filter(col("label") == 0)

    pos_train, pos_test = positive_df.randomSplit([TRAIN_RATIO, 1.0 - TRAIN_RATIO], seed=SEED)
    neg_train, neg_test = negative_df.randomSplit([TRAIN_RATIO, 1.0 - TRAIN_RATIO], seed=SEED)

    train_df = pos_train.unionByName(neg_train).orderBy(F.rand(seed=SEED))
    test_df = pos_test.unionByName(neg_test).orderBy(F.rand(seed=SEED + 1))

    train_df = train_df.persist(StorageLevel.MEMORY_AND_DISK)
    test_df = test_df.persist(StorageLevel.MEMORY_AND_DISK)

    print(f"Training rows: {train_df.count()}")
    print(f"Testing rows : {test_df.count()}")
    print("\nTraining label distribution:")
    train_df.groupBy("label").count().orderBy("label").show()
    print("Testing label distribution:")
    test_df.groupBy("label").count().orderBy("label").show()

    return train_df, test_df


def add_class_weight(train_df: DataFrame, test_df: DataFrame) -> Tuple[DataFrame, DataFrame, Dict[str, float]]:
    print_header("CLASS WEIGHT FOR IMBALANCED LABELS")
    counts = {int(row["label"]): int(row["count"]) for row in train_df.groupBy("label").count().collect()}
    total = sum(counts.values())
    count_good = counts.get(0, 0)
    count_bad = counts.get(1, 0)

    if count_good == 0 or count_bad == 0:
        raise ValueError("Both Good Loan and Bad Loan classes are required for training.")

    weight_good = total / (2.0 * count_good)
    weight_bad = total / (2.0 * count_bad)

    print(f"Good Loan count: {count_good}, class weight: {weight_good:.4f}")
    print(f"Bad Loan count : {count_bad}, class weight: {weight_bad:.4f}")

    def with_weight(df: DataFrame) -> DataFrame:
        return df.withColumn(
            "class_weight",
            when(col("label") == 1, lit(float(weight_bad))).otherwise(lit(float(weight_good))),
        )

    train_weighted = with_weight(train_df).persist(StorageLevel.MEMORY_AND_DISK)
    test_weighted = with_weight(test_df).persist(StorageLevel.MEMORY_AND_DISK)

    return train_weighted, test_weighted, {
        "weight_good": weight_good,
        "weight_bad": weight_bad,
        "count_good": count_good,
        "count_bad": count_bad,
    }


# =============================================================================
# FEATURE DETECTION AND FEATURE SELECTION PREPROCESSING
# =============================================================================


def detect_feature_columns(df: DataFrame) -> Tuple[List[str], List[str]]:
    print_header("FEATURE COLUMN DETECTION")
    drop_set = set(ALWAYS_DROP_COLUMNS)
    if EXCLUDE_PROXY_RISK_FEATURES:
        drop_set.update(PROXY_RISK_FEATURES)

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for field in df.schema.fields:
        name = field.name
        if name in drop_set:
            continue
        if isinstance(field.dataType, NumericType):
            numeric_cols.append(name)
        elif isinstance(field.dataType, (StringType, BooleanType)):
            categorical_cols.append(name)

    print(f"Numeric candidate features ({len(numeric_cols)}):")
    for col_name in numeric_cols:
        print(f"- {col_name}")

    print(f"\nCategorical candidate features ({len(categorical_cols)}):")
    for col_name in categorical_cols:
        print(f"- {col_name}")

    if EXCLUDE_PROXY_RISK_FEATURES:
        print("\nProxy risk features excluded:")
        for col_name in get_existing_columns(df, PROXY_RISK_FEATURES):
            print(f"- {col_name}")
    else:
        print("\nProxy risk features are kept for the current project setting.")

    if not numeric_cols and not categorical_cols:
        raise ValueError("No candidate feature columns detected.")

    return numeric_cols, categorical_cols


def build_feature_selection_working_df(
    df: DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> Tuple[DataFrame, List[str], List[str], Dict[str, str], Dict[str, str]]:
    """
    Build a clean temporary DataFrame with safe column names for Spark ML stages.
    Returns:
      working_df,
      numeric_work_cols,
      categorical_work_cols,
      raw_to_work,
      work_to_raw
    """
    print_header("BUILD FEATURE SELECTION WORKING DATAFRAME")
    used_names = set(["label"])
    raw_to_work: Dict[str, str] = {}
    work_to_raw: Dict[str, str] = {}
    select_exprs = [col("label").cast("double").alias("label")]

    for raw_col in numeric_cols:
        safe = sanitize_name(raw_col, used_names)
        work_col = f"{safe}__num"
        used_names.add(work_col)
        raw_to_work[raw_col] = work_col
        work_to_raw[work_col] = raw_col
        select_exprs.append(col(raw_col).cast("double").alias(work_col))

    for raw_col in categorical_cols:
        safe = sanitize_name(raw_col, used_names)
        work_col = f"{safe}__cat"
        used_names.add(work_col)
        raw_to_work[raw_col] = work_col
        work_to_raw[work_col] = raw_col
        select_exprs.append(col(raw_col).cast("string").alias(work_col))

    working_df = df.select(*select_exprs)

    numeric_work_cols = [raw_to_work[col_name] for col_name in numeric_cols]
    categorical_work_cols = [raw_to_work[col_name] for col_name in categorical_cols]

    print("Working numeric columns:")
    for col_name in numeric_work_cols:
        print(f"- {col_name} -> {work_to_raw[col_name]}")

    print("Working categorical columns:")
    for col_name in categorical_work_cols:
        print(f"- {col_name} -> {work_to_raw[col_name]}")

    return working_df, numeric_work_cols, categorical_work_cols, raw_to_work, work_to_raw


def get_feature_names_from_metadata(df: DataFrame, vector_col: str) -> List[str]:
    """Extract vector element names from Spark ML metadata."""
    field = df.schema[vector_col]
    metadata = field.metadata
    attrs = metadata.get("ml_attr", {}).get("attrs", {})
    named_attrs = []
    for attr_group in attrs.values():
        named_attrs.extend(attr_group)

    if named_attrs:
        named_attrs = sorted(named_attrs, key=lambda x: x.get("idx", 0))
        return [item.get("name", f"{vector_col}_{idx}") for idx, item in enumerate(named_attrs)]

    num_attrs = metadata.get("ml_attr", {}).get("num_attrs")
    if num_attrs is not None:
        return [f"{vector_col}_{i}" for i in range(int(num_attrs))]

    return []


def map_vector_name_to_raw_feature(
    vector_feature_name: str,
    work_to_raw: Dict[str, str],
) -> str:
    """
    Map a vector element name back to the original raw feature name.
    Handles names created by VectorAssembler, StringIndexer, OneHotEncoder,
    and QuantileDiscretizer.
    """
    sorted_work_cols = sorted(work_to_raw.keys(), key=len, reverse=True)
    for work_col in sorted_work_cols:
        base = work_col.replace("__num", "").replace("__cat", "")
        candidates = [
            work_col,
            f"{work_col}__bucket",
            f"{work_col}__idx",
            f"{work_col}__ohe",
            base,
        ]
        for prefix in candidates:
            if vector_feature_name == prefix or vector_feature_name.startswith(prefix + "_"):
                return work_to_raw[work_col]

    return vector_feature_name


def build_chi_square_pipeline(
    numeric_work_cols: List[str],
    categorical_work_cols: List[str],
) -> Tuple[Pipeline, List[str]]:
    stages = []
    chi_inputs: List[str] = []

    for col_name in numeric_work_cols:
        bucket_col = f"{col_name}__bucket"
        discretizer = QuantileDiscretizer(
            inputCol=col_name,
            outputCol=bucket_col,
            numBuckets=FS_CHI_NUM_BUCKETS,
            handleInvalid="keep",
            relativeError=0.01,
        )
        stages.append(discretizer)
        chi_inputs.append(bucket_col)

    for col_name in categorical_work_cols:
        idx_col = f"{col_name}__idx"
        ohe_col = f"{col_name}__ohe"
        indexer = StringIndexer(inputCol=col_name, outputCol=idx_col, handleInvalid="keep")
        encoder = OneHotEncoder(inputCols=[idx_col], outputCols=[ohe_col], handleInvalid="keep")
        stages.extend([indexer, encoder])
        chi_inputs.append(ohe_col)

    assembler = VectorAssembler(inputCols=chi_inputs, outputCol="chi_features", handleInvalid="keep")
    stages.append(assembler)
    return Pipeline(stages=stages), chi_inputs


def build_rf_feature_selection_pipeline(
    numeric_work_cols: List[str],
    categorical_work_cols: List[str],
) -> Pipeline:
    stages = []
    rf_inputs: List[str] = []

    for col_name in numeric_work_cols:
        rf_inputs.append(col_name)

    for col_name in categorical_work_cols:
        idx_col = f"{col_name}__idx_rf"
        ohe_col = f"{col_name}__ohe_rf"
        indexer = StringIndexer(inputCol=col_name, outputCol=idx_col, handleInvalid="keep")
        encoder = OneHotEncoder(inputCols=[idx_col], outputCols=[ohe_col], handleInvalid="keep")
        stages.extend([indexer, encoder])
        rf_inputs.append(ohe_col)

    assembler = VectorAssembler(inputCols=rf_inputs, outputCol="rf_features", handleInvalid="keep")
    rf = RandomForestClassifier(
        featuresCol="rf_features",
        labelCol="label",
        predictionCol="fs_prediction",
        probabilityCol="fs_probability",
        rawPredictionCol="fs_rawPrediction",
        numTrees=FS_RF_NUM_TREES,
        maxDepth=FS_RF_MAX_DEPTH,
        seed=SEED,
    )
    stages.extend([assembler, rf])
    return Pipeline(stages=stages)


def normalize_scores(score_dict: Dict[str, float]) -> Dict[str, float]:
    if not score_dict:
        return {}
    max_value = max([safe_float(value) for value in score_dict.values()])
    if max_value <= 0:
        return {key: 0.0 for key in score_dict.keys()}
    return {key: safe_float(value) / max_value for key, value in score_dict.items()}


def perform_feature_selection(
    train_df: DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> Tuple[List[str], List[Dict]]:
    """
    Fit Chi-Square and Random Forest feature selection on training data only.
    """
    print_header("SUPERVISED FEATURE SELECTION ON TRAINING DATA ONLY")

    working_df, numeric_work_cols, categorical_work_cols, _, work_to_raw = build_feature_selection_working_df(
        train_df,
        numeric_cols,
        categorical_cols,
    )
    working_df = working_df.persist(StorageLevel.MEMORY_AND_DISK)

    # -------------------------------------------------------------------------
    # Method 1: Chi-Square Test
    # -------------------------------------------------------------------------
    print_subheader("Method 1: Chi-Square Test")
    chi_pipeline, _ = build_chi_square_pipeline(numeric_work_cols, categorical_work_cols)
    chi_model = chi_pipeline.fit(working_df)
    chi_df = chi_model.transform(working_df).select("label", "chi_features")
    chi_result = ChiSquareTest.test(chi_df, "chi_features", "label").head()

    chi_feature_names = get_feature_names_from_metadata(chi_df, "chi_features")
    chi_statistics = list(chi_result["statistics"])
    chi_p_values = list(chi_result["pValues"])

    chi_scores_by_raw: Dict[str, float] = {}
    chi_p_min_by_raw: Dict[str, float] = {}

    for index, stat_value in enumerate(chi_statistics):
        feature_name = chi_feature_names[index] if index < len(chi_feature_names) else f"chi_feature_{index}"
        raw_feature = map_vector_name_to_raw_feature(feature_name, work_to_raw)
        stat = safe_float(stat_value)
        p_value = safe_float(chi_p_values[index], default=1.0) if index < len(chi_p_values) else 1.0
        chi_scores_by_raw[raw_feature] = chi_scores_by_raw.get(raw_feature, 0.0) + stat
        chi_p_min_by_raw[raw_feature] = min(chi_p_min_by_raw.get(raw_feature, 1.0), p_value)

    print("Chi-Square completed.")

    # -------------------------------------------------------------------------
    # Method 2: Random Forest Feature Importance
    # -------------------------------------------------------------------------
    print_subheader("Method 2: Random Forest Feature Importance")
    rf_pipeline = build_rf_feature_selection_pipeline(numeric_work_cols, categorical_work_cols)
    rf_model = rf_pipeline.fit(working_df)
    rf_transformed = rf_model.transform(working_df).select("rf_features")
    rf_feature_names = get_feature_names_from_metadata(rf_transformed, "rf_features")
    rf_stage = rf_model.stages[-1]
    rf_importances = list(rf_stage.featureImportances)

    rf_scores_by_raw: Dict[str, float] = {}
    for index, importance_value in enumerate(rf_importances):
        feature_name = rf_feature_names[index] if index < len(rf_feature_names) else f"rf_feature_{index}"
        raw_feature = map_vector_name_to_raw_feature(feature_name, work_to_raw)
        importance = safe_float(importance_value)
        rf_scores_by_raw[raw_feature] = rf_scores_by_raw.get(raw_feature, 0.0) + importance

    print("Random Forest importance completed.")

    all_features = sorted(set(numeric_cols + categorical_cols))
    chi_norm = normalize_scores(chi_scores_by_raw)
    rf_norm = normalize_scores(rf_scores_by_raw)

    score_rows: List[Dict] = []
    for feature in all_features:
        chi_score = safe_float(chi_scores_by_raw.get(feature, 0.0))
        rf_score = safe_float(rf_scores_by_raw.get(feature, 0.0))
        chi_normalized = safe_float(chi_norm.get(feature, 0.0))
        rf_normalized = safe_float(rf_norm.get(feature, 0.0))
        combined_score = 0.5 * chi_normalized + 0.5 * rf_normalized

        score_rows.append({
            "feature": feature,
            "feature_type": "numeric" if feature in numeric_cols else "categorical",
            "chi_square_statistic": round_float(chi_score, 8),
            "chi_square_min_p_value": round_float(chi_p_min_by_raw.get(feature, 1.0), 10),
            "chi_square_score_normalized": round_float(chi_normalized, 8),
            "random_forest_importance": round_float(rf_score, 8),
            "random_forest_score_normalized": round_float(rf_normalized, 8),
            "combined_score": round_float(combined_score, 8),
        })

    score_rows = sorted(score_rows, key=lambda row: row["combined_score"], reverse=True)

    selected_features = [row["feature"] for row in score_rows[: min(TOP_N_FEATURES, len(score_rows))]]
    removed_features = [feature for feature in all_features if feature not in selected_features]

    print_subheader("Selected raw features")
    for i, feature in enumerate(selected_features, start=1):
        row = next(item for item in score_rows if item["feature"] == feature)
        print(f"{i:02d}. {feature:<35} combined_score={row['combined_score']:.6f}")

    print(f"\nSelected features: {len(selected_features)}")
    print(f"Removed features : {len(removed_features)}")

    write_dicts_to_csv(score_rows, FEATURE_SELECTION_SCORES_CSV)
    selected_rows = []
    for i, feature in enumerate(selected_features, start=1):
        source_row = next(row for row in score_rows if row["feature"] == feature)
        selected_rows.append({"rank": i, **source_row})
    write_dicts_to_csv(selected_rows, SELECTED_FEATURES_CSV)

    plot_feature_selection_scores(score_rows, top_n=15)
    plot_selected_vs_removed(selected_features, removed_features)

    working_df.unpersist()
    return selected_features, score_rows


# =============================================================================
# MODEL PREPROCESSING, TRAINING, AND EVALUATION
# =============================================================================


def split_selected_feature_types(
    selected_features: List[str],
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> Tuple[List[str], List[str]]:
    selected_set = set(selected_features)
    selected_numeric = [col_name for col_name in numeric_cols if col_name in selected_set]
    selected_categorical = [col_name for col_name in categorical_cols if col_name in selected_set]
    return selected_numeric, selected_categorical


def build_model_preprocessing_stages(
    numeric_cols: List[str],
    categorical_cols: List[str],
    use_scaler: bool,
) -> Tuple[List, str]:
    stages = []
    feature_inputs: List[str] = []

    for categorical_col in categorical_cols:
        idx_col = f"{categorical_col}_idx"
        ohe_col = f"{categorical_col}_ohe"
        stages.append(StringIndexer(inputCol=categorical_col, outputCol=idx_col, handleInvalid="keep"))
        stages.append(OneHotEncoder(inputCols=[idx_col], outputCols=[ohe_col], handleInvalid="keep"))
        feature_inputs.append(ohe_col)

    feature_inputs.extend(numeric_cols)

    assembler = VectorAssembler(inputCols=feature_inputs, outputCol="raw_features", handleInvalid="keep")
    stages.append(assembler)

    if use_scaler:
        scaler = StandardScaler(inputCol="raw_features", outputCol="scaled_features", withMean=False, withStd=True)
        stages.append(scaler)
        return stages, "scaled_features"

    return stages, "raw_features"


def get_model_feature_names(df: DataFrame, features_col: str) -> List[str]:
    names = get_feature_names_from_metadata(df, features_col)
    if names:
        return names
    try:
        size = df.select(features_col).head()[0].size
        return [f"feature_{i}" for i in range(size)]
    except Exception:
        return []


def evaluate_confusion(predictions: DataFrame, prediction_col: str = "prediction") -> Dict[str, int]:
    rows = predictions.groupBy("label", prediction_col).count().collect()
    values = {(int(row["label"]), int(row[prediction_col])): int(row["count"]) for row in rows}
    tn = values.get((0, 0), 0)
    fp = values.get((0, 1), 0)
    fn = values.get((1, 0), 0)
    tp = values.get((1, 1), 0)
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def metrics_from_confusion(confusion: Dict[str, int]) -> Dict[str, float]:
    tn = confusion.get("tn", 0)
    fp = confusion.get("fp", 0)
    fn = confusion.get("fn", 0)
    tp = confusion.get("tp", 0)

    total = tn + fp + fn + tp
    accuracy = safe_divide(tp + tn, total)
    bad_precision = safe_divide(tp, tp + fp)
    bad_recall = safe_divide(tp, tp + fn)
    bad_f1 = safe_divide(2 * bad_precision * bad_recall, bad_precision + bad_recall)
    good_recall = safe_divide(tn, tn + fp)
    balanced_accuracy = (good_recall + bad_recall) / 2.0

    return {
        "accuracy_manual": accuracy,
        "bad_precision": bad_precision,
        "bad_recall": bad_recall,
        "bad_f1": bad_f1,
        "good_recall": good_recall,
        "balanced_accuracy": balanced_accuracy,
    }


def evaluate_default_metrics(predictions: DataFrame) -> Dict[str, float]:
    binary_roc = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )
    binary_pr = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR",
    )

    accuracy_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    f1_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    precision_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedPrecision"
    )
    recall_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")

    confusion = evaluate_confusion(predictions, prediction_col="prediction")
    confusion_metrics = metrics_from_confusion(confusion)

    return {
        "accuracy": round_float(accuracy_eval.evaluate(predictions)),
        "f1": round_float(f1_eval.evaluate(predictions)),
        "weighted_precision": round_float(precision_eval.evaluate(predictions)),
        "weighted_recall": round_float(recall_eval.evaluate(predictions)),
        "roc_auc": round_float(binary_roc.evaluate(predictions)),
        "pr_auc": round_float(binary_pr.evaluate(predictions)),
        "bad_precision_default": round_float(confusion_metrics["bad_precision"]),
        "bad_recall_default": round_float(confusion_metrics["bad_recall"]),
        "bad_f1_default": round_float(confusion_metrics["bad_f1"]),
        "balanced_accuracy": round_float(confusion_metrics["balanced_accuracy"]),
        **confusion,
    }


def add_bad_probability(predictions: DataFrame) -> DataFrame:
    return predictions.withColumn("probability_array", vector_to_array(col("probability"))).withColumn(
        "prob_bad", col("probability_array")[1]
    )


def evaluate_thresholds(predictions: DataFrame, model_name: str) -> List[Dict]:
    print_subheader(f"Threshold analysis: {model_name}")
    pred_prob = add_bad_probability(predictions)
    threshold_rows: List[Dict] = []

    for threshold in DECISION_THRESHOLDS:
        pred_t = pred_prob.withColumn(
            "prediction_threshold",
            when(col("prob_bad") >= lit(float(threshold)), lit(1.0)).otherwise(lit(0.0)),
        )
        confusion = evaluate_confusion(pred_t, prediction_col="prediction_threshold")
        metric = metrics_from_confusion(confusion)

        threshold_rows.append({
            "model": model_name,
            "threshold": threshold,
            "tn": confusion["tn"],
            "fp": confusion["fp"],
            "fn": confusion["fn"],
            "tp": confusion["tp"],
            "accuracy": round_float(metric["accuracy_manual"]),
            "balanced_accuracy": round_float(metric["balanced_accuracy"]),
            "bad_precision": round_float(metric["bad_precision"]),
            "bad_recall": round_float(metric["bad_recall"]),
            "bad_f1": round_float(metric["bad_f1"]),
            "good_recall": round_float(metric["good_recall"]),
        })

    print(f"{'threshold':<12}{'bad_precision':<16}{'bad_recall':<14}{'bad_f1':<12}{'balanced_acc':<14}")
    for row in threshold_rows:
        print(
            f"{row['threshold']:<12.2f}"
            f"{row['bad_precision']:<16.4f}"
            f"{row['bad_recall']:<14.4f}"
            f"{row['bad_f1']:<12.4f}"
            f"{row['balanced_accuracy']:<14.4f}"
        )

    return threshold_rows


def get_best_threshold_row(threshold_rows: List[Dict]) -> Dict:
    if not threshold_rows:
        return {}
    return sorted(threshold_rows, key=lambda row: row["bad_f1"], reverse=True)[0]


def train_single_model(
    model_name: str,
    train_df: DataFrame,
    test_df: DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> Tuple[PipelineModel, DataFrame, Dict, List[Dict]]:
    print_header(f"TRAINING MODEL - {model_name}")

    if model_name == "Logistic Regression":
        preprocessing_stages, final_features_col = build_model_preprocessing_stages(
            numeric_cols, categorical_cols, use_scaler=True
        )
        classifier = LogisticRegression(
            featuresCol=final_features_col,
            labelCol="label",
            weightCol="class_weight",
            predictionCol="prediction",
            probabilityCol="probability",
            rawPredictionCol="rawPrediction",
            maxIter=LR_MAX_ITER,
            regParam=0.05,
            elasticNetParam=0.0,
        )
    elif model_name == "Random Forest":
        preprocessing_stages, final_features_col = build_model_preprocessing_stages(
            numeric_cols, categorical_cols, use_scaler=False
        )
        classifier = RandomForestClassifier(
            featuresCol=final_features_col,
            labelCol="label",
            weightCol="class_weight",
            predictionCol="prediction",
            probabilityCol="probability",
            rawPredictionCol="rawPrediction",
            numTrees=RF_NUM_TREES,
            maxDepth=RF_MAX_DEPTH,
            seed=SEED,
        )
    elif model_name == "Gradient Boosted Trees":
        preprocessing_stages, final_features_col = build_model_preprocessing_stages(
            numeric_cols, categorical_cols, use_scaler=False
        )
        classifier = GBTClassifier(
            featuresCol=final_features_col,
            labelCol="label",
            weightCol="class_weight",
            predictionCol="prediction",
            probabilityCol="probability",
            rawPredictionCol="rawPrediction",
            maxIter=GBT_MAX_ITER,
            maxDepth=GBT_MAX_DEPTH,
            stepSize=0.05,
            seed=SEED,
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    pipeline = Pipeline(stages=preprocessing_stages + [classifier])
    fitted_model = pipeline.fit(train_df)
    predictions = fitted_model.transform(test_df).persist(StorageLevel.MEMORY_AND_DISK)

    default_metrics = evaluate_default_metrics(predictions)
    threshold_rows = evaluate_thresholds(predictions, model_name)
    best_threshold = get_best_threshold_row(threshold_rows)

    result = {
        "model": model_name,
        **default_metrics,
        "best_threshold_by_bad_f1": best_threshold.get("threshold", 0.5),
        "bad_precision_best_threshold": round_float(best_threshold.get("bad_precision", 0.0)),
        "bad_recall_best_threshold": round_float(best_threshold.get("bad_recall", 0.0)),
        "bad_f1_best_threshold": round_float(best_threshold.get("bad_f1", 0.0)),
        "balanced_accuracy_best_threshold": round_float(best_threshold.get("balanced_accuracy", 0.0)),
        "selected_numeric_features": ", ".join(numeric_cols),
        "selected_categorical_features": ", ".join(categorical_cols),
    }

    print_subheader(f"Evaluation summary: {model_name}")
    for key in [
        "accuracy", "f1", "roc_auc", "pr_auc", "balanced_accuracy",
        "bad_precision_default", "bad_recall_default", "bad_f1_default",
        "best_threshold_by_bad_f1", "bad_precision_best_threshold",
        "bad_recall_best_threshold", "bad_f1_best_threshold",
    ]:
        print(f"{key:<35}: {result[key]}")

    return fitted_model, predictions, result, threshold_rows


def train_and_evaluate_models(
    train_df: DataFrame,
    test_df: DataFrame,
    selected_numeric: List[str],
    selected_categorical: List[str],
) -> Tuple[List[Dict], List[Dict], Dict[str, PipelineModel], Dict[str, DataFrame]]:
    print_header("MODEL TRAINING AND EVALUATION")

    model_names = ["Logistic Regression", "Random Forest", "Gradient Boosted Trees"]
    model_rows: List[Dict] = []
    threshold_rows_all: List[Dict] = []
    model_objects: Dict[str, PipelineModel] = {}
    prediction_objects: Dict[str, DataFrame] = {}

    for model_name in model_names:
        fitted_model, predictions, result, threshold_rows = train_single_model(
            model_name,
            train_df,
            test_df,
            selected_numeric,
            selected_categorical,
        )
        model_rows.append(result)
        threshold_rows_all.extend(threshold_rows)
        model_objects[model_name] = fitted_model
        prediction_objects[model_name] = predictions

    model_rows = sorted(model_rows, key=lambda row: safe_float(row.get(MODEL_SELECTION_METRIC, 0.0)), reverse=True)

    print_header("MODEL COMPARISON")
    print(
        f"{'Model':<28}{'Accuracy':<10}{'F1':<10}{'ROC-AUC':<10}"
        f"{'PR-AUC':<10}{'BadRecall*':<12}{'BadF1*':<10}"
    )
    print("-" * 90)
    for row in model_rows:
        print(
            f"{row['model']:<28}"
            f"{row['accuracy']:<10.4f}"
            f"{row['f1']:<10.4f}"
            f"{row['roc_auc']:<10.4f}"
            f"{row['pr_auc']:<10.4f}"
            f"{row['bad_recall_best_threshold']:<12.4f}"
            f"{row['bad_f1_best_threshold']:<10.4f}"
        )
    print("* BadRecall and BadF1 are calculated at each model's best threshold by Bad Loan F1.")

    write_dicts_to_csv(model_rows, MODEL_METRICS_CSV)
    write_dicts_to_csv(threshold_rows_all, THRESHOLD_ANALYSIS_CSV)
    plot_model_metric_comparison(model_rows)

    return model_rows, threshold_rows_all, model_objects, prediction_objects


# =============================================================================
# FEATURE IMPORTANCE AND REPORTING
# =============================================================================


def extract_final_feature_importance(
    model_name: str,
    fitted_model: PipelineModel,
    predictions: DataFrame,
) -> List[Dict]:
    classifier = fitted_model.stages[-1]

    # The classifier's featuresCol can be raw_features or scaled_features.
    features_col = classifier.getFeaturesCol()
    feature_names = get_model_feature_names(predictions, features_col)

    rows: List[Dict] = []

    if hasattr(classifier, "coefficients"):
        coefficients = list(classifier.coefficients)
        for idx, coef in enumerate(coefficients):
            feature_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            rows.append({
                "model": model_name,
                "feature": feature_name,
                "importance": round_float(abs(safe_float(coef)), 8),
                "signed_coefficient": round_float(safe_float(coef), 8),
                "interpretation": "increases Bad Loan risk" if safe_float(coef) > 0 else "decreases Bad Loan risk",
            })
    elif hasattr(classifier, "featureImportances"):
        importances = list(classifier.featureImportances)
        for idx, importance in enumerate(importances):
            feature_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            rows.append({
                "model": model_name,
                "feature": feature_name,
                "importance": round_float(safe_float(importance), 8),
                "signed_coefficient": "",
                "interpretation": "higher value means stronger contribution in tree splits",
            })

    rows = sorted(rows, key=lambda row: safe_float(row["importance"]), reverse=True)
    return rows


def save_confusion_matrices(
    model_rows: List[Dict],
    threshold_rows: List[Dict],
) -> List[Dict]:
    confusion_rows: List[Dict] = []

    for model_row in model_rows:
        model_name = model_row["model"]
        default_row = {
            "model": model_name,
            "threshold_type": "default_0.50",
            "threshold": 0.50,
            "tn": model_row.get("tn", 0),
            "fp": model_row.get("fp", 0),
            "fn": model_row.get("fn", 0),
            "tp": model_row.get("tp", 0),
        }
        confusion_rows.append(default_row)

        best_threshold = model_row.get("best_threshold_by_bad_f1", 0.5)
        match = [
            row for row in threshold_rows
            if row["model"] == model_name and abs(float(row["threshold"]) - float(best_threshold)) < 1e-9
        ]
        if match:
            best_row = match[0]
            confusion_rows.append({
                "model": model_name,
                "threshold_type": "best_bad_f1",
                "threshold": best_threshold,
                "tn": best_row.get("tn", 0),
                "fp": best_row.get("fp", 0),
                "fn": best_row.get("fn", 0),
                "tp": best_row.get("tp", 0),
            })

    write_dicts_to_csv(confusion_rows, CONFUSION_MATRIX_CSV)
    return confusion_rows


def save_selected_dataset(df: DataFrame, selected_features: List[str]) -> None:
    if not SAVE_SELECTED_DATASET_TO_HDFS:
        return

    print_header("SAVE SELECTED RAW DATASET TO HDFS")
    output_cols = selected_features + ["label"]
    optional_cols = get_existing_columns(df, ["class_weight"])
    output_cols.extend(optional_cols)
    selected_df = df.select(*output_cols)
    selected_df.write.mode("overwrite").parquet(HDFS_SELECTED_DATA_OUTPUT_PATH)
    print(f"Selected dataset saved to: {HDFS_SELECTED_DATA_OUTPUT_PATH}")
    print(f"Selected dataset columns: {len(output_cols)}")


def save_best_model(
    model_rows: List[Dict],
    model_objects: Dict[str, PipelineModel],
) -> str:
    best_model_name = model_rows[0]["model"]
    best_model = model_objects[best_model_name]

    if SAVE_BEST_MODEL_TO_HDFS:
        print_header("SAVE BEST MODEL TO HDFS")
        best_model.write().overwrite().save(HDFS_BEST_MODEL_OUTPUT_PATH)
        print(f"Best model selected by {MODEL_SELECTION_METRIC}: {best_model_name}")
        print(f"Best model saved to: {HDFS_BEST_MODEL_OUTPUT_PATH}")

    return best_model_name


def print_final_conclusion(best_model_row: Dict) -> None:
    print_header("FINAL CONCLUSION")
    print(f"Best model by {MODEL_SELECTION_METRIC}: {best_model_row['model']}")
    print(f"PR-AUC: {best_model_row['pr_auc']:.4f}")
    print(f"ROC-AUC: {best_model_row['roc_auc']:.4f}")
    print(f"Best Bad Loan threshold: {best_model_row['best_threshold_by_bad_f1']:.2f}")
    print(f"Bad Loan Recall at best threshold: {best_model_row['bad_recall_best_threshold']:.4f}")
    print(f"Bad Loan F1 at best threshold: {best_model_row['bad_f1_best_threshold']:.4f}")
    print("\nInterpretation guide:")
    print("- Feature Selection was fitted only on the training set to avoid test leakage.")
    print("- Chi-Square captures statistical dependence between features and the label.")
    print("- Random Forest importance captures nonlinear and interaction-based predictive signal.")
    print("- PR-AUC and Bad Loan Recall are important because Bad Loan is the minority/risk class.")
    print("- Threshold analysis supports business decision making instead of blindly using 0.50.")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    ensure_output_dirs()
    spark = create_spark_session()

    df = load_dataset(spark)
    validate_dataset(df)
    label_rows = print_label_distribution(df)

    # Detect all candidate features from the full dataset schema.
    numeric_cols, categorical_cols = detect_feature_columns(df)

    # Correct Data Science practice: split first, then fit Feature Selection on train only.
    train_df, test_df = stratified_train_test_split(df)

    # Feature Selection uses raw train_df before class_weight is added.
    selected_features, feature_score_rows = perform_feature_selection(
        train_df=train_df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )

    selected_numeric, selected_categorical = split_selected_feature_types(
        selected_features,
        numeric_cols,
        categorical_cols,
    )

    print_header("FINAL SELECTED FEATURE SET FOR MODELING")
    print("Selected numeric features:")
    for col_name in selected_numeric:
        print(f"- {col_name}")
    print("Selected categorical features:")
    for col_name in selected_categorical:
        print(f"- {col_name}")

    # Add class weights after feature selection; final models use class_weight.
    train_weighted, test_weighted, class_weight_info = add_class_weight(train_df, test_df)

    # Optional: save selected raw dataset for reproducibility.
    save_selected_dataset(df, selected_features)

    model_rows, threshold_rows, model_objects, prediction_objects = train_and_evaluate_models(
        train_df=train_weighted,
        test_df=test_weighted,
        selected_numeric=selected_numeric,
        selected_categorical=selected_categorical,
    )

    best_model_name = save_best_model(model_rows, model_objects)
    best_model_row = model_rows[0]
    best_threshold = safe_float(best_model_row.get("best_threshold_by_bad_f1", 0.50), 0.50)

    confusion_rows = save_confusion_matrices(model_rows, threshold_rows)

    # Feature importance for every final model.
    all_importance_rows: List[Dict] = []
    for model_name, fitted_model in model_objects.items():
        importance_rows = extract_final_feature_importance(
            model_name=model_name,
            fitted_model=fitted_model,
            predictions=prediction_objects[model_name],
        )
        all_importance_rows.extend(importance_rows)
    write_dicts_to_csv(all_importance_rows, FINAL_FEATURE_IMPORTANCE_CSV)

    # Report-ready figures for the best model.
    plot_threshold_tradeoff(threshold_rows, best_model_name)

    best_confusion = [
        row for row in confusion_rows
        if row["model"] == best_model_name and row["threshold_type"] == "best_bad_f1"
    ]
    if best_confusion:
        plot_confusion_matrix(best_confusion[0], best_model_name, best_threshold)

    plot_feature_importance(all_importance_rows, best_model_name, top_n=15)

    write_json(
        {
            "input_path": HDFS_INPUT_PATH,
            "selected_dataset_output_path": HDFS_SELECTED_DATA_OUTPUT_PATH,
            "best_model_output_path": HDFS_BEST_MODEL_OUTPUT_PATH,
            "top_n_features": TOP_N_FEATURES,
            "exclude_proxy_risk_features": EXCLUDE_PROXY_RISK_FEATURES,
            "model_selection_metric": MODEL_SELECTION_METRIC,
            "selected_features": selected_features,
            "selected_numeric_features": selected_numeric,
            "selected_categorical_features": selected_categorical,
            "label_distribution": label_rows,
            "class_weight_info": class_weight_info,
            "best_model": best_model_name,
            "best_model_metrics": best_model_row,
        },
        METADATA_JSON,
    )

    print_final_conclusion(best_model_row)

    # Release persisted dataframes.
    for pred_df in prediction_objects.values():
        pred_df.unpersist()
    train_weighted.unpersist()
    test_weighted.unpersist()
    train_df.unpersist()
    test_df.unpersist()
    df.unpersist()

    spark.stop()


if __name__ == "__main__":
    main()