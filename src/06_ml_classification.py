# -*- coding: utf-8 -*-
"""
FILE 06 - Classification Feature Selection + MLlib Modeling + Stream Data Split

Flow:
1. Read preprocessed classification data from HDFS.
2. Split data 7:2:1 into train/test/stream simulation.
3. Run classification feature selection on train data only.
   - Chi-square score after discretizing numeric features.
   - Random Forest feature importance.
   - Combined score = 0.5 * normalized chi-square + 0.5 * normalized RF importance.
4. Train Logistic Regression, Random Forest, and GBTClassifier.
5. Evaluate models and save report artifacts.
6. Save best classification PipelineModel to HDFS.
7. Save 10 percent stream simulation data for Kafka producer.
"""

import csv
import math
import os

from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, QuantileDiscretizer, StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.stat import ChiSquareTest
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, lit, rand, when
from pyspark.sql.types import BooleanType, NumericType, StringType

SEED = 42
TOP_N_FEATURES = 12

INPUT_PATH = (
    "hdfs://localhost:9000/bigdata/prosper_loan/processed/"
    "prosper_loan_preprocessed_classification"
)
MODEL_OUTPUT_PATH = (
    "hdfs://localhost:9000/bigdata/prosper_loan/models/classification/"
    "best_classification_pipeline_model"
)
STREAM_DATA_HDFS_PATH = (
    "hdfs://localhost:9000/bigdata/prosper_loan/streaming/"
    "classification_simulation_data"
)

OUTPUT_DIR = os.path.join("outputs", "06_ml_classification")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
STREAM_JSON_DIR = os.path.join(OUTPUT_DIR, "streaming_json_sample")

FEATURE_SCORE_CSV = os.path.join(TABLE_DIR, "classification_feature_score.csv")
MODEL_METRICS_CSV = os.path.join(TABLE_DIR, "model_metrics.csv")
CONFUSION_MATRIX_CSV = os.path.join(TABLE_DIR, "confusion_matrix.csv")
BEST_MODEL_INFO_TXT = os.path.join(TABLE_DIR, "best_model_info.txt")

LABEL_DISTRIBUTION_FIG = os.path.join(FIGURE_DIR, "01_label_distribution.png")
FEATURE_SCORE_FIG = os.path.join(FIGURE_DIR, "02_classification_feature_selection.png")
MODEL_COMPARISON_FIG = os.path.join(FIGURE_DIR, "03_model_comparison.png")
CONFUSION_MATRIX_FIG = os.path.join(FIGURE_DIR, "04_confusion_matrix_best_model.png")

DROP_COLUMNS = {
    "label", "LoanStatus", "features", "raw_features", "scaled_features",
    "prediction", "rawPrediction", "probability", "class_weight"
}


def print_header(title):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def ensure_dirs():
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(STREAM_JSON_DIR, exist_ok=True)


def write_dicts_to_csv(rows, path):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved table: {path}")


def safe_float(value):
    if value is None:
        return 0.0
    try:
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return value
    except Exception:
        return 0.0


def normalize(score_dict):
    if not score_dict:
        return {}
    max_value = max(score_dict.values())
    if max_value <= 0:
        return {k: 0.0 for k in score_dict}
    return {k: safe_float(v) / max_value for k, v in score_dict.items()}


def detect_feature_columns(df):
    numeric_cols = []
    categorical_cols = []
    for field in df.schema.fields:
        name = field.name
        if name in DROP_COLUMNS:
            continue
        if isinstance(field.dataType, NumericType):
            numeric_cols.append(name)
        elif isinstance(field.dataType, (StringType, BooleanType)):
            categorical_cols.append(name)
    return numeric_cols, categorical_cols


def stratified_split_7_2_1(df):
    print_header("SPLIT DATA 7:2:1")
    train_parts = []
    test_parts = []
    stream_parts = []
    for label_value in [0, 1]:
        label_df = df.filter(col("label") == label_value)
        train_part, test_part, stream_part = label_df.randomSplit([0.7, 0.2, 0.1], seed=SEED)
        train_parts.append(train_part)
        test_parts.append(test_part)
        stream_parts.append(stream_part)
    train_df = train_parts[0].unionByName(train_parts[1]).orderBy(rand(seed=SEED)).persist(StorageLevel.MEMORY_AND_DISK)
    test_df = test_parts[0].unionByName(test_parts[1]).orderBy(rand(seed=SEED + 1)).persist(StorageLevel.MEMORY_AND_DISK)
    stream_df = stream_parts[0].unionByName(stream_parts[1]).orderBy(rand(seed=SEED + 2)).persist(StorageLevel.MEMORY_AND_DISK)
    print(f"Train rows : {train_df.count()}")
    print(f"Test rows  : {test_df.count()}")
    print(f"Stream rows: {stream_df.count()}")
    print("Train label distribution:")
    train_df.groupBy("label").count().orderBy("label").show()
    print("Test label distribution:")
    test_df.groupBy("label").count().orderBy("label").show()
    print("Stream label distribution:")
    stream_df.groupBy("label").count().orderBy("label").show()
    return train_df, test_df, stream_df


def add_class_weight(train_df, test_df):
    print_header("ADD CLASS WEIGHT")
    counts = {int(r["label"]): int(r["count"]) for r in train_df.groupBy("label").count().collect()}
    total = counts.get(0, 0) + counts.get(1, 0)
    weight_good = total / (2.0 * counts.get(0, 1))
    weight_bad = total / (2.0 * counts.get(1, 1))
    print(f"Good Loan weight: {weight_good:.4f}")
    print(f"Bad Loan weight : {weight_bad:.4f}")
    def apply_weight(df):
        return df.withColumn(
            "class_weight",
            when(col("label") == 1, lit(float(weight_bad))).otherwise(lit(float(weight_good)))
        )
    return apply_weight(train_df), apply_weight(test_df)


def plot_label_distribution(df):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows = df.groupBy("label").count().orderBy("label").collect()
    labels = ["Good Loan" if int(r["label"]) == 0 else "Bad Loan" for r in rows]
    values = [int(r["count"]) for r in rows]
    plt.figure(figsize=(7, 4.5))
    plt.bar(labels, values)
    plt.title("Label Distribution")
    plt.xlabel("Loan class")
    plt.ylabel("Number of loans")
    for i, value in enumerate(values):
        plt.text(i, value, f"{value:,}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(LABEL_DISTRIBUTION_FIG, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {LABEL_DISTRIBUTION_FIG}")


def plot_feature_scores(rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    top_rows = rows[:20][::-1]
    plt.figure(figsize=(10, 7))
    plt.barh([r["feature"] for r in top_rows], [r["combined_score"] for r in top_rows])
    plt.title("Hybrid Feature Selection Diagnostic")
    plt.xlabel("Combined feature score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FEATURE_SCORE_FIG, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {FEATURE_SCORE_FIG}")


def plot_model_comparison(rows):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return
    models = [r["model"] for r in rows]
    metrics = ["accuracy", "f1", "roc_auc", "pr_auc"]
    x = np.arange(len(models))
    width = 0.18
    plt.figure(figsize=(11, 6))
    for i, metric in enumerate(metrics):
        values = [r[metric] for r in rows]
        plt.bar(x + (i - 1.5) * width, values, width, label=metric)
    plt.xticks(x, models, rotation=15, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Metric value")
    plt.title("Classification Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(MODEL_COMPARISON_FIG, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {MODEL_COMPARISON_FIG}")


def plot_confusion_matrix(confusion, model_name):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    matrix = [[confusion["tn"], confusion["fp"]], [confusion["fn"], confusion["tp"]]]
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xticks([0, 1], ["Pred Good", "Pred Bad"])
    plt.yticks([0, 1], ["Actual Good", "Actual Bad"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{matrix[i][j]:,}", ha="center", va="center", fontsize=12)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_FIG, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {CONFUSION_MATRIX_FIG}")


def classification_feature_selection(train_df, numeric_cols, categorical_cols):
    print_header("CLASSIFICATION FEATURE SELECTION")
    stages = []
    fs_cols = []
    for c in numeric_cols:
        out_col = f"{c}__bucket"
        stages.append(QuantileDiscretizer(inputCol=c, outputCol=out_col, numBuckets=5, handleInvalid="keep"))
        fs_cols.append(out_col)
    for c in categorical_cols:
        out_col = f"{c}__idx"
        stages.append(StringIndexer(inputCol=c, outputCol=out_col, handleInvalid="keep"))
        fs_cols.append(out_col)
    stages.append(VectorAssembler(inputCols=fs_cols, outputCol="fs_features", handleInvalid="keep"))
    fs_model = Pipeline(stages=stages).fit(train_df)
    fs_df = fs_model.transform(train_df).select("label", "fs_features").persist(StorageLevel.MEMORY_AND_DISK)

    chi_result = ChiSquareTest.test(fs_df, "fs_features", "label").head()
    chi_stats = [safe_float(x) for x in list(chi_result["statistics"])]
    raw_features = numeric_cols + categorical_cols
    chi_scores = {raw_features[i]: chi_stats[i] if i < len(chi_stats) else 0.0 for i in range(len(raw_features))}

    rf = RandomForestClassifier(
        featuresCol="fs_features", labelCol="label", numTrees=80, maxDepth=6, seed=SEED,
        predictionCol="fs_prediction", probabilityCol="fs_probability", rawPredictionCol="fs_rawPrediction"
    )
    rf_model = rf.fit(fs_df)
    rf_importances = [safe_float(x) for x in rf_model.featureImportances.toArray()]
    rf_scores = {raw_features[i]: rf_importances[i] if i < len(rf_importances) else 0.0 for i in range(len(raw_features))}

    chi_norm = normalize(chi_scores)
    rf_norm = normalize(rf_scores)
    rows = []
    for feature in raw_features:
        combined = 0.5 * chi_norm.get(feature, 0.0) + 0.5 * rf_norm.get(feature, 0.0)
        rows.append({
            "feature": feature,
            "feature_type": "numeric" if feature in numeric_cols else "categorical",
            "chi_square_score": round(chi_scores.get(feature, 0.0), 8),
            "chi_square_score_normalized": round(chi_norm.get(feature, 0.0), 8),
            "random_forest_importance": round(rf_scores.get(feature, 0.0), 8),
            "random_forest_importance_normalized": round(rf_norm.get(feature, 0.0), 8),
            "combined_score": round(combined, 8),
        })
    rows = sorted(rows, key=lambda r: r["combined_score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["selected"] = "YES" if rank <= TOP_N_FEATURES else "NO"
    selected = [r["feature"] for r in rows[:TOP_N_FEATURES]]
    write_dicts_to_csv(rows, FEATURE_SCORE_CSV)
    plot_feature_scores(rows)
    print("Selected features:")
    for i, feature in enumerate(selected, start=1):
        print(f"{i:02d}. {feature}")
    return selected


def split_feature_types(selected_features, numeric_cols, categorical_cols):
    selected = set(selected_features)
    return [c for c in numeric_cols if c in selected], [c for c in categorical_cols if c in selected]


def build_model_stages(numeric_cols, categorical_cols, use_scaler):
    stages = []
    feature_inputs = []
    for c in categorical_cols:
        idx_col = f"{c}__idx_model"
        ohe_col = f"{c}__ohe_model"
        stages.append(StringIndexer(inputCol=c, outputCol=idx_col, handleInvalid="keep"))
        stages.append(OneHotEncoder(inputCols=[idx_col], outputCols=[ohe_col], handleInvalid="keep"))
        feature_inputs.append(ohe_col)
    feature_inputs.extend(numeric_cols)
    if use_scaler:
        stages.append(VectorAssembler(inputCols=feature_inputs, outputCol="raw_features", handleInvalid="keep"))
        stages.append(StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=False))
    else:
        stages.append(VectorAssembler(inputCols=feature_inputs, outputCol="features", handleInvalid="keep"))
    return stages


def confusion_values(predictions):
    rows = predictions.groupBy("label", "prediction").count().collect()
    values = {(int(r["label"]), int(r["prediction"])): int(r["count"]) for r in rows}
    return {
        "tn": values.get((0, 0), 0),
        "fp": values.get((0, 1), 0),
        "fn": values.get((1, 0), 0),
        "tp": values.get((1, 1), 0),
    }


def evaluate_model(predictions, model_name):
    accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy").evaluate(predictions)
    f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1").evaluate(predictions)
    precision = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision").evaluate(predictions)
    recall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall").evaluate(predictions)
    roc_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC").evaluate(predictions)
    pr_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR").evaluate(predictions)
    cm = confusion_values(predictions)
    bad_precision = cm["tp"] / (cm["tp"] + cm["fp"]) if (cm["tp"] + cm["fp"]) else 0.0
    bad_recall = cm["tp"] / (cm["tp"] + cm["fn"]) if (cm["tp"] + cm["fn"]) else 0.0
    return {
        "model": model_name,
        "accuracy": round(float(accuracy), 6),
        "f1": round(float(f1), 6),
        "weighted_precision": round(float(precision), 6),
        "weighted_recall": round(float(recall), 6),
        "roc_auc": round(float(roc_auc), 6),
        "pr_auc": round(float(pr_auc), 6),
        "bad_loan_precision": round(float(bad_precision), 6),
        "bad_loan_recall": round(float(bad_recall), 6),
        **cm,
    }


def train_models(train_df, test_df, numeric_cols, categorical_cols):
    print_header("TRAIN AND EVALUATE CLASSIFICATION MODELS")
    model_specs = [
        ("Logistic Regression", LogisticRegression(labelCol="label", featuresCol="features", weightCol="class_weight", maxIter=60), True),
        ("Random Forest", RandomForestClassifier(labelCol="label", featuresCol="features", weightCol="class_weight", numTrees=100, maxDepth=6, seed=SEED), False),
        ("GBTClassifier", GBTClassifier(labelCol="label", featuresCol="features", weightCol="class_weight", maxIter=50, maxDepth=5, seed=SEED), False),
    ]
    metrics = []
    models = {}
    predictions = {}
    for name, estimator, use_scaler in model_specs:
        stages = build_model_stages(numeric_cols, categorical_cols, use_scaler)
        model = Pipeline(stages=stages + [estimator]).fit(train_df)
        pred = model.transform(test_df).persist(StorageLevel.MEMORY_AND_DISK)
        pred.count()
        row = evaluate_model(pred, name)
        metrics.append(row)
        models[name] = model
        predictions[name] = pred
        print(f"{name}: Accuracy={row['accuracy']}, F1={row['f1']}, ROC-AUC={row['roc_auc']}, PR-AUC={row['pr_auc']}")
    metrics = sorted(metrics, key=lambda r: r["pr_auc"], reverse=True)
    write_dicts_to_csv(metrics, MODEL_METRICS_CSV)
    plot_model_comparison(metrics)
    return metrics, models, predictions


def save_best_model_info(best_row):
    with open(BEST_MODEL_INFO_TXT, "w", encoding="utf-8") as f:
        f.write("Best classification model\n")
        f.write(f"Model: {best_row['model']}\n")
        f.write(f"Selection metric: PR-AUC\n")
        f.write(f"Accuracy: {best_row['accuracy']}\n")
        f.write(f"F1: {best_row['f1']}\n")
        f.write(f"ROC-AUC: {best_row['roc_auc']}\n")
        f.write(f"PR-AUC: {best_row['pr_auc']}\n")
    print(f"Saved best model info: {BEST_MODEL_INFO_TXT}")


def save_streaming_data(stream_df):
    print_header("SAVE 10 PERCENT STREAMING SIMULATION DATA")
    stream_input_df = stream_df.drop("label", "class_weight")
    stream_input_df.write.mode("overwrite").parquet(STREAM_DATA_HDFS_PATH)
    stream_input_df.coalesce(1).write.mode("overwrite").json(STREAM_JSON_DIR)
    print(f"Saved HDFS stream simulation data: {STREAM_DATA_HDFS_PATH}")
    print(f"Saved local JSON sample for Kafka producer: {STREAM_JSON_DIR}")


def main():
    ensure_dirs()
    spark = SparkSession.builder.appName("Prosper_File06_Classification_MLlib").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    print_header("LOAD CLASSIFICATION DATASET")
    df = spark.read.parquet(INPUT_PATH).persist(StorageLevel.MEMORY_AND_DISK)
    df.count()
    if "label" not in df.columns:
        raise ValueError("Input dataset must contain label column.")
    print(f"Rows: {df.count()}")
    print(f"Columns: {len(df.columns)}")
    df.groupBy("label").agg(count("*").alias("count")).orderBy("label").show()
    plot_label_distribution(df)

    numeric_cols, categorical_cols = detect_feature_columns(df)
    print(f"Numeric feature candidates: {len(numeric_cols)}")
    print(f"Categorical feature candidates: {len(categorical_cols)}")

    train_df, test_df, stream_df = stratified_split_7_2_1(df)
    train_df, test_df = add_class_weight(train_df, test_df)

    selected_features = classification_feature_selection(train_df, numeric_cols, categorical_cols)
    selected_numeric, selected_categorical = split_feature_types(selected_features, numeric_cols, categorical_cols)
    print(f"Selected numeric features: {selected_numeric}")
    print(f"Selected categorical features: {selected_categorical}")

    metric_rows, model_objects, prediction_objects = train_models(train_df, test_df, selected_numeric, selected_categorical)
    best_row = metric_rows[0]
    best_name = best_row["model"]
    best_model = model_objects[best_name]
    best_predictions = prediction_objects[best_name]

    print_header("BEST CLASSIFICATION MODEL")
    print(f"Best model: {best_name}")
    print(f"PR-AUC: {best_row['pr_auc']}")
    best_model.write().overwrite().save(MODEL_OUTPUT_PATH)
    print(f"Saved best model to HDFS: {MODEL_OUTPUT_PATH}")
    save_best_model_info(best_row)

    cm = confusion_values(best_predictions)
    cm_row = {"model": best_name, **cm}
    write_dicts_to_csv([cm_row], CONFUSION_MATRIX_CSV)
    plot_confusion_matrix(cm, best_name)

    save_streaming_data(stream_df)
    spark.stop()


if __name__ == "__main__":
    main()