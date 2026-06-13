# -*- coding: utf-8 -*-
"""
FILE 07 - Regression Feature Selection + MLlib Modeling

Flow:
1. Read preprocessed regression data from HDFS.
2. Split train/test.
3. Run regression feature selection on train data only.
   - Pearson correlation for numeric features.
   - Random Forest Regressor feature importance.
   - Combined score = 0.5 * normalized correlation + 0.5 * normalized RF importance.
4. Train Linear Regression, Random Forest Regressor, and GBTRegressor.
5. Evaluate with RMSE, MAE, R2, and MAPE.
6. Save best regression PipelineModel and report artifacts.
"""

import csv
import math
import os

from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import Imputer, OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.regression import GBTRegressor, LinearRegression, RandomForestRegressor
from pyspark.sql import SparkSession
from pyspark.sql.functions import abs as spark_abs, avg, col, count, max as spark_max, min as spark_min, rand, stddev
from pyspark.sql.types import BooleanType, NumericType, StringType

SEED = 42
TRAIN_RATIO = 0.8
TOP_N_FEATURES = 12
TARGET_COL = "BorrowerAPR"

INPUT_PATH = (
    "hdfs://localhost:9000/bigdata/prosper_loan/processed/"
    "prosper_loan_preprocessed_regression"
)
MODEL_OUTPUT_PATH = (
    "hdfs://localhost:9000/bigdata/prosper_loan/models/regression/"
    "best_borrower_apr_model"
)

OUTPUT_DIR = os.path.join("outputs", "07_spark_ml_regression")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

DATASET_SUMMARY_CSV = os.path.join(TABLE_DIR, "regression_dataset_summary.csv")
FEATURE_SCORE_CSV = os.path.join(TABLE_DIR, "regression_feature_score.csv")
MODEL_METRICS_CSV = os.path.join(TABLE_DIR, "model_metrics.csv")
PREDICTION_SAMPLE_CSV = os.path.join(TABLE_DIR, "best_model_prediction_sample.csv")
BEST_MODEL_INFO_TXT = os.path.join(TABLE_DIR, "best_model_info.txt")

TARGET_DISTRIBUTION_FIG = os.path.join(FIGURE_DIR, "01_borrower_apr_distribution.png")
FEATURE_SCORE_FIG = os.path.join(FIGURE_DIR, "02_regression_feature_selection.png")
MODEL_COMPARISON_FIG = os.path.join(FIGURE_DIR, "03_model_comparison.png")
ACTUAL_VS_PREDICTED_FIG = os.path.join(FIGURE_DIR, "04_actual_vs_predicted_best_model.png")
RESIDUAL_DISTRIBUTION_FIG = os.path.join(FIGURE_DIR, "05_residual_distribution_best_model.png")

DROP_COLUMNS = {TARGET_COL, "features", "raw_features", "prediction"}


def print_header(title):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def ensure_dirs():
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)


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


def summarize_dataset(df):
    row = df.select(
        count(col(TARGET_COL)).alias("rows"),
        avg(col(TARGET_COL)).alias("mean"),
        spark_min(col(TARGET_COL)).alias("min"),
        spark_max(col(TARGET_COL)).alias("max"),
        stddev(col(TARGET_COL)).alias("stddev"),
    ).collect()[0]
    summary = {
        "rows": int(row["rows"]),
        "columns": len(df.columns),
        "target": TARGET_COL,
        "target_mean": round(safe_float(row["mean"]), 6),
        "target_min": round(safe_float(row["min"]), 6),
        "target_max": round(safe_float(row["max"]), 6),
        "target_stddev": round(safe_float(row["stddev"]), 6),
    }
    write_dicts_to_csv([summary], DATASET_SUMMARY_CSV)
    return summary


def train_test_split(df):
    print_header("TRAIN / TEST SPLIT")
    train_df, test_df = df.randomSplit([TRAIN_RATIO, 1.0 - TRAIN_RATIO], seed=SEED)
    train_df = train_df.persist(StorageLevel.MEMORY_AND_DISK)
    test_df = test_df.persist(StorageLevel.MEMORY_AND_DISK)
    print(f"Train rows: {train_df.count()}")
    print(f"Test rows : {test_df.count()}")
    return train_df, test_df


def plot_target_distribution(df):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    values = [float(r[TARGET_COL]) for r in df.select(TARGET_COL).orderBy(rand(seed=SEED)).limit(20000).collect()]
    plt.figure(figsize=(9, 5.5))
    plt.hist(values, bins=40)
    plt.xlabel("BorrowerAPR")
    plt.ylabel("Frequency")
    plt.title("Distribution of BorrowerAPR")
    plt.tight_layout()
    plt.savefig(TARGET_DISTRIBUTION_FIG, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {TARGET_DISTRIBUTION_FIG}")


def plot_feature_scores(rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    top_rows = rows[:20][::-1]
    plt.figure(figsize=(10, 7))
    plt.barh([r["feature"] for r in top_rows], [r["combined_score"] for r in top_rows])
    plt.title("Regression Feature Diagnostic for BorrowerAPR")
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
    rmse = [r["rmse"] for r in rows]
    mae = [r["mae"] for r in rows]
    r2 = [r["r2"] for r in rows]
    x = np.arange(len(models))
    width = 0.25
    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.bar(x - width / 2, rmse, width, label="RMSE")
    ax1.bar(x + width / 2, mae, width, label="MAE")
    ax1.set_ylabel("Error")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha="right")
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(x, r2, marker="o", label="R2")
    ax2.set_ylabel("R2")
    ax2.legend(loc="upper right")
    plt.title("Regression Model Comparison")
    plt.tight_layout()
    plt.savefig(MODEL_COMPARISON_FIG, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {MODEL_COMPARISON_FIG}")


def sample_predictions(predictions, sample_size=5000):
    return predictions.select(TARGET_COL, "prediction").orderBy(rand(seed=SEED)).limit(sample_size).toPandas()


def plot_actual_vs_predicted(predictions):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    pdf = sample_predictions(predictions, 5000)
    plt.figure(figsize=(7, 7))
    plt.scatter(pdf[TARGET_COL], pdf["prediction"], alpha=0.35)
    min_value = min(pdf[TARGET_COL].min(), pdf["prediction"].min())
    max_value = max(pdf[TARGET_COL].max(), pdf["prediction"].max())
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")
    plt.xlabel("Actual BorrowerAPR")
    plt.ylabel("Predicted BorrowerAPR")
    plt.title("Actual vs Predicted BorrowerAPR")
    plt.tight_layout()
    plt.savefig(ACTUAL_VS_PREDICTED_FIG, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {ACTUAL_VS_PREDICTED_FIG}")


def plot_residual_distribution(predictions):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    pdf = predictions.select((col(TARGET_COL) - col("prediction")).alias("residual")).orderBy(rand(seed=SEED)).limit(20000).toPandas()
    plt.figure(figsize=(9, 5.5))
    plt.hist(pdf["residual"], bins=50)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Residual: actual - predicted")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution for Best Regression Model")
    plt.tight_layout()
    plt.savefig(RESIDUAL_DISTRIBUTION_FIG, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {RESIDUAL_DISTRIBUTION_FIG}")


def correlation_scores(train_df, numeric_cols):
    scores = {}
    for c in numeric_cols:
        try:
            value = train_df.stat.corr(c, TARGET_COL)
        except Exception:
            value = 0.0
        scores[c] = abs(safe_float(value))
    return scores


def rf_importance_scores(train_df, numeric_cols, categorical_cols):
    stages = []
    indexed_cols = []
    for c in categorical_cols:
        idx_col = f"{c}__idx_rf"
        stages.append(StringIndexer(inputCol=c, outputCol=idx_col, handleInvalid="keep"))
        indexed_cols.append(idx_col)
    feature_cols = numeric_cols + indexed_cols
    stages.append(VectorAssembler(inputCols=feature_cols, outputCol="rf_features", handleInvalid="keep"))
    stages.append(RandomForestRegressor(labelCol=TARGET_COL, featuresCol="rf_features", numTrees=80, maxDepth=6, seed=SEED))
    model = Pipeline(stages=stages).fit(train_df)
    rf_model = model.stages[-1]
    importances = [safe_float(x) for x in rf_model.featureImportances.toArray()]
    raw_features = numeric_cols + categorical_cols
    return {raw_features[i]: importances[i] if i < len(importances) else 0.0 for i in range(len(raw_features))}


def regression_feature_selection(train_df, numeric_cols, categorical_cols):
    print_header("REGRESSION FEATURE SELECTION")
    corr_scores = correlation_scores(train_df, numeric_cols)
    rf_scores = rf_importance_scores(train_df, numeric_cols, categorical_cols)
    corr_norm = normalize(corr_scores)
    rf_norm = normalize(rf_scores)
    rows = []
    for feature in numeric_cols + categorical_cols:
        combined = 0.5 * corr_norm.get(feature, 0.0) + 0.5 * rf_norm.get(feature, 0.0)
        rows.append({
            "feature": feature,
            "feature_type": "numeric" if feature in numeric_cols else "categorical",
            "absolute_correlation": round(corr_scores.get(feature, 0.0), 8),
            "correlation_score_normalized": round(corr_norm.get(feature, 0.0), 8),
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


def build_preprocessing_stages(numeric_cols, categorical_cols, use_scaler):
    stages = []
    numeric_imputed_cols = []
    if numeric_cols:
        numeric_imputed_cols = [f"{c}__imputed" for c in numeric_cols]
        stages.append(Imputer(inputCols=numeric_cols, outputCols=numeric_imputed_cols))
    ohe_cols = []
    for c in categorical_cols:
        idx_col = f"{c}__idx"
        ohe_col = f"{c}__ohe"
        stages.append(StringIndexer(inputCol=c, outputCol=idx_col, handleInvalid="keep"))
        stages.append(OneHotEncoder(inputCols=[idx_col], outputCols=[ohe_col], handleInvalid="keep"))
        ohe_cols.append(ohe_col)
    feature_inputs = numeric_imputed_cols + ohe_cols
    if use_scaler:
        stages.append(VectorAssembler(inputCols=feature_inputs, outputCol="raw_features", handleInvalid="keep"))
        stages.append(StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=False))
    else:
        stages.append(VectorAssembler(inputCols=feature_inputs, outputCol="features", handleInvalid="keep"))
    return stages


def evaluate_regression(predictions, model_name):
    rmse = RegressionEvaluator(labelCol=TARGET_COL, predictionCol="prediction", metricName="rmse").evaluate(predictions)
    mae = RegressionEvaluator(labelCol=TARGET_COL, predictionCol="prediction", metricName="mae").evaluate(predictions)
    r2 = RegressionEvaluator(labelCol=TARGET_COL, predictionCol="prediction", metricName="r2").evaluate(predictions)
    mape_row = predictions.select(
        avg(spark_abs((col(TARGET_COL) - col("prediction")) / col(TARGET_COL))).alias("mape")
    ).collect()[0]
    return {
        "model": model_name,
        "rmse": round(safe_float(rmse), 6),
        "mae": round(safe_float(mae), 6),
        "r2": round(safe_float(r2), 6),
        "mape": round(safe_float(mape_row["mape"]), 6),
    }


def baseline_mean_model(train_df, test_df):
    mean_value = safe_float(train_df.select(avg(col(TARGET_COL)).alias("mean_value")).collect()[0]["mean_value"])
    predictions = test_df.withColumn("prediction", col(TARGET_COL) * 0 + mean_value)
    metrics = evaluate_regression(predictions, "Baseline Mean Predictor")
    metrics["baseline_mean_prediction"] = round(mean_value, 6)
    return predictions, metrics


def train_regression_models(train_df, test_df, numeric_cols, categorical_cols):
    print_header("TRAIN REGRESSION MODELS")
    _, baseline_metrics = baseline_mean_model(train_df, test_df)
    metric_rows = [baseline_metrics]
    models = {}
    predictions = {}
    specs = [
        ("Linear Regression", LinearRegression(labelCol=TARGET_COL, featuresCol="features", maxIter=60, regParam=0.01), True),
        ("Random Forest Regressor", RandomForestRegressor(labelCol=TARGET_COL, featuresCol="features", numTrees=100, maxDepth=8, seed=SEED), False),
        ("GBTRegressor", GBTRegressor(labelCol=TARGET_COL, featuresCol="features", maxIter=60, maxDepth=5, stepSize=0.05, seed=SEED), False),
    ]
    for name, estimator, use_scaler in specs:
        stages = build_preprocessing_stages(numeric_cols, categorical_cols, use_scaler)
        model = Pipeline(stages=stages + [estimator]).fit(train_df)
        pred = model.transform(test_df).persist(StorageLevel.MEMORY_AND_DISK)
        pred.count()
        metrics = evaluate_regression(pred, name)
        metric_rows.append(metrics)
        models[name] = model
        predictions[name] = pred
        print(f"{name}: RMSE={metrics['rmse']}, MAE={metrics['mae']}, R2={metrics['r2']}, MAPE={metrics['mape']}")
    write_dicts_to_csv(metric_rows, MODEL_METRICS_CSV)
    plot_model_comparison(metric_rows)
    return metric_rows, models, predictions


def save_prediction_sample(predictions):
    rows = []
    sample = predictions.select(
        col(TARGET_COL).alias("actual_borrower_apr"),
        col("prediction").alias("predicted_borrower_apr"),
        spark_abs(col(TARGET_COL) - col("prediction")).alias("absolute_error"),
    ).orderBy(rand(seed=SEED)).limit(200).collect()
    for r in sample:
        rows.append({
            "actual_borrower_apr": round(safe_float(r["actual_borrower_apr"]), 6),
            "predicted_borrower_apr": round(safe_float(r["predicted_borrower_apr"]), 6),
            "absolute_error": round(safe_float(r["absolute_error"]), 6),
        })
    write_dicts_to_csv(rows, PREDICTION_SAMPLE_CSV)


def save_best_model_info(best_row):
    with open(BEST_MODEL_INFO_TXT, "w", encoding="utf-8") as f:
        f.write("Best regression model\n")
        f.write(f"Model: {best_row['model']}\n")
        f.write("Selection metric: lowest RMSE\n")
        f.write(f"RMSE: {best_row['rmse']}\n")
        f.write(f"MAE: {best_row['mae']}\n")
        f.write(f"R2: {best_row['r2']}\n")
        f.write(f"MAPE: {best_row['mape']}\n")
    print(f"Saved best model info: {BEST_MODEL_INFO_TXT}")


def main():
    ensure_dirs()
    spark = SparkSession.builder.appName("Prosper_File07_Regression_MLlib").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    print_header("LOAD REGRESSION DATASET")
    df = spark.read.parquet(INPUT_PATH).persist(StorageLevel.MEMORY_AND_DISK)
    df = df.filter(col(TARGET_COL).isNotNull()).persist(StorageLevel.MEMORY_AND_DISK)
    df.count()
    if TARGET_COL not in df.columns:
        raise ValueError(f"{TARGET_COL} is required for regression.")
    summary = summarize_dataset(df)
    print(f"Rows: {summary['rows']}, Columns: {summary['columns']}")
    print(f"BorrowerAPR mean: {summary['target_mean']}")
    plot_target_distribution(df)

    numeric_cols, categorical_cols = detect_feature_columns(df)
    print(f"Numeric feature candidates: {len(numeric_cols)}")
    print(f"Categorical feature candidates: {len(categorical_cols)}")
    print("BorrowerAPR is excluded from features because it is the regression target.")

    train_df, test_df = train_test_split(df)
    selected_features = regression_feature_selection(train_df, numeric_cols, categorical_cols)
    selected_numeric, selected_categorical = split_feature_types(selected_features, numeric_cols, categorical_cols)
    print(f"Selected numeric features: {selected_numeric}")
    print(f"Selected categorical features: {selected_categorical}")

    metric_rows, model_objects, prediction_objects = train_regression_models(train_df, test_df, selected_numeric, selected_categorical)
    candidate_rows = [r for r in metric_rows if r["model"] != "Baseline Mean Predictor"]
    best_row = sorted(candidate_rows, key=lambda r: (r["rmse"], r["mae"], -r["r2"]))[0]
    best_name = best_row["model"]
    best_model = model_objects[best_name]
    best_predictions = prediction_objects[best_name]

    print_header("BEST REGRESSION MODEL")
    print(f"Best model: {best_name}")
    print(f"RMSE={best_row['rmse']}, MAE={best_row['mae']}, R2={best_row['r2']}, MAPE={best_row['mape']}")
    best_model.write().overwrite().save(MODEL_OUTPUT_PATH)
    print(f"Saved best model to HDFS: {MODEL_OUTPUT_PATH}")
    save_best_model_info(best_row)

    plot_actual_vs_predicted(best_predictions)
    plot_residual_distribution(best_predictions)
    save_prediction_sample(best_predictions)
    spark.stop()


if __name__ == "__main__":
    main()