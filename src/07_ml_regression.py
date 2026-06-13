import os
import csv
import math

from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    avg,
    min as spark_min,
    max as spark_max,
    stddev,
    abs as spark_abs,
    rand,
)
from pyspark.sql.types import StringType, BooleanType
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    Imputer,
    VectorAssembler,
    StandardScaler,
)
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator


INPUT_PATH = "hdfs://localhost:9000/bigdata/prosper_loan/processed/prosper_loan_preprocessed_regression"
MODEL_OUTPUT_PATH = "hdfs://localhost:9000/bigdata/prosper_loan/models/regression/best_borrower_apr_model"

TARGET_COL = "BorrowerAPR"
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

OUTPUT_DIR = os.path.join("outputs", "07_spark_ml_regression")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

DATASET_SUMMARY_CSV = os.path.join(TABLE_DIR, "07_regression_dataset_summary.csv")
FEATURE_SCORE_CSV = os.path.join(TABLE_DIR, "07_regression_feature_scores.csv")
MODEL_METRICS_CSV = os.path.join(TABLE_DIR, "07_regression_model_metrics.csv")
PREDICTION_SAMPLE_CSV = os.path.join(TABLE_DIR, "07_best_model_prediction_sample.csv")

TARGET_DISTRIBUTION_FIG = os.path.join(FIGURE_DIR, "07_borrower_apr_distribution.png")
FEATURE_SCORE_FIG = os.path.join(FIGURE_DIR, "07_regression_feature_scores.png")
MODEL_COMPARISON_FIG = os.path.join(FIGURE_DIR, "07_regression_model_comparison.png")
ACTUAL_VS_PREDICTED_FIG = os.path.join(FIGURE_DIR, "07_actual_vs_predicted_gbt.png")
RESIDUAL_DISTRIBUTION_FIG = os.path.join(FIGURE_DIR, "07_residual_distribution_gbt.png")


def print_header(title):
    print("\n" + "=" * 86)
    print(title)
    print("=" * 86)


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_dicts_to_csv(rows, path):
    if not rows:
        return
    ensure_parent_dir(path)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved table: {path}")


def detect_feature_columns(df, target_col=TARGET_COL):
    numeric_cols = []
    categorical_cols = []

    for field in df.schema.fields:
        name = field.name
        if name == target_col:
            continue
        if isinstance(field.dataType, (StringType, BooleanType)):
            categorical_cols.append(name)
        else:
            numeric_cols.append(name)

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
        "target_mean": float(row["mean"]),
        "target_min": float(row["min"]),
        "target_max": float(row["max"]),
        "target_stddev": float(row["stddev"]),
    }

    write_dicts_to_csv([summary], DATASET_SUMMARY_CSV)
    return summary


def plot_target_distribution(df, output_path, sample_size=20000):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed. Skip target distribution figure.")
        return

    values = [
        float(row[TARGET_COL])
        for row in df.select(TARGET_COL).orderBy(rand(seed=RANDOM_SEED)).limit(sample_size).collect()
    ]

    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=40, edgecolor="white", color="#2F80ED")
    plt.xlabel("BorrowerAPR")
    plt.ylabel("Frequency")
    plt.title("Distribution of BorrowerAPR")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {output_path}")


def train_test_split(df):
    train_df, test_df = df.randomSplit([TRAIN_RATIO, 1 - TRAIN_RATIO], seed=RANDOM_SEED)
    return (
        train_df.persist(StorageLevel.MEMORY_AND_DISK),
        test_df.persist(StorageLevel.MEMORY_AND_DISK),
    )


def minmax_normalize(scores):
    if not scores:
        return {}
    max_value = max(scores.values())
    if max_value <= 0:
        return {key: 0.0 for key in scores}
    return {key: value / max_value for key, value in scores.items()}


def correlation_scores(train_df, numeric_cols):
    rows = []
    scores = {}

    for c in numeric_cols:
        try:
            corr_value = train_df.stat.corr(c, TARGET_COL)
        except Exception:
            corr_value = None

        if corr_value is None or math.isnan(corr_value):
            corr_value = 0.0

        abs_corr = abs(float(corr_value))
        rows.append({
            "feature": c,
            "pearson_correlation": float(corr_value),
            "absolute_correlation": abs_corr,
        })
        scores[c] = abs_corr

    return rows, scores


def rf_regression_importance_scores(train_df, numeric_cols, categorical_cols):
    indexed_cols = [f"{c}__idx_rf" for c in categorical_cols]
    stages = []

    if categorical_cols:
        stages.append(StringIndexer(
            inputCols=categorical_cols,
            outputCols=indexed_cols,
            handleInvalid="keep",
        ))

    feature_cols = numeric_cols + indexed_cols
    stages.append(VectorAssembler(
        inputCols=feature_cols,
        outputCol="rf_features",
        handleInvalid="keep",
    ))

    rf = RandomForestRegressor(
        labelCol=TARGET_COL,
        featuresCol="rf_features",
        numTrees=80,
        maxDepth=6,
        seed=RANDOM_SEED,
    )
    stages.append(rf)

    model = Pipeline(stages=stages).fit(train_df)
    rf_model = model.stages[-1]
    importances = list(rf_model.featureImportances.toArray())

    raw_features = numeric_cols + categorical_cols
    rows = []
    scores = {}

    for i, feature in enumerate(raw_features):
        importance = float(importances[i]) if i < len(importances) else 0.0
        rows.append({"feature": feature, "rf_importance": importance})
        scores[feature] = importance

    return rows, scores


def regression_feature_reason(feature):
    reasons = {
        "EstimatedLoss": "Strong Prosper risk signal closely related to loan pricing.",
        "ProsperScore": "Internal Prosper risk score reflecting borrower risk level.",
        "CreditScoreRangeLower": "Borrower credit quality indicator.",
        "LoanOriginalAmount": "Loan exposure size that may affect pricing.",
        "BankcardUtilization": "Credit utilization pressure.",
        "DelinquenciesLast7Years": "Long-term delinquency history.",
        "CurrentDelinquencies": "Current delinquency risk.",
        "InquiriesLast6Months": "Recent credit-seeking behavior.",
        "DebtToIncomeRatio": "Debt burden relative to income.",
        "TotalTrades": "Credit history depth.",
        "Term": "Loan duration and risk exposure period.",
        "listing_year": "Loan vintage or period effect.",
        "listing_month": "Possible monthly period effect.",
        "loan_origination_year": "Loan origination period effect.",
        "loan_origination_month": "Possible origination month effect.",
        "credit_history_days": "Length of borrower credit history.",
        "credit_pull_to_origination_days": "Operational timing signal.",
        "IncomeRange": "Borrower repayment capacity.",
        "IsBorrowerHomeowner": "Borrower profile information.",
    }
    return reasons.get(feature, "Retained after regression feature diagnostic.")


def regression_feature_diagnostic(train_df, numeric_cols, categorical_cols):
    corr_rows, corr_scores = correlation_scores(train_df, numeric_cols)
    _, rf_scores = rf_regression_importance_scores(train_df, numeric_cols, categorical_cols)

    corr_norm = minmax_normalize(corr_scores)
    rf_norm = minmax_normalize(rf_scores)

    all_features = numeric_cols + categorical_cols
    rows = []

    for feature in all_features:
        combined_score = 0.5 * corr_norm.get(feature, 0.0) + 0.5 * rf_norm.get(feature, 0.0)
        rows.append({
            "feature": feature,
            "absolute_correlation": corr_scores.get(feature, 0.0),
            "correlation_score_normalized": corr_norm.get(feature, 0.0),
            "rf_importance": rf_scores.get(feature, 0.0),
            "rf_importance_normalized": rf_norm.get(feature, 0.0),
            "combined_score": combined_score,
            "selection_decision": "KEEP",
        })

    rows = sorted(rows, key=lambda row: row["combined_score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    write_dicts_to_csv(rows, FEATURE_SCORE_CSV)
    return rows


def plot_feature_scores(rows, output_path, top_n=20):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed. Skip feature score figure.")
        return

    top_rows = rows[:top_n]
    features = [r["feature"] for r in top_rows][::-1]
    scores = [r["combined_score"] for r in top_rows][::-1]

    plt.figure(figsize=(11, 7))
    plt.barh(features, scores, color="#2F80ED")
    plt.xlabel("Combined feature score")
    plt.ylabel("Feature")
    plt.title("Regression Feature Diagnostic for BorrowerAPR")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {output_path}")


def build_preprocessing_stages(numeric_cols, categorical_cols, use_scaler=False):
    stages = []
    numeric_imputed_cols = [f"{c}__imputed" for c in numeric_cols]
    categorical_index_cols = [f"{c}__idx" for c in categorical_cols]
    categorical_ohe_cols = [f"{c}__ohe" for c in categorical_cols]

    if numeric_cols:
        stages.append(Imputer(inputCols=numeric_cols, outputCols=numeric_imputed_cols))

    if categorical_cols:
        stages.append(StringIndexer(
            inputCols=categorical_cols,
            outputCols=categorical_index_cols,
            handleInvalid="keep",
        ))
        stages.append(OneHotEncoder(
            inputCols=categorical_index_cols,
            outputCols=categorical_ohe_cols,
            handleInvalid="keep",
        ))

    stages.append(VectorAssembler(
        inputCols=numeric_imputed_cols + categorical_ohe_cols,
        outputCol="raw_features",
        handleInvalid="keep",
    ))

    if use_scaler:
        stages.append(StandardScaler(
            inputCol="raw_features",
            outputCol="features",
            withStd=True,
            withMean=False,
        ))
    else:
        stages.append(VectorAssembler(inputCols=["raw_features"], outputCol="features"))

    return stages


def evaluate_regression(predictions, model_name):
    rmse_eval = RegressionEvaluator(labelCol=TARGET_COL, predictionCol="prediction", metricName="rmse")
    mae_eval = RegressionEvaluator(labelCol=TARGET_COL, predictionCol="prediction", metricName="mae")
    r2_eval = RegressionEvaluator(labelCol=TARGET_COL, predictionCol="prediction", metricName="r2")

    rmse = float(rmse_eval.evaluate(predictions))
    mae = float(mae_eval.evaluate(predictions))
    r2 = float(r2_eval.evaluate(predictions))

    extra = predictions.select(
        avg(spark_abs(col(TARGET_COL) - col("prediction"))).alias("mae_check"),
        avg(spark_abs((col(TARGET_COL) - col("prediction")) / col(TARGET_COL))).alias("mape"),
    ).collect()[0]

    return {
        "model": model_name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": float(extra["mape"]),
    }


def train_regression_model(model_name, estimator, train_df, test_df, numeric_cols, categorical_cols, use_scaler=False):
    stages = build_preprocessing_stages(numeric_cols, categorical_cols, use_scaler=use_scaler)
    pipeline = Pipeline(stages=stages + [estimator])
    model = pipeline.fit(train_df)
    predictions = model.transform(test_df).persist(StorageLevel.MEMORY_AND_DISK)
    predictions.count()
    metrics = evaluate_regression(predictions, model_name)
    return model, predictions, metrics


def baseline_mean_model(train_df, test_df):
    mean_value = float(train_df.select(avg(col(TARGET_COL)).alias("mean_apr")).collect()[0]["mean_apr"])
    predictions = test_df.withColumn("prediction", col(TARGET_COL) * 0 + mean_value)
    metrics = evaluate_regression(predictions, "Baseline Mean Predictor")
    metrics["baseline_mean_prediction"] = mean_value
    return predictions, metrics


def plot_model_comparison(rows, output_path):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib or numpy is not installed. Skip model comparison figure.")
        return

    models = [r["model"] for r in rows]
    rmse = [r["rmse"] for r in rows]
    mae = [r["mae"] for r in rows]
    r2 = [r["r2"] for r in rows]

    x = np.arange(len(models))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(x - width / 2, rmse, width, label="RMSE", color="#2F80ED")
    ax1.bar(x + width / 2, mae, width, label="MAE", color="#56CCF2")
    ax1.set_ylabel("Error")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha="right")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(x, r2, marker="o", linewidth=2, label="R2", color="#EB5757")
    ax2.set_ylabel("R2")
    ax2.legend(loc="upper right")

    plt.title("Regression Model Comparison")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {output_path}")


def sample_predictions_to_pandas(predictions, sample_size=5000):
    return (
        predictions
        .select(TARGET_COL, "prediction")
        .orderBy(rand(seed=RANDOM_SEED))
        .limit(sample_size)
        .toPandas()
    )


def plot_actual_vs_predicted(predictions, output_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed. Skip actual vs predicted figure.")
        return

    pdf = sample_predictions_to_pandas(predictions)
    plt.figure(figsize=(7, 7))
    plt.scatter(pdf[TARGET_COL], pdf["prediction"], alpha=0.35, color="#2F80ED", edgecolors="none")
    min_value = min(pdf[TARGET_COL].min(), pdf["prediction"].min())
    max_value = max(pdf[TARGET_COL].max(), pdf["prediction"].max())
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="#EB5757")
    plt.xlabel("Actual BorrowerAPR")
    plt.ylabel("Predicted BorrowerAPR")
    plt.title("Actual vs Predicted BorrowerAPR")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {output_path}")


def plot_residual_distribution(predictions, output_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed. Skip residual distribution figure.")
        return

    pdf = (
        predictions
        .select((col(TARGET_COL) - col("prediction")).alias("residual"))
        .orderBy(rand(seed=RANDOM_SEED))
        .limit(20000)
        .toPandas()
    )

    plt.figure(figsize=(10, 6))
    plt.hist(pdf["residual"], bins=50, color="#2F80ED", edgecolor="white")
    plt.axvline(0, linestyle="--", color="#EB5757")
    plt.xlabel("Residual: actual - predicted")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution for Best Regression Model")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {output_path}")



def save_prediction_sample(predictions, path, sample_size=200):
    rows = []
    sample = (
        predictions
        .select(TARGET_COL, "prediction")
        .withColumn("absolute_error", spark_abs(col(TARGET_COL) - col("prediction")))
        .orderBy(rand(seed=RANDOM_SEED))
        .limit(sample_size)
        .collect()
    )
    for row in sample:
        rows.append({
            "actual_borrower_apr": float(row[TARGET_COL]),
            "predicted_borrower_apr": float(row["prediction"]),
            "absolute_error": float(row["absolute_error"]),
        })
    write_dicts_to_csv(rows, path)


def main():
    spark = SparkSession.builder.appName("Prosper Loan Spark ML Regression").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    print_header("LOAD REGRESSION DATASET")
    df = spark.read.parquet(INPUT_PATH).persist(StorageLevel.MEMORY_AND_DISK)
    df.count()

    if TARGET_COL not in df.columns:
        raise ValueError(f"{TARGET_COL} is required for regression.")

    df = df.filter(col(TARGET_COL).isNotNull()).persist(StorageLevel.MEMORY_AND_DISK)
    summary = summarize_dataset(df)
    print(f"Rows: {summary['rows']}, Columns: {summary['columns']}")
    print(f"BorrowerAPR mean: {summary['target_mean']:.6f}")

    numeric_cols, categorical_cols = detect_feature_columns(df)
    print(f"Numeric features: {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    print("BorrowerAPR is excluded from features because it is the regression target.")

    plot_target_distribution(df, TARGET_DISTRIBUTION_FIG)

    train_df, test_df = train_test_split(df)
    train_count = train_df.count()
    test_count = test_df.count()
    print(f"Train rows: {train_count}")
    print(f"Test rows : {test_count}")

    print_header("REGRESSION FEATURE DIAGNOSTIC")
    feature_rows = regression_feature_diagnostic(train_df, numeric_cols, categorical_cols)
    plot_feature_scores(feature_rows, FEATURE_SCORE_FIG)

    print_header("TRAIN REGRESSION MODELS")
    _, baseline_metrics = baseline_mean_model(train_df, test_df)

    lr = LinearRegression(
        labelCol=TARGET_COL,
        featuresCol="features",
        maxIter=80,
        regParam=0.01,
        elasticNetParam=0.0,
    )

    rf = RandomForestRegressor(
        labelCol=TARGET_COL,
        featuresCol="features",
        numTrees=100,
        maxDepth=8,
        seed=RANDOM_SEED,
    )

    gbt = GBTRegressor(
        labelCol=TARGET_COL,
        featuresCol="features",
        maxIter=80,
        maxDepth=5,
        stepSize=0.05,
        seed=RANDOM_SEED,
    )

    model_objects = {}
    prediction_objects = {}
    metric_rows = [baseline_metrics]

    for name, estimator, use_scaler in [
        ("Linear Regression", lr, True),
        ("Random Forest Regressor", rf, False),
        ("GBTRegressor", gbt, False),
    ]:
        model, predictions, metrics = train_regression_model(
            model_name=name,
            estimator=estimator,
            train_df=train_df,
            test_df=test_df,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            use_scaler=use_scaler,
        )
        model_objects[name] = model
        prediction_objects[name] = predictions
        metric_rows.append(metrics)
        print(f"{name}: RMSE={metrics['rmse']:.6f}, MAE={metrics['mae']:.6f}, R2={metrics['r2']:.6f}, MAPE={metrics['mape']:.6f}")

    write_dicts_to_csv(metric_rows, MODEL_METRICS_CSV)
    plot_model_comparison(metric_rows, MODEL_COMPARISON_FIG)

    candidate_rows = [row for row in metric_rows if row["model"] != "Baseline Mean Predictor"]
    best_row = sorted(candidate_rows, key=lambda row: (row["rmse"], row["mae"], -row["r2"]))[0]
    best_model_name = best_row["model"]
    best_model = model_objects[best_model_name]
    best_predictions = prediction_objects[best_model_name]

    print_header("BEST REGRESSION MODEL")
    print(f"Best model: {best_model_name}")
    print(f"RMSE={best_row['rmse']:.6f}, MAE={best_row['mae']:.6f}, R2={best_row['r2']:.6f}, MAPE={best_row['mape']:.6f}")

    plot_actual_vs_predicted(best_predictions, ACTUAL_VS_PREDICTED_FIG)
    plot_residual_distribution(best_predictions, RESIDUAL_DISTRIBUTION_FIG)
    save_prediction_sample(best_predictions, PREDICTION_SAMPLE_CSV)


    best_model.write().overwrite().save(MODEL_OUTPUT_PATH)
    print(f"Best model saved to: {MODEL_OUTPUT_PATH}")

    spark.stop()


if __name__ == "__main__":
    main()