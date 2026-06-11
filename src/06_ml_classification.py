# ============================================================
# Good Loan / Bad Loan Classification
# ============================================================
#
# Input:
#   hdfs://localhost:9000/bigdata/prosper_loan/processed/
#   prosper_loan_preprocessed_classification
#
# This dataset was already created in the preprocessing step.
# It already contains:
#   - label = 0 for Good Loan
#   - label = 1 for Bad Loan
#
# Models used:
#   1. Logistic Regression
#   2. Random Forest Classifier
#   3. Gradient Boosting Classifier
#
# Outputs:
#   - Console logs for dataset validation, model metrics, and confusion matrix
#   - Local CSV summary table:
#       outputs/tables/classification_model_results.csv
#   - Optional HDFS model outputs:
#       hdfs://localhost:9000/bigdata/prosper_loan/models/classification/
#
# ============================================================


import csv
import os
from typing import Dict, List, Tuple

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, lit, when
from pyspark.sql.types import StringType, BooleanType

from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler,
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

HDFS_CLASSIFICATION_INPUT_PATH = (
    "hdfs://localhost:9000/bigdata/prosper_loan/processed/"
    "prosper_loan_preprocessed_classification"
)

HDFS_MODEL_OUTPUT_BASE_PATH = (
    "hdfs://localhost:9000/bigdata/prosper_loan/models/classification"
)

LOCAL_TABLE_OUTPUT_DIR = os.path.join("outputs", "tables")
CLASSIFICATION_RESULT_CSV = os.path.join(
    LOCAL_TABLE_OUTPUT_DIR,
    "classification_model_results.csv",
)


# ============================================================
# GENERAL UTILITY FUNCTIONS
# ============================================================

def print_header(title: str) -> None:
    """Print a clear section header in the terminal."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def ensure_local_output_dir() -> None:
    """Create local output folder for result tables if it does not exist."""
    os.makedirs(LOCAL_TABLE_OUTPUT_DIR, exist_ok=True)


def print_dataset_overview(df, label_col: str = "label") -> None:
    """
    Print dataset size, schema, and label distribution.

    Why:
    Before training ML models, we must confirm:
      - the dataset is not empty,
      - the label column exists,
      - class distribution is reasonable.
    """
    print_header("DATASET OVERVIEW")

    total_rows = df.count()
    total_cols = len(df.columns)

    print(f"Input path: {HDFS_CLASSIFICATION_INPUT_PATH}")
    print(f"Total rows: {total_rows}")
    print(f"Total columns: {total_cols}")

    print("\nColumn list:")
    for idx, column_name in enumerate(df.columns, start=1):
        print(f"{idx:02d}. {column_name}")

    print("\nSchema:")
    df.printSchema()

    if label_col not in df.columns:
        raise ValueError(f"Required label column '{label_col}' does not exist.")

    print("\nLabel distribution:")
    label_rows = (
        df.groupBy(label_col)
        .agg(count("*").alias("count"))
        .orderBy(label_col)
        .collect()
    )

    print("label | meaning   | count | percentage")
    print("----------------------------------------")
    for row in label_rows:
        label_value = int(row[label_col])
        row_count = int(row["count"])
        pct = (row_count / total_rows) * 100 if total_rows > 0 else 0.0
        meaning = "Good Loan" if label_value == 0 else "Bad Loan"
        print(f"{label_value:<5} | {meaning:<9} | {row_count:<5} | {pct:.2f}%")


def get_feature_columns(df, label_col: str = "label") -> Tuple[List[str], List[str], List[str]]:
    """
    Identify feature columns automatically.

    In this classification task:
      - label is the target, so it is excluded from features.
      - BorrowerAPR is intentionally kept as a feature based on the user's decision.
      - All remaining columns are used as input features.

    Spark ML requires:
      - categorical string columns to be indexed and one-hot encoded,
      - numeric columns to be assembled directly.
    """
    feature_cols = [c for c in df.columns if c != label_col]

    categorical_cols = []
    numeric_cols = []

    for field in df.schema.fields:
        col_name = field.name

        if col_name == label_col:
            continue

        if isinstance(field.dataType, (StringType, BooleanType)):
            categorical_cols.append(col_name)
        else:
            numeric_cols.append(col_name)

    print_header("FEATURE COLUMN DETECTION")
    print(f"Label column: {label_col}")
    print(f"Total feature columns: {len(feature_cols)}")

    print("\nNumeric feature columns:")
    if numeric_cols:
        for col_name in numeric_cols:
            print(f"- {col_name}")
    else:
        print("- None")

    print("\nCategorical feature columns:")
    if categorical_cols:
        for col_name in categorical_cols:
            print(f"- {col_name}")
    else:
        print("- None")

    if "BorrowerAPR" in feature_cols:
        print(
            "\nDecision: BorrowerAPR is kept as a classification feature. "
            "The model therefore uses pricing information together with borrower "
            "and loan characteristics."
        )

    return feature_cols, numeric_cols, categorical_cols


def build_preprocessing_stages(
    numeric_cols: List[str],
    categorical_cols: List[str],
    use_scaler: bool,
) -> Tuple[List, str]:
    """
    Build Spark ML preprocessing stages.

    For categorical variables:
      StringIndexer converts text categories into numeric indexes.
      OneHotEncoder converts category indexes into sparse binary vectors.

    For numeric variables:
      Numeric columns are directly used in VectorAssembler.

    For Logistic Regression:
      StandardScaler is used because linear models are sensitive to feature scale.

    For tree-based models:
      StandardScaler is not required because decision trees split features by thresholds.
    """
    stages = []

    indexed_cols = []
    encoded_cols = []

    for c in categorical_cols:
        index_col = f"{c}_idx"
        encoded_col = f"{c}_ohe"

        indexer = StringIndexer(
            inputCol=c,
            outputCol=index_col,
            handleInvalid="keep",
        )

        indexed_cols.append(index_col)
        encoded_cols.append(encoded_col)
        stages.append(indexer)

    if categorical_cols:
        encoder = OneHotEncoder(
            inputCols=indexed_cols,
            outputCols=encoded_cols,
            handleInvalid="keep",
        )
        stages.append(encoder)

    assembler_inputs = numeric_cols + encoded_cols

    if not assembler_inputs:
        raise ValueError("No valid feature columns found for VectorAssembler.")

    assembler = VectorAssembler(
        inputCols=assembler_inputs,
        outputCol="features",
        handleInvalid="keep",
    )
    stages.append(assembler)

    if use_scaler:
        scaler = StandardScaler(
            inputCol="features",
            outputCol="scaled_features",
            withStd=True,
            withMean=False,
        )
        stages.append(scaler)
        final_features_col = "scaled_features"
    else:
        final_features_col = "features"

    return stages, final_features_col


def get_feature_names_from_metadata(transformed_df, features_col: str = "features") -> List[str]:
    """
    Extract readable feature names from Spark vector metadata.

    Why:
    Tree models return feature importance by vector index.
    This helper maps vector index back to the original feature/encoded feature name.

    If metadata is not available, the function returns generic feature names.
    """
    metadata = transformed_df.schema[features_col].metadata

    attrs = []
    if "ml_attr" in metadata and "attrs" in metadata["ml_attr"]:
        for attr_type in ["numeric", "binary"]:
            attrs.extend(metadata["ml_attr"]["attrs"].get(attr_type, []))

        attrs = sorted(attrs, key=lambda x: x["idx"])
        return [a["name"] for a in attrs]

    vector_size = metadata.get("ml_attr", {}).get("num_attrs", 0)
    if vector_size == 0:
        return []

    return [f"feature_{i}" for i in range(vector_size)]


def compute_confusion_matrix(predictions, label_col: str = "label") -> Dict[str, int]:
    """
    Compute confusion matrix values.

    In this project:
      label 0 = Good Loan
      label 1 = Bad Loan

    Confusion matrix terms:
      TN: actual Good, predicted Good
      FP: actual Good, predicted Bad
      FN: actual Bad, predicted Good
      TP: actual Bad, predicted Bad
    """
    matrix_rows = (
        predictions
        .select(col(label_col).cast("int").alias("label_int"),
                col("prediction").cast("int").alias("prediction_int"))
        .groupBy("label_int", "prediction_int")
        .agg(count("*").alias("count"))
        .collect()
    )

    matrix = {
        "TN": 0,
        "FP": 0,
        "FN": 0,
        "TP": 0,
    }

    for row in matrix_rows:
        actual = row["label_int"]
        pred = row["prediction_int"]
        row_count = int(row["count"])

        if actual == 0 and pred == 0:
            matrix["TN"] = row_count
        elif actual == 0 and pred == 1:
            matrix["FP"] = row_count
        elif actual == 1 and pred == 0:
            matrix["FN"] = row_count
        elif actual == 1 and pred == 1:
            matrix["TP"] = row_count

    return matrix


def evaluate_classification_model(predictions, model_name: str) -> Dict[str, float]:
    """
    Evaluate one classification model using multiple metrics.

    Why multiple metrics:
      Accuracy alone can be misleading when class distribution is imbalanced.
      This dataset has more Good Loans than Bad Loans, so F1, Recall, Precision,
      ROC-AUC and PR-AUC are also reported.
    """
    label_col = "label"

    evaluator_accuracy = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="accuracy",
    )

    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="f1",
    )

    evaluator_precision = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="weightedPrecision",
    )

    evaluator_recall = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="weightedRecall",
    )

    evaluator_auc = BinaryClassificationEvaluator(
        labelCol=label_col,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )

    evaluator_pr = BinaryClassificationEvaluator(
        labelCol=label_col,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR",
    )

    accuracy = evaluator_accuracy.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)
    precision = evaluator_precision.evaluate(predictions)
    recall = evaluator_recall.evaluate(predictions)
    auc = evaluator_auc.evaluate(predictions)
    aupr = evaluator_pr.evaluate(predictions)

    confusion = compute_confusion_matrix(predictions, label_col=label_col)

    tn = confusion["TN"]
    fp = confusion["FP"]
    fn = confusion["FN"]
    tp = confusion["TP"]

    # Bad Loan recall is important because label 1 means risky/bad loan.
    # It shows how many actual bad loans the model successfully detects.
    bad_loan_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Bad Loan precision shows how reliable the model is when it predicts Bad Loan.
    bad_loan_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    print_header(f"EVALUATION - {model_name}")
    print(f"Accuracy           : {accuracy:.4f}")
    print(f"F1-score           : {f1:.4f}")
    print(f"Weighted Precision : {precision:.4f}")
    print(f"Weighted Recall    : {recall:.4f}")
    print(f"ROC-AUC            : {auc:.4f}")
    print(f"PR-AUC             : {aupr:.4f}")

    print("\nConfusion Matrix:")
    print("Actual \\ Predicted | Good Loan (0) | Bad Loan (1)")
    print("--------------------------------------------------")
    print(f"Good Loan (0)      | {tn:<13} | {fp:<12}")
    print(f"Bad Loan (1)       | {fn:<13} | {tp:<12}")

    print("\nBad Loan specific metrics:")
    print(f"Bad Loan Precision : {bad_loan_precision:.4f}")
    print(f"Bad Loan Recall    : {bad_loan_recall:.4f}")

    return {
        "model": model_name,
        "accuracy": accuracy,
        "f1": f1,
        "weighted_precision": precision,
        "weighted_recall": recall,
        "roc_auc": auc,
        "pr_auc": aupr,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "bad_loan_precision": bad_loan_precision,
        "bad_loan_recall": bad_loan_recall,
    }


def save_results_to_csv(results: List[Dict[str, float]]) -> None:
    """
    Save model comparison results to a local CSV file.

    This file can be inserted into the report as the classification result table.
    """
    ensure_local_output_dir()

    fieldnames = [
        "model",
        "accuracy",
        "f1",
        "weighted_precision",
        "weighted_recall",
        "roc_auc",
        "pr_auc",
        "tn",
        "fp",
        "fn",
        "tp",
        "bad_loan_precision",
        "bad_loan_recall",
    ]

    with open(CLASSIFICATION_RESULT_CSV, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            rounded_row = dict(row)

            for key in [
                "accuracy",
                "f1",
                "weighted_precision",
                "weighted_recall",
                "roc_auc",
                "pr_auc",
                "bad_loan_precision",
                "bad_loan_recall",
            ]:
                rounded_row[key] = round(float(rounded_row[key]), 6)

            writer.writerow(rounded_row)

    print(f"\nClassification result table saved to: {CLASSIFICATION_RESULT_CSV}")


def print_top_feature_importance(
    fitted_pipeline_model,
    transformed_train_df,
    model_stage_index: int,
    model_name: str,
    top_n: int = 15,
) -> None:
    """
    Print top feature importance for tree-based models.

    Random Forest and GBT provide featureImportances.
    Logistic Regression uses coefficients instead, so it is not handled here.
    """
    print_header(f"TOP FEATURE IMPORTANCE - {model_name}")

    feature_names = get_feature_names_from_metadata(transformed_train_df, features_col="features")

    model = fitted_pipeline_model.stages[model_stage_index]

    if not hasattr(model, "featureImportances"):
        print(f"{model_name} does not provide featureImportances.")
        return

    importances = model.featureImportances.toArray().tolist()

    importance_rows = []
    for idx, importance_value in enumerate(importances):
        feature_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        importance_rows.append((feature_name, importance_value))

    importance_rows = sorted(importance_rows, key=lambda x: x[1], reverse=True)

    print(f"Top {top_n} features:")
    for rank, (feature_name, importance_value) in enumerate(importance_rows[:top_n], start=1):
        print(f"{rank:02d}. {feature_name:<50} {importance_value:.6f}")


def print_logistic_regression_coefficients(
    fitted_pipeline_model,
    transformed_train_df,
    model_stage_index: int,
    top_n: int = 15,
) -> None:
    """
    Print the largest absolute Logistic Regression coefficients.

    Why:
    Logistic Regression is the baseline model and is easier to interpret.
    A positive coefficient means the feature increases the probability of label 1
    or Bad Loan, while a negative coefficient means the feature reduces that risk.
    """
    print_header("TOP LOGISTIC REGRESSION COEFFICIENTS")

    feature_names = get_feature_names_from_metadata(transformed_train_df, features_col="features")

    model = fitted_pipeline_model.stages[model_stage_index]

    coefficients = model.coefficients.toArray().tolist()

    coef_rows = []
    for idx, coef_value in enumerate(coefficients):
        feature_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        coef_rows.append((feature_name, coef_value, abs(coef_value)))

    coef_rows = sorted(coef_rows, key=lambda x: x[2], reverse=True)

    print(f"Top {top_n} coefficients by absolute value:")
    for rank, (feature_name, coef_value, abs_value) in enumerate(coef_rows[:top_n], start=1):
        direction = "increases Bad Loan risk" if coef_value > 0 else "decreases Bad Loan risk"
        print(f"{rank:02d}. {feature_name:<50} coef={coef_value:.6f} | {direction}")


# ============================================================
# MODEL TRAINING FUNCTIONS
# ============================================================

def train_logistic_regression(train_df, test_df, numeric_cols, categorical_cols):
    """
    Train Logistic Regression.

    Role in project:
      Logistic Regression is the baseline model.
      It is fast, stable, and interpretable through coefficients.
    """
    print_header("TRAINING MODEL 1 - LOGISTIC REGRESSION")

    preprocessing_stages, final_features_col = build_preprocessing_stages(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        use_scaler=True,
    )

    lr = LogisticRegression(
        featuresCol=final_features_col,
        labelCol="label",
        predictionCol="prediction",
        rawPredictionCol="rawPrediction",
        probabilityCol="probability",
        maxIter=100,
        regParam=0.01,
        elasticNetParam=0.0,
    )

    stages = preprocessing_stages + [lr]

    pipeline = Pipeline(stages=stages)
    pipeline_model = pipeline.fit(train_df)

    predictions = pipeline_model.transform(test_df)

    result = evaluate_classification_model(
        predictions=predictions,
        model_name="Logistic Regression",
    )

    # For coefficient names, transform train data using preprocessing stages only.
    preprocessing_model = Pipeline(stages=preprocessing_stages).fit(train_df)
    transformed_train = preprocessing_model.transform(train_df)

    print_logistic_regression_coefficients(
        fitted_pipeline_model=pipeline_model,
        transformed_train_df=transformed_train,
        model_stage_index=len(stages) - 1,
        top_n=15,
    )

    return pipeline_model, predictions, result


def train_random_forest(train_df, test_df, numeric_cols, categorical_cols):
    """
    Train Random Forest Classifier.

    Role in project:
      Random Forest is an ensemble model.
      It combines many decision trees, usually performs more stably than a single tree,
      and can capture non-linear relationships between features and the Good/Bad label.
    """
    print_header("TRAINING MODEL 2 - RANDOM FOREST CLASSIFIER")

    preprocessing_stages, final_features_col = build_preprocessing_stages(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        use_scaler=False,
    )

    rf = RandomForestClassifier(
        featuresCol=final_features_col,
        labelCol="label",
        predictionCol="prediction",
        rawPredictionCol="rawPrediction",
        probabilityCol="probability",
        numTrees=100,
        maxDepth=8,
        seed=42,
    )

    stages = preprocessing_stages + [rf]

    pipeline = Pipeline(stages=stages)
    pipeline_model = pipeline.fit(train_df)

    predictions = pipeline_model.transform(test_df)

    result = evaluate_classification_model(
        predictions=predictions,
        model_name="Random Forest Classifier",
    )

    preprocessing_model = Pipeline(stages=preprocessing_stages).fit(train_df)
    transformed_train = preprocessing_model.transform(train_df)

    print_top_feature_importance(
        fitted_pipeline_model=pipeline_model,
        transformed_train_df=transformed_train,
        model_stage_index=len(stages) - 1,
        model_name="Random Forest Classifier",
        top_n=15,
    )

    return pipeline_model, predictions, result


def train_gbt_classifier(train_df, test_df, numeric_cols, categorical_cols):
    """
    Train Gradient Boosting Classifier.

    Role in project:
      GBT is the strongest model in this classification comparison.
      It trains trees sequentially, where each new tree attempts to correct errors
      made by previous trees.

    Note:
      GBT can train slower than Logistic Regression and Random Forest.
      If runtime is too long, reduce maxIter from 50 to 30.
    """
    print_header("TRAINING MODEL 3 - GRADIENT BOOSTING CLASSIFIER")

    preprocessing_stages, final_features_col = build_preprocessing_stages(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        use_scaler=False,
    )

    gbt = GBTClassifier(
        featuresCol=final_features_col,
        labelCol="label",
        predictionCol="prediction",
        rawPredictionCol="rawPrediction",
        probabilityCol="probability",
        maxIter=50,
        maxDepth=5,
        stepSize=0.05,
        seed=42,
    )

    stages = preprocessing_stages + [gbt]

    pipeline = Pipeline(stages=stages)
    pipeline_model = pipeline.fit(train_df)

    predictions = pipeline_model.transform(test_df)

    result = evaluate_classification_model(
        predictions=predictions,
        model_name="Gradient Boosting Classifier",
    )

    preprocessing_model = Pipeline(stages=preprocessing_stages).fit(train_df)
    transformed_train = preprocessing_model.transform(train_df)

    print_top_feature_importance(
        fitted_pipeline_model=pipeline_model,
        transformed_train_df=transformed_train,
        model_stage_index=len(stages) - 1,
        model_name="Gradient Boosting Classifier",
        top_n=15,
    )

    return pipeline_model, predictions, result


def save_best_model(model_results: List[Dict[str, float]], model_objects: Dict[str, object]) -> None:
    """
    Save the best model to HDFS based on ROC-AUC.

    Why ROC-AUC:
      This is a binary classification task.
      ROC-AUC evaluates how well the model separates Good Loan and Bad Loan
      across classification thresholds.
    """
    print_header("SAVE BEST MODEL")

    best_result = sorted(
        model_results,
        key=lambda row: row["roc_auc"],
        reverse=True,
    )[0]

    best_model_name = best_result["model"]
    best_model = model_objects[best_model_name]

    safe_model_name = (
        best_model_name
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    output_path = f"{HDFS_MODEL_OUTPUT_BASE_PATH}/{safe_model_name}"

    best_model.write().overwrite().save(output_path)

    print(f"Best model selected by ROC-AUC: {best_model_name}")
    print(f"Best ROC-AUC: {best_result['roc_auc']:.4f}")
    print(f"Model saved to: {output_path}")


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    spark = (
        SparkSession.builder
        .appName("Prosper Loan Good Bad Classification")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    # ========================================================
    # STEP 1 - READ CLASSIFICATION DATASET
    # ========================================================
    #
    # This dataset was produced by the preprocessing file.
    # It already contains label and does not contain LoanStatus.
    # ========================================================
    print_header("READ CLASSIFICATION DATASET FROM HDFS")
    df = spark.read.parquet(HDFS_CLASSIFICATION_INPUT_PATH)

    # Cache the dataset because it is reused by 3 model training pipelines.
    # This can also be mentioned in the performance optimization section.
    df = df.cache()

    print_dataset_overview(df, label_col="label")

    # ========================================================
    # STEP 2 - IDENTIFY FEATURE COLUMNS
    # ========================================================
    #
    # User decision:
    #   Keep BorrowerAPR as one of the classification features.
    #
    # Therefore:
    #   feature columns = all columns except label.
    # ========================================================
    feature_cols, numeric_cols, categorical_cols = get_feature_columns(
        df=df,
        label_col="label",
    )

    # ========================================================
    # STEP 3 - TRAIN / TEST SPLIT
    # ========================================================
    #
    # 80% training data
    # 20% testing data
    #
    # seed=42 makes the split reproducible.
    # ========================================================
    print_header("TRAIN / TEST SPLIT")

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    train_df = train_df.cache()
    test_df = test_df.cache()

    print(f"Training rows: {train_df.count()}")
    print(f"Testing rows : {test_df.count()}")

    print("\nTraining label distribution:")
    train_df.groupBy("label").agg(count("*").alias("count")).orderBy("label").show()

    print("Testing label distribution:")
    test_df.groupBy("label").agg(count("*").alias("count")).orderBy("label").show()

    # ========================================================
    # STEP 4 - TRAIN 3 CLASSIFICATION MODELS
    # ========================================================

    model_results = []
    model_objects = {}

    lr_model, lr_predictions, lr_result = train_logistic_regression(
        train_df=train_df,
        test_df=test_df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )
    model_results.append(lr_result)
    model_objects["Logistic Regression"] = lr_model

    rf_model, rf_predictions, rf_result = train_random_forest(
        train_df=train_df,
        test_df=test_df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )
    model_results.append(rf_result)
    model_objects["Random Forest Classifier"] = rf_model

    gbt_model, gbt_predictions, gbt_result = train_gbt_classifier(
        train_df=train_df,
        test_df=test_df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )
    model_results.append(gbt_result)
    model_objects["Gradient Boosting Classifier"] = gbt_model

    # ========================================================
    # STEP 5 - MODEL COMPARISON
    # ========================================================

    print_header("CLASSIFICATION MODEL COMPARISON")

    sorted_results = sorted(
        model_results,
        key=lambda row: row["roc_auc"],
        reverse=True,
    )

    print(
        f"{'Model':35} "
        f"{'Accuracy':>10} "
        f"{'F1':>10} "
        f"{'Precision':>12} "
        f"{'Recall':>10} "
        f"{'ROC-AUC':>10} "
        f"{'PR-AUC':>10} "
        f"{'Bad Recall':>12}"
    )
    print("-" * 115)

    for row in sorted_results:
        print(
            f"{row['model']:35} "
            f"{row['accuracy']:10.4f} "
            f"{row['f1']:10.4f} "
            f"{row['weighted_precision']:12.4f} "
            f"{row['weighted_recall']:10.4f} "
            f"{row['roc_auc']:10.4f} "
            f"{row['pr_auc']:10.4f} "
            f"{row['bad_loan_recall']:12.4f}"
        )

    save_results_to_csv(model_results)

    # ========================================================
    # STEP 6 - SAVE BEST MODEL
    # ========================================================
    #
    # The best model is selected using ROC-AUC.
    # This is useful because this is a binary classification problem.
    # ========================================================
    save_best_model(model_results, model_objects)

    # ========================================================
    # STEP 7 - FINAL CONCLUSION
    # ========================================================

    best_model = sorted_results[0]

    print_header("FINAL CONCLUSION")
    print(f"Best classification model by ROC-AUC: {best_model['model']}")
    print(f"ROC-AUC: {best_model['roc_auc']:.4f}")
    print(f"F1-score: {best_model['f1']:.4f}")
    print(f"Bad Loan Recall: {best_model['bad_loan_recall']:.4f}")
    print(
        "\nInterpretation guide:\n"
        "- Accuracy shows overall correct predictions.\n"
        "- F1-score balances precision and recall.\n"
        "- ROC-AUC shows the model's ability to separate Good and Bad loans.\n"
        "- Bad Loan Recall is important because missing a bad loan can create credit risk.\n"
        "- BorrowerAPR is included as a feature, so the classification model uses both "
        "borrower/loan characteristics and pricing information."
    )

    # Release cached dataframes.
    df.unpersist()
    train_df.unpersist()
    test_df.unpersist()

    spark.stop()


if __name__ == "__main__":
    main()