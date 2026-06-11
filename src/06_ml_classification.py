from pyspark.sql.functions import col, count
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
import csv
import os


# ============================================================
# FILE 06 OVERVIEW
# ============================================================
#
# Input:
# Dataset created by File 05: prosper_loan_preprocessed_classification
#
# Main tasks:
# 1. Read the prepared classification dataset from HDFS.
# 2. Build Spark ML feature pipelines for numeric and categorical variables.
# 3. Split the data into training and testing sets.
# 4. Train three classification models:
#    Logistic Regression, Random Forest Classifier, and Gradient Boosting Classifier.
# 5. Evaluate the models using Accuracy, F1-score, Precision, Recall, ROC-AUC,
#    PR-AUC, and confusion matrix.
# 6. Save the model comparison table locally.
# 7. Save the best model back to HDFS.
#
# Outputs:
# classification_model_results.csv
# best classification model saved in HDFS
# ============================================================


HDFS_CLASSIFICATION_INPUT_PATH = (
    "hdfs://localhost:9000/bigdata/prosper_loan/processed/"
    "prosper_loan_preprocessed_classification"
)

HDFS_CLASSIFICATION_MODEL_OUTPUT_PATH = (
    "hdfs://localhost:9000/bigdata/prosper_loan/models/"
    "prosper_loan_best_classification_model"
)

LOCAL_TABLE_OUTPUT_DIR = os.path.join("outputs", "tables")
CLASSIFICATION_RESULTS_PATH = os.path.join(
    LOCAL_TABLE_OUTPUT_DIR,
    "classification_model_results.csv",
)


CATEGORICAL_COLUMNS = [
    "IncomeRange",
    "EmploymentStatus",
    "IsBorrowerHomeowner",
]


def print_header(title):
    print(f"\n========== {title} ==========")


def get_existing_cols(df, cols):
    return [column_name for column_name in cols if column_name in df.columns]


def get_feature_columns(df):
    feature_cols = [column_name for column_name in df.columns if column_name != "label"]

    categorical_cols = get_existing_cols(df, CATEGORICAL_COLUMNS)
    numeric_cols = []

    for field in df.schema.fields:
        column_name = field.name

        if column_name == "label" or column_name in categorical_cols:
            continue

        if isinstance(field.dataType, (StringType, BooleanType)):
            categorical_cols.append(column_name)
        else:
            numeric_cols.append(column_name)

    return feature_cols, numeric_cols, categorical_cols


def validate_classification_input(df):
    print_header("CLASSIFICATION MODELING INPUT")

    if "label" not in df.columns:
        raise ValueError("label column is required for classification modeling.")

    print(f"Classification dataset loaded from: {HDFS_CLASSIFICATION_INPUT_PATH}")
    print(f"Rows: {df.count()}")
    print(f"Columns: {len(df.columns)}")

    if "BorrowerAPR" in df.columns:
        print("BorrowerAPR is kept as an input feature for classification.")

    label_count = df.select("label").distinct().count()
    print(f"Number of label classes: {label_count}")


def print_feature_summary(numeric_cols, categorical_cols):
    print_header("FEATURE PIPELINE SUMMARY")
    print(f"Numeric features: {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")

    if categorical_cols:
        print("Categorical columns encoded by StringIndexer and OneHotEncoder:")
        for column_name in categorical_cols:
            print(f"- {column_name}")


def build_feature_pipeline(numeric_cols, categorical_cols, use_scaler):
    stages = []
    indexed_cols = []
    encoded_cols = []

    for column_name in categorical_cols:
        index_col = f"{column_name}_index"
        encoded_col = f"{column_name}_encoded"

        indexer = StringIndexer(
            inputCol=column_name,
            outputCol=index_col,
            handleInvalid="keep",
        )

        stages.append(indexer)
        indexed_cols.append(index_col)
        encoded_cols.append(encoded_col)

    if categorical_cols:
        encoder = OneHotEncoder(
            inputCols=indexed_cols,
            outputCols=encoded_cols,
            handleInvalid="keep",
        )
        stages.append(encoder)

    assembler = VectorAssembler(
        inputCols=numeric_cols + encoded_cols,
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
        features_col = "scaled_features"
    else:
        features_col = "features"

    return stages, features_col


def compute_confusion_matrix(predictions):
    rows = (
        predictions
        .select(
            col("label").cast("int").alias("actual"),
            col("prediction").cast("int").alias("predicted"),
        )
        .groupBy("actual", "predicted")
        .agg(count("*").alias("count"))
        .collect()
    )

    result = {
        "true_negative": 0,
        "false_positive": 0,
        "false_negative": 0,
        "true_positive": 0,
    }

    for row in rows:
        actual = row["actual"]
        predicted = row["predicted"]
        value = int(row["count"])

        if actual == 0 and predicted == 0:
            result["true_negative"] = value
        elif actual == 0 and predicted == 1:
            result["false_positive"] = value
        elif actual == 1 and predicted == 0:
            result["false_negative"] = value
        elif actual == 1 and predicted == 1:
            result["true_positive"] = value

    return result


def evaluate_model(predictions, model_name):
    evaluator_accuracy = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy",
    )
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1",
    )
    evaluator_precision = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedPrecision",
    )
    evaluator_recall = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedRecall",
    )
    evaluator_auc = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )
    evaluator_pr = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR",
    )

    accuracy = evaluator_accuracy.evaluate(predictions)
    f1_score = evaluator_f1.evaluate(predictions)
    precision = evaluator_precision.evaluate(predictions)
    recall = evaluator_recall.evaluate(predictions)
    roc_auc = evaluator_auc.evaluate(predictions)
    pr_auc = evaluator_pr.evaluate(predictions)

    matrix = compute_confusion_matrix(predictions)
    tn = matrix["true_negative"]
    fp = matrix["false_positive"]
    fn = matrix["false_negative"]
    tp = matrix["true_positive"]

    bad_loan_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    bad_loan_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    print_header(f"EVALUATION - {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1_score:.4f}")
    print(f"Weighted precision: {precision:.4f}")
    print(f"Weighted recall: {recall:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")

    print("Confusion matrix:")
    print("Actual / Predicted | Good Loan (0) | Bad Loan (1)")
    print(f"Good Loan (0)      | {tn:<13} | {fp:<12}")
    print(f"Bad Loan (1)       | {fn:<13} | {tp:<12}")

    print(f"Bad Loan precision: {bad_loan_precision:.4f}")
    print(f"Bad Loan recall: {bad_loan_recall:.4f}")

    return {
        "model": model_name,
        "accuracy": accuracy,
        "f1_score": f1_score,
        "weighted_precision": precision,
        "weighted_recall": recall,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "bad_loan_precision": bad_loan_precision,
        "bad_loan_recall": bad_loan_recall,
    }


def train_logistic_regression(train_df, test_df, numeric_cols, categorical_cols):
    print_header("TRAIN LOGISTIC REGRESSION")

    stages, features_col = build_feature_pipeline(
        numeric_cols,
        categorical_cols,
        use_scaler=True,
    )

    model = LogisticRegression(
        featuresCol=features_col,
        labelCol="label",
        maxIter=100,
        regParam=0.01,
        elasticNetParam=0.0,
    )

    pipeline = Pipeline(stages=stages + [model])
    fitted_pipeline = pipeline.fit(train_df)
    predictions = fitted_pipeline.transform(test_df)

    result = evaluate_model(predictions, "Logistic Regression")
    return fitted_pipeline, result


def train_random_forest(train_df, test_df, numeric_cols, categorical_cols):
    print_header("TRAIN RANDOM FOREST CLASSIFIER")

    stages, features_col = build_feature_pipeline(
        numeric_cols,
        categorical_cols,
        use_scaler=False,
    )

    model = RandomForestClassifier(
        featuresCol=features_col,
        labelCol="label",
        numTrees=100,
        maxDepth=8,
        seed=42,
    )

    pipeline = Pipeline(stages=stages + [model])
    fitted_pipeline = pipeline.fit(train_df)
    predictions = fitted_pipeline.transform(test_df)

    result = evaluate_model(predictions, "Random Forest Classifier")
    return fitted_pipeline, result


def train_gradient_boosting(train_df, test_df, numeric_cols, categorical_cols):
    print_header("TRAIN GRADIENT BOOSTING CLASSIFIER")

    stages, features_col = build_feature_pipeline(
        numeric_cols,
        categorical_cols,
        use_scaler=False,
    )

    model = GBTClassifier(
        featuresCol=features_col,
        labelCol="label",
        maxIter=50,
        maxDepth=5,
        stepSize=0.05,
        seed=42,
    )

    pipeline = Pipeline(stages=stages + [model])
    fitted_pipeline = pipeline.fit(train_df)
    predictions = fitted_pipeline.transform(test_df)

    result = evaluate_model(predictions, "Gradient Boosting Classifier")
    return fitted_pipeline, result


def save_results_locally(results):
    os.makedirs(LOCAL_TABLE_OUTPUT_DIR, exist_ok=True)

    fieldnames = [
        "model",
        "accuracy",
        "f1_score",
        "weighted_precision",
        "weighted_recall",
        "roc_auc",
        "pr_auc",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
        "bad_loan_precision",
        "bad_loan_recall",
    ]

    with open(CLASSIFICATION_RESULTS_PATH, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            row = dict(result)
            for column_name in [
                "accuracy",
                "f1_score",
                "weighted_precision",
                "weighted_recall",
                "roc_auc",
                "pr_auc",
                "bad_loan_precision",
                "bad_loan_recall",
            ]:
                row[column_name] = round(float(row[column_name]), 6)

            writer.writerow(row)

    print(f"Classification model results saved to: {CLASSIFICATION_RESULTS_PATH}")


def print_model_comparison(results):
    print_header("CLASSIFICATION MODEL COMPARISON")

    sorted_results = sorted(
        results,
        key=lambda row: row["roc_auc"],
        reverse=True,
    )

    print(
        f"{'Model':35} "
        f"{'Accuracy':>10} "
        f"{'F1':>10} "
        f"{'ROC-AUC':>10} "
        f"{'PR-AUC':>10} "
        f"{'Bad Recall':>12}"
    )
    print("-" * 95)

    for result in sorted_results:
        print(
            f"{result['model']:35} "
            f"{result['accuracy']:10.4f} "
            f"{result['f1_score']:10.4f} "
            f"{result['roc_auc']:10.4f} "
            f"{result['pr_auc']:10.4f} "
            f"{result['bad_loan_recall']:12.4f}"
        )

    return sorted_results[0]


def save_best_model(best_result, model_objects):
    print_header("SAVE BEST CLASSIFICATION MODEL")

    best_model_name = best_result["model"]
    best_model = model_objects[best_model_name]

    best_model.write().overwrite().save(HDFS_CLASSIFICATION_MODEL_OUTPUT_PATH)

    print(f"Best model: {best_model_name}")
    print(f"Best ROC-AUC: {best_result['roc_auc']:.4f}")
    print(f"Model output path: {HDFS_CLASSIFICATION_MODEL_OUTPUT_PATH}")


def main():
    spark = (
        SparkSession.builder
        .appName("Prosper Loan Classification Modeling")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # ============================================================
    # STEP 1: READ PREPROCESSED CLASSIFICATION DATASET
    # ============================================================
    print_header("READ PREPROCESSED CLASSIFICATION DATASET")
    df = spark.read.parquet(HDFS_CLASSIFICATION_INPUT_PATH).cache()
    validate_classification_input(df)

    # ============================================================
    # STEP 2: PREPARE FEATURE COLUMNS
    # ============================================================
    print_header("PREPARE FEATURE COLUMNS")
    feature_cols, numeric_cols, categorical_cols = get_feature_columns(df)
    print_feature_summary(numeric_cols, categorical_cols)

    # ============================================================
    # STEP 3: SPLIT TRAINING AND TESTING DATA
    #
    # The data is split into 80% training and 20% testing sets.
    # ============================================================
    print_header("TRAIN TEST SPLIT")
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    train_df = train_df.cache()
    test_df = test_df.cache()

    print(f"Training rows: {train_df.count()}")
    print(f"Testing rows: {test_df.count()}")

    # ============================================================
    # STEP 4: TRAIN CLASSIFICATION MODELS
    # ============================================================
    model_results = []
    model_objects = {}

    lr_model, lr_result = train_logistic_regression(
        train_df,
        test_df,
        numeric_cols,
        categorical_cols,
    )
    model_results.append(lr_result)
    model_objects["Logistic Regression"] = lr_model

    rf_model, rf_result = train_random_forest(
        train_df,
        test_df,
        numeric_cols,
        categorical_cols,
    )
    model_results.append(rf_result)
    model_objects["Random Forest Classifier"] = rf_model

    gbt_model, gbt_result = train_gradient_boosting(
        train_df,
        test_df,
        numeric_cols,
        categorical_cols,
    )
    model_results.append(gbt_result)
    model_objects["Gradient Boosting Classifier"] = gbt_model

    # ============================================================
    # STEP 5: COMPARE AND SAVE RESULTS
    # The best model is selected by ROC-AUC because this is a binary classification
    # task and ROC-AUC measures class separation quality.
    # ============================================================
    best_result = print_model_comparison(model_results)
    save_results_locally(model_results)
    save_best_model(best_result, model_objects)

    print_header("CLASSIFICATION MODELING CONCLUSION")
    print(f"Best classification model: {best_result['model']}")
    print(f"ROC-AUC: {best_result['roc_auc']:.4f}")
    print(f"F1-score: {best_result['f1_score']:.4f}")
    print(f"Bad Loan recall: {best_result['bad_loan_recall']:.4f}")
    print("Conclusion: Spark ML classification modeling completed successfully.")

    df.unpersist()
    train_df.unpersist()
    test_df.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()