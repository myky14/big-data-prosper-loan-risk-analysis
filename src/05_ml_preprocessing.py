from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    when,
    count,
    isnan,
    sum as spark_sum,
    year,
    month,
    datediff,
    to_date,
    lit,
    percentile_approx,
    avg,
    min as spark_min,
    max as spark_max,
    stddev,
)
from pyspark.sql.types import DoubleType, IntegerType
import csv
import math
import os


# ============================================================
# FILE 05 OVERVIEW
# ============================================================
#
# Input:
# The script reads the EDA-reduced Prosper Loan dataset created by File 04:
# prosper_loan_eda_reduced
#
# Main tasks:
# 1. Read the cleaned feature set from HDFS.
# 2. Check and remove duplicate loan records.
# 3. Validate that important columns are still available.
# 4. Remove features that are too sparse to be reliable.
# 5. Review and handle missing values.
# 6. Create useful date-based features from raw date columns.
# 7. Standardize numeric and categorical data types.
# 8. Create a classification dataset for Good Loan / Bad Loan prediction.
# 9. Create a regression dataset for BorrowerAPR prediction.
# 10. Save the preprocessed outputs back to HDFS.
#
# Why this file exists:
# File 04 focuses on EDA-based feature reduction. This file prepares the data
# for later Feature Selection and Machine Learning Modeling by cleaning values,
# creating targets, and saving task-specific datasets.
#
# Outputs:
# prosper_loan_preprocessed_classification
# prosper_loan_preprocessed_regression
#
# These outputs are intended for File 06 Feature Selection and later Spark ML
# modeling scripts.
# ============================================================


HDFS_INPUT_PATH = "hdfs://localhost:9000/bigdata/prosper_loan/processed/prosper_loan_eda_reduced"
HDFS_CLASSIFICATION_OUTPUT_PATH = (
    "hdfs://localhost:9000/bigdata/prosper_loan/processed/"
    "prosper_loan_preprocessed_classification"
)
HDFS_REGRESSION_OUTPUT_PATH = (
    "hdfs://localhost:9000/bigdata/prosper_loan/processed/"
    "prosper_loan_preprocessed_regression"
)

OUTPUT_ROOT = os.path.join("outputs", "05_ml_preprocessing")
TABLES_DIR = os.path.join(OUTPUT_ROOT, "tables")
MISSING_VALUE_SUMMARY_PATH = os.path.join(
    TABLES_DIR,
    "missing_value_summary.csv",
)


NUMERIC_COLUMNS = [
    "Term",
    "BorrowerAPR",
    "EstimatedLoss",
    "ProsperScore",
    "CreditScoreRangeLower",
    "InquiriesLast6Months",
    "CurrentDelinquencies",
    "DelinquenciesLast7Years",
    "BankcardUtilization",
    "TotalTrades",
    "DebtToIncomeRatio",
    "TotalProsperLoans",
    "ProsperPaymentsLessThanOneMonthLate",
    "ProsperPaymentsOneMonthPlusLate",
    "LoanOriginalAmount",
    "Investors",
    "listing_year",
    "listing_month",
    "loan_origination_year",
    "loan_origination_month",
    "credit_history_days",
    "credit_pull_to_origination_days",
]

CATEGORICAL_COLUMNS = [
    "IncomeRange",
    "EmploymentStatus",
    "IsBorrowerHomeowner",
]

RAW_DATE_COLUMNS = [
    "ListingCreationDate",
    "LoanOriginationDate",
    "DateCreditPulled",
    "FirstRecordedCreditLine",
]

HIGH_MISSING_VALUE_COLUMNS = [
    "TotalProsperLoans",
    "ProsperPaymentsLessThanOneMonthLate",
    "ProsperPaymentsOneMonthPlusLate",
]


def print_header(title):
    print(f"\n========== {title} ==========")


def print_status(title, df):
    print_header(title)
    print(f"Rows: {df.count()}")
    print(f"Columns: {len(df.columns)}")
    print("Column list:")
    for index, column_name in enumerate(df.columns, start=1):
        print(f"{index:02d}. {column_name}")


def get_existing_cols(df, cols):
    return [column_name for column_name in cols if column_name in df.columns]


def safe_drop(df, cols):
    existing_cols = get_existing_cols(df, cols)
    if not existing_cols:
        return df
    return df.drop(*existing_cols)


def remove_duplicate_rows(df):
    print_header("DUPLICATE CHECKING AND REMOVAL")

    # Duplicate removal ensures each loan record is counted only once and prevents
    # repeated rows from biasing downstream model training.
    # Count rows before removing duplicates so the cleaning effect is measurable.
    before_duplicate_count = df.count()

    # Remove duplicate records across all columns. This keeps one copy of each
    # identical loan row and prevents the same information from being learned twice.
    df = df.dropDuplicates()

    # Count rows after cleaning to show how many duplicated records were removed.
    after_duplicate_count = df.count()
    duplicate_removed_count = before_duplicate_count - after_duplicate_count

    print(f"Row count before duplicate removal: {before_duplicate_count}")
    print(f"Row count after duplicate removal: {after_duplicate_count}")
    print(f"Duplicate rows removed: {duplicate_removed_count}")

    return df


def drop_high_missing_value_columns(df):
    print_header("HIGH-MISSING-VALUE FEATURE REMOVAL")
    print(
        "Columns with missing value percentages above 80% are removed because they "
        "are unreliable and may introduce noise into the machine learning models."
    )

    columns_before = len(df.columns)

    # File 04 may already have removed these columns. This check prevents errors
    # by dropping only the features that still exist in the current DataFrame.
    existing_drop_cols = get_existing_cols(df, HIGH_MISSING_VALUE_COLUMNS)

    if existing_drop_cols:
        print("High-missing-value columns removed:")
        for column_name in existing_drop_cols:
            print(f"- {column_name}")
        df = df.drop(*existing_drop_cols)
    else:
        print(
            "No configured high-missing-value columns were found. They may have "
            "already been removed by File 04 EDA-based feature reduction."
        )

    print(f"Columns before removal: {columns_before}")
    print(f"Columns removed: {len(existing_drop_cols)}")
    print(f"Columns after removal: {len(df.columns)}")
    print("Remaining columns after high-missing-value cleanup:")
    for index, column_name in enumerate(df.columns, start=1):
        print(f"{index:02d}. {column_name}")

    return df


def validate_dataset_schema(df, stage_description):
    print_header("BASIC SCHEMA AND DATA VALIDATION")

    # Row and column counts help confirm that preprocessing did not accidentally
    # remove all records or remove too many features.
    print(f"Total row count {stage_description}: {df.count()}")
    print(f"Total column count {stage_description}: {len(df.columns)}")

    # LoanStatus is needed for classification. BorrowerAPR is needed for regression.
    # Checking them early makes pipeline problems easier to diagnose.
    important_columns = ["LoanStatus", "BorrowerAPR"]
    print("Important columns before target construction:")
    for column_name in important_columns:
        status = "present" if column_name in df.columns else "missing"
        print(f"- {column_name}: {status}")


def calculate_missing_summary(df):
    # Missing-value analysis shows which features may need imputation or removal.
    # This script counts NULL values only. NaN handling for numeric values happens
    # later during numeric imputation.
    total_rows = df.count()
    agg_exprs = [
        spark_sum(when(col(column_name).isNull(), 1).otherwise(0)).alias(column_name)
        for column_name in df.columns
    ]

    missing_counts = df.select(agg_exprs).collect()[0].asDict()

    summary = []
    for column_name, missing_count in missing_counts.items():
        missing_count = int(missing_count or 0)
        missing_percent = 0.0
        if total_rows > 0:
            missing_percent = round((missing_count / total_rows) * 100, 2)
        summary.append(
            {
                "column": column_name,
                "missing_count": missing_count,
                "missing_percent": missing_percent,
            }
        )

    return sorted(summary, key=lambda row: row["missing_percent"], reverse=True)


def save_missing_summary_locally(missing_summary):
    os.makedirs(TABLES_DIR, exist_ok=True)
    with open(MISSING_VALUE_SUMMARY_PATH, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["column", "missing_count", "missing_percent"],
        )
        writer.writeheader()
        writer.writerows(missing_summary)

    print(f"Missing value summary saved to: {MISSING_VALUE_SUMMARY_PATH}")


def create_date_features(df):
    created_features = []

    if "ListingCreationDate" in df.columns:
        # Listing year and month capture broad timing patterns without keeping
        # the full raw date string.
        df = df.withColumn(
            "listing_year",
            year(to_date(col("ListingCreationDate"))),
        ).withColumn(
            "listing_month",
            month(to_date(col("ListingCreationDate"))),
        )
        created_features.extend(["listing_year", "listing_month"])

    if "LoanOriginationDate" in df.columns:
        # Loan origination year and month can help later models learn changes in
        # lending behavior over time.
        df = df.withColumn(
            "loan_origination_year",
            year(to_date(col("LoanOriginationDate"))),
        ).withColumn(
            "loan_origination_month",
            month(to_date(col("LoanOriginationDate"))),
        )
        created_features.extend(["loan_origination_year", "loan_origination_month"])

    if "LoanOriginationDate" in df.columns and "FirstRecordedCreditLine" in df.columns:
        # Credit history length is a borrower risk signal. A longer credit history
        # often gives lenders more evidence about repayment behavior.
        df = df.withColumn(
            "credit_history_days",
            datediff(
                to_date(col("LoanOriginationDate")),
                to_date(col("FirstRecordedCreditLine")),
            ),
        )
        created_features.append("credit_history_days")

    if "LoanOriginationDate" in df.columns and "DateCreditPulled" in df.columns:
        # This feature measures the time between the credit pull and loan origination.
        # It can reveal whether the credit information was recent.
        df = df.withColumn(
            "credit_pull_to_origination_days",
            datediff(
                to_date(col("LoanOriginationDate")),
                to_date(col("DateCreditPulled")),
            ),
        )
        created_features.append("credit_pull_to_origination_days")

    # After extracting useful date features, raw date columns are removed so later
    # ML steps do not need to handle date strings directly.
    df = safe_drop(df, RAW_DATE_COLUMNS)

    print("Date features created from raw temporal columns.")
    if created_features:
        print("Created date features:")
        for feature_name in created_features:
            print(f"- {feature_name}")
    else:
        print("No expected raw temporal columns were found for date feature engineering.")

    return df


def standardize_data_types(df):
    existing_numeric_cols = get_existing_cols(df, NUMERIC_COLUMNS)
    existing_categorical_cols = get_existing_cols(df, CATEGORICAL_COLUMNS)

    for column_name in existing_numeric_cols:
        # Numeric columns are cast to DoubleType because Spark ML algorithms expect
        # numerical features to have consistent numeric types.
        df = df.withColumn(column_name, col(column_name).cast(DoubleType()))

    for column_name in existing_categorical_cols:
        if column_name == "IsBorrowerHomeowner":
            # Convert homeowner status into clear string categories so it can be
            # encoded later in the modeling stage.
            df = df.withColumn(
                column_name,
                when(col(column_name).isNull(), lit("Unknown"))
                .when(col(column_name).cast("boolean") == lit(True), lit("true"))
                .when(col(column_name).cast("boolean") == lit(False), lit("false"))
                .otherwise(col(column_name).cast("string")),
            )
        else:
            # Keep categorical variables as strings. Encoding is intentionally left
            # for the later ML modeling stage.
            df = df.withColumn(column_name, col(column_name).cast("string"))

    print("Numeric columns cast to DoubleType:")
    if existing_numeric_cols:
        for column_name in existing_numeric_cols:
            print(f"- {column_name}")
    else:
        print("- None found")

    print("Categorical columns retained for later encoding:")
    if existing_categorical_cols:
        for column_name in existing_categorical_cols:
            print(f"- {column_name}")
    else:
        print("- None found")

    if "IsBorrowerHomeowner" in existing_categorical_cols:
        print('IsBorrowerHomeowner standardized to string values: "true", "false", "Unknown".')

    return df


def get_numeric_fill_value(df, column_name):
    # Median imputation is used because it is less sensitive to extreme values
    # than the mean. This fills missing numeric values without changing the row count.
    valid_rows = df.filter(
        col(column_name).isNotNull() & (~isnan(col(column_name).cast("double")))
    )
    median_row = valid_rows.select(
        percentile_approx(col(column_name).cast("double"), 0.5).alias("median")
    ).collect()[0]
    fill_value = median_row["median"]

    if fill_value is None:
        return 0.0
    if isinstance(fill_value, float) and math.isnan(fill_value):
        return 0.0
    return float(fill_value)


def fill_numeric_with_median(df, numeric_cols):
    existing_numeric_cols = get_existing_cols(df, numeric_cols)

    if not existing_numeric_cols:
        print("No expected numeric columns found for median imputation.")
        return df

    for column_name in existing_numeric_cols:
        fill_value = get_numeric_fill_value(df, column_name)

        # Replace NULL and NaN values with the column median so Spark ML receives
        # complete numeric inputs later.
        df = df.withColumn(
            column_name,
            when(
                col(column_name).isNull() | isnan(col(column_name).cast("double")),
                lit(fill_value),
            )
            .otherwise(col(column_name))
            .cast(DoubleType()),
        )
        print(f"{column_name}: missing numeric values filled with {fill_value}")

    return df


def fill_categorical_with_unknown(df):
    existing_categorical_cols = get_existing_cols(df, CATEGORICAL_COLUMNS)

    if not existing_categorical_cols:
        print("No expected categorical columns found for Unknown imputation.")
        return df

    for column_name in existing_categorical_cols:
        # Unknown is used as an explicit category. This preserves rows while making
        # missing categorical values visible to later encoding steps.
        df = df.withColumn(
            column_name,
            when(col(column_name).isNull(), lit("Unknown"))
            .otherwise(col(column_name).cast("string")),
        )
        print(f"{column_name}: missing categorical values filled with Unknown")

    return df


def print_label_distribution(df):
    total_rows = df.count()
    label_rows = (
        df.groupBy("label")
        .agg(count("*").alias("count"))
        .orderBy("label")
        .collect()
    )

    print("label | count | percentage")
    print("--------------------------")
    for row in label_rows:
        percentage = 0.0
        if total_rows > 0:
            percentage = (row["count"] / total_rows) * 100
        label_name = "Good Loan" if row["label"] == 0 else "Bad Loan"
        print(f"{label_name} / label {row['label']} | {row['count']} | {percentage:.2f}%")


def build_classification_dataset(df_shared):
    if "LoanStatus" not in df_shared.columns:
        raise ValueError("LoanStatus is required to construct the classification label.")

    rows_before = df_shared.count()
    finalized_statuses = ["Completed", "Chargedoff", "Defaulted"]

    # Only finalized loans are used for classification because their final outcome
    # is known. Current loans are excluded because their repayment result could
    # still change in the future.
    finalized_count = df_shared.filter(col("LoanStatus").isin(finalized_statuses)).count()
    excluded_count = rows_before - finalized_count

    print("Classification status validation:")
    print(f"Rows with finalized statuses: {finalized_count}")
    print(f"Rows excluded because they are not finalized: {excluded_count}")

    df_classification = (
        df_shared.withColumn(
            "label",
            # Chargedoff and Defaulted loans are bad loans because they represent
            # credit risk events where the borrower did not successfully repay.
            when(col("LoanStatus").isin("Chargedoff", "Defaulted"), lit(1))
            # Completed loans are good loans because borrowers fully repaid their
            # obligations.
            .when(col("LoanStatus") == "Completed", lit(0))
            # Current and other non-finalized loans are not assigned a label because
            # their final outcome is still unknown.
            .otherwise(lit(None))
            .cast(IntegerType()),
        )
        .filter(col("label").isNotNull())
    )

    # BorrowerAPR may be removed in the classification ML file if treated as pricing leakage.
    df_classification = safe_drop(df_classification, ["LoanStatus"])

    print("Classification target mapping:")
    print("- Completed -> Good Loan -> label 0")
    print("- Chargedoff, Defaulted -> Bad Loan -> label 1")
    print("- Current and other non-finalized statuses are excluded.")
    print(f"Rows before finalized-status filtering: {rows_before}")
    print(f"Rows after finalized-status filtering: {df_classification.count()}")
    print_label_distribution(df_classification)
    print("Conclusion: Classification dataset uses only final loan outcomes to reduce label noise.")

    return df_classification


def print_borrower_apr_summary(df_regression):
    summary_row = (
        df_regression.select(
            count(col("BorrowerAPR")).alias("count"),
            avg(col("BorrowerAPR")).alias("mean"),
            spark_min(col("BorrowerAPR")).alias("min"),
            spark_max(col("BorrowerAPR")).alias("max"),
            stddev(col("BorrowerAPR")).alias("standard_deviation"),
        )
        .collect()[0]
    )

    print("BorrowerAPR summary:")
    print(f"count: {summary_row['count']}")
    print(f"mean: {summary_row['mean']}")
    print(f"min: {summary_row['min']}")
    print(f"max: {summary_row['max']}")
    print(f"standard deviation: {summary_row['standard_deviation']}")


def build_regression_dataset(df_shared):
    if "BorrowerAPR" not in df_shared.columns:
        raise ValueError("BorrowerAPR is required as the regression target.")

    rows_before = df_shared.count()

    # BorrowerAPR is the target variable for the regression task. Rows without APR
    # information cannot be used for training because the expected answer is missing.
    missing_borrower_apr_count = df_shared.filter(
        col("BorrowerAPR").isNull() | isnan(col("BorrowerAPR"))
    ).count()

    # Keep only rows where BorrowerAPR is available and valid.
    df_regression = df_shared.filter(
        col("BorrowerAPR").isNotNull() & (~isnan(col("BorrowerAPR")))
    )
    df_regression = safe_drop(df_regression, ["LoanStatus"])

    print(f"Rows with missing or NaN BorrowerAPR before filtering: {missing_borrower_apr_count}")
    print(f"Rows before BorrowerAPR filtering: {rows_before}")
    print(f"Rows after BorrowerAPR filtering: {df_regression.count()}")
    print_borrower_apr_summary(df_regression)
    print("Conclusion: Regression dataset keeps BorrowerAPR as the prediction target.")

    return df_regression


def main():
    spark = (
        SparkSession.builder
        .appName("Prosper Loan ML Preprocessing")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # ============================================================
    # STEP 1: READ EDA-REDUCED DATASET FROM HDFS
    # ============================================================
    #
    # Purpose:
    # Read the dataset produced by File 04, which already completed EDA,
    # distribution analysis, correlation analysis, and EDA-based feature reduction.
    #
    # Why:
    # File 05 should not start from the earlier File 03 dataset because File 04
    # may have removed additional weak, redundant, or high-missing-value features.
    #
    # Output:
    # A Spark DataFrame containing the EDA-reduced loan records.
    #
    # Next step:
    # Check for duplicate rows so each loan record is counted only once.
    # ============================================================
    print_header("READ EDA-REDUCED DATASET")
    df = spark.read.parquet(HDFS_INPUT_PATH)
    print(f"EDA-reduced dataset loaded from: {HDFS_INPUT_PATH}")

    print_status("DATASET SUMMARY", df)

    # ============================================================
    # STEP 2: DUPLICATE CHECKING AND REMOVAL
    # ============================================================
    #
    # Purpose:
    # Detect and remove fully duplicated rows from the dataset.
    #
    # Why:
    # Duplicate loan records can make model training unfair because the same
    # borrower or loan pattern may be counted more than once.
    #
    # Output:
    # A DataFrame with duplicate rows removed, plus a console log showing how
    # many rows were removed.
    #
    # Next step:
    # Validate that important columns still exist after duplicate removal.
    # ============================================================
    df = remove_duplicate_rows(df)

    # ============================================================
    # STEP 3: BASIC SCHEMA AND DATA VALIDATION
    # ============================================================
    #
    # Purpose:
    # Confirm that the dataset still has rows, columns, and required target fields.
    #
    # Why:
    # LoanStatus is needed to build the classification target. BorrowerAPR is
    # needed to build the regression target. If either is missing, later outputs
    # cannot be created correctly.
    #
    # Output:
    # Console validation logs showing row count, column count, and whether key
    # target columns are present.
    #
    # Next step:
    # Remove any known unreliable features that may still exist after File 04.
    # ============================================================
    validate_dataset_schema(df, "after duplicate removal")

    # BorrowerRate is removed if it is still present because it is a direct pricing
    # proxy for BorrowerAPR and can leak information into downstream modeling.
    if "BorrowerRate" in df.columns:
        df = df.drop("BorrowerRate")
        print("BorrowerRate removed because it overlaps strongly with BorrowerAPR.")
        print(f"Columns after BorrowerRate removal: {len(df.columns)}")
    else:
        print("BorrowerRate was not present in the reduced dataset.")

    # ============================================================
    # STEP 4: HIGH-MISSING-VALUE FEATURE REMOVAL
    # ============================================================
    #
    # Purpose:
    # Remove specific Prosper-history columns if they still exist in the dataset.
    #
    # Why:
    # These columns were identified as having more than 80% missing values.
    # Features with this much missing information are less reliable and may add
    # noise to later machine learning models.
    #
    # Output:
    # A DataFrame without the configured high-missing-value columns, if those
    # columns are present.
    #
    # Next step:
    # Review the remaining missing values before imputation.
    # ============================================================
    df = drop_high_missing_value_columns(df)

    # ============================================================
    # STEP 5: MISSING VALUE ANALYSIS AND HANDLING
    # ============================================================
    #
    # Purpose:
    # Summarize missing values and later fill missing numeric and categorical
    # values in a consistent way.
    #
    # Why:
    # Spark ML models generally require complete feature values. Missing values
    # can cause model training errors or reduce model quality if handled poorly.
    #
    # Output:
    # A local missing-value summary table and, after type standardization, a
    # DataFrame with numeric NULL/NaN values filled and categorical NULL values
    # replaced by "Unknown".
    #
    # Next step:
    # Create date-based features before casting feature data types.
    # ============================================================
    print_header("MISSING VALUE ANALYSIS")
    missing_summary = calculate_missing_summary(df)
    print(f"{'column':35} {'missing_count':15} {'missing_percent':15}")
    for row in missing_summary[:30]:
        print(
            f"{row['column']:35} "
            f"{row['missing_count']:<15} "
            f"{row['missing_percent']:<15}"
        )
    save_missing_summary_locally(missing_summary)

    # ============================================================
    # STEP 6: DATE FEATURE ENGINEERING
    # ============================================================
    #
    # Purpose:
    # Convert useful raw date columns into numeric features such as year, month,
    # credit history length, and credit-pull timing.
    #
    # Why:
    # Raw date strings are difficult for ML models to use directly. Derived date
    # features turn time information into simpler numeric values.
    #
    # Output:
    # A shared DataFrame with new date features and without the original raw date
    # columns.
    #
    # Next step:
    # Standardize data types so numeric and categorical columns are consistent.
    # ============================================================
    print_header("SHARED DATE FEATURE ENGINEERING")
    df_shared = create_date_features(df)

    # ============================================================
    # STEP 7: DATA TYPE STANDARDIZATION
    # ============================================================
    #
    # Purpose:
    # Cast expected numeric columns to DoubleType and categorical columns to strings.
    #
    # Why:
    # Consistent data types make later feature selection and ML modeling easier.
    # Numeric columns must be numeric for calculations, while categorical columns
    # should remain strings until encoding is performed in a later modeling file.
    #
    # Output:
    # A shared preprocessed DataFrame with standardized feature types and filled
    # missing values.
    #
    # Next step:
    # Build task-specific datasets for classification and regression.
    # ============================================================
    print_header("SHARED DATA TYPE STANDARDIZATION")
    df_shared = standardize_data_types(df_shared)

    # Missing value handling is performed after type standardization so numeric
    # imputation and NaN checks operate on Spark numeric columns consistently.
    print_header("SHARED MISSING VALUE HANDLING")
    df_shared = fill_numeric_with_median(df_shared, NUMERIC_COLUMNS)
    df_shared = fill_categorical_with_unknown(df_shared)

    # Validate the shared preprocessed dataframe before target-specific outputs are built.
    validate_dataset_schema(df_shared, "after preprocessing")

    # ============================================================
    # STEP 8: CLASSIFICATION TARGET CONSTRUCTION
    # ============================================================
    #
    # Purpose:
    # Create the Good Loan / Bad Loan target variable for classification.
    #
    # Why:
    # Credit risk classification requires a clear final outcome. Completed loans
    # are successful repayments, while Chargedoff and Defaulted loans are credit
    # risk events.
    #
    # Output:
    # A classification DataFrame with label 0 for Good Loan and label 1 for Bad
    # Loan. Current and other non-finalized loans are excluded.
    #
    # Next step:
    # Build the regression dataset for BorrowerAPR prediction.
    # ============================================================
    print_header("BUILD CLASSIFICATION DATASET")
    df_classification = build_classification_dataset(df_shared)
    print(f"Classification rows: {df_classification.count()}")
    print(f"Classification columns: {len(df_classification.columns)}")

    # ============================================================
    # STEP 9: REGRESSION TARGET CONSTRUCTION
    # ============================================================
    #
    # Purpose:
    # Prepare the dataset for predicting BorrowerAPR.
    #
    # Why:
    # BorrowerAPR is a risk-pricing value. Rows without BorrowerAPR cannot be used
    # for supervised regression because the target value is missing.
    #
    # Output:
    # A regression DataFrame containing only rows with valid BorrowerAPR values.
    #
    # Next step:
    # Save both task-specific datasets to HDFS.
    # ============================================================
    print_header("BUILD REGRESSION DATASET")
    df_regression = build_regression_dataset(df_shared)
    print(f"Regression rows: {df_regression.count()}")
    print(f"Regression columns: {len(df_regression.columns)}")

    # ============================================================
    # STEP 10: SAVE PREPROCESSED DATASETS TO HDFS
    # ============================================================
    #
    # Purpose:
    # Save the classification and regression datasets for the next stages of the
    # project pipeline.
    #
    # Why:
    # HDFS storage allows later Spark jobs to read these prepared datasets without
    # repeating all preprocessing steps.
    #
    # Output:
    # prosper_loan_preprocessed_classification
    # prosper_loan_preprocessed_regression
    #
    # Next step:
    # File 06 can perform feature selection, and later files can train Spark ML
    # models using these outputs.
    # ============================================================
    print_header("SAVE PREPROCESSED DATASETS")
    df_classification.write.mode("overwrite").parquet(HDFS_CLASSIFICATION_OUTPUT_PATH)
    df_regression.write.mode("overwrite").parquet(HDFS_REGRESSION_OUTPUT_PATH)

    print("Preprocessed datasets saved successfully.")
    print(f"Classification rows and columns: {df_classification.count()} rows, {len(df_classification.columns)} columns")
    print(f"Regression rows and columns: {df_regression.count()} rows, {len(df_regression.columns)} columns")
    print(f"Classification output path: {HDFS_CLASSIFICATION_OUTPUT_PATH}")
    print(f"Regression output path: {HDFS_REGRESSION_OUTPUT_PATH}")

    spark.stop()


if __name__ == "__main__":
    main()