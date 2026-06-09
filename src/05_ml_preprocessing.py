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


HDFS_INPUT_PATH = "hdfs://localhost:9000/bigdata/prosper_loan/processed/prosper_loan_reduced"
HDFS_CLASSIFICATION_OUTPUT_PATH = (
    "hdfs://localhost:9000/bigdata/prosper_loan/processed/"
    "prosper_loan_preprocessed_classification"
)
HDFS_REGRESSION_OUTPUT_PATH = (
    "hdfs://localhost:9000/bigdata/prosper_loan/processed/"
    "prosper_loan_preprocessed_regression"
)

LOCAL_TABLE_OUTPUT_DIR = os.path.join("outputs", "tables")
MISSING_VALUE_SUMMARY_PATH = os.path.join(
    LOCAL_TABLE_OUTPUT_DIR,
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


def calculate_missing_summary(df):
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
    os.makedirs(LOCAL_TABLE_OUTPUT_DIR, exist_ok=True)
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
        df = df.withColumn(
            "listing_year",
            year(to_date(col("ListingCreationDate"))),
        ).withColumn(
            "listing_month",
            month(to_date(col("ListingCreationDate"))),
        )
        created_features.extend(["listing_year", "listing_month"])

    if "LoanOriginationDate" in df.columns:
        df = df.withColumn(
            "loan_origination_year",
            year(to_date(col("LoanOriginationDate"))),
        ).withColumn(
            "loan_origination_month",
            month(to_date(col("LoanOriginationDate"))),
        )
        created_features.extend(["loan_origination_year", "loan_origination_month"])

    if "LoanOriginationDate" in df.columns and "FirstRecordedCreditLine" in df.columns:
        df = df.withColumn(
            "credit_history_days",
            datediff(
                to_date(col("LoanOriginationDate")),
                to_date(col("FirstRecordedCreditLine")),
            ),
        )
        created_features.append("credit_history_days")

    if "LoanOriginationDate" in df.columns and "DateCreditPulled" in df.columns:
        df = df.withColumn(
            "credit_pull_to_origination_days",
            datediff(
                to_date(col("LoanOriginationDate")),
                to_date(col("DateCreditPulled")),
            ),
        )
        created_features.append("credit_pull_to_origination_days")

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
        df = df.withColumn(column_name, col(column_name).cast(DoubleType()))

    for column_name in existing_categorical_cols:
        if column_name == "IsBorrowerHomeowner":
            df = df.withColumn(
                column_name,
                when(col(column_name).isNull(), lit("Unknown"))
                .when(col(column_name).cast("boolean") == lit(True), lit("true"))
                .when(col(column_name).cast("boolean") == lit(False), lit("false"))
                .otherwise(col(column_name).cast("string")),
            )
        else:
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
        print(f"{row['label']} | {row['count']} | {percentage:.2f}%")


def build_classification_dataset(df_shared):
    if "LoanStatus" not in df_shared.columns:
        raise ValueError("LoanStatus is required to construct the classification label.")

    df_classification = (
        df_shared.withColumn(
            "label",
            when(col("LoanStatus").isin("Chargedoff", "Defaulted"), lit(1))
            .when(col("LoanStatus") == "Completed", lit(0))
            .otherwise(lit(None))
            .cast(IntegerType()),
        )
        .filter(col("label").isNotNull())
    )

    # BorrowerAPR may be removed in the classification ML file if treated as pricing leakage.
    df_classification = safe_drop(df_classification, ["LoanStatus"])

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

    df_regression = df_shared.filter(
        col("BorrowerAPR").isNotNull() & (~isnan(col("BorrowerAPR")))
    )
    df_regression = safe_drop(df_regression, ["LoanStatus"])

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

    print_header("READ REDUCED DATASET")
    df = spark.read.parquet(HDFS_INPUT_PATH)
    print(f"Reduced dataset loaded from: {HDFS_INPUT_PATH}")

    print_status("DATASET SUMMARY", df)

    if "BorrowerRate" in df.columns:
        df = df.drop("BorrowerRate")
        print("BorrowerRate removed because it overlaps strongly with BorrowerAPR.")
        print(f"Columns after BorrowerRate removal: {len(df.columns)}")
    else:
        print("BorrowerRate was not present in the reduced dataset.")

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

    print_header("SHARED DATE FEATURE ENGINEERING")
    df_shared = create_date_features(df)

    print_header("SHARED DATA TYPE STANDARDIZATION")
    df_shared = standardize_data_types(df_shared)

    print_header("SHARED MISSING VALUE HANDLING")
    df_shared = fill_numeric_with_median(df_shared, NUMERIC_COLUMNS)
    df_shared = fill_categorical_with_unknown(df_shared)

    print_header("BUILD CLASSIFICATION DATASET")
    df_classification = build_classification_dataset(df_shared)
    print(f"Classification rows: {df_classification.count()}")
    print(f"Classification columns: {len(df_classification.columns)}")

    print_header("BUILD REGRESSION DATASET")
    df_regression = build_regression_dataset(df_shared)
    print(f"Regression rows: {df_regression.count()}")
    print(f"Regression columns: {len(df_regression.columns)}")

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
