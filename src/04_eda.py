from pyspark.sql import SparkSession
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


HDFS_REDUCED_DATASET_PATH = (
    "hdfs://localhost:9000/bigdata/prosper_loan/processed/prosper_loan_reduced"
)
OUTPUT_DIR = os.path.join("outputs", "charts")
income_order = [
    "Not displayed",
    "Not employed",
    "$0",
    "$1-24,999",
    "$25,000-49,999",
    "$50,000-74,999",
    "$75,000-99,999",
    "$100,000+"
]


def print_aggregated_table(data, label_col):
    print(f"{label_col} | count")
    print("-" * (len(label_col) + 9))
    for _, row in data.iterrows():
        print(f"{str(row[label_col]):<30} | {row['count']}")


def main():
    spark = (
        SparkSession.builder.appName("ProsperLoanEDA")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    print("========== READ REDUCED DATASET ==========")
    df = spark.read.parquet(HDFS_REDUCED_DATASET_PATH)
    print(f"Reduced dataset loaded from: {HDFS_REDUCED_DATASET_PATH}")

    print("========== DATASET SUMMARY ==========")
    total_rows = df.count()
    columns = df.columns
    total_columns = len(columns)

    print(f"Total rows: {total_rows}")
    print(f"Total columns: {total_columns}")
    print("Columns:")
    for column in columns:
        print(f"- {column}")

    print("========== DISTRIBUTION ANALYSIS ==========")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("========== LOAN STATUS DISTRIBUTION ==========")
    loan_status_pd = (
        df.groupBy("LoanStatus")
        .count()
        .orderBy("count", ascending=False)
        .toPandas()
    )
    print_aggregated_table(loan_status_pd, "LoanStatus")

    print("========== INCOME RANGE DISTRIBUTION ==========")
    income_df = (
        df.groupBy("IncomeRange")
        .count()
    )
    income_pd = income_df.toPandas()
    income_pd["IncomeRange"] = pd.Categorical(
        income_pd["IncomeRange"],
        categories=income_order,
        ordered=True
    )
    income_pd = income_pd.sort_values("IncomeRange")
    print("Income ranges are displayed from lowest to highest earning groups for easier interpretation.")
    print_aggregated_table(income_pd, "IncomeRange")

    print("========== PROSPER SCORE DISTRIBUTION ==========")
    prosper_score_pd = (
        df.groupBy("ProsperScore")
        .count()
        .orderBy("ProsperScore")
        .toPandas()
    )
    print_aggregated_table(prosper_score_pd, "ProsperScore")

    print("========== SAVE CHARTS ==========")
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=loan_status_pd,
        y="LoanStatus",
        x="count"
    )
    plt.title("Loan Status Distribution")
    plt.xlabel("Count")
    plt.ylabel("LoanStatus")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loan_status_distribution.png"), dpi=300)
    plt.close()
    print(
        "Conclusion: Current and Completed loans dominate the dataset, while default "
        "and delinquent loans represent a much smaller portion. This indicates class "
        "imbalance that should be considered in later machine learning tasks."
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=income_pd,
        x="IncomeRange",
        y="count"
    )
    plt.title("Income Range Distribution")
    plt.xlabel("IncomeRange")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "income_range_distribution.png"), dpi=300)
    plt.close()
    print("Conclusion: IncomeRange distribution helps describe borrower repayment capacity.")

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=prosper_score_pd,
        x="ProsperScore",
        y="count"
    )
    plt.title("Prosper Score Distribution")
    plt.xlabel("ProsperScore")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "prosper_score_distribution.png"), dpi=300)
    plt.close()
    print(
        "Conclusion: ProsperScore distribution shows the risk profile of borrowers "
        "in the dataset."
    )

    print("========== EDA COMPLETED ==========")
    spark.stop()


if __name__ == "__main__":
    main()
