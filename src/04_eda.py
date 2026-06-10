from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
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

    # ============================================================
    # 4.2 Correlation Analysis and Feature Reduction
    # ============================================================

    print("========== CORRELATION ANALYSIS AND FEATURE REDUCTION ==========")

    target_col = "BorrowerAPR"

    # Các biến này không nên đưa vào ML vì có thể gây nhiễu hoặc leakage.
    # BorrowerRate gần như mang cùng ý nghĩa với BorrowerAPR.
    # Các biến LP_* thường là thông tin sau khi khoản vay đã diễn ra,
    # nên nếu dùng để dự đoán rủi ro ban đầu sẽ làm mô hình bị sai lệch.
    manual_remove_candidates = [
        "BorrowerRate",
        "LP_CustomerPayments",
        "LP_CustomerPrincipalPayments",
        "LP_InterestandFees",
        "LP_ServiceFees",
        "LP_CollectionFees",
        "LP_GrossPrincipalLoss",
        "LP_NetPrincipalLoss",
        "LP_NonPrincipalRecoverypayments",
        "Investors",
        "Recommendations"
    ]

    manual_remove_features = [
        col for col in manual_remove_candidates
        if col in df.columns
    ]

    print("Features removed before correlation analysis because of leakage or low modeling value:")
    for col in manual_remove_features:
        print(f"- {col}")

    # Lấy các biến numerical, nhưng bỏ các biến leakage/proxy trước
    numerical_features = [
        field.name
        for field in df.schema.fields
        if field.dataType.simpleString() in ["int", "bigint", "long", "double", "float"]
        and field.name not in manual_remove_features
    ]

    print("Numerical features used for correlation analysis:")
    for feature in numerical_features:
        print(f"- {feature}")

    print("Number of numerical features:", len(numerical_features))

    numerical_features_pd = df.select(numerical_features).toPandas()

    corr_matrix = numerical_features_pd.corr(method="pearson")

    # ============================================================
    # 4.2.1 Correlation with BorrowerAPR
    # ============================================================

    print("========== CORRELATION WITH BORROWER APR ==========")

    corr_with_apr = (
        corr_matrix[target_col]
        .drop(target_col)
        .sort_values(key=lambda x: x.abs(), ascending=False)
        .reset_index()
    )

    corr_with_apr.columns = ["Feature", "CorrelationWithBorrowerAPR"]
    corr_with_apr["AbsCorrelation"] = corr_with_apr["CorrelationWithBorrowerAPR"].abs()

    print(corr_with_apr)

    corr_with_apr.to_csv(
        os.path.join(OUTPUT_DIR, "correlation_with_borrower_apr.csv"),
        index=False
    )

    plot_data = corr_with_apr.sort_values("CorrelationWithBorrowerAPR")

    plt.figure(figsize=(10, 7))
    plt.barh(
        plot_data["Feature"],
        plot_data["CorrelationWithBorrowerAPR"]
    )
    plt.axvline(0, linewidth=1)
    plt.title("Correlation with BorrowerAPR")
    plt.xlabel("Pearson correlation")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "correlation_with_borrower_apr.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print("Saved chart: correlation_with_borrower_apr.png")
    print("Saved table: correlation_with_borrower_apr.csv")

    # ============================================================
    # 4.2.2 High Correlation Between Features
    # ============================================================

    print("========== HIGH CORRELATION BETWEEN FEATURES ==========")

    # 0.85 phù hợp hơn 0.7 vì 0.7 có thể loại quá nhiều biến.
    # Chỉ những cặp tương quan rất cao mới bị xem là trùng thông tin mạnh.
    threshold = 0.85

    corr_pairs = (
        corr_matrix
        .where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )

    corr_pairs.columns = ["Feature_1", "Feature_2", "Correlation"]
    corr_pairs["AbsCorrelation"] = corr_pairs["Correlation"].abs()

    high_corr_pairs = (
        corr_pairs[corr_pairs["AbsCorrelation"] >= threshold]
        .sort_values("AbsCorrelation", ascending=False)
        .reset_index(drop=True)
    )

    print(high_corr_pairs)

    high_corr_pairs.to_csv(
        os.path.join(OUTPUT_DIR, "high_correlation_pairs.csv"),
        index=False
    )

    print("Saved table: high_correlation_pairs.csv")

    # ============================================================
    # 4.2.3 Automatically Suggest Redundant Features to Remove
    # ============================================================

    print("========== SUGGESTED REDUNDANT FEATURES TO REMOVE ==========")

    redundant_remove_features = set()

    for _, row in high_corr_pairs.iterrows():
        feature_1 = row["Feature_1"]
        feature_2 = row["Feature_2"]

        # Không loại target
        if feature_1 == target_col or feature_2 == target_col:
            continue

        corr_1 = abs(corr_matrix.loc[feature_1, target_col]) if feature_1 in corr_matrix.index else 0
        corr_2 = abs(corr_matrix.loc[feature_2, target_col]) if feature_2 in corr_matrix.index else 0

        # Nếu hai biến trùng thông tin, giữ lại biến có tương quan mạnh hơn với BorrowerAPR
        if corr_1 >= corr_2:
            redundant_remove_features.add(feature_2)
        else:
            redundant_remove_features.add(feature_1)

    redundant_remove_features = sorted(list(redundant_remove_features))

    print("Redundant features suggested to remove based on high correlation:")
    if len(redundant_remove_features) == 0:
        print("- No redundant numerical features found based on the selected threshold.")
    else:
        for col in redundant_remove_features:
            print(f"- {col}")

    # ============================================================
    # 4.2.4 Final Feature Removal List for MLlib
    # ============================================================

    print("========== FINAL FEATURES REMOVED FOR MLLIB ==========")

    final_remove_features = sorted(
        list(set(manual_remove_features + redundant_remove_features))
    )

    feature_removal_summary = pd.DataFrame({
        "RemovedFeature": final_remove_features,
        "Reason": [
            "Manual removal: leakage, target proxy, or low modeling value"
            if feature in manual_remove_features
            else "Automatic removal: highly correlated with another numerical feature"
            for feature in final_remove_features
        ]
    })

    print(feature_removal_summary)

    feature_removal_summary.to_csv(
        os.path.join(OUTPUT_DIR, "removed_features_for_mllib.csv"),
        index=False
    )

    print("Saved table: removed_features_for_mllib.csv")

    # Tạo dataset mới cho MLlib sau khi loại các biến không cần thiết
    df_mllib_ready = df.drop(*final_remove_features)

    print("========== DATASET AFTER FEATURE REDUCTION ==========")
    print(f"Original number of columns: {len(df.columns)}")
    print(f"Number of removed columns: {len(final_remove_features)}")
    print(f"Remaining number of columns: {len(df_mllib_ready.columns)}")

    print("Remaining columns for MLlib:")
    for col in df_mllib_ready.columns:
        print(f"- {col}")

    # Nếu muốn lưu dataset đã giảm feature để dùng cho MLlib
    MLLIB_READY_DATASET_PATH = (
        "hdfs://localhost:9000/bigdata/prosper_loan/processed/prosper_loan_mllib_ready"
    )

    df_mllib_ready.write.mode("overwrite").parquet(MLLIB_READY_DATASET_PATH)

    print(f"Saved MLlib-ready dataset to: {MLLIB_READY_DATASET_PATH}")

    # ============================================================
    # 4.2.5 Correlation Matrix Heatmap
    # ============================================================

    print("========== CORRELATION MATRIX HEATMAP ==========")

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        center=0,
        annot=False,
        linewidths=0.5
    )
    plt.title("Correlation Matrix of Numerical Features")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "correlation_matrix_heatmap.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print("Saved chart: correlation_matrix_heatmap.png")

    print(
        "Conclusion: Correlation analysis helps identify variables that are either "
        "too similar to each other or may cause data leakage. These variables are "
        "removed before MLlib modeling to reduce noise, avoid target leakage, and "
        "make the model more reliable."
    )

    print("========== EDA COMPLETED ==========")
    spark.stop()


if __name__ == "__main__":
    main()