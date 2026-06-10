import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


HDFS_REDUCED_DATASET_PATH = (
    "hdfs://localhost:9000/bigdata/prosper_loan/processed/prosper_loan_reduced"
)
HDFS_EDA_REDUCED_DATASET_PATH = (
    "hdfs://localhost:9000/bigdata/prosper_loan/processed/prosper_loan_eda_reduced"
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
    "$100,000+",
]

numeric_type_names = {
    "byte",
    "short",
    "int",
    "bigint",
    "long",
    "float",
    "double",
}

manual_remove_reason_by_feature = {
    "BorrowerRate": (
        "Manual removal: overlaps strongly with BorrowerAPR and is excluded as a "
        "risk-pricing proxy."
    ),
    "LP_CustomerPayments": (
        "Manual removal: post-loan payment accounting field that may leak outcome "
        "information."
    ),
    "LP_CustomerPrincipalPayments": (
        "Manual removal: post-loan principal payment field that may leak outcome "
        "information."
    ),
    "LP_InterestandFees": (
        "Manual removal: post-loan interest and fee accounting field."
    ),
    "LP_ServiceFees": (
        "Manual removal: post-loan service fee accounting field."
    ),
    "LP_CollectionFees": (
        "Manual removal: post-loan collection fee field related to loan performance."
    ),
    "LP_GrossPrincipalLoss": (
        "Manual removal: loss field that directly reflects post-loan outcome."
    ),
    "LP_NetPrincipalLoss": (
        "Manual removal: loss field that directly reflects post-loan outcome."
    ),
    "LP_NonPrincipalRecoverypayments": (
        "Manual removal: recovery payment field observed after loan performance."
    ),
    "Investors": (
        "Manual removal: platform funding participation signal with limited borrower "
        "credit-risk business value for the EDA-reduced feature set."
    ),
    "Recommendations": (
        "Manual removal: platform/social signal with limited direct credit-risk "
        "business value."
    ),
}


def ensure_output_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def print_table_from_pandas(data):
    if data.empty:
        print("No rows to display.")
        return

    print(data.to_string(index=False))


def print_feature_list(title, features):
    print(f"{title} ({len(features)}):")
    if not features:
        print("- None")
        return

    for feature in features:
        print(f"- {feature}")


def calculate_missing_summary(df, total_rows, excluded_columns=None):
    excluded_columns = set(excluded_columns or [])
    columns = [column for column in df.columns if column not in excluded_columns]

    if not columns:
        return pd.DataFrame(columns=["column", "missing_count", "missing_percent"])

    null_expressions = [
        F.sum(F.when(F.col(column).isNull(), 1).otherwise(0)).alias(column)
        for column in columns
    ]
    missing_counts = df.agg(*null_expressions).collect()[0].asDict()

    missing_summary = []
    for column in columns:
        missing_count = missing_counts[column]
        missing_percent = (
            (missing_count / total_rows) * 100
            if total_rows > 0
            else 0
        )
        missing_summary.append(
            {
                "column": column,
                "missing_count": missing_count,
                "missing_percent": round(missing_percent, 2),
            }
        )

    return (
        pd.DataFrame(missing_summary)
        .sort_values("missing_percent", ascending=False)
        .reset_index(drop=True)
    )


def get_numeric_features(df, excluded_columns=None):
    excluded_columns = set(excluded_columns or [])
    numeric_features = []

    for field in df.schema.fields:
        data_type = field.dataType.simpleString()
        is_numeric = data_type in numeric_type_names or data_type.startswith("decimal")
        if is_numeric and field.name not in excluded_columns:
            numeric_features.append(field.name)

    return numeric_features


def calculate_correlation_matrix(df, numeric_features):
    matrix = pd.DataFrame(
        np.eye(len(numeric_features)),
        index=numeric_features,
        columns=numeric_features,
    )

    for index_1, feature_1 in enumerate(numeric_features):
        for index_2 in range(index_1 + 1, len(numeric_features)):
            feature_2 = numeric_features[index_2]
            correlation = (
                df.select(F.corr(F.col(feature_1), F.col(feature_2)).alias("corr"))
                .collect()[0]["corr"]
            )

            if correlation is None:
                correlation = np.nan

            matrix.loc[feature_1, feature_2] = correlation
            matrix.loc[feature_2, feature_1] = correlation

    return matrix


def build_high_correlation_pairs(corr_matrix, threshold):
    corr_pairs = (
        corr_matrix
        .where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )
    corr_pairs.columns = ["Feature_1", "Feature_2", "Correlation"]
    corr_pairs["AbsCorrelation"] = corr_pairs["Correlation"].abs()

    return (
        corr_pairs[corr_pairs["AbsCorrelation"] >= threshold]
        .sort_values("AbsCorrelation", ascending=False)
        .reset_index(drop=True)
    )


def add_removal_record(records, feature, stage, reason):
    if feature is None:
        return

    records.append(
        {
            "RemovedFeature": feature,
            "RemovalStage": stage,
            "Reason": reason,
        }
    )


def drop_features_if_present(df, features):
    existing_features = [feature for feature in features if feature in df.columns]
    if not existing_features:
        return df

    return df.drop(*existing_features)


def main():
    spark = (
        SparkSession.builder.appName("ProsperLoanEDAFeatureReduction")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    ensure_output_dirs()
    removal_records = []

    print("========== READ REDUCED DATASET ==========")
    df = spark.read.parquet(HDFS_REDUCED_DATASET_PATH)
    print(f"Reduced dataset loaded from: {HDFS_REDUCED_DATASET_PATH}")

    print("========== DATASET SUMMARY ==========")
    total_rows = df.count()
    original_columns = df.columns
    original_column_count = len(original_columns)

    print(f"Total rows: {total_rows}")
    print(f"Total columns: {original_column_count}")
    print("Columns:")
    for column in original_columns:
        print(f"- {column}")

    if "BorrowerRate" in original_columns:
        print(
            "Note: BorrowerRate exists in the current dataset but is ignored in EDA "
            "correlation analysis because it overlaps strongly with BorrowerAPR."
        )

    print("========== 4.1 DISTRIBUTION ANALYSIS ==========")

    print("========== LOAN OUTCOME DISTRIBUTION ==========")
    outcome_col = (
        F.when(F.col("LoanStatus") == "Completed", "Good Loan")
        .when(F.col("LoanStatus").isin("Chargedoff", "Defaulted"), "Bad Loan")
        .otherwise("Other / Not Finalized")
    )
    loan_outcome_pd = (
        df.withColumn("outcome_group", outcome_col)
        .groupBy("outcome_group")
        .count()
        .orderBy(F.col("count").desc())
        .toPandas()
    )
    if total_rows > 0:
        loan_outcome_pd["percentage"] = (
            loan_outcome_pd["count"] / total_rows * 100
        ).round(2)
    else:
        loan_outcome_pd["percentage"] = 0
    print_table_from_pandas(loan_outcome_pd[["outcome_group", "count", "percentage"]])

    plt.figure(figsize=(8, 8))
    plt.pie(
        loan_outcome_pd["count"],
        labels=loan_outcome_pd["outcome_group"],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"width": 0.45},
    )
    plt.title("Loan Outcome Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loan_outcome_distribution.png"), dpi=300)
    plt.close()
    print(
        "Conclusion: The dataset contains a large proportion of active or non-finalized "
        "loans, so only final outcomes should be used later when constructing the "
        "classification target."
    )

    print("========== LOAN STATUS DISTRIBUTION ==========")
    loan_status_pd = (
        df.groupBy("LoanStatus")
        .count()
        .orderBy(F.col("count").desc())
        .toPandas()
    )
    print_table_from_pandas(loan_status_pd)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=loan_status_pd, y="LoanStatus", x="count", color="#4C78A8")
    plt.title("LoanStatus Distribution")
    plt.xlabel("Count")
    plt.ylabel("LoanStatus")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loan_status_distribution.png"), dpi=300)
    plt.close()
    print(
        "Conclusion: Current and Completed loans dominate the raw status distribution, "
        "while Defaulted and Chargedoff loans form a smaller but important risk group."
    )

    print("========== INCOME RANGE DISTRIBUTION ==========")
    income_pd = df.groupBy("IncomeRange").count().toPandas()
    income_pd["IncomeRange"] = pd.Categorical(
        income_pd["IncomeRange"],
        categories=income_order,
        ordered=True,
    )
    income_pd = income_pd.sort_values("IncomeRange").reset_index(drop=True)
    print_table_from_pandas(income_pd)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=income_pd, x="IncomeRange", y="count", color="#59A14F")
    plt.title("IncomeRange Distribution")
    plt.xlabel("IncomeRange")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "income_range_distribution.png"), dpi=300)
    plt.close()
    print(
        "Conclusion: Borrowers are concentrated mainly in middle-income groups, while "
        "unemployed or zero-income groups are much smaller but may represent higher "
        "repayment risk."
    )

    print("========== PROSPER SCORE DISTRIBUTION ==========")
    prosper_score_pd = (
        df.groupBy("ProsperScore")
        .count()
        .orderBy(F.col("ProsperScore").asc_nulls_last())
        .toPandas()
    )
    print_table_from_pandas(prosper_score_pd)

    prosper_score_plot_pd = prosper_score_pd.copy()
    prosper_score_plot_pd["ProsperScoreLabel"] = (
        prosper_score_plot_pd["ProsperScore"].astype("string").fillna("Missing")
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=prosper_score_plot_pd,
        x="ProsperScoreLabel",
        y="count",
        color="#F28E2B",
    )
    plt.plot(
        range(len(prosper_score_plot_pd)),
        prosper_score_plot_pd["count"],
        color="#1F2937",
        marker="o",
        linewidth=2,
    )
    plt.title("ProsperScore Distribution")
    plt.xlabel("ProsperScore")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "prosper_score_distribution.png"), dpi=300)
    plt.close()
    print(
        "Conclusion: ProsperScore is concentrated in the middle range, suggesting that "
        "most borrowers fall into moderate-risk groups."
    )

    print("========== MISSING VALUE DISTRIBUTION ==========")
    missing_summary_pd = calculate_missing_summary(
        df,
        total_rows,
        excluded_columns=["BorrowerRate"],
    )
    top_missing_pd = missing_summary_pd.head(10)
    print_table_from_pandas(top_missing_pd)

    plot_missing_pd = top_missing_pd.sort_values("missing_percent", ascending=True)
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=plot_missing_pd,
        x="missing_percent",
        y="column",
        color="#E15759",
    )
    plt.title("Top Missing Value Percentages by Feature")
    plt.xlabel("Missing Percent")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "missing_value_distribution.png"), dpi=300)
    plt.close()
    print(
        "Conclusion: Features with missing value percentages above 80% are strong "
        "candidates for removal during EDA-based feature reduction."
    )

    print("========== FEATURES REMOVED BY DISTRIBUTION/MISSING VALUE ANALYSIS ==========")
    distribution_candidate_features = [
        "TotalProsperLoans",
        "ProsperPaymentsLessThanOneMonthLate",
        "ProsperPaymentsOneMonthPlusLate",
    ]
    missing_percent_by_column = dict(
        zip(missing_summary_pd["column"], missing_summary_pd["missing_percent"])
    )
    distribution_removed_features = []

    for feature in distribution_candidate_features:
        missing_percent = missing_percent_by_column.get(feature)
        if missing_percent is not None and missing_percent >= 80:
            distribution_removed_features.append(feature)
            add_removal_record(
                removal_records,
                feature,
                "Distribution/Missing Value Analysis",
                "Missing value percentage above 80%.",
            )

    distribution_removal_pd = pd.DataFrame(
        [
            {
                "Feature": feature,
                "MissingPercent": missing_percent_by_column[feature],
                "Reason": "Missing value percentage above 80%.",
            }
            for feature in distribution_removed_features
        ],
        columns=["Feature", "MissingPercent", "Reason"],
    )
    print_table_from_pandas(distribution_removal_pd)
    print(
        "These features are removed because more than 80% of their values are missing, "
        "making them less reliable for downstream modeling."
    )

    df_after_distribution = drop_features_if_present(df, distribution_removed_features)

    print("========== 4.2 CORRELATION ANALYSIS ==========")
    print(
        "BorrowerAPR is used as a risk-pricing proxy for correlation analysis. "
        "Classification labels will be constructed later from LoanStatus."
    )

    print("========== MANUAL LEAKAGE OR PROXY REMOVAL BEFORE CORRELATION ==========")
    manual_remove_features = [
        feature
        for feature in manual_remove_reason_by_feature
        if feature in df_after_distribution.columns
    ]
    manual_removal_pd = pd.DataFrame(
        [
            {
                "Feature": feature,
                "Reason": manual_remove_reason_by_feature[feature],
            }
            for feature in manual_remove_features
        ],
        columns=["Feature", "Reason"],
    )
    print_table_from_pandas(manual_removal_pd)
    for feature in manual_remove_features:
        add_removal_record(
            removal_records,
            feature,
            "Manual Leakage or Proxy Removal",
            manual_remove_reason_by_feature[feature],
        )

    df_for_correlation = drop_features_if_present(
        df_after_distribution,
        manual_remove_features,
    )

    target_col = "BorrowerAPR"
    excluded_from_correlation = set(distribution_removed_features + manual_remove_features)
    excluded_from_correlation.add("BorrowerRate")
    numeric_features = get_numeric_features(
        df_for_correlation,
        excluded_columns=excluded_from_correlation,
    )

    print("Numerical features used for correlation analysis:")
    print_feature_list("Features", numeric_features)

    if target_col not in numeric_features:
        print(
            "BorrowerAPR is not available after EDA filtering, so Chapter 4.2 "
            "correlation analysis is skipped."
        )
        corr_matrix = pd.DataFrame()
        corr_with_apr = pd.DataFrame(
            columns=["Feature", "CorrelationWithBorrowerAPR", "AbsCorrelation"]
        )
        high_corr_pairs = pd.DataFrame(
            columns=["Feature_1", "Feature_2", "Correlation", "AbsCorrelation"]
        )
        redundant_remove_features = []
    else:
        corr_matrix = calculate_correlation_matrix(df_for_correlation, numeric_features)

        print("========== 4.2.1 Correlation with BorrowerAPR ==========")
        corr_with_apr = (
            corr_matrix[target_col]
            .drop(target_col)
            .dropna()
            .sort_values(key=lambda values: values.abs(), ascending=False)
            .reset_index()
        )
        corr_with_apr.columns = ["Feature", "CorrelationWithBorrowerAPR"]
        corr_with_apr["AbsCorrelation"] = (
            corr_with_apr["CorrelationWithBorrowerAPR"].abs()
        )

        print_table_from_pandas(corr_with_apr)
        corr_with_apr.to_csv(
            os.path.join(OUTPUT_DIR, "correlation_with_borrower_apr.csv"),
            index=False,
        )

        plot_data = corr_with_apr.sort_values("CorrelationWithBorrowerAPR")
        plt.figure(figsize=(10, 7))
        plt.barh(
            plot_data["Feature"],
            plot_data["CorrelationWithBorrowerAPR"],
            color="#4C78A8",
        )
        plt.axvline(0, linewidth=1, color="#1F2937")
        plt.title("Correlation with BorrowerAPR")
        plt.xlabel("Pearson Correlation")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, "correlation_with_borrower_apr.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print("Saved chart: correlation_with_borrower_apr.png")
        print("Saved table: correlation_with_borrower_apr.csv")
        print(
            "Conclusion: Features with stronger correlation to BorrowerAPR may be "
            "useful for the regression task because BorrowerAPR reflects risk-based "
            "loan pricing."
        )

        print("========== 4.2.2 High Correlation Between Features ==========")
        threshold = 0.85
        high_corr_pairs = build_high_correlation_pairs(corr_matrix, threshold)
        print_table_from_pandas(high_corr_pairs)
        high_corr_pairs.to_csv(
            os.path.join(OUTPUT_DIR, "high_correlation_pairs.csv"),
            index=False,
        )
        print("Saved table: high_correlation_pairs.csv")
        print(
            "Conclusion: Highly correlated feature pairs suggest redundant information "
            "and support EDA-based feature reduction."
        )

        print("========== 4.2.3 Redundant Feature Detection ==========")
        redundant_remove_features = set()
        target_correlations = corr_matrix[target_col].abs()

        for _, row in high_corr_pairs.iterrows():
            feature_1 = row["Feature_1"]
            feature_2 = row["Feature_2"]

            if feature_1 == target_col or feature_2 == target_col:
                continue

            corr_1 = target_correlations.get(feature_1, 0)
            corr_2 = target_correlations.get(feature_2, 0)

            if pd.isna(corr_1):
                corr_1 = 0
            if pd.isna(corr_2):
                corr_2 = 0

            if corr_1 >= corr_2:
                redundant_remove_features.add(feature_2)
            else:
                redundant_remove_features.add(feature_1)

        redundant_remove_features = sorted(redundant_remove_features)
        print_feature_list(
            "Redundant features suggested for removal",
            redundant_remove_features,
        )
        print(
            "Conclusion: When two predictors contain similar information, the feature "
            "with weaker relationship to BorrowerAPR is removed as redundant."
        )

        print("========== 4.2.4 Feature Reduction Based on Correlation Analysis ==========")
        correlation_removal_pd = pd.DataFrame(
            [
                {
                    "Feature": feature,
                    "Reason": (
                        "Highly correlated with another numerical feature and has "
                        "weaker correlation with BorrowerAPR."
                    ),
                }
                for feature in redundant_remove_features
            ],
            columns=["Feature", "Reason"],
        )
        print_table_from_pandas(correlation_removal_pd)
        for feature in redundant_remove_features:
            add_removal_record(
                removal_records,
                feature,
                "Correlation Redundancy Removal",
                (
                    "Highly correlated with another numerical feature and has weaker "
                    "correlation with BorrowerAPR."
                ),
            )
        print(
            "Conclusion: Correlation-based feature reduction removes redundant "
            "numerical variables while keeping the stronger risk-pricing signal."
        )

        print("========== 4.2.5 Correlation Matrix Heatmap ==========")
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            cmap="coolwarm",
            center=0,
            annot=False,
            linewidths=0.5,
        )
        plt.title("Correlation Matrix of Numerical Features")
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, "correlation_matrix_heatmap.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print("Saved chart: correlation_matrix_heatmap.png")
        print(
            "Conclusion: The heatmap provides a compact overview of numerical feature "
            "relationships and highlights areas of redundancy."
        )

    removed_features_by_stage_pd = pd.DataFrame(
        removal_records,
        columns=["RemovedFeature", "RemovalStage", "Reason"],
    ).drop_duplicates(subset=["RemovedFeature"], keep="first")
    removed_features_by_stage_pd.to_csv(
        os.path.join(OUTPUT_DIR, "removed_features_by_eda.csv"),
        index=False,
    )

    final_remove_features = removed_features_by_stage_pd["RemovedFeature"].tolist()
    df_eda_reduced = drop_features_if_present(df, final_remove_features)

    print("========== EDA FEATURE REDUCTION SUMMARY ==========")
    print(f"Original columns before EDA reduction: {original_column_count}")
    print_feature_list(
        "Features removed in Chapter 4.1",
        distribution_removed_features,
    )
    print_feature_list(
        "Features removed in Chapter 4.2 manual removal",
        manual_remove_features,
    )
    print_feature_list(
        "Features removed in Chapter 4.2 correlation redundancy",
        redundant_remove_features,
    )
    print(f"Final columns after EDA reduction: {len(df_eda_reduced.columns)}")
    print(f"HDFS path of saved EDA-reduced dataset: {HDFS_EDA_REDUCED_DATASET_PATH}")
    print("Saved table: removed_features_by_eda.csv")

    print("========== SAVE EDA-REDUCED DATASET TO HDFS ==========")
    df_eda_reduced.write.mode("overwrite").parquet(HDFS_EDA_REDUCED_DATASET_PATH)
    print(f"Saved EDA-reduced dataset to: {HDFS_EDA_REDUCED_DATASET_PATH}")

    print("========== EDA COMPLETED ==========")
    spark.stop()


if __name__ == "__main__":
    main()
