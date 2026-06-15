from pyspark.sql import SparkSession


HDFS_INPUT_PATH = "hdfs://localhost:9000/bigdata/prosper_loan/raw/prosperLoanData.csv"
HDFS_OUTPUT_PATH = "hdfs://localhost:9000/bigdata/prosper_loan/processed/prosper_loan_reduced"

spark = (
    SparkSession.builder
    .appName("Prosper Loan Domain Feature Reduction")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

feature_progress = []


def print_status(step_name, dataframe):
    print(f"\n========== {step_name} ==========")
    print(f"Rows: {dataframe.count()}")
    print(f"Columns: {len(dataframe.columns)}")


def print_feature_list(title, columns):
    print(f"{title} ({len(columns)}):")
    if not columns:
        print("- None")
        return

    for column_name in columns:
        print(f"- {column_name}")


def record_feature_progress(step_name, dataframe):
    feature_progress.append((step_name, len(dataframe.columns)))


def print_initial_domain_filtering_summary(columns_before, columns_removed, columns_remaining):
    print("\n========== INITIAL DOMAIN FILTERING SUMMARY ==========")
    print(f"Columns before filtering: {columns_before}")
    print(f"Columns removed: {columns_removed}")
    print(f"Columns remaining: {columns_remaining}")

    print("\nRemoved column categories:")
    print("- Identifier columns")
    print("- Duplicate loan identifiers")
    print("- Internal platform tracking fields")
    print("- Historical payment/recovery accounting fields")
    print("- Post-loan outcome leakage variables")

    print("\nReason:")
    print(
        "These columns are not suitable as predictive features because they either "
        "uniquely identify records, contain duplicated identifiers, represent internal "
        "platform bookkeeping fields, or leak information that would not be available "
        "at loan origination time."
    )

    print("\nConclusion:")
    print(
        f"Initial domain filtering reduced the dataset from {columns_before} to "
        f"{columns_remaining} features."
    )
    print(
        "The removed attributes do not contribute meaningful predictive information "
        "for loan risk modeling and may introduce noise or data leakage."
    )
    print(
        "The remaining features will be evaluated further through domain-based grouping "
        "and feature analysis."
    )


def print_feature_reduction_progress():
    print("\n========== FEATURE REDUCTION PROGRESS ==========")
    print(f"{'Step':<45} Columns")
    print("-" * 55)
    for step_name, column_count in feature_progress:
        print(f"{step_name:<45} {column_count}")


def safe_drop(dataframe, columns_to_drop):
    existing_columns = [col_name for col_name in columns_to_drop if col_name in dataframe.columns]
    missing_columns = [col_name for col_name in columns_to_drop if col_name not in dataframe.columns]

    if missing_columns:
        print_feature_list("Missing columns skipped", missing_columns)

    if not existing_columns:
        return dataframe

    print_feature_list("Dropped columns", existing_columns)
    return dataframe.drop(*existing_columns)


print("\n========== READ RAW DATA FROM HDFS ==========")

df = spark.read.csv(
    HDFS_INPUT_PATH,
    header=True,
    inferSchema=True
)

print_status("RAW DATASET", df)
record_feature_progress("Raw Dataset", df)

df_reduced = df
df_reduced.createOrReplaceTempView("prosper_loan_reduced")


# GROUP 0: Initial Domain Filtering
# Remove identifiers and post-origination fields that may cause leakage.
cols_group_0 = [
    "ListingKey",
    "ListingNumber",
    "LoanKey",
    "LoanNumber",
    "MemberKey",
    "GroupKey",
    "LP_CustomerPayments",
    "LP_CustomerPrincipalPayments",
    "LP_InterestandFees",
    "LP_ServiceFees",
    "LP_CollectionFees",
    "LP_GrossPrincipalLoss",
    "LP_NetPrincipalLoss",
    "LP_NonPrincipalRecoverypayments",
    "LoanCurrentDaysDelinquent",
    "LoanFirstDefaultedCycleNumber",
    "LoanMonthsSinceOrigination",
    "CurrentlyInGroup",
    "ClosedDate"
]

group_0_columns_before = len(df_reduced.columns)
df_reduced = safe_drop(df_reduced, cols_group_0)
df_reduced.createOrReplaceTempView("prosper_loan_reduced")
print_status("GROUP 0 - AFTER INITIAL DOMAIN FILTERING", df_reduced)
record_feature_progress("Initial Domain Filtering", df_reduced)
print_initial_domain_filtering_summary(
    group_0_columns_before,
    group_0_columns_before - len(df_reduced.columns),
    len(df_reduced.columns)
)


# GROUP 1: Internal Risk Metrics
# Check whether ProsperScore summarizes internal pricing and risk signals.
print("\n========== GROUP 1 - INTERNAL RISK METRICS ==========")

spark.sql("""
SELECT
    ProsperScore,
    COUNT(*) AS total_loans,
    ROUND(AVG(BorrowerAPR), 4) AS avg_apr,
    ROUND(
        AVG(
            CASE
                WHEN LoanStatus IN ('Chargedoff', 'Defaulted') THEN 1
                WHEN LoanStatus = 'Completed' THEN 0
                ELSE NULL
            END
        ) * 100, 2
    ) AS bad_loan_rate_percent
FROM prosper_loan_reduced
WHERE ProsperScore IS NOT NULL
GROUP BY ProsperScore
ORDER BY ProsperScore
""").show(truncate=False)

print("Conclusion: ProsperScore is retained because it captures both loan pricing and credit risk patterns.")

cols_group_1 = [
    "ProsperRating (Alpha)",
    "ProsperRating (numeric)",
    "CreditGrade"
]

df_reduced = safe_drop(df_reduced, cols_group_1)
df_reduced.createOrReplaceTempView("prosper_loan_reduced")
print_status("GROUP 1 - AFTER DROP", df_reduced)
record_feature_progress("After Group 1 - Internal Risk Metrics", df_reduced)


# GROUP 2: Income and Employment
# Check borrower repayment capacity through income and employment status.
print("\n========== GROUP 2 - INCOME AND EMPLOYMENT ==========")

spark.sql("""
SELECT
    IncomeRange,
    EmploymentStatus,
    COUNT(*) AS total_loans,
    ROUND(AVG(BorrowerAPR), 4) AS avg_apr,
    ROUND(
        AVG(
            CASE
                WHEN LoanStatus IN ('Chargedoff', 'Defaulted') THEN 1
                WHEN LoanStatus = 'Completed' THEN 0
                ELSE NULL
            END
        ) * 100, 2
    ) AS bad_loan_rate_percent
FROM prosper_loan_reduced
WHERE IncomeRange IS NOT NULL
  AND EmploymentStatus IS NOT NULL
GROUP BY IncomeRange, EmploymentStatus
HAVING COUNT(*) >= 100
ORDER BY IncomeRange, EmploymentStatus
""").show(100, truncate=False)

print("Conclusion: IncomeRange and EmploymentStatus are retained because they represent borrower repayment capacity.")

cols_group_2 = [
    "IncomeVerifiable",
    "EmploymentStatusDuration",
    "StatedMonthlyIncome"
]

df_reduced = safe_drop(df_reduced, cols_group_2)
df_reduced.createOrReplaceTempView("prosper_loan_reduced")
print_status("GROUP 2 - AFTER DROP", df_reduced)
record_feature_progress("After Group 2 - Income and Employment", df_reduced)


# GROUP 3: Credit History
# Compare credit score and DTI as complementary credit risk dimensions.
print("\n========== GROUP 3 - CREDIT HISTORY ==========")

spark.sql("""
SELECT
    CASE
        WHEN CreditScoreRangeLower < 600 THEN '<600'
        WHEN CreditScoreRangeLower < 660 THEN '600-659'
        WHEN CreditScoreRangeLower < 720 THEN '660-719'
        ELSE '720+'
    END AS credit_score_group,
    CASE
        WHEN DebtToIncomeRatio < 0.10 THEN 'Low DTI'
        WHEN DebtToIncomeRatio < 0.30 THEN 'Medium DTI'
        ELSE 'High DTI'
    END AS dti_group,
    COUNT(*) AS total_loans,
    ROUND(AVG(BorrowerAPR), 4) AS avg_apr,
    ROUND(
        AVG(
            CASE
                WHEN LoanStatus IN ('Chargedoff', 'Defaulted') THEN 1
                WHEN LoanStatus = 'Completed' THEN 0
                ELSE NULL
            END
        ) * 100, 2
    ) AS bad_loan_rate_percent
FROM prosper_loan_reduced
WHERE CreditScoreRangeLower IS NOT NULL
  AND DebtToIncomeRatio IS NOT NULL
GROUP BY
    CASE
        WHEN CreditScoreRangeLower < 600 THEN '<600'
        WHEN CreditScoreRangeLower < 660 THEN '600-659'
        WHEN CreditScoreRangeLower < 720 THEN '660-719'
        ELSE '720+'
    END,
    CASE
        WHEN DebtToIncomeRatio < 0.10 THEN 'Low DTI'
        WHEN DebtToIncomeRatio < 0.30 THEN 'Medium DTI'
        ELSE 'High DTI'
    END
HAVING COUNT(*) >= 100
ORDER BY credit_score_group, dti_group
""").show(100, truncate=False)

print("Conclusion: CreditScoreRangeLower and DebtToIncomeRatio are retained because they capture different but complementary credit risk dimensions.")

cols_group_3 = [
    "CreditScoreRangeUpper",
    "OpenCreditLines",
    "OpenRevolvingAccounts"
]

df_reduced = safe_drop(df_reduced, cols_group_3)
df_reduced.createOrReplaceTempView("prosper_loan_reduced")
print_status("GROUP 3 - AFTER DROP", df_reduced)
record_feature_progress("After Group 3 - Credit History", df_reduced)


# GROUP 4: Debt and Credit Utilization
# Check credit utilization as a direct indicator of revolving credit pressure.
print("\n========== GROUP 4 - DEBT AND CREDIT UTILIZATION ==========")

spark.sql("""
SELECT
    CASE
        WHEN BankcardUtilization < 0.30 THEN 'Low Utilization'
        WHEN BankcardUtilization < 0.70 THEN 'Medium Utilization'
        ELSE 'High Utilization'
    END AS utilization_group,
    COUNT(*) AS total_loans,
    ROUND(AVG(BorrowerAPR), 4) AS avg_apr,
    ROUND(
        AVG(
            CASE
                WHEN LoanStatus IN ('Chargedoff', 'Defaulted') THEN 1
                WHEN LoanStatus = 'Completed' THEN 0
                ELSE NULL
            END
        ) * 100, 2
    ) AS bad_loan_rate_percent
FROM prosper_loan_reduced
WHERE BankcardUtilization IS NOT NULL
GROUP BY
    CASE
        WHEN BankcardUtilization < 0.30 THEN 'Low Utilization'
        WHEN BankcardUtilization < 0.70 THEN 'Medium Utilization'
        ELSE 'High Utilization'
    END
ORDER BY avg_apr DESC
""").show(truncate=False)

print("Conclusion: BankcardUtilization is retained because it reflects revolving credit pressure.")

cols_group_4 = [
    "AvailableBankcardCredit",
    "RevolvingCreditBalance",
    "OpenRevolvingMonthlyPayment"
]

df_reduced = safe_drop(df_reduced, cols_group_4)
df_reduced.createOrReplaceTempView("prosper_loan_reduced")
print_status("GROUP 4 - AFTER DROP", df_reduced)
record_feature_progress("After Group 4 - Debt and Credit Utilization", df_reduced)


# GROUP 5: Delinquency and Public Records
# Keep core delinquency history and remove weaker public-record indicators.
print("\n========== GROUP 5 - DELINQUENCY AND PUBLIC RECORDS ==========")

spark.sql("""
WITH base AS (
    SELECT
        CASE
            WHEN CurrentDelinquencies = 0 THEN 'No Current Delinquency'
            WHEN CurrentDelinquencies <= 2 THEN '1-2 Current Delinquencies'
            ELSE '3+ Current Delinquencies'
        END AS current_delinquency_group,

        CASE
            WHEN DelinquenciesLast7Years = 0 THEN 'No 7Y Delinquency'
            WHEN DelinquenciesLast7Years <= 2 THEN '1-2 in 7Y'
            WHEN DelinquenciesLast7Years <= 5 THEN '3-5 in 7Y'
            ELSE '5+ in 7Y'
        END AS delinquency_history_group,

        BorrowerAPR,
        LoanStatus,
        AmountDelinquent,
        PublicRecordsLast10Years,
        PublicRecordsLast12Months
    FROM prosper_loan_reduced
    WHERE CurrentDelinquencies IS NOT NULL
      AND DelinquenciesLast7Years IS NOT NULL
      AND BorrowerAPR IS NOT NULL
),
agg AS (
    SELECT
        current_delinquency_group,
        delinquency_history_group,
        COUNT(*) AS total_loans,

        ROUND(AVG(BorrowerAPR), 4) AS avg_apr,

        ROUND(
            AVG(
                CASE
                    WHEN LoanStatus IN ('Chargedoff', 'Defaulted') THEN 1.0
                    WHEN LoanStatus = 'Completed' THEN 0.0
                    ELSE NULL
                END
            ) * 100, 2
        ) AS bad_loan_rate_percent,

        ROUND(AVG(COALESCE(AmountDelinquent, 0)), 2) AS avg_amount_delinquent,
        ROUND(AVG(COALESCE(PublicRecordsLast10Years, 0)), 2) AS avg_public_records_10y,
        ROUND(AVG(COALESCE(PublicRecordsLast12Months, 0)), 2) AS avg_public_records_12m
    FROM base
    GROUP BY current_delinquency_group, delinquency_history_group
    HAVING COUNT(*) >= 100
)
SELECT
    *,
    DENSE_RANK() OVER (ORDER BY avg_apr DESC) AS apr_risk_rank,
    DENSE_RANK() OVER (
        ORDER BY COALESCE(bad_loan_rate_percent, -1) DESC
    ) AS default_risk_rank
FROM agg
ORDER BY apr_risk_rank, default_risk_rank
""").show(50, truncate=False)

print(
    "Conclusion: CurrentDelinquencies and DelinquenciesLast7Years are retained because "
    "they jointly capture current and historical delinquency risk. AmountDelinquent and "
    "PublicRecordsLast12Months are removed as weaker or overlapping delinquency-related signals."
)

cols_group_5 = [
    "AmountDelinquent",
    "PublicRecordsLast12Months"
]

df_reduced = safe_drop(df_reduced, cols_group_5)
df_reduced.createOrReplaceTempView("prosper_loan_reduced")
print_status("GROUP 5 - AFTER DROP", df_reduced)
record_feature_progress("After Group 5 - Delinquency and Public Records", df_reduced)


# GROUP 6: Previous Prosper Borrowing History
# Retain prior Prosper borrowing experience and late-payment behavior.
print("\n========== GROUP 6 - PREVIOUS PROSPER BORROWING HISTORY ==========")

spark.sql("""
WITH base AS (
    SELECT
        CASE
            WHEN COALESCE(TotalProsperLoans, 0) = 0 THEN 'No Previous Loan'
            WHEN COALESCE(TotalProsperLoans, 0) = 1 THEN '1 Previous Loan'
            ELSE '2+ Previous Loans'
        END AS prosper_loan_history,

        CASE
            WHEN COALESCE(ProsperPaymentsOneMonthPlusLate, 0) = 0
                THEN 'No 1M+ Late Payment'
            ELSE 'Has 1M+ Late Payment'
        END AS prosper_late_payment_group,

        BorrowerAPR,
        LoanStatus,
        TotalProsperPaymentsBilled,
        OnTimeProsperPayments,
        ProsperPrincipalBorrowed,
        ProsperPrincipalOutstanding
    FROM prosper_loan_reduced
    WHERE BorrowerAPR IS NOT NULL
),
agg AS (
    SELECT
        prosper_loan_history,
        prosper_late_payment_group,
        COUNT(*) AS total_loans,

        ROUND(AVG(BorrowerAPR), 4) AS avg_apr,

        ROUND(
            AVG(
                CASE
                    WHEN LoanStatus IN ('Chargedoff', 'Defaulted') THEN 1.0
                    WHEN LoanStatus = 'Completed' THEN 0.0
                    ELSE NULL
                END
            ) * 100, 2
        ) AS bad_loan_rate_percent,

        ROUND(AVG(COALESCE(OnTimeProsperPayments, 0)), 2) AS avg_on_time_payments,
        ROUND(AVG(COALESCE(TotalProsperPaymentsBilled, 0)), 2) AS avg_payments_billed,
        ROUND(AVG(COALESCE(ProsperPrincipalBorrowed, 0)), 2) AS avg_principal_borrowed,
        ROUND(AVG(COALESCE(ProsperPrincipalOutstanding, 0)), 2) AS avg_principal_outstanding
    FROM base
    GROUP BY prosper_loan_history, prosper_late_payment_group
    HAVING COUNT(*) >= 100
)
SELECT
    *,
    DENSE_RANK() OVER (ORDER BY avg_apr DESC) AS apr_risk_rank,
    DENSE_RANK() OVER (
        ORDER BY COALESCE(bad_loan_rate_percent, -1) DESC
    ) AS default_risk_rank
FROM agg
ORDER BY apr_risk_rank, default_risk_rank
""").show(50, truncate=False)

print(
    "Conclusion: TotalProsperLoans is retained to represent prior platform experience, "
    "and ProsperPaymentsOneMonthPlusLate is retained as a direct late-payment risk signal. "
    "Payment count and principal amount variables are removed because they mainly describe "
    "transaction volume and may overlap with prior borrowing history."
)

cols_group_6 = [
    "TotalProsperPaymentsBilled",
    "OnTimeProsperPayments",
    "ProsperPrincipalBorrowed",
    "ProsperPrincipalOutstanding"
]

df_reduced = safe_drop(df_reduced, cols_group_6)
df_reduced.createOrReplaceTempView("prosper_loan_reduced")
print_status("GROUP 6 - AFTER DROP", df_reduced)
record_feature_progress("After Group 6 - Previous Prosper Borrowing History", df_reduced)


# GROUP 7: Loan Structure and Funding
# Keep core loan size, term, and investor participation features.
print("\n========== GROUP 7 - LOAN STRUCTURE AND FUNDING CHECK ==========")

spark.sql("""
WITH base AS (
    SELECT
        CASE
            WHEN LoanOriginalAmount < 5000 THEN 'Small Loan'
            WHEN LoanOriginalAmount < 15000 THEN 'Medium Loan'
            ELSE 'Large Loan'
        END AS loan_size_group,

        Term,

        CASE
            WHEN Investors < 50 THEN 'Low Investor Interest'
            WHEN Investors < 200 THEN 'Medium Investor Interest'
            ELSE 'High Investor Interest'
        END AS investor_group,

        BorrowerAPR,
        LoanStatus,
        MonthlyLoanPayment,
        PercentFunded,
        Recommendations,
        InvestmentFromFriendsCount,
        InvestmentFromFriendsAmount
    FROM prosper_loan_reduced
    WHERE LoanOriginalAmount IS NOT NULL
      AND Term IS NOT NULL
      AND Investors IS NOT NULL
      AND BorrowerAPR IS NOT NULL
),
agg AS (
    SELECT
        loan_size_group,
        Term,
        investor_group,
        COUNT(*) AS total_loans,

        ROUND(AVG(BorrowerAPR), 4) AS avg_apr,

        ROUND(
            AVG(
                CASE
                    WHEN LoanStatus IN ('Chargedoff', 'Defaulted') THEN 1.0
                    WHEN LoanStatus = 'Completed' THEN 0.0
                    ELSE NULL
                END
            ) * 100, 2
        ) AS bad_loan_rate_percent,

        ROUND(AVG(MonthlyLoanPayment), 2) AS avg_monthly_payment,
        ROUND(AVG(PercentFunded), 4) AS avg_percent_funded,
        ROUND(AVG(Recommendations), 2) AS avg_recommendations,
        ROUND(AVG(InvestmentFromFriendsCount), 2) AS avg_friend_invest_count,
        ROUND(AVG(InvestmentFromFriendsAmount), 2) AS avg_friend_invest_amount
    FROM base
    GROUP BY loan_size_group, Term, investor_group
    HAVING COUNT(*) >= 100
)
SELECT
    *,
    DENSE_RANK() OVER (ORDER BY avg_apr DESC) AS apr_rank,
    DENSE_RANK() OVER (
        ORDER BY COALESCE(bad_loan_rate_percent, -1) DESC
    ) AS default_risk_rank
FROM agg
ORDER BY apr_rank, default_risk_rank
""").show(80, truncate=False)

print(
    "Conclusion: LoanOriginalAmount, Term, and Investors are retained because they represent "
    "loan size, contract structure, and market funding interest. MonthlyLoanPayment, "
    "Recommendations, InvestmentFromFriendsCount, and InvestmentFromFriendsAmount are removed "
    "because they are derivative, sparse, or provide limited additional business value."
)

cols_group_7 = [
    "MonthlyLoanPayment",
    "Recommendations",
    "InvestmentFromFriendsCount",
    "InvestmentFromFriendsAmount"
]

df_reduced = safe_drop(df_reduced, cols_group_7)
df_reduced.createOrReplaceTempView("prosper_loan_reduced")
print_status("GROUP 7 - AFTER DROP", df_reduced)
record_feature_progress("After Group 7 - Loan Structure and Funding", df_reduced)


# GROUP 8: Estimated Pricing Components
# Remove pricing-output variables while retaining EstimatedLoss as the core expected risk signal.
print("\n========== GROUP 8 - ESTIMATED LOSS AND PRICING LEAKAGE CHECK ==========")

spark.sql("""
WITH corr_check AS (
    SELECT
        ROUND(corr(BorrowerAPR, BorrowerRate), 4) AS corr_apr_borrower_rate,
        ROUND(corr(BorrowerAPR, LenderYield), 4) AS corr_apr_lender_yield,
        ROUND(corr(BorrowerAPR, EstimatedEffectiveYield), 4) AS corr_apr_effective_yield,
        ROUND(corr(BorrowerAPR, EstimatedReturn), 4) AS corr_apr_estimated_return,
        ROUND(corr(EstimatedLoss, EstimatedReturn), 4) AS corr_loss_return
    FROM prosper_loan_reduced
),
base AS (
    SELECT
        CASE
            WHEN EstimatedLoss < 0.05 THEN 'Low Expected Loss'
            WHEN EstimatedLoss < 0.10 THEN 'Medium Expected Loss'
            ELSE 'High Expected Loss'
        END AS expected_loss_group,

        BorrowerAPR,
        LoanStatus,
        EstimatedLoss,
        BorrowerRate,
        LenderYield,
        EstimatedEffectiveYield,
        EstimatedReturn
    FROM prosper_loan_reduced
    WHERE EstimatedLoss IS NOT NULL
      AND BorrowerAPR IS NOT NULL
),
agg AS (
    SELECT
        expected_loss_group,
        COUNT(*) AS total_loans,

        ROUND(AVG(BorrowerAPR), 4) AS avg_apr,
        ROUND(AVG(EstimatedLoss), 4) AS avg_estimated_loss,
        ROUND(AVG(BorrowerRate), 4) AS avg_borrower_rate,
        ROUND(AVG(LenderYield), 4) AS avg_lender_yield,
        ROUND(AVG(EstimatedEffectiveYield), 4) AS avg_effective_yield,
        ROUND(AVG(EstimatedReturn), 4) AS avg_estimated_return,

        ROUND(
            AVG(
                CASE
                    WHEN LoanStatus IN ('Chargedoff', 'Defaulted') THEN 1.0
                    WHEN LoanStatus = 'Completed' THEN 0.0
                    ELSE NULL
                END
            ) * 100, 2
        ) AS bad_loan_rate_percent
    FROM base
    GROUP BY expected_loss_group
)
SELECT
    agg.*,
    corr_check.corr_apr_borrower_rate,
    corr_check.corr_apr_lender_yield,
    corr_check.corr_apr_effective_yield,
    corr_check.corr_apr_estimated_return,
    corr_check.corr_loss_return,
    DENSE_RANK() OVER (ORDER BY avg_apr DESC) AS apr_rank,
    DENSE_RANK() OVER (
        ORDER BY COALESCE(bad_loan_rate_percent, -1) DESC
    ) AS default_risk_rank
FROM agg
CROSS JOIN corr_check
ORDER BY apr_rank, default_risk_rank
""").show(truncate=False)

print(
    "Conclusion: EstimatedLoss is retained because it represents expected credit risk. "
    "BorrowerRate, LenderYield, EstimatedEffectiveYield, and EstimatedReturn are removed "
    "because they are pricing-output variables that are highly related to BorrowerAPR and may "
    "introduce target leakage or redundant pricing information in the APR prediction task."
)

cols_group_8 = [
    "BorrowerRate",
    "LenderYield",
    "EstimatedEffectiveYield",
    "EstimatedReturn"
]

df_reduced = safe_drop(df_reduced, cols_group_8)
df_reduced.createOrReplaceTempView("prosper_loan_reduced")
print_status("GROUP 8 - AFTER PRICING COMPONENTS REDUCTION", df_reduced)
record_feature_progress("After Group 8 - Estimated Pricing Components", df_reduced)


# GROUP 9: Credit Capacity and Trade Structure
# Keep compact credit capacity signals and remove overlapping trade history fields.
print("\n========== GROUP 9 - DELINQUENCY AND CREDIT HISTORY RISK CHECK ==========")

spark.sql("""
SELECT
    CASE
        WHEN CurrentDelinquencies = 0 THEN 'No Current Delinquency'
        WHEN CurrentDelinquencies <= 2 THEN '1-2 Current Delinquencies'
        ELSE '3+ Current Delinquencies'
    END AS delinquency_group,

    CASE
        WHEN TotalTrades < 10 THEN 'Thin Credit History'
        WHEN TotalTrades < 30 THEN 'Medium Credit History'
        ELSE 'Strong Credit History'
    END AS credit_history_group,

    COUNT(*) AS total_loans,
    ROUND(AVG(BorrowerAPR), 4) AS avg_apr,

    ROUND(
        SUM(CASE
            WHEN LoanStatus IN ('Chargedoff', 'Defaulted') THEN 1
            ELSE 0
        END) / COUNT(*),
        4
    ) AS bad_loan_rate

FROM prosper_loan_reduced
WHERE CurrentDelinquencies IS NOT NULL
  AND TotalTrades IS NOT NULL
  AND BorrowerAPR IS NOT NULL
  AND LoanStatus IS NOT NULL
GROUP BY
    CASE
        WHEN CurrentDelinquencies = 0 THEN 'No Current Delinquency'
        WHEN CurrentDelinquencies <= 2 THEN '1-2 Current Delinquencies'
        ELSE '3+ Current Delinquencies'
    END,
    CASE
        WHEN TotalTrades < 10 THEN 'Thin Credit History'
        WHEN TotalTrades < 30 THEN 'Medium Credit History'
        ELSE 'Strong Credit History'
    END
ORDER BY bad_loan_rate DESC, avg_apr DESC
""").show(truncate=False)


spark.sql("""
SELECT
    ROUND(corr(CurrentCreditLines, TotalCreditLinespast7years), 4) AS corr_credit_history,
    ROUND(corr(CurrentCreditLines, TotalTrades), 4) AS corr_credit_trade,
    ROUND(corr(TotalTrades, `TradesNeverDelinquent (percentage)`), 4) AS corr_trade_quality
FROM prosper_loan_reduced
""").show(truncate=False)


cols_group_9 = [
    "TradesNeverDelinquent (percentage)",
    "PublicRecordsLast10Years",
    "TotalCreditLinespast7years"
]

df_reduced = safe_drop(df_reduced, cols_group_9)
df_reduced.createOrReplaceTempView("prosper_loan_reduced")
print_status("GROUP 9 - AFTER CREDIT CAPACITY REDUCTION", df_reduced)
record_feature_progress("After Group 9 - Credit Capacity and Trade Structure", df_reduced)

# GROUP 10: Borrower Profile and Demographic Features
# Remove high-cardinality borrower descriptors and keep home ownership as the compact profile signal.
print("\n========== GROUP 10 - BORROWER PROFILE AND DEMOGRAPHIC FEATURES ==========")

spark.sql("""
SELECT
    CASE
        WHEN IsBorrowerHomeowner = true THEN 'Homeowner'
        ELSE 'Non-Homeowner'
    END AS homeowner_group,
    COUNT(*) AS total_loans,
    ROUND(AVG(BorrowerAPR), 4) AS avg_apr
FROM prosper_loan_reduced
WHERE IsBorrowerHomeowner IS NOT NULL
GROUP BY
    CASE
        WHEN IsBorrowerHomeowner = true THEN 'Homeowner'
        ELSE 'Non-Homeowner'
    END
ORDER BY avg_apr DESC
""").show(truncate=False)

spark.sql("""
SELECT
    EmploymentStatus,
    COUNT(*) AS total_loans,
    ROUND(AVG(BorrowerAPR), 4) AS avg_apr
FROM prosper_loan_reduced
WHERE EmploymentStatus IS NOT NULL
GROUP BY EmploymentStatus
HAVING COUNT(*) > 500
ORDER BY avg_apr DESC
""").show(truncate=False)

spark.sql("""
SELECT
    CASE
        WHEN IsBorrowerHomeowner = true THEN 'Homeowner'
        ELSE 'Non-Homeowner'
    END AS homeowner_group,
    EmploymentStatus,
    COUNT(*) AS total_loans,
    ROUND(AVG(BorrowerAPR), 4) AS avg_apr
FROM prosper_loan_reduced
WHERE IsBorrowerHomeowner IS NOT NULL
  AND EmploymentStatus IS NOT NULL
GROUP BY
    CASE
        WHEN IsBorrowerHomeowner = true THEN 'Homeowner'
        ELSE 'Non-Homeowner'
    END,
    EmploymentStatus
HAVING COUNT(*) > 200
ORDER BY total_loans DESC
""").show(truncate=False)

spark.sql("""
SELECT
    BorrowerState,
    COUNT(*) AS total_loans,
    ROUND(AVG(BorrowerAPR), 4) AS avg_apr
FROM prosper_loan_reduced
WHERE BorrowerState IS NOT NULL
GROUP BY BorrowerState
HAVING COUNT(*) > 1000
ORDER BY total_loans DESC
LIMIT 15
""").show(truncate=False)

spark.sql("""
SELECT
    Occupation,
    COUNT(*) AS total_loans,
    ROUND(AVG(BorrowerAPR), 4) AS avg_apr
FROM prosper_loan_reduced
WHERE Occupation IS NOT NULL
GROUP BY Occupation
HAVING COUNT(*) > 500
ORDER BY total_loans DESC
LIMIT 15
""").show(truncate=False)

print("Conclusion: IsBorrowerHomeowner is retained, while BorrowerState, Occupation, and EmploymentStatus are removed to reduce weak or redundant profile information.")

cols_group_10 = [
    "BorrowerState",
    "Occupation",
    "EmploymentStatus"
]

df_reduced = safe_drop(df_reduced, cols_group_10)
df_reduced.createOrReplaceTempView("prosper_loan_reduced")
print_status("GROUP 10 - AFTER DROP", df_reduced)
record_feature_progress("After Group 10 - Borrower Profile", df_reduced)


# GROUP 11: Credit Inquiry and Credit Activity Features
# Keep recent inquiry and trade-depth signals, remove overlapping credit activity variables.
print("\n========== GROUP 11 - CREDIT INQUIRY AND CREDIT ACTIVITY FEATURES ==========")

spark.sql("""
WITH corr_check AS (
    SELECT
        ROUND(corr(InquiriesLast6Months, TotalInquiries), 4) AS corr_recent_total_inquiries,
        ROUND(corr(CurrentCreditLines, TotalTrades), 4) AS corr_current_lines_total_trades,
        ROUND(corr(TradesOpenedLast6Months, InquiriesLast6Months), 4) AS corr_recent_trades_inquiries
    FROM prosper_loan_reduced
),
base AS (
    SELECT
        CASE
            WHEN InquiriesLast6Months = 0 THEN 'No Recent Inquiry'
            WHEN InquiriesLast6Months <= 2 THEN '1-2 Recent Inquiries'
            ELSE '3+ Recent Inquiries'
        END AS recent_inquiry_group,

        CASE
            WHEN TotalTrades < 10 THEN 'Thin Credit History'
            WHEN TotalTrades < 30 THEN 'Medium Credit History'
            ELSE 'Long Credit History'
        END AS trade_history_group,

        BorrowerAPR,
        LoanStatus,
        TotalInquiries,
        TradesOpenedLast6Months,
        CurrentCreditLines,
        TotalTrades
    FROM prosper_loan_reduced
    WHERE InquiriesLast6Months IS NOT NULL
      AND TotalTrades IS NOT NULL
      AND BorrowerAPR IS NOT NULL
),
agg AS (
    SELECT
        recent_inquiry_group,
        trade_history_group,
        COUNT(*) AS total_loans,

        ROUND(AVG(BorrowerAPR), 4) AS avg_apr,

        ROUND(
            AVG(
                CASE
                    WHEN LoanStatus IN ('Chargedoff', 'Defaulted') THEN 1.0
                    WHEN LoanStatus = 'Completed' THEN 0.0
                    ELSE NULL
                END
            ) * 100, 2
        ) AS bad_loan_rate_percent,

        ROUND(AVG(TotalInquiries), 2) AS avg_total_inquiries,
        ROUND(AVG(TradesOpenedLast6Months), 2) AS avg_recent_trades_opened,
        ROUND(AVG(CurrentCreditLines), 2) AS avg_current_credit_lines
    FROM base
    GROUP BY recent_inquiry_group, trade_history_group
    HAVING COUNT(*) >= 100
)
SELECT
    agg.*,
    corr_check.corr_recent_total_inquiries,
    corr_check.corr_current_lines_total_trades,
    corr_check.corr_recent_trades_inquiries,
    DENSE_RANK() OVER (ORDER BY avg_apr DESC) AS apr_rank,
    DENSE_RANK() OVER (
        ORDER BY COALESCE(bad_loan_rate_percent, -1) DESC
    ) AS default_risk_rank
FROM agg
CROSS JOIN corr_check
ORDER BY apr_rank, default_risk_rank
""").show(80, truncate=False)

print(
    "Conclusion: InquiriesLast6Months and TotalTrades are retained because they represent "
    "recent credit-seeking behavior and the depth of credit history. TotalInquiries, "
    "CurrentCreditLines, and TradesOpenedLast6Months are removed because they overlap with "
    "the retained inquiry and trade-history signals."
)

cols_group_11 = [
    "TotalInquiries",
    "CurrentCreditLines",
    "TradesOpenedLast6Months"
]

df_reduced = safe_drop(df_reduced, cols_group_11)
df_reduced.createOrReplaceTempView("prosper_loan_reduced")
print_status("GROUP 11 - AFTER DROP", df_reduced)
record_feature_progress("After Group 11 - Credit Inquiry and Activity", df_reduced)


# GROUP 12: Temporal and Listing Information Redundancy Analysis
# Retain Investors as the stronger funding signal and remove weak descriptive listing/time variables.
print("\n========== GROUP 12 - TEMPORAL AND LISTING INFORMATION REDUNDANCY ANALYSIS ==========")

spark.sql("""
WITH base AS (
    SELECT
        YEAR(LoanOriginationDate) AS orig_year,
        QUARTER(LoanOriginationDate) AS orig_quarter,

        `ListingCategory (numeric)` AS listing_category,

        CASE
            WHEN PercentFunded >= 1.0 THEN 'Fully Funded'
            WHEN PercentFunded >= 0.8 THEN 'Mostly Funded'
            ELSE 'Low or Medium Funded'
        END AS funding_group,

        BorrowerAPR,
        LoanStatus,
        ScorexChangeAtTimeOfListing,
        PercentFunded,
        Investors
    FROM prosper_loan_reduced
    WHERE LoanOriginationDate IS NOT NULL
      AND `ListingCategory (numeric)` IS NOT NULL
      AND PercentFunded IS NOT NULL
      AND BorrowerAPR IS NOT NULL
),
agg AS (
    SELECT
        orig_year,
        orig_quarter,
        listing_category,
        funding_group,
        COUNT(*) AS total_loans,

        ROUND(AVG(BorrowerAPR), 4) AS avg_apr,

        ROUND(
            AVG(
                CASE
                    WHEN LoanStatus IN ('Chargedoff', 'Defaulted') THEN 1.0
                    WHEN LoanStatus = 'Completed' THEN 0.0
                    ELSE NULL
                END
            ) * 100, 2
        ) AS bad_loan_rate_percent,

        ROUND(AVG(PercentFunded), 4) AS avg_percent_funded,
        ROUND(AVG(Investors), 2) AS avg_investors,
        ROUND(AVG(ScorexChangeAtTimeOfListing), 2) AS avg_score_change,

        ROUND(
            AVG(
                CASE
                    WHEN ScorexChangeAtTimeOfListing < 0 THEN 1.0
                    ELSE 0.0
                END
            ) * 100, 2
        ) AS score_decrease_percent
    FROM base
    GROUP BY orig_year, orig_quarter, listing_category, funding_group
    HAVING COUNT(*) >= 100
),
time_check AS (
    SELECT
        *,
        LAG(avg_apr) OVER (
            PARTITION BY listing_category, funding_group
            ORDER BY orig_year, orig_quarter
        ) AS prev_quarter_avg_apr
    FROM agg
)
SELECT
    *,
    ROUND(avg_apr - prev_quarter_avg_apr, 4) AS apr_change_vs_prev_quarter,
    DENSE_RANK() OVER (
        PARTITION BY orig_year, orig_quarter
        ORDER BY avg_apr DESC
    ) AS quarterly_apr_rank
FROM time_check
ORDER BY orig_year, orig_quarter, quarterly_apr_rank
""").show(100, truncate=False)

spark.sql("""
SELECT
    ROUND(corr(PercentFunded, Investors), 4) AS corr_percent_funded_investors
FROM prosper_loan_reduced
WHERE PercentFunded IS NOT NULL
  AND Investors IS NOT NULL
""").show(truncate=False)

print(
    "Conclusion: Investors is retained as the stronger funding-market signal. "
    "ListingCategory (numeric), PercentFunded, LoanOriginationQuarter, and "
    "ScorexChangeAtTimeOfListing are removed because they are weak, unstable, or mostly "
    "descriptive listing and temporal variables after stronger credit-risk features are retained."
)

cols_group_12 = [
    "ListingCategory (numeric)",
    "PercentFunded",
    "LoanOriginationQuarter",
    "ScorexChangeAtTimeOfListing"
]

df_reduced = safe_drop(df_reduced, cols_group_12)
df_reduced.createOrReplaceTempView("prosper_loan_reduced")
print_status("GROUP 12 - AFTER DROP", df_reduced)
record_feature_progress("Final Reduced Dataset", df_reduced)


print_feature_reduction_progress()
print("\nConclusion:")
print(
    "The feature analysis workflow reduced the dataset from the original "
    f"{feature_progress[0][1]} attributes to the final reduced feature set."
)
print(
    "This step helps remove identifiers, redundant variables, high-leakage attributes, "
    "and weak business-value features before EDA, preprocessing, and machine learning."
)


print("\n========== REMAINING COLUMNS ==========")
for index, column_name in enumerate(df_reduced.columns, start=1):
    print(f"{index:02d}. {column_name}")

print("\n========== SAVE REDUCED DATASET TO HDFS ==========")
df_reduced.write.mode("overwrite").parquet(HDFS_OUTPUT_PATH)

print(f"Saved reduced dataset to: {HDFS_OUTPUT_PATH}")
print(f"Remaining rows: {df_reduced.count()}")
print(f"Remaining columns: {len(df_reduced.columns)}")
print("Domain feature reduction completed successfully.")

spark.stop()
