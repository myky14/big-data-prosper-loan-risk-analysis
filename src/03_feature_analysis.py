from pyspark.sql import SparkSession


HDFS_INPUT_PATH = "hdfs://localhost:9000/bigdata/prosper_loan/raw/prosperLoanData.csv"
HDFS_OUTPUT_PATH = "hdfs://localhost:9000/bigdata/prosper_loan/processed/prosper_loan_reduced"


spark = (
    SparkSession.builder
    .appName("Prosper Loan Domain Feature Reduction")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


def print_status(step_name, dataframe):
    print(f"\n========== {step_name} ==========")
    print(f"Rows: {dataframe.count()}")
    print(f"Columns: {len(dataframe.columns)}")


def safe_drop(dataframe, columns_to_drop):
    existing_columns = [col_name for col_name in columns_to_drop if col_name in dataframe.columns]
    missing_columns = [col_name for col_name in columns_to_drop if col_name not in dataframe.columns]

    if missing_columns:
        print("Missing columns skipped:", missing_columns)

    if not existing_columns:
        return dataframe

    print("Dropped columns:", existing_columns)
    return dataframe.drop(*existing_columns)


print("\n========== READ RAW DATA FROM HDFS ==========")

df = spark.read.csv(
    HDFS_INPUT_PATH,
    header=True,
    inferSchema=True
)

print_status("RAW DATASET", df)

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

df_reduced = safe_drop(df_reduced, cols_group_0)
df_reduced.createOrReplaceTempView("prosper_loan_reduced")
print_status("GROUP 0 - AFTER INITIAL DOMAIN FILTERING", df_reduced)


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


# GROUP 5: Delinquency and Public Records
# Keep core delinquency history and remove weaker public-record indicators.
print("\n========== GROUP 5 - DELINQUENCY AND PUBLIC RECORDS ==========")

spark.sql("""
SELECT
    CASE
        WHEN CurrentDelinquencies = 0 THEN 'No Delinquency'
        WHEN CurrentDelinquencies <= 2 THEN '1-2 Delinquencies'
        ELSE '3+ Delinquencies'
    END AS delinquency_group,
    COUNT(*) AS total_loans,
    ROUND(AVG(BorrowerAPR), 4) AS avg_apr
FROM prosper_loan_reduced
WHERE CurrentDelinquencies IS NOT NULL
GROUP BY
    CASE
        WHEN CurrentDelinquencies = 0 THEN 'No Delinquency'
        WHEN CurrentDelinquencies <= 2 THEN '1-2 Delinquencies'
        ELSE '3+ Delinquencies'
    END
ORDER BY avg_apr DESC
""").show(truncate=False)

print("Conclusion: CurrentDelinquencies is retained as a concise delinquency signal.")

cols_group_5 = [
    "AmountDelinquent",
    "PublicRecordsLast12Months"
]

df_reduced = safe_drop(df_reduced, cols_group_5)
df_reduced.createOrReplaceTempView("prosper_loan_reduced")
print_status("GROUP 5 - AFTER DROP", df_reduced)


# GROUP 6: Previous Prosper Borrowing History
# Retain prior Prosper borrowing experience and late-payment behavior.
print("\n========== GROUP 6 - PREVIOUS PROSPER BORROWING HISTORY ==========")

spark.sql("""
SELECT
    CASE
        WHEN TotalProsperLoans = 0 THEN 'No Previous Loan'
        WHEN TotalProsperLoans = 1 THEN '1 Loan'
        ELSE '2+ Loans'
    END AS prosper_history_group,
    COUNT(*) AS total_loans,
    ROUND(AVG(BorrowerAPR), 4) AS avg_apr
FROM prosper_loan_reduced
WHERE TotalProsperLoans IS NOT NULL
GROUP BY
    CASE
        WHEN TotalProsperLoans = 0 THEN 'No Previous Loan'
        WHEN TotalProsperLoans = 1 THEN '1 Loan'
        ELSE '2+ Loans'
    END
ORDER BY avg_apr DESC
""").show(truncate=False)

print("Conclusion: TotalProsperLoans is retained to represent prior platform experience.")

cols_group_6 = [
    "TotalProsperPaymentsBilled",
    "OnTimeProsperPayments",
    "ProsperPrincipalBorrowed",
    "ProsperPrincipalOutstanding"
]

df_reduced = safe_drop(df_reduced, cols_group_6)
df_reduced.createOrReplaceTempView("prosper_loan_reduced")
print_status("GROUP 6 - AFTER DROP", df_reduced)


# GROUP 7: Loan Structure and Funding
# Keep core loan size, term, and investor participation features.
print("\n========== GROUP 7 - LOAN STRUCTURE AND FUNDING ==========")

spark.sql("""
SELECT
    Term,
    COUNT(*) AS total_loans,
    ROUND(AVG(BorrowerAPR), 4) AS avg_apr
FROM prosper_loan_reduced
WHERE Term IS NOT NULL
GROUP BY Term
ORDER BY Term
""").show(truncate=False)

print("Conclusion: Term is retained because loan structure affects pricing.")

cols_group_7 = [
    "MonthlyLoanPayment",
    "Recommendations",
    "InvestmentFromFriendsCount",
    "InvestmentFromFriendsAmount"
]

df_reduced = safe_drop(df_reduced, cols_group_7)
df_reduced.createOrReplaceTempView("prosper_loan_reduced")
print_status("GROUP 7 - AFTER DROP", df_reduced)


# GROUP 8: Estimated Pricing Components
# Remove highly related pricing outputs while keeping core pricing variables.
print("\n========== GROUP 8 - ESTIMATED PRICING COMPONENTS ==========")

spark.sql("""
SELECT
    CASE
        WHEN BorrowerRate < 0.10 THEN 'Low Rate'
        WHEN BorrowerRate < 0.20 THEN 'Medium Rate'
        ELSE 'High Rate'
    END AS rate_group,
    COUNT(*) AS total_loans,
    ROUND(AVG(BorrowerAPR), 4) AS avg_apr
FROM prosper_loan_reduced
WHERE BorrowerRate IS NOT NULL
GROUP BY
    CASE
        WHEN BorrowerRate < 0.10 THEN 'Low Rate'
        WHEN BorrowerRate < 0.20 THEN 'Medium Rate'
        ELSE 'High Rate'
    END
ORDER BY avg_apr DESC
""").show(truncate=False)

print("Conclusion: BorrowerRate is retained and overlapping yield variables are removed.")

cols_group_8 = [
    "LenderYield",
    "EstimatedEffectiveYield",
    "EstimatedReturn"
]

df_reduced = safe_drop(df_reduced, cols_group_8)
df_reduced.createOrReplaceTempView("prosper_loan_reduced")
print_status("GROUP 8 - AFTER DROP", df_reduced)


# GROUP 9: Credit Capacity and Trade Structure
# Keep compact credit capacity signals and remove overlapping trade history fields.
print("\n========== GROUP 9 - CREDIT CAPACITY AND TRADE STRUCTURE ==========")

spark.sql("""
SELECT
    CASE
        WHEN TotalTrades < 10 THEN 'Low Trade Count'
        WHEN TotalTrades < 25 THEN 'Medium Trade Count'
        ELSE 'High Trade Count'
    END AS trade_count_group,
    COUNT(*) AS total_loans,
    ROUND(AVG(BorrowerAPR), 4) AS avg_apr
FROM prosper_loan_reduced
WHERE TotalTrades IS NOT NULL
GROUP BY
    CASE
        WHEN TotalTrades < 10 THEN 'Low Trade Count'
        WHEN TotalTrades < 25 THEN 'Medium Trade Count'
        ELSE 'High Trade Count'
    END
ORDER BY avg_apr DESC
""").show(truncate=False)

print("Conclusion: TotalTrades is retained as a compact credit history depth signal.")

cols_group_9 = [
    "TradesNeverDelinquent (percentage)",
    "PublicRecordsLast10Years",
    "TotalCreditLinespast7years"
]

df_reduced = safe_drop(df_reduced, cols_group_9)
df_reduced.createOrReplaceTempView("prosper_loan_reduced")
print_status("GROUP 9 - AFTER DROP", df_reduced)


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


# GROUP 11: Credit Inquiry and Credit Activity Features
# Keep recent inquiry and trade-depth signals, remove overlapping credit activity variables.
print("\n========== GROUP 11 - CREDIT INQUIRY AND CREDIT ACTIVITY FEATURES ==========")

spark.sql("""
SELECT
    CASE
        WHEN InquiriesLast6Months = 0 THEN 'No Inquiry'
        WHEN InquiriesLast6Months <= 2 THEN '1-2 Inquiries'
        ELSE '3+ Inquiries'
    END AS inquiry_group,
    COUNT(*) AS total_loans,
    ROUND(AVG(BorrowerAPR), 4) AS avg_apr
FROM prosper_loan_reduced
WHERE InquiriesLast6Months IS NOT NULL
GROUP BY
    CASE
        WHEN InquiriesLast6Months = 0 THEN 'No Inquiry'
        WHEN InquiriesLast6Months <= 2 THEN '1-2 Inquiries'
        ELSE '3+ Inquiries'
    END
ORDER BY avg_apr DESC
""").show(truncate=False)

spark.sql("""
SELECT
    corr(InquiriesLast6Months, TotalInquiries) AS corr_recent_total_inquiries,
    corr(CurrentCreditLines, TotalTrades) AS corr_current_credit_lines_total_trades,
    corr(TradesOpenedLast6Months, InquiriesLast6Months) AS corr_recent_trades_recent_inquiries
FROM prosper_loan_reduced
""").show(truncate=False)

print("Conclusion: InquiriesLast6Months and TotalTrades are retained, while TotalInquiries, CurrentCreditLines, and TradesOpenedLast6Months are removed as overlapping activity signals.")

cols_group_11 = [
    "TotalInquiries",
    "CurrentCreditLines",
    "TradesOpenedLast6Months"
]

df_reduced = safe_drop(df_reduced, cols_group_11)
df_reduced.createOrReplaceTempView("prosper_loan_reduced")
print_status("GROUP 11 - AFTER DROP", df_reduced)


# GROUP 12: Temporal and Listing Information Redundancy Analysis
# Retain Investors as the stronger funding signal and remove weak descriptive listing/time variables.
print("\n========== GROUP 12 - TEMPORAL AND LISTING INFORMATION REDUNDANCY ANALYSIS ==========")

spark.sql("""
SELECT
    `ListingCategory (numeric)` AS listing_category,
    COUNT(*) AS total_loans,
    ROUND(AVG(BorrowerAPR), 4) AS avg_apr
FROM prosper_loan_reduced
WHERE `ListingCategory (numeric)` IS NOT NULL
GROUP BY `ListingCategory (numeric)`
ORDER BY avg_apr DESC
""").show(100, truncate=False)

spark.sql("""
SELECT
    corr(PercentFunded, Investors) AS corr_percent_funded_investors
FROM prosper_loan_reduced
WHERE PercentFunded IS NOT NULL
""").show(truncate=False)

spark.sql("""
SELECT
    LoanOriginationQuarter,
    COUNT(*) AS total_loans,
    ROUND(AVG(BorrowerAPR), 4) AS avg_apr
FROM prosper_loan_reduced
WHERE LoanOriginationQuarter IS NOT NULL
GROUP BY LoanOriginationQuarter
ORDER BY LoanOriginationQuarter
""").show(100, truncate=False)

spark.sql("""
SELECT
    CASE
        WHEN ScorexChangeAtTimeOfListing < 0 THEN 'Negative'
        WHEN ScorexChangeAtTimeOfListing > 0 THEN 'Positive'
        ELSE 'No Change'
    END AS score_change_group,
    COUNT(*) AS total_loans,
    ROUND(AVG(BorrowerAPR), 4) AS avg_apr
FROM prosper_loan_reduced
WHERE ScorexChangeAtTimeOfListing IS NOT NULL
GROUP BY
    CASE
        WHEN ScorexChangeAtTimeOfListing < 0 THEN 'Negative'
        WHEN ScorexChangeAtTimeOfListing > 0 THEN 'Positive'
        ELSE 'No Change'
    END
ORDER BY avg_apr DESC
""").show(truncate=False)

print("Conclusion: Investors is retained, while ListingCategory (numeric), PercentFunded, LoanOriginationQuarter, and ScorexChangeAtTimeOfListing are removed as weak listing or temporal descriptors.")

cols_group_12 = [
    "ListingCategory (numeric)",
    "PercentFunded",
    "LoanOriginationQuarter",
    "ScorexChangeAtTimeOfListing"
]

df_reduced = safe_drop(df_reduced, cols_group_12)
df_reduced.createOrReplaceTempView("prosper_loan_reduced")
print_status("GROUP 12 - AFTER DROP", df_reduced)


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
