import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, when

DATA_PATH = "data/raw/prosperLoanData.csv"

spark = SparkSession.builder \
    .appName("Check Prosper Loan Dataset") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("\n========== DATASET VALIDATION ==========")

if not os.path.exists(DATA_PATH):
    print(f"File not found: {DATA_PATH}")
    spark.stop()
    exit()

df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)

row_count = df.count()
column_count = len(df.columns)

print(f"Dataset name: Prosper Loan Dataset")
print(f"Source: Kaggle")
print(f"Number of rows: {row_count}")
print(f"Number of columns: {column_count}")
print(f"Rows > 100000: {row_count > 100000}")
print(f"Columns > 10: {column_count > 10}")

print("\n========== SELECTED SCHEMA ==========")
selected_columns = [
    "ListingKey",
    "LoanStatus",
    "Term",
    "BorrowerAPR",
    "BorrowerRate",
    "ProsperScore",
    "CreditScoreRangeLower",
    "CreditScoreRangeUpper",
    "DebtToIncomeRatio",
    "IncomeRange",
    "EmploymentStatus",
    "Occupation",
    "LoanOriginalAmount",
    "MonthlyLoanPayment",
    "Investors"
]

df.select(selected_columns).printSchema()

print("\n========== SAMPLE DATA ==========")
df.select(selected_columns).show(5, truncate=False)

print("\n========== TOP 15 MISSING VALUE COLUMNS ==========")
missing_row = df.select([
    spark_sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in df.columns
]).collect()[0].asDict()

missing_list = sorted(
    missing_row.items(),
    key=lambda x: x[1],
    reverse=True
)

for column, missing_count in missing_list[:15]:
    missing_percent = (missing_count / row_count) * 100
    print(f"{column}: {missing_count} missing values ({missing_percent:.2f}%)")

print("\n========== CONCLUSION ==========")
if row_count > 100000 and column_count > 10:
    print("The dataset satisfies the Big Data project requirement.")
else:
    print("The dataset does not satisfy the Big Data project requirement.")

spark.stop()