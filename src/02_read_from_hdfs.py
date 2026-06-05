from pyspark.sql import SparkSession

HDFS_PATH = "hdfs:///bigdata/prosper_loan/raw/prosperLoanData.csv"

spark = SparkSession.builder \
    .appName("Read Prosper Loan From HDFS") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("\n========== READ DATA FROM HDFS ==========")

print(f"HDFS Path: {HDFS_PATH}")

df = spark.read.csv(
    HDFS_PATH,
    header=True,
    inferSchema=True
)

row_count = df.count()
column_count = len(df.columns)

print("\n========== DATASET SUMMARY ==========")
print(f"Number of rows: {row_count}")
print(f"Number of columns: {column_count}")

print("\n========== SELECTED SCHEMA ==========")
df.printSchema()

print("\n========== SAMPLE DATA FROM HDFS ==========")
df.select(
    "LoanStatus",
    "BorrowerAPR",
    "ProsperScore",
    "CreditScoreRangeLower",
    "DebtToIncomeRatio",
    "IncomeRange",
    "Occupation",
    "EmploymentStatus",
    "LoanOriginalAmount",
    "MonthlyLoanPayment"
).show(10, truncate=False)

print("\n========== CONCLUSION ==========")
print("Spark successfully read Prosper Loan Dataset directly from HDFS.")

spark.stop()