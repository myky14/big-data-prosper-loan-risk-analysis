# Spark Installation Steps

## 1. Mục tiêu

* Cài đặt Apache Spark
* Kết nối Spark với Hadoop HDFS
* Đọc dữ liệu Prosper Loan từ HDFS
* Chạy thử Spark SQL và Spark MLlib

---

## 2. Môi trường thực hiện

### Thông tin hệ thống

* Hệ điều hành:
* Java Version:
* Hadoop Version:
* Spark Version:

Screenshot:

* [ ] System Information
* [ ] Spark Version

---

## 3. Cài đặt Apache Spark

### Tải Spark

Nguồn tải:

* URL:

### Giải nén Spark

```bash
tar -xzf spark-*.tgz
```

Screenshot:

* [ ] Spark Folder

---

## 4. Cấu hình biến môi trường

### Cập nhật .bashrc

Ví dụ:

```bash
export SPARK_HOME=...
export PATH=$PATH:$SPARK_HOME/bin
```

Screenshot:

* [ ] Environment Variables

---

## 5. Kiểm tra Spark

### Kiểm tra phiên bản

```bash
spark-submit --version
```

Screenshot:

* [ ] Spark Version

---

## 6. Khởi động PySpark

### Chạy PySpark

```bash
pyspark
```

### Tạo SparkSession

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("ProsperLoanProject") \
    .getOrCreate()
```

Screenshot:

* [ ] PySpark Running

---

## 7. Kết nối Spark với HDFS

### Đọc dữ liệu từ HDFS

Ví dụ:

```python
df = spark.read.csv(
    "hdfs:///bigdata/prosper_loan/raw/prosperLoanData.csv",
    header=True,
    inferSchema=True
)
```

Screenshot:

* [ ] Dataset Loaded From HDFS

---

## 8. Kiểm tra dữ liệu

### Xem Schema

```python
df.printSchema()
```

### Xem dữ liệu mẫu

```python
df.show(5)
```

### Kiểm tra số dòng và số cột

```python
print(df.count())
print(len(df.columns))
```

Screenshot:

* [ ] Schema
* [ ] Sample Data
* [ ] Record Count

---

## 9. Thực thi Spark SQL

### Tạo Temporary View

```python
df.createOrReplaceTempView("prosper_loan")
```

### Chạy truy vấn thử

```python
spark.sql("""
SELECT COUNT(*) AS total_records
FROM prosper_loan
""").show()
```

Screenshot:

* [ ] Spark SQL Result

---

## 10. Kiểm tra Spark Web UI

### Spark UI

URL:

http://localhost:4040

Screenshot:

* [ ] Spark Jobs
* [ ] Spark Executors

---

## 11. Kết quả đạt được

* Spark cài đặt thành công
* Spark kết nối được HDFS
* Dataset được đọc thành công từ HDFS
* Spark SQL hoạt động bình thường
* Sẵn sàng cho bước phân tích dữ liệu và Machine Learning
