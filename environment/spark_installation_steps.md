# Spark Installation Steps

## 1. Mục tiêu

Phần này trình bày các bước cài đặt Apache Spark, cấu hình biến môi trường, kết nối Spark với Hadoop HDFS và kiểm tra Spark hoạt động trên dữ liệu Prosper Loan.

Mục tiêu thực hiện gồm:

- Cài đặt Apache Spark
- Cấu hình Spark trên môi trường Windows
- Kết nối Spark với Hadoop HDFS
- Đọc dữ liệu Prosper Loan từ HDFS
- Chạy thử Spark SQL để kiểm tra hệ thống
- Chuẩn bị môi trường cho Spark MLlib

---

## 2. Môi trường thực hiện

### Thông tin hệ thống

- Hệ điều hành: Microsoft Windows 10 Pro
- Java Version: OpenJDK 8
- Hadoop Version: Apache Hadoop 3.4.1
- Spark Version: Apache Spark
- Công cụ: Command Prompt/PowerShell, PySpark, Jupyter Notebook

<img width="1259" height="1004" alt="image" src="https://github.com/user-attachments/assets/467ed4b4-200e-4a5d-9aa0-f3b41f4ff0ed" />
<img width="716" height="469" alt="image" src="https://github.com/user-attachments/assets/1f6fd890-5938-4150-9ca3-28c153aef1db" />

---

## 3. Cài đặt Apache Spark

### 3.1. Tải Spark

Nhóm tải Apache Spark từ trang chính thức của Apache Spark.

Thông tin lựa chọn:

- Package type: Pre-built for Apache Hadoop
- File tải về: `spark-*.tgz`

Nguồn tải:
```text
https://spark.apache.org/downloads.html
```

### 3.2. Giải nén Spark
Sau khi tải về, nhóm giải nén Spark vào thư mục cài đặt trên máy cá nhân.
```bash
tar -xzf spark-*.tgz
```

<img width="545" height="479" alt="image" src="https://github.com/user-attachments/assets/a0786a11-1fdc-4815-a061-6009bcc3d97d" />

---

## 4. Cấu hình biến môi trường

Sau khi giải nén Spark, nhóm cấu hình biến môi trường để hệ điều hành có thể nhận diện và thực thi các lệnh Spark từ terminal.

### 4.1. Cấu hình SPARK_HOME
Tạo biến môi trường hệ thống:
```text
Variable name: SPARK_HOME
Variable value: D:\BIGDATA_THM\SPARK\spark-3.5.8-bin-hadoop3
```
<img width="1170" height="655" alt="image" src="https://github.com/user-attachments/assets/a4edfabd-af3c-4e30-9f6c-9b6762b20997" />

### 4.2. Cấu hình Path

Thêm đường dẫn sau vào biến Path:
```text
%SPARK_HOME%\bin
```
Việc cấu hình này giúp người dùng có thể chạy các lệnh như spark-submit, spark-shell và pyspark trực tiếp từ Command Prompt/PowerShell.
<img width="831" height="482" alt="image" src="https://github.com/user-attachments/assets/65280bfe-9705-4535-b2e1-2a598c1e21e0" />

## 5. Kiểm tra Spark

### Kiểm tra phiên bản
Sau khi cấu hình biến môi trường, nhóm mở Command Prompt/PowerShell mới và chạy lệnh:

```bash
spark-submit --version
```

Kết quả hiển thị thông tin phiên bản Spark, cho thấy Spark đã được cài đặt và cấu hình thành công
<img width="715" height="321" alt="image" src="https://github.com/user-attachments/assets/25d17670-83e1-42d4-8dfc-50a07e766ee2" />

## 6. Khởi động PySpark

### 6.1. Chạy PySpark

Để kiểm tra Spark có thể chạy trong môi trường Python, nhóm khởi động PySpark bằng lệnh:
```bash
pyspark
```
Nếu PySpark khởi động thành công và hiển thị SparkSession, điều này cho thấy Spark có thể hoạt động với Python.

<img width="954" height="359" alt="image" src="https://github.com/user-attachments/assets/2e5206cc-96c7-40e0-b2c5-a07fe7f44823" />


### Tạo SparkSession

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("ProsperLoanProject") \
    .getOrCreate()
```
<img width="1200" height="176" alt="image" src="https://github.com/user-attachments/assets/ae53d6b7-8c0e-4155-b335-67ebe57b4906" />

---
## 7. Kết quả đạt được
* Spark cài đặt thành công
* Spark kết nối được HDFS
* Dataset được đọc thành công từ HDFS
* Spark SQL hoạt động bình thường
* Sẵn sàng cho bước phân tích dữ liệu và Machine Learning
