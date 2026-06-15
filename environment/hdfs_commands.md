# HDFS Commands Log

## Mục tiêu

Ghi lại các lệnh HDFS đã sử dụng trong quá trình lưu trữ và quản lý Prosper Loan Dataset trên Hadoop Distributed File System (HDFS).

---
## 1. Kiểm tra HDFS
### 1.1. Tạo thư mục
Tạo thư mục lưu trữ dữ liệu Prosper Loan Dataset trên HDFS:
```bash
hdfs dfs -mkdir -p /bigdata/prosper_loan/raw
```

### 1.2. Kiểm tra cấu trúc thư mục HDFS
Chạy lệnh sau để kiểm tra thư mục đã được tạo:
```bash
hdfs dfs -ls /
```

<img width="1426" height="472" alt="image" src="https://github.com/user-attachments/assets/44a7f8be-6a37-4a00-8e3a-03dd45f50db0" />

---

## 2. Upload Dataset lên HDFS

### 2.1.Upload file
Upload file Prosper Loan Dataset từ máy local lên HDFS:
```bash
hdfs dfs -put prosperLoanData.csv /bigdata/prosper_loan/raw/
```

### 2.2. Kiểm tra file
Chạy lệnh sau để kiểm tra file dữ liệu đã được upload thành công:
```bash
hdfs dfs -ls /bigdata/prosper_loan/raw/
```

<img width="1430" height="531" alt="image" src="https://github.com/user-attachments/assets/ad692302-8816-450b-aae6-31ee5daf83a9" />

---

# 3. Các lệnh sử dụng trong quá trình thực hiện

| Lệnh            | Mục đích             |
| --------------- | -------------------- |
| hdfs dfs -ls    | Liệt kê file/thư mục |
| hdfs dfs -mkdir | Tạo thư mục          |
| hdfs dfs -put   | Upload file          |
| hdfs dfs -cat   | Xem nội dung         |
| hdfs dfs -du    | Kiểm tra dung lượng  |
| hdfs dfs -stat  | Xem thông tin file   |

---

# Kết luận

* HDFS hoạt động bình thường.
* Dataset Prosper Loan được lưu thành công trên HDFS.
* Dữ liệu sẵn sàng cho bước xử lý bằng Apache Spark.
