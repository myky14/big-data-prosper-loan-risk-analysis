# HDFS Commands Log

## Mục tiêu

Ghi lại các lệnh HDFS đã sử dụng trong quá trình lưu trữ và quản lý Prosper Loan Dataset trên Hadoop Distributed File System (HDFS).

---

# 1. Kiểm tra HDFS

### Kiểm tra thư mục gốc

```bash
hdfs dfs -ls /
```

Kết quả:

* Mô tả ngắn kết quả trả về.

📸 Screenshot:

* [ ] Terminal hiển thị danh sách thư mục HDFS

---

# 2. Tạo thư mục dự án

### Tạo thư mục chính

```bash
hdfs dfs -mkdir -p /bigdata/prosper_loan/raw
```

### Kiểm tra thư mục

```bash
hdfs dfs -ls /bigdata/prosper_loan
```

Kết quả:

* Thư mục raw được tạo thành công.

📸 Screenshot:

* [ ] Terminal hiển thị thư mục prosper_loan

---

# 3. Upload Dataset lên HDFS

### Upload file CSV

```bash
hdfs dfs -put prosperLoanData.csv /bigdata/prosper_loan/raw/
```

### Kiểm tra file

```bash
hdfs dfs -ls /bigdata/prosper_loan/raw/
```

Kết quả:

* Dataset được lưu thành công trên HDFS.

📸 Screenshot:

* [ ] Terminal hiển thị file prosperLoanData.csv trong HDFS

---

# 4. Kiểm tra nội dung Dataset

### Xem vài dòng đầu

```bash
hdfs dfs -cat /bigdata/prosper_loan/raw/prosperLoanData.csv | head
```

Kết quả:

* Hiển thị dữ liệu đầu vào.

📸 Screenshot:

* [ ] Terminal hiển thị dữ liệu mẫu

---

# 5. Kiểm tra dung lượng Dataset

### Xem kích thước file

```bash
hdfs dfs -du -h /bigdata/prosper_loan/raw/
```

Kết quả:

* Hiển thị dung lượng dataset.

📸 Screenshot:

* [ ] Terminal hiển thị kích thước file

---

# 6. Kiểm tra thông tin file

### Xem chi tiết file

```bash
hdfs dfs -stat "%n %b bytes" /bigdata/prosper_loan/raw/prosperLoanData.csv
```

Kết quả:

* Hiển thị tên và dung lượng file.

📸 Screenshot:

* [ ] Terminal hiển thị thông tin file

---

# 7. Kiểm tra trên NameNode UI

### Truy cập NameNode

URL:

http://localhost:9870

### Kiểm tra thư mục

* /bigdata
* /prosper_loan
* /raw

Kết quả:

* Dataset xuất hiện trên giao diện HDFS.

📸 Screenshot:

* [ ] NameNode UI
* [ ] Dataset trong HDFS

---

# 8. Các lệnh sử dụng trong quá trình thực hiện

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
