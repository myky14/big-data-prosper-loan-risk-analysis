# Hadoop Installation Steps

## 1. Mục tiêu

* Cài đặt Apache Hadoop Single Node
* Khởi tạo HDFS
* Upload Prosper Loan Dataset lên HDFS
* Chứng minh Spark có thể đọc dữ liệu từ HDFS

---

## 2. Môi trường thực hiện

### Thông tin máy

* Hệ điều hành:
* CPU:
* RAM:
* Java Version:
* Hadoop Version:

Screenshot:

* [ ] System Information
* [ ] Java Version

---

## 3. Cài đặt Java

### Kiểm tra Java

```bash
java -version
```

Screenshot:

* [ ] Java Version

---

## 4. Cài đặt Hadoop

### Tải Hadoop

Nguồn tải:

* URL:

### Giải nén Hadoop

```bash
tar -xzf hadoop-*.tar.gz
```

Screenshot:

* [ ] Hadoop Folder

---

## 5. Cấu hình Hadoop

### File core-site.xml

Giải thích ngắn chức năng.

Screenshot:

* [ ] core-site.xml

### File hdfs-site.xml

Giải thích ngắn chức năng.

Screenshot:

* [ ] hdfs-site.xml

### File mapred-site.xml

Giải thích ngắn chức năng.

Screenshot:

* [ ] mapred-site.xml

### File yarn-site.xml

Giải thích ngắn chức năng.

Screenshot:

* [ ] yarn-site.xml

---

## 6. Khởi tạo HDFS

### Format NameNode

```bash
hdfs namenode -format
```

Screenshot:

* [ ] Format Successful

---

## 7. Khởi động Hadoop

### Start DFS

```bash
start-dfs.sh
```

### Start YARN

```bash
start-yarn.sh
```

### Kiểm tra tiến trình

```bash
jps
```

Screenshot:

* [ ] JPS Result

---

## 8. Kiểm tra HDFS

### Tạo thư mục

```bash
hdfs dfs -mkdir -p /bigdata/prosper_loan/raw
```

### Kiểm tra

```bash
hdfs dfs -ls /
```

Screenshot:

* [ ] HDFS Folder Structure

---

## 9. Upload Dataset lên HDFS

### Upload file

```bash
hdfs dfs -put prosperLoanData.csv /bigdata/prosper_loan/raw/
```

### Kiểm tra file

```bash
hdfs dfs -ls /bigdata/prosper_loan/raw/
```

Screenshot:

* [ ] Dataset Uploaded Successfully

---

## 10. Kiểm tra Hadoop Web UI

### NameNode UI

URL:
http://localhost:9870

Screenshot:

* [ ] NameNode Dashboard

### DataNode UI

Screenshot:

* [ ] DataNode Information

---

## 11. Kết quả đạt được

* Hadoop cài đặt thành công
* HDFS hoạt động bình thường
* Dataset được lưu trên HDFS
* Sẵn sàng cho bước Spark Processing
