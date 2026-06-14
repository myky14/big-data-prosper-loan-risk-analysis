# Kafka Streaming Simulation Setup

## 1. Mục tiêu

* Cài đặt Apache Kafka trên môi trường Windows.
* Tạo Kafka topic để mô phỏng luồng hồ sơ khoản vay mới.
* Sử dụng Kafka Producer để gửi dữ liệu streaming simulation vào topic.
* Sử dụng Spark Structured Streaming để đọc dữ liệu từ Kafka.
* Tải mô hình phân loại tốt nhất đã lưu từ batch MLlib pipeline.
* Dự đoán kết quả Good Loan / Bad Loan cho từng hồ sơ vay mới.
* Ghi kết quả dự đoán xuống HDFS để phục vụ cảnh báo rủi ro tín dụng.

Trong đồ án Prosper Loan Risk Analysis, dữ liệu gốc là dữ liệu lịch sử dạng batch. Vì vậy, phần streaming không nhằm thay thế pipeline batch đã xây dựng, mà nhằm mô phỏng tình huống thực tế khi hệ thống liên tục tiếp nhận các hồ sơ vay mới. Mô hình classification đã được huấn luyện ở bước batch sẽ được tái sử dụng để dự đoán rủi ro cho dữ liệu mới theo cơ chế gần thời gian thực.

---

## 2. Môi trường thực hiện

### Thông tin môi trường Kafka

| Thành phần      | Cấu hình sử dụng                                        |
| --------------- | ------------------------------------------------------- |
| Hệ điều hành    | Windows 10/11                                           |
| Java            | OpenJDK 17 trở lên                                      |
| Apache Kafka    | Kafka 3.9.2                                             |
| Scala version   | 2.13                                                    |
| Thư mục cài đặt | `C:\kafka`                                              |
| Broker local    | `localhost:9092`                                        |
| Kafka topic     | `prosper_loan_stream`                                   |
| Terminal        | Windows PowerShell, chạy file `.bat` thông qua `cmd /c` |

Lưu ý: Kafka được triển khai như một thành phần bổ sung cho phần streaming simulation. Việc cài đặt Kafka không làm thay đổi cấu hình Hadoop, HDFS hoặc Spark đã thiết lập trước đó.

---

## 3. Cài đặt Apache Kafka

### 3.1. Tải Kafka

Nhóm sử dụng Apache Kafka phiên bản 3.9.2 với Scala 2.13 để triển khai phần streaming simulation.

Thông tin tải:

* Kafka version: Apache Kafka 3.9.2
* Scala version: 2.13
* Package: Binary package
* File tải về: `kafka_2.13-3.9.2.tgz`

Sau khi tải về, nhóm giải nén file `.tgz`, sau đó giải nén tiếp file `.tar` để thu được thư mục Kafka. Để hạn chế lỗi đường dẫn dài trên Windows, thư mục Kafka được đặt tại:

```bash
C:\kafka
```

Cấu trúc thư mục Kafka sau khi giải nén gồm:

```bash
C:\kafka
   |-- bin
   |-- config
   |-- libs
   |-- LICENSE
   |-- NOTICE
   |-- site-docs
```

---

## 4. Kiểm tra Java và Kafka

Trước khi khởi động Kafka, nhóm kiểm tra Java và các file thực thi của Kafka.

Chạy các lệnh sau trong PowerShell:

```powershell
cd C:\kafka

java -version
echo $env:JAVA_HOME
(Get-Item C:\kafka\bin\windows\kafka-storage.bat).Length
```

Kết quả hợp lệ:

* `java -version` hiển thị OpenJDK 17 hoặc cao hơn.
* `JAVA_HOME` trỏ đúng đến thư mục JDK.
* File `kafka-storage.bat` có kích thước lớn hơn 0 bytes.

Việc kiểm tra này giúp đảm bảo Kafka đã được giải nén đúng và có thể chạy các file batch trên Windows.

---

## 5. Cấu hình Kafka

### 5.1. Tạo thư mục lưu log Kafka

Kafka cần một thư mục riêng để lưu dữ liệu broker và metadata. Nhóm tạo thư mục log tại:

```powershell
cd C:\kafka
mkdir C:\kafka\kraft-combined-logs
```

### 5.2. Cấu hình file server.properties

Mở file:

```bash
C:\kafka\config\server.properties
```

Tìm dòng `log.dirs` và chỉnh thành đường dẫn Windows:

```bash
log.dirs=C:/kafka/kraft-combined-logs
```

Có thể kiểm tra lại bằng lệnh:

```powershell
Select-String "log.dirs" .\config\server.properties
```

### 5.3. Tạo Cluster ID và format Kafka storage

Kafka chạy ở KRaft mode cần format storage trước khi khởi động broker. Trước tiên, tạo Cluster ID:

```powershell
cd C:\kafka
cmd /c ".\bin\windows\kafka-storage.bat random-uuid"
```

Sau khi lệnh trên in ra UUID, copy UUID đó và chạy lệnh format:

```powershell
cmd /c ".\bin\windows\kafka-storage.bat format -t <UUID_VUA_COPY> -c .\config\server.properties --standalone"
```

Ví dụ:

```powershell
cmd /c ".\bin\windows\kafka-storage.bat format -t 0e4efee8-a9ed-4ad0-a5ee-c71a994e268c -c .\config\server.properties --standalone"
```

Kiểm tra thư mục log:

```powershell
dir C:\kafka\kraft-combined-logs
```

Nếu format thành công, thư mục này sẽ có file `meta.properties`.

---

## 6. Khởi động Kafka Broker

Sau khi hoàn tất cấu hình, khởi động Kafka broker bằng lệnh:

```powershell
cd C:\kafka
cmd /c ".\bin\windows\kafka-server-start.bat .\config\server.properties"
```

Terminal đang chạy Kafka broker cần được giữ mở trong suốt quá trình demo streaming. Nếu đóng terminal này, Kafka broker sẽ dừng hoạt động.

---

## 7. Tạo và kiểm tra Kafka Topic

### 7.1. Tạo topic prosper_loan_stream

Mở một PowerShell mới và chạy:

```powershell
cd C:\kafka
cmd /c ".\bin\windows\kafka-topics.bat --create --topic prosper_loan_stream --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1"
```

Topic `prosper_loan_stream` được sử dụng để nhận các hồ sơ khoản vay mới từ Kafka Producer.

### 7.2. Kiểm tra danh sách topic

```powershell
cmd /c ".\bin\windows\kafka-topics.bat --list --bootstrap-server localhost:9092"
```

### 7.3. Xem thông tin chi tiết của topic

```powershell
cmd /c ".\bin\windows\kafka-topics.bat --describe --topic prosper_loan_stream --bootstrap-server localhost:9092"
```

---

## 8. Kiểm tra Producer và Consumer thủ công

Trước khi tích hợp Spark Structured Streaming, nhóm kiểm tra Kafka bằng cách gửi và đọc một message JSON đơn giản.

### 8.1. Chạy Kafka Console Producer

Mở PowerShell thứ nhất:

```powershell
cd C:\kafka
cmd /c ".\bin\windows\kafka-console-producer.bat --topic prosper_loan_stream --bootstrap-server localhost:9092"
```

Nhập thử một JSON message:

```json
{"LoanOriginalAmount": 5000, "Term": 36, "IncomeRange": "$25,000-49,999"}
```

### 8.2. Chạy Kafka Console Consumer

Mở PowerShell thứ hai:

```powershell
cd C:\kafka
cmd /c ".\bin\windows\kafka-console-consumer.bat --topic prosper_loan_stream --from-beginning --bootstrap-server localhost:9092"
```

Nếu consumer hiển thị đúng JSON vừa gửi, Kafka topic đã hoạt động thành công.

---

## 9. Tích hợp Kafka Producer cho dữ liệu Prosper Loan

Sau khi mô hình classification được huấn luyện ở batch pipeline, nhóm sử dụng 10% dữ liệu được tách riêng làm streaming simulation data. Tập dữ liệu này đóng vai trò mô phỏng các hồ sơ khoản vay mới được gửi liên tục vào hệ thống.

File thực hiện:

```bash
src/08_kafka_producer.py
```

Vai trò của file:

* Đọc các bản ghi mô phỏng streaming dưới dạng JSON.
* Gửi từng hồ sơ khoản vay vào Kafka topic `prosper_loan_stream`.
* Tạo độ trễ giữa các bản ghi để mô phỏng dữ liệu đến theo thời gian.

Chạy Kafka Producer:

```bash
python src/08_kafka_producer.py --topic prosper_loan_stream --bootstrap-server localhost:9092 --delay 0.5
```

Trong đó:

* `--topic`: tên Kafka topic nhận dữ liệu.
* `--bootstrap-server`: địa chỉ Kafka broker.
* `--delay`: thời gian chờ giữa hai bản ghi liên tiếp.

---

## 10. Tích hợp Spark Structured Streaming

File thực hiện:

```bash
src/09_spark_structured_streaming_prediction.py
```

Vai trò của file:

* Đọc dữ liệu từ Kafka topic `prosper_loan_stream`.
* Chuyển dữ liệu JSON thành DataFrame theo schema của tập streaming simulation.
* Tải mô hình classification tốt nhất đã lưu dưới dạng PipelineModel.
* Dự đoán khoản vay thuộc nhóm Good Loan hoặc Bad Loan.
* Ghi kết quả dự đoán xuống HDFS.

Chạy Spark Structured Streaming:

```bash
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:<SPARK_VERSION> src/09_spark_structured_streaming_prediction.py
```

Ví dụ với Spark 3.5.8:

```bash
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.8 src/09_spark_structured_streaming_prediction.py
```

Trong quá trình chạy, Spark Structured Streaming đọc dữ liệu từ Kafka theo cơ chế micro-batch. Mỗi nhóm bản ghi mới được Spark xử lý như một batch nhỏ, sau đó áp dụng mô hình đã lưu để sinh ra kết quả dự đoán.

---

## 11. Kiểm tra kết quả dự đoán trên HDFS

Sau khi Spark Structured Streaming chạy và ghi output, kiểm tra kết quả trên HDFS:

```bash
hdfs dfs -ls /bigdata/prosper_loan/streaming/predictions
```

Kết quả mong đợi là thư mục output có các file parquet chứa kết quả dự đoán, bao gồm các thông tin như:

* thời điểm xử lý,
* nhãn dự đoán,
* nhóm dự đoán Good Loan hoặc Bad Loan,
* một số đặc trưng chính của hồ sơ vay.

---

## 12. Kết quả đạt được

Sau khi hoàn thành phần Kafka và Spark Structured Streaming, nhóm đạt được các kết quả sau:

* Apache Kafka được cài đặt và khởi động thành công trên Windows.
* Kafka topic `prosper_loan_stream` được tạo để nhận dữ liệu streaming.
* Kafka Producer gửi được dữ liệu hồ sơ vay dưới dạng JSON vào topic.
* Spark Structured Streaming đọc được dữ liệu từ Kafka theo cơ chế micro-batch.
* Mô hình classification tốt nhất đã lưu từ batch MLlib pipeline được tải lại và tái sử dụng.
* Hệ thống dự đoán được kết quả Good Loan / Bad Loan cho các hồ sơ vay mới.
* Kết quả dự đoán được ghi xuống HDFS để phục vụ lưu trữ, kiểm tra và phân tích tiếp theo.

Phần streaming simulation cho thấy pipeline của nhóm có thể mở rộng từ xử lý batch sang xử lý dữ liệu gần thời gian thực. Trong bối cảnh tín dụng, điều này có ý nghĩa thực tế vì các hồ sơ vay mới có thể phát sinh liên tục, và hệ thống có thể hỗ trợ cảnh báo sớm các khoản vay có rủi ro cao.
