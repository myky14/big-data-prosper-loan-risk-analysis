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
| Thư mục cài đặt | `D:\BIGDATA_THM`                                        |
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
D:\kafka
```

Cấu trúc thư mục Kafka sau khi giải nén gồm:

<img width="553" height="254" alt="image" src="https://github.com/user-attachments/assets/b6279a29-4e15-43c0-a4e4-247ed65d9509" />


---

## 4. Kiểm tra Java và Kafka

Trước khi khởi động Kafka, nhóm kiểm tra Java và các file thực thi của Kafka.

Chạy các lệnh sau trong PowerShell:

```powershell
cd D:\kafka

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
cd D:\kafka
mkdir D:\kafka\kraft-combined-logs
```
<img width="774" height="258" alt="image" src="https://github.com/user-attachments/assets/5308942d-b424-4822-8468-86958604a55f" />

### 5.2. Cấu hình file server.properties

Mở file:

```bash
D:\BIGDATA_THM\KAFKA\config\server.properties
```

Tìm dòng `log.dirs` và chỉnh thành đường dẫn Windows:

```bash
log.dirs=D:/kafka/kraft-combined-logs
```
<img width="1301" height="392" alt="image" src="https://github.com/user-attachments/assets/bdaa9741-e391-4962-b6dd-d283574e2c88" />

Nhóm kiểm tra lại bằng lệnh:

```powershell
Select-String "log.dirs" .\config\server.properties
```
<img width="964" height="111" alt="image" src="https://github.com/user-attachments/assets/e4719cb6-2576-4f0c-b7cb-bdaa58c70131" />


### 5.3. Tạo Cluster ID và format Kafka storage

Kafka chạy ở KRaft mode cần format storage trước khi khởi động broker. Trước tiên, tạo Cluster ID:

```powershell
cd C:\kafka
cmd /c ".\bin\windows\kafka-storage.bat random-uuid"
```

Sau khi lệnh trên in ra UUID, copy UUID đó và chạy lệnh format:

```powershell
cmd /c ".\bin\windows\kafka-storage.bat format -t <CODE_UUID> -c .\config\server.properties --standalone"
```

Nếu format thành công, thư mục này sẽ có file `meta.properties`.
<img width="1122" height="264" alt="image" src="https://github.com/user-attachments/assets/c52bb34e-5bac-49a5-8e63-7f8f39d7f82d" />

---

## 6. Khởi động Kafka Broker

Sau khi hoàn tất cấu hình, khởi động Kafka broker bằng lệnh:

```powershell
cd D:\kafka
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
<img width="1674" height="296" alt="image" src="https://github.com/user-attachments/assets/1cf35045-0a63-46d9-8135-f995d0d15c2d" />

### 7.2. Kiểm tra danh sách topic

```powershell
cmd /c ".\bin\windows\kafka-topics.bat --list --bootstrap-server localhost:9092"
```
<img width="1246" height="76" alt="image" src="https://github.com/user-attachments/assets/6e315e35-18a9-49c4-93a2-e30737013c58" />

### 7.3. Xem thông tin chi tiết của topic

```powershell
cmd /c ".\bin\windows\kafka-topics.bat --describe --topic prosper_loan_stream --bootstrap-server localhost:9092"
```
<img width="1719" height="91" alt="image" src="https://github.com/user-attachments/assets/2462f001-cbea-42ed-8c38-c3d92e7fddd5" />

---

## 8. Kiểm tra Producer và Consumer thủ công

Trước khi tích hợp Spark Structured Streaming, nhóm kiểm tra Kafka bằng cách gửi và đọc một message JSON đơn giản.

### 8.1. Chạy Kafka Console Producer

Mở PowerShell thứ nhất:

```powershell
cd C:\kafka
cmd /c ".\bin\windows\kafka-console-producer.bat --topic prosper_loan_stream --bootstrap-server localhost:9092"
```
<img width="1799" height="72" alt="image" src="https://github.com/user-attachments/assets/d99dfc78-64f9-4047-b660-c4ebd446dcee" />

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
<img width="1920" height="454" alt="image" src="https://github.com/user-attachments/assets/fd638edc-fcc8-45b4-b91d-ddb62d8c3d66" />

---

## 9. Kết quả đạt được

Sau khi hoàn thành phần Kafka và Spark Structured Streaming, nhóm đạt được các kết quả sau:

* Apache Kafka được cài đặt và khởi động thành công trên Windows.
* Kafka topic `prosper_loan_stream` được tạo để nhận dữ liệu streaming.
* Kafka Producer gửi được dữ liệu hồ sơ vay dưới dạng JSON vào topic.
* Spark Structured Streaming đọc được dữ liệu từ Kafka theo cơ chế micro-batch.
* Mô hình classification tốt nhất đã lưu từ batch MLlib pipeline được tải lại và tái sử dụng.
* Hệ thống dự đoán được kết quả Good Loan / Bad Loan cho các hồ sơ vay mới.
* Kết quả dự đoán được ghi xuống HDFS để phục vụ lưu trữ, kiểm tra và phân tích tiếp theo.

Phần streaming simulation cho thấy pipeline của nhóm có thể mở rộng từ xử lý batch sang xử lý dữ liệu gần thời gian thực. Trong bối cảnh tín dụng, điều này có ý nghĩa thực tế vì các hồ sơ vay mới có thể phát sinh liên tục, và hệ thống có thể hỗ trợ cảnh báo sớm các khoản vay có rủi ro cao.
