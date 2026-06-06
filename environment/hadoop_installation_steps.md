# Hadoop Installation Steps

## 1. Mục tiêu

* Cài đặt Apache Hadoop Single Node
* Khởi tạo HDFS
* Upload Prosper Loan Dataset lên HDFS
* Chứng minh Spark có thể đọc dữ liệu từ HDFS

## 2. Môi trường thực hiện

### Thông tin máy
- Hệ điều hành: Microsoft Windows 10 Pro
- Phiên bản hệ điều hành: 10.0.19044 Build 19044
- Bộ xử lý: 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz
- Bộ nhớ RAM: 8.00 GB
- Công cụ kiểm tra: System Information của Windows

<img width="1259" height="1004" alt="Untitled design" src="https://github.com/user-attachments/assets/b7c9ed7f-8a2f-42a0-897c-79dcd1208cb9" />

## 3. Cài đặt Java
### Tải Hadoop
Để Hadoop có thể hoạt động trên Windows, nhóm sử dụng OpenJDK 8 làm môi trường chạy Java. Bộ cài được tải từ trang OpenLogic OpenJDK với các thông tin lựa chọn như sau:
- Java version: OpenJDK 8
- Operating system: Windows
- Architecture: x64
- Package type: JDK

Nguồn tải: 
https://www.openlogic.com/openjdk-downloads?field_java_parent_version_target_id=416&field_operating_system_target_id=436&field_architecture_target_id=391&field_java_package_target_id=396

Sau khi tải về, thư mục JDK được giải nén vào đường dẫn:
```bash
D:\BIGDATA_THM\JAVA\openlogic-openjdk-8u492-b09-windows-x64
```
### Set biến môi trường
Cấu hình biến môi trường JAVA_HOME và bổ sung đường dẫn Java vào biến Path.
- Tạo biến hệ thống JAVA_HOME
- Giá trị của JAVA_HOME trỏ đến thư mục cài đặt JDK 8
<img width="877" height="966" alt="Untitled design (1)" src="https://github.com/user-attachments/assets/a605079a-de21-4a5d-a80e-6a1fb89d4313" />
- Thêm %JAVA_HOME%\bin vào biến Path
<img width="877" height="966" alt="Untitled design (2)" src="https://github.com/user-attachments/assets/f3641dc3-df94-4a0d-bd9b-50ea4e6e3692" />

### Kiểm tra Java
Mở Command Prompt/PowerShell mới và thực hiện lệnh sau để kiểm tra phiên bản Java:

```bash
java -version
```

<img width="814" height="283" alt="image" src="https://github.com/user-attachments/assets/435c8d55-117f-4b84-9c6c-fb25a0e79e0f" />


## 4. Cài đặt Hadoop

### Tải Hadoop
Nhóm sử dụng Apache Hadoop phiên bản 3.4.1 để triển khai mô hình Hadoop Single Node trên hệ điều hành Windows.
- Hadoop version: 3.4.1
- Package: Binary package
- File tải về: `hadoop-3.4.1.tar.gz`
- Nguồn tải:
  
* URL: https://www.apache.org/dyn/closer.cgi/hadoop/common/hadoop-3.4.1/hadoop-3.4.1-src.tar.gz

### Giải nén Hadoop

```bash
tar -xzf hadoop-*.tar.gz
```

<img width="557" height="470" alt="image" src="https://github.com/user-attachments/assets/44721dad-4ffb-49f6-afdc-88a7452c128a" />

### Cấu hình biến môi trường 
<img width="877" height="966" alt="Untitled design (3)" src="https://github.com/user-attachments/assets/a5acdbbe-0040-4148-81f3-fec80a5ed48c" />

### Kiểm tra lại Command Prompt/PowerShell mới và chạy lệnh:
```bash
hadoop version
```
<img width="1103" height="208" alt="image" src="https://github.com/user-attachments/assets/58be316d-f708-4a19-a2ad-ca4b52be7bf7" />

---

## 5. Cấu hình Hadoop

### File core-site.xml
File core-site.xml được sử dụng để cấu hình địa chỉ mặc định của hệ thống tệp phân tán HDFS. Trong cấu hình này, thuộc tính fs.defaultFS được thiết lập là hdfs://localhost:9000, cho biết Hadoop sẽ kết nối đến NameNode chạy trên máy local tại cổng 9000.
```bash
<configuration>
    <property>
        <name>fs.defaultFS</name>
        <value>hdfs://localhost:9000</value>
    </property>
</configuration>
```
Screenshot:
<img width="910" height="435" alt="image" src="https://github.com/user-attachments/assets/619c7304-5ef2-4c1b-b1c7-837511a7a6fd" />

### File hdfs-site.xml
File hdfs-site.xml được sử dụng để cấu hình các thông số liên quan đến HDFS. Do hệ thống được triển khai theo mô hình Single Node, số lượng bản sao dữ liệu được thiết lập là 1 thông qua thuộc tính dfs.replication.

Ngoài ra, file này cũng khai báo thư mục lưu trữ metadata của NameNode và dữ liệu block của DataNode. Trước khi cấu hình, tạo hai thư mục lưu trữ dữ liệu cho HDFS là datanode và namenode:
<img width="555" height="223" alt="image" src="https://github.com/user-attachments/assets/455ac189-5fb7-4de1-a96c-48b455243675" />

Nội dung cấu hình: 
```bash
<configuration>
    <property>
        <name>dfs.replication</name>
        <value>1</value>
    </property>

    <property>
        <name>dfs.namenode.name.dir</name>
        <value>file:///D:/BIGDATA_THM/HADOOP/hadoop-3.4.1/data/namenode</value>
    </property>

    <property>
        <name>dfs.datanode.data.dir</name>
        <value>file:///D:/BIGDATA_THM/HADOOP/hadoop-3.4.1/data/datanode</value>
    </property>
</configuration>
```
Screenshot:

<img width="1189" height="611" alt="image" src="https://github.com/user-attachments/assets/134d5727-b70e-40a8-9123-a8acbc99fd82" />

### File mapred-site.xml
File mapred-site.xml được sử dụng để cấu hình framework thực thi MapReduce. Trong cấu hình này, thuộc tính mapreduce.framework.name được thiết lập là yarn, cho phép các tác vụ MapReduce được thực thi thông qua YARN.
```bash
<configuration>
    <property>
        <name>mapreduce.framework.name</name>
        <value>yarn</value>
    </property>
</configuration>
```
Screenshot:

<img width="1011" height="421" alt="image" src="https://github.com/user-attachments/assets/63b4669f-8953-4a3b-8097-c109ed90cbe8" />

### File yarn-site.xml

File yarn-site.xml được sử dụng để cấu hình YARN. Trong cấu hình này, dịch vụ mapreduce_shuffle được khai báo để NodeManager có thể hỗ trợ quá trình thực thi các tác vụ MapReduce trên YARN.
```bash
<configuration>
    <property>
        <name>yarn.nodemanager.aux-services</name>
        <value>mapreduce_shuffle</value>
    </property>

    <property>
        <name>yarn.nodemanager.aux-services.mapreduce.shuffle.class</name>
        <value>org.apache.hadoop.mapred.ShuffleHandler</value>
    </property>
</configuration>
```
Screenshot:
<img width="1190" height="613" alt="image" src="https://github.com/user-attachments/assets/c65b9044-471e-467f-a397-30dfc81a4cec" />

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
