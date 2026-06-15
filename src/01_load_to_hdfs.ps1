Write-Host "===== LOAD PROSPER LOAN DATASET TO HDFS ====="

$LocalFile = "data/raw/prosperLoanData.csv"
$HdfsDir = "/bigdata/prosper_loan/raw"

Write-Host "`nStep 1: Check local file"

if (-not (Test-Path $LocalFile)) {
    Write-Host "File not found: $LocalFile"
    exit
}

Write-Host "File found."

Write-Host "`nStep 2: Create HDFS directory"

hdfs dfs -mkdir -p $HdfsDir

Write-Host "`nStep 3: Upload dataset"

hdfs dfs -put -f $LocalFile $HdfsDir

Write-Host "`nStep 4: Verify upload"

hdfs dfs -ls $HdfsDir

Write-Host "`nStep 5: Check file size"

hdfs dfs -du -h $HdfsDir

Write-Host "`n===== UPLOAD COMPLETED ====="