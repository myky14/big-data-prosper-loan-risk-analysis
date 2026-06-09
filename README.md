# Prosper Loan Risk Analysis using Hadoop & Spark

## Project Overview

This project analyzes loan risk and borrower behavior using the Prosper Loan Dataset with Apache Hadoop and Apache Spark.

Developed as a group academic project for the Big Data course, the project demonstrates a complete big data analytics workflow, including distributed data processing, feature selection, business insight generation, and machine learning for credit risk prediction.

### Project Goals

* Analyze factors affecting loan performance and borrower risk
* Generate business insights using Spark SQL
* Perform feature selection using domain knowledge and data-driven analysis
* Build machine learning models for:

  * Regression: Predict BorrowerAPR
  * Classification: Predict Good Loan / Bad Loan

---

## Dataset

**Dataset:** Prosper Loan Dataset

**Source:**
https://www.kaggle.com/datasets/nurudeenabdulsalaam/prosper-loan-dataset

### Dataset Characteristics

| Metric      | Value                            |
| ----------- | -------------------------------- |
| Records     | 113,937+                         |
| Features    | 81                               |
| Domain      | Financial Services / Credit Risk |
| Time Period | 2005 – 2014                      |

The dataset contains borrower demographics, credit history, loan characteristics, risk scores, and loan performance information.

---

## Business Understanding

Peer-to-peer lending platforms must balance loan growth and credit risk.

This project investigates:

* Which borrower characteristics are associated with higher credit risk?
* Which factors influence loan pricing (BorrowerAPR)?
* Can historical borrower information predict loan outcomes?
* How effective are Prosper's internal risk assessment metrics?

---

## Project Architecture

[TODO: Insert architecture diagram]

Suggested screenshot:

* Hadoop HDFS
* Spark Processing Layer
* Spark SQL Analytics
* Spark MLlib
* Business Insights & Prediction Outputs

---

## Methodology

### Phase 1: Data Understanding

* Dataset exploration
* Schema analysis
* Feature categorization
* Missing value inspection

### Phase 2: Feature Selection

* Domain-driven feature grouping
* Spark SQL analysis
* Correlation analysis
* Redundancy reduction
* Data leakage detection

Features reduced from:

81 Features → 24 Features

### Phase 3: Business Insight Analysis

Spark SQL was used to identify:

* High-risk borrower segments
* Credit score behavior
* Income and loan performance relationships
* Default risk patterns
* Geographic and demographic trends

### Phase 4: Machine Learning

#### Regression Task

Target:

BorrowerAPR

Objective:

Predict annual borrowing cost using borrower credit information.

#### Classification Task

Target:

Good Loan / Bad Loan

Objective:

Predict loan quality using borrower risk indicators.

---

## Spark SQL Business Insights

Examples of business questions explored:

* Which borrower groups have the highest default risk?
* Does ProsperScore effectively measure borrower risk?
* How does Debt-to-Income Ratio impact loan outcomes?
* Do homeowners have lower loan risk?
* How does previous repayment behavior affect future loans?

[TODO: Insert SQL result screenshots]

Suggested screenshots:

* ProsperScore vs Bad Loan Rate
* Credit Score vs Default Risk
* Income Range Analysis
* Debt-to-Income Ratio Analysis
* Loan Amount Analysis

---

## Machine Learning Results

### Regression

Model Objective:

Predict BorrowerAPR

Metrics:

[TODO: Insert RMSE, MAE, R²]

### Classification

Model Objective:

Predict Good Loan / Bad Loan

Metrics:

[TODO: Insert Accuracy, Precision, Recall, F1-score]

[TODO: Insert confusion matrix screenshot]

---

## Technologies

* Apache Hadoop
* HDFS
* Apache Spark
* Spark SQL
* Spark MLlib
* Python
* Jupyter Notebook
* Pandas
* Matplotlib
* Seaborn

---

## Repository Structure

```text
Prosper-Loan-Risk-Analysis/
│
├── Data/
├── Notebooks/
├── Spark_SQL/
├── Feature_Selection/
├── Machine_Learning/
├── Visualizations/
├── Screenshots/
├── Reports/
└── README.md
```

---

## Setup

### Prerequisites

* Python 3.x
* Apache Hadoop
* Apache Spark
* Jupyter Notebook

### Run Project

```bash
git clone <repository-url>

cd Prosper-Loan-Risk-Analysis

jupyter notebook
```

---

## Status

Project Status:

Completed Academic Project

Course:

Big Data

Version:

v1.0

---

## Team Members

* Nguyen Du My Ky
* Tran Hong Mai
* Duong Thanh Ngoc

---

## References

* Prosper Loan Dataset (Kaggle)
* Apache Hadoop Documentation
* Apache Spark Documentation
* Spark MLlib Documentation

```
```
