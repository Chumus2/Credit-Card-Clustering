# Credit Card Customer Segmentation

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-Blue?style=flat)
![HDBSCAN](https://img.shields.io/badge/HDBSCAN-Density--Based-teal)

Segmenting credit card users into behavior groups using PCA and HDBSCAN based on spending, balances, and payment habits

---

## 📌 Overview

Understanding customer behavior allows financial institutions to personalize credit limits, tailor marketing campaigns, and mitigate churn risk. 
This project performs *unsupervised behavioral clustering* on **~9,000 credit cardholders**, isolates noise/anomalies, and 
maps out actionable customer personas using density-based algorithms.

---

## 🛠️ Tech Stack and Methods

* **Language and Analysis:** Python (`pandas`, `numpy`, `seaborn`, `matplotlib`)
* **Preprocessing:** `log1p` transformation for right-skewed data + `StandardScaler`
* **Dimensionality Reduction:** **PCA** (6 components retaining **84.2%** of total variance)
* **Clustering Algorithms:** **KMeans**, **DBSCAN**, **HDBSCAN**
* **Hyperparameter Tuning:** **Optuna Framework** (50 trials per algorithm)
* **Visualization:** Custom dual-layer heatmap using row-wise `MinMaxScaler`

---

## 📋 Features Used

The dataset contains behavioral metrics collected over 6 months:

* **Account & Balance:**
  * `BALANCE`: Amount of money left in the account to make purchases.
  * `BALANCE_FREQUENCY`: How frequently the balance is updated (score between 0 and 1).
  * `CREDIT_LIMIT`: Limit of credit card for user.
* **Purchasing Behavior:**
  * `PURCHASES`: Total amount of purchases made from account.
  * `ONEOFF_PURCHASES`: Maximum purchase amount done in one-go.
  * `INSTALLMENTS_PURCHASES`: Amount of purchase done in installment.
  * `PURCHASES_FREQUENCY`: How frequently purchases are being made (0 to 1).
  * `ONEOFF_PURCHASES_FREQUENCY`: Frequency of one-off purchases (0 to 1).
  * `PURCHASES_INSTALLMENTS_FREQUENCY`: Frequency of installment purchases (0 to 1).
  * `PURCHASES_TRX`: Number of purchase transactions made.
* **Cash Advances & Payments:**
  * `CASH_ADVANCE`: Cash in advance given by the user.
  * `CASH_ADVANCE_FREQUENCY`: How frequently the cash in advance has been paid.
  * `CASH_ADVANCE_TRX`: Number of transactions made with "Cash in Advance".
  * `PAYMENTS`: Amount of payment done by user.
  * `MINIMUM_PAYMENTS`: Minimum amount of payments made by user.
  * `PRC_FULL_PAYMENT`: Percent of full payment paid by user.
  * `TENURE`: Tenure of credit card service for user.

---

## 📊 Model Evaluation

| Model | Silhouette Score | Davies-Bouldin | Calinski-Harabasz | Key Takeaway |
| :--- | :---: | :---: | :---: | :--- |
| **KMeans** | 0.271 | 1.334 | **3000.5** | Forces 100% data assignment; weak on non-spherical clusters. |
| **DBSCAN** | **0.300** | 0.910 | 1968.4 | Global `eps` fails across regions of varying density. |
| **HDBSCAN** *(Selected)* | 0.240 | **0.783** | 97.6 | **Best performance.** Adapts to variable density & isolates noise (~11.5%). |

---

## 💡 Why HDBSCAN was Selected as the Final Model

While **KMeans** achieved a higher Calinski-Harabasz score (due to its bias toward spherical clusters) and 
**DBSCAN** had a slightly higher Silhouette score, 
**HDBSCAN was chosen as the primary model** for the following business and statistical reasons:

1. **Handles Complex Data:** Financial behavior naturally varies in density. HDBSCAN finds clusters of different shapes and densities better than fixed-rule models.
2. **Filters Out Noise:** HDBSCAN isolated **~11.5% of random outliers/anomalies** instead of forcing them into real user groups. This keeps customer profiles clean.
3. **Best Cluster Separation:** Achieved the lowest **Davies-Bouldin score (0.783)**, showing clear distinction between customer groups.

---

## 🎯 Target Customer Segments

* **Active Core Cardholders (~87.7%)**: Primary high-value segment with high credit limits ($4,466), regular payments, and strong purchasing activity ($967).
* **Installment-Only Newcomers (~0.5%)**: Near-zero balance ($2.1) users who strictly use credit cards for installment purchases ($186).
* **Cash Borrowers (~0.3%)**: Segment relying almost exclusively on cash advances ($1,185) with zero point-of-sale purchases.
* **Outliers / Noise (~11.5%)**: Anomalous transaction behavior isolated by HDBSCAN to maintain clean segment integrity.

---

## 📁 Repository Structure

```text
├── Data/
│   ├── Processed/     # Scaled & transformed features
│   └── Final/         # Final dataset with cluster assignments
├── Models/            # Serialized HDBSCAN model (.pkl)
├── Notebooks/         # EDA, Clustering, Feature-Engineering and Profiling notebooks
├── Src/               # Modular Python scripts (data loading, plotting, model evaluation)
└── README.md
