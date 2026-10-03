# Valor: Customer Lifetime Value (CLV) Prediction Using Machine Learning
 
> **Project Acronym:** **VALOR** (*Value Assessment & Lifetime Outlook via Regression*)  
> **Dataset:** [Kaggle IBM Watson Marketing Customer Value Analysis](https://www.kaggle.com/datasets/pankajchoudhary/ibm-watson-marketing-customer-value-data) (9,134 records, 24 features)  
> **Interactive App:** Streamlit Prototype (`app.py`)  

---

## Project Overview
Customer Lifetime Value (CLV) quantifies the total financial value a commercial enterprise expects from a customer over their relationship. **Valor** implements an end-to-end Machine Learning pipeline that ingests authentic Kaggle transaction and policy data, compares multiple regression algorithms, segments customers via unsupervised K-Means clustering, and provides an interactive Streamlit decision-support application.

The project notebook is structured **identically** to academic mini-project benchmarks (like the Stroke Prediction reference notebook), with 70 fully executed cells, dedicated exploratory data analysis, visual **Insight** annotations under every plot, model comparison bar charts, and K-Means customer segmentation.

---

## Interactive Decision-Support Dashboard

The platform includes an interactive decision-support application built with Streamlit and styled with an editorial design system. Each functional tab is detailed below:

### 1. Real-Time Customer Lifetime Value Prediction
<img src="Assest/Screenshot 2026-10-03 at 08.17.43.png" alt="Predict Customer CLV Dashboard" width="100%" />

- **Interactive Feature Inputs:** Underwriters and marketers can configure 21 demographic and policy attributes including monthly premium, active policies, income, coverage tier, and vehicle classification.
- **Instant Model Inference:** Executes the trained Random Forest model in real time to calculate expected CLV and assign the customer to an actionable tier (VIP, Moderate, or Developing).
- **Strategic Decision Intelligence:** Automatically outputs recommended retention budgets, annualized premiums, and tailored customer management protocols to guide commercial decisions.

---

### 2. Model Comparative Study & Evaluation Metrics
<img src="Assest/Screenshot 2026-10-03 at 08.18.11.png" alt="Model Comparative Study" width="100%" />

- **Comprehensive Algorithm Benchmarking:** Displays holdout evaluation metrics across all five regression algorithms evaluated on 1,827 unseen test customers (20% split).
- **Multi-Metric Comparison Table:** Directly compares Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Variance Explained ($R^2$).
- **Visual Performance Diagnostics:** Clean comparative bar charts highlight the significant performance advantage of non-linear tree ensembles over traditional linear regression.

---

### 3. Unsupervised Customer Segmentation (K-Means)
<img src="Assest/Screenshot 2026-10-03 at 08.18.26.png" alt="K-Means Customer Segments" width="100%" />

- **Multi-Dimensional Clustering:** Segments the customer population into three distinct cohorts ($k=3$) based on annual income, monthly premium auto, and total claim history without target label leakage.
- **Interactive Segmentation Scatterplot:** Visually demonstrates the separation between Moderate-Income Regulars, High-Income Low-Claim clients, and High-Premium VIP patrons.
- **Actionable Commercial Playbooks:** Maps data-driven marketing strategies, concierge account assignment, and upselling opportunities to each specific cluster profile.

---

### 4. Kaggle Dataset Explorer & Audit Catalog
<img src="Assest/Screenshot 2026-10-03 at 08.18.37.png" alt="Dataset Explorer" width="100%" />

- **Complete Data Catalog:** Provides transparent inspection of the underlying IBM Watson Marketing dataset comprising 9,134 records and 24 policyholder attributes.
- **Granular Record Inspection:** Allows users and evaluators to browse individual records, verify underwriting inputs, and examine policy renewal offer distributions.
- **Statistical Distribution Profiling:** Features embedded summary metrics detailing descriptive statistics, standard deviations, and quartile ranges for full auditability.

---

## Machine Learning Algorithms Compared
1. **Linear Regression** (Parametric Baseline with Feature Standardization)
2. **Decision Tree Regressor** (Non-parametric tree with max depth control)
3. **Random Forest Regressor** (Ensemble Bagging of 100 decision trees — **Best Performer**)
4. **Gradient Boosting Regressor** (Sequential boosting ensemble)
5. **XGBoost Regressor** (Extreme Gradient Boosting optimized ensemble)
6. **K-Means Clustering** ($k=3$) (Unsupervised customer segmentation on financial dimensions)

---

## Comparative Study Results (1,827 Unseen Test Customers)

| Algorithm | MAE ($) | MSE ($) | RMSE ($) | $R^2$ Score | Rank |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | $3,984.72 | 43,528,551.91 | $6,597.62 | 0.1554 | 5 |
| **Decision Tree** | $1,702.14 | 17,951,494.79 | $4,236.92 | 0.6517 | 4 |
| **Gradient Boosting** | $1,688.47 | 16,943,894.25 | $4,116.30 | 0.6712 | 3 |
| **XGBoost** | $1,686.50 | 16,877,130.61 | $4,108.18 | 0.6725 | 2 |
| **Random Forest (Best)** | **$1,500.27** | **16,013,133.05** | **$4,001.64** | **0.6893** | **1** |

---

## K-Means Customer Segments ($k=3$)

| Segment | Cohort Profile | Mean CLV | Mean Income | Mean Premium | Strategic Action |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **Cluster 0** | Moderate-Income Regulars | $7,410.71 | $14,853.37 | $87.42 | Digital renewal reminders & automated self-service |
| **Cluster 1** | High-Income Low-Claim Cohort | $7,569.92 | $66,276.08 | $85.19 | Extended coverage upselling & multi-vehicle discounts |
| **Cluster 2** | High-Premium VIP Patrons | **$14,250.53** | $26,599.45 | **$152.38** | Dedicated concierge account manager & rapid claims |

---

## Repository Structure
```
VALOR/
├── website/                       # Streamlit Web Application & Presentation UI
│   ├── .streamlit/config.toml     # Custom Theme & UI configuration
│   ├── app.py                     # Streamlit web app with interactive prediction & intelligence
│   ├── requirements.txt           # Web application dependencies
│   ├── README.md                  # Web application guide
│   ├── data -> ../data            # Symlinked data access
│   ├── models -> ../models        # Symlinked model access
│   └── plots -> ../plots          # Symlinked presentation figures
├── Assest/                        # UI Demonstration Screenshots
│   ├── Screenshot 2026-10-03 at 08.17.43.png
│   ├── Screenshot 2026-10-03 at 08.18.11.png
│   ├── Screenshot 2026-10-03 at 08.18.26.png
│   └── Screenshot 2026-10-03 at 08.18.37.png
├── data/                          # Clean Datasets & Evaluation Metrics
│   ├── customer_clv_data.csv      # IBM Watson CLV dataset (9,134 records)
│   ├── feature_cols.json          # Input feature column order
│   ├── metrics_comparison.json    # Regression benchmarks JSON
│   └── model_comparison.csv       # Regression benchmarks CSV
├── models/                        # Serialized Machine Learning Models
│   ├── best_model.pkl             # Top performer (Random Forest Regressor)
│   ├── all_models.pkl             # Dictionary of all 5 trained models
│   ├── kmeans_model.pkl           # Fitted K-Means clustering model (k=3)
│   ├── scaler.pkl                 # StandardScaler fitted on features
│   ├── cluster_scaler.pkl         # Scaler for financial attributes
│   └── label_encoders.pkl         # Encoders for categorical attributes
├── notebooks/                     # Jupyter Notebooks
│   └── Valor_CLV_Prediction.ipynb # Fully executed Jupyter Notebook with outputs & plots
├── plots/                         # High-resolution Presentation Plots (300 DPI)
│   ├── actual_vs_predicted.png
│   ├── eda_correlation_heatmap.png
│   ├── eda_income_distribution.png
│   ├── eda_monthly_premium.png
│   ├── eda_target_distribution.png
│   ├── kmeans_customer_clusters.png
│   └── model_comparison_bar.png
├── docs/                          # Academic & Project Documentation
│   ├── Project_Report.docx        # Professional Project Report (Word Document)
│   └── Project_Report.md          # Project Report in Markdown format
├── app.py                         # Root launcher delegating to website/app.py
├── train_models.py                # Pipeline script for training, benchmarking & plots
├── Valor_CLV_Prediction.ipynb     # Root notebook link
└── README.md                      # Comprehensive project documentation
```

---

## Getting Started

### 1. Launch Interactive Streamlit Website
Run the web application directly via terminal:
```bash
streamlit run website/app.py
```
*(Or run `streamlit run app.py` from project root)*

The web dashboard is hosted locally at: `http://localhost:8501`

### 2. Run the Jupyter Notebook
To view or re-execute the notebook via terminal:
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/Valor_CLV_Prediction.ipynb
```

### 3. Re-train Models & Generate Figures
To run the full regression training pipeline and update all models & plots:
```bash
python3 train_models.py
```

### 4. Generate the Project Report (.docx)
To compile the academic project report document with tables and embedded figures:
```bash
python3 create_report_docx.py
```
