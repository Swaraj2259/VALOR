# Project Report: Customer Lifetime Value (CLV) Prediction Using Machine Learning

**Project Name:** Valor — Value Assessment & Lifetime Outlook via Regression  
**Case Study:** Case Study 158: Customer Lifetime Value Prediction  
**Program:** B.Tech Computer Science & Engineering | Semester V  
**Domain:** Machine Learning / Customer Intelligence & Predictive Analytics  
**Dataset:** [Kaggle IBM Watson Marketing Customer Value Analysis](https://www.kaggle.com/datasets/pankajchoudhary/ibm-watson-marketing-customer-value-data) (9,134 Records, 24 Attributes)  
**Notebook Reference:** [Valor_CLV_Prediction.ipynb](file:///Users/swarajwattamwar/Documents/VALOR/Valor_CLV_Prediction.ipynb)  
**Word Report:** [Project_Report.docx](file:///Users/swarajwattamwar/Documents/VALOR/Project_Report.docx)  

---

## 1. Abstract
Customer Lifetime Value (CLV) is one of the foundational metrics in customer relationship management, quantifying the aggregate financial revenue an enterprise expects from a customer throughout their active engagement. Accurately estimating future CLV empowers retail and insurance leadership to optimize Customer Acquisition Costs (CAC), tailor loyalty programs, and deploy proactive retention interventions.

This project, titled **Valor**, implements a complete machine learning lifecycle using the official **Kaggle IBM Watson Marketing Customer Value Analysis** dataset, analyzing 9,134 policyholder records across 24 demographic, policy, and behavioral dimensions. Four distinct regression algorithms were evaluated and rigorously benchmarked: **Linear Regression (Baseline), Decision Tree Regressor, Random Forest Regressor, and Gradient Boosting Regressor**. Furthermore, unsupervised **K-Means clustering ($k=3$)** was deployed to segment the customer base into actionable operational tiers.

Our empirical comparative evaluation shows that **Random Forest Regressor** achieved the highest accuracy, explaining **68.93%** of variance ($R^2 = 0.6893$) on unseen test data, with a Root Mean Squared Error (RMSE) of **$4,001.64** and a Mean Absolute Error (MAE) of **$1,500.27**, dramatically outperforming the linear baseline ($R^2 = 0.1554$). Furthermore, an interactive web prototype was developed in **Streamlit**, enabling real-time CLV forecasting, customer tier segmentation (Budget, Moderate, High-Value VIP), and targeted retention strategy recommendations.

---

## 2. Problem Statement & Case Study Objectives
Commercial organizations manage thousands of customer policies annually, yet marketing budgets are often allocated uniformly without understanding individual customer value trajectories. Treating all customers identically leads to over-spending on unprofitable, price-sensitive shoppers and under-investing in high-value, loyal patrons.

### Primary Objectives:
1. **Ingest and Clean Authentic Kaggle Data:** Process 9,134 records from the IBM Watson Marketing Customer Value dataset, ensuring complete data hygiene.
2. **Conduct Rigorous Exploratory Data Analysis (EDA):** Quantify how monthly premiums, policy count, claim amounts, and coverage tiers drive customer lifetime value.
3. **Train and Benchmark Regression Algorithms:** Evaluate 4 regression models on standardized metrics: MAE, MSE, RMSE, and $R^2$ Score.
4. **Deploy Unsupervised Customer Segmentation:** Execute K-Means clustering ($k=3$) on financial attributes to isolate high-value VIP customer segments.
5. **Develop an Interactive Decision-Support Prototype:** Deploy a Streamlit web application providing instant predictions, dynamic segmentation, and targeted retention recommendations.

---

## 3. Dataset & Feature Engineering

The project utilizes the Kaggle **IBM Watson Marketing Customer Value Analysis** dataset:

| Feature Name | Data Type | Role | Description |
| :--- | :--- | :--- | :--- |
| **`Customer Lifetime Value`** | Numeric (Float) | **Target Variable** | Historical aggregate lifetime revenue ($1,898 to $83,325) |
| **`Monthly Premium Auto`** | Numeric (Integer) | Key Predictor | Monthly policy cost ($61 to $298) |
| **`Total Claim Amount`** | Numeric (Float) | Key Predictor | Cumulative historical claim expenses ($0.09 to $2,893) |
| **`Number of Policies`** | Numeric (Integer) | Predictor | Total insurance policies maintained (1 to 9) |
| **`Income`** | Numeric (Integer) | Predictor | Annual reported income ($0 to $100,000) |
| **`Coverage`** | Categorical | Predictor | Policy tier: `Basic`, `Extended`, `Premium` |
| **`Vehicle Class`** | Categorical | Predictor | Category: `Four-Door Car`, `SUV`, `Sports Car`, `Luxury` |
| **`EmploymentStatus`** | Categorical | Predictor | Status: `Employed`, `Unemployed`, `Retired`, `Medical Leave` |
| **`Months Since Inception`** | Numeric (Integer) | Predictor | Tenure of policy relationship in months (0 to 99) |
| **`Months Since Last Claim`** | Numeric (Integer) | Predictor | Recency of last recorded claim event (0 to 35) |

### Preprocessing Pipeline:
- **Identifier Removal:** Dropped `Customer` (unique ID) and `Effective To Date` (timestamp) as non-predictive operational artifacts.
- **Categorical Encoding:** Applied Scikit-Learn's `LabelEncoder` across 14 categorical features.
- **Train/Test Partitioning:** Split into 80% training (7,307 samples) and 20% unseen testing (1,827 samples) with fixed seed 42.
- **Feature Standardization:** Standardized numerical columns using `StandardScaler` for linear baseline models.

---

## 4. Exploratory Data Analysis (EDA) Insights

1. **Target Distribution:** Customer Lifetime Value exhibits a classic right-skewed Pareto distribution (mean: $8,004.94, median: $5,780.18), with 75% of customers below $8,962 and high-value outliers exceeding $40,000.
2. **Monthly Premium Impact:** Monthly premium is the strongest individual predictor ($r = 0.40$), with customers paying higher monthly rates compounding greater lifetime value.
3. **Coverage Tier Analysis:** Policyholders on `Extended` and `Premium` tiers generate significantly higher upper-quartile CLV compared to `Basic` coverage tiers, highlighting the ROI of upselling.
4. **Income Bimodality:** Income displays a bimodal shape with a distinct zero-income cohort (unemployed/students/retirees) alongside an evenly distributed working class from $20k to $100k.

---

## 5. Model Evaluation & Comparative Study

All models were evaluated on the **1,827 unseen test samples**:

| Algorithm | MAE ($) | MSE ($) | RMSE ($) | $R^2$ Score | Rank |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression (Baseline)** | $3,984.72 | 43,528,551.91 | $6,597.62 | 0.1554 | 5 |
| **Decision Tree Regressor** | $1,702.14 | 17,951,494.79 | $4,236.92 | 0.6517 | 4 |
| **Gradient Boosting Regressor** | $1,688.47 | 16,943,894.25 | $4,116.30 | 0.6712 | 3 |
| **XGBoost Regressor** | $1,686.50 | 16,877,130.61 | $4,108.18 | 0.6725 | 2 |
| **Random Forest Regressor (Ensemble)** | **$1,500.27** | **16,013,133.05** | **$4,001.64** | **0.6893** | **1 (Best)** |

### Analytical Rationale:
Linear Regression fails to capture threshold effects ($R^2 = 0.1554$), whereas non-linear tree models excel because customer value depends on multi-variable interactions (e.g. number of policies interacting with monthly premium rates and vehicle tiers). **Random Forest** achieved the lowest prediction error and the highest explained variance ($R^2 = 0.6893$), closely followed by **XGBoost** ($R^2 = 0.6725$) and **Gradient Boosting** ($R^2 = 0.6712$).

---

## 6. K-Means Customer Segmentation ($k=3$)

Unsupervised K-Means clustering was executed on `Income`, `Monthly Premium Auto`, and `Total Claim Amount` without utilizing the target label:

| Cluster | Segment Profile | Mean CLV | Mean Income | Mean Premium | Operational Strategy |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **Cluster 0** | Moderate-Income Regulars | $7,410.71 | $14,853.37 | $87.42 | Digital self-service, automated renewal incentives |
| **Cluster 1** | High-Income Low-Claim Cohort | $7,569.92 | $66,276.08 | $85.19 | Coverage upselling, multi-vehicle policy bundling |
| **Cluster 2** | High-Premium VIP Patrons | **$14,250.53** | $26,599.45 | **$152.38** | Dedicated concierge account manager, priority claims |

---

## 7. Streamlit Web Application Prototype (`app.py`)

An interactive web application was deployed to bridge the gap between machine learning models and operational business decisions:
- **Interactive Attribute Sliders:** Dynamically configure monthly premium, policy count, claim expenses, income, coverage level, and vehicle class.
- **Real-Time Prediction:** Instant calculation of estimated CLV using the trained Random Forest model.
- **Customer Tier & Segment Badge:** Displays value category (VIP, Moderate, Budget) and K-Means cluster assignment.
- **Actionable Business Recommendations:** Prescribes budget caps for CAC/retention and tailored upselling actions.
- **Model Comparison & Dataset Explorer Tabs:** Allows stakeholders to inspect benchmarking charts and explore raw Kaggle records.

---

## 8. Conclusion & Future Scope

### Conclusions:
- Machine learning regression models reliably predict customer lifetime financial value, with ensemble methods explaining ~69% of variance.
- Tree-based models substantially outperform linear models due to strong non-linear interactions between policy volume, coverage level, and premium amounts.
- Unsupervised K-Means clustering isolates a high-value customer cohort ($14,250 average value), enabling resource prioritization.

### Future Scope:
- Implement BG/NBD and Gamma-Gamma probabilistic models for transaction frequency and monetary estimation.
- Introduce dynamic time-to-event survival analysis to forecast policy cancellation probabilities.
- Integrate real-time webhook APIs into enterprise CRM systems (Salesforce, HubSpot) for automated tier updates.
