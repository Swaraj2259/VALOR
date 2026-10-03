# 💎 Valor Web Application (Streamlit)

Interactive Customer Lifetime Value (CLV) Prediction & Strategic Customer Intelligence Dashboard.

## 🚀 Quick Launch

From the project root:
```bash
streamlit run website/app.py
```

Or from inside this directory:
```bash
cd website
streamlit run app.py
```

The application will be available at: `http://localhost:8501`

## 🌟 Features

1. **🎯 Predict Customer CLV**:
   - Interactive policyholder configuration sliders & selectors (Coverage, Premium, Vehicle Class, Income, Complaints).
   - Real-time prediction powered by the best-performing **Random Forest Regressor** ($R^2 = 0.6893$).
   - Dynamic tier segmentation: **VIP High-Value**, **Moderate-Value**, or **Budget / Developing**.
   - Strategic actionable marketing & retention budget recommendation.

2. **📊 Model Comparative Study**:
   - Comprehensive benchmark table comparing 5 algorithms (Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost).
   - Comparative bar charts for MAE, RMSE, and $R^2$ variance explained.

3. **🔍 K-Means Customer Segments**:
   - Unsupervised segmentation ($k=3$) based on Income, Monthly Premium, and Total Claim Amount.
   - High-resolution cluster visualizations and strategic cohort profiles.

4. **📁 Kaggle Dataset Explorer**:
   - Live data explorer for IBM Watson Marketing Customer Value Analysis (9,134 records).
   - Summary statistics and attribute distribution insights.
