"""
Project: Valor - Value Assessment & Lifetime Outlook via Regression
Script: app.py
Dataset: IBM Watson Marketing Customer Value Analysis (Kaggle)
Description: Interactive Streamlit Web Application for Customer Lifetime Value (CLV) Prediction.
             Allows users to input customer policy attributes and obtain predicted CLV,
             customer tier segmentation, K-Means clustering assignment, and marketing recommendations.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ==========================================
# Page Configuration (Clean, No Emojis)
# ==========================================
st.set_page_config(
    page_title="Valor | Customer Lifetime Value Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# Custom CSS Design System
# Based on Warm Terracotta & Editorial Palette
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Geist+Mono:wght@400;500;600;700&display=swap');

:root {
  --card: #f5f4ef;
  --ring: #c96442;
  --input: #b4b2a7;
  --muted: #ede9de;
  --accent: #e9e6dc;
  --border: #dad9d4;
  --radius: 0.85rem;
  --chart-1: #b05730;
  --chart-2: #9c87f5;
  --chart-3: #ded8c4;
  --chart-4: #dbd3f0;
  --chart-5: #b4552d;
  --popover: #ffffff;
  --primary: #c96442;
  --sidebar: #f5f4ee;
  --spacing: 0.25rem;
  --font-mono: 'Geist Mono', ui-monospace, monospace;
  --font-sans: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
  --secondary: #e9e6dc;
  --background: #faf9f5;
  --foreground: #3d3929;
  --destructive: #141413;
  --card-foreground: #141413;
  --muted-foreground: #6e6d68;
  --accent-foreground: #28261b;
  --popover-foreground: #28261b;
  --primary-foreground: #ffffff;
}

/* Global App Container */
html, body, [data-testid="stAppViewContainer"], .stApp {
    background-color: var(--background) !important;
    color: var(--foreground) !important;
    font-family: var(--font-sans) !important;
}

header[data-testid="stHeader"] {
    background-color: var(--background) !important;
}

/* Typography Overrides */
h1, h2, h3, h4, h5, h6 {
    font-family: var(--font-sans) !important;
    color: var(--foreground) !important;
    letter-spacing: -0.02em !important;
    font-weight: 600 !important;
}

p, span, label, div {
    font-family: var(--font-sans);
}

/* Header & Brand Banner */
.brand-pill {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--primary);
    background-color: var(--muted);
    border: 1px solid var(--border);
    padding: 3px 10px;
    border-radius: 9999px;
    margin-bottom: 0.5rem;
}

.main-title {
    font-size: 2.15rem;
    font-weight: 700;
    color: var(--foreground);
    margin-bottom: 0.25rem;
    letter-spacing: -0.025em;
    line-height: 1.2;
}

.sub-title {
    font-size: 0.95rem;
    color: var(--muted-foreground);
    margin-bottom: 1.5rem;
    line-height: 1.5;
}

.sub-title b {
    color: var(--foreground);
    font-weight: 500;
}

/* Navigation Tabs */
div[data-baseweb="tab-list"] {
    background-color: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 1.75rem !important;
    padding-bottom: 0px !important;
    margin-bottom: 1.5rem !important;
}

button[data-baseweb="tab"] {
    font-family: var(--font-sans) !important;
    font-weight: 500 !important;
    font-size: 0.92rem !important;
    color: var(--muted-foreground) !important;
    background-color: transparent !important;
    border: none !important;
    padding: 10px 0 !important;
    transition: color 0.15s ease !important;
}

button[data-baseweb="tab"]:hover {
    color: var(--foreground) !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--primary) !important;
    font-weight: 600 !important;
    border-bottom: 2px solid var(--primary) !important;
}

/* Cards & Surfaces */
.clv-card {
    background-color: var(--card);
    border: 1px solid var(--border);
    border-top: 3px solid var(--primary);
    border-radius: var(--radius);
    padding: 26px 22px;
    text-align: center;
    margin-bottom: 1.2rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

.clv-eyebrow {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted-foreground);
    margin-bottom: 6px;
}

.clv-value {
    font-family: var(--font-mono);
    font-size: 2.75rem;
    font-weight: 700;
    color: var(--card-foreground);
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin-bottom: 12px;
}

.tier-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.78rem;
    letter-spacing: 0.02em;
    border: 1px solid var(--border);
}

.tier-vip {
    background-color: #f2ece4;
    color: #9c4c2e;
    border-color: #dec8b8;
}

.tier-mod {
    background-color: var(--muted);
    color: var(--secondary-foreground);
    border-color: var(--border);
}

.tier-budget {
    background-color: var(--accent);
    color: var(--muted-foreground);
    border-color: var(--border);
}

/* Mini Metric Cards */
.mini-metric {
    background-color: var(--card);
    border: 1px solid var(--border);
    border-radius: 0.65rem;
    padding: 14px 16px;
    height: 100%;
}

.mini-metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted-foreground);
    margin-bottom: 4px;
}

.mini-metric-val {
    font-family: var(--font-mono);
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--card-foreground);
    margin-bottom: 2px;
}

.mini-metric-sub {
    font-size: 0.78rem;
    color: var(--primary);
    font-weight: 500;
}

/* Strategy Box */
.strategy-box {
    background-color: var(--card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--primary);
    border-radius: 0.65rem;
    padding: 16px 18px;
    margin-top: 0.8rem;
    margin-bottom: 1.25rem;
}

.strategy-title {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--primary);
    margin-bottom: 6px;
}

.strategy-text {
    font-size: 0.9rem;
    color: var(--foreground);
    line-height: 1.5;
}

/* Factor Row Items */
.factors-card {
    background-color: var(--card);
    border: 1px solid var(--border);
    border-radius: 0.65rem;
    padding: 6px 16px;
    margin-top: 0.6rem;
}

.factor-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 9px 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.88rem;
}

.factor-item:last-child {
    border-bottom: none;
}

.factor-item-label {
    color: var(--muted-foreground);
}

.factor-item-val {
    font-family: var(--font-mono);
    font-weight: 500;
    color: var(--foreground);
}

/* Cluster Profile Cards */
.cluster-card {
    background-color: var(--card);
    border: 1px solid var(--border);
    border-radius: 0.65rem;
    padding: 16px 18px;
    margin-bottom: 12px;
}

.cluster-card-title {
    font-size: 0.92rem;
    font-weight: 600;
    color: var(--foreground);
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

.cluster-card-tag {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--muted-foreground);
    background-color: var(--muted);
    border: 1px solid var(--border);
    padding: 2px 8px;
    border-radius: 9999px;
}

.cluster-card-metrics {
    font-size: 0.85rem;
    color: var(--muted-foreground);
    line-height: 1.5;
    margin-bottom: 6px;
}

.cluster-card-strategy {
    font-size: 0.82rem;
    color: var(--foreground);
}

/* Streamlit Widget Elements */
div[data-testid="stExpander"] {
    background-color: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    box-shadow: none !important;
    margin-top: 0.5rem;
}

div[data-testid="stExpander"] details summary {
    font-family: var(--font-sans) !important;
    font-weight: 500 !important;
    color: var(--foreground) !important;
}

div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border-color: var(--border) !important;
    border-radius: 0.5rem !important;
    font-family: var(--font-sans) !important;
}

div[data-baseweb="select"] > div:focus-within {
    border-color: var(--ring) !important;
    box-shadow: 0 0 0 1px var(--ring) !important;
}

/* Sliders */
.stSlider div[data-testid="stThumbValue"] {
    font-family: var(--font-mono) !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# Path Resolution Helper
# ==========================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))

def resolve_path(filename, subfolder=""):
    candidates = [
        os.path.join(APP_DIR, subfolder, filename) if subfolder else "",
        os.path.join(PROJECT_ROOT, subfolder, filename) if subfolder else "",
        os.path.join(APP_DIR, filename),
        os.path.join(PROJECT_ROOT, filename),
        filename
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return filename

# ==========================================
# Load Saved Artifacts
# ==========================================
@st.cache_resource
def load_artifacts():
    best_model = joblib.load(resolve_path("best_model.pkl", "models"))
    scaler = joblib.load(resolve_path("scaler.pkl", "models"))
    label_encoders = joblib.load(resolve_path("label_encoders.pkl", "models"))
    kmeans = joblib.load(resolve_path("kmeans_model.pkl", "models"))
    cluster_scaler = joblib.load(resolve_path("cluster_scaler.pkl", "models"))
    with open(resolve_path("feature_cols.json", "data")) as f:
        feature_cols = json.load(f)

    # Optional / non-blocking load for all_models (prevents cross-version Cython _loss unpickling crashes)
    all_models = {}
    try:
        all_models_path = resolve_path("all_models.pkl", "models")
        if os.path.exists(all_models_path):
            all_models = joblib.load(all_models_path)
    except Exception:
        all_models = {}

    return best_model, all_models, scaler, label_encoders, kmeans, cluster_scaler, feature_cols

@st.cache_data
def load_dataset():
    csv_path = resolve_path("customer_clv_data.csv", "data")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

@st.cache_data
def load_comparison_data():
    comp_path = resolve_path("model_comparison.csv", "data")
    if os.path.exists(comp_path):
        return pd.read_csv(comp_path)
    return None

try:
    best_model, all_models, scaler, label_encoders, kmeans, cluster_scaler, feature_cols = load_artifacts()
    df_raw = load_dataset()
    comparison_df = load_comparison_data()
except Exception as e:
    st.error(f"Error loading models or artifacts: {e}. Please ensure model files are present.")
    st.stop()


# ==========================================
# App Header (Editorial, No Emojis)
# ==========================================
st.markdown('<div class="brand-pill">Valor Analytics Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">Customer Lifetime Value Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title"><b>Dataset:</b> Kaggle IBM Watson Marketing Customer Value Analysis &nbsp;|&nbsp; <b>Domain:</b> Customer Intelligence & Predictive Modeling</div>', unsafe_allow_html=True)

# Main Navigation Tabs (Clean, No Emojis)
tab_predict, tab_comparison, tab_clusters, tab_dataset = st.tabs([
    "Predict Customer CLV",
    "Model Comparative Study",
    "K-Means Customer Segments",
    "Dataset Explorer"
])

# ==========================================
# TAB 1: PREDICTION INTERFACE
# ==========================================
with tab_predict:
    col_input, col_output = st.columns([1.1, 1.2], gap="large")

    with col_input:
        st.subheader("Policyholder & Demographics Input")
        st.caption("Configure the policyholder attributes below to calculate predicted customer lifetime value.")

        c1, c2 = st.columns(2)
        with c1:
            monthly_premium = st.slider("Monthly Premium Auto ($)", 60, 300, 95, help="Monthly insurance premium.")
            num_policies = st.slider("Number of Policies", 1, 9, 2, help="Active insurance policies held.")
            total_claim = st.slider("Total Claim Amount ($)", 0, 3000, 420, help="Cumulative historical claim amount.")
            income = st.slider("Annual Income ($)", 0, 100000, 48000, step=1000)

        with c2:
            coverage = st.selectbox("Coverage Level", label_encoders['Coverage'].classes_, index=1)
            vehicle_class = st.selectbox("Vehicle Class", label_encoders['Vehicle Class'].classes_, index=0)
            employment = st.selectbox("Employment Status", label_encoders['EmploymentStatus'].classes_, index=1)
            education = st.selectbox("Education Level", label_encoders['Education'].classes_, index=0)

        with st.expander("Additional Policy Attributes"):
            c3, c4 = st.columns(2)
            with c3:
                state = st.selectbox("State", label_encoders['State'].classes_, index=1)
                gender = st.selectbox("Gender", label_encoders['Gender'].classes_, index=0)
                marital = st.selectbox("Marital Status", label_encoders['Marital Status'].classes_, index=1)
                location = st.selectbox("Location Code", label_encoders['Location Code'].classes_, index=1)
                policy_type = st.selectbox("Policy Type", label_encoders['Policy Type'].classes_, index=1)
            with c4:
                policy = st.selectbox("Policy", label_encoders['Policy'].classes_, index=4)
                renew_offer = st.selectbox("Renew Offer Type", label_encoders['Renew Offer Type'].classes_, index=0)
                sales_channel = st.selectbox("Sales Channel", label_encoders['Sales Channel'].classes_, index=0)
                vehicle_size = st.selectbox("Vehicle Size", label_encoders['Vehicle Size'].classes_, index=1)
                response = st.selectbox("Marketing Campaign Response", label_encoders['Response'].classes_, index=0)
                months_last_claim = st.slider("Months Since Last Claim", 0, 35, 15)
                months_inception = st.slider("Months Since Policy Inception", 0, 99, 45)
                open_complaints = st.slider("Number of Open Complaints", 0, 5, 0)

        # Build feature dictionary
        input_data = {
            "State": state,
            "Response": response,
            "Coverage": coverage,
            "Education": education,
            "EmploymentStatus": employment,
            "Gender": gender,
            "Income": income,
            "Location Code": location,
            "Marital Status": marital,
            "Monthly Premium Auto": monthly_premium,
            "Months Since Last Claim": months_last_claim,
            "Months Since Policy Inception": months_inception,
            "Number of Open Complaints": open_complaints,
            "Number of Policies": num_policies,
            "Policy Type": policy_type,
            "Policy": policy,
            "Renew Offer Type": renew_offer,
            "Sales Channel": sales_channel,
            "Total Claim Amount": total_claim,
            "Vehicle Class": vehicle_class,
            "Vehicle Size": vehicle_size
        }

        # Encode categorical variables
        encoded_dict = {}
        for col in feature_cols:
            val = input_data[col]
            if col in label_encoders:
                encoded_dict[col] = label_encoders[col].transform([str(val)])[0]
            else:
                encoded_dict[col] = float(val)

        input_df = pd.DataFrame([encoded_dict])[feature_cols]

    with col_output:
        st.subheader("Prediction & Strategic Intelligence")

        # Predict using Best Model (Random Forest)
        predicted_clv = float(best_model.predict(input_df)[0])
        predicted_clv = max(predicted_clv, 1500.0)

        # Cluster prediction using K-Means
        cluster_input = np.array([[income, monthly_premium, total_claim]])
        cluster_input_scaled = cluster_scaler.transform(cluster_input)
        assigned_cluster = int(kmeans.predict(cluster_input_scaled)[0])

        cluster_labels = {
            0: ("Moderate-Income Regular", "Moderate Tier"),
            1: ("High-Income Policyholder", "Affluent Tier"),
            2: ("High-Premium VIP", "Prime Value Tier")
        }
        cluster_name, cluster_tier = cluster_labels.get(assigned_cluster, ("Standard Tier", "Base Tier"))

        # Value Tier classification
        if predicted_clv >= 12000:
            tier_class = "tier-vip"
            tier_name = "VIP High-Value Customer"
            retention_budget = "$750 - $1,200"
            rec_text = "Assign dedicated account manager, prioritize concierge claim processing, and offer exclusive multi-vehicle premium discounts."
        elif predicted_clv >= 6000:
            tier_class = "tier-mod"
            tier_name = "Moderate-Value Customer"
            retention_budget = "$300 - $600"
            rec_text = "Deploy targeted coverage upselling (Extended -> Premium), bundled insurance promotions, and semi-annual loyalty check-ins."
        else:
            tier_class = "tier-budget"
            tier_name = "Budget / Developing Customer"
            retention_budget = "$100 - $250"
            rec_text = "Engage via automated digital self-service channels, digital renewal reminders, and cross-sell entry-level auto add-ons."

        # Hero CLV Card
        st.markdown(f"""
        <div class="clv-card">
            <div class="clv-eyebrow">Estimated Customer Lifetime Value</div>
            <div class="clv-value">${predicted_clv:,.2f}</div>
            <div class="tier-badge {tier_class}">{tier_name}</div>
        </div>
        """, unsafe_allow_html=True)

        # Metrics Row
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="mini-metric">
                <div class="mini-metric-label">K-Means Cluster</div>
                <div class="mini-metric-val">Cluster {assigned_cluster}</div>
                <div class="mini-metric-sub">{cluster_name}</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="mini-metric">
                <div class="mini-metric-label">Retention Budget</div>
                <div class="mini-metric-val">{retention_budget}</div>
                <div class="mini-metric-sub">Recommended Allocation</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="mini-metric">
                <div class="mini-metric-label">Annualized Premium</div>
                <div class="mini-metric-val">${monthly_premium * 12:,.0f}/yr</div>
                <div class="mini-metric-sub">${monthly_premium}/mo billing</div>
            </div>
            """, unsafe_allow_html=True)

        # Strategic Action Callout
        st.markdown(f"""
        <div class="strategy-box">
            <div class="strategy-title">Recommended Strategic Action</div>
            <div class="strategy-text">{rec_text}</div>
        </div>
        """, unsafe_allow_html=True)

        # Key Driving Factors Table
        prem_status = "Above Average" if monthly_premium > 93 else "Standard"
        policy_status = "Multi-Policy Account" if num_policies > 1 else "Single Policy"

        st.markdown(f"""
        <div style="font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--foreground); margin-top: 1rem; margin-bottom: 4px;">
            Key Contributing Attributes
        </div>
        <div class="factors-card">
            <div class="factor-item">
                <span class="factor-item-label">Monthly Premium</span>
                <span class="factor-item-val">${monthly_premium} ({prem_status})</span>
            </div>
            <div class="factor-item">
                <span class="factor-item-label">Active Policies</span>
                <span class="factor-item-val">{num_policies} ({policy_status})</span>
            </div>
            <div class="factor-item">
                <span class="factor-item-label">Coverage Tier</span>
                <span class="factor-item-val">{coverage}</span>
            </div>
            <div class="factor-item">
                <span class="factor-item-label">Vehicle Classification</span>
                <span class="factor-item-val">{vehicle_class}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# TAB 2: MODEL COMPARISON
# ==========================================
with tab_comparison:
    st.subheader("Regression Algorithms Benchmark Study")
    st.caption("Evaluation across 7,307 training records (80%) and 1,827 unseen holdout test samples (20%).")

    if comparison_df is not None:
        st.dataframe(comparison_df.style.highlight_min(subset=['MAE', 'RMSE'], color='#ede9de')
                                        .highlight_max(subset=['R2 Score'], color='#e9e6dc'), use_container_width=True)

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            fig1, ax1 = plt.subplots(figsize=(6, 3.8))
            fig1.patch.set_facecolor('#faf9f5')
            ax1.set_facecolor('#f5f4ef')
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['left'].set_color('#dad9d4')
            ax1.spines['bottom'].set_color('#dad9d4')
            ax1.tick_params(colors='#3d3929', which='both', labelsize=9)
            ax1.grid(axis='y', linestyle='--', alpha=0.5, color='#dad9d4')

            bar_width = 0.35
            models = comparison_df["Model"]
            x = np.arange(len(models))

            ax1.bar(x - bar_width/2, comparison_df["MAE"], width=bar_width, label='MAE ($)', color='#b05730', edgecolor='none')
            ax1.bar(x + bar_width/2, comparison_df["RMSE"], width=bar_width, label='RMSE ($)', color='#9c87f5', edgecolor='none')

            ax1.set_title("Error Metrics (MAE & RMSE in USD) - Lower is Better", fontsize=10, fontweight='600', color='#3d3929', pad=12)
            ax1.set_ylabel("USD ($)", fontsize=9, color='#3d3929')
            ax1.set_xticks(x)
            ax1.set_xticklabels(models, rotation=20, ha='right')
            ax1.legend(frameon=False, fontsize=8)
            plt.tight_layout()
            st.pyplot(fig1)

        with col_b2:
            fig2, ax2 = plt.subplots(figsize=(6, 3.8))
            fig2.patch.set_facecolor('#faf9f5')
            ax2.set_facecolor('#f5f4ef')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['left'].set_color('#dad9d4')
            ax2.spines['bottom'].set_color('#dad9d4')
            ax2.tick_params(colors='#3d3929', which='both', labelsize=9)
            ax2.grid(axis='y', linestyle='--', alpha=0.5, color='#dad9d4')

            ax2.bar(models, comparison_df["R2 Score"], width=0.45, color='#c96442', edgecolor='none')
            ax2.set_title("Variance Explained (R2 Score) - Higher is Better", fontsize=10, fontweight='600', color='#3d3929', pad=12)
            ax2.set_ylabel("R2 Score", fontsize=9, color='#3d3929')
            ax2.set_ylim(0, 1.0)
            ax2.set_xticklabels(models, rotation=20, ha='right')
            plt.tight_layout()
            st.pyplot(fig2)

    st.markdown("""
    <div class="strategy-box">
        <div class="strategy-title">Benchmark Insight</div>
        <div class="strategy-text">
            <b>Random Forest Regressor</b> achieves top performance with an R2 of <b>0.6893</b> and Mean Absolute Error of <b>$1,500.27</b>. Linear Regression achieves an R2 of only <b>0.1554</b>, confirming that customer lifetime value is governed by non-linear relationships across policy counts, coverage levels, and vehicle classes.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# TAB 3: K-MEANS CUSTOMER CLUSTERS
# ==========================================
with tab_clusters:
    st.subheader("Unsupervised Customer Segmentation (K-Means, k=3)")
    st.caption("Clustering executed on Income, Monthly Premium Auto, and Total Claim Amount without target labels.")

    c_img1, c_img2 = st.columns([1.2, 1])
    with c_img1:
        cluster_plot_path = resolve_path("kmeans_customer_clusters.png", "plots")
        if os.path.exists(cluster_plot_path):
            st.image(cluster_plot_path, caption="Customer Segmentation: Income vs Monthly Premium", use_container_width=True)
    with c_img2:
        st.markdown("""
        <div class="cluster-card">
            <div class="cluster-card-title">
                <span>Cluster 0: Moderate-Income Regulars</span>
                <span class="cluster-card-tag">Core Base</span>
            </div>
            <div class="cluster-card-metrics">
                <b>Mean CLV:</b> $7,410.71 &nbsp;|&nbsp; <b>Mean Income:</b> ~$14,850 &nbsp;|&nbsp; <b>Monthly Premium:</b> ~$87/mo
            </div>
            <div class="cluster-card-strategy">
                <b>Action:</b> Automated cross-selling and streamlined digital self-service policy management.
            </div>
        </div>

        <div class="cluster-card">
            <div class="cluster-card-title">
                <span>Cluster 1: High-Income Low-Claim Cohort</span>
                <span class="cluster-card-tag">Affluent</span>
            </div>
            <div class="cluster-card-metrics">
                <b>Mean CLV:</b> $7,569.92 &nbsp;|&nbsp; <b>Mean Income:</b> ~$66,275 &nbsp;|&nbsp; <b>Monthly Premium:</b> ~$85/mo
            </div>
            <div class="cluster-card-strategy">
                <b>Action:</b> Premium umbrella liability cross-selling, multi-line bundling, and wealth-tier campaigns.
            </div>
        </div>

        <div class="cluster-card" style="border-left: 3px solid var(--primary);">
            <div class="cluster-card-title">
                <span>Cluster 2: High-Premium VIP Patrons</span>
                <span class="cluster-card-tag" style="color: var(--primary); background-color: var(--accent);">High Value</span>
            </div>
            <div class="cluster-card-metrics">
                <b>Mean CLV:</b> $14,250.53 &nbsp;|&nbsp; <b>Mean Income:</b> ~$26,600 &nbsp;|&nbsp; <b>Monthly Premium:</b> ~$152/mo
            </div>
            <div class="cluster-card-strategy">
                <b>Action:</b> Dedicated executive relationship manager, expedited claim processing, and tailored retention incentives.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# TAB 4: DATASET EXPLORER
# ==========================================
with tab_dataset:
    st.subheader("Kaggle IBM Watson Marketing Customer Value Dataset")
    st.caption("Kaggle benchmark repository (9,134 records, 24 attributes).")

    if df_raw is not None:
        st.write(f"Sample records ({df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns):")
        st.dataframe(df_raw.head(50), use_container_width=True)

        st.subheader("Statistical Summary")
        st.dataframe(df_raw.describe(), use_container_width=True)
