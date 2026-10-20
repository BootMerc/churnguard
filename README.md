<div align="center">

```
                                                ╔══════════════════════════════════════════════════════════════════╗
                                                ║                                                                  ║
                                                ║        ██████╗██╗  ██╗██╗   ██╗██████╗ ███╗   ██╗              ║
                                                ║       ██╔════╝██║  ██║██║   ██║██╔══██╗████╗  ██║              ║
                                                ║       ██║     ███████║██║   ██║██████╔╝██╔██╗ ██║              ║
                                                ║       ██║     ██╔══██║██║   ██║██╔══██╗██║╚██╗██║              ║
                                                ║       ╚██████╗██║  ██║╚██████╔╝██║  ██║██║ ╚████║              ║
                                                ║        ╚═════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝              ║
                                                ║                  G U A R D   A I  — v 2.0                       ║
                                                ║            Customer Churn Prediction Intelligence                ║
                                                ╚══════════════════════════════════════════════════════════════════╝
```

# 🔮 ChurnGuard AI — Customer Churn Prediction

**End-to-end ML system for subscription-based customer churn prediction.**
Built with LightGBM · SMOTE · SHAP · Streamlit · Scikit-learn

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-brightgreen?style=flat-square)](https://lightgbm.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b?style=flat-square&logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?style=flat-square)](https://scikit-learn.org)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple?style=flat-square)](https://shap.readthedocs.io)

</div>

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Business Problem](#-business-problem--motivation)
- [Dataset Columns](#-dataset-columns-kaggle)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Model Performance](#-model-performance)
- [Installation & Setup](#-installation--setup)
- [Running the Project](#-running-the-project)
- [Streamlit Dashboard](#-streamlit-dashboard)
- [Feature Engineering](#-feature-engineering)
- [Imbalanced Data Strategy](#-imbalanced-data-strategy)
- [SHAP Explainability](#-shap-explainability)
- [Tech Stack](#-tech-stack)

---

## 🎯 Project Overview

ChurnGuard AI is a **portfolio-ready, production-style ML pipeline** built on the Kaggle
Customer Churn Dataset. It covers the full ML lifecycle end-to-end:

| Phase | Detail |
|-------|--------|
| **EDA** | Distributions, correlations, churn-rate breakdowns by every feature |
| **Preprocessing** | Imputation → scaling/encoding via leak-free sklearn Pipelines |
| **Feature Engineering** | 6 new features: SpendPerMonth, EngagementRate, RiskScore, LongTermCustomer, Inactive, HighSpender |
| **Imbalanced Data** | 4 strategies compared: SMOTE, Class Weights, RandomOver/UnderSampler |
| **Model Comparison** | 7 classifiers: LR · RF · GB · XGBoost · LightGBM · CatBoost · SVM |
| **Hyperparameter Tuning** | RandomizedSearchCV — 30 iterations × 5-fold StratifiedKFold |
| **Threshold Optimization** | Decision threshold tuned for maximum F1 score |
| **Explainability** | SHAP TreeExplainer — global beeswarm + bar plots |
| **Deployment** | Streamlit dashboard — single prediction + batch CSV scoring |

---

## 💼 Business Problem & Motivation

> **Acquiring a new customer costs 5–7× more than retaining an existing one.**

Proactively identifying at-risk customers lets retention teams intervene with:
targeted discounts, satisfaction surveys, contract upgrade offers, or direct outreach.

### Why Recall > Accuracy?

| Error Type | Business Cost |
|-----------|--------------|
| **False Negative** — missed churner | ❌ Customer leaves → permanent revenue loss |
| **False Positive** — wrong alert | ⚠️ Wasted retention outreach → minor cost |

The cost asymmetry means we **optimize for Recall and ROC-AUC**, not accuracy.
The classification threshold is tuned to maximize F1 rather than using a naive 0.50 cutoff.

---

## 📊 Dataset Columns (Kaggle)

| Column | Type | Description |
|--------|------|-------------|
| `CustomerID` | ID | Dropped before modeling |
| `Age` | Numerical | Customer age |
| `Gender` | Categorical | Male / Female |
| `Tenure` | Numerical | Months as a customer |
| `Usage Frequency` | Numerical | Product interactions per month |
| `Support Calls` | Numerical | Inbound support contacts |
| `Payment Delay` | Numerical | Days of billing delay |
| `Subscription Type` | Categorical | Basic / Standard / Premium |
| `Contract Length` | Categorical | Monthly / Quarterly / Annual |
| `Total Spend` | Numerical | Lifetime customer spend ($) |
| `Last Interaction` | Numerical | Days since last contact |
| `Churn` | **Target** | 0 = Retained, 1 = Churned |

> ✅ Column names with spaces are handled automatically — no manual renaming needed.

---

## 📁 Project Structure

```
churnguard-ai/
│
├── 📓 customer_churn_prediction.ipynb   ← Full ML pipeline (29 executed cells)
├── 🚀 app.py                            ← Streamlit dashboard (4 pages)
├── 📦 churn_model.joblib               ← Serialized model + pipeline artifacts
├── 📋 model_metadata.json              ← Metrics & column config (consumed by app)
├── 📄 requirements.txt                 ← All Python dependencies
├── 📖 README.md                        ← This file
│
└── 📊 Data (place in project root)
    ├── customer_churn_dataset-training-master.csv
    └── customer_churn_dataset-testing-master.csv
```

---

## 🔬 Methodology

### Preprocessing Pipeline

```
Raw Kaggle CSV
    │
    ├── Rename columns (spaces → underscores internally)
    ├── Drop CustomerID
    │
    ├── Feature Engineering (+6 derived features)
    │
    └── ColumnTransformer (inside ImbPipeline — no leakage)
         ├── Numerical  → SimpleImputer(median) → StandardScaler
         └── Categorical → SimpleImputer(mode)  → OneHotEncoder
              │
              ↓
           SMOTE (minority class oversampling)
              │
              ↓
           LightGBM Classifier (tuned)
              │
              ↓
           Threshold Optimization → Final Prediction
```

### Models Compared

| Model | Imbalance Handling |
|-------|-------------------|
| Logistic Regression | class_weight='balanced' |
| Random Forest | class_weight='balanced' |
| Gradient Boosting | — |
| XGBoost | scale_pos_weight |
| **LightGBM** ✅ | class_weight='balanced' |
| CatBoost | class_weights={0:1, 1:3} |
| SVM (RBF) | class_weight='balanced' |

---

## 📈 Model Performance

### Final Model: LightGBM (Tuned + SMOTE)

| Metric | Score |
|--------|-------|
| **ROC-AUC** | **1.0000** |
| **F1 Score** | **0.9976** |
| **Recall** | **0.9964** |
| **Precision** | **0.9988** |
| **Accuracy** | **0.9987** |
| Optimal Threshold | **0.34** |

### Resampling Strategy Comparison

| Strategy | AUC | F1 | Recall |
|----------|-----|----|--------|
| Class Weights | 0.979 | 0.968 | 0.961 |
| **SMOTE** ✅ | **0.999** | **0.998** | **0.996** |
| RandomOverSampler | 0.997 | 0.995 | 0.993 |
| RandomUnderSampler | 0.981 | 0.971 | 0.965 |

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.9+
- pip

### 1 — Clone / download the project
```bash
git clone https://github.com/your-username/churnguard-ai.git
cd churnguard-ai
```

### 2 — Virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3 — Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Step 1 — Place the Kaggle CSVs in the project root
```
customer_churn_dataset-training-master.csv
customer_churn_dataset-testing-master.csv
```
The notebook reads them by filename — no path changes needed.

### Step 2 — Run the notebook
```bash
jupyter notebook customer_churn_prediction.ipynb
```
Run **all cells top-to-bottom**. The notebook will:
- Perform complete EDA and save 16 plots
- Compare 7 classifiers with SMOTE
- Tune LightGBM with 30-iter RandomizedSearchCV
- Optimize the decision threshold
- Compute SHAP values
- Save `churn_model.joblib` and `model_metadata.json`

> ⏱️ Expected runtime: **8–15 min** depending on CPU

### Step 3 — Launch the Streamlit app
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

> ⚠️ The notebook **must be run first** to generate `churn_model.joblib`.

---

## 🌐 Streamlit Dashboard

Four pages, dark futuristic cyberpunk theme:

| Page | Features |
|------|----------|
| 🔮 **Single Prediction** | 10-field input form · Animated gauge · Risk level badge · Churn probability · Risk factor breakdown |
| 📤 **Batch Prediction** | CSV upload (Kaggle format) · KPI cards · Risk distribution chart · Probability histogram · Download scored CSV |
| 📊 **Model Intelligence** | ROC & PR curves · Threshold optimization chart · Feature importance · Strategy comparison · 7-model comparison |
| ℹ️ **System Info** | Pipeline architecture · Column reference · Engineered features · Tech stack |

### Design System
- **Fonts:** Orbitron (headers) + Rajdhani (body) + Share Tech Mono (code/badges)
- **Colors:** Cyan `#00d4ff` · Pink `#ff6b9d` · Violet `#c77dff` · Amber `#ffb347` · Green `#56ff9e`
- **FX:** Animated CSS grid background · Glassmorphism cards · Scanline overlay · Glowing borders

---

## 🧬 Feature Engineering

```python
# Normalized spend rate
df['SpendPerMonth']    = df['TotalSpend'] / (df['Tenure'] + 1)

# Product engagement per month of tenure
df['EngagementRate']   = df['UsageFrequency'] / (df['Tenure'] + 1)

# Composite risk signal
df['RiskScore']        = df['SupportCalls'] + df['PaymentDelay'] * 1.5

# Loyalty milestone flag
df['LongTermCustomer'] = (df['Tenure'] > 24).astype(int)

# Recency disengagement flag
df['Inactive']         = (df['LastInteraction'] > 20).astype(int)

# High-value customer flag
df['HighSpender']      = (df['TotalSpend'] > df['TotalSpend'].median()).astype(int)
```

---

## ⚖️ Imbalanced Data Strategy

SMOTE was selected after comparing all four strategies:

```
Strategy              Pipeline
──────────────────────────────────────────────────────────────────
Class Weights     →   Preprocessor → LightGBM(class_weight='balanced')
SMOTE ✅ (best)   →   Preprocessor → SMOTE → LightGBM
RandomOverSampler →   Preprocessor → RandomOverSampler → LightGBM
RandomUnderSampler→   Preprocessor → RandomUnderSampler → LightGBM
```

All resampling runs **inside the ImbPipeline**, applied only to training folds during
cross-validation — preventing any data leakage from the validation/test sets.

---

## 🔬 SHAP Explainability

SHAP (SHapley Additive exPlanations) explains individual predictions and global feature impact:

```
Top churn drivers identified by SHAP:

  Support Calls      ↑ High calls = strong churn push
  Payment Delay      ↑ Delays = billing disengagement
  Contract (Monthly) ↑ Present = strong churn push
  Tenure             ↑ Short = churn risk
  Usage Frequency    ↓ Low usage = disengagement
  RiskScore          ↑ High composite = churn push
  Last Interaction   ↑ Long silence = churn precursor
  Subscription Basic ↑ Low switching cost
```

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| pandas | ≥2.0 | Data wrangling & EDA |
| numpy | ≥1.24 | Numerical operations |
| scikit-learn | ≥1.3 | Pipelines, preprocessing, metrics, tuning |
| lightgbm | ≥4.0 | Primary classifier |
| xgboost | ≥2.0 | Comparison classifier |
| catboost | ≥1.2 | Comparison classifier |
| imbalanced-learn | ≥0.11 | SMOTE & resampling strategies |
| shap | ≥0.43 | Model explainability |
| matplotlib | ≥3.7 | Static plots (notebook) |
| seaborn | ≥0.12 | Heatmaps & styled charts |
| plotly | ≥5.17 | Interactive charts (app) |
| streamlit | ≥1.28 | Web dashboard |
| joblib | ≥1.3 | Model serialization |

---

## 📝 License

MIT License — free to use, modify, and distribute.

---

<div align="center">

**Built for GitHub portfolios · Internship applications · University projects**

```
⬡ CHURNGUARD AI — PREDICTING CHURN. PRESERVING REVENUE. ⬡
```

</div>
