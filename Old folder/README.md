# 🌾 Drought Prediction using Machine Learning and Ensemble Models with Explainable AI (SHAP & LIME)

A comprehensive machine learning pipeline for predicting drought conditions across Indian districts using meteorological features, ensemble models, and explainable AI techniques.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Visualizations](#visualizations)
- [Key Findings](#key-findings)
- [Future Scope](#future-scope)

---

## 🎯 Overview

This project implements a drought prediction system that:

- Analyzes meteorological data from **49 districts** across **5 Indian states** (Andhra Pradesh, Bihar, Gujarat, Karnataka, Maharashtra)
- Utilizes **time-lagged features** (up to 6 months prior) to capture temporal dependencies
- Compares multiple ML models: Random Forest, SVM, KNN, and XGBoost
- Addresses class imbalance using **SMOTE** and **weighted loss functions**
- Provides model interpretability using **SHAP** (SHapley Additive exPlanations) and **LIME** (Local Interpretable Model-agnostic Explanations)
- Generates **probabilistic policy insights** for drought mitigation

---

## 📊 Dataset

### Source
The dataset contains historical meteorological data from districts across five Indian states.

### Features

| Feature | Description |
|---------|-------------|
| `STATE` | Indian state name |
| `DISTRICT` | District name |
| `YEAR` | Year of observation |
| `mon` | Month of observation |
| `AvgTemp` | Average temperature |
| `PotentialEvapTran` | Potential evapotranspiration |
| `Precipitation` | Precipitation amount |
| `SPI` | Standardized Precipitation Index |
| `VapourPres` | Vapour pressure |
| `WetDayFreq` | Frequency of wet days |
| `CLASS` | Target variable (0: No Drought, 1: Drought) |

### Data Statistics
- **Total Records:** 29,988 observations
- **Districts:** 49 unique districts
- **States:** 5 (Andhra Pradesh, Bihar, Gujarat, Karnataka, Maharashtra)
- **Time Span:** Multiple decades of historical data
- **Class Distribution:** Imbalanced (drought events are minority class ~7%)

---

## 🔬 Methodology

### 1. Data Preprocessing
- Handled duplicate district names (AURANGABAD exists in both Bihar and Maharashtra)
- Label encoding for categorical variables (STATE, DISTRICT)
- Cyclical encoding for temporal features (month, year) using sine/cosine transformation
- Standard scaling for numerical features

### 2. Feature Engineering
- Created **time-lagged features** (t-1 to t-6 months) for all meteorological variables
- Generated 36 additional lagged features capturing historical patterns
- Total features after engineering: **42 features**

### 3. Models Evaluated
| Model | Description |
|-------|-------------|
| Random Forest | Ensemble of 400 decision trees |
| SVM | Support Vector Machine with RBF kernel |
| KNN | K-Nearest Neighbors |
| XGBoost | Gradient boosting with hyperparameter tuning |

### 4. Handling Class Imbalance
Three approaches compared:
1. **XGBoost with Weighted Loss** (scale_pos_weight parameter)
2. **XGBoost without SMOTE and Unweighted Loss**
3. **XGBoost with SMOTE** (Synthetic Minority Oversampling)

### 5. Hyperparameter Optimization
- **RandomizedSearchCV** with 3-fold cross-validation
- **StratifiedKFold** (5-fold) for model evaluation
- Optimized parameters: learning_rate, n_estimators, max_depth, min_child_weight, gamma, subsample, colsample_bytree, reg_alpha, reg_lambda

---

## 📁 Project Structure

```
Drought Prediction Final/
│
├── Code/
│   └── Drought_Prediction_using_ML_and_Ensemble_Models_alongwith_xAI_with_SHAP_&_LIME___Final_Notebook.ipynb
|   └── xgbmodel_files.pkl
│
├── Data/
│   └── final.csv                    # Raw dataset
│
├── Figures/
│   ├── Average Temperature Time series.png
│   ├── Potential Evapotranspiration Time series.png
│   ├── Precipitation Time series.png
│   ├── Standardized Precipitation Index (SPI) Time series.png
│   ├── Vapour Pressure Time series.png
│   ├── Frequency of Wet Days Time series.png
│   ├── Comparison of Model Performances.png
│   ├── Comparison of XGBoost Model Performance with and without SMOTE and Weighted Loss.png
│   ├── LIME Explain Plot Gujarat Patan Dec 2022 Drought.png
│   ├── LIME Explain Plot Bihar Siwan Oct 2022 Non-Drought.png
│   ├── lime_gujarat_drought.html
│   ├── lime_bihar_non_drought.html
│   ├── Predictions Jan 2023.png
│   └── Workflow diagram.png
│
├── Research Paper/
│   └── (Research paper documents)
│
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage

### Running the Notebook

1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Navigate to `Code/` folder and open the notebook

3. Update file paths in the notebook to point to local `Data/` directory:
   ```python
   # Change from Google Drive path:
   # df = pd.read_csv('/content/drive/MyDrive/drought_flood_prediction/final.csv')
   
   # To local path:
   df = pd.read_csv('../Data/final.csv')
   ```

4. Run cells sequentially

### Quick Start - Load Pre-trained Model

```python
import pickle

# Load saved model and preprocessors
with open('../Code/xgbmodel_files.pkl', 'rb') as f:
    xgbmodel_files = pickle.load(f)

best_xgb_model = xgbmodel_files[0]  # Trained XGBoost model
sc = xgbmodel_files[1]               # StandardScaler
le1 = xgbmodel_files[2]              # LabelEncoder for DISTRICT
le2 = xgbmodel_files[3]              # LabelEncoder for STATE
```
---

## 📊 Visualizations

The `Figures/` directory contains:

1. **Time Series Plots** - Historical trends of meteorological variables across states and districts
2. **Model Comparison Charts** - Performance metrics comparison across different models
3. **SHAP Plots** - Beeswarm plots showing feature importance and impact direction
4. **LIME Explanations** - Local interpretability for individual predictions
5. **Prediction Maps** - Folium-based interactive maps showing predicted drought regions

---

## 🔑 Key Findings

1. **XGBoost outperforms** traditional ML models for drought prediction
2. **Weighted loss function** is more effective than SMOTE for handling class imbalance
3. **SPI (Standardized Precipitation Index)** is the most important predictor, especially 1-2 months prior
4. **Precipitation patterns 6 months prior** have significant predictive power
5. **State-specific feature importance** varies, suggesting localized drought dynamics
6. **Uncertainty quantification** shows most predictions fall in low-uncertainty bins

---

## 🔮 Future Scope

- Integration of **satellite imagery** and remote sensing data
- **Real-time prediction pipeline** with automated data ingestion
- **Deep learning models** (LSTM, Transformer) for sequential patterns
- Expansion to **more states and districts** across India
- **Web dashboard** for visualization and monitoring
- Integration with **government early warning systems**

