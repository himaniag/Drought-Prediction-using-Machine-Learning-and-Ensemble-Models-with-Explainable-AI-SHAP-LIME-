# Drought-Prediction-using-Machine-Learning-and-Ensemble-Models-with-Explainable-AI-SHAP-LIME-
A comprehensive machine learning framework for district-level drought prediction across India using temporal feature engineering, ensemble learning, Explainable AI (SHAP & LIME), uncertainty quantification, probabilistic policy insights, and cross-dataset generalization.
## 📋 Table of Contents

- Overview
- Dataset
- Methodology
- Project Structure
- Code
- Installation
- Usage
- Experimental Results
- Explainable AI
- Cross-Dataset Generalization
- Visualizations
- Repository Outputs
- Key Findings
- Future Scope
- Citation
- ### Dataset Access

The dataset used in this study can be accessed from:

🔗 **Dataset Link**

```text
https://PASTE_YOUR_DATASET_LINK_HERE
## 🔬 Methodology

The proposed framework follows an end-to-end machine learning pipeline for drought prediction, from data preprocessing to model interpretation and cross-dataset validation.

### 1. Data Preprocessing

The raw meteorological dataset was preprocessed to ensure data quality and consistency.

- Handled duplicate district names (e.g., **AURANGABAD** in both Bihar and Maharashtra)
- Label encoding for categorical variables (`STATE`, `DISTRICT`)
- Cyclical encoding for temporal variables (`MONTH` and `YEAR`) using sine and cosine transformations
- Standard scaling of numerical features
- Data quality verification and preprocessing for model training

---

### 2. Feature Engineering

To capture temporal dependencies in drought evolution, historical lag-based features were generated.

- Created **time-lagged features (t−1 to t−6 months)** for all meteorological variables
- Generated **36 additional lagged features**
- Combined original and lagged variables to obtain a total of **42 predictive features**
- Preserved historical climate patterns for improved forecasting performance

---

### 3. Machine Learning Models

Four supervised machine learning algorithms were evaluated and compared.

| Model | Description |
|-------|-------------|
| Random Forest | Ensemble of 400 decision trees |
| SVM | Support Vector Machine with RBF kernel |
| KNN | K-Nearest Neighbors classifier |
| XGBoost | Gradient Boosting with hyperparameter optimization |

Model performance was compared using multiple classification metrics to identify the best-performing approach.

---

### 4. Handling Class Imbalance

Since drought events represent a minority class, three different imbalance handling strategies were investigated.

1. **XGBoost with Weighted Loss** (`scale_pos_weight`)
2. **XGBoost without SMOTE and Unweighted Loss**
3. **XGBoost with SMOTE** (Synthetic Minority Oversampling Technique)

The effectiveness of each strategy was evaluated using classification metrics and Precision–Recall analysis.

---

### 5. Hyperparameter Optimization

The XGBoost model was optimized using **RandomizedSearchCV** with cross-validation.

- **RandomizedSearchCV**
- **3-fold Cross Validation**
- **StratifiedKFold (5-fold)** for model evaluation

Optimized parameters include:

- `learning_rate`
- `n_estimators`
- `max_depth`
- `min_child_weight`
- `gamma`
- `subsample`
- `colsample_bytree`
- `reg_alpha`
- `reg_lambda`

---

### 6. Model Evaluation

The predictive performance of all models was evaluated using multiple classification metrics.

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- PR-AUC (Precision–Recall Area Under Curve)
- Confusion Matrix
- Precision–Recall Curve

A comparative analysis was performed to identify the most reliable model for drought prediction.

---

### 7. Explainable Artificial Intelligence (XAI)

To improve model transparency and interpretability, Explainable AI techniques were incorporated.

#### SHAP (SHapley Additive exPlanations)

SHAP was used to provide:

- Global feature importance
- State-wise feature importance
- Feature contribution analysis
- Beeswarm and summary plots
- Feature interaction insights

#### LIME (Local Interpretable Model-agnostic Explanations)

LIME was used for local interpretation by explaining individual drought and non-drought predictions through feature contribution analysis.

---

### 8. Uncertainty Quantification

Model prediction confidence was assessed using probability-based uncertainty analysis.

- Estimated prediction probabilities
- Categorized predictions into uncertainty levels
- Identified high-confidence and low-confidence predictions
- Improved the reliability of drought forecasting for decision support

---

### 9. Probabilistic Policy Insights

To facilitate practical decision-making, probabilistic analyses were conducted.

This analysis includes:

- Probability distributions of drought predictions
- State-wise probabilistic insights
- Feature-wise probability interpretation
- Decision-support information for policymakers and disaster management authorities

---

### 10. District-Level Drought Prediction

The trained XGBoost model was applied to generate district-wise drought predictions.

The framework provides:

- District-level drought classification
- State-wise prediction summaries
- Spatial drought visualization
- Prediction maps for early warning and climate resilience planning

---

### 11. Cross-Dataset Generalization

A separate notebook evaluates the robustness of the proposed framework using an independent dataset.

This analysis investigates:

- Model transferability
- Prediction consistency
- Generalization capability
- Performance on unseen datasets
- Robustness across different data distributions

Cross-dataset validation demonstrates the applicability of the proposed framework beyond the original training dataset.
## 📁 Project Structure

```text
Drought-Prediction/
│
├── Code/
│   ├── Drought_Prediction_using_ML_and_Ensemble_Models_alongwith_xAI_with_SHAP_&_LIME.ipynb
│   │      # Main notebook containing data preprocessing, feature engineering,
│   │      # model development, evaluation, SHAP & LIME explainability
│   │
│   ├── Drought_Prediction_Cross_Dataset_Generalization.ipynb
│   │      # External dataset evaluation for testing model robustness and
│   │      # cross-dataset generalization
│   │
│   └── xgbmodel_files.pkl
│          # Saved XGBoost model and preprocessing objects
│
├── Data/
│   ├── final.csv
│   │      # Historical meteorological dataset used for model development
│   │
│   └── README.md
│          # Instructions for accessing the dataset
│
├── Figures/
│   ├── Workflow Diagram.png
│   ├── Time Series Plots/
│   ├── Model Performance Comparison.png
│   ├── XGBoost Comparison (Weighted Loss vs SMOTE).png
│   ├── SHAP Summary Plot.png
│   ├── SHAP Beeswarm Plot.png
│   ├── LIME Explain Plot - Gujarat Patan (Drought).png
│   ├── LIME Explain Plot - Bihar Siwan (Non-Drought).png
│   ├── Prediction Maps.png
│   ├── Uncertainty Analysis.png
│   └── Probabilistic Insights.png
│
├── Results/
│   ├── performance_with_diff_model_configs_v1.1.csv
│   │      # Performance metrics for all evaluated machine learning models
│   │
│   ├── shap_top5_feats_per_state_v1.1.csv
│   │      # Top five SHAP features for each state
│   │
│   └── Probabilistic Insights for Top 2 Important Features and Different States v1.1.csv
│          # State-wise probabilistic feature analysis
│
├── Research Paper/
│   └── Manuscript and supplementary documents
│
├── requirements.txt
│      # Python package dependencies
│
├── LICENSE
│      # Repository license
│
└── README.md
       # Project documentation
```
## 💻 Code

### Main Drought Prediction Notebook

| Platform | Link |
|----------|------|
| 📄 GitHub Notebook | **PASTE_GITHUB_NOTEBOOK_LINK_HERE** |
| ▶ Google Colab |https://colab.research.google.com/drive/10fywWcECadQL6g3LJeuIap27zIEwor46?usp=sharing|

---

### Cross-Dataset Generalization Notebook

| Platform | Link |
|----------|------|
| 📄 GitHub Notebook | **PASTE_GITHUB_NOTEBOOK_LINK_HERE** |
| ▶ Google Colab | https://colab.research.google.com/drive/1UHo7rJlxWrxvcOtNckATvJwx6_ED2b_P?usp=sharing|

---
## 💻 Usage

### Running the Main Notebook

1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Navigate to the `Code/` directory and open:

   ```text
   New_Drought_Prediction_using_ML_and_Ensemble_Models_alongwith_xAI_with_SHAP_&_LIME.ipynb
   ```

3. Update the dataset path according to your local directory.

   ```python
   # Example

   df = pd.read_csv("../Data/final.csv")
   ```

4. Execute all notebook cells sequentially to perform:

   - Data preprocessing
   - Feature engineering
   - Model training
   - Hyperparameter optimization
   - Class imbalance handling
   - Model evaluation
   - SHAP analysis
   - LIME explanations
   - Uncertainty quantification
   - District-level drought prediction
   - Probabilistic policy insights

---

### Cross-Dataset Generalization

To evaluate the robustness of the trained model on an independent dataset, open:

```text
Code/Drought_Prediction_Cross_Dataset_Generalization.ipynb
```

Run all cells sequentially to perform:

- External dataset preprocessing
- Feature alignment
- Model inference
- Cross-dataset evaluation
- Generalization analysis

---

### Quick Start – Load the Pre-trained Model

```python
import pickle

# Load saved model and preprocessing objects
with open("../Code/xgbmodel_files.pkl", "rb") as f:
    xgbmodel_files = pickle.load(f)

best_xgb_model = xgbmodel_files[0]      # Trained XGBoost model
sc = xgbmodel_files[1]                  # StandardScaler
le_district = xgbmodel_files[2]         # LabelEncoder for DISTRICT
le_state = xgbmodel_files[3]            # LabelEncoder for STATE
```
## 📈 Experimental Results

The proposed framework was evaluated using multiple machine learning models and different class imbalance handling strategies.

### Model Comparison

The following models were compared:

- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- XGBoost

Performance was assessed using:

- Accuracy
- Precision
- Recall
- F1-score
- PR-AUC (Precision–Recall Area Under Curve)

---

### Class Imbalance Analysis

Three different approaches were investigated to address the imbalanced nature of drought events:

- XGBoost with Weighted Loss
- XGBoost without SMOTE (Unweighted Loss)
- XGBoost with SMOTE

A comparative performance analysis demonstrates the effectiveness of each strategy for drought prediction.

---

### Explainability Analysis

To improve model transparency and interpretability, the framework incorporates:

- SHAP global feature importance
- State-wise SHAP feature analysis
- SHAP summary and beeswarm plots
- LIME explanations for individual drought and non-drought predictions

---

### Uncertainty and Probabilistic Analysis

The framework further evaluates prediction reliability through:

- Prediction probability estimation
- Uncertainty quantification
- State-wise probabilistic feature insights
- Decision-support information for drought mitigation

---

### Cross-Dataset Evaluation

A separate notebook evaluates the trained model on an independent dataset to assess:

- Prediction consistency
- Model robustness
- Generalization capability
- Transferability across different datasets

---

### Generated Outputs

Running the notebooks produces:

- Performance comparison tables
- Confusion matrices
- Precision–Recall curves
- SHAP visualizations
- LIME explanation plots
- State-wise SHAP feature rankings
- Probabilistic insight reports
- District-level drought prediction maps
- Cross-dataset evaluation results
- ## 📊 Visualizations

The `Figures/` directory contains visualizations that illustrate the data characteristics, model performance, explainability, and prediction outcomes.

### Time Series Analysis
- Average Temperature Time Series
- Potential Evapotranspiration Time Series
- Precipitation Time Series
- Standardized Precipitation Index (SPI) Time Series
- Vapour Pressure Time Series
- Wet Day Frequency Time Series

### Model Performance
- Comparison of Random Forest, SVM, KNN, and XGBoost models
- Comparison of XGBoost with Weighted Loss, SMOTE, and Unweighted Loss
- Confusion Matrix
- Precision–Recall Curve
- Performance Metrics Comparison

### Explainable AI
- SHAP Summary Plot
- SHAP Beeswarm Plot
- State-wise SHAP Feature Importance
- LIME Explanation for Drought Prediction
- LIME Explanation for Non-Drought Prediction

### Prediction and Analysis
- District-Level Drought Prediction Maps
- Uncertainty Quantification
- Probabilistic Policy Insights
- Workflow Diagram

## 🔑 Key Findings

- **XGBoost consistently outperformed** Random Forest, SVM, and KNN for district-level drought prediction.
- **Weighted Loss** proved more effective than SMOTE in handling severe class imbalance while maintaining better predictive performance.
- **Temporal lag features (up to six months)** significantly improved the model's ability to capture drought evolution patterns.
- **Standardized Precipitation Index (SPI)** and precipitation-related variables were identified as the most influential predictors through SHAP analysis.
- **SHAP and LIME** enhanced model interpretability by providing both global feature importance and local prediction explanations.
- **State-wise feature importance** revealed regional differences in drought-driving factors, highlighting the need for localized drought assessment.
- **Uncertainty quantification** and **probabilistic policy insights** improved confidence in predictions and supported informed decision-making.
- **Cross-dataset generalization** demonstrated the robustness and transferability of the proposed framework on unseen datasets.

## 🔮 Future Scope

The proposed framework can be further extended through the following enhancements:

- Integration of **satellite imagery** and remote sensing products (e.g., MODIS, Sentinel, Landsat).
- Incorporation of **real-time weather data** for continuous drought monitoring.
- Development of **deep learning models** such as LSTM, GRU, and Transformer architectures for long-term temporal forecasting.
- Expansion to **additional states and districts** across India for nationwide drought prediction.
- Integration of **soil moisture, vegetation indices (NDVI, EVI)**, and other environmental variables.
- Development of an **interactive web dashboard** for visualization and real-time monitoring.
- Deployment as a **decision support system** for agricultural planning and disaster management.
- Integration with **government early warning systems** to assist policymakers in drought mitigation and climate resilience planning.

---
