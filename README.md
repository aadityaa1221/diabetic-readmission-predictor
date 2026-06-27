# 🏥 Hospital Readmission Risk Predictor

An end-to-end machine learning project designed to predict the 30-day hospital readmission risk for diabetic patients. This repository contains the complete pipeline: from data preprocessing and exploratory data analysis (EDA) to model training and the deployment of an interactive web application.

## 📋 Project Overview

Hospital readmissions are a critical metric for healthcare quality and cost. This tool evaluates patient demographics, medical history, and clinical test results to generate a readmission risk score. If a patient's probability exceeds 50%, they are flagged as **High Risk**, indicating a potential need for proactive follow-up care.

## 🏗️ Technical Architecture & Transparency

To provide a clear understanding of the project's technical composition, the architecture is divided into custom implementations and utilized external libraries.

**Original Implementations:**
* **Data Pipeline:** Custom preprocessing logic to handle categorical mapping, feature selection, and class imbalance scaling.
* **Predictive Modeling:** Training and hyperparameter tuning of an original `XGBClassifier` optimized for log-loss and AUC.
* **Thresholding & Logic:** Custom probability thresholding and visual mapping logic for the user interface.

**Utilized Libraries:**
* **Model Training:** `XGBoost` for gradient boosting, `Scikit-Learn` for data splitting and label encoding.
* **Explainability:** `SHAP` (SHapley Additive exPlanations) for model interpretability and feature importance analysis.
* **Application Framework:** `Streamlit` for rapid UI development and interactive state management.
* **Data Visualization:** `Matplotlib` and `Seaborn` for static ROC curves, confusion matrices, and dynamic risk gauges.

## 📂 Repository Structure

* `app.py`: The main Streamlit application script containing the UI and prediction logic.
* `diabetic_readmission_predictor.ipynb`: The Jupyter Notebook documenting EDA, data cleaning, XGBoost training, and SHAP evaluation.
* `readmission_model.pkl`: The serialized XGBoost classifier.
* `model_features.pkl`: The exact feature order required for model inference.
* `encoders.pkl`: The label encoder mappings used during training to ensure consistent input formatting.
* `requirements.txt`: The environment dependencies.

## 🚀 Running the Application Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR-USERNAME/diabetic-readmission-predictor.git](https://github.com/YOUR-USERNAME/diabetic-readmission-predictor.git)
   cd diabetic-readmission-predictor
