# 🏦 Credit Risk Prediction Using Machine Learning

## 📌 Project Overview

This project focuses on predicting whether a loan applicant will default or not using machine learning models.

The goal is to help financial institutions assess credit risk and make more informed lending decisions.

We build an end-to-end machine learning pipeline from data exploration to model deployment.

---

## 🎯 Objective

- Predict loan default risk (binary classification problem)
- Compare multiple machine learning models
- Handle class imbalance in the dataset
- Identify the most important risk factors
- Build an interpretable and deployable ML model

---

## 📊 Dataset Description

The dataset contains information about loan applicants, including:

- Demographic information (age, income, employment length)
- Loan details (amount, interest rate, purpose)
- Credit history
- Target variable:
  - `loan_status` → 0 = No Default, 1 = Default

---

## 🧠 Machine Learning Pipeline

The project follows a structured workflow:

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training
- Model Evaluation
- Model Explainability
- Final Model Selection

---

## 🤖 Models Used

### 📌 Logistic Regression (Baseline)
A simple and interpretable model used as a baseline.  
Handles class imbalance using `class_weight='balanced'`.

---

### 🌳 Random Forest
An ensemble model based on decision trees.  
Captures non-linear relationships and improves stability.

---

### 🚀 XGBoost (Final Model)
A powerful gradient boosting model optimized for tabular data.  
Selected as the final model due to best performance.

---

## 📈 Evaluation Metrics

The models are evaluated using:

- Confusion Matrix
- Precision, Recall, F1-score
- ROC-AUC Score
- Precision-Recall Curve

---

## 🏆 Final Model

The final selected model is **XGBoost**, because it:

- Achieved the highest ROC-AUC score
- Balanced precision and recall effectively
- Performed best on imbalanced data
- Generalized well on unseen data

---

## 🔍 Model Explainability

We use interpretability techniques to understand predictions:

- Feature Importance (XGBoost)
- SHAP values (global + local explanations)
- ROC Curve
- Precision-Recall Curve

These help explain why a loan is predicted as risky or safe.

---

## ⚙️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn
- SHAP
- Joblib

---

## 🚀 How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/your-username/credit-risk-project.git
cd credit-risk-project