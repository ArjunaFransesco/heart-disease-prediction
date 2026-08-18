# 🫀 Cardiovascular Heart Disease Prediction & Clinical Decision Support

[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000.svg?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning pipeline and clinical diagnostic web application for predicting cardiovascular disease risk from patient clinical biomarkers (UCI Cleveland Heart Disease Dataset).

---

## 🔬 Key Highlights & Architecture

- **Dataset**: UCI Cleveland Heart Disease Dataset (303 patient records, 14 clinical features including `age`, `sex`, `cp`, `trestbps`, `chol`, `thalach`, `oldpeak`, `ca`, `thal`).
- **Data Preprocessing**: `ColumnTransformer` with `StandardScaler` for numerical metrics and categorical feature preservation.
- **Model Comparisons**: Benchmark evaluation comparing **Logistic Regression**, **Random Forest**, and **Gradient Boosting**.
- **Model Evaluation**: ROC-AUC metric tuning reaching **~0.89+ AUC**, with stratified 5-fold cross-validation.
- **Decision Support Web UI**: Interactive medical-grade clinical risk scoring application built with Flask & Tailwind CSS.

---

## 📊 Feature Importance & Biomarkers

1. **Chest Pain Type (`cp`)**: Strong predictor of ischemic cardiac events.
2. **Maximum Heart Rate (`thalach`)**: Inverse correlation with exercise-induced cardiac strain.
3. **ST Depression (`oldpeak`)**: Significant indicator of myocardial ischemia under stress.
4. **Major Vessels (`ca`)**: Number of vessels colored by fluoroscopy indicating arterial patency.
5. **Thalassemia (`thal`)**: Blood disorder and stress test defect indicators.

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/ArjunaFransesco/heart-disease-prediction.git
cd heart-disease-prediction

# Run web app
python app/main.py
```
Open `http://localhost:5000` in your web browser.
