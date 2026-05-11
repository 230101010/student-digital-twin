# Student Digital Twin for Exam Success Prediction 

**CSS 324: Introduction to Machine Learning — Final Project**

## Overview
A simplified digital twin of a student using behavioral and academic features to predict exam success probability.
## Project Structure
## Features
- **Data:** 5 integrated raw datasets from different sources (9,113 records, 73 features)
- **EDA:** Statistical summaries, correlation analysis, visualizations
- **Models:** Logistic Regression, Random Forest, XGBoost with GridSearchCV hyperparameter tuning
- **Best Model:** Tuned Random Forest — Accuracy: 86%, F1-score: 0.87
- **Demo:** Interactive Streamlit application with real-time risk prediction
├── data/raw/ # 5 source datasets (CSV)
├── data/processed/ # master_dataset.csv
├── model/ # best_model.pkl, scaler.pkl, feature_means.pkl
├── app/ # Streamlit demo (app.py)
├── notebooks/ # Jupyter notebook
├── README.md
└── requirements.txt



## How to Run the Demo
pip install -r requirements.txt
cd app && streamlit run app.py


## Key Findings
- **Top predictors:** Attendance (34%), Hours Studied (16%), Previous Scores (7%)
- **Model performance:** 86% accuracy, 87% F1-score on test set (1,727 samples)
- **Risk categories:** Low / Medium / High based on predicted probability

## Author
Abu Akerke — CSS 324 Final Project, Spring 2026
