#  ML-Based Credit Score & Loan Approval Predictor

A machine learning web application built using **Streamlit** that predicts **loan approval / creditworthiness** based on applicant financial and personal details.  
The application trains a **Gaussian Naive Bayes** model in real time and provides instant predictions through an interactive user interface.

---

##  Project Overview

Banks and financial institutions evaluate multiple factors before approving a loan.  
This project demonstrates how **machine learning** can be used to analyze applicant data and predict loan approval, which serves as an indicator of **creditworthiness**.

The system allows users to input personal and financial details through a Streamlit-based UI and generates predictions using a trained ML model.

---

##  Features

- Interactive **Streamlit UI**
- Real-time model training (no pickle or saved model)
- Automatic data preprocessing
- Missing value handling using imputation
- Encoding of categorical features
- Feature scaling and engineering
- Gaussian Naive Bayes classification
- Clean and beginner-friendly design
- Suitable for academic projects and viva

---

##  Machine Learning Methodology

- **Algorithm:** Gaussian Naive Bayes  
- **Target Variable:** Loan_Approved  

### Data Preprocessing
- Missing numerical values handled using `SimpleImputer (mean)`
- Missing categorical values handled using `SimpleImputer (most frequent)`
- Label Encoding for ordinal features
- One-Hot Encoding for nominal categorical features
- Feature engineering (squared and logarithmic transformations)
- Feature scaling using `StandardScaler`

---

##  Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn

---
