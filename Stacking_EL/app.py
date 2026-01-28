from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import streamlit as st
import pandas as pd
import numpy as np
import os

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config("🎯 Smart Loan Approval System - Stacking Model", layout="centered")

# ===============================
# TITLE & DESCRIPTION
# ===============================
st.markdown("""
## 🎯 Smart Loan Approval System – Stacking Model
This system uses a **Stacking Ensemble Machine Learning model** to predict whether a loan will be approved by combining multiple ML models for better decision making.
""")

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "train_u6lujuX_CVtuZ9i.csv")
    return pd.read_csv(file_path)

df = load_data()

# ===============================
# DATA CLEANING
# ===============================
df = df.drop(columns=["Loan_ID"])

df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)
df["Dependents"].fillna(df["Dependents"].mode()[0], inplace=True)
df["Self_Employed"].fillna(df["Self_Employed"].mode()[0], inplace=True)
df["Married"].fillna(df["Married"].mode()[0], inplace=True)

df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median(), inplace=True)
df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)
df["Credit_History"].fillna(df["Credit_History"].mode()[0], inplace=True)

# ===============================
# ENCODING
# ===============================
le = LabelEncoder()
cat_cols = ["Gender","Married","Dependents","Education","Self_Employed","Property_Area","Loan_Status"]

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# ===============================
# TRAIN TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===============================
# SCALING
# ===============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# STACKING MODEL
# ===============================
base_models = [
    ('lr', LogisticRegression()),
    ('dt', DecisionTreeClassifier(max_depth=3)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
]

meta_model = LogisticRegression()

stack_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5
)

stack_model.fit(X_train, y_train)

# ===============================
# SIDEBAR INPUTS
# ===============================
st.sidebar.header("📥 Enter Applicant Details")

applicant_income = st.sidebar.number_input("Applicant Income", min_value=0, step=1000)
co_applicant_income = st.sidebar.number_input("Co-Applicant Income", min_value=0, step=1000)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, step=1000)
loan_amount_term = st.sidebar.number_input("Loan Amount Term", min_value=0, step=12)

credit_history = st.sidebar.radio("Credit History", ("Yes", "No"))
employment_status = st.sidebar.selectbox("Employment Status", ("Salaried", "Self-Employed"))
property_area = st.sidebar.selectbox("Property Area", ("Urban", "Semi-Urban", "Rural"))

credit_history_val = 1 if credit_history == "Yes" else 0
employment_status_val = 0 if employment_status == "Salaried" else 1
property_area_val = {"Urban": 2, "Semi-Urban": 1, "Rural": 0}[property_area]

# ===============================
# MODEL ARCHITECTURE DISPLAY
# ===============================
st.markdown("""
### 🧩 Stacking Model Architecture

**Base Models Used:**
- Logistic Regression
- Decision Tree
- Random Forest

**Meta Model Used:**
- Logistic Regression

📌 In stacking, predictions from base models are used as inputs to the meta-model.
""")

# ===============================
# PREDICTION BUTTON
# ===============================
if st.button("🔘 Check Loan Eligibility (Stacking Model)"):

    input_data = np.array([[0, 1, 0, 1, employment_status_val,
                            applicant_income, co_applicant_income,
                            loan_amount, loan_amount_term,
                            credit_history_val, property_area_val]])

    input_scaled = scaler.transform(input_data)

    # Base model predictions
    lr_pred = base_models[0][1].fit(X_train, y_train).predict(input_scaled)[0]
    dt_pred = base_models[1][1].fit(X_train, y_train).predict(input_scaled)[0]
    rf_pred = base_models[2][1].fit(X_train, y_train).predict(input_scaled)[0]

    final_pred = stack_model.predict(input_scaled)[0]
    prob = stack_model.predict_proba(input_scaled)[0][1] * 100

    # ===============================
    # OUTPUT SECTION
    # ===============================
    st.markdown("## 📊 Prediction Results")

    def show_result(pred):
        return "✅ Approved" if pred == 1 else "❌ Rejected"

    st.write("### 📊 Base Model Predictions")
    st.write(f"• Logistic Regression → {show_result(lr_pred)}")
    st.write(f"• Decision Tree → {show_result(dt_pred)}")
    st.write(f"• Random Forest → {show_result(rf_pred)}")

    st.markdown("### 🧠 Final Stacking Decision")

    if final_pred == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write(f"📈 Confidence Score: **{prob:.2f}%**")

    # ===============================
    # BUSINESS EXPLANATION
    # ===============================
    st.markdown("### 💼 Business Explanation")

    if final_pred == 1:
        st.write("""
        Based on the applicant's income, credit history, and combined predictions from multiple models,  
        the applicant is likely to repay the loan.  
        Therefore, the stacking model predicts **loan approval**.
        """)
    else:
        st.write("""
        Based on the applicant's income, credit history, and combined predictions from multiple models,  
        the applicant is unlikely to repay the loan.  
        Therefore, the stacking model predicts **loan rejection**.
        """)

# ===============================
# MODEL ACCURACY (OPTIONAL)
# ===============================
st.markdown("### 📈 Model Accuracy")
y_pred = stack_model.predict(X_test)
st.write(f"Stacking Model Accuracy: **{accuracy_score(y_test, y_pred) * 100:.2f}%**")
