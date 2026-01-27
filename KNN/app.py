import os
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Customer Risk Prediction System (KNN)",
    layout="centered",
)

# ================= LOAD CSS FILE =================
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ================= TITLE =================
st.markdown("""
<div class="app-title">
  <h1>Customer Risk Prediction System (KNN)</h1>
  <p>This system predicts customer risk by comparing them with similar customers.</p>
</div>
""", unsafe_allow_html=True)

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "credit_risk_dataset.csv")
    return pd.read_csv(file_path)

df = load_data()

# ================= DATA PREPROCESSING =================
df["person_emp_length"].fillna(df["person_emp_length"].mean(), inplace=True)
df["loan_int_rate"].fillna(df["loan_int_rate"].mean(), inplace=True)

# Label Encoding
le = LabelEncoder()
cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Feature Selection
X = df[["person_age", "person_income", "loan_amnt", "cb_person_cred_hist_length"]]
y = df["loan_status"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================= SIDEBAR INPUTS =================
st.sidebar.header("📋 Enter Applicant Details")

age = st.sidebar.slider("Age", 18, 70, 30)
income = st.sidebar.number_input("Annual Income", min_value=0, step=1000)
loan_amt = st.sidebar.number_input("Loan Amount", min_value=0, step=1000)
cred_hist = st.sidebar.slider("Credit History Length (years)", 0, 30, 5)

k_value = st.sidebar.slider("K Value (Neighbors)", 1, 15, 5, step=2)

# ================= PREDICTION =================
if st.button("Predict Customer Risk"):

    input_data = np.array([[age, income, loan_amt, cred_hist]])
    input_scaled = scaler.transform(input_data)

    knn = KNeighborsClassifier(n_neighbors=k_value)
    knn.fit(X_train, y_train)

    prediction = knn.predict(input_scaled)[0]

    # ===== RESULT =====
    st.markdown("## 🔍 Prediction Result")

    if prediction == 1:
        st.markdown("<div class='result-box high-risk'>🔴 High Risk Customer</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result-box low-risk'>🟢 Low Risk Customer</div>", unsafe_allow_html=True)

    # ===== KNN EXPLANATION =====
    distances, indices = knn.kneighbors(input_scaled)

    neighbor_classes = y_train.iloc[indices[0]].values
    unique, counts = np.unique(neighbor_classes, return_counts=True)
    majority_class = unique[np.argmax(counts)]

    st.markdown("## 🧠 KNN Explanation")
    st.markdown(f"""
    <div class="info-section">
    <b>Number of neighbors (K):</b> {k_value}<br>
    <b>Majority class among neighbors:</b> {majority_class}
    </div>
    """, unsafe_allow_html=True)

    # ===== NEAREST NEIGHBORS TABLE =====
    neighbors_df = df.iloc[y_train.iloc[indices[0]].index][
        ["person_age", "person_income", "loan_amnt", "cb_person_cred_hist_length", "loan_status"]
    ]
    st.markdown("### 📊 Nearest Customers")
    st.dataframe(neighbors_df)

    # ===== BUSINESS INSIGHT =====
    st.markdown("## 💡 Business Insight")
    st.markdown("""
    <div class="info-section">
    This decision is based on similarity with nearby customers in feature space.
    Customers with similar age, income, loan amount, and credit history influenced the prediction.
    </div>
    """, unsafe_allow_html=True)
    