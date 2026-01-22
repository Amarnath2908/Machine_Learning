from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import os


st.set_page_config("Smart Loan Approval System", layout="centered")

# ---------- Load CSS ----------
def load_css(filename):
    css_path = os.path.join(os.path.dirname(__file__), filename)
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")


# ---------- Title ----------
st.markdown("""
<div class="card">
<h1> Smart Loan Approval System </h1>
<p>This system uses Support Vector Machines (SVM) to predict loan approval.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Load Data ----------
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "traindata.csv")
    return pd.read_csv(file_path)

df = load_data()

# ---------- Data Cleaning ----------
df = df.drop(columns=["Loan_ID"])

df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)
df["Dependents"].fillna(df["Dependents"].mode()[0], inplace=True)
df["Self_Employed"].fillna(df["Self_Employed"].mode()[0], inplace=True)
df["Married"].fillna(df["Married"].mode()[0], inplace=True)

df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median(), inplace=True)
df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)
df["Credit_History"].fillna(df["Credit_History"].mode()[0], inplace=True)

# ---------- Encoding ----------
label_encoders = {}
for c in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c])
    label_encoders[c] = le

# ---------- Features & Target ----------
features = [
    "ApplicantIncome",
    "LoanAmount",
    "Credit_History",
    "Self_Employed",
    "Married",
    "Education",
    "CoapplicantIncome",
    "Loan_Amount_Term",
    "Property_Area"
]

X = df[features]
y = df["Loan_Status"]

# ---------- Train-Test Split ----------
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- Scaling ----------
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# ---------- Sidebar Inputs ----------
st.sidebar.header("📋 Enter Applicant Details")

applicant_income = st.sidebar.number_input("Applicant Income", min_value=0, step=1000)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, step=1000)

credit_history = st.sidebar.radio("Credit History", ("Yes", "No"))
employment_status = st.sidebar.selectbox("Employment Status", ("Employed", "Self Employed"))
property_area = st.sidebar.selectbox("Property Area", ("Urban", "Semiurban", "Rural"))

credit_history_val = 1 if credit_history == "Yes" else 0
employment_status_val = 0 if employment_status == "Employed" else 1
property_area_val = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

# ---------- Model Selection ----------
st.sidebar.header("⚙️ Choose SVM Kernel")

kernel_choice = st.sidebar.radio(
    "Select Kernel",
    ("Linear SVM", "Polynomial SVM", "RBF SVM")
)

kernel_map = {
    "Linear SVM": "linear",
    "Polynomial SVM": "poly",
    "RBF SVM": "rbf"
}

kernel = kernel_map[kernel_choice]

model = SVC(kernel=kernel, probability=True)
model.fit(x_train, y_train)

# ---------- Prediction ----------
st.markdown("## 🔮 Loan Eligibility Prediction")

if st.button("Check Loan Eligibility"):

    # IMPORTANT: Feature order must match training data
    user_data = np.array([[
        applicant_income,      # ApplicantIncome
        loan_amount,           # LoanAmount
        credit_history_val,    # Credit_History
        employment_status_val, # Self_Employed
        1,                     # Married (default Yes)
        1,                     # Education (default Graduate)
        0,                     # CoapplicantIncome (default 0)
        360,                   # Loan_Amount_Term (default)
        property_area_val      # Property_Area
    ]])

    user_data = scaler.transform(user_data)

    prediction = model.predict(user_data)[0]
    prob = model.predict_proba(user_data)[0].max() * 100

    # ---------- Output ----------
    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.markdown("### 📊 Prediction Details")
    st.write(f"**Kernel Used:** {kernel_choice}")
    st.write(f"**Confidence Score:** {prob:.2f}%")

    # ---------- Business Explanation ----------
    st.markdown("### 🧠 Business Explanation")

    if prediction == 1:
        st.info("Based on credit history and income pattern, the applicant is likely to repay the loan.")
    else:
        st.warning("Based on credit history and income pattern, the applicant is unlikely to repay the loan.")

# ---------- Model Performance ----------
st.markdown("---")
st.subheader("📈 Model Performance on Test Data")

y_pred = model.predict(x_test)
st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.text(classification_report(y_test, y_pred))

fig, ax = plt.subplots()
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot(ax=ax)
st.pyplot(fig)
