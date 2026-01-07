from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import os

st.set_page_config("Logistic Regression", layout="centered")

# ---------- Load CSS ----------
def load_css(filename):
    css_path = os.path.join(os.path.dirname(__file__), filename)
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# ---------- Title ----------
st.markdown("""
<div class="card">
<h1> Logistic Regression </h1>
</div>
""", unsafe_allow_html=True)

# ---------- Load Data ----------
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "Telco_customer.csv")
    return pd.read_csv(file_path)
# Load data FIRST
df = load_data()



st.subheader("Dataset Preview")
# ---------- Encoding (SAFE METHOD) ----------
le = LabelEncoder()
cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

st.dataframe(df.head())

X = df.drop("Churn", axis=1)
y = df["Churn"]

# ---------- Train Test Split ----------
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- Scaling ----------
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# ---------- Model ----------
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# ---------- Prediction ----------
y_pred = model.predict(x_test)

st.subheader("Model Prediction")
st.write("Prediction:",y_pred)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")
st.write(f"Accuracy: **{accuracy:.2f}**")

# ---------- Classification Report ----------
st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

# ---------- Confusion Matrix ----------
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test,y_pred)
st.write("True Positive:",cm[0,0])
st.write("False Negative:",cm[1,1])
fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    cmap="Blues",
    values_format="d",
    display_labels=["No", "Yes"],
    ax=ax
)
st.pyplot(fig)
