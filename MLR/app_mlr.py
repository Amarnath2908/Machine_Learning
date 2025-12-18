import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score



#Page config

st.set_page_config("Multiple Linear Regression",layout = "centered")

#Load css
def load_css(filename):
    css_path = os.path.join(os.path.dirname(__file__), filename)
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style1.css")

#Title

st.markdown("""
        <div class="card">
        <h1> Multiple Linear Regression</h1>
            <p> Predict <b>Tip Amount<b> from <b> Total Bill </b> using Linear Regression..</p>    
            </div>

            """,unsafe_allow_html=True)

#Load Data

@st.cache_data
def load_data():
    return sns.load_dataset("tips")

df = load_data()

#Dataset Preview 
st.markdown('<div class = "card">',unsafe_allow_html=True)
st.subheader("Dataset Preview",anchor=False)
st.dataframe(df[["total_bill","size","tip"]].head())
st.markdown('<div>',unsafe_allow_html=True)

# Prepare Data

x,y = df[["total_bill","size"]],df["tip"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


#Train Model

model = LinearRegression()
model.fit(x_train_scaled,y_train)
y_pred = model.predict(x_test_scaled)


#Metrics

mae = mean_absolute_error(y_test,y_pred)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test,y_pred)
adj_r2 = 1-(1-r2)*(len(y_test)-1)/(len(y_test)-2)

#Visualization

st.markdown('<div class = "card">',unsafe_allow_html=True)
st.subheader("Total Bill vs Tip(with Multiple Linear Regression)")
fig,ax = plt.subplots()
ax.scatter(df["total_bill"],df["tip"],alpha = 0.6)
x_scaled_full = scaler.transform(x)
ax.plot(df["total_bill"],model.predict(x_scaled_full),color="red")
ax.set_xlabel("Total bill($)")
ax.set_ylabel("Tip {$}")
st.pyplot(fig)
st.markdown('</div>',unsafe_allow_html=True)

#performance metrics

st.markdown('<div class= "card">',unsafe_allow_html=True)
st.subheader("Model performance",anchor=False)
c1,c2 = st.columns(2)
c1.metric("MAE",f"{mae:.2f}")
c2.metric("RMSE",f"{rmse:.2f}")
c3,c4 = st.columns(2)
c3.metric("R2",f"{r2:.3f}")
c4.metric("Adj R2",f"{adj_r2: .3f}")
st.markdown('</div>',unsafe_allow_html=True)

#m & c (slope and intercept)

st.markdown(f"""
            <div class = "card">
            <h3>Model Interception </h3>
            <p>
            <b>Coefficient (Total bill):</b> {model.coef_[0]:.3f}<br>
            <b>Coefficient (Group size):</b> {model.coef_[1]:.3f}<br>
            <b>Intercept:</b> {model.intercept_:.3f}
            </p>
            <p>
            Tip depends on <b>bill amount </b> and <b> number of people</b>.
            </p>
            </div>
            """,unsafe_allow_html=True)



#Prediction

st.markdown('<div class = "card">',unsafe_allow_html=True)
st.subheader("Predict Tip Amount:")
bill = st.slider(
    "Total Bill ($)",
    min_value=float(df["total_bill"].min()),
    max_value=float(df["total_bill"].max()),
    value=30.0
)
size = st.slider(
    "Group size",
    min_value=int(df["size"].min()),
    max_value=int(df["size"].max()),
    value=2
)

tip = model.predict(scaler.transform([[bill,size]]))[0]
st.markdown(f'<div class = "prediction-box">Predicted Tip:${tip:.2f}</div> ',unsafe_allow_html=True)
st.markdown('</div>',unsafe_allow_html=True)
