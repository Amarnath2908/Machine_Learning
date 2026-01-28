import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config("🟢 Customer Segmentation Dashboard", layout="centered")

# ===============================
# LOAD CSS
# ===============================
def load_css(filename):
    css_path = os.path.join(os.path.dirname(__file__), filename)
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# ===============================
# TITLE & DESCRIPTION
# ===============================
st.markdown("""
<div class="main-title">🟢 Customer Segmentation Dashboard</div>
<div class="subtitle">
This system uses <b>K-Means Clustering</b> to group customers based on their purchasing behavior and similarities.
</div>
""", unsafe_allow_html=True)

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "Wholesale customers data.csv")
    return pd.read_csv(file_path)

df = load_data()

st.markdown("### 📂 Dataset Preview")
st.dataframe(df.head())

# ===============================
# NUMERICAL FEATURES
# ===============================
num_features = df.select_dtypes(include=np.number).columns.tolist()

# ===============================
# SIDEBAR CONTROLS
# ===============================
st.sidebar.header("🎛️ Clustering Controls")

feature1 = st.sidebar.selectbox("Select Feature 1", num_features)
feature2 = st.sidebar.selectbox("Select Feature 2", [f for f in num_features if f != feature1])

k_value = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=10, value=5)
random_state = st.sidebar.number_input("Random State (Optional)", min_value=0, value=42)

run_btn = st.sidebar.button("🟦 Run Clustering")

# ===============================
# RUN CLUSTERING
# ===============================
if run_btn:

    # Select features
    X = df[[feature1, feature2]].values

    # Scaling (important for KMeans)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans model
    kmeans = KMeans(n_clusters=k_value, init="k-means++", random_state=random_state, n_init=10)
    y_kmeans = kmeans.fit_predict(X_scaled)

    # ===============================
    # VISUALIZATION SECTION
    # ===============================
    st.markdown("## 📊 Cluster Visualization")

    plt.figure(figsize=(8, 6))

    colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'orange', 'purple', 'brown', 'pink']

    for i in range(k_value):
        plt.scatter(X_scaled[y_kmeans == i, 0], X_scaled[y_kmeans == i, 1],
                    s=80, color=colors[i], label=f'Cluster {i}')

    # Cluster centers
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                s=300, color='black', label='Centroids')

    plt.title(f'Customer Clusters ({feature1} vs {feature2})')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.legend()
    st.pyplot(plt)

    # ===============================
    # CLUSTER SUMMARY SECTION
    # ===============================
    st.markdown("## 📋 Cluster Summary")

    df_clustered = df.copy()
    df_clustered["Cluster"] = y_kmeans

    summary = df_clustered.groupby("Cluster")[[feature1, feature2]].agg(["mean", "count"])
    st.dataframe(summary)

    # ===============================
    # BUSINESS INTERPRETATION
    # ===============================
    st.markdown("## 💼 Business Interpretation")

    cluster_means = df_clustered.groupby("Cluster")[[feature1, feature2]].mean()

    interpretations = []
    for cluster_id in range(k_value):
        mean1 = cluster_means.loc[cluster_id, feature1]
        mean2 = cluster_means.loc[cluster_id, feature2]

        if mean1 > cluster_means[feature1].mean() and mean2 > cluster_means[feature2].mean():
            text = f"🟢 Cluster {cluster_id}: High-spending customers across selected categories."
        elif mean1 < cluster_means[feature1].mean() and mean2 < cluster_means[feature2].mean():
            text = f"🟡 Cluster {cluster_id}: Budget-conscious customers with low spending."
        else:
            text = f"🔵 Cluster {cluster_id}: Moderate spenders with selective purchasing behavior."

        interpretations.append(text)

    for t in interpretations:
        st.markdown(f"- {t}")

    # ===============================
    # USER GUIDANCE BOX
    # ===============================
    st.markdown("""
    <div class="business-box">
    💡 Customers in the same cluster exhibit similar purchasing behaviour and can be targeted with similar business strategies.
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("👈 Select features and click **Run Clustering** to see results.")
