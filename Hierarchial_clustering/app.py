import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.cluster.hierarchy as sch

from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="🟣 News Topic Discovery Dashboard", layout="centered")

# ===============================
# TITLE & DESCRIPTION
# ===============================

st.title("🟣 News Topic Discovery Dashboard")

st.write("""
This system uses **Hierarchical Clustering** to automatically group similar news articles 
based on textual similarity.  
Discover hidden themes without defining categories upfront.
""")

# ===============================
# SIDEBAR – DATASET UPLOAD
# ===============================

st.sidebar.header("📂 Dataset Handling")

uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    except:
        try:
            df = pd.read_csv(uploaded_file, encoding='latin-1')
        except:
            df = pd.read_csv(uploaded_file, encoding='cp1252')

    st.sidebar.success("Dataset Uploaded Successfully")
else:
    st.sidebar.info("Please upload a CSV file to continue")
    st.stop()

st.write("### Dataset Preview")
st.dataframe(df.head())

# ===============================
# AUTO DETECT TEXT COLUMN
# ===============================

st.sidebar.header("📝 Text Column Selection")

text_columns = df.select_dtypes(include=['object']).columns.tolist()

if len(text_columns) == 0:
    st.error("No text column detected in dataset")
    st.stop()

text_col = st.sidebar.selectbox("Select Text Column", text_columns)

# ===============================
# TEXT VECTORIZATION CONTROLS
# ===============================

st.sidebar.header("📝 Text Vectorization Controls")

max_features = st.sidebar.slider(
    "Maximum TF-IDF Features",
    min_value=100,
    max_value=2000,
    value=1000
)

use_stopwords = st.sidebar.checkbox("Remove English Stopwords", value=True)

ngram_option = st.sidebar.selectbox(
    "N-gram Range",
    ["Unigrams", "Bigrams", "Unigrams + Bigrams"]
)

if ngram_option == "Unigrams":
    ngram_range = (1, 1)
elif ngram_option == "Bigrams":
    ngram_range = (2, 2)
else:
    ngram_range = (1, 2)

# ===============================
# HIERARCHICAL CLUSTERING CONTROLS
# ===============================

st.sidebar.header("🌳 Hierarchical Clustering Controls")

linkage_method = st.sidebar.selectbox(
    "Linkage Method",
    ["ward", "complete", "average", "single"]
)

distance_metric = st.sidebar.selectbox(
    "Distance Metric",
    ["euclidean"]
)

subset_size = st.sidebar.slider(
    "Number of Articles for Dendrogram",
    min_value=20,
    max_value=200,
    value=50
)

# ===============================
# TF-IDF PROCESSING
# ===============================

stop_words = "english" if use_stopwords else None

vectorizer = TfidfVectorizer(
    max_features=max_features,
    stop_words=stop_words,
    ngram_range=ngram_range
)

X = vectorizer.fit_transform(df[text_col]).toarray()

st.write("### TF-IDF Matrix Shape:", X.shape)

# ===============================
# DENDROGRAM GENERATION
# ===============================

st.write("## Dendrogram Section")

if st.button("🟦 Generate Dendrogram"):

    st.write("### Hierarchical Clustering Dendrogram")

    if len(X) < subset_size:
        subset_size = len(X)

    X_subset = X[:subset_size]

    fig = plt.figure(figsize=(10, 6))

    sch.dendrogram(
        sch.linkage(X_subset, method=linkage_method)
    )

    plt.title("Dendrogram")
    plt.xlabel("Article Index")
    plt.ylabel("Distance")

    st.pyplot(fig)

    st.info("Inspect the dendrogram to decide the optimal number of clusters.")

# ===============================
# APPLY CLUSTERING
# ===============================

st.write("## Clustering Control")

num_clusters = st.number_input(
    "Select Number of Clusters After Inspecting Dendrogram",
    min_value=2,
    max_value=10,
    value=3
)

if st.button("🟩 Apply Clustering"):

    model = AgglomerativeClustering(
        n_clusters=num_clusters,
        metric=distance_metric,
        linkage=linkage_method
    )

    labels = model.fit_predict(X)

    df["Cluster"] = labels

    st.success("Clustering Applied Successfully!")

    # ===============================
    # PCA VISUALIZATION
    # ===============================

    st.write("## Clustering Visualization (PCA Projection)")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig2, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10")

    ax.set_title("2D PCA Projection of Clusters")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")

    st.pyplot(fig2)

    # ===============================
    # CLUSTER SUMMARY
    # ===============================

    st.write("## Cluster Summary")

    feature_names = vectorizer.get_feature_names_out()

    summary = []

    for i in range(num_clusters):

    # Create boolean mask for rows belonging to cluster i
        mask = (df["Cluster"] == i).values

    # Select only those rows from X using mask
        X_cluster = X[mask]

    # Sum TF-IDF scores and find top 10 terms
        top_terms = np.argsort(X_cluster.sum(axis=0))[-10:]

        keywords = [feature_names[t] for t in top_terms]

        summary.append({
        "Cluster ID": i,
        "Number of Articles": int(mask.sum()),
        "Top Keywords": ", ".join(keywords)
        })


    summary_df = pd.DataFrame(summary)

    st.table(summary_df)

    # ===============================
    # SILHOUETTE SCORE
    # ===============================

    score = silhouette_score(X, labels)

    st.write("## Validation Section")

    st.metric("Silhouette Score", round(score, 3))

    st.write("""
    **Interpretation**
    - Close to 1 → Well-separated clusters  
    - Close to 0 → Overlapping clusters  
    - Negative → Poor clustering
    """)

    # ===============================
    # BUSINESS INTERPRETATION
    # ===============================

    st.write("## Business Interpretation")

    for i in range(num_clusters):
        st.write(f"🟣 **Cluster {i}:** Articles focused on themes related to the main keywords shown above.")

    st.info("""
    Articles grouped in the same cluster share similar vocabulary and themes.  
    These clusters can be used for automatic tagging, recommendations, and content organization.
    """)
