import streamlit as st
import sys
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Fix path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import clean_text
from config import MODEL_PATH, VECTORIZER_PATH, DATA_PATH
from src.predict import SentimentPredictor
from src.bert_model import BERTSentiment

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    return SentimentPredictor(), BERTSentiment()

ml_model, bert_model = load_models()

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Sentiment Dashboard", layout="wide")

st.title("🚀 Advanced Sentiment Analysis Dashboard")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Controls")
model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Machine Learning", "BERT (Advanced)"]
)

# =========================
# INPUT
# =========================
st.subheader("💬 Analyze Review")

user_input = st.text_area("Enter your review:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Enter a review")
    else:
        if model_choice == "Machine Learning":
            result = ml_model.predict(user_input)
        else:
            result = bert_model.predict(user_input)

        st.success(f"Sentiment: {result['label']}")
        st.info(f"Confidence: {result['confidence']}%")

# =========================
# MODEL PERFORMANCE
# =========================
st.subheader("📊 Model Performance")

df["clean"] = df["review"].astype(str).apply(clean_text)

vectorizer = joblib.load(VECTORIZER_PATH)
model = joblib.load(MODEL_PATH)

X = vectorizer.transform(df["clean"])
y = df["sentiment"]
preds = model.predict(X)

# Metrics
accuracy = accuracy_score(y, preds)
precision = precision_score(y, preds, average='weighted')
recall = recall_score(y, preds, average='weighted')
f1 = f1_score(y, preds, average='weighted')

# Display metrics
col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{accuracy:.2f}")
col2.metric("Precision", f"{precision:.2f}")
col3.metric("Recall", f"{recall:.2f}")
col4.metric("F1 Score", f"{f1:.2f}")

# =========================
# ML vs BERT COMPARISON
# =========================
st.subheader("⚔️ ML vs BERT Comparison")

sample_texts = df["review"].head(100)

ml_preds = [ml_model.predict(t)["label"] for t in sample_texts]
bert_preds = [bert_model.predict(t)["label"] for t in sample_texts]

ml_score = sum(["Positive" in p for p in ml_preds]) / len(ml_preds)
bert_score = sum(["Positive" in p for p in bert_preds]) / len(bert_preds)

fig, ax = plt.subplots()
ax.bar(["ML Model", "BERT"], [ml_score, bert_score])
ax.set_title("Positive Prediction Comparison")
st.pyplot(fig)

# =========================
# CONFUSION MATRIX
# =========================
st.subheader("📊 Confusion Matrix")

cm = confusion_matrix(y, preds)
fig2, ax2 = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax2)
st.pyplot(fig2)

# =========================
# PIE CHART
# =========================
st.subheader("🥧 Sentiment Distribution")

sentiment_counts = df["sentiment"].value_counts()

fig3, ax3 = plt.subplots()
ax3.pie(sentiment_counts, labels=["Negative", "Positive", "Neutral"], autopct='%1.1f%%')
ax3.set_title("Sentiment Distribution")
st.pyplot(fig3)

# =========================
# LIVE CHART (REAL-TIME)
# =========================
st.subheader("📈 Live Sentiment Simulation")

import numpy as np

chart_data = pd.DataFrame(
    np.random.randn(20, 1),
    columns=["Sentiment Score"]
)

st.line_chart(chart_data)

# =========================
# DATA PREVIEW
# =========================
st.subheader("📋 Dataset Preview")
st.dataframe(df.head(20))