import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from preprocess import clean_text

MODEL_DIR = Path("models")
MODEL_FILES = {
    "Naive Bayes": "naive_bayes.joblib",
    "Logistic Regression": "logistic_regression.joblib",
    "SVM": "svm.joblib",
}

st.set_page_config(page_title="Email Spam Classifier", page_icon="📧", layout="wide")

if not (MODEL_DIR / "vectorizer.joblib").exists():
    st.error("Models illa. Mudhalla terminal-la `python train.py` run pannunga.")
    st.stop()


@st.cache_resource
def load_models():
    vec = joblib.load(MODEL_DIR / "vectorizer.joblib")
    models = {n: joblib.load(MODEL_DIR / f) for n, f in MODEL_FILES.items()}
    return vec, models


@st.cache_data
def load_metrics():
    with open(MODEL_DIR / "metrics.json") as f:
        return json.load(f)


vectorizer, models = load_models()
metrics = load_metrics()

SAMPLES = {
    "Spam example": "WINNER!! You have won a £1000 cash prize. Call 09061701461 now or click http://claim-prize.com to claim!",
    "Ham example": "Hey, are we still meeting at 5 pm today? Let me know if you want me to bring the notes.",
}

st.title("📧 Email Spam Classifier")
st.caption("NLP + Machine Learning (Naive Bayes, Logistic Regression, SVM)")

tab1, tab2 = st.tabs(["🔍 Predict", "📊 Dashboard"])

# ---------------------------------------------------------------- Predict
with tab1:
    left, right = st.columns([2, 1])

    with right:
        model_name = st.selectbox("Model", list(models.keys()), index=1)
        st.write("Sample try pannunga:")
        for label, text in SAMPLES.items():
            if st.button(label):
                st.session_state["msg"] = text

    with left:
        message = st.text_area("Email / message text inga paste pannunga",
                               key="msg", height=200)
        check = st.button("Check", type="primary")

    if check:
        if not message.strip():
            st.warning("Mudhalla ethaavadhu text type pannunga.")
        else:
            X = vectorizer.transform([clean_text(message)])
            spam_prob = float(models[model_name].predict_proba(X)[0][1])
            if spam_prob >= 0.5:
                st.error(f"🚨 SPAM  (confidence: {spam_prob:.1%})")
            else:
                st.success(f"✅ HAM / Legitimate  (spam probability: {spam_prob:.1%})")
            st.progress(min(max(spam_prob, 0.0), 1.0))

            # Intha message-la ethana spam-indicating words irukku
            spam_words = dict(metrics["top_spam_words"])
            found = [w for w in clean_text(message).split() if w in spam_words]
            if found:
                st.write("Spam-indicating words:", ", ".join(sorted(set(found))))

# -------------------------------------------------------------- Dashboard
with tab2:
    results = pd.DataFrame(metrics["results"]).set_index("Model")
    cc = metrics["class_counts"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total messages", cc["ham"] + cc["spam"])
    c2.metric("Ham", cc["ham"])
    c3.metric("Spam", cc["spam"])

    st.subheader("Model comparison")
    st.dataframe(results.style.format("{:.2%}"), use_container_width=True)
    st.bar_chart(results)

    st.subheader("Confusion matrices")
    cols = st.columns(3)
    for col, (name, cm) in zip(cols, metrics["confusion_matrices"].items()):
        fig, ax = plt.subplots(figsize=(3.5, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(name)
        col.pyplot(fig)
        plt.close(fig)

    st.subheader("Top keywords")
    k1, k2 = st.columns(2)
    for col, key, title, color in [(k1, "top_spam_words", "Spam-indicating words", "crimson"),
                                   (k2, "top_ham_words", "Ham-indicating words", "seagreen")]:
        words, scores = zip(*metrics[key][:15])
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.barh(words[::-1], [abs(s) for s in scores][::-1], color=color)
        ax.set_title(title)
        ax.set_xlabel("Importance (Logistic Regression coefficient)")
        fig.tight_layout()
        col.pyplot(fig)
        plt.close(fig)
