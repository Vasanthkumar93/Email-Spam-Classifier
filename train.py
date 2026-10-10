"""Run: python train.py
Dataset load -> preprocess -> TF-IDF -> 3 models train -> evaluate -> save."""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from preprocess import clean_text

DATA_PATH = "spam.csv"
OUT_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)

# 1. Load data ---------------------------------------------------------
df = pd.read_csv(DATA_PATH, encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "text"]
total_rows = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Rows: {total_rows} -> {len(df)} (duplicates removed)")
print(df["label"].value_counts(), "\n")

df["y"] = (df["label"] == "spam").astype(int)      # spam=1, ham=0
df["clean"] = df["text"].apply(clean_text)

# 2. Split -------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["y"], test_size=0.2, random_state=42, stratify=df["y"])

# 3. TF-IDF ------------------------------------------------------------
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
Xtr = vectorizer.fit_transform(X_train)
Xte = vectorizer.transform(X_test)

# 4. Models ------------------------------------------------------------
models = {
    "Naive Bayes": MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(max_iter=1000, C=10, class_weight="balanced"),
    # LinearSVC-ku probability illa, so calibrate panrom
    "SVM": CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=5),
}

results, matrices = [], {}
for name, model in models.items():
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
    })
    matrices[name] = confusion_matrix(y_test, pred).tolist()
    joblib.dump(model, OUT_DIR / f"{name.replace(' ', '_').lower()}.joblib")

res_df = pd.DataFrame(results).round(4)
print(res_df.to_string(index=False))

# 5. Top spam / ham words (Logistic Regression coefficients) -----------
lr = models["Logistic Regression"]
words = vectorizer.get_feature_names_out()
coef = lr.coef_[0]
order = coef.argsort()
top_spam = [(words[i], float(coef[i])) for i in order[::-1][:20]]
top_ham = [(words[i], float(coef[i])) for i in order[:20]]

# 6. Save --------------------------------------------------------------
joblib.dump(vectorizer, OUT_DIR / "vectorizer.joblib")
with open(OUT_DIR / "metrics.json", "w") as f:
    json.dump({
        "results": res_df.to_dict(orient="records"),
        "confusion_matrices": matrices,
        "top_spam_words": top_spam,
        "top_ham_words": top_ham,
        "class_counts": {"ham": int((df.y == 0).sum()), "spam": int((df.y == 1).sum())},
        "test_size": int(len(y_test)),
    }, f, indent=2)

print("\nSaved everything in 'models/' folder. Ippo: streamlit run app.py")
