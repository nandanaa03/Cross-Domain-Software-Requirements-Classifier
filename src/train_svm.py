import matplotlib
matplotlib.use("Agg")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

# Load cleaned data
df = pd.read_csv("data/cleaned_requirements.csv")

X = df["clean_text"]
y = df["label"]

# Stratified Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 🔹 PIPELINE (TF-IDF + SVM together)
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),   # unigrams + bigrams
        max_df=0.9,
        min_df=2
    )),
    ("clf", LinearSVC(class_weight="balanced"))
])

# Train
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#Heapmap of Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=["functional", "non-functional"])

plt.figure(figsize=(6,5))
sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="rocket",
            xticklabels=["Functional", "Non-Functional"],
            yticklabels=["Functional", "Non-Functional"])

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (SVM + TF-IDF Bigrams)")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# F1 bar chart
report = classification_report(y_test, y_pred, output_dict=True)

labels = ["Functional", "Non-Functional"]
f1_scores = [
    report["functional"]["f1-score"],
    report["non-functional"]["f1-score"]
]

plt.figure(figsize=(6,5))
plt.bar(labels, f1_scores,
        color=["#2E004F", "#FF4D6D"])
plt.ylim(0, 1)
plt.ylabel("F1 Score")
plt.title("F1 Score per Class")
plt.tight_layout()
plt.savefig("f1_scores.png")
plt.close()

# ================================================
# CROSS-DOMAIN EVALUATION ON PURE
# ================================================

print("\n" + "="*50)
print("CROSS-DOMAIN EVALUATION")
print("Train: PROMISE → Test: PURE")
print("="*50)

# Load PURE
pure_df = pd.read_csv("data/pure/Pure_Annotate_Dataset.csv", encoding="latin-1")
pure_df["label"] = pure_df["NFR_boolean"].apply(
    lambda x: "non-functional" if x == 1 else "functional"
)

# Preprocess PURE using same cleaning
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

pure_df["clean_text"] = pure_df["sentence"].astype(str).apply(clean_text)

# Retrain on full PROMISE
full_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=2)),
    ("clf", LinearSVC(class_weight="balanced"))
])
full_pipeline.fit(X, y)

# Test on PURE
X_pure = pure_df["clean_text"]
y_pure = pure_df["label"]
y_pure_pred = full_pipeline.predict(X_pure)

macro_f1 = f1_score(y_pure, y_pure_pred, average="macro")

print(f"\nMacro F1 Score: {macro_f1:.3f}")
print("\nDetailed Report:")
print(classification_report(y_pure, y_pure_pred,
      target_names=["Functional", "Non-Functional"]))
print("="*50)
print("This is your cross-domain result for the paper")
print("="*50)

# ================================================
# SAVE RESULTS
# ================================================
import os
import json

os.makedirs("results", exist_ok=True)

results = {
    "experiment": "TF-IDF + SVM Cross-Domain",
    "train": "PROMISE",
    "test": "PURE",
    "macro_f1": round(macro_f1, 3),
    "classification_report": classification_report(
        y_pure, y_pure_pred,
        target_names=["Functional", "Non-Functional"],
        output_dict=True
    )
}

with open("results/tfidf_svm_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Results saved to results/tfidf_svm_results.json")