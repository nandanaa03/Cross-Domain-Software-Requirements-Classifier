import pandas as pd
import numpy as np
import os
import json

from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from preprocessing import clean_text


# ==============================
# 1. Load PROMISE Dataset
# ==============================

print("Loading PROMISE dataset...")
df = pd.read_csv("data/cleaned_requirements.csv")

X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ==============================
# 2. Load SBERT Model
# ==============================

print("Loading Sentence-BERT model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ==============================
# 3. Encode PROMISE Sentences
# ==============================

print("Encoding PROMISE training data...")
X_train_emb = model.encode(X_train.tolist(), show_progress_bar=True)

print("Encoding PROMISE test data...")
X_test_emb = model.encode(X_test.tolist(), show_progress_bar=True)

# ==============================
# 4. Train Linear SVM on Embeddings
# ==============================

clf = LinearSVC(class_weight="balanced")
clf.fit(X_train_emb, y_train)

# ==============================
# 5. In-Domain Evaluation (PROMISE)
# ==============================

print("\n=== In-Domain Evaluation (PROMISE) ===")

promise_preds = clf.predict(X_test_emb)
promise_macro_f1 = f1_score(y_test, promise_preds, average="macro")

print("PROMISE Macro F1:", round(promise_macro_f1, 4))
print(classification_report(y_test, promise_preds))

# ==============================
# 6. Cross-Domain Evaluation (PURE)
# ==============================

print("\n=== Cross-Domain Evaluation (PROMISE -> PURE) ===")

pure_df = pd.read_csv(
    r"data/pure/Pure_Annotate_dataset.csv",
    encoding="latin1"
)

# Map labels
pure_df["label"] = pure_df["NFR_boolean"].apply(
    lambda x: "non-functional" if x == 1 else "functional"
)

# Clean PURE text
pure_df["clean_text"] = pure_df["sentence"].apply(clean_text)

X_pure = pure_df["clean_text"]
y_pure = pure_df["label"]

print("Encoding PURE dataset...")
X_pure_emb = model.encode(X_pure.tolist(), show_progress_bar=True)

pure_preds = clf.predict(X_pure_emb)
pure_macro_f1 = f1_score(y_pure, pure_preds, average="macro")

print("PURE Macro F1:", round(pure_macro_f1, 4))
print(classification_report(y_pure, pure_preds))

# ==============================
# 7. Save Results
# ==============================

os.makedirs("results", exist_ok=True)

results = {
    "experiment": "SBERT + SVM Cross-Domain",
    "train": "PROMISE",
    "test": "PURE",
    "in_domain_macro_f1": round(promise_macro_f1, 3),
    "cross_domain_macro_f1": round(pure_macro_f1, 3),
    "cross_domain_report": classification_report(
        y_pure, pure_preds,
        target_names=["Functional", "Non-Functional"],
        output_dict=True
    )
}

with open("results/sbert_svm_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nResults saved to results/sbert_svm_results.json")
