import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score


# ==============================
# Load cleaned dataset
# ==============================

df = pd.read_csv("data/cleaned_requirements.csv")

X = df["clean_text"]
y = df["label"]

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 5-fold cross validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Hyperparameter grid
param_grid = {
    "tfidf__ngram_range": [(1,1), (1,2), (1,3)],
    "clf__C": [0.1, 1, 5, 10]
}

# ==============================
# 1️⃣ Linear SVM
# ==============================

svm_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LinearSVC(class_weight="balanced"))
])

svm_grid = GridSearchCV(
    svm_pipeline,
    param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

svm_grid.fit(X_train, y_train)

print("\nBest SVM Params:", svm_grid.best_params_)

svm_preds = svm_grid.best_estimator_.predict(X_test)
svm_f1 = f1_score(y_test, svm_preds, average="macro")

print("\nOptimized Linear SVM")
print("Macro F1:", round(svm_f1, 4))
print(classification_report(y_test, svm_preds))


# ==============================
# 2️⃣ Logistic Regression
# ==============================

log_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(
        class_weight="balanced",
        max_iter=1000
    ))
])

log_grid = GridSearchCV(
    log_pipeline,
    param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

log_grid.fit(X_train, y_train)

print("\nBest Logistic Params:", log_grid.best_params_)

log_preds = log_grid.best_estimator_.predict(X_test)
log_f1 = f1_score(y_test, log_preds, average="macro")

print("\nOptimized Logistic Regression")
print("Macro F1:", round(log_f1, 4))
print(classification_report(y_test, log_preds))


# ==============================
# Comparison Table
# ==============================

comparison = pd.DataFrame({
    "Model": ["Linear SVM", "Logistic Regression"],
    "Macro F1": [svm_f1, log_f1]
})

print("\n===== Experimental Comparison =====")
print(comparison)


# ==============================
# Error Analysis
# ==============================

results = pd.DataFrame({
    "text": X_test,
    "true_label": y_test,
    "predicted_label": svm_preds if svm_f1 >= log_f1 else log_preds
})

errors = results[
    (results["true_label"] == "functional") &
    (results["predicted_label"] != "functional")
]

print("\nMisclassified Functional:", len(errors))
errors.to_csv("misclassified_functional.csv", index=False)

# ==============================
# 6️⃣ Cross-Domain Evaluation (PURE)
# ==============================

print("\n==============================")
print("Cross-Domain Evaluation: PROMISE → PURE")
print("==============================")

import re
from preprocessing import clean_text  # use SAME function


# Load PURE dataset
pure_df = pd.read_csv(
    r"data/pure/Pure_Annotate_dataset.csv",
    encoding="latin1"
)

# Map labels
pure_df["label"] = pure_df["NFR_boolean"].apply(
    lambda x: "non-functional" if x == 1 else "functional"
)

# Clean PURE sentences
pure_df["clean_text"] = pure_df["sentence"].apply(clean_text)

X_pure = pure_df["clean_text"]
y_pure = pure_df["label"]

# Use BEST model from PROMISE
best_model = svm_grid.best_estimator_ if svm_f1 >= log_f1 else log_grid.best_estimator_

pure_preds = best_model.predict(X_pure)

from sklearn.metrics import classification_report, f1_score

pure_macro_f1 = f1_score(y_pure, pure_preds, average="macro")

print("\nPURE Macro F1:", round(pure_macro_f1, 4))
print("\nClassification Report on PURE:")
print(classification_report(y_pure, pure_preds))

# ==============================
# SAVE RESULTS
# ==============================
import os
import json

os.makedirs("results", exist_ok=True)

results_data = {
    "experiment": "Hyperparameter Optimization (GridSearchCV)",
    "train": "PROMISE",
    "best_svm_params": svm_grid.best_params_,
    "best_svm_macro_f1": round(svm_f1, 3),
    "best_logistic_params": log_grid.best_params_,
    "best_logistic_macro_f1": round(log_f1, 3),
    "cross_domain_macro_f1": round(pure_macro_f1, 3),
    "cross_domain_report": classification_report(
        y_pure, pure_preds,
        output_dict=True
    )
}

with open("results/optimize_models_results.json", "w") as f:
    json.dump(results_data, f, indent=4)

print("\nResults saved to results/optimize_models_results.json")