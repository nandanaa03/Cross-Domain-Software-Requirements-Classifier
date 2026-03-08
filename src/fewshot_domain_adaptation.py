import pandas as pd
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from preprocessing import clean_text


# =====================================================
# 1. Load PROMISE (Source Domain)
# =====================================================

print("Loading PROMISE...")
promise_df = pd.read_csv("data/cleaned_requirements.csv")

X_promise = promise_df["clean_text"]
y_promise = promise_df["label"]


# =====================================================
# 2. Load PURE (Target Domain)
# =====================================================

print("Loading PURE...")
pure_df = pd.read_csv(
    r"data/pure/Pure_Annotate_dataset.csv",
    encoding="latin1"
)

pure_df["label"] = pure_df["NFR_boolean"].apply(
    lambda x: "non-functional" if x == 1 else "functional"
)

pure_df["clean_text"] = pure_df["sentence"].apply(clean_text)

X_pure = pure_df["clean_text"]
y_pure = pure_df["label"]

# Split PURE into few-shot pool (20%) + locked evaluation set (80%)
X_pure_train_pool, X_pure_test, y_pure_train_pool, y_pure_test = train_test_split(
    X_pure,
    y_pure,
    test_size=0.8,
    stratify=y_pure,
    random_state=42
)


# =====================================================
# 3. Load SBERT
# =====================================================

print("Loading SBERT...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode all PURE test data once (fixed evaluation set)
print("Encoding PURE evaluation set...")
X_pure_test_emb = model.encode(X_pure_test.tolist(), show_progress_bar=True)


# =====================================================
# 4. Few-Shot Sizes
# =====================================================

few_shot_sizes = [0, 10, 50, 100, 500]

results = []
detailed_results = {}

for k in few_shot_sizes:

    print(f"\n=== Few-Shot Size: {k} ===")

    # Sample k PURE examples
    if k > 0:
        X_few = X_pure_train_pool.sample(n=k, random_state=42)
        y_few = y_pure_train_pool.loc[X_few.index]
    else:
        X_few = pd.Series([], dtype=str)
        y_few = pd.Series([], dtype=str)

    # Combine PROMISE + Few-shot PURE
    X_train = pd.concat([X_promise, X_few])
    y_train = pd.concat([y_promise, y_few])

    # Encode training
    X_train_emb = model.encode(X_train.tolist(), show_progress_bar=True)

    # Train classifier
    clf = LinearSVC(class_weight="balanced")
    clf.fit(X_train_emb, y_train)

    # Evaluate on PURE test set
    preds = clf.predict(X_pure_test_emb)
    macro_f1 = f1_score(y_pure_test, preds, average="macro")

    print("PURE Macro F1:", round(macro_f1, 4))

    results.append((k, macro_f1))
    detailed_results[str(k)] = {
        "macro_f1": round(macro_f1, 3),
        "report": classification_report(
            y_pure_test, preds,
            target_names=["Functional", "Non-Functional"],
            output_dict=True
        )
    }


# =====================================================
# 5. Final Summary
# =====================================================

print("\n=== Few-Shot Results ===")
print(f"{'Samples':<12} {'Macro F1'}")
print("-" * 25)
for k, score in results:
    print(f"{k:<12} {round(score, 3)}")


# =====================================================
# 6. Save Results
# =====================================================

os.makedirs("results", exist_ok=True)

output = {
    "experiment": "SBERT Few-Shot Domain Adaptation",
    "train_source": "PROMISE",
    "target_domain": "PURE",
    "pure_split": "20% adaptation pool / 80% locked test set",
    "few_shot_sizes": few_shot_sizes,
    "results": detailed_results
}

with open("results/sbert_fewshot_results.json", "w") as f:
    json.dump(output, f, indent=4)

# Save summary CSV
summary_df = pd.DataFrame(results, columns=["samples", "macro_f1"])
summary_df.to_csv("results/sbert_fewshot_summary.csv", index=False)

print("\nResults saved to results/sbert_fewshot_results.json")
print("Summary saved to results/sbert_fewshot_summary.csv")
