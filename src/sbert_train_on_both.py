import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score
from preprocessing import clean_text


# =====================================================
# 1️⃣ Load PROMISE
# =====================================================

print("Loading PROMISE dataset...")

promise_df = pd.read_csv("data/cleaned_requirements.csv")

promise_X = promise_df["clean_text"]
promise_y = promise_df["label"]


# =====================================================
# 2️⃣ Load PURE
# =====================================================

print("Loading PURE dataset...")

pure_df = pd.read_csv(
    r"data/pure/Pure_Annotate_dataset.csv",
    encoding="latin1"
)

pure_df["label"] = pure_df["NFR_boolean"].apply(
    lambda x: "non-functional" if x == 1 else "functional"
)

pure_df["clean_text"] = pure_df["sentence"].apply(clean_text)

pure_X = pure_df["clean_text"]
pure_y = pure_df["label"]


# =====================================================
# 3️⃣ Combine Datasets (TRAIN ON BOTH)
# =====================================================

print("Combining datasets...")

X_train = pd.concat([promise_X, pure_X])
y_train = pd.concat([promise_y, pure_y])


# =====================================================
# 4️⃣ Load SBERT
# =====================================================

print("Loading SBERT model...")
model = SentenceTransformer("all-MiniLM-L6-v2")


# =====================================================
# 5️⃣ Encode Training Data
# =====================================================

print("Encoding combined training data...")
X_train_emb = model.encode(X_train.tolist(), show_progress_bar=True)


# =====================================================
# 6️⃣ Train Classifier
# =====================================================

clf = LinearSVC(class_weight="balanced")
clf.fit(X_train_emb, y_train)


# =====================================================
# 7️⃣ Evaluate on PROMISE
# =====================================================

print("\n=== Evaluation on PROMISE ===")

X_promise_emb = model.encode(promise_X.tolist(), show_progress_bar=True)

promise_preds = clf.predict(X_promise_emb)
promise_macro_f1 = f1_score(promise_y, promise_preds, average="macro")

print("PROMISE Macro F1:", round(promise_macro_f1, 4))
print(classification_report(promise_y, promise_preds))


# =====================================================
# 8️⃣ Evaluate on PURE
# =====================================================

print("\n=== Evaluation on PURE ===")

X_pure_emb = model.encode(pure_X.tolist(), show_progress_bar=True)

pure_preds = clf.predict(X_pure_emb)
pure_macro_f1 = f1_score(pure_y, pure_preds, average="macro")

print("PURE Macro F1:", round(pure_macro_f1, 4))
print(classification_report(pure_y, pure_preds))