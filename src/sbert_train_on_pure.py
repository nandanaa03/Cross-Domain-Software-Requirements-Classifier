import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from preprocessing import clean_text


# =====================================================
# 1️⃣ Load PURE Dataset (TRAIN SOURCE)
# =====================================================

print("Loading PURE dataset...")

pure_df = pd.read_csv(
    r"C:\Users\samri\Downloads\puredataset\Pure_Annotate_Dataset.csv",
    encoding="latin1"
)

# Map labels
pure_df["label"] = pure_df["NFR_boolean"].apply(
    lambda x: "non-functional" if x == 1 else "functional"
)

# Clean sentences
pure_df["clean_text"] = pure_df["sentence"].apply(clean_text)

X = pure_df["clean_text"]
y = pure_df["label"]

# Train/test split PURE (in-domain evaluation)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# =====================================================
# 2️⃣ Load SBERT Model
# =====================================================

print("Loading SBERT model...")
model = SentenceTransformer("all-MiniLM-L6-v2")


# =====================================================
# 3️⃣ Encode PURE Training Data
# =====================================================

print("Encoding PURE training data...")
X_train_emb = model.encode(X_train.tolist(), show_progress_bar=True)

print("Encoding PURE test data...")
X_test_emb = model.encode(X_test.tolist(), show_progress_bar=True)


# =====================================================
# 4️⃣ Train Classifier
# =====================================================

clf = LinearSVC(class_weight="balanced")
clf.fit(X_train_emb, y_train)


# =====================================================
# 5️⃣ In-Domain Evaluation (PURE → PURE)
# =====================================================

print("\n=== In-Domain Evaluation (PURE) ===")

pure_preds = clf.predict(X_test_emb)
pure_macro_f1 = f1_score(y_test, pure_preds, average="macro")

print("PURE Macro F1:", round(pure_macro_f1, 4))
print(classification_report(y_test, pure_preds))


# =====================================================
# 6️⃣ Cross-Domain Evaluation (PURE → PROMISE)
# =====================================================

print("\n=== Cross-Domain Evaluation (PURE → PROMISE) ===")

promise_df = pd.read_csv("data/cleaned_requirements.csv")

X_promise = promise_df["clean_text"]
y_promise = promise_df["label"]

print("Encoding PROMISE dataset...")
X_promise_emb = model.encode(X_promise.tolist(), show_progress_bar=True)

promise_preds = clf.predict(X_promise_emb)
promise_macro_f1 = f1_score(y_promise, promise_preds, average="macro")

print("PROMISE Macro F1:", round(promise_macro_f1, 4))
print(classification_report(y_promise, promise_preds))