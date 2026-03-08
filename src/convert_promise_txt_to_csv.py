import pandas as pd

def load_txt(file_path, label):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]
    return pd.DataFrame({
        "text": lines,
        "label": label
    })

# Load functional and non-functional requirements
fr_df = load_txt("data/raw/fr.txt", "functional")
nfr_df = load_txt("data/raw/nfr.txt", "non-functional")

# Combine
df = pd.concat([fr_df, nfr_df], ignore_index=True)

# Save as CSV
df.to_csv("data/raw/promise.csv", index=False)

print("PROMISE dataset converted successfully!")
print(df.head())
print("\nClass distribution:")
print(df["label"].value_counts())