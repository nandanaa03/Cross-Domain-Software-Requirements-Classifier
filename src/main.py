import pandas as pd
from preprocessing import clean_text

pd.set_option("display.max_colwidth", None)

# load raw data
df = pd.read_csv("data/sample_requirements.csv")

# preprocess
df["clean_text"] = df["text"].apply(clean_text)

# save processed data
df.to_csv("data/cleaned_requirements.csv", index=False)

print("Cleaned data saved to data/cleaned_requirements.csv")
print(df.to_string(index=False))
