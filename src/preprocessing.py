import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

if __name__ == "__main__":
    df = pd.read_csv("data/raw/promise.csv")
    df["clean_text"] = df["text"].apply(clean_text)
    df["text_length"] = df["clean_text"].apply(lambda x: len(x.split()))

    df.to_csv("data/cleaned_requirements.csv", index=False)
    print(df)
