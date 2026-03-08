# BERT Cross-Domain Classifier
# Train on PROMISE → Test on PURE

import pandas as pd
import numpy as np
import torch
import os
import json
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ================================================
# STEP 1 — LOAD DATA
# ================================================
print("Step 1: Loading datasets...")

train_df = pd.read_csv('data/raw/promise.csv')
test_df = pd.read_csv('data/pure/Pure_Annotate_Dataset.csv', encoding='latin-1')

train_texts = train_df['text'].astype(str).tolist()
train_labels = [1 if label == 'non-functional' else 0
                for label in train_df['label']]

test_texts = test_df['sentence'].astype(str).tolist()
test_labels = test_df['NFR_boolean'].astype(int).tolist()

print(f"PROMISE: {len(train_texts)} requirements")
print(f"PURE: {len(test_texts)} requirements")
print(f"PROMISE — FR: {train_labels.count(0)}, NFR: {train_labels.count(1)}")
print(f"PURE — FR: {test_labels.count(0)}, NFR: {test_labels.count(1)}")

# ================================================
# STEP 2 — TOKENIZER
# ================================================
print("\nStep 2: Loading BERT tokenizer...")
print("First run will download files. Takes 2-3 minutes...")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("Tokenizer ready")

# ================================================
# STEP 3 — DATASET CLASS
# ================================================
print("\nStep 3: Preparing datasets...")

class RequirementsDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors=None
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx])
        }
        return item

train_dataset = RequirementsDataset(train_texts, train_labels)
test_dataset = RequirementsDataset(test_texts, test_labels)

print(f"Train dataset ready: {len(train_dataset)} samples")
print(f"Test dataset ready: {len(test_dataset)} samples")

# ================================================
# STEP 4 — LOAD BERT MODEL
# ================================================
print("\nStep 4: Loading BERT model...")
print("First run downloads ~400MB. Please wait...")

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)
print("BERT model loaded")

# ================================================
# STEP 5 — TRAINING SETTINGS
# ================================================
print("\nStep 5: Setting up training...")

training_args = TrainingArguments(
    output_dir='./bert_output',
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=50,
    weight_decay=0.01,
    logging_steps=20,
    evaluation_strategy='epoch',
    save_strategy='no',
    report_to='none'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# ================================================
# STEP 6 — TRAIN
# ================================================
print("\nStep 6: Training BERT on PROMISE...")
print("This takes 30-60 minutes on CPU")
print("Do NOT close the terminal")
print("-" * 50)

trainer.train()

# ================================================
# STEP 7 — EVALUATE ON PURE
# ================================================
print("\n" + "=" * 50)
print("Step 7: Evaluating on PURE...")

output = trainer.predict(test_dataset)
predicted = output.predictions.argmax(axis=1)
true = np.array(test_labels)

macro_f1 = f1_score(true, predicted, average='macro')
report = classification_report(
    true,
    predicted,
    target_names=['Functional', 'Non-Functional'],
    output_dict=True
)

print("\n" + "=" * 50)
print("FINAL RESULTS")
print("Train: PROMISE | Test: PURE")
print("=" * 50)
print(f"\nMacro F1 Score: {macro_f1:.3f}")
print("\nDetailed Report:")
print(classification_report(
    true,
    predicted,
    target_names=['Functional', 'Non-Functional']
))
print("=" * 50)

# ================================================
# STEP 8 — SAVE RESULTS
# ================================================
os.makedirs("results", exist_ok=True)

results = {
    "experiment": "Fine-tuned BERT Cross-Domain",
    "model": "bert-base-uncased",
    "train": "PROMISE",
    "test": "PURE",
    "epochs": 4,
    "batch_size": 8,
    "max_length": 128,
    "macro_f1": round(macro_f1, 3),
    "classification_report": report
}

with open("results/bert_classifier_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Results saved to results/bert_classifier_results.json")
