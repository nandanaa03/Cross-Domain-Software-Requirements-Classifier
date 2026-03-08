# BERT Few-Shot Domain Adaptation
# Train on PROMISE + N samples from PURE
# Test on held-out PURE test set

import pandas as pd
import numpy as np
import torch
import random
import os
import json
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ================================================
# STEP 1 — LOAD DATA
# ================================================
print("Step 1: Loading datasets...")

promise_df = pd.read_csv('data/raw/promise.csv')
pure_df = pd.read_csv(
    'data/pure/Pure_Annotate_Dataset.csv',
    encoding='latin-1'
)

promise_texts = promise_df['text'].astype(str).tolist()
promise_labels = [
    1 if label == 'non-functional' else 0
    for label in promise_df['label']
]

pure_texts = pure_df['sentence'].astype(str).tolist()
pure_labels = pure_df['NFR_boolean'].astype(int).tolist()

print(f"PROMISE: {len(promise_texts)} requirements")
print(f"PURE: {len(pure_texts)} requirements")

# ================================================
# STEP 2 — SPLIT PURE INTO POOL AND TEST
# ================================================
print("\nStep 2: Splitting PURE into adaptation pool and test set...")

(
    pure_pool_texts,
    pure_test_texts,
    pure_pool_labels,
    pure_test_labels
) = train_test_split(
    pure_texts,
    pure_labels,
    test_size=0.8,
    random_state=42,
    stratify=pure_labels
)

print(f"PURE adaptation pool: {len(pure_pool_texts)} samples")
print(f"PURE test set: {len(pure_test_texts)} samples")
print(f"Test FR: {pure_test_labels.count(0)}")
print(f"Test NFR: {pure_test_labels.count(1)}")

# ================================================
# STEP 3 — TOKENIZER
# ================================================
print("\nStep 3: Loading BERT tokenizer...")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("Tokenizer ready")

# ================================================
# STEP 4 — DATASET CLASS
# ================================================
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

# ================================================
# STEP 5 — FEW SHOT EXPERIMENT
# ================================================

sample_sizes = [0, 10, 50, 100, 500]

results = []
detailed_results = {}

print("\n" + "="*50)
print("Starting BERT Few-Shot Adaptation Experiment")
print("="*50)

for n_samples in sample_sizes:

    print(f"\nRunning experiment: {n_samples} PURE samples added")
    print("-"*40)

    if n_samples == 0:
        train_texts = promise_texts.copy()
        train_labels = promise_labels.copy()
        print("Training on PROMISE only")

    else:
        random.seed(42)
        indices = random.sample(
            range(len(pure_pool_texts)),
            min(n_samples, len(pure_pool_texts))
        )

        few_shot_texts = [pure_pool_texts[i] for i in indices]
        few_shot_labels = [pure_pool_labels[i] for i in indices]

        train_texts = promise_texts + few_shot_texts
        train_labels = promise_labels + few_shot_labels

        print(f"Training on PROMISE + {n_samples} PURE samples")
        print(f"Total training size: {len(train_texts)}")

    train_dataset = RequirementsDataset(train_texts, train_labels)
    test_dataset = RequirementsDataset(pure_test_texts, pure_test_labels)

    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=f'./bert_fewshot_output_{n_samples}',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=30,
        weight_decay=0.01,
        logging_steps=50,
        evaluation_strategy='no',
        save_strategy='no',
        report_to='none'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    print("Training...")
    trainer.train()

    print("Evaluating on PURE test set...")
    output = trainer.predict(test_dataset)
    predicted = output.predictions.argmax(axis=1)
    true = np.array(pure_test_labels)

    macro_f1 = f1_score(true, predicted, average='macro')
    report = classification_report(
        true,
        predicted,
        target_names=['Functional', 'Non-Functional'],
        output_dict=True
    )

    results.append({
        'samples': n_samples,
        'macro_f1': round(macro_f1, 3)
    })

    detailed_results[str(n_samples)] = {
        'macro_f1': round(macro_f1, 3),
        'report': report
    }

    print(f"Samples: {n_samples} | Macro F1: {macro_f1:.3f}")

# ================================================
# STEP 6 — PRINT FINAL RESULTS
# ================================================
print("\n" + "="*50)
print("BERT FEW-SHOT ADAPTATION RESULTS")
print("="*50)
print(f"\n{'Samples':<15} {'Macro F1':<15}")
print("-"*30)

for r in results:
    print(f"{r['samples']:<15} {r['macro_f1']:<15}")

print("\n" + "="*50)
print("COMPARISON WITH SBERT FEW-SHOT RESULTS")
print("="*50)

sbert_results = {
    0: 0.266,
    10: 0.408,
    50: 0.529,
    100: 0.574,
    500: 0.654
}

print(f"\n{'Samples':<15} {'SBERT F1':<15} {'BERT F1':<15}")
print("-"*45)

for r in results:
    n = r['samples']
    print(f"{n:<15} {sbert_results[n]:<15} {r['macro_f1']:<15}")

print("\n" + "="*50)
print("DONE. Copy these results into your paper.")
print("="*50)

# ================================================
# STEP 7 — SAVE RESULTS
# ================================================
os.makedirs("results", exist_ok=True)

output_data = {
    "experiment": "BERT Few-Shot Domain Adaptation",
    "model": "bert-base-uncased",
    "train_source": "PROMISE",
    "target_domain": "PURE",
    "pure_split": "20% adaptation pool / 80% locked test set",
    "epochs": 3,
    "batch_size": 8,
    "sample_sizes": sample_sizes,
    "results": detailed_results,
    "sbert_comparison": sbert_results
}

with open("results/bert_fewshot_results.json", "w") as f:
    json.dump(output_data, f, indent=4)

results_df = pd.DataFrame(results)
results_df.to_csv("results/bert_fewshot_summary.csv", index=False)

print("\nResults saved to results/bert_fewshot_results.json")
print("Summary saved to results/bert_fewshot_summary.csv")
