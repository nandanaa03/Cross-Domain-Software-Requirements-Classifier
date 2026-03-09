# Cross-Domain Software Requirements Classification
### Evaluating Software Requirement Classifiers Under Cross-Domain Conditions with Few-Shot Adaptation

---

## What is this?

Most ML models for requirement classification look great on paper — until you test them outside the dataset they were trained on. This project investigates *why* that happens, *how bad* it actually gets, and *how quickly* performance can be recovered with minimal target-domain supervision.

We compared three approaches for classifying Functional vs Non-Functional Requirements, tested them under cross-domain conditions, and measured how quickly performance recovers when a small amount of target-domain data is added.

---

## Datasets

**PROMISE** — 626 requirements, heavily skewed toward NFR (89% Non-Functional)  
**PURE** — 11,440 requirements, opposite skew (83% Functional)

The label distribution literally flips between datasets. That's the core challenge this project addresses.

---

## Models

| Model | Description |
|-------|-------------|
| TF-IDF + Linear SVM | Lexical baseline using unigrams + bigrams |
| Sentence-BERT + SVM | Semantic embeddings (all-MiniLM-L6-v2) with LinearSVC |
| Fine-tuned BERT | BERT (bert-base-uncased) fine-tuned on PROMISE, tested on PURE |

---

## Results

**Cross-domain performance (trained on PROMISE, tested on PURE):**

| Model | Macro F1 |
|-------|----------|
| TF-IDF + SVM | 0.187 |
| SBERT + SVM | 0.282 |
| Fine-tuned BERT | 0.267 |

**SBERT few-shot domain adaptation:**

| PURE Samples Added | SBERT Macro F1 |
|--------------------|----------------|
| 0 | 0.266 |
| 10 | 0.408 |
| 50 | 0.529 |
| 100 | 0.574 |
| 500 | 0.655 |

**BERT few-shot domain adaptation:**

| PURE Samples Added | BERT Macro F1 |
|--------------------|---------------|
| 0 | 0.201 |
| 10 | 0.567 |
| 50 | 0.612 |
| 100 | 0.587 |
| 500 | 0.773 |

Key observation: Fine-tuned BERT outperforms SBERT under few-shot conditions, reaching Macro F1 of 0.773 at 500 samples compared to SBERT's 0.654. However all models suffer severe degradation without any target-domain samples.

---

## Project Structure

```
semantic_classifier/
├── data/
│   ├── raw/              # PROMISE txt and csv files
│   └── pure/             # PURE annotated dataset
├── src/
│   ├── preprocessing.py                # Text cleaning and lemmatization
│   ├── main.py                         # Data preprocessing entry point
│   ├── convert_promise_txt_to_csv.py   # Convert raw PROMISE txt to csv
│   ├── train_svm.py                    # TF-IDF + SVM baseline
│   ├── optimize_models.py              # Hyperparameter tuning
│   ├── sbert.py                        # SBERT cross-domain evaluation
│   ├── sbert_train_on_both.py          # SBERT trained on both datasets
│   ├── sbert_train_on_pure.py          # SBERT trained on PURE only
│   ├── fewshot_domain_adaptation.py    # SBERT few-shot adaptation
│   ├── bert_classifier.py              # Fine-tuned BERT cross-domain
│   ├── bert_fewshot_adaptation.py      # BERT few-shot adaptation
│   ├── analyze_label_distribution.py   # Label distribution analysis
│   ├── plot_fewshot_curve.py           # Plot few-shot results
│   └── generate_visuals.py             # Generate all paper visuals
├── fewshot_adaptation_curve.png
├── bert_vs_sbert_fewshot.png
├── model_comparison_chart.png
├── confusion_matrix.png
├── bert_confusion_matrix.png
├── label_distribution.png
├── f1_scores.png
└── requirements.txt
```

---

## How to Run

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Step 1 — Preprocess data:**
```bash
python src/convert_promise_txt_to_csv.py
python src/main.py
```

**Step 2 — Run baseline (TF-IDF + SVM):**
```bash
python src/train_svm.py
```

**Step 3 — Run SBERT cross-domain evaluation:**
```bash
python src/sbert.py
```

**Step 4 — Run SBERT few-shot adaptation:**
```bash
python src/fewshot_domain_adaptation.py
```

**Step 5 — Run fine-tuned BERT:**
```bash
python src/bert_classifier.py
```

**Step 6 — Run BERT few-shot adaptation:**
```bash
python src/bert_fewshot_adaptation.py
```

**Step 7 — Generate all visuals:**
```bash
python src/generate_visuals.py
```

> ⚠️ BERT training takes 30–60 minutes on CPU. GPU recommended.

---

## Key Findings

- All three models suffer severe performance degradation under cross-domain conditions
- Lexical models (TF-IDF) degrade most sharply under domain shift
- Semantic embeddings (SBERT) provide marginally better cross-domain robustness than TF-IDF
- Label distribution shift — majority class flipping from NFR to FR — is the primary factor in performance degradation
- Even 10 labeled target-domain samples produce a 54% improvement in Macro F1
- Fine-tuned BERT outperforms SBERT under few-shot conditions but requires significantly more computational resources

---

## Reproducibility

All experiments use fixed random seeds where applicable. Results may vary slightly across hardware configurations. BERT experiments require approximately 4–6 hours on CPU. GPU is strongly recommended for BERT training and few-shot adaptation experiments.

---