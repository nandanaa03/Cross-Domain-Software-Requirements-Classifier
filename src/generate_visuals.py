# generate_visuals.py
# Generates all visuals for the research paper
# Loads results from results/ folder automatically

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import json
import os
import warnings
warnings.filterwarnings('ignore')

print("Generating all visuals...")

# ================================================
# LOAD RESULTS FROM FILES
# ================================================

# Load SBERT few-shot results
try:
    sbert_summary = pd.read_csv("results/sbert_fewshot_summary.csv")
    samples = sbert_summary["samples"].tolist()
    sbert_f1 = sbert_summary["macro_f1"].tolist()
    print("Loaded SBERT few-shot results from results/sbert_fewshot_summary.csv")
except FileNotFoundError:
    print("WARNING: sbert_fewshot_summary.csv not found — using hardcoded values")
    samples = [0, 10, 50, 100, 500]
    sbert_f1 = [0.266, 0.408, 0.529, 0.574, 0.654]

# Load BERT few-shot results
try:
    bert_summary = pd.read_csv("results/bert_fewshot_summary.csv")
    bert_fewshot_f1 = bert_summary["macro_f1"].tolist()
    print("Loaded BERT few-shot results from results/bert_fewshot_summary.csv")
except FileNotFoundError:
    print("WARNING: bert_fewshot_summary.csv not found — using hardcoded values")
    bert_fewshot_f1 = [0.201, 0.567, 0.612, 0.587, 0.773]

# Load BERT cross-domain result
try:
    with open("results/bert_classifier_results.json") as f:
        bert_results = json.load(f)
    bert_cross_domain_f1 = bert_results["macro_f1"]
    print("Loaded BERT cross-domain result from results/bert_classifier_results.json")
except FileNotFoundError:
    print("WARNING: bert_classifier_results.json not found — using hardcoded value")
    bert_cross_domain_f1 = 0.228

# Load SBERT cross-domain result
try:
    with open("results/sbert_svm_results.json") as f:
        sbert_results = json.load(f)
    sbert_cross_domain_f1 = sbert_results["cross_domain_macro_f1"]
    print("Loaded SBERT cross-domain result from results/sbert_svm_results.json")
except FileNotFoundError:
    print("WARNING: sbert_svm_results.json not found — using hardcoded value")
    sbert_cross_domain_f1 = 0.266

# Load TF-IDF cross-domain result
try:
    with open("results/tfidf_svm_results.json") as f:
        tfidf_results = json.load(f)
    tfidf_cross_domain_f1 = tfidf_results["macro_f1"]
    print("Loaded TF-IDF cross-domain result from results/tfidf_svm_results.json")
except FileNotFoundError:
    print("WARNING: tfidf_svm_results.json not found — using hardcoded value")
    tfidf_cross_domain_f1 = 0.187

# Model comparison
models = ['TF-IDF\n+SVM', 'SBERT\n+SVM', 'Fine-tuned\nBERT']
cross_domain_f1 = [tfidf_cross_domain_f1, sbert_cross_domain_f1, bert_cross_domain_f1]

# BERT confusion matrix values from actual results
bert_true = [0] * 9532 + [1] * 1908
bert_pred = (
    [0] * 953 + [1] * 8579 +
    [0] * 191 + [1] * 1717
)

os.makedirs("results", exist_ok=True)

# ================================================
# VISUAL 1 — BERT CONFUSION MATRIX
# ================================================
print("\nCreating BERT confusion matrix...")

fig, ax = plt.subplots(figsize=(7, 6))

cm_bert = confusion_matrix(bert_true, bert_pred)
cm_bert_pct = cm_bert.astype('float') / cm_bert.sum(axis=1)[:, np.newaxis] * 100

labels = ['Functional', 'Non-Functional']
sns.heatmap(
    cm_bert_pct,
    annot=True,
    fmt='.1f',
    cmap='Blues',
    xticklabels=labels,
    yticklabels=labels,
    ax=ax,
    linewidths=0.5,
    linecolor='gray',
    annot_kws={"size": 13}
)

ax.set_xlabel('Predicted Label', fontsize=12, labelpad=10)
ax.set_ylabel('True Label', fontsize=12, labelpad=10)
ax.set_title('BERT Confusion Matrix\n(Train: PROMISE → Test: PURE)', fontsize=13, pad=15)

for i in range(2):
    for j in range(2):
        ax.text(j + 0.5, i + 0.72,
                f'n={cm_bert[i][j]:,}',
                ha='center', va='center',
                fontsize=9, color='gray')

plt.tight_layout()
plt.savefig('bert_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: bert_confusion_matrix.png")

# ================================================
# VISUAL 2 — COMBINED MODEL COMPARISON BAR CHART
# ================================================
print("Creating model comparison bar chart...")

fig, ax = plt.subplots(figsize=(9, 6))

colors = ['#d32f2f', '#1976d2', '#388e3c']
bars = ax.bar(models, cross_domain_f1, color=colors, width=0.5,
              edgecolor='white', linewidth=1.2)

for bar, val in zip(bars, cross_domain_f1):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.008,
        f'{val:.3f}',
        ha='center', va='bottom',
        fontsize=13, fontweight='bold'
    )

ax.axhline(y=0.99, color='orange', linestyle='--', linewidth=1.5, alpha=0.8)
ax.text(2.42, 0.992, 'Same-domain\nbenchmark (0.99)',
        fontsize=9, color='orange', va='bottom', ha='right')

ax.set_ylabel('Macro F1 Score', fontsize=12)
ax.set_title('Cross-Domain Performance Comparison\n(Train: PROMISE → Test: PURE)',
             fontsize=13, pad=15)
ax.set_ylim(0, 1.05)
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('model_comparison_chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: model_comparison_chart.png")

# ================================================
# VISUAL 3 — BERT vs SBERT FEW-SHOT CURVE
# ================================================
print("Creating BERT vs SBERT few-shot curve...")

fig, ax = plt.subplots(figsize=(9, 6))

# SBERT line
ax.plot(samples, sbert_f1,
        marker='o', markersize=8, linewidth=2.5,
        color='#1976d2', label='Sentence-BERT + SVM',
        zorder=3)

for x, y in zip(samples, sbert_f1):
    ax.annotate(f'{y:.3f}',
                (x, y),
                textcoords="offset points",
                xytext=(0, 12),
                ha='center', fontsize=9, color='#1976d2')

# BERT line
ax.plot(samples, bert_fewshot_f1,
        marker='s', markersize=8, linewidth=2.5,
        color='#388e3c', label='Fine-tuned BERT',
        zorder=3)

for x, y in zip(samples, bert_fewshot_f1):
    ax.annotate(f'{y:.3f}',
                (x, y),
                textcoords="offset points",
                xytext=(0, -18),
                ha='center', fontsize=9, color='#388e3c')

ax.set_xlabel('Number of PURE Samples Added to Training', fontsize=12)
ax.set_ylabel('Macro F1 Score', fontsize=12)
ax.set_title('Few-Shot Domain Adaptation Curve\nSBERT vs BERT',
             fontsize=13, pad=15)
ax.set_xticks(samples)
ax.set_ylim(0.1, 0.85)
ax.grid(alpha=0.3, linestyle='--')
ax.legend(fontsize=11, loc='lower right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('bert_vs_sbert_fewshot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: bert_vs_sbert_fewshot.png")

# ================================================
# DONE
# ================================================
print("\nAll visuals generated successfully.")
print("Files saved:")
print("  bert_confusion_matrix.png")
print("  model_comparison_chart.png")
print("  bert_vs_sbert_fewshot.png")
