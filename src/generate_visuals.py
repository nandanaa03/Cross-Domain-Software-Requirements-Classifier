# generate_visuals.py
# Generates all visuals for the research paper
# Loads results from results/ folder automatically

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import json
import os
import warnings
warnings.filterwarnings('ignore')

print("Generating all visuals...")
os.makedirs("results", exist_ok=True)

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
    sbert_f1 = [0.266, 0.409, 0.529, 0.574, 0.655]

# Load BERT few-shot results
try:
    bert_summary = pd.read_csv("results/bert_fewshot_summary.csv")
    bert_fewshot_f1 = bert_summary["macro_f1"].tolist()
    print("Loaded BERT few-shot results from results/bert_fewshot_summary.csv")
except FileNotFoundError:
    print("WARNING: bert_fewshot_summary.csv not found — using hardcoded values")
    bert_fewshot_f1 = [0.201, 0.567, 0.612, 0.587, 0.773]

# Load BERT cross-domain result + classification report if available
try:
    with open("results/bert_classifier_results.json") as f:
        bert_results = json.load(f)
    bert_cross_domain_f1 = bert_results["macro_f1"]
    bert_report = bert_results.get("classification_report", None)
    print("Loaded BERT cross-domain result from results/bert_classifier_results.json")
except FileNotFoundError:
    print("WARNING: bert_classifier_results.json not found — using hardcoded value")
    bert_cross_domain_f1 = 0.228
    bert_report = None

# Load SBERT cross-domain result
try:
    with open("results/sbert_svm_results.json") as f:
        sbert_results = json.load(f)
    sbert_cross_domain_f1 = sbert_results["cross_domain_macro_f1"]
    print("Loaded SBERT cross-domain result from results/sbert_svm_results.json")
except FileNotFoundError:
    print("WARNING: sbert_svm_results.json not found — using hardcoded value")
    sbert_cross_domain_f1 = 0.282

# Load TF-IDF cross-domain result + classification report
try:
    with open("results/tfidf_svm_results.json") as f:
        tfidf_results = json.load(f)
    tfidf_cross_domain_f1 = tfidf_results["macro_f1"]
    tfidf_report = tfidf_results.get("classification_report", None)
    print("Loaded TF-IDF cross-domain result from results/tfidf_svm_results.json")
except FileNotFoundError:
    print("WARNING: tfidf_svm_results.json not found — using hardcoded value")
    tfidf_cross_domain_f1 = 0.187
    tfidf_report = None

# Model comparison
models = ['TF-IDF\n+SVM', 'SBERT\n+SVM', 'Fine-tuned\nBERT']
cross_domain_f1 = [tfidf_cross_domain_f1, sbert_cross_domain_f1, bert_cross_domain_f1]
labels = ['Functional', 'Non-Functional']

# ================================================
# VISUAL 1 — TF-IDF CONFUSION MATRIX (from results file)
# ================================================
print("\nCreating TF-IDF confusion matrix...")

if tfidf_report:
    fr_support = int(tfidf_report["Functional"]["support"])
    nfr_support = int(tfidf_report["Non-Functional"]["support"])
    fr_recall = tfidf_report["Functional"]["recall"]
    nfr_recall = tfidf_report["Non-Functional"]["recall"]
    tp_fr = int(round(fr_recall * fr_support))
    fn_fr = fr_support - tp_fr
    tp_nfr = int(round(nfr_recall * nfr_support))
    fn_nfr = nfr_support - tp_nfr
    tfidf_cm = np.array([[tp_fr, fn_fr],
                         [fn_nfr, tp_nfr]])
else:
    # From known results: FR recall=0.05, support=9532 | NFR recall=0.93, support=1908
    tfidf_cm = np.array([[480, 9052],
                         [134, 1774]])

fig, ax = plt.subplots(figsize=(7, 6))
tfidf_cm_pct = tfidf_cm.astype('float') / tfidf_cm.sum(axis=1)[:, np.newaxis] * 100

sns.heatmap(
    tfidf_cm_pct,
    annot=True,
    fmt='.1f',
    cmap='Greys',
    xticklabels=labels,
    yticklabels=labels,
    ax=ax,
    linewidths=0.5,
    linecolor='white',
    annot_kws={"size": 13}
)

for i in range(2):
    for j in range(2):
        ax.text(j + 0.5, i + 0.72,
                f'n={tfidf_cm[i][j]:,}',
                ha='center', va='center',
                fontsize=9, color='white')

ax.set_xlabel('Predicted Label', fontsize=12, labelpad=10)
ax.set_ylabel('True Label', fontsize=12, labelpad=10)
ax.set_title('TF-IDF + SVM Confusion Matrix\n(Train: PROMISE → Test: PURE)', fontsize=13, pad=15)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: confusion_matrix.png")

# ================================================
# VISUAL 2 — BERT CONFUSION MATRIX
# ================================================
print("Creating BERT confusion matrix...")

if bert_report:
    fr_key = "Functional" if "Functional" in bert_report else "0"
    nfr_key = "Non-Functional" if "Non-Functional" in bert_report else "1"
    fr_support = int(bert_report[fr_key]["support"])
    nfr_support = int(bert_report[nfr_key]["support"])
    fr_recall = bert_report[fr_key]["recall"]
    nfr_recall = bert_report[nfr_key]["recall"]
    tp_fr = int(round(fr_recall * fr_support))
    fn_fr = fr_support - tp_fr
    tp_nfr = int(round(nfr_recall * nfr_support))
    fn_nfr = nfr_support - tp_nfr
    bert_cm = np.array([[tp_fr, fn_fr],
                        [fn_nfr, tp_nfr]])
    title_note = "Train: PROMISE → Test: PURE"
else:
    # Approximate from known results: FR recall=0.10, NFR recall=0.90
    bert_cm = np.array([[953, 8579],
                        [191, 1717]])
    title_note = "Train: PROMISE → Test: PURE (approximate)"

fig, ax = plt.subplots(figsize=(7, 6))
bert_cm_pct = bert_cm.astype('float') / bert_cm.sum(axis=1)[:, np.newaxis] * 100

sns.heatmap(
    bert_cm_pct,
    annot=True,
    fmt='.1f',
    cmap='Greys',
    xticklabels=labels,
    yticklabels=labels,
    ax=ax,
    linewidths=0.5,
    linecolor='gray',
    annot_kws={"size": 13}
)

for i in range(2):
    for j in range(2):
        ax.text(j + 0.5, i + 0.72,
                f'n={bert_cm[i][j]:,}',
                ha='center', va='center',
                fontsize=9, color='gray')

ax.set_xlabel('Predicted Label', fontsize=12, labelpad=10)
ax.set_ylabel('True Label', fontsize=12, labelpad=10)
ax.set_title(f'BERT Confusion Matrix\n({title_note})', fontsize=13, pad=15)
plt.tight_layout()
plt.savefig('bert_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: bert_confusion_matrix.png")

# ================================================
# VISUAL 3 — MODEL COMPARISON BAR CHART
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
# VISUAL 4 — SBERT FEW-SHOT CURVE
# ================================================
print("Creating SBERT few-shot curve...")

fig, ax = plt.subplots(figsize=(9, 6))

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

ax.set_xlabel('Number of PURE Samples Added to Training', fontsize=12)
ax.set_ylabel('Macro F1 Score', fontsize=12)
ax.set_title('SBERT Few-Shot Domain Adaptation Curve\n(Train: PROMISE → Test: PURE)',
             fontsize=13, pad=15)
ax.set_xticks(samples)
ax.set_ylim(0.1, 0.85)
ax.grid(alpha=0.3, linestyle='--')
ax.legend(fontsize=11, loc='lower right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('fewshot_adaptation_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fewshot_adaptation_curve.png")

# ================================================
# VISUAL 5 — BERT vs SBERT FEW-SHOT CURVE
# ================================================
print("Creating BERT vs SBERT few-shot curve...")

fig, ax = plt.subplots(figsize=(9, 6))

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
print("  confusion_matrix.png         — TF-IDF confusion matrix")
print("  bert_confusion_matrix.png    — BERT confusion matrix")
print("  model_comparison_chart.png   — all 3 models bar chart")
print("  fewshot_adaptation_curve.png — SBERT few-shot curve")
print("  bert_vs_sbert_fewshot.png    — BERT vs SBERT curve")
if not bert_report:
    print("\nNOTE: BERT confusion matrix uses approximate values.")
    print("Rerun bert_classifier.py to save full report, then rerun this file.")
