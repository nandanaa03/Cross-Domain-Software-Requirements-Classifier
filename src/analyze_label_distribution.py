import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json

# Load PROMISE
promise_df = pd.read_csv("data/cleaned_requirements.csv")
promise_dist = promise_df["label"].value_counts(normalize=True)
promise_counts = promise_df["label"].value_counts()

print("PROMISE Label Distribution:")
print(promise_dist)
print("\n")

# Load PURE
pure_df = pd.read_csv(
    r"data/pure/Pure_Annotate_dataset.csv",
    encoding="latin1"
)

pure_df["label"] = pure_df["NFR_boolean"].apply(
    lambda x: "non-functional" if x == 1 else "functional"
)

pure_dist = pure_df["label"].value_counts(normalize=True)
pure_counts = pure_df["label"].value_counts()

print("PURE Label Distribution:")
print(pure_dist)

# ================================================
# LABEL DISTRIBUTION CHART
# ================================================

promise_fr = promise_dist.get('functional', 0) * 100
promise_nfr = promise_dist.get('non-functional', 0) * 100

pure_fr = pure_dist.get('functional', 0) * 100
pure_nfr = pure_dist.get('non-functional', 0) * 100

fig, axes = plt.subplots(1, 2, figsize=(10, 6))

colors_promise = ['#1976d2', '#d32f2f']
colors_pure = ['#1976d2', '#d32f2f']

# PROMISE pie
axes[0].pie(
    [promise_fr, promise_nfr],
    labels=['Functional\n({:.1f}%)'.format(promise_fr),
            'Non-Functional\n({:.1f}%)'.format(promise_nfr)],
    colors=colors_promise,
    autopct='',
    startangle=90,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)
axes[0].set_title('PROMISE Dataset\n(626 requirements)', fontsize=13, pad=15)

# PURE pie
axes[1].pie(
    [pure_fr, pure_nfr],
    labels=['Functional\n({:.1f}%)'.format(pure_fr),
            'Non-Functional\n({:.1f}%)'.format(pure_nfr)],
    colors=colors_pure,
    autopct='',
    startangle=90,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)
axes[1].set_title('PURE Dataset\n(11,440 requirements)', fontsize=13, pad=15)

fig.suptitle('Label Distribution Comparison\nPROMISE vs PURE',
             fontsize=14, fontweight='bold', y=1.02)

fig.text(0.5, 0.02,
         'Majority class completely reverses between datasets — worst-case domain shift',
         ha='center', fontsize=10, color='#b71c1c', style='italic')

plt.tight_layout()
plt.savefig('label_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: label_distribution.png")

# ================================================
# SAVE DISTRIBUTION NUMBERS TO RESULTS
# ================================================
os.makedirs("results", exist_ok=True)

distribution_data = {
    "PROMISE": {
        "total": int(promise_counts.sum()),
        "functional": int(promise_counts.get('functional', 0)),
        "non_functional": int(promise_counts.get('non-functional', 0)),
        "functional_pct": round(promise_fr, 1),
        "non_functional_pct": round(promise_nfr, 1)
    },
    "PURE": {
        "total": int(pure_counts.sum()),
        "functional": int(pure_counts.get('functional', 0)),
        "non_functional": int(pure_counts.get('non-functional', 0)),
        "functional_pct": round(pure_fr, 1),
        "non_functional_pct": round(pure_nfr, 1)
    }
}

with open("results/label_distribution.json", "w") as f:
    json.dump(distribution_data, f, indent=4)

print("Distribution saved to results/label_distribution.json")
print("\nSummary:")
print(f"PROMISE — FR: {distribution_data['PROMISE']['functional']} ({distribution_data['PROMISE']['functional_pct']}%) | NFR: {distribution_data['PROMISE']['non_functional']} ({distribution_data['PROMISE']['non_functional_pct']}%)")
print(f"PURE    — FR: {distribution_data['PURE']['functional']} ({distribution_data['PURE']['functional_pct']}%) | NFR: {distribution_data['PURE']['non_functional']} ({distribution_data['PURE']['non_functional_pct']}%)")
