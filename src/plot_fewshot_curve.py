import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

few_shot = [0, 10, 50, 100, 500]
f1_scores = [0.2663, 0.4087, 0.5295, 0.5743, 0.6546]

plt.figure(figsize=(8,6))
plt.plot(few_shot, f1_scores, marker='o')
plt.xlabel("Number of Labeled PURE Samples")
plt.ylabel("Macro F1 on PURE")
plt.title("Few-Shot Domain Adaptation Curve")
plt.grid(True)
plt.savefig("fewshot_adaptation_curve.png")