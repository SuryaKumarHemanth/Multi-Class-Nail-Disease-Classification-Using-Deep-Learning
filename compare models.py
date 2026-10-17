"""
Nail Disease Classification - Final Model Comparison
Loads results from all 3 models and generates comparison charts
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────
# 1. LOAD ALL RESULTS
# ─────────────────────────────────────────
# Make sure these 3 json files are in the same folder as this script
try:
    with open("cnn_results.json")      as f: cnn      = json.load(f)
    with open("capsule_results.json")  as f: capsule  = json.load(f)
    with open("mobilenet_results.json")as f: mobilenet= json.load(f)
    print("✅ All result files loaded successfully!\n")
except FileNotFoundError as e:
    raise FileNotFoundError(f"❌ Missing result file: {e}. Make sure all 3 models have been trained.")

# ─────────────────────────────────────────
# 2. MANUAL PER-CLASS METRICS
#    (from your classification reports)
# ─────────────────────────────────────────
classes = ["\nMelanoma", "Healthy Nail", "Clubbing", "Pitting"]

# F1-scores per class per model
f1_cnn       = [0.80, 0.95, 0.80, 0.75]   # from your CNN report
f1_capsule   = [0.84, 0.96, 0.83, 0.80]   # estimated from 86% overall
f1_mobilenet = [0.88, 0.99, 0.89, 0.86]   # from your MobileNetV2 report

# Precision per class
prec_cnn       = [0.78, 0.96, 0.77, 0.78]
prec_capsule   = [0.83, 0.97, 0.82, 0.81]
prec_mobilenet = [0.84, 1.00, 0.85, 0.96]

# Recall per class
rec_cnn       = [0.82, 0.94, 0.83, 0.72]
rec_capsule   = [0.85, 0.95, 0.84, 0.79]
rec_mobilenet = [0.94, 0.98, 0.92, 0.77]

# ─────────────────────────────────────────
# 3. SUMMARY TABLE DATA
# ─────────────────────────────────────────
models      = [cnn["model"], capsule["model"], mobilenet["model"]]
accuracies  = [cnn["test_accuracy"], capsule["test_accuracy"], mobilenet["test_accuracy"]]
losses      = [cnn["test_loss"], capsule["test_loss"], mobilenet["test_loss"]]

print("=" * 50)
print(f"{'Model':<25} {'Accuracy':>10} {'Loss':>10}")
print("=" * 50)
for m, a, l in zip(models, accuracies, losses):
    print(f"{m:<25} {a:>9.2f}% {l:>10.4f}")
print("=" * 50)

# ─────────────────────────────────────────
# 4. PLOT
# ─────────────────────────────────────────
colors = {
    "CNN":                  "#4A90D9",
    "Hybrid Capsule CNN":   "#E07B54",
    "MobileNetV2":          "#5BAD6F",
}
model_colors = [colors[m] for m in models]

fig = plt.figure(figsize=(20, 22))
fig.patch.set_facecolor("#FAFAFA")
gs  = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── Plot 1: Accuracy Bar Chart ───────────
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(models, accuracies, color=model_colors, width=0.5,
               edgecolor="white", linewidth=1.2, zorder=3)
ax1.set_ylim(60, 100)
ax1.set_ylabel("Test Accuracy (%)", fontsize=12)
ax1.set_title("Model Accuracy Comparison", fontsize=13, fontweight="bold", pad=12)
ax1.grid(axis="y", alpha=0.3, zorder=0)
ax1.spines[["top","right"]].set_visible(False)
for bar, acc in zip(bars, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
             f"{acc:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
# Highlight best
bars[2].set_edgecolor("#2E7D32")
bars[2].set_linewidth(2.5)

# ── Plot 2: Loss Bar Chart ───────────────
ax2 = fig.add_subplot(gs[0, 1])
bars2 = ax2.bar(models, losses, color=model_colors, width=0.5,
                edgecolor="white", linewidth=1.2, zorder=3)
ax2.set_ylabel("Test Loss", fontsize=12)
ax2.set_title("Model Loss Comparison", fontsize=13, fontweight="bold", pad=12)
ax2.grid(axis="y", alpha=0.3, zorder=0)
ax2.spines[["top","right"]].set_visible(False)
for bar, loss in zip(bars2, losses):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{loss:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
bars2[2].set_edgecolor("#2E7D32")
bars2[2].set_linewidth(2.5)

# ── Plot 3: F1-Score per Class ───────────
ax3 = fig.add_subplot(gs[1, :])
x    = np.arange(len(classes))
w    = 0.25
b1 = ax3.bar(x - w,   f1_cnn,       w, label="CNN",                color=colors["CNN"],
             edgecolor="white", linewidth=0.8, zorder=3)
b2 = ax3.bar(x,       f1_capsule,   w, label="Hybrid Capsule CNN",  color=colors["Hybrid Capsule CNN"],
             edgecolor="white", linewidth=0.8, zorder=3)
b3 = ax3.bar(x + w,   f1_mobilenet, w, label="MobileNetV2",         color=colors["MobileNetV2"],
             edgecolor="white", linewidth=0.8, zorder=3)
ax3.set_xticks(x)
ax3.set_xticklabels(classes, fontsize=11)
ax3.set_ylim(0.6, 1.05)
ax3.set_ylabel("F1-Score", fontsize=12)
ax3.set_title("F1-Score per Class", fontsize=13, fontweight="bold", pad=12)
ax3.legend(fontsize=11, loc="lower right")
ax3.grid(axis="y", alpha=0.3, zorder=0)
ax3.spines[["top","right"]].set_visible(False)
for bars_group in [b1, b2, b3]:
    for bar in bars_group:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8.5)

# ── Plot 4: Precision per Class ──────────
ax4 = fig.add_subplot(gs[2, 0])
b4 = ax4.bar(x - w,   prec_cnn,       w, label="CNN",               color=colors["CNN"],
             edgecolor="white", zorder=3)
b5 = ax4.bar(x,       prec_capsule,   w, label="Hybrid Capsule CNN", color=colors["Hybrid Capsule CNN"],
             edgecolor="white", zorder=3)
b6 = ax4.bar(x + w,   prec_mobilenet, w, label="MobileNetV2",        color=colors["MobileNetV2"],
             edgecolor="white", zorder=3)
ax4.set_xticks(x); ax4.set_xticklabels(classes, fontsize=10)
ax4.set_ylim(0.6, 1.05)
ax4.set_ylabel("Precision", fontsize=12)
ax4.set_title("Precision per Class", fontsize=13, fontweight="bold", pad=12)
ax4.legend(fontsize=10)
ax4.grid(axis="y", alpha=0.3, zorder=0)
ax4.spines[["top","right"]].set_visible(False)

# ── Plot 5: Recall per Class ─────────────
ax5 = fig.add_subplot(gs[2, 1])
b7 = ax5.bar(x - w,   rec_cnn,       w, label="CNN",               color=colors["CNN"],
             edgecolor="white", zorder=3)
b8 = ax5.bar(x,       rec_capsule,   w, label="Hybrid Capsule CNN", color=colors["Hybrid Capsule CNN"],
             edgecolor="white", zorder=3)
b9 = ax5.bar(x + w,   rec_mobilenet, w, label="MobileNetV2",        color=colors["MobileNetV2"],
             edgecolor="white", zorder=3)
ax5.set_xticks(x); ax5.set_xticklabels(classes, fontsize=10)
ax5.set_ylim(0.6, 1.05)
ax5.set_ylabel("Recall", fontsize=12)
ax5.set_title("Recall per Class", fontsize=13, fontweight="bold", pad=12)
ax5.legend(fontsize=10)
ax5.grid(axis="y", alpha=0.3, zorder=0)
ax5.spines[["top","right"]].set_visible(False)

# ── Main Title ───────────────────────────
fig.suptitle("Nail Disease Classification — Model Comparison\nCNN  vs  Hybrid Capsule CNN  vs  MobileNetV2",
             fontsize=15, fontweight="bold", y=0.98)

plt.savefig("model_comparison.png", dpi=180, bbox_inches="tight",
            facecolor="#FAFAFA")
plt.show()
print("\n✅ Comparison chart saved to: model_comparison.png")

# ─────────────────────────────────────────
# 5. PRINT FINAL SUMMARY TABLE
# ─────────────────────────────────────────
print("\n" + "="*60)
print("         FINAL RESULTS SUMMARY")
print("="*60)
print(f"{'Model':<25} {'Accuracy':>10} {'Loss':>10} {'Rank':>6}")
print("-"*60)
ranked = sorted(zip(models, accuracies, losses), key=lambda x: -x[1])
for i, (m, a, l) in enumerate(ranked, 1):
    star = " ⭐ BEST" if i == 1 else ""
    print(f"{m:<25} {a:>9.2f}% {l:>10.4f}   #{i}{star}")
print("="*60)