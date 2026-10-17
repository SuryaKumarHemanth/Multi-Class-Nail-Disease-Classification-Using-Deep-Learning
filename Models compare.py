# ─────────────────────────────────────────
# 10. PLOTS
# ─────────────────────────────────────────
acc      = history_p1.history["accuracy"]     + history_p2.history["accuracy"]
val_acc  = history_p1.history["val_accuracy"] + history_p2.history["val_accuracy"]
loss     = history_p1.history["loss"]         + history_p2.history["loss"]
val_loss = history_p1.history["val_loss"]     + history_p2.history["val_loss"]
phase_boundary = len(history_p1.history["accuracy"])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("MobileNetV2 Improved — Nail Disease Classification",
             fontsize=14, fontweight="bold")

axes[0].plot(acc,     label="Train Acc", color="royalblue", linewidth=2)
axes[0].plot(val_acc, label="Val Acc",   color="tomato",    linewidth=2)
axes[0].axvline(x=phase_boundary, color="gray", linestyle="--",
                alpha=0.7, label="Fine-tune start")
axes[0].set_title("Accuracy over Epochs")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(loss,     label="Train Loss", color="royalblue", linewidth=2)
axes[1].plot(val_loss, label="Val Loss",   color="tomato",    linewidth=2)
axes[1].axvline(x=phase_boundary, color="gray", linestyle="--",
                alpha=0.7, label="Fine-tune start")
axes[1].set_title("Loss over Epochs")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=class_labels, yticklabels=class_labels, ax=axes[2])
axes[2].set_title("Confusion Matrix")
axes[2].set_xlabel("Predicted Label"); axes[2].set_ylabel("True Label")

plt.tight_layout()
plt.savefig("mobilenet_improved_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n📈 Plots saved to: mobilenet_improved_results.png")
print("✅ Done! Run compare_models.py for final comparison chart.")