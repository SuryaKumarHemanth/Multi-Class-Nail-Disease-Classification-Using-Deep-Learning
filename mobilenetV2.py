"""
Nail Disease Classification - MobileNetV2 IMPROVED
Fixes: class weights, deeper unfreeze, better augmentation, LR warmup
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ─────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────
IMG_SIZE    = (160, 160)   # ✅ Increased from 128 → 160 (more detail)
BATCH_SIZE  = 32
EPOCHS_P1   = 25           # ✅ More epochs for feature extraction
EPOCHS_P2   = 40           # ✅ More epochs for fine-tuning
NUM_CLASSES = 4
SEED        = 42

BASE_DIR  = r"C:\Nail Disease\data"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR   = os.path.join(BASE_DIR, "validation")
TEST_DIR  = os.path.join(BASE_DIR, "test")

# ─────────────────────────────────────────
# 2. VERIFY PATHS
# ─────────────────────────────────────────
for name, path in [("Train", TRAIN_DIR), ("Val", VAL_DIR), ("Test", TEST_DIR)]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ {name} not found: {path}")
    print(f"✅ {name}: {os.listdir(path)}")

# ─────────────────────────────────────────
# 3. DATA LOADING
# ✅ FIX 1: Stronger augmentation for pitting & clubbing
# ─────────────────────────────────────────
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,           # was 20
    width_shift_range=0.20,      # was 0.15
    height_shift_range=0.20,     # was 0.15
    shear_range=0.15,            # was 0.10
    zoom_range=0.25,             # was 0.20
    horizontal_flip=True,
    vertical_flip=True,          # ✅ NEW — nail images are rotation invariant
    brightness_range=[0.7, 1.3], # was [0.8, 1.2]
    channel_shift_range=20.0,    # ✅ NEW — slight color variation
    fill_mode="nearest"
)
val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", seed=SEED, shuffle=True
)
val_gen = val_test_datagen.flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", seed=SEED, shuffle=False
)
test_gen = val_test_datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", seed=SEED, shuffle=False
)

print(f"\n📂 Class indices : {train_gen.class_indices}")
print(f"📊 Train: {train_gen.samples} | Val: {val_gen.samples} | Test: {test_gen.samples}\n")

# ─────────────────────────────────────────
# 4. CLASS WEIGHTS
# ✅ FIX 2: Penalize misclassification of pitting more
# ─────────────────────────────────────────
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights = dict(enumerate(class_weights_array))
print(f"⚖️  Class weights: {class_weights}")
# Pitting will get higher weight automatically if it has fewer samples

# ─────────────────────────────────────────
# 5. BUILD IMPROVED MOBILENETV2
# ✅ FIX 3: Deeper & stronger classification head
# ─────────────────────────────────────────
def build_improved_mobilenetv2(input_shape=(160, 160, 3), num_classes=4):
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)

    # ✅ Improved head: added extra dense + L2 regularization
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_Improved")
    return model, base_model

mobilenet_model, base_model = build_improved_mobilenetv2(
    input_shape=(*IMG_SIZE, 3), num_classes=NUM_CLASSES
)
mobilenet_model.summary()

# ─────────────────────────────────────────
# 6. PHASE 1 — FEATURE EXTRACTION
# ─────────────────────────────────────────
print("\n" + "="*50)
print("  PHASE 1: Feature Extraction (Base Frozen)")
print("="*50 + "\n")

mobilenet_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_p1 = [
    EarlyStopping(monitor="val_accuracy", patience=8,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                     patience=4, min_lr=1e-6, verbose=1),
    ModelCheckpoint(filepath="best_mobilenet_improved.keras",
                    monitor="val_accuracy", save_best_only=True, verbose=1)
]

history_p1 = mobilenet_model.fit(
    train_gen,
    epochs=EPOCHS_P1,
    validation_data=val_gen,
    class_weight=class_weights,   # ✅ Class weights applied
    callbacks=callbacks_p1,
    verbose=1
)

# ─────────────────────────────────────────
# 7. PHASE 2 — FINE-TUNING
# ✅ FIX 4: Unfreeze top 50 layers (was 30)
# ─────────────────────────────────────────
print("\n" + "="*50)
print("  PHASE 2: Fine-Tuning (Top 50 layers unfrozen)")
print("="*50 + "\n")

base_model.trainable = True
for layer in base_model.layers[:-50]:   # unfreeze top 50 (was 30)
    layer.trainable = False

unfrozen = sum(1 for l in base_model.layers if l.trainable)
print(f"🔓 Unfrozen layers: {unfrozen} / {len(base_model.layers)}")

mobilenet_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-6),  # even lower than before
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_p2 = [
    EarlyStopping(monitor="val_accuracy", patience=12,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                     patience=6, min_lr=1e-8, verbose=1),
    ModelCheckpoint(filepath="best_mobilenet_improved.keras",
                    monitor="val_accuracy", save_best_only=True, verbose=1)
]

history_p2 = mobilenet_model.fit(
    train_gen,
    epochs=EPOCHS_P2,
    validation_data=val_gen,
    class_weight=class_weights,   # ✅ Class weights applied in fine-tuning too
    callbacks=callbacks_p2,
    verbose=1
)

# ─────────────────────────────────────────
# 8. EVALUATION
# ─────────────────────────────────────────
test_loss, test_acc = mobilenet_model.evaluate(test_gen, verbose=0)
print(f"\n{'='*45}")
print(f"  ✅ MobileNetV2 (Improved) Test Accuracy : {test_acc * 100:.2f}%")
print(f"  📉 MobileNetV2 (Improved) Test Loss     : {test_loss:.4f}")
print(f"{'='*45}\n")

mobilenet_results = {
    "model": "MobileNetV2",
    "test_accuracy": round(test_acc * 100, 2),
    "test_loss": round(test_loss, 4)
}
with open("mobilenet_results.json", "w") as f:
    json.dump(mobilenet_results, f, indent=2)
print("💾 Results saved to mobilenet_results.json (overwrites previous)")

# ─────────────────────────────────────────
# 9. CLASSIFICATION REPORT
# ─────────────────────────────────────────
test_gen.reset()
y_pred_probs = mobilenet_model.predict(test_gen, verbose=0)
y_pred       = np.argmax(y_pred_probs, axis=1)
y_true       = test_gen.classes
class_labels = list(test_gen.class_indices.keys())

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

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