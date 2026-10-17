"""
Nail Disease Classification - CNN Model
Classes: Clubbing, Pitting, Healthy, Melanoma
Environment: VS Code / PyCharm (Local)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

# ─────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────
IMG_SIZE    = (128, 128)
BATCH_SIZE  = 32
EPOCHS      = 50
NUM_CLASSES = 4
SEED        = 42

# ✅ CHANGE THIS to your actual path
# Windows example : r"C:\Users\YourName\Desktop\Nail Disease\data"
# Mac/Linux example: "/home/yourname/projects/Nail Disease/data"
BASE_DIR  = r"C:\Nail Disease\data"   # <-- UPDATE THIS

TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR   = os.path.join(BASE_DIR, "validation")
TEST_DIR  = os.path.join(BASE_DIR, "test")

# ─────────────────────────────────────────
# 2. VERIFY PATHS EXIST
# ─────────────────────────────────────────
for name, path in [("Train", TRAIN_DIR), ("Val", VAL_DIR), ("Test", TEST_DIR)]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ {name} directory not found: {path}")
    classes = os.listdir(path)
    print(f"✅ {name}: {path}")
    print(f"   Classes found: {classes}\n")

# ─────────────────────────────────────────
# 3. DATA AUGMENTATION & LOADING
# ─────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest"
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    seed=SEED,
    shuffle=True
)

val_gen = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    seed=SEED,
    shuffle=False
)

test_gen = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    seed=SEED,
    shuffle=False
)

print("📂 Class indices:", train_gen.class_indices)
print(f"📊 Train samples : {train_gen.samples}")
print(f"📊 Val samples   : {val_gen.samples}")
print(f"📊 Test samples  : {test_gen.samples}\n")

# ─────────────────────────────────────────
# 4. CNN MODEL ARCHITECTURE
# ─────────────────────────────────────────
def build_cnn(input_shape=(128, 128, 3), num_classes=4):
    model = models.Sequential([
        # Block 1 — 32 filters
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2 — 64 filters
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3 — 128 filters
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.30),

        # Block 4 — 256 filters
        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.30),

        # Classifier Head
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax")
    ], name="Nail_CNN")

    return model

cnn_model = build_cnn(input_shape=(*IMG_SIZE, 3), num_classes=NUM_CLASSES)
cnn_model.summary()

# ─────────────────────────────────────────
# 5. COMPILE
# ─────────────────────────────────────────
cnn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ─────────────────────────────────────────
# 6. CALLBACKS
# ─────────────────────────────────────────
callbacks = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        filepath="best_cnn_model.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
]

# ─────────────────────────────────────────
# 7. TRAIN
# ─────────────────────────────────────────
print("\n🚀 Starting CNN Training...\n")
history = cnn_model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=1
)

# ─────────────────────────────────────────
# 8. EVALUATION ON TEST SET
# ─────────────────────────────────────────
test_loss, test_acc = cnn_model.evaluate(test_gen, verbose=0)
print(f"\n{'='*45}")
print(f"  ✅ CNN Test Accuracy : {test_acc * 100:.2f}%")
print(f"  📉 CNN Test Loss     : {test_loss:.4f}")
print(f"{'='*45}\n")

# Save results for final 3-model comparison chart
cnn_results = {
    "model": "CNN",
    "test_accuracy": round(test_acc * 100, 2),
    "test_loss": round(test_loss, 4)
}
with open("cnn_results.json", "w") as f:
    json.dump(cnn_results, f, indent=2)
print("💾 Results saved to cnn_results.json  (needed for final comparison!)")

# ─────────────────────────────────────────
# 9. CLASSIFICATION REPORT
# ─────────────────────────────────────────
test_gen.reset()
y_pred_probs = cnn_model.predict(test_gen, verbose=0)
y_pred       = np.argmax(y_pred_probs, axis=1)
y_true       = test_gen.classes
class_labels = list(test_gen.class_indices.keys())

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# ─────────────────────────────────────────
# 10. PLOTS — Accuracy, Loss, Confusion Matrix
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("CNN — Nail Disease Classification", fontsize=14, fontweight="bold")

# Accuracy curve
axes[0].plot(history.history["accuracy"],     label="Train Acc", color="royalblue", linewidth=2)
axes[0].plot(history.history["val_accuracy"], label="Val Acc",   color="tomato",    linewidth=2)
axes[0].set_title("Accuracy over Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss curve
axes[1].plot(history.history["loss"],     label="Train Loss", color="royalblue", linewidth=2)
axes[1].plot(history.history["val_loss"], label="Val Loss",   color="tomato",    linewidth=2)
axes[1].set_title("Loss over Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=class_labels,
    yticklabels=class_labels,
    ax=axes[2]
)
axes[2].set_title("Confusion Matrix")
axes[2].set_xlabel("Predicted Label")
axes[2].set_ylabel("True Label")

plt.tight_layout()
plt.savefig("cnn_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n📈 Plots saved to: cnn_results.png")
print("✅ CNN training complete! Run Capsule CNN next.")