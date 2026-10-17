"""
Nail Disease Classification - Hybrid Capsule CNN Model
Classes: Acral_Lentiginous_Melanoma, Healthy_Nail, clubbing, pitting
Environment: VS Code / PyCharm (Local)

Architecture:
  CNN Backbone (feature extraction)
  → PrimaryCaps (convert feature maps to capsules)
  → DigitCaps   (dynamic routing between capsules)
  → Classification via capsule length
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

# ─────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────
IMG_SIZE        = (128, 128)
BATCH_SIZE      = 32
EPOCHS          = 50
NUM_CLASSES     = 4
SEED            = 42
ROUTING_ITERS   = 3       # dynamic routing iterations

BASE_DIR  = r"C:\Nail Disease\data"   # <-- SAME AS BEFORE
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR   = os.path.join(BASE_DIR, "validation")
TEST_DIR  = os.path.join(BASE_DIR, "test")

# ─────────────────────────────────────────
# 2. VERIFY PATHS
# ─────────────────────────────────────────
for name, path in [("Train", TRAIN_DIR), ("Val", VAL_DIR), ("Test", TEST_DIR)]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ {name} directory not found: {path}")
    print(f"✅ {name}: {path}  →  {os.listdir(path)}")

# ─────────────────────────────────────────
# 3. DATA LOADING
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
print(f"📊 Train samples : {train_gen.samples}")
print(f"📊 Val samples   : {val_gen.samples}")
print(f"📊 Test samples  : {test_gen.samples}\n")

# ─────────────────────────────────────────
# 4. CAPSULE COMPONENTS
# ─────────────────────────────────────────

def squash(vectors, axis=-1):
    """Squash activation — keeps direction, squashes magnitude to (0,1)."""
    s_squared_norm = K.sum(K.square(vectors), axis=axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors


class PrimaryCapsule(layers.Layer):
    """
    Converts CNN feature maps into primary capsules.
    Each capsule is an 8-D vector representing a feature cluster.
    """
    def __init__(self, num_capsules, dim_capsule, kernel_size=9, strides=2, **kwargs):
        super(PrimaryCapsule, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsule  = dim_capsule
        self.conv = layers.Conv2D(
            filters=num_capsules * dim_capsule,
            kernel_size=kernel_size,
            strides=strides,
            padding="valid",
            activation="relu"
        )

    def call(self, inputs):
        output = self.conv(inputs)
        # Reshape to (batch, total_capsules, dim_capsule)
        batch_size = tf.shape(output)[0]
        h, w, c    = output.shape[1], output.shape[2], output.shape[3]
        total_caps = h * w * (c // self.dim_capsule)
        output = tf.reshape(output, (batch_size, total_caps, self.dim_capsule))
        return squash(output)

    def get_config(self):
        config = super().get_config()
        config.update({"num_capsules": self.num_capsules, "dim_capsule": self.dim_capsule})
        return config


class DigitCapsule(layers.Layer):
    """
    Final capsule layer with dynamic routing.
    Each class gets one 16-D capsule; its length = class probability.
    """
    def __init__(self, num_capsules, dim_capsule, routing_iters=3, **kwargs):
        super(DigitCapsule, self).__init__(**kwargs)
        self.num_capsules  = num_capsules
        self.dim_capsule   = dim_capsule
        self.routing_iters = routing_iters

    def build(self, input_shape):
        self.input_num_caps = input_shape[1]
        self.input_dim_caps = input_shape[2]
        self.W = self.add_weight(
            name="W",
            shape=(1, self.input_num_caps, self.num_capsules,
                   self.dim_capsule, self.input_dim_caps),
            initializer="glorot_uniform",
            trainable=True
        )
        super(DigitCapsule, self).build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        # Expand inputs for broadcasting: (batch, input_caps, 1, input_dim, 1)
        inputs_expand = tf.expand_dims(tf.expand_dims(inputs, 2), 4)
        inputs_tiled  = tf.tile(inputs_expand, [1, 1, self.num_capsules, 1, 1])

        # Prediction vectors û: (batch, input_caps, num_caps, dim_caps, 1)
        W_tiled = tf.tile(self.W, [batch_size, 1, 1, 1, 1])
        u_hat   = tf.squeeze(tf.matmul(W_tiled, inputs_tiled), axis=4)

        # Dynamic routing
        b = tf.zeros([batch_size, self.input_num_caps, self.num_capsules, 1])
        for _ in range(self.routing_iters):
            c     = tf.nn.softmax(b, axis=2)
            s     = tf.reduce_sum(c * u_hat, axis=1, keepdims=True)
            v     = squash(s, axis=-1)
            b    += tf.reduce_sum(u_hat * tf.tile(v, [1, self.input_num_caps, 1, 1]), axis=-1, keepdims=True)

        return tf.squeeze(v, axis=1)   # (batch, num_caps, dim_caps)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_capsules": self.num_capsules,
            "dim_capsule": self.dim_capsule,
            "routing_iters": self.routing_iters
        })
        return config


class CapsuleLength(layers.Layer):
    """Computes length of each capsule vector → class probability."""
    def call(self, inputs):
        return tf.sqrt(tf.reduce_sum(tf.square(inputs), axis=-1))

    def get_config(self):
        return super().get_config()


# ─────────────────────────────────────────
# 5. HYBRID CAPSULE CNN MODEL
# ─────────────────────────────────────────
def build_hybrid_capsule_cnn(input_shape=(128, 128, 3), num_classes=4):
    inputs = layers.Input(shape=input_shape)

    # ── CNN Backbone ──────────────────────
    x = layers.Conv2D(32, (3,3), activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.30)(x)

    # ── Primary Capsules ──────────────────
    # Converts CNN feature maps → capsule vectors
    primary_caps = PrimaryCapsule(
        num_capsules=8,
        dim_capsule=8,
        kernel_size=3,
        strides=1,
        name="primary_capsules"
    )(x)

    # ── Digit Capsules (with routing) ─────
    digit_caps = DigitCapsule(
        num_capsules=num_classes,
        dim_capsule=16,
        routing_iters=ROUTING_ITERS,
        name="digit_capsules"
    )(primary_caps)

    # ── Output: capsule length = class prob ─
    outputs = CapsuleLength(name="capsule_length")(digit_caps)

    model = models.Model(inputs=inputs, outputs=outputs, name="Hybrid_Capsule_CNN")
    return model


capsule_model = build_hybrid_capsule_cnn(input_shape=(*IMG_SIZE, 3), num_classes=NUM_CLASSES)
capsule_model.summary()

# ─────────────────────────────────────────
# 6. CUSTOM MARGIN LOSS
# ─────────────────────────────────────────
def margin_loss(y_true, y_pred):
    """
    Capsule margin loss — penalizes both false positives and false negatives.
    Better than cross-entropy for capsule networks.
    """
    L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))
    return tf.reduce_mean(tf.reduce_sum(L, axis=1))

# ─────────────────────────────────────────
# 7. COMPILE
# ─────────────────────────────────────────
capsule_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=margin_loss,
    metrics=["accuracy"]
)

# ─────────────────────────────────────────
# 8. CALLBACKS
# ─────────────────────────────────────────
callbacks = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=12,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=6,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        filepath="best_capsule_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
]

# ─────────────────────────────────────────
# 9. TRAIN
# ─────────────────────────────────────────
print("\n🚀 Starting Hybrid Capsule CNN Training...\n")
history = capsule_model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=1
)

# ─────────────────────────────────────────
# 10. EVALUATION
# ─────────────────────────────────────────
test_loss, test_acc = capsule_model.evaluate(test_gen, verbose=0)
print(f"\n{'='*45}")
print(f"  ✅ Capsule CNN Test Accuracy : {test_acc * 100:.2f}%")
print(f"  📉 Capsule CNN Test Loss     : {test_loss:.4f}")
print(f"{'='*45}\n")

# Save results for final comparison
capsule_results = {
    "model": "Hybrid Capsule CNN",
    "test_accuracy": round(test_acc * 100, 2),
    "test_loss": round(test_loss, 4)
}
with open("capsule_results.json", "w") as f:
    json.dump(capsule_results, f, indent=2)
print("💾 Results saved to capsule_results.json")

# ─────────────────────────────────────────
# 11. CLASSIFICATION REPORT
# ─────────────────────────────────────────
test_gen.reset()
y_pred_probs = capsule_model.predict(test_gen, verbose=0)
y_pred       = np.argmax(y_pred_probs, axis=1)
y_true       = test_gen.classes
class_labels = list(test_gen.class_indices.keys())

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# ─────────────────────────────────────────
# 12. PLOTS
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Hybrid Capsule CNN — Nail Disease Classification", fontsize=14, fontweight="bold")

axes[0].plot(history.history["accuracy"],     label="Train Acc", color="royalblue", linewidth=2)
axes[0].plot(history.history["val_accuracy"], label="Val Acc",   color="tomato",    linewidth=2)
axes[0].set_title("Accuracy over Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history["loss"],     label="Train Loss", color="royalblue", linewidth=2)
axes[1].plot(history.history["val_loss"], label="Val Loss",   color="tomato",    linewidth=2)
axes[1].set_title("Loss over Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Purples",
    xticklabels=class_labels,
    yticklabels=class_labels,
    ax=axes[2]
)
axes[2].set_title("Confusion Matrix")
axes[2].set_xlabel("Predicted Label")
axes[2].set_ylabel("True Label")

plt.tight_layout()
plt.savefig("capsule_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n📈 Plots saved to: capsule_results.png")
print("✅ Capsule CNN training complete! Run MobileNetV2 next.")