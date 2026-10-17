"""
Nail Disease Classification - Flask Web App
Fix: Conv2D inside PrimaryCapsule moved to build() so weights load correctly
"""

from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import keras
import numpy as np
import os
import io
import base64
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras import layers, backend as K

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024


# ─────────────────────────────────────────
# CAPSULE HELPER
# ─────────────────────────────────────────
def squash(vectors, axis=-1):
    s_squared_norm = K.sum(K.square(vectors), axis=axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors


# ─────────────────────────────────────────
# CUSTOM LAYERS
# ─────────────────────────────────────────

@keras.saving.register_keras_serializable()
class PrimaryCapsule(layers.Layer):
    def __init__(self, num_capsules, dim_capsule, kernel_size=9, strides=2, **kwargs):
        super().__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsule  = dim_capsule
        self.kernel_size  = kernel_size
        self.strides      = strides
        # ✅ DO NOT create Conv2D here — create it in build()

    def build(self, input_shape):
        # ✅ Conv2D created inside build() so its weights
        #    are properly registered and restored on load
        self.conv = layers.Conv2D(
            filters=self.num_capsules * self.dim_capsule,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding="valid",
            activation="relu",
            name="primary_conv"
        )
        # Build the child layer explicitly with the expected input shape
        self.conv.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        output     = self.conv(inputs)
        batch_size = tf.shape(output)[0]
        h = output.shape[1]
        w = output.shape[2]
        c = output.shape[3]
        total_caps = h * w * (c // self.dim_capsule)
        output = tf.reshape(output, (batch_size, total_caps, self.dim_capsule))
        return squash(output)

    def compute_output_shape(self, input_shape):
        h = (input_shape[1] - self.kernel_size) // self.strides + 1
        w = (input_shape[2] - self.kernel_size) // self.strides + 1
        total_caps = h * w * self.num_capsules
        return (input_shape[0], total_caps, self.dim_capsule)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "num_capsules": self.num_capsules,
            "dim_capsule":  self.dim_capsule,
            "kernel_size":  self.kernel_size,
            "strides":      self.strides,
        })
        return cfg


@keras.saving.register_keras_serializable()
class DigitCapsule(layers.Layer):
    def __init__(self, num_capsules, dim_capsule, routing_iters=3, **kwargs):
        super().__init__(**kwargs)
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
        super().build(input_shape)

    def call(self, inputs):
        batch_size    = tf.shape(inputs)[0]
        inputs_expand = tf.expand_dims(tf.expand_dims(inputs, 2), 4)
        inputs_tiled  = tf.tile(inputs_expand, [1, 1, self.num_capsules, 1, 1])
        W_tiled       = tf.tile(self.W, [batch_size, 1, 1, 1, 1])
        u_hat         = tf.squeeze(tf.matmul(W_tiled, inputs_tiled), axis=4)
        b = tf.zeros([batch_size, self.input_num_caps, self.num_capsules, 1])
        for _ in range(self.routing_iters):
            c  = tf.nn.softmax(b, axis=2)
            s  = tf.reduce_sum(c * u_hat, axis=1, keepdims=True)
            v  = squash(s, axis=-1)
            b += tf.reduce_sum(
                u_hat * tf.tile(v, [1, self.input_num_caps, 1, 1]),
                axis=-1, keepdims=True
            )
        return tf.squeeze(v, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_capsules, self.dim_capsule)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "num_capsules":  self.num_capsules,
            "dim_capsule":   self.dim_capsule,
            "routing_iters": self.routing_iters,
        })
        return cfg


@keras.saving.register_keras_serializable()
class CapsuleLength(layers.Layer):
    def call(self, inputs):
        return tf.sqrt(tf.reduce_sum(tf.square(inputs), axis=-1))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])

    def get_config(self):
        return super().get_config()


CUSTOM_OBJECTS = {
    "PrimaryCapsule": PrimaryCapsule,
    "DigitCapsule":   DigitCapsule,
    "CapsuleLength":  CapsuleLength,
}

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
CLASS_NAMES = [
    "Acral Lentiginous Melanoma",
    "Healthy Nail",
    "Clubbing",
    "Pitting"
]

CLASS_INFO = {
    "Acral Lentiginous Melanoma": {
        "color":  "#E74C3C",
        "advice": "This may indicate a serious condition. Please consult a dermatologist immediately.",
        "icon":   "⚠️"
    },
    "Healthy Nail": {
        "color":  "#27AE60",
        "advice": "Your nail appears healthy. Maintain good nail hygiene.",
        "icon":   "✅"
    },
    "Clubbing": {
        "color":  "#E67E22",
        "advice": "Nail clubbing may indicate respiratory or cardiovascular issues. Consult a doctor.",
        "icon":   "🔶"
    },
    "Pitting": {
        "color":  "#8E44AD",
        "advice": "Nail pitting is often associated with psoriasis. Consult a dermatologist.",
        "icon":   "🔷"
    }
}

MODEL_CONFIG = {
    "cnn": {
        "path":       "best_cnn_model.h5",
        "name":       "CNN (Baseline)",
        "img_size":   (128, 128),
        "preprocess": "rescale",
        "accuracy":   "75.00%",
        "custom":     False
    },
    "capsule": {
        "path":       "best_capsule_model.keras",
        "name":       "Hybrid Capsule CNN",
        "img_size":   (128, 128),
        "preprocess": "rescale",
        "accuracy":   "86.00%",
        "custom":     True
    },
    "mobilenet": {
        "path":       "best_mobilenet_improved.keras",
        "name":       "MobileNetV2 (Best)",
        "img_size":   (160, 160),
        "preprocess": "mobilenet",
        "accuracy":   "90.20%",
        "custom":     False
    }
}

loaded_models = {}


def load_model(model_key):
    if model_key not in loaded_models:
        config = MODEL_CONFIG[model_key]
        path   = config["path"]

        if not os.path.exists(path):
            return None, f"Model file '{path}' not found."

        try:
            if config["custom"]:
                model = tf.keras.models.load_model(
                    path,
                    custom_objects=CUSTOM_OBJECTS,
                    compile=False
                )
            else:
                model = tf.keras.models.load_model(path, compile=False)

            loaded_models[model_key] = model
            print(f"✅ Loaded: {config['name']}")

        except Exception as e:
            return None, f"Error loading model: {str(e)}"

    return loaded_models[model_key], None


def preprocess_image(image_bytes, img_size, preprocess_type):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(img_size)
    arr = np.array(img, dtype=np.float32)
    if preprocess_type == "rescale":
        arr = arr / 255.0
    elif preprocess_type == "mobilenet":
        arr = mobilenet_preprocess(arr)
    return np.expand_dims(arr, axis=0)


# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", models=MODEL_CONFIG)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    model_key = request.form.get("model", "mobilenet")
    if model_key not in MODEL_CONFIG:
        return jsonify({"error": "Invalid model selection."}), 400

    image_bytes = file.read()
    allowed = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
    if file.content_type not in allowed:
        return jsonify({"error": "Please upload a JPG or PNG image."}), 400

    model, err = load_model(model_key)
    if err:
        return jsonify({"error": err}), 500

    config = MODEL_CONFIG[model_key]
    try:
        img_array = preprocess_image(
            image_bytes, config["img_size"], config["preprocess"]
        )
    except Exception as e:
        return jsonify({"error": f"Could not process image: {str(e)}"}), 400

    preds      = model.predict(img_array, verbose=0)[0]
    pred_idx   = int(np.argmax(preds))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(preds[pred_idx]) * 100

    class_scores = [
        {
            "class": CLASS_NAMES[i],
            "score": round(float(preds[i]) * 100, 2),
            "color": CLASS_INFO[CLASS_NAMES[i]]["color"]
        }
        for i in range(len(CLASS_NAMES))
    ]
    class_scores.sort(key=lambda x: x["score"], reverse=True)

    img_b64  = base64.b64encode(image_bytes).decode("utf-8")
    img_mime = file.content_type

    return jsonify({
        "predicted_class": pred_class,
        "confidence":      round(confidence, 2),
        "advice":          CLASS_INFO[pred_class]["advice"],
        "color":           CLASS_INFO[pred_class]["color"],
        "icon":            CLASS_INFO[pred_class]["icon"],
        "model_used":      config["name"],
        "model_accuracy":  config["accuracy"],
        "class_scores":    class_scores,
        "image_b64":       img_b64,
        "image_mime":      img_mime,
    })


if __name__ == "__main__":
    print("\n🚀 Nail Disease Classification Web App")
    print("=" * 45)
    for key, cfg in MODEL_CONFIG.items():
        status = "✅" if os.path.exists(cfg["path"]) else "❌ MISSING"
        print(f"  {status}  {cfg['name']} ({cfg['path']})")
    print("=" * 45)
    print("  Open: http://127.0.0.1:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)