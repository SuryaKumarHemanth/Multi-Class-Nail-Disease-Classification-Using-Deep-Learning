from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
# ================= LOAD MODEL =================
model = tf.keras.models.load_model("best_capsule_model.keras")

# ================= PREDICTIONS =================
y_pred = model.predict(test_gen)
y_pred_classes = np.argmax(y_pred, axis=1)

y_true = test_gen.classes
class_names = list(test_gen.class_indices.keys())

# ================= REPORT =================
report = classification_report(
    y_true,
    y_pred_classes,
    target_names=class_names,
    output_dict=True
)

# ================= CONVERT TO TABLE =================
df = pd.DataFrame(report).transpose()

# Keep only class rows
df = df.iloc[:4, :3]  # precision, recall, f1-score

# Rename columns
df.columns = ["Precision", "Recall", "F1-Score"]

print("\n📊 Class-wise Performance Table:\n")
print(df)