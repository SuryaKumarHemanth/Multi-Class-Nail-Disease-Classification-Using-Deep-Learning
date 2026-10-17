import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load your trained model
model = load_model("best_mobilenet_improved.keras")  # change name if needed

# Class names
classes = ["Healthy Nail", "Melanoma", "Clubbing", "Pitting"]

st.title("Nail Disease Classification")

uploaded_file = st.file_uploader("Upload Nail Image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.write(f"### Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")