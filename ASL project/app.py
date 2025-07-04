import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# --- CONFIG ---
IMG_SIZE = 64
classes = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'SPACE', 'DELETE', 'NOTHING'
]

# --- Load your trained model ---
model = load_model('asl_model.keras')  # This must be in the same folder as app.py

# --- Streamlit UI ---
st.set_page_config(page_title="ASL Detection App", layout="centered")
st.title("🤟 ASL Sign Language Detection")
st.write("Upload an image of a hand sign to detect the ASL letter or command.")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 3)

    # Predict
    prediction = model.predict(img)
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Output
    st.success(f"Predicted Sign: **{predicted_class}**")
    st.info(f"Confidence: **{confidence:.2f}%**")
