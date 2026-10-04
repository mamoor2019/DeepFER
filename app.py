from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


MODEL_PATH = Path(__file__).with_name("best_cnn_model.keras")
EMOTION_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]


st.set_page_config(page_title="Facial Emotion Recognition", layout="centered")
st.title("Facial Emotion Recognition")


@st.cache_resource
def load_emotion_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


uploaded_file = st.file_uploader(
    "Upload a cropped face image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    preview_column, result_column = st.columns(2)

    with preview_column:
        st.image(image, caption="Uploaded image", use_container_width=True)

    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH.name}")
    else:
        resized_image = image.resize((48, 48))
        model_input = np.asarray(resized_image, dtype=np.float32) / 255.0
        model_input = np.expand_dims(model_input, axis=0)

        probabilities = load_emotion_model().predict(model_input, verbose=0)[0]
        predicted_index = int(np.argmax(probabilities))

        with result_column:
            st.subheader(EMOTION_LABELS[predicted_index].title())
            st.metric("Confidence", f"{probabilities[predicted_index]:.1%}")

        probability_table = {
            "Emotion": [label.title() for label in EMOTION_LABELS],
            "Probability": probabilities,
        }
        st.subheader("Class probabilities")
        st.bar_chart(probability_table, x="Emotion", y="Probability")
else:
    st.info("Upload a close-up face crop to run a prediction.")