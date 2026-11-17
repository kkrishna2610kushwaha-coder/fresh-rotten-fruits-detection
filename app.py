
import streamlit as st
import tensorflow as tf
import numpy as np
import json
import pandas as pd
from PIL import Image
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="Fresh & Rotten Fruits Detection",
    page_icon="🍎",
    layout="wide"
)

st.title("🍎 Fresh and Rotten Fruits Detection")
st.subheader("AI Based Fruit Classification Using Deep Learning CNN")

st.write(
    "Upload a fruit image to check whether it is Fresh or Rotten."
)

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fruit_model.keras")

# Check model availability
if not os.path.exists("fruit_model.keras"):
    st.error("Model not found! Please run train_model.py first.")
    st.stop()

model = load_model()

# Load class names
with open("classes.json", "r") as f:
    class_names = json.load(f)

# Image uploader
st.divider()

uploaded_file = st.file_uploader(
    "Upload Fruit Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Fruit Image",
        width=350
    )

    if st.button("Detect Fruit Condition"):

        # Preprocessing
        image = image.resize((128, 128))

        image_array = np.array(
            image,
            dtype=np.float32
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # Prediction
        prediction = model.predict(image_array)

        rotten_probability = float(prediction[0][0])

        if rotten_probability >= 0.5:
            result = "Rotten"
            confidence = rotten_probability * 100
        else:
            result = "Fresh"
            confidence = (1 - rotten_probability) * 100

        st.divider()

        # Display result
        if result == "Fresh":
            st.success("🍏 Fruit Condition: FRESH")
        else:
            st.error("⚠️ Fruit Condition: ROTTEN")

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

        st.progress(
            int(confidence)
        )

        # Save prediction history
        record = {
            "Date": datetime.now().strftime("%Y-%m-%d"),
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Image": uploaded_file.name,
            "Prediction": result,
            "Confidence": round(confidence, 2)
        }

        history_file = "fruit_records.csv"

        new_record = pd.DataFrame([record])

        if os.path.exists(history_file):
            old_records = pd.read_csv(history_file)
            all_records = pd.concat(
                [old_records, new_record],
                ignore_index=True
            )
        else:
            all_records = new_record

        all_records.to_csv(
            history_file,
            index=False
        )

        st.success("Prediction saved successfully!")

# Prediction history
st.divider()
st.header("📊 Fruit Detection History")

if os.path.exists("fruit_records.csv"):

    records = pd.read_csv("fruit_records.csv")

    st.dataframe(
        records,
        use_container_width=True
    )

    st.download_button(
        "Download Detection Report",
        data=records.to_csv(index=False),
        file_name="fruit_detection_report.csv",
        mime="text/csv"
    )

else:
    st.info("No detection records available yet.")

st.divider()

st.caption(
    "Developed using Python | TensorFlow | CNN | Streamlit"
)
