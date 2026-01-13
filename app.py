import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json

# Load model
model = tf.keras.models.load_model("fashion_cnn_model.h5")

# Load labels
with open("labels.json") as f:
    class_names = json.load(f)

st.set_page_config(page_title="Fashion AI", layout="centered")

st.title("👕 Fashion Item Classifier")
st.write("Upload a clothing image and AI will classify it.")

uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("L")   # grayscale
    img = img.resize((28,28))
    st.image(uploaded, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(img)/255.0
    img_array = img_array.reshape(1,28,28,1)

    pred = model.predict(img_array)
    class_id = np.argmax(pred)
    confidence = np.max(pred)*100

    st.markdown("### 🧠 Prediction")
    st.success(f"{class_names[class_id]}  ({confidence:.2f}%)")

    st.bar_chart(pred[0])
