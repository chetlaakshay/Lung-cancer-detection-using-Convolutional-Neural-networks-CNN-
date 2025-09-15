import streamlit as st
import numpy as np
import cv2
from tensorflow import keras
from PIL import Image

# Load the pre-trained model
loaded_model = keras.models.load_model('lung_cancer_detection_model.h5')

# Set page configuration
st.set_page_config(
    page_title="Lung Cancer Detection",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for a sleek interface
st.markdown("""
    <style>
        body {
            background-color: #0e1117;
            color: #ffffff;
        }
        .stApp {
            background-color: #0e1117;
        }
        .header-title {
            font-size: 2.5em;
            font-weight: 700;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1.5em;
        }
        .upload-area {
            border-radius: 10px;
            border: 2px dashed #1f77b4;
            padding: 2em;
            text-align: center;
            margin-bottom: 2em;
            background-color: #1b1e23;
        }
        .upload-area:hover {
            border-color: #1f88e5;
        }
        .prediction-button {
            background-color: #1f77b4;
            color: #ffffff;
            font-size: 1.2em;
            padding: 0.5em 1.5em;
            border-radius: 10px;
            margin-top: 1em;
        }
        .prediction-button:hover {
            background-color: #1f88e5;
        }
        .result {
            font-size: 1.8em;
            color: #76d7c4;
            text-align: center;
            margin-top: 2em;
        }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown("<div class='header-title'>Lung Cancer Detection</div>", unsafe_allow_html=True)

# File Uploader
st.markdown("<div class='upload-area'>Upload an image of a lung tissue sample, and the model will predict whether it is Lung Adenocarcinoma (lung_aca), Normal Lung (lung_n), or Lung Squamous Cell Carcinoma (lung_scc).</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Convert the image to the format needed for prediction
    img_array = np.array(image.convert('RGB'))
    img_resized = cv2.resize(img_array, (256, 256))
    img_expanded = np.expand_dims(img_resized, axis=0)
    
    # Predict button
    if st.button('Predict', key="prediction-button"):
        prediction = loaded_model.predict(img_expanded)
        predicted_class = np.argmax(prediction, axis=1)[0]
        classes = ['lung_aca', 'lung_n', 'lung_scc']
        st.markdown(f"<div class='result'>The predicted class for the image is: {classes[predicted_class]}</div>", unsafe_allow_html=True)
