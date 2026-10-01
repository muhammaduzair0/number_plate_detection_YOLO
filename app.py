from ultralytics import YOLO
import streamlit as st
from PIL import Image

# YOLO model load
model = YOLO("best.pt")

st.title("YOLO Object Detection")

# Image upload
uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Convert Streamlit UploadedFile buffer to a PIL Image
    image = Image.open(uploaded_file)

    # YOLO prediction
    results = model.predict(image)

    # Detection result plot (returns BGR numpy array)
    result_image = results[0].plot()

    # Show result (convert BGR to RGB so colors render correctly in Streamlit)
    st.image(result_image, channels="BGR", caption="Detection Result")