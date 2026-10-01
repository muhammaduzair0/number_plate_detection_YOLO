import re

import easyocr
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


@st.cache_resource
def load_model():
    return YOLO("best.pt")


@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(["en"], gpu=False)


def clean_plate_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def read_plate_text(reader: easyocr.Reader, plate_image: Image.Image) -> str:
    ocr_results = reader.readtext(np.array(plate_image))
    if not ocr_results:
        return "No text detected"

    text_candidates = []
    for _, text, confidence in ocr_results:
        cleaned_text = clean_plate_text(text)
        if cleaned_text and confidence >= 0.2:
            text_candidates.append((cleaned_text, confidence))

    if not text_candidates:
        return "No text detected"

    text_candidates.sort(key=lambda item: item[1], reverse=True)
    return text_candidates[0][0]

st.title("YOLO Object Detection")
st.write("Upload an image to detect the number plate and read the registration number.")

# Image upload
uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Convert Streamlit UploadedFile buffer to a PIL Image
    image = Image.open(uploaded_file).convert("RGB")

    model = load_model()
    reader = load_ocr_reader()

    # YOLO prediction
    results = model.predict(image)

    # Detection result plot (returns BGR numpy array)
    result_image = results[0].plot()

    # Show result (convert BGR to RGB so colors render correctly in Streamlit)
    st.image(result_image, channels="BGR", caption="Detection Result")

    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        st.subheader("OCR Results")
        for index, box in enumerate(boxes, start=1):
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            cropped_plate = image.crop((x1, y1, x2, y2))
            plate_text = read_plate_text(reader, cropped_plate)

            st.write(f"Plate {index}: {plate_text}")
            st.image(cropped_plate, caption=f"Crop {index}", use_container_width=True)
    else:
        st.info("No number plate detected for OCR.")