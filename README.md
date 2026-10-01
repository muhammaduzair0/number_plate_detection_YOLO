## About This Project

I built this vehicle number plate detection application using a custom-trained YOLO model and a Streamlit web interface. Users can upload an image, view detected license plates with bounding boxes, and read the plate text with OCR.

The model detects a single class, `License-Plate`. Its saved validation results report 94.85% precision, 78.23% recall, and 85.76% mAP@50.

The current version detects the plate region and runs OCR on each crop to extract registration numbers.

## Features

- Detect license plates with YOLO
- Crop each detected plate automatically
- Run OCR on the crop and display the extracted text
- Show both the annotated image and the plate crops

## Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```
