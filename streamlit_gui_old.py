import streamlit as st
import easyocr
import numpy as np
import cv2
from pdf2image import convert_from_bytes
from PIL import Image
import re

###############################
#  OCR LOADING
###############################

@st.cache_resource
def load_ocr_reader():
    # Add more languages if needed: ['en','ms','ch_sim','ta']
    return easyocr.Reader(['en', 'ms'])  

reader = load_ocr_reader()


###############################
#  CLEANING FUNCTION
###############################
def clean_text(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


###############################
#  OCR FUNCTION WITH BOUNDING BOXES
###############################
def ocr_image(img):
    results = reader.readtext(img)

    extracted_text = []
    boxed_image = img.copy()

    for (bbox, text, prob) in results:
        extracted_text.append(text)

        # Draw bounding boxes
        pts = np.array(bbox, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(boxed_image, [pts], True, (0, 255, 0), 2)

        # Put text label
        cv2.putText(
            boxed_image, text, 
            (pts[0][0][0], pts[0][0][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    return boxed_image, clean_text(" ".join(extracted_text))


###############################
#  PDF OCR PROCESSING
###############################
def ocr_pdf(file):
    pages = convert_from_bytes(file.read())
    full_text = ""
    images_with_boxes = []

    for page in pages:
        page_np = np.array(page)
        boxed, text = ocr_image(page_np)
        full_text += text + "\n"
        images_with_boxes.append(boxed)

    return images_with_boxes, clean_text(full_text)


###############################
#  STREAMLIT UI
###############################
st.header("📄 OCR – Optical Character Recognition Module")

option = st.radio("Choose Input Method:", [
    "Upload Image",
    "Upload PDF",
    "Use Camera"
])

###############################
#  UPLOAD IMAGE
###############################
if option == "Upload Image":
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if file:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        boxed_img, extracted = ocr_image(img)

        st.subheader("🖼 OCR Result (with Bounding Boxes)")
        st.image(boxed_img, channels="BGR")

        st.subheader("📌 Extracted Text")
        st.text_area("Output", extracted, height=200)

        if st.button("Analyze with Fake News Model"):
            result = predict_news(extracted)     # <-- your prediction function
            st.success(f"Prediction: {result}")


###############################
#  UPLOAD PDF
###############################
elif option == "Upload PDF":
    file = st.file_uploader("Upload a PDF", type=["pdf"])

    if file:
        images, extracted = ocr_pdf(file)

        st.subheader("📄 OCR on PDF Pages (with Bounding Boxes)")
        for idx, img in enumerate(images):
            st.image(img, caption=f"Page {idx+1}", channels="BGR")

        st.subheader("📌 Extracted Text")
        st.text_area("Output", extracted, height=300)

        if st.button("Analyze Extracted Text"):
            result = predict_news(extracted)
            st.success(f"Prediction: {result}")


###############################
#  CAMERA CAPTURE
###############################
elif option == "Use Camera":
    camera_img = st.camera_input("Capture an image")

    if camera_img:
        img = Image.open(camera_img)
        img = np.array(img)

        boxed_img, extracted = ocr_image(img)

        st.subheader("📸 OCR Result (with Bounding Boxes)")
        st.image(boxed_img, channels="BGR")

        st.subheader("📌 Extracted Text")
        st.text_area("Output", extracted, height=200)

        if st.button("Analyze Captured Text"):
            result = predict_news(extracted)
            st.success(f"Prediction: {result}")
