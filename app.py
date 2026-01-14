import gc
import torch
import streamlit as st
import easyocr
import numpy as np
import cv2
from PIL import Image
from pdf2image import convert_from_bytes
from langdetect import detect
import re
from gensim.models.doc2vec import Doc2Vec
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity

# Memory Management
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# =========================
# Page config
# =========================
st.set_page_config(page_title="OCR + Translation + Auto-Summary", layout="wide")
st.title("📰 OCR → Auto-Translate & Summarize → Fake-News")
st.write("Mode: Upload Image / PDF / Camera. Edit Raw text, then Translate & Summarize automatically.")

# =========================
# Helpers / Utilities
# =========================
def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unknown"

# =========================
# Load models (cached)
# =========================
@st.cache_resource(show_spinner=False)
def load_ocr_reader():
    try:
        gpu = torch.cuda.is_available()
        return easyocr.Reader(["en", "ms"], gpu=gpu)
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_translator_model():
    try:
        from transformers import MarianTokenizer, MarianMTModel
        model_name = "Helsinki-NLP/opus-mt-en-ms"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        model_type = "marian"
    except Exception:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        model_name = "mesolitica/nanot5-small-malaysian-translation-v2.1"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model_type = "seq2seq"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device, model_type

@st.cache_resource(show_spinner=False)
def load_summarizer():
    from transformers import pipeline
    # mT5 is excellent for multilingual (Malay) abstractive summarization
    return pipeline("summarization", model="google/mt5-small", device=0 if torch.cuda.is_available() else -1)

@st.cache_resource(show_spinner=False)
def load_gnn_assets():
    from gat_model import GAT 
    d2v = Doc2Vec.load("models/doc2vec_model.d2v")
    model = GAT(in_dim=100, hid_dim=128, out_dim=2)
    model.load_state_dict(torch.load("models/gat_fake_news.pt", map_location="cpu"))
    model.eval()
    train_data = torch.load("models/fake_news_data_object3.pt", map_location="cpu")
    return d2v, model, train_data

# =========================
# Processing Functions
# =========================
def preprocess_image(img: np.ndarray, sharpen: bool = False, threshold: bool = True) -> np.ndarray:
    if img is None: return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    if threshold:
        proc = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 3)
    else:
        proc = denoised
    if sharpen:
        gaussian = cv2.GaussianBlur(proc, (0, 0), 2.0)
        proc = cv2.addWeighted(proc, 1.5, gaussian, -0.5, 0)
    return proc

def ocr_image(img: np.ndarray, preprocess_opts: dict) -> tuple:
    reader = load_ocr_reader()
    if reader is None: return img.copy(), ""
    proc = preprocess_image(img, sharpen=preprocess_opts["sharpen"], threshold=preprocess_opts["threshold"])
    results = reader.readtext(proc)
    extracted_texts = []
    boxed = img.copy()
    for (bbox, text, conf) in results:
        if conf >= 0.3:
            extracted_texts.append(text.strip())
            p1, _, p3, _ = bbox
            p1, p3 = tuple(map(int, p1)), tuple(map(int, p3))
            color = (0, 255, 0) if conf >= 0.7 else (0, 255, 255)
            cv2.rectangle(boxed, p1, p3, color, 2)
    return boxed, clean_text(" ".join(extracted_texts))

def ocr_pdf_bytes(file_bytes, preprocess_opts):
    pages = convert_from_bytes(file_bytes)
    full_text, images_with_boxes = "", []
    for page in pages:
        page_np = np.array(page)
        page_bgr = cv2.cvtColor(page_np, cv2.COLOR_RGB2BGR)
        boxed, text = ocr_image(page_bgr, preprocess_opts)
        full_text += text + "\n "
        images_with_boxes.append(boxed)
    return images_with_boxes, clean_text(full_text)

def translate_en_to_malay(text: str) -> str:
    if not text.strip(): return ""
    try:
        tok, model, device, m_type = load_translator_model()
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        outputs = model.generate(**inputs, max_length=512)
        return tok.decode(outputs[0], skip_special_tokens=True)
    except:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source='auto', target='ms').translate(text)

def predict_news(text: str) -> str:
    d2v, gat, train_data = load_gnn_assets()
    new_vec = d2v.infer_vector(text.split()).reshape(1, -1)
    sims = cosine_similarity(new_vec, train_data.x.numpy())
    top_k_idx = np.argsort(-sims[0])[:5]
    combined_x = torch.cat([torch.tensor(new_vec), train_data.x[top_k_idx]], dim=0)
    rows = np.array([0, 0, 0, 0, 0, 1, 2, 3, 4, 5]); cols = np.array([1, 2, 3, 4, 5, 0, 0, 0, 0, 0])
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    with torch.no_grad():
        out = gat(combined_x, edge_index)
        probs = torch.softmax(out[0], dim=0)
        prediction = probs.argmax().item()
        confidence = probs[prediction].item()
    if prediction == 1:
        st.error(f"### Result: LIKELY FAKE")
        st.progress(confidence)
        return f"GNN Prediction: Misinformation ({confidence:.1%} confidence)."
    else:
        st.success(f"### Result: LIKELY REAL")
        st.progress(confidence)
        return f"GNN Prediction: Authentic ({confidence:.1%} confidence)."

# =========================
# UI Layout
# =========================
st.sidebar.header("OCR Settings")
sharpen = st.sidebar.checkbox("Sharpen image", value=False)
threshold = st.sidebar.checkbox("Adaptive thresholding", value=True)
mode = st.sidebar.radio("Input mode", ["Manual Only", "Upload Image", "Upload PDF", "Use Camera"])

if "raw_text" not in st.session_state: st.session_state["raw_text"] = ""
if "malay_text" not in st.session_state: st.session_state["malay_text"] = ""
if "summary_text" not in st.session_state: st.session_state["summary_text"] = ""

left_col, right_col = st.columns(2)

with left_col:
    st.subheader("1. Input / OCR")
    if mode == "Upload Image":
        up = st.file_uploader("Image", type=["png","jpg","jpeg"])
        if up:
            img = np.array(Image.open(up).convert("RGB"))
            boxed, ext = ocr_image(img, {"sharpen": sharpen, "threshold": threshold})
            st.image(boxed, caption="OCR Result", channels="BGR")
            st.session_state["raw_text"] = ext
    elif mode == "Upload PDF":
        up = st.file_uploader("PDF", type=["pdf"])
        if up:
            imgs, ext = ocr_pdf_bytes(up.read(), {"sharpen": sharpen, "threshold": threshold})
            st.image(imgs[0], caption="Page 1 Preview", channels="BGR")
            st.session_state["raw_text"] = ext
    elif mode == "Use Camera":
        cam = st.camera_input("Capture")
        if cam:
            img = np.array(Image.open(cam).convert("RGB"))
            boxed, ext = ocr_image(img, {"sharpen": sharpen, "threshold": threshold})
            st.session_state["raw_text"] = ext

    st.markdown("### Raw OCR / Manual Text")
    edited_raw = st.text_area("Edit text here:", value=st.session_state["raw_text"], height=200, key="main_raw")
    st.session_state["raw_text"] = edited_raw

    if st.button("Translate & Auto-Summarize", type="primary"):
        if not edited_raw.strip():
            st.warning("No text to process.")
        else:
            with st.spinner("Processing..."):
                # Step 1: Translate
                lang = detect_language(edited_raw)
                m_text = translate_en_to_malay(edited_raw) if lang == "en" else edited_raw
                st.session_state["malay_text"] = m_text
                # Step 2: Auto-Summarize
                try:
                    summ_pipe = load_summarizer()
                    s_res = summ_pipe(m_text, max_length=100, min_length=30, do_sample=False, num_beams=4, early_stopping=True)[0]['summary_text']
                    st.session_state["summary_text"] = s_res
                except:
                    st.session_state["summary_text"] = "Summarization skipped (text too short or error)."
                st.rerun()

with right_col:
    st.subheader("2. Final Text & Analysis")
    current_malay = st.text_area("Malay Translation:", value=st.session_state["malay_text"], height=200, key="main_malay")
    st.session_state["malay_text"] = current_malay

    if st.session_state["summary_text"]:
        st.info(f"**Auto-Summary (Malay):**\n\n{st.session_state['summary_text']}")

    st.write("---")
    if st.button("Analyze Fake News"):
        if current_malay.strip():
            with st.spinner("GNN Analyzing..."):
                res = predict_news(current_malay)
                st.write(res)
        else: st.warning("No Malay text to analyze.")

st.markdown("---")
st.caption("Status: GNN Model fully integrated (GATv2 + Doc2Vec Inductive Inference).")