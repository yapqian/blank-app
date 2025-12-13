# app.py
import streamlit as st
import easyocr
import numpy as np
import cv2
from PIL import Image
from pdf2image import convert_from_bytes
from langdetect import detect
import re
import torch

# =========================
# Page config
# =========================
st.set_page_config(page_title="OCR + EN→MS Translator + Summarizer + Fake-News", layout="wide")
st.title("📰 OCR → Translate (EN→MS) → Summarize → Fake-News Detector")
st.write("Mode: Upload Image / Upload PDF / Camera — OR type text manually. (B1 behaviour: raw editable + translate button)")

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
    # English-only OCR reader (fastest/most stable)
    try:
        return easyocr.Reader(["en"], gpu=False)
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_translator_model():
    # EN -> MS translation model (mesolitica nanot5-small-malaysian-translation)
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    model_name = "mesolitica/nanot5-small-malaysian-translation-v2.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

@st.cache_resource(show_spinner=False)
def load_summarizer():
    # multilingual summarizer (mT5 small) — works for Malay reasonably
    # Note: smaller models are faster but less fluent; change if you prefer another.
    from transformers import pipeline
    return pipeline("summarization", model="google/mt5-small", device=0 if torch.cuda.is_available() else -1)

reader = load_ocr_reader()
translator_tok, translator_model = load_translator_model()
summarizer = load_summarizer()

# =========================
# Translation function (EN -> MS)
# =========================
def translate_en_to_malay(text: str, max_length: int = 512) -> str:
    text = text.strip()
    if text == "":
        return ""
    # Some Seq2Seq Malay models expect a task prompt — nanot5 expects "terjemah ke Melayu: " prefix
    prefix = "terjemah ke Melayu: "
    input_text = prefix + text
    inputs = translator_tok.encode(input_text, return_tensors="pt", truncation=True, max_length=max_length)
    outputs = translator_model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
    translated = translator_tok.decode(outputs[0], skip_special_tokens=True)
    return translated

# =========================
# Preprocessing helpers
# =========================
def preprocess_image(img: np.ndarray, sharpen: bool = False, threshold: bool = True) -> np.ndarray:
    """Convert to grayscale, denoise, threshold, optionally sharpen."""
    if img is None:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    if threshold:
        proc = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 2
        )
    else:
        proc = blur
    if sharpen:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        proc = cv2.filter2D(proc, -1, kernel)
    return proc

# =========================
# OCR functions
# =========================
def ocr_image(img: np.ndarray, preprocess_opts: dict) -> tuple:
    """Return boxed image (RGB/BGR) and extracted raw text (string)."""
    if img is None:
        return None, ""
    proc = preprocess_image(img, sharpen=preprocess_opts["sharpen"], threshold=preprocess_opts["threshold"])
    # easyocr expects rgb or gray; provide processed image
    try:
        results = reader.readtext(proc)
    except Exception:
        results = []
    extracted = " ".join([r[1] for r in results]) if results else ""
    boxed = img.copy()
    for (bbox, text, prob) in results:
        # bbox is 4 points; draw rectangle using pts 0 and 2
        p1, p2, p3, p4 = bbox
        p1 = tuple(map(int, p1)); p3 = tuple(map(int, p3))
        cv2.rectangle(boxed, p1, p3, (0, 255, 0), 2)
        cv2.putText(boxed, text, (p1[0], max(p1[1]-8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    return boxed, clean_text(extracted)

def ocr_pdf_bytes(file_bytes: bytes, preprocess_opts: dict) -> tuple:
    pages = convert_from_bytes(file_bytes)
    images = []
    full_text = []
    for page in pages:
        page_np = np.array(page.convert("RGB"))
        boxed, extracted = ocr_image(page_np, preprocess_opts)
        images.append(boxed)
        full_text.append(extracted)
    return images, "\n".join(full_text).strip()

# =========================
# Fake news placeholder
# =========================
def predict_news(text: str) -> str:
    # Placeholder — replace with your model integration
    t = text.lower()
    if "tipu" in t or "bohong" in t or "vaccin" in t:  # naive heuristics
        return "🔴 Likely FAKE (placeholder)"
    return "🟢 Likely REAL (placeholder)"

# =========================
# UI Layout
# =========================
st.sidebar.header("Preprocessing")
sharpen = st.sidebar.checkbox("Sharpen image", value=False)
threshold = st.sidebar.checkbox("Adaptive thresholding", value=True)

st.sidebar.markdown("---")
st.sidebar.header("Input mode")
mode = st.sidebar.radio("Choose input", ["Manual Only", "Upload Image", "Upload PDF", "Use Camera"])

# Shared session state keys
if "raw_text" not in st.session_state:
    st.session_state["raw_text"] = ""      # editable raw OCR / manual text (user edits)
if "malay_text" not in st.session_state:
    st.session_state["malay_text"] = ""    # translated/target Malay (editable)
if "last_mode" not in st.session_state:
    st.session_state["last_mode"] = mode

# Two-column main layout: left for OCR/controls, right for final actions
left_col, right_col = st.columns([1, 1])

# ---------- LEFT: Input / OCR ----------
with left_col:
    st.subheader("Input / OCR")
    # Manual typing is always available in Raw box; but we also show upload widgets
    if mode == "Upload Image":
        uploaded = st.file_uploader("Upload image", type=["png","jpg","jpeg"])
        if uploaded:
            img = np.array(Image.open(uploaded).convert("RGB"))
            preprocess_opts = {"sharpen": sharpen, "threshold": threshold}
            preview = preprocess_image(img, sharpen=sharpen, threshold=threshold)
            st.image(preview, caption="Preprocessed preview", channels="GRAY")
            boxed, extracted = ocr_image(img, preprocess_opts)
            st.image(boxed, caption="OCR bounding boxes", channels="BGR")
            # Insert OCR into raw_text (editable)
            st.session_state["raw_text"] = extracted
    elif mode == "Upload PDF":
        uploaded = st.file_uploader("Upload PDF", type=["pdf"])
        if uploaded:
            file_bytes = uploaded.read()
            preprocess_opts = {"sharpen": sharpen, "threshold": threshold}
            images, extracted = ocr_pdf_bytes(file_bytes, preprocess_opts)
            st.subheader("PDF OCR - pages")
            for i, im in enumerate(images):
                st.image(im, caption=f"Page {i+1}", channels="BGR")
            st.session_state["raw_text"] = extracted
    elif mode == "Use Camera":
        camera_img = st.camera_input("Use camera to capture")
        if camera_img:
            img = np.array(Image.open(camera_img).convert("RGB"))
            preprocess_opts = {"sharpen": sharpen, "threshold": threshold}
            preview = preprocess_image(img, sharpen=sharpen, threshold=threshold)
            st.image(preview, caption="Preprocessed preview", channels="GRAY")
            boxed, extracted = ocr_image(img, preprocess_opts)
            st.image(boxed, caption="OCR bounding boxes", channels="BGR")
            st.session_state["raw_text"] = extracted

    # Manual input always available as editable Raw OCR Text
    st.markdown("### Raw OCR / Manual Text (editable)")
    st.session_state["raw_text"] = st.text_area(
        "Edit or type the original text here (this is the source text):",
        st.session_state["raw_text"],
        height=220
    )

    # Translate button: user edits raw, then presses Translate to populate Malay box
    if st.button("Translate → Malay"):
        raw = st.session_state["raw_text"].strip()
        if raw == "":
            st.warning("Raw text is empty — type or upload first.")
        else:
            lang = detect_language(raw)
            # if detected English, translate to Malay; if Malay, copy; else try to translate if English-like
            if lang == "en":
                with st.spinner("Translating English → Malay..."):
                    try:
                        malay = translate_en_to_malay(raw)
                    except Exception as e:
                        st.error(f"Translation failed: {e}")
                        malay = raw
            else:
                # keep as Malay (assume raw is Malay) — if user typed English and detection failed, still attempt translation
                if lang == "ms" or lang == "id" or lang == "unknown":
                    # if detection says Malay/Indo/unknown, copy raw into malay box but user can change
                    malay = raw
                else:
                    # other languages: attempt translation (best-effort)
                    try:
                        malay = translate_en_to_malay(raw)
                    except Exception:
                        malay = raw
            st.session_state["malay_text"] = malay
            st.success("Translated/copied into Malay box (editable).")

# ---------- RIGHT: Malay box, summarization, analysis ----------
with right_col:
    st.subheader("Malay Text (final) — editable")
    st.session_state["malay_text"] = st.text_area(
        "Malay text for summarization & analysis:",
        st.session_state["malay_text"],
        height=260
    )

    # Summarization (operate on Malay text)
    if st.button("Summarize Malay Text"):
        malay_input = st.session_state["malay_text"].strip()
        if malay_input == "":
            st.warning("Malay text is empty — translate or enter Malay text first.")
        else:
            with st.spinner("Generating summary..."):
                try:
                    summary = summarizer(malay_input, max_length=120, min_length=40, do_sample=False)[0]["summary_text"]
                except Exception as e:
                    # fallback: summarize shorter text or just return first 300 chars
                    summary = malay_input if len(malay_input.split()) < 40 else malay_input[:400] + "..."
                st.subheader("Summary (Malay)")
                st.text_area("Summary output", summary, height=200)

    # Fake news analysis
    if st.button("Analyze Fake News (Malay)"):
        malay_input = st.session_state["malay_text"].strip()
        if malay_input == "":
            st.warning("Malay text is empty — translate or enter Malay text first.")
        else:
            with st.spinner("Running fake-news analysis (placeholder)..."):
                result = predict_news(malay_input)
            st.subheader("Fake News Result")
            st.success(result)

    # Optionally, show detected language for raw text
    st.markdown("---")
    st.write("Detected language for raw text:", detect_language(st.session_state["raw_text"] or ""))

# Footer / notes
st.markdown("---")
st.caption("Notes: • 'Translate → Malay' will try to translate English to Malay and copy Malay as-is. • Replace predict_news() with your real classifier for production.")

