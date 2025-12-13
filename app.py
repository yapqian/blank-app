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
    # English-only OCR reader with GPU support if available
    try:
        gpu = torch.cuda.is_available()
        return easyocr.Reader(["en"], gpu=gpu)
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_translator_model():
    # EN -> MS translation model (mesolitica nanot5-small-malaysian-translation)
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    model_name = "mesolitica/nanot5-small-malaysian-translation-v2.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # Move model to available device (GPU if available) for faster inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model.to(device)
    except Exception:
        # best-effort; if moving fails, continue on CPU
        device = torch.device("cpu")
    return tokenizer, model, device

@st.cache_resource(show_spinner=False)
def load_summarizer():
    # multilingual summarizer (mT5 small) — works for Malay reasonably
    # Note: smaller models are faster but less fluent; change if you prefer another.
    from transformers import pipeline
    return pipeline("summarization", model="google/mt5-small", device=0 if torch.cuda.is_available() else -1)

# Don't load models at startup to avoid OOM on Streamlit Cloud
# Models will be loaded on-demand via @st.cache_resource
# reader = load_ocr_reader()
# translator_tok, translator_model = load_translator_model()
# summarizer = load_summarizer()

# =========================
# Translation function (EN -> MS)
# =========================
def translate_en_to_malay(text: str, max_length: int = 512) -> str:
    text = text.strip()
    if text == "":
        return ""
    # Load translator on-demand
    try:
        translator_tok, translator_model, device = load_translator_model()
    except Exception as e:
        # If loading the heavy model fails, return original text as a fallback
        return text

    # Some Seq2Seq Malay models expect a task prompt — nanot5 expects "terjemah ke Melayu: " prefix
    prefix = "terjemah ke Melayu: "
    input_text = prefix + text

    # Helper: chunk long input into sentence-based pieces to avoid token limit/OOM
    def _chunk_text_for_translation(s: str, tokenizer, max_len: int):
        import re
        s = s.strip()
        if not s:
            return []
        # split by sentence endings
        sents = re.split(r'(?<=[.!?])\s+', s)
        chunks = []
        cur = ""
        for sent in sents:
            cand = (cur + " " + sent).strip() if cur else sent
            # estimate token length using tokenizer
            try:
                tok_len = len(tokenizer.encode(cand, add_special_tokens=False))
            except Exception:
                tok_len = len(cand.split())
            if tok_len <= max_len - 16:
                cur = cand
            else:
                if cur:
                    chunks.append(cur)
                # if single sentence too long, truncate by characters as last resort
                try:
                    sent_len = len(tokenizer.encode(sent, add_special_tokens=False))
                except Exception:
                    sent_len = len(sent.split())
                if sent_len > max_len - 16:
                    chunks.append(sent[: max_len * 2])
                    cur = ""
                else:
                    cur = sent
        if cur:
            chunks.append(cur)
        return chunks

    max_len = max_length
    chunks = _chunk_text_for_translation(input_text, translator_tok, max_len)
    if not chunks:
        return ""

    translated_parts = []
    for chunk in chunks:
        try:
            inputs = translator_tok(chunk, return_tensors="pt", truncation=True, max_length=max_len)
            # move inputs to device
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            outputs = translator_model.generate(**inputs, max_length=max_len, num_beams=4, early_stopping=True, no_repeat_ngram_size=2)
            out = translator_tok.decode(outputs[0], skip_special_tokens=True)
            translated_parts.append(out)
        except Exception:
            # fallback: append raw chunk
            translated_parts.append(chunk)

    translated = " ".join(translated_parts)
    return translated

# =========================
# Preprocessing helpers
# =========================
def preprocess_image(img: np.ndarray, sharpen: bool = False, threshold: bool = True) -> np.ndarray:
    """Convert to grayscale, denoise, threshold, optionally sharpen for better OCR."""
    if img is None:
        return img
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Bilateral filtering: preserves edges while removing noise (better than Gaussian for OCR)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    if threshold:
        # Adaptive thresholding with larger block size for better sensitivity
        proc = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            41, 3  # Larger blockSize (41) for better sensitivity
        )
    else:
        proc = denoised
    
    if sharpen:
        # Unsharp mask: better than simple kernel for text sharpening
        gaussian = cv2.GaussianBlur(proc, (0, 0), 2.0)
        proc = cv2.addWeighted(proc, 1.5, gaussian, -0.5, 0)
    
    return proc

# =========================
# OCR functions
# =========================
def ocr_image(img: np.ndarray, preprocess_opts: dict) -> tuple:
    """Return boxed image (RGB/BGR) and extracted raw text with confidence filtering."""
    if img is None:
        return None, ""
    
    # Load reader on-demand
    reader = load_ocr_reader()
    if reader is None:
        return img.copy(), ""
    
    proc = preprocess_image(img, sharpen=preprocess_opts["sharpen"], threshold=preprocess_opts["threshold"])
    
    # easyocr.readtext returns list of (bbox, text, confidence)
    try:
        results = reader.readtext(proc)
    except Exception:
        results = []
    
    # Filter by confidence threshold (0.0-1.0) — keep only high-confidence detections
    min_confidence = 0.3  # Adjust: lower = more sensitive, higher = more strict
    filtered_results = [(bbox, text, conf) for bbox, text, conf in results if conf >= min_confidence]
    
    # Extract text with confidence info
    extracted_texts = []
    for bbox, text, conf in filtered_results:
        # Clean text: strip whitespace
        clean = text.strip()
        if clean and len(clean) > 1:  # Ignore single characters
            extracted_texts.append(clean)
    
    extracted = " ".join(extracted_texts)
    
    # Draw bounding boxes on original image
    boxed = img.copy()
    for (bbox, text, conf) in filtered_results:
        # bbox is 4 points (corners); draw rectangle
        p1, p2, p3, p4 = bbox
        p1 = tuple(map(int, p1))
        p3 = tuple(map(int, p3))
        
        # Color based on confidence: green (high), yellow (medium), red (low)
        if conf >= 0.7:
            color = (0, 255, 0)  # Green
        elif conf >= 0.5:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        
        cv2.rectangle(boxed, p1, p3, color, 2)
        # Draw text and confidence
        label = f"{text} ({conf:.2f})"
        cv2.putText(boxed, label, (p1[0], max(p1[1]-8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    return boxed, clean_text(extracted)

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
                    summarizer = load_summarizer()  # Load on-demand
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

