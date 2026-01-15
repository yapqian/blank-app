import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import easyocr
import re
import unicodedata
import docx
from PIL import Image
from pdf2image import convert_from_bytes
from deep_translator import GoogleTranslator
from transformers import pipeline
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sage_model import MultiRelationalGNN

# --- CONFIGURATION ---
st.set_page_config(page_title="Malay News AI Verification", layout="wide")

# Initialize Session State to handle "Not Responding" issues
if 'content' not in st.session_state:
    st.session_state['content'] = ""

def reset_app():
    st.session_state['content'] = ""
    st.rerun()

# --- RESOURCE LOADING ---
@st.cache_resource
def load_all_resources():
    reader = easyocr.Reader(['ms', 'en'])
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    st_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    df_orig = pd.read_csv('final_merged_fake_news_data.csv')
    le_source = LabelEncoder()
    le_source.fit(df_orig['site_url'].unique())
    
    model = MultiRelationalGNN(128, 2, 384, len(le_source.classes_))
    checkpoint = torch.load('best_gnn_malay_model.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return reader, summarizer, st_model, model, le_source

try:
    reader, summarizer, st_model, gnn_model, le_source = load_all_resources()
except Exception as e:
    st.error(f"Missing files: {e}")

def clean_text(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return " ".join(text.split()).lower()

# --- UI LAYOUT ---
st.title("🛡️ Unified Malay News Verification System")
st.markdown("---")

col_input, col_analysis = st.columns([1, 1])

with col_input:
    st.subheader("📥 Input & Extraction")
    mode = st.selectbox("Select Input Method", ["Manual Typing", "Camera Scan", "Image Upload", "PDF Document", "Word Document"])
    
    # Extraction Logic
    extracted = ""
    if mode == "Camera Scan":
        cam = st.camera_input("Scan")
        if cam: extracted = " ".join([t[1] for t in reader.readtext(np.array(Image.open(cam)))])
    elif mode == "Image Upload":
        img = st.file_uploader("Upload Image", type=['jpg','png','jpeg'])
        if img: extracted = " ".join([t[1] for t in reader.readtext(np.array(Image.open(img)))])
    elif mode == "PDF Document":
        pdf = st.file_uploader("Upload PDF", type=['pdf'])
        if pdf:
            for pg in convert_from_bytes(pdf.read()):
                extracted += " ".join([t[1] for t in reader.readtext(np.array(pg))]) + " "
    elif mode == "Word Document":
        doc_f = st.file_uploader("Upload Word", type=['docx'])
        if doc_f: extracted = "\n".join([p.text for p in docx.Document(doc_f).paragraphs])

    # If file was uploaded, update session state
    if extracted:
        st.session_state['content'] = extracted

    # MASTER TEXT BOX (Responsive)
    final_news_text = st.text_area("Final Content (Review / Edit here):", 
                                   value=st.session_state['content'], 
                                   height=400, 
                                   key="master_box")
    # Sync typing back to state
    st.session_state['content'] = final_news_text

with col_analysis:
    st.subheader("🔍 AI Intelligence")
    if final_news_text:
        with st.expander("Language Tools (Malay Focus)"):
            if st.button("Translate Everything to Malay"):
                chunks = [final_news_text[i:i+4000] for i in range(0, len(final_news_text), 4000)]
                translated = [GoogleTranslator(source='auto', target='ms').translate(c) for c in chunks]
                st.write(" ".join(translated))

            if st.button("Generate Malay Summary"):
                raw = summarizer(final_news_text[:1024], max_length=100, min_length=30)[0]['summary_text']
                st.info(GoogleTranslator(source='auto', target='ms').translate(raw))

        st.write("---")
        st.subheader("GNN Prediction")
        source = st.text_input("Source URL", "facebook.com")
        
        if st.button("Run Graph Analysis"):
            with st.spinner("Processing..."):
                # Auto-Malay for GNN accuracy
                malay_text = GoogleTranslator(source='auto', target='ms').translate(final_news_text[:4000])
                emb = torch.tensor(st_model.encode([clean_text(malay_text)]), dtype=torch.float)
                
                data = HeteroData()
                data['article'].x = emb
                data['source'].x = torch.eye(len(le_source.classes_))
                s_idx = le_source.transform([source])[0] if source in le_source.classes_ else 0
                data['article','published_by','source'].edge_index = torch.tensor([[0],[s_idx]])
                data['article','same_day','article'].edge_index = torch.empty((2,0), dtype=torch.long)

                with torch.no_grad():
                    out = gnn_model(data.x_dict, data.edge_index_dict)
                    prob = F.softmax(out['article'], dim=-1)
                    pred = out['article'].argmax(dim=-1).item()
                
                label = "REAL NEWS" if pred == 1 else "FAKE NEWS"
                if pred == 1: st.success(f"{label} ({prob[0][pred]*100:.2f}%)")
                else: st.error(f"{label} ({prob[0][pred]*100:.2f}%)")

        if st.button("🔄 Reset App"):
            reset_app()
    else:
        st.info("Awaiting input from the left panel.")