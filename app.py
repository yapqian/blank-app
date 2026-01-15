import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import cv2
import easyocr
import re
import unicodedata
import docx
from PIL import Image
from pdf2image import convert_from_bytes
from googletrans import Translator
from transformers import pipeline
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sage_model import MultiRelationalGNN

# --- SETTINGS ---
st.set_page_config(page_title="Malay News Verification System", layout="wide")

@st.cache_resource
def load_all_resources():
    reader = easyocr.Reader(['ms', 'en'])
    translator = Translator()
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    st_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2') # 384 dim
    
    # Load Training Metadata
    df_orig = pd.read_csv('final_merged_fake_news_data.csv')
    le_source = LabelEncoder()
    le_source.fit(df_orig['site_url'].unique())
    
    # Init GraphSAGE Model (Matches your .pt file weights)
    model = MultiRelationalGNN(
        hidden_channels=128, 
        out_channels=2, 
        article_in_channels=384, 
        source_in_channels=len(le_source.classes_)
    )
    checkpoint = torch.load('best_gnn_malay_model.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return reader, translator, summarizer, st_model, model, le_source

reader, translator, summarizer, st_model, gnn_model, le_source = load_all_resources()

# --- PREPROCESSING ---
def final_clean_text(text):
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'http\S+|www\S+|<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return " ".join(text.split()).lower()

# --- UI ---
st.title("🛡️ Unified Malay News Verification System")
st.markdown("---")

col_input, col_analysis = st.columns([1, 1])

with col_input:
    st.subheader("📥 Input Methods")
    input_mode = st.selectbox("Select Input", ["Manual Typing", "Camera Scan", "Image Upload", "PDF Document", "Word Document"])
    
    extracted_text = ""

    if input_mode == "Manual Typing":
        extracted_text = st.text_area("Type news here:", height=300)
    elif input_mode == "Camera Scan":
        cam_file = st.camera_input("Scan News")
        if cam_file:
            img = Image.open(cam_file)
            res = reader.readtext(np.array(img))
            extracted_text = " ".join([t[1] for t in res])
    elif input_mode == "Image Upload":
        img_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
        if img_file:
            extracted_text = " ".join([t[1] for t in reader.readtext(np.array(Image.open(img_file)))])
    elif input_mode == "PDF Document":
        pdf_file = st.file_uploader("Upload PDF", type=['pdf'])
        if pdf_file:
            images = convert_from_bytes(pdf_file.read())
            for pg in images:
                extracted_text += " ".join([t[1] for t in reader.readtext(np.array(pg))]) + "\n"
    elif input_mode == "Word Document":
        doc_file = st.file_uploader("Upload Word", type=['docx'])
        if doc_file:
            doc = docx.Document(doc_file)
            extracted_text = "\n".join([p.text for p in doc.paragraphs])

    final_text = st.text_area("Final Text to Check:", value=extracted_text, height=200)

with col_analysis:
    st.subheader("🔍 AI Intelligence")
    if final_text:
        # Translation & Summary
        with st.expander("Language Tools"):
            target = st.selectbox("Translate to", ["en", "ms", "zh-cn"])
            if st.button("Translate"):
                st.write(translator.translate(final_text, dest=target).text)
            if st.button("Summarize"):
                st.info(summarizer(final_text[:1024], max_length=100)[0]['summary_text'])

        # GNN Prediction
        st.write("---")
        st.subheader("GNN Model Verification")
        source_url = st.text_input("Source URL (e.g., hmetro.com.my)", "facebook.com")
        
        if st.button("Predict"):
            cleaned = final_clean_text(final_text)
            emb = torch.tensor(st_model.encode([cleaned]), dtype=torch.float)
            
            data = HeteroData()
            data['article'].x = emb
            data['source'].x = torch.eye(len(le_source.classes_))
            s_idx = le_source.transform([source_url])[0] if source_url in le_source.classes_ else 0
            data['article', 'published_by', 'source'].edge_index = torch.tensor([[0], [s_idx]], dtype=torch.long)

            with torch.no_grad():
                out = gnn_model(data.x_dict, data.edge_index_dict)
                prob = F.softmax(out['article'], dim=-1)
                pred = out['article'].argmax(dim=-1).item()
            
            label = "REAL" if pred == 1 else "FAKE"
            st.metric("GNN Result", label, f"{prob[0][pred]*100:.2f}% Confidence")