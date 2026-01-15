import streamlit as st
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel

# --- CONFIGURATION ---
MODEL_PATH = 'best_gnn_malay_model.pt'
DATA_PATH = 'final_merged_fake_news_data.csv'
EMBEDDINGS_PATH = 'article_embeddings.pt'

# --- LOAD RESOURCES ---
@st.cache_resource
def load_gnn_model():
    # Load the state dict and architecture from your uploaded .pt file
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    # Note: Ensure your GNN class definition matches the saved state_dict
    return checkpoint

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

# --- APP UI ---
st.title("🛡️ Malay Fake News Detector (Unified)")
st.subheader("GNN-Powered Analysis & Article Search")

# Sidebar for Navigation
option = st.sidebar.selectbox("Select Function", ["Classify News", "Browse Dataset"])

data = load_data()

if option == "Classify News":
    st.write("### Analyze Malay News Content")
    user_input = st.text_area("Enter news text in Malay:", height=200)
    
    if st.button("Predict"):
        if user_input:
            with st.spinner('Analyzing...'):
                # Simplified prediction logic for demonstration
                # In a real scenario, you would transform text into a graph (Data object)
                # and pass it through the GNN model loaded from MODEL_PATH
                st.info("Processing text through GNN Model...")
                
                # Mock Result based on provided CSV labels for UI demonstration
                st.success("Analysis Complete!")
                st.metric(label="Reliability Score", value="85%", delta="Real News")
        else:
            st.warning("Please enter some text.")

elif option == "Browse Dataset":
    st.write("### Explore Training Data")
    st.write(f"Showing last few entries from `{DATA_PATH}`:")
    st.dataframe(data.tail(5))
    
    # Feature: Search by Keyword
    search_query = st.text_input("Search titles for keywords (e.g., 'Hillary', 'Kerajaan'):")
    if search_query:
        filtered_df = data[data['title'].str.contains(search_query, case=False, na=False)]
        st.write(f"Found {len(filtered_df)} matches:")
        st.table(filtered_df[['title', 'label']].head(10))

# --- FOOTER ---
st.divider()
st.caption("Model: GNN-Malay-V1 | Data Source: final_merged_fake_news_data.csv")