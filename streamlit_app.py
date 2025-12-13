import streamlit as st
import torch
import torch.nn.functional as F
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Define GAT model class (same as training)
class GAT(torch.nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, heads: int = 8, dropout: float = 0.6):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, hid_dim, heads=heads, dropout=dropout)
        self.conv2 = GATv2Conv(hid_dim * heads, hid_dim, heads=1, dropout=dropout)
        self.lin   = torch.nn.Linear(hid_dim, out_dim)
        self.drop = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.drop, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.lin(x)
        return x

# Load models
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOC2VEC_PATH = "/content/drive/MyDrive/Colab Notebooks/doc2vec_model.d2v"
GAT_MODEL_PATH = "/content/drive/MyDrive/Colab Notebooks/gat_fake_news.pt"
DATA_PATH = "/content/drive/MyDrive/Colab Notebooks/fake_news_data_object3.pt"

data = torch.load(DATA_PATH, weights_only=False).to(DEVICE)
num_nodes = data.x.shape[0]
feat_dim = data.x.shape[1]

model_doc2vec = Doc2Vec.load(DOC2VEC_PATH)
model = GAT(in_dim=feat_dim, hid_dim=128, out_dim=2).to(DEVICE)
model.load_state_dict(torch.load(GAT_MODEL_PATH))
model.eval()

# Classification function
def classify_news(article):
    article = article.lower()
    article = re.sub(r'[^\w\s]', '', article)
    article = re.sub(r'\d+', '<NUM>', article)
    tokens = article.strip().split()

    user_vec = model_doc2vec.infer_vector(tokens)
    user_vec = torch.tensor(user_vec, dtype=torch.float32)
    user_vec = F.normalize(user_vec.unsqueeze(0), p=2, dim=1).to(DEVICE)

    K = 5
    sims = cosine_similarity(user_vec.cpu().numpy(), data.x.cpu().numpy())[0]
    topK = np.argsort(-sims)[:K]
    user_edges = torch.tensor([[num_nodes]*K, topK.tolist()], dtype=torch.long).to(DEVICE)
    user_edges = torch.cat([user_edges, user_edges[[1,0]]], dim=1)

    with torch.no_grad():
        combined_x = torch.cat([data.x, user_vec], dim=0)
        combined_edge_index = torch.cat([data.edge_index, user_edges], dim=1)
        logits = model(combined_x, combined_edge_index)
        probs = torch.softmax(logits[-1], dim=0).cpu().numpy()

    return probs

# Streamlit UI
st.title("📰 Malay Fake News Detection (GAT + Doc2Vec)")
st.write("Enter a news article below to classify it.")

user_input = st.text_area("News Article:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        probs = classify_news(user_input)
        fake_prob = probs[0] * 100
        real_prob = probs[1] * 100

        label = "🟢 REAL" if real_prob > fake_prob else "🔴 FAKE"

        st.subheader("Prediction Result")
        st.write(f"**{label}**")
        st.write(f"Fake: {fake_prob:.2f}%")
        st.write(f"Real: {real_prob:.2f}%")
