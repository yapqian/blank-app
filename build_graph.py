import torch
import pandas as pd
import numpy as np
import gc
from gensim.models import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

DATA_PATH = "data/malaya_fake_news_preprocessed_dataframe.pkl"
DOC2VEC_PATH = "models/doc2vec_model.d2v"
SAVE_PT = "models/fake_news_data_object3.pt"

df = pd.read_pickle(DATA_PATH)

# Use only a subset to avoid memory issues
MAX_SAMPLES = 5000
if len(df) > MAX_SAMPLES:
    print(f"Using {MAX_SAMPLES} samples out of {len(df)} (to avoid OOM)")
    df = df.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)

model = Doc2Vec.load(DOC2VEC_PATH)

# Build embeddings
print("Building embeddings...")
emb_np = np.stack([model.dv[str(i)] for i in range(len(df))])
embeddings = torch.tensor(emb_np, dtype=torch.float32)
embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

num_nodes = embeddings.shape[0]
print(f"Number of nodes: {num_nodes}")

# KNN graph
print("Computing similarity matrix...")
sim = cosine_similarity(emb_np)
np.fill_diagonal(sim, 0)
K = 5
nbr_idx = np.argsort(-sim, axis=1)[:, :K]

rows = np.repeat(np.arange(num_nodes), K)
cols = nbr_idx.flatten()

edge_index = torch.tensor([rows, cols], dtype=torch.long)
edge_attr  = torch.tensor(sim[rows, cols], dtype=torch.float32)

print("Making graph undirected...")
edge_index = to_undirected(edge_index)
edge_attr = torch.cat([edge_attr, edge_attr])

labels = torch.tensor(df["label"].values, dtype=torch.long)

# Train/Val/Test split
idx = torch.randperm(num_nodes)
train_cut = int(0.8 * num_nodes)
val_cut = int(0.9 * num_nodes)

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
test_mask  = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[idx[:train_cut]] = True
val_mask[idx[train_cut:val_cut]] = True
test_mask[idx[val_cut:]] = True

gc.collect()

data = Data(
    x=embeddings,
    edge_index=edge_index,
    edge_attr=edge_attr,
    y=labels,
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask
)

torch.save(data, SAVE_PT)
print(f"✔ Graph data saved to {SAVE_PT}")
