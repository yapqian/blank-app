import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import wandb

DATA_PATH = "models/fake_news_data_object3.pt"
MODEL_SAVE = "models/gat_fake_news.pt"

data = torch.load(DATA_PATH, weights_only=False)

device = torch.device("cpu")
data = data.to(device)

class GAT(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, heads=8, dropout=0.6):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, hid_dim, heads=heads, dropout=dropout)
        self.conv2 = GATv2Conv(hid_dim*heads, hid_dim, heads=1, dropout=dropout)
        self.lin   = torch.nn.Linear(hid_dim, out_dim)
        self.drop = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.drop, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        return self.lin(x)

feat_dim = data.x.shape[1]
model = GAT(in_dim=feat_dim, hid_dim=128, out_dim=2).to(device)
opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-4)

wandb.init(project="fake-news-gnn", name="gat-doc2vec-codespace")

best_val = float("inf")
pat = 20
counter = 0

for epoch in range(1, 301):
    model.train()
    opt.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    opt.step()

    model.eval()
    with torch.no_grad():
        val_out = out[data.val_mask]
        val_loss = F.cross_entropy(val_out, data.y[data.val_mask])

        preds = val_out.argmax(dim=1).cpu()
        trues = data.y[data.val_mask].cpu()

        val_f1 = f1_score(trues, preds)

        wandb.log({"val_loss": val_loss, "val_f1": val_f1})

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), MODEL_SAVE)
            counter = 0
        else:
            counter += 1

    if counter >= pat:
        break

print(f"✔ Model saved to {MODEL_SAVE}")
