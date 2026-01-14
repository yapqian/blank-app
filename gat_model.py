# gat_model.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

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