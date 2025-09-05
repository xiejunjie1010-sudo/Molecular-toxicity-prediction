# core/models/gcn_wrapper.py
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
import torch

class Net(torch.nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=12):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin   = torch.nn.Linear(hidden, out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_add_pool(x, batch)
        return self.lin(x)          # 返回 logits (未 sigmoid)
