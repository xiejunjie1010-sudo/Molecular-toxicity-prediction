import torch, torch.nn.functional as F
from torch_geometric.nn import GATConv, global_add_pool

class Net(torch.nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=12, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden, heads=heads)
        self.conv2 = GATConv(hidden*heads, hidden, heads=1, concat=False)
        self.lin   = torch.nn.Linear(hidden, out_dim)
    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        x = F.elu(self.conv1(x, ei))
        x = F.elu(self.conv2(x, ei))
        x = global_add_pool(x, batch)
        return self.lin(x)
