import torch, torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_add_pool

class Net(torch.nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=12):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.lin   = torch.nn.Linear(hidden, out_dim)
    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, ei))
        x = F.relu(self.conv2(x, ei))
        x = global_add_pool(x, batch)
        return self.lin(x)
