import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class ResidualGATv2Layer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=8, edge_dim=1, dropout=0.25):
        super().__init__()
        self.gat = GATv2Conv(
            in_channels, out_channels, heads=heads,
            edge_dim=edge_dim, add_self_loops=False,
            dropout=dropout
        ).jittable()

        self.out_dim = out_channels * heads
        self.use_proj = in_channels != self.out_dim
        self.proj = nn.Linear(in_channels, self.out_dim) if self.use_proj else nn.Identity()

    def forward(self, x, edge_index):
        x_gat = self.gat(x, edge_index)
        x_res = self.proj(x)
        return F.gelu(x_gat + x_res)

class GATv2ResidualNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = ResidualGATv2Layer(161 + 32, 16, heads=8)     # → (B, 128)
        self.layer2 = ResidualGATv2Layer(16 * 8, 16, heads=8)       # → (B, 128)
        self.layer3 = ResidualGATv2Layer(16 * 8, 16, heads=8)       # → (B, 128)
        self.layer4 = ResidualGATv2Layer(16 * 8, 8, heads=8)        # → (B, 64)
        self.lin1 = nn.Linear(8 * 8, 32)

    def forward(self, x, edge_index):
        x = self.layer1(x, edge_index)
        x = self.layer2(x, edge_index)
        x = self.layer3(x, edge_index)
        x = self.layer4(x, edge_index)
        x = self.lin1(x)
        return x
