import torch
import torch.nn as nn
from torch.nn import functional as F
import copy
import math
from torch.nn import Conv1d
from structure_rank.Model.resnet import ResNet, ResNetBlock
from structure_rank.Model.attention import Pair2Pair, Templ_emb
from structure_rank.Model.IPA import Voxel
# from structure_rank.Model.GCN import GCNConv
from torch_geometric.nn import GCNConv, GATv2Conv, GAT
from torch_geometric.nn.pool import SAGPooling
from structure_rank.Model.feature_fusion import FeatureFusionMLP


class ModelRank(torch.nn.Module):
    def __init__(self,
                 onebody_size=74,
                 twobody_size=32,
                 num_update=1, abag=False,
                 n_module=3, n_layer=1,
                 d_pair=128,
                 n_head_pair=4, r_ff=2,
                 n_resblock=1, p_drop=0.1,
                 index_dim=6, out_index=64,
                 performer_L_opts=None, performer_N_opts=None):

        super(ModelRank, self).__init__()
        self.num_update = num_update
        self.abag = abag
        self.GATv2Conv = GATv2Conv(161 + 32, 16, heads=8, edge_dim=1, add_self_loops=False, dropout=0.25).jittable()
        self.GATv2Conv1 = GATv2Conv(16 * 8, 16, heads=8, edge_dim=1, add_self_loops=False, dropout=0.25).jittable()
        self.GATv2Conv2 = GATv2Conv(16 * 8, 16, heads=8, edge_dim=1, add_self_loops=False, dropout=0.25).jittable()
        self.GATv2Conv3 = GATv2Conv(16 * 8, 8, heads=8, edge_dim=1, add_self_loops=False, dropout=0.25).jittable()
        self.lin1 = nn.Linear(8 * 8, 1)

        self.add_module("conv1d_1", torch.nn.Conv1d(640, d_pair // 2, 1, padding=0, bias=True))
        self.add_module("conv2d_1", torch.nn.Conv2d(twobody_size + 16, d_pair, 1, padding=0, bias=True))

        # 融合 f2d + profile + entropy
        self.fusion = FeatureFusionMLP(profile_dim=36, entropy_dim=1, orientation_dim=6, hidden_dim=40, out_dim=16)
        self.add_module("conv2d_2", torch.nn.Conv2d(d_pair * 2, d_pair, 1, padding=0, bias=True))
        self.add_module("rep", torch.nn.Conv1d(512, d_pair // 2, 1, padding=0, bias=True))
        self.add_module("inorm_1", torch.nn.InstanceNorm2d(d_pair * 2, eps=1e-06, affine=True))
        self.add_module("inorm_2", torch.nn.InstanceNorm2d(d_pair, eps=1e-06, affine=True))
        self.add_module("sample", torch.nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True))
        self.pair_updata = _get_clones(Pair2Pair(n_layer=1,
                                                 n_att_head=n_head_pair,
                                                 n_feat=d_pair,
                                                 r_ff=r_ff,
                                                 p_drop=p_drop,
                                                 performer_L_opts=performer_L_opts
                                                 ), num_update)
        self.mask_conv2d = torch.nn.Conv2d(d_pair, 1, 1, padding=0, bias=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.bn1 = nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(128, 128)
        self.voxel = Voxel(20, 20)
        self.sag1 = SAGPooling(161 + 32, 0.5)
        #######################################
        # voro升维
        self.voro = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )
        #######################################

    def forward(self, data):
        # print("Forward pass of ModelRank")
        voro_features = data['voro_features']
        # print("voro_features:", voro_features.shape)

        voro_normal = data['voro_normal']
        # print("voro_normal:", voro_normal.shape)

        # 拼接 voro_features 和 voro_normal
        voro = torch.cat([voro_features, voro_normal], dim=1)  # (1, 5, 142)
        # print("voro (after concat):", voro.shape)

        # 变换形状以适应 nn.Linear
        voro = voro.squeeze(0).permute(1, 0)  # (142, 5)  (L, 5)
        # print("voro (before Linear):", voro.shape)

        # 通过 voro 层升维
        voro = self.voro(voro)  # (142, 64)
        # print("voro (after Linear):", voro.shape)

        # 调整形状回到 (1, 64, L)
        voro = voro.permute(1, 0).unsqueeze(0)  # (1, 64, 142)
        # print("voro (final shape):", voro.shape)
        
        
        if '_1d' not in data:
            raise KeyError("missing _1d feature")
        f1d = data['_1d']
            
        # print("f1d:", f1d.shape)
        feat = data['feat'].permute(0, 2, 1).to(torch.float32)
        # print("feat:", feat.shape)

        
        mpnn = data['mpnn'].to(torch.float32)
        if mpnn.ndim == 2:
            if mpnn.shape[1] != 32:
                raise ValueError(f"Unexpected shape: {tuple(mpnn.shape)}, expected (L, 32)")
            mpnn = mpnn.transpose(0, 1).unsqueeze(0)
        elif mpnn.ndim == 3 and mpnn.shape[-1] == 32:
            mpnn = mpnn.permute(0, 2, 1)
        elif mpnn.ndim == 3 and mpnn.shape[1] == 32:
            pass
        else:
            raise ValueError(f"Unexpected shape: {tuple(mpnn.shape)}, expected (*, L, 32)")
        # print("✅ mpnn shape", mpnn.shape)

        new_f1d = torch.cat((f1d, feat, voro, mpnn), dim=-2)
        # print("new_f1d shape", new_f1d.shape)
        node = new_f1d.squeeze(0)
        node = node.permute(1, 0)
        node = node.to(torch.float32)
        
        edge = data["adj"].squeeze().permute(1, 0).contiguous().long().to(node.device)
        
        x = self.sag1(node, edge)
        node = x[0]
        edge = x[1]
        x = F.gelu(self.GATv2Conv(node, edge))
        x = F.gelu(self.GATv2Conv1(x, edge))
        x = F.gelu(self.GATv2Conv2(x, edge))
        x = self.GATv2Conv3(x, edge)
        x = self.lin1(x)
        # score1d = torch.mean(x, dim=0)

        nres = f1d.shape[2]
        vidx = data["vidx"].squeeze(0)
        val = data["val"].squeeze(0)
        f3d = self.voxel(vidx, val, nres).permute(1, 0).unsqueeze(0)
        new_f3d = F.gelu(self._modules["conv1d_1"](f3d))
        temp1 = tile(new_f3d.unsqueeze(3), 3, nres)
        temp2 = tile(new_f3d.unsqueeze(2), 2, nres)

        f2d = data['_2d'].permute(0, 3, 1, 2)  # [B, 32, L, L]
        # print("f2d shape:", f2d.shape)
        profile = data['profile']               # [B, L, L, 36]
        entropy = data['entropy']               # [B, L, L, 1]
        mask_3d = data['mask_3d']    # [B, L, L, 1]
        af_orientation = data['af_orientation']
        
        # print("profile shape:", profile.shape)
        # print("entropy shape:", entropy.shape)
        # print("mask_3d shape:", mask_3d.shape)
        
        # 转换 mask
        mask_4d = mask_3d.float().permute(0, 3, 1, 2)  # [B, 1, L, L]
        # print("mask_4d shape:", mask_4d.shape)
        
        fused_feat = self.fusion(profile, entropy, af_orientation)  # [B, 16, L, L]
        # print("fused_feat shape:", fused_feat.shape)

        # 屏蔽无效区域
        fused_feat = fused_feat * mask_4d
        # print("fused_feat after masking shape:", fused_feat.shape)

        # 与 f2d 拼接
        f2d_fused = torch.cat([f2d, fused_feat], dim=1)  # [B, twobody_size+16, L, L]
        # print("f2d_fused after concat shape:", f2d_fused.shape)

        out_conv_f2d = self._modules["conv2d_1"](f2d_fused)
        #############################################
        pair = torch.cat([temp1, temp2, out_conv_f2d], dim=1)
        pair = F.gelu(self._modules["inorm_1"](pair))
        pair = self._modules["conv2d_2"](pair)
        pair = F.gelu(self._modules["inorm_2"](pair))
        pair = self._modules["sample"](pair)
        pair = pair.permute(0, 2, 3, 1)
        for i_m in range(self.num_update):
            pair = self.pair_updata[i_m](pair)
        pair = pair.permute(0, 3, 1, 2)
        mask = self.mask_conv2d(pair)
        product_tensor = x * mask
        sum_tensor = torch.mean(product_tensor, dim=(1, 2, 3))
        result_tensor = sum_tensor.reshape(1, 1)
        score = result_tensor
        return score


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# Calculates LDDT based on estogram
def calculate_LDDT(estogram, mask, center=7):
    # Get on the same device as indices
    device = estogram.device

    # Remove diagonal from calculation
    nres = mask.shape[-1]
    # torch.mul(a,b) 矩阵点乘，对应位置元素相乘
    # torch.ones((n,m)),生成n*m的矩阵，所  有元素都为1 # torch.eye(n),生成n*n的矩阵，对角线元素全为1，其他全为0
    mask = torch.mul(mask, torch.ones((nres, nres)).to(device) - torch.eye(nres).to(device))  # 将对角线元素置零
    masked = torch.mul(estogram, mask)
    #    print("mask: ", mask)
    #    print("estogram: ", estogram)
    #    print("masked: ", masked)

    p0 = (masked[center]).sum(axis=0)
    p1 = (masked[center - 1] + masked[center + 1]).sum(axis=0)
    p2 = (masked[center - 2] + masked[center + 2]).sum(axis=0)
    p3 = (masked[center - 3] + masked[center + 3]).sum(axis=0)
    p4 = mask.sum(axis=0)

    return 0.25 * (4.0 * p0 + 3.0 * p1 + 2.0 * p2 + p3) / p4


def tile(a, dim, n_tile):
    # Get on the same device as indices
    device = a.device

    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))

    order_index = (
        torch.arange(init_dim, device=device).unsqueeze(1)
        + init_dim * torch.arange(n_tile, device=device).unsqueeze(0)
    ).reshape(-1)
    return torch.index_select(a, dim, order_index)


# tf.scatter_nd-like function implemented with torch.scatter_add
def scatter_nd(indices, updates, shape):
    # Get on the same device as indices
    device = indices.device

    # Initialize empty array
    size = math.prod(shape)  # prod(x),x的各个元素的乘积：nres*24*24*24*self.num_restype
    out = torch.zeros(size).to(device)

    # Get flattened index (Calculation needs to be done in long to preserve indexing precision)
    temp = torch.as_tensor(
        [int(math.prod(shape[i + 1:])) for i in range(len(shape))],
        dtype=torch.long,
        device=device,
    )
    flattened_indices = torch.mul(indices.long(), temp).sum(dim=1)

    # Scatter_add
    out = out.scatter_add(0, flattened_indices, updates)

    # Reshape
    return out.view(shape)
