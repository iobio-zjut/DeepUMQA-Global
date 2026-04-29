import torch
import torch.nn as nn

class FeatureFusionMLP(nn.Module):
    def __init__(self, profile_dim=36, entropy_dim=1, orientation_dim=6, hidden_dim=40, out_dim=16):
        super(FeatureFusionMLP, self).__init__()

        # Learnable scalar weights
        self.profile_weight = nn.Parameter(torch.tensor(0.1))
        self.entropy_weight = nn.Parameter(torch.tensor(0.1))
        self.orientation_weight = nn.Parameter(torch.tensor(0.1))

        # 升维 entropy（投影为 4 维）
        self.entropy_proj = nn.Conv2d(entropy_dim, 4, kernel_size=1)

        # profile + entropy_proj + orientation → MLP
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(profile_dim + 4 + orientation_dim, hidden_dim, kernel_size=1),
            nn.InstanceNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1)
        )

    def forward(self, profile, entropy, orientation):
        """
        profile:     [B, L, L, 36]
        entropy:     [B, L, L, 1]
        orientation: [B, L, L, 6]
        return: fused: [B, out_dim, L, L]
        """
        # 通道放前 + 权重缩放
        profile_scaled = profile.permute(0, 3, 1, 2) * self.profile_weight      # [B, 36, L, L]
        entropy_scaled = entropy.permute(0, 3, 1, 2) * self.entropy_weight      # [B, 1, L, L]
        orientation_scaled = orientation.permute(0, 3, 1, 2) * self.orientation_weight  # [B, 6, L, L]

        entropy_proj = self.entropy_proj(entropy_scaled)  # [B, 4, L, L]

        # 拼接：36 + 4 + 6 = 46
        x = torch.cat([profile_scaled, entropy_proj, orientation_scaled], dim=1)  # [B, 46, L, L]

        fused = self.fusion_proj(x)  # [B, out_dim, L, L]
        return fused
