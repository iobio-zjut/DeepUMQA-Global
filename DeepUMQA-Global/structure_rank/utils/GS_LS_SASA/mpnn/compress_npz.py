# compress_npz_CASP16.py

import sys
import os
import numpy as np
import torch
import torch.nn as nn

# ======= 模型定义 =======
class LinearReduce256to64(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(256, 64)

    def forward(self, x):  # x: (L*W, 256)
        return self.linear(x)

def create_model2(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 32)
    )

# ======= 主压缩函数 =======
def compress_npz(npz_path):
    try:
        data = np.load(npz_path)
        if "h_ES_encoder" not in data:
            print(f"⚠️ Skipping {npz_path}, no h_ES_encoder found.")
            return

        h = data["h_ES_encoder"]  # shape: (1, L, 48, 256) or (L, 48, 256)
        h = np.squeeze(h)

        if h.ndim != 3 or h.shape[2] != 256:
            print(f"❌ Skipping {npz_path}, unexpected shape = {h.shape}")
            return

        L, W, C = h.shape
        h_tensor = torch.tensor(h, dtype=torch.float32).view(-1, C)  # (L*W, 256)

        model1 = LinearReduce256to64()
        model2 = create_model2(64 * W)
        model1.eval()
        model2.eval()

        with torch.no_grad():
            h_64 = model1(h_tensor)                  # (L*W, 64)
            h_64 = h_64.view(L, W, 64)               # (L, 48, 64)
            h_merge = h_64.reshape(L, -1)            # (L, 48*64)
            h_final = model2(h_merge)                # (L, 32)

        np.savez_compressed(npz_path, h_ES_encoder=h_final.numpy())
        print(f"✅ Compressed and saved: {npz_path}")

    except Exception as e:
        print(f"❌ Error processing {npz_path}: {e}")

# ======= 命令行入口 =======
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compress_npz_CASP16.py <npz_file>")
        sys.exit(1)
    compress_npz(sys.argv[1])
