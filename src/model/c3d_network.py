import torch.nn as nn
from src.blocks.c3d_block import C3DBlock
from src.blocks.feature_head import FeatureHead

class C3DNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # ---- C3D backbone ----
        self.block1 = C3DBlock(3, 64, pool=True, pool_kernel=(1,2,2), pool_stride=(1,2,2))
        self.block2 = C3DBlock(64, 128, pool=True)
        self.block3a = C3DBlock(128, 256)
        self.block3b = C3DBlock(256, 256, pool=True)
        self.block4a = C3DBlock(256, 512)
        self.block4b = C3DBlock(512, 512, pool=True)
        self.block5a = C3DBlock(512, 512)
        self.block5b = C3DBlock(512, 512, pool=True)

        # ---- Feature head (fc6 + fc7) ----
        # 4096 entries: 512 * 1 * 4 * 4 (assuming 16 frames)
        self.head = FeatureHead(in_dim=8192)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3a(x)
        x = self.block3b(x)
        x = self.block4a(x)
        x = self.block4b(x)
        x = self.block5a(x)
        x = self.block5b(x)

        x = self.head(x)
        return x
