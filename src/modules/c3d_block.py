import torch.nn as nn
from src.layers.conv3d_layer import Conv3DLayer
from src.layers.relu_layer import ReLULayer
from src.layers.pool3d_layer import Pool3DLayer

class C3DBlock(nn.Module):
    def __init__(self, in_c, out_c, pool=False, pool_kernel=(2,2,2), pool_stride=(2,2,2)):
        super().__init__()

        self.conv = Conv3DLayer(in_c, out_c)
        self.relu = ReLULayer()
        self.pool = Pool3DLayer(pool_kernel, pool_stride) if pool else None

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        if self.pool:
            x = self.pool(x)
        return x
