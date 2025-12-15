import torch.nn as nn
from src.layers.flatten_layer import FlattenLayer
from src.layers.fc_layer import FCLayer
from src.layers.pool3d_layer import Pool3DLayer

class FeatureHead(nn.Module):
    def __init__(self, in_dim, use_pool=False):
        super().__init__()

        self.pool = Pool3DLayer((2,2,2),(2,2,2)) if use_pool else None
        self.flatten = FlattenLayer()
        self.fc1 = FCLayer(in_dim, 4096)
        self.fc2 = FCLayer(4096, 4096)

    def forward(self, x):
        if self.pool:
            x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
