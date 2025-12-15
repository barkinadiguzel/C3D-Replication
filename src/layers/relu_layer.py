import torch.nn as nn

class ReLULayer(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        return self.relu(x)
