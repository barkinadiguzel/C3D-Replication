import torch.nn as nn

class Pool3DLayer(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.pool = nn.MaxPool3d(
            kernel_size=kernel_size,
            stride=stride
        )

    def forward(self, x):
        return self.pool(x)
