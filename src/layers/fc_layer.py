import torch.nn as nn

class FCLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return self.dropout(x)
