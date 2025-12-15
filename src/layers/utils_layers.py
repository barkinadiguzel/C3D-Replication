import torch.nn as nn

def init_c3d_weights(module):
    if isinstance(module, nn.Conv3d):
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, 0, 0.01)
        nn.init.constant_(module.bias, 0)
