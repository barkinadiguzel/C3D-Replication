# Input video clip
NUM_FRAMES = 16          # C3D paper default
INPUT_CHANNELS = 3      # RGB
FRAME_HEIGHT = 112
FRAME_WIDTH = 112

# 3D convolution
CONV_KERNEL = (3, 3, 3)
CONV_STRIDE = (1, 1, 1)
CONV_PADDING = (1, 1, 1)

# Pooling
POOL_KERNEL = (2, 2, 2)
POOL_STRIDE = (2, 2, 2)

# First pooling keeps time
POOL1_KERNEL = (1, 2, 2)
POOL1_STRIDE = (1, 2, 2)

# Fully connected layers
FC_DIM = 4096
DROPOUT = 0.5

# Feature dimension (fc7 output)
FEATURE_DIM = 4096

# Weight initialization
WEIGHT_INIT = "c3d"

# Notes:
# - No classifier head
# - No training config
# - This replication is for feature extraction only
