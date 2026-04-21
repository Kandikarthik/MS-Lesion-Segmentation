import torch.nn as nn
from monai.networks.nets import BasicUNet

def get_model():
    return BasicUNet(
        spatial_dims=3,
        in_channels=3,
        out_channels=1,
        features=(16, 32, 64, 128, 256, 32),
        act="LEAKYRELU",
        norm="INSTANCE",
        dropout=0.2
    )