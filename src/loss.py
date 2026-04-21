import torch.nn as nn
def get_loss(loss_name):
    if loss_name == "ce":
        return nn.BCEWithLogitsLoss()
    elif loss_name == "focal":
        try:
            from monai.losses import FocalLoss
            return FocalLoss()
        except ImportError:
            return nn.BCEWithLogitsLoss() # Fallback
    else:
        raise ValueError("Unknown loss: " + loss_name)
