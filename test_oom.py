import torch
from monai.networks.nets import BasicUNet
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print("Device:", device)
model = BasicUNet(spatial_dims=3, in_channels=3, out_channels=1, features=(32, 64, 128, 256, 512, 1024)).to(device)
x = torch.randn(1, 3, 96, 96, 96).to(device)
try:
    out = model(x)
    loss = out.sum()
    loss.backward()
    print("Success")
except Exception as e:
    print("Error:", e)
