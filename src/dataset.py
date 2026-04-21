from monai.transforms import *
from monai.data import Dataset

def get_transforms(train=True):
    t = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"])
    ]

    if train:
        t += [
            RandCropByPosNegLabeld(
                keys=["image","label"],
                label_key="label",
                spatial_size=(96,96,96),
                pos=3, neg=1, num_samples=4
            ),
            RandFlipd(keys=["image","label"], prob=0.5),
            RandRotate90d(keys=["image","label"], prob=0.5),
        ]

    return Compose(t)

def get_dataset(data_dicts, train=True):
    return Dataset(data=data_dicts, transform=get_transforms(train))