import os
from glob import glob
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    ScaleIntensityd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d
)
from monai.data import Dataset
from monai.inferers import sliding_window_inference

from model import get_model
from loss import get_loss
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


# 🔷 Dice Score
def dice_score(pred, gt):
    inter = (pred * gt).sum()
    return (2 * inter) / (pred.sum() + gt.sum() + 1e-5)


# 🔷 Load MSLesSeg dataset
def load_data():
    data = []

    for flair in glob("data/raw/MSLesSeg/**/*FLAIR*.nii*", recursive=True):
        t1 = flair.replace("FLAIR", "T1")
        t2 = flair.replace("FLAIR", "T2")
        mask = flair.replace("FLAIR", "MASK")

        if os.path.exists(t1) and os.path.exists(t2) and os.path.exists(mask):
            data.append({
                "image": [flair, t1, t2],
                "label": mask
            })

    print(f"✅ Total samples loaded: {len(data)}")
    return data


# 🔷 Transforms
def get_transforms(train=True):

    t = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
    ]

    if train:
        t += [
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=3, neg=1, num_samples=1
            ),
            RandFlipd(keys=["image", "label"], prob=0.5),
            RandRotate90d(keys=["image", "label"], prob=0.5),
        ]

    return Compose(t)


# 🔷 Training function
def train(train_loader, val_loader):

    model = get_model().to(device)

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=25, T_mult=2
    )

    loss_fn = get_loss("ce")
    best_dice = 0

    print("🔥 Training started...")

    for epoch in range(200):

        # 🔁 Switch loss at epoch 150
        if epoch == 150:
            print("🔁 Switching to DiceFocalLoss")
            loss_fn = get_loss("focal")
            for g in optimizer.param_groups:
                g["lr"] = 5e-5

        model.train()
        total_loss = 0

        for batch in train_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)

            optimizer.zero_grad()
            out = model(x)

            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        # 🔍 Validation
        model.eval()
        total_dice = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device)
                y = batch["label"].to(device)

                pred_raw = sliding_window_inference(
                    x, (96, 96, 96), 4, model
                )

                best = 0
                for t in [0.1, 0.2, 0.3, 0.5]:
                    pred = (torch.sigmoid(pred_raw) > t).float()
                    d = dice_score(pred, y)

                    if d > best:
                        best = d

                total_dice += best.item()

        val_dice = total_dice / len(val_loader)

        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Dice: {val_dice:.4f}")

        # 💾 Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/best_model.pth")
            print("✅ Best model saved!")

    print("🔥 Final Best Dice:", best_dice)


# 🔷 MAIN EXECUTION
if __name__ == "__main__":

    print("🚀 Script started")

    data = load_data()

    if len(data) == 0:
        print("❌ No data found. Check dataset path!")
        exit()

    # Split dataset
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    # Create datasets
    train_ds = Dataset(data=train_data, transform=get_transforms(True))
    val_ds = Dataset(data=val_data, transform=get_transforms(False))

    from monai.data import list_data_collate

    # Dataloaders
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, collate_fn=list_data_collate
    )
    val_loader = DataLoader(val_ds, batch_size=1)

    # Start training
    train(train_loader, val_loader)