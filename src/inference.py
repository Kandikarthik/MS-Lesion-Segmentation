import sys
import torch
import nibabel as nib
from monai.inferers import sliding_window_inference
from model import get_model

def load_weights(weight_path):
    print(f"Loading model with weights from {weight_path}...")
    model = get_model()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def run_inference(model, device, image_tensor):
    print("Running sliding window inference...")
    with torch.no_grad():
        x = image_tensor.to(device)
        # Use an ROI size matching training
        pred_raw = sliding_window_inference(x, (96, 96, 96), 4, model)
        pred = (torch.sigmoid(pred_raw) > 0.5).float()
    return pred

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/inference.py <path_to_pth> <path_to_nifti_to_test>")
        print("Example: python src/inference.py models/best_model.pth path/to/image.nii.gz")
        sys.exit(1)

    weight_path = sys.argv[1]
    image_path = sys.argv[2]
    
    # Load Model
    model, device = load_weights(weight_path)
    
    # Note: inference requires the same 3 channels used in training (FLAIR, T1, T2)
    flair_path = image_path
    t1_path = flair_path.replace("FLAIR", "T1")
    t2_path = flair_path.replace("FLAIR", "T2")

    from monai.transforms import LoadImaged, EnsureChannelFirstd, ResampleToMatchd, ConcatItemsd, ScaleIntensityd, Compose
    
    transform = Compose([
        LoadImaged(keys=["flair", "t1", "t2"]),
        EnsureChannelFirstd(keys=["flair", "t1", "t2"]),
        ResampleToMatchd(keys=["t1", "t2"], key_dst="flair"),
        ConcatItemsd(keys=["flair", "t1", "t2"], name="image"),
        ScaleIntensityd(keys=["image"])
    ])
    
    print(f"Loading images:\n- {flair_path}\n- {t1_path}\n- {t2_path}")
    
    data = {"flair": flair_path, "t1": t1_path, "t2": t2_path}
    transformed = transform(data)
    img = transformed["image"].unsqueeze(0) # add batch dimension
    
    pred = run_inference(model, device, img)
    output_path = "prediction_mask.nii.gz"
    
    # Save output
    pred_np = pred.squeeze().cpu().numpy()
    nib.save(nib.Nifti1Image(pred_np, affine=None), output_path)
    print(f"✅ Prediction saved to {output_path}")