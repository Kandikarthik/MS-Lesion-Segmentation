import sys
import os
import nibabel as nib

path = "data/raw/Mendeley/Patient-1/"
flair = os.path.join(path, "1-Flair.nii")
mask = os.path.join(path, "1-LesionSeg-Flair.nii")

f_img = nib.load(flair).get_fdata()
m_img = nib.load(mask).get_fdata()

print("Flair shape:", f_img.shape)
print("Mask shape:", m_img.shape)
