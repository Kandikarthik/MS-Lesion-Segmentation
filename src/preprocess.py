import SimpleITK as sitk

def n4_bias_correction(image):
    mask = sitk.OtsuThreshold(image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    return corrector.Execute(image, mask)

def normalize(img):
    return (img - img.mean()) / (img.std() + 1e-5)