import numpy as np
from scipy.ndimage import label

def remove_small_components(mask, min_size=8):
    labeled, num = label(mask)
    sizes = np.bincount(labeled.ravel())

    for i in range(1, num+1):
        if sizes[i] < min_size:
            mask[labeled == i] = 0

    return mask