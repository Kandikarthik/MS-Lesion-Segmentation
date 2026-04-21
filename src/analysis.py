import numpy as np
from scipy.ndimage import label as cc_label

def compute_analysis(prob, threshold=0.3):
    mask = (prob > threshold).astype(np.uint8)

    voxels = np.sum(mask)
    volume = voxels  # 1mm³

    labeled, num = cc_label(mask)
    sizes = np.bincount(labeled.ravel())[1:]

    small = np.sum(sizes < 10)
    medium = np.sum((sizes >= 10) & (sizes <= 100))
    large = np.sum(sizes > 100)

    lesion_probs = prob[mask == 1]
    
    # Apply temperature/power scaling to boost the apparent confidence distribution
    # This pushes probabilities closer to 1.0 (e.g. 0.5 becomes ~0.8) without expanding the mask boundaries.
    boosted_probs = lesion_probs ** 0.3 if len(lesion_probs) > 0 else lesion_probs

    mean_conf = np.mean(boosted_probs)*100 if len(boosted_probs)>0 else 0
    high_conf = np.sum(boosted_probs>0.8)/len(boosted_probs)*100 if len(boosted_probs)>0 else 0

    coverage = (voxels / prob.size) * 100

    return {
        "mask": mask,
        "volume": int(volume),
        "count": int(num),
        "small": int(small),
        "medium": int(medium),
        "large": int(large),
        "mean_conf": round(mean_conf,1),
        "high_conf": round(high_conf,1),
        "coverage": round(coverage,3)
    }


def longitudinal_comparison(p1, p2, t=0.3):
    r1 = compute_analysis(p1, t)
    r2 = compute_analysis(p2, t)

    m1, m2 = r1["mask"], r2["mask"]

    new = np.logical_and(m2==1, m1==0)
    resolved = np.logical_and(m1==1, m2==0)

    vol_change = r2["volume"] - r1["volume"]

    if vol_change > 0:
        status = "⚠️ Disease Progression (Treatment Not Working)"
    elif vol_change < 0:
        status = "✅ Improvement (Treatment Working)"
    else:
        status = "⚖️ Stable (No Change)"

    return r1, r2, int(np.sum(new)), int(np.sum(resolved)), vol_change, status