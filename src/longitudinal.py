import numpy as np

def progression_analysis(m1, m2):

    new = np.logical_and(m2==1, m1==0)
    resolved = np.logical_and(m1==1, m2==0)

    change = np.sum(m2) - np.sum(m1)

    if change > 0:
        status = "Progression"
    elif change < 0:
        status = "Improvement"
    else:
        status = "Stable"

    return new.sum(), resolved.sum(), change, status