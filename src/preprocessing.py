import numpy as np
from scipy.ndimage import label


def remove_small_objects(mask, min_size):
    labeled, num = label(mask)

    output = np.zeros_like(mask)

    for i in range(1, num+1):
        component = (labeled == i)
        if component.sum() >= min_size:
            output[component] = 1

    return output