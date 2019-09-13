import numpy as np


def scale(x, old_min, old_max, new_min=0, new_max=1):
    x = np.array(x)
    return new_min + ((x - old_min)*(new_max - new_min))/(old_max - old_min)
