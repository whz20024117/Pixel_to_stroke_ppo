import numpy as np
from ppo.config import config


def scale(x, old_min, old_max, new_min=0, new_max=1):
    x = np.array(x)
    return new_min + ((x - old_min)*(new_max - new_min))/(old_max - old_min)


def move_point(x, y):
    x_old, y_old = x, y
    x = x_old + np.random.randint(-3, 4)
    y = y_old + np.random.randint(-3, 4)

    while True:
        if x <= 0 or x >= config['STATE_DIM'][0] or x == x_old:
            x = x_old + np.random.randint(-1, 1)
        else:
            break

    while True:
        if y <= 0 or y >= config['STATE_DIM'][1] or y == y_old:
            y = y_old + np.random.randint(-1, 1)
        else:
            break

    return x, y
