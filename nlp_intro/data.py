import numpy as np


def draw_uniform_data(n_points, *ranges):
    return np.asarray([np.random.uniform(*r, n_points) for r in ranges]).T
