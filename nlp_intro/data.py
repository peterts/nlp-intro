import numpy as np


def create_2d_2class_data(x_range, y_range, n_points, model, bias):
    x = np.random.uniform(*x_range, n_points)
    y = np.random.uniform(*y_range, n_points)
    features = np.asarray([x, y]).T
    bias = np.random.normal(0, bias, n_points)
    label = (y > (model(x) + bias)).astype(int)
    return features, label

