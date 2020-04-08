import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()  # for numerical stability
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()


def sigmoid(x: float) -> float:
    sigmoid_range = 34.538776394910684

    # for numerical stability
    if x <= -sigmoid_range:
        return 1e-15
    if x >= sigmoid_range:
        return 1.0 - 1e-15

    return 1.0 / (1.0 + np.exp(-x))
