import math
from typing import Optional

import numpy as np


# Function to find the closest perfect square
# taking minimum steps to reach from a number
def getClosestPerfectSquare(n: int):
    while True:
        if math.sqrt(n) - math.floor(math.sqrt(n)) == 0:
            return n
        n += 1


def reshape_and_tile_images(array: np.ndarray, shape: tuple = (28, 28), n_cols: Optional[int] = None) -> np.ndarray:
    """
    Given an np array of shape (n_samples, sample_dim_flattened) it converts it to an big np array where all samples are
    arranged in a matrix like fashion.
    :param np.ndarray array:
    :param tuple shape:
    :param (optional) n_cols:
    :return:
    """
    n_samples, sample_flattened_dim = array.shape
    if n_cols is None:
        n_cols = int(math.sqrt(getClosestPerfectSquare(n_samples)))  # aiming for a square image grid
    n_rows = int(math.ceil(float(n_samples) / n_cols))

    def cell(i, j):
        array_index = i * n_cols + j
        if i * n_cols + j < n_samples:
            return array[array_index].reshape(*shape, order='C')
        else:
            return np.zeros(shape)

    return np.concatenate([
        np.concatenate([cell(i, j) for j in range(n_cols)], axis=1)
        for i in range(n_rows)], axis=0)
