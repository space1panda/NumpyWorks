import numpy as np


def mseloss(Y, activation):
    return np.power(np.subtract(Y.flatten(),activation), 2).mean()