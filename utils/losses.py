import numpy as np


def mseloss(Y, activation):
    return np.power(np.subtract(Y.flatten(),activation), 2).mean()

def logloss(Y, activation):
    return (-Y * np.log(activation) - (1 - Y) * np.log(1 - activation)).mean()
