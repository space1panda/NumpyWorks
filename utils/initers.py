import numpy as np


def init_lin(num_features):
    return np.random.randn(num_features)*0.01

def init_lin_w_bias(num_features):
    return {'w':np.random.randn(num_features)*0.01, 'b':0}
