import numpy as np


def sgd(lr, params, grads):
    params -= lr * grads
    return params

def sgd_v2(lr, params, grads):
    params['w'] -= lr * grads['grad_w']
    params['b'] -= lr* grads['grad_b']
    return params

def decay_lr(num_epochs, rate, each, lr):
    return lr * (rate ** int(num_epochs/each))

def grad_cost_w(y, y_estimate, x):
    return -(1.0/len(y)) * (y.flatten()- y_estimate).dot(x)

def grad_cost_b(y, y_estimate):
    return -(1.0/len(y)) * np.sum(y.flatten()- y_estimate)



