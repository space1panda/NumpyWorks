import numpy as np


def sgd(lr, params, grads):
    params -= lr * grads
    return params

def decay_lr(num_epochs, rate, each, lr):
    return lr * (rate ** int(num_epochs/each))


