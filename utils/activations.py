import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    pass

def relu(z):
    pass

def leakyrelu(z):
    pass

def maxout(z):
    pass

def softmax(z):
    pass

"""activation derivatives"""

def dsigmoid(z):
    return sigmoid(z) *(1 - sigmoid(z))

def dtanh(dz):
    pass

def drelu(dz):
    pass

def dleakyrelu(dz):
    pass

def dmaxout(dz):
    pass

def dsotfmax(dz):
    pass
