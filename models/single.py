from utils.activations import *
from utils.optimizers import *

class LinearRegression:
    def __init__(self, params):
        self.params = params
        print(">>>Building LinReg model...")

    def forward(self, x):
        return x.dot(self.params).flatten()

    def backwards(self, x, y, y_estimate):
        return -(1.0 / len(y)) * (y.flatten() - y_estimate).dot(x)


class LogisticRegression:
    def __init__(self, params):
        self.params = params
        print(">>>Building 1-neuron LogReg model...")

    def forward(self, x):
        linear = x.dot(self.params['w']).flatten() + self.params['b']
        return sigmoid(linear)

    def backwards(self, x, y, y_estimate):
        grads = {}
        grads.update({'grad_w': grad_cost_w(y, y_estimate, x),
                      'grad_b': grad_cost_b(y, y_estimate)})
        return grads
