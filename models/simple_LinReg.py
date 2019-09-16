import numpy as np
from matplotlib import pyplot as plt
from utils import mse, simple_params_init


class SimpleLinearRegression:
    def __init__(self, *params):
        self.w = params[0]
        self.b = params[1]

    def linear(self, x):
        h = x.dot(self.w) + self.b
        return h

    def optimize(self, x, y, num_iters, alpha=0.03):
        losses = []
        N = float(len(y))
        for _ in range(num_iters):
            h = self.linear(x)
            error = y.flatten() - h
            loss = mse(y, h)
            losses.append(loss)
            gradient_w = -(2 / N) * np.sum(error.dot(x))
            gradient_b = -(2 / N) * np.sum(error)
            self.w -= alpha * gradient_w
            self.b -= alpha * gradient_b
        plt.plot(range(num_iters), losses)
        plt.show()
        return (self.w, self.b), losses[-1]

    def predict(self, x, *params):
        return x.dot(params[0]) + params[1]

data_x = np.random.randint(10, size=100)[:, np.newaxis]
data_y = np.sin(data_x) + 0.1*np.power(data_x,2) + 0.5*np.random.randn(100,1)
data_x = data_x / np.max(data_x)
order = np.random.permutation(len(data_x))
portion = 20
test_x = data_x[order[:portion]]
test_y = data_y[order[:portion]]
train_x = data_x[order[portion:]]
train_y = data_y[order[portion:]]

model = SimpleLinearRegression(simple_params_init())
params, _ = model.optimize(x=train_x, y=train_y, num_iters=2000)

h = model.predict(test_x, params[0], params[1])
print(mse(test_y, h))

