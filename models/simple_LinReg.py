import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from utils import mse, linearparams_init


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
            if _ % 100 ==0:
                print(f'Loss on iter {_}: {loss}')
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

data_x = np.array([np.linspace(1,20,100).flatten(),np.linspace(1,20,100).flatten()]).T
print(data_x.shape)
data_y = np.sin(data_x) + 0.1*np.power(data_x,2) + 0.5*np.random.randn(100,2)
plt.scatter(data_x, data_y)
plt.show()
data_x = data_x / np.max(data_x)
order = np.random.permutation(len(data_x))
portion = 20
test_x = data_x[order[:portion]]
test_y = data_y[order[:portion]]
train_x = data_x[order[portion:]]
train_y = data_y[order[portion:]]

w,b = linearparams_init(data_x.shape[-1])
model = SimpleLinearRegression(w,b)
params, final_loss = model.optimize(x=train_x, y=train_y, num_iters=2500)

plt.scatter(train_x, train_y)
plt.plot(train_x, model.predict(train_x, params[0], params[1]))
plt.show()

h = model.predict(test_x, params[0], params[1])

plt.scatter(test_x, test_y)
plt.plot(test_x, model.predict(test_x, params[0], params[1]))
plt.show()
print(f'Loss on training set: {final_loss}, Loss on test_set: {mse(test_y, h)}')