import numpy as np


data_x = np.linspace(1.0, 10.0, 100)[:, np.newaxis]
data_y = np.sin(data_x) + 0.1*np.power(data_x,2) + 0.5*np.random.randn(100,1)
data_x /= np.max(data_x)
print(data_x[0], data_y[0])


def test():
    return [0,1,2]

print(test()[0])