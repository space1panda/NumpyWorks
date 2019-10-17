import numpy as np


def batch_gen(x, y, bs):
    for mb in range(int(len(x)/bs)):
        yield x[mb*bs:mb*bs+bs], y[mb*bs:mb*bs+bs]

lin_formula_input = lambda x: np.linspace(1.0, 7.0, x)[:, np.newaxis]
lin_formula_output = lambda x: np.power(x, 0.5) - 0.5*np.sin(x) - np.power(x, 0.2) + 0.3*np.random.randn(len(x),1)