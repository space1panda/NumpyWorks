import numpy as np


def batch_gen(x, y, bs):
    for mb in range(int(len(x)/bs)):
        yield x[mb*bs:mb*bs+bs], y[mb*bs:mb*bs+bs]