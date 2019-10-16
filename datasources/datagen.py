import numpy as np
from utils.data_transforming import lin_formula_input, lin_formula_output

class LinearDataGenerator:

    def __init__(self, ds_len, poly_order, test_split):
        print(">>>Generating linear dataset...")
        data_x = lin_formula_input(ds_len)
        self.data_y = lin_formula_output(data_x)
        data_x = data_x / np.max(data_x)
        self.data_x = np.power(data_x, range(poly_order))
        self.test_x = self.data_x[int(ds_len - ds_len * test_split):]
        self.test_y = self.data_y[int(ds_len - ds_len * test_split):]
        x = self.data_x[:int(ds_len - ds_len * test_split)]
        y = self.data_y[:int(ds_len - ds_len * test_split)]
        order = np.random.permutation(len(x))
        self._x = x[order]
        self._y = y[order]

    def __getitem__(self, item):
        return self._x[item], self._y[item]

    def __len__(self):
        return len(self._x)

    def gettest(self):
        return self.test_x, self.test_y


class BinaryClassDataGenerator:
    def __init__(self, ds_len, class_split_factor, test_split, data_dist, class_mix_factor):
        print(">>>Generating binary class dataset...")
        gen_data = np.zeros(shape=(ds_len, 3))
        gen_data[:, 0] = np.arange(ds_len, dtype=float) + data_dist * np.random.randn(ds_len)
        gen_data[:, 1] = np.arange(ds_len, dtype=float) - class_mix_factor * np.random.randn(ds_len)
        gen_data[:, 0:-1] = gen_data[:, 0:2] / gen_data[:, 0:2].max()
        gen_data[:, -1][int(ds_len * class_split_factor):] = 1
        self.data = gen_data
        order = np.random.permutation(len(self.data))
        self._x = self.data[order][:, 0:2]
        self._y = self.data[order][:, 2]
        self.test_x = self._x[int(ds_len * test_split):]
        self.test_y = self._y[int(ds_len * test_split):]
        self._x = self._x[:int(ds_len * test_split)]
        self._y = self._y[:int(ds_len * test_split)]

    def __getitem__(self, item):
        return self._x[item], self._y[item]

    def __len__(self):
        return len(self._x)

    def gettest(self):
        return self.test_x, self.test_y