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


