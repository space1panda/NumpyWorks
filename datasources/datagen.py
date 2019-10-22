import numpy as np
import matplotlib.pyplot as plt


class GenericDataGenerator:
    def __init__(self, ds_len, num_features, test_split):
        """
        :param ds_len: length of the generated dataset (number of rows)
        :param num_features: number of features (depth of the dataset)
        :param test_split: ratio at which test set is cut from data corpus
        """
        self.ds_len = ds_len
        self.num_features = num_features
        self.test_split = test_split
        mask = np.zeros(shape=(ds_len, num_features+1))
        self._x = mask[:,:-1]
        self._y = mask[:,-1]

    def __getitem__(self, item):
        return self._x[item], self._y[item]

    def __len__(self):
        return len(self._x)


class PolynomLinDataGenerator(GenericDataGenerator):

    def __init__(self, ds_len, num_features, test_split):
        super().__init__(ds_len, num_features, test_split)
        print(">>>Generating linear dataset...")
        self._x = np.linspace(1.0, 10.0, self.ds_len)[:, np.newaxis]
        self._y = np.power(self._x, 1.5) + 15*np.sin(self._x) - 10*np.cos(self._x) + 5*np.random.randn(self.ds_len,1)
        self._x /= self._x.max()
        self._allx = np.power(self._x, range(self.num_features))
        order = np.random.permutation(self.ds_len)
        self._x = self._allx[order]
        self._y = self._y[order]
        self.test_x = self._x[int(self.ds_len - self.ds_len * self.test_split):]
        self.test_y = self._y[int(self.ds_len - self.ds_len * self.test_split):]
        self._x = self._x[:int(self.ds_len-self.ds_len*self.test_split)]
        self._y = self._y[:int(self.ds_len-self.ds_len*self.test_split)]


class BinClassDataGenerator(GenericDataGenerator):
    def __init__(self, ds_len,  test_split, class_split_factor, data_dist, num_features=2):
        """
        :param class_split_factor: split percentage of 2 classes
        :param data_dist: dataset breadth across X-axis
        """
        super().__init__(ds_len, num_features, test_split)
        print(">>>Generating binary class dataset...")
        self.class_split_factor = class_split_factor
        self.data_dist = data_dist
        self._x[:, 0] = np.arange(ds_len, dtype=float) + self.data_dist * np.random.randn(ds_len)
        self._x[:, 1] = -1*np.arange(ds_len, dtype=float) - 20 * np.random.randn(ds_len) + np.sin(self._x[:, 0])
        self._x = self._x / self._x.max()
        self._y[int(self.ds_len * self.class_split_factor):] = 1
        order = np.random.permutation(self.ds_len)
        self._x = self._x[order]
        self._y = self._y[order]
        self.test_x = self._x[int(self.ds_len - self.ds_len * self.test_split):]
        self.test_y = self._y[int(self.ds_len - self.ds_len * self.test_split):]
        self._x = self._x[:int(self.ds_len - self.ds_len * self.test_split)]
        self._y = self._y[:int(self.ds_len - self.ds_len * self.test_split)]

