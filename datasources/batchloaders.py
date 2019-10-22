from utils.data_transforming import batch_gen
from datasources.datagen import *

class LinBatchLoader:

    def __init__(self, x, y, bs, valid_split):

        print(">>>Batching train and validation sets...")

        self.bs = bs
        self._x = x
        self._y = y

        self.train_x = self._x[:int((len(self._y) - len(self._y) * valid_split))]
        self.train_y = self._y[:int((len(self._y) - len(self._y) * valid_split))]

        self.valid_x = self._x[int((len(self._y) - len(self._y) * valid_split)):]
        self.valid_y = self._y[int((len(self._y) - len(self._y) * valid_split)):]

    def getloaders(self):
        return batch_gen(self.train_x, self.train_y, self.bs), batch_gen(self.valid_x,
                                                                               self.valid_y, self.bs)