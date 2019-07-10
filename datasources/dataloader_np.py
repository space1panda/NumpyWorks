from datasources.chardatasource import CharDatasource
from utils.utils_np import chunks

class RNNDataLoader(CharDatasource):

    def __init__(self, batch_size, split_ratio, **kwargs):

        """ CharDatasource is used as parent object.
        """

        self._batch_size = batch_size
        super().__init__(**kwargs)

        """Splitting dataset into train/test
        """
        train_set = (self._tokens[:int(split_ratio*len(self._tokens))],
                     self._targets[:int(split_ratio*len(self._targets))])
        valid_set = (self._tokens[int(split_ratio*len(self._tokens)):],
                    self._targets[int(split_ratio*len(self._targets)):])

        """Even though we are having pretty small dataset, it's a 
        good practice to use generators instead of keeping
        additional data arrays to free up memory
        """

        train_x_loader = chunks(train_set[0], self._batch_size)
        train_y_loader = chunks(train_set[1], self._batch_size)
        valid_x_loader = chunks(valid_set[0], self._batch_size)
        valid_y_loader = chunks(valid_set[1], self._batch_size)

        self._batches_train = []
        self._batches_valid = []

        while True:
            try:
                x_train, y_train = next(train_x_loader), next(train_y_loader)
                x_valid, y_valid = next(valid_x_loader), next(valid_y_loader)
                if len(x_train) == self._batch_size:
                    self._batches_train.append((x_train, y_train))
                if len(x_valid) == self._batch_size:
                    self._batches_valid.append((x_valid, y_valid))


            except StopIteration:
                break


test = RNNDataLoader(path="/home/yegor/Desktop/projects/MLProjects/assets/npl_lang_models/test.txt",
                     seq_len=10,batch_size=5,split_ratio=0.8)
print(test._tokens[0])