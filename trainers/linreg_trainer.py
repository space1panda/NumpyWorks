import numpy as np
import pickle
from utils.losses import *
from utils.optimizers import *
from utils.misc import *


class GenericTrainer:
    def __init__(self, dataloader, model, num_epochs, lr, each, decay_rate):
        self.num_epochs = num_epochs
        self.each = each
        self.decay_rate = decay_rate
        self.model = model
        self.dataloader = dataloader
        self.lr = lr

    def train_epoch(self):
        raise NotImplementedError("Implement parameter update method for all train batches")

    def valid_epoch(self):
        raise NotImplementedError("Implement validation method for all valid batches")

    def train(self):
        train_losses = []
        valid_losses = []
        for e in range(self.num_epochs):
            train_epoch_loss = self.train_epoch()
            train_losses.append(train_epoch_loss)
            valid_epoch_loss = self.valid_epoch()
            valid_losses.append(valid_epoch_loss)
            self.lr = decay_lr(self.num_epochs, self.decay_rate, self.each, self.lr)

            if e % self.each == 0:
                print(f"Train error on epoch {e} : {train_epoch_loss}")
                print(f"Valid error on epoch {e} : {valid_epoch_loss}")
                if self.lr != decay_lr(self.num_epochs, self.decay_rate, self.each, self.lr):
                    print(f"Adjusted lr: {self.lr}")

        return self.model.params, train_losses, valid_losses

    def save_model(self, save_path):
        print(f">>> Saving trained model to {save_path}")
        pickle.dump(self.model, open(save_path, "wb"))

class LinearTrainer(GenericTrainer):
    def __init__(self, dataloader, model, num_epochs, lr, each, decay_rate):
        super().__init__(dataloader, model, num_epochs, lr, each, decay_rate)

    def train_epoch(self):
        train_batch, _ = self.dataloader.getloaders()
        train_epoch_loss = []
        while True:
            try:
                x, y = next(train_batch)
                y_estimate = self.model.forward(x)
                mini_loss = mseloss(y, y_estimate)
                train_epoch_loss.append(mini_loss)
                gradient = self.model.backwards(x, y, y_estimate)
                self.model.params = sgd(self.lr, self.model.params, gradient)
            except StopIteration:
                break
        return np.sum(train_epoch_loss) / len(train_epoch_loss)

    def valid_epoch(self):
        valid_epoch_loss = []
        _, valid_loader = self.dataloader.getloaders()
        while True:
            try:
                x, y = next(valid_loader)
                y_estimate = self.model.forward(x)
                mini_loss = mseloss(y, y_estimate)
                valid_epoch_loss.append(mini_loss)
            except StopIteration:
                break
        return np.sum(valid_epoch_loss) / len(valid_epoch_loss)

    def test_model(self, test_set):
        y_estimate = self.model.forward(test_set.test_x)
        test_loss = mseloss(test_set.test_y, y_estimate)
        return test_loss


class LogisticTrainer(GenericTrainer):
    def __init__(self, dataloader, model, num_epochs, lr, each, decay_rate):
        super().__init__(dataloader, model, num_epochs, lr, each, decay_rate)

    def train_epoch(self):
        train_batch, _ = self.dataloader.getloaders()
        train_epoch_loss = []
        while True:
            try:
                x, y = next(train_batch)
                y_estimate = self.model.forward(x)
                mini_loss = logloss(y, y_estimate)
                train_epoch_loss.append(mini_loss)
                gradient = self.model.backwards(x, y, y_estimate)
                self.model.params = sgd_v2(self.lr, self.model.params, gradient)
            except StopIteration:
                break
        return np.sum(train_epoch_loss) / len(train_epoch_loss)

    def valid_epoch(self):
        valid_epoch_loss = []
        _, valid_loader = self.dataloader.getloaders()
        while True:
            try:
                x, y = next(valid_loader)
                y_estimate = self.model.forward(x)
                mini_loss = logloss(y, y_estimate)
                valid_epoch_loss.append(mini_loss)
            except StopIteration:
                break
        return np.sum(valid_epoch_loss) / len(valid_epoch_loss)

    def test_model(self, test_set):
        y_estimate = self.model.forward(test_set.test_x)
        test_loss = logloss(test_set.test_y, y_estimate)
        result = np.round(y_estimate)
        ps = precision(result, test_set.test_y)
        rec = recall(result, test_set.test_y)
        f1 = 2 * (ps * rec) / (ps + rec)
        print(
            f"\nError : {test_loss} ---- Precision {np.round(ps, 2)} ---- Recall {np.round(rec, 2)} ---- F1 Score {np.round(f1, 2)}")
        return test_loss


