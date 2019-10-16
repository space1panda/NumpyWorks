import numpy as np
import pickle
from utils.losses import *
from utils.optimizers import *
from utils.misc import *

class LinearTrainer:
    def __init__(self, dataloader, model, num_epochs, lr, each, decay_rate):
        self.num_epochs = num_epochs
        self.model = model
        self.lr = lr
        self.each = each
        self.decay_rate = decay_rate
        self.dataloader = dataloader

    def train_epoch(self):
        train_batch, _ = self.dataloader._getloaders()
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
        _, valid_loader = self.dataloader._getloaders()
        while True:
            try:
                x, y = next(valid_loader)
                y_estimate = self.model.forward(x)
                mini_loss = mseloss(y, y_estimate)
                valid_epoch_loss.append(mini_loss)
            except StopIteration:
                break
        return np.sum(valid_epoch_loss) / len(valid_epoch_loss)

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

    def test_model(self):
        x, y = self.dataloader.gettest()
        y_estimate = self.model.forward(x)
        test_loss = mseloss(y, y_estimate)
        return test_loss

    def save_model(self, save_path):
        print(f">>> Saving trained model to {save_path}")
        pickle.dump(self.model, open(save_path, "wb"))


class LogisticTrainer:
    def __init__(self, dataloader, model, num_epochs, lr, each, decay_rate):
        self.num_epochs = num_epochs
        self.model = model
        self.lr = lr
        self.each = each
        self.decay_rate = decay_rate
        self.dataloader = dataloader

    def train_epoch(self):
        train_batch, _ = self.dataloader._getloaders()
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
        _, valid_loader = self.dataloader._getloaders()
        while True:
            try:
                x, y = next(valid_loader)
                y_estimate = self.model.forward(x)
                mini_loss = logloss(y, y_estimate)
                valid_epoch_loss.append(mini_loss)
            except StopIteration:
                break
        return np.sum(valid_epoch_loss) / len(valid_epoch_loss)

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

    def test_model(self):
        x, y = self.dataloader.gettest()
        y_estimate = self.model.forward(x)
        test_loss = logloss(y, y_estimate)
        result = np.round(y_estimate)
        ps = precision(result, y)
        rec = recall(result, y)
        f1 = 2 * (ps * rec) / (ps + rec)
        print(
            f"\nError : {test_loss} ---- Precision {np.round(ps, 2)} ---- Recall {np.round(rec, 2)} ---- F1 Score {np.round(f1, 2)}")
        return test_loss

    def save_model(self, save_path):
        print(f">>> Saving trained model to {save_path}")
        pickle.dump(self.model, open(save_path, "wb"))


