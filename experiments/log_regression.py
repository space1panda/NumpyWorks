import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from models.single import LogisticRegression
from datasources.datagen import BinClassDataGenerator
from datasources.batchloaders import LinBatchLoader
from trainers.linreg_trainer import LogisticTrainer
from utils.initers import init_lin_w_bias

def main(config):
    model = LogisticRegression(init_lin_w_bias(config["datagen"]["num_features"]))

    datagen = BinClassDataGenerator(**config["datagen"])
    x,y = datagen._x, datagen._y
    dataloader = LinBatchLoader(x,y,**config["dataloader"])
    figure1 = plt.figure()
    sns.set_style('white')
    sns.scatterplot(dataloader.train_x[:, 0], dataloader.train_x[:, 1],
                    hue=dataloader.train_y)
    plt.legend(loc='best')
    plt.xlabel('Param1')
    plt.ylabel('Param2')
    plt.title(f"Data Structure")
    figure1.savefig("tmp/dataset.png")
    print(">>>Saving dataset.png to tmp")


    trainer = LogisticTrainer(dataloader=dataloader, model=model, **config['trainer'])
    model.params, train_losses, valid_losses = trainer.train()
    trainer.test_model(test_set=datagen)
    trainer.save_model(config['save_path'])
    slope = -(model.params['w'][0] / model.params['w'][1])
    intercept = -(model.params['b'] / model.params['w'][-1])

    figure2 = plt.figure()
    sns.set_style('white')
    sns.scatterplot(datagen.test_x[:, 0], datagen.test_x[:, 1],
                    hue=datagen.test_y)
    ax = plt.gca()
    ax.autoscale(False)
    x_vals = np.array(ax.get_xlim())
    y_vals = intercept + (slope * x_vals)
    plt.plot(x_vals, y_vals, c="b", label="Model")
    plt.legend(loc='best')
    plt.xlabel('Param1')
    plt.ylabel('Param2')
    plt.title(f"Model Performance")
    figure2.savefig("tmp/model_performance.png")
    print(">>>Saving model_performance.png to tmp")


    figure3 = plt.figure()

    plt.plot(train_losses, 'b', label="Train loss")
    plt.plot(valid_losses, 'r', label="Valid loss")
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f"Optimization")
    figure3.savefig("tmp/loss_optimization.png")
    print(">>>Saving loss_optimization.png to tmp")

    return model