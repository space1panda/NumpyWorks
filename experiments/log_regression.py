import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import pickle
import json
from models.single import LogisticRegression
from datasources.datagen import BinClassDataGenerator
from datasources.batchloaders import LinBatchLoader
from trainers.linreg_trainer import LogisticTrainer
from utils.initers import init_lin_w_bias

def main(config, ext_args):
    if config['transfer']:
        print(">>>Loading saved model...")
        model = pickle.load(open(config["load_path"], 'rb'))
    else:
        model = LogisticRegression(init_lin_w_bias(config["datagen"]["num_features"]))

    datagen = BinClassDataGenerator(**config["datagen"])
    x,y = datagen._x, datagen._y
    dataloader = LinBatchLoader(x,y,**config["dataloader"])
    if ext_args.noplot:
        plt.figure()
        sns.set_style('white')
        sns.scatterplot(dataloader.train_x[:, 0], dataloader.train_x[:, 1],
                        hue=dataloader.train_y)
        plt.legend(loc='best')
        plt.xlabel('Param1')
        plt.ylabel('Param2')
        plt.title(f"Data Structure")
        plt.show()


    trainer = LogisticTrainer(dataloader=dataloader, model=model, **config['trainer'])
    model.params, train_losses, valid_losses = trainer.train()
    trainer.test_model(test_set=datagen)
    trainer.save_model(config['save_path'])
    if ext_args.noplot:

        slope = -(model.params['w'][0] / model.params['w'][1])
        intercept = -(model.params['b'] / model.params['w'][-1])

        plt.figure()
        sns.set_style('white')
        sns.scatterplot(datagen.test_x[:, 0], datagen.test_x[:, 1],
                        hue=datagen.test_y)
        ax = plt.gca()
        ax.autoscale(False)
        x_vals = np.array(ax.get_xlim())
        y_vals = intercept + (slope * x_vals)
        plt.plot(x_vals, y_vals, c="b")


        plt.figure()

        plt.plot(train_losses, 'b')
        plt.plot(valid_losses, 'r')

        plt.show()

    return model

default_config = "log_config.json"

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training configuration:')
    parser.add_argument('-c', '--config', default=default_config, type=str,
                        help='config file (default: ./<script_filename>.json)')
    parser.add_argument('-p', '--noplot', default=False, type=bool,
                        help='Run Training without pyplots')

    ext_args = parser.parse_args()
    config = json.load(open(ext_args.config))

    main(config, ext_args)