import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import matplotlib.pyplot as plt
import json
import argparse
import pickle
import numpy as np
from models.single import LinearRegression
from datasources.datagen import PolynomLinDataGenerator
from datasources.batchloaders import LinBatchLoader
from trainers.linreg_trainer import LinearTrainer
from utils.initers import init_lin

def main(config, ext_args):
    if config['transfer']:
        print(">>>Loading saved model...")
        model = pickle.load(open(config["load_path"], 'rb'))
    else:
        model = LinearRegression(init_lin(config["datagen"]["num_features"]))

    datagen = PolynomLinDataGenerator(**config["datagen"])
    x,y = datagen._x, datagen._y
    dataloader = LinBatchLoader(x,y,**config["dataloader"])
    if ext_args.noplot:
        plt.figure(1)
        plt.scatter(dataloader.train_x[:, 1], dataloader.train_y, c='y', label='Train Set')
        plt.scatter(dataloader.valid_x[:, 1], dataloader.valid_y, c='r', label='Validation Set')
        plt.scatter(datagen.test_x[:, 1], datagen.test_y, c='b', label='Test Set')
        plt.title("Data Structure")
        plt.legend(loc='best')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    trainer = LinearTrainer(dataloader=dataloader, model=model, **config['trainer'])
    model.params, train_losses, valid_losses = trainer.train()
    trainer.test_model(test_set=datagen)
    trainer.save_model(config['save_path'])
    if ext_args.noplot:
        plt.figure(2)
        plt.plot(range(config["trainer"]['num_epochs']), train_losses, label='Train')
        plt.plot(range(config["trainer"]['num_epochs']), valid_losses, label='Validate')
        plt.grid()
        plt.legend(loc='best')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f"Error Optimization , Test Error: {trainer.test_model(datagen)}")

        plt.figure(3)
        plt.plot(datagen._allx[:,1], datagen._allx.dot(model.params), c='b', label='Model', linewidth=3.0)
        """plt.scatter(dataloader.train_x[:, 1], dataloader.train_y, c='y', label='Train Set')
        plt.scatter(dataloader.valid_x[:, 1], dataloader.valid_y, c='r', label='Validation Set')"""
        plt.scatter(datagen.test_x[:, 1], datagen.test_y, c='g', label='Test Set')
        plt.grid()
        plt.legend(loc='best')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f"Model Performance, dataset_len: {config['datagen']['ds_len']}, polynomial order: "
                  f"{config['datagen']['num_features']}")

        plt.show()

    return model

default_config = "lin_config.json"

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training configuration:')
    parser.add_argument('-c', '--config', default=default_config, type=str,
                        help='config file (default: ./<script_filename>.json)')
    parser.add_argument('-p', '--noplot', default=False, type=bool,
                        help='Run Training without pyplots')

    ext_args = parser.parse_args()
    config = json.load(open(ext_args.config))

    main(config, ext_args)