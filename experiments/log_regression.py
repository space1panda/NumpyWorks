import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from models.single import LogisticRegression
from datasources.batchloader import BinBatchLoader
from trainers.linreg_trainer import LogisticTrainer
from utils.initers import init_lin_w_bias






dataloader = BinBatchLoader(ds_len=1000, class_split_factor=0.7, test_split=0.9, data_dist=1000,
                 class_mix_factor=50, bs=50, valid_split=0.1)
print(dataloader.train_x.shape, dataloader.valid_x.shape, dataloader.test_x.shape)
params = init_lin_w_bias(2)
model = LogisticRegression(params)
trainer = LogisticTrainer(dataloader=dataloader, model=model, num_epochs=10000,
                          lr=0.1, each=2000, decay_rate=1)

model.params, train_losses, valid_losses = trainer.train()

trainer.test_model()

slope = -(model.params['w'][0] / model.params['w'][1])
intercept = -(model.params['b'] / model.params['w'][-1])

plt.figure()
sns.set_style('white')
sns.scatterplot(dataloader.train_x[:,0], dataloader.train_x[:,1],
                hue=dataloader.train_y);
plt.figure()
sns.scatterplot(dataloader.test_x[:,0], dataloader.test_x[:,1],
                hue=dataloader.test_y);

ax = plt.gca()
ax.autoscale(False)
x_vals = np.array(ax.get_xlim())
y_vals = intercept + (slope * x_vals)

plt.plot(x_vals, y_vals, c="k")

plt.figure()

plt.plot(train_losses, 'b')
plt.plot(valid_losses, 'r')

plt.show()