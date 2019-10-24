import matplotlib.pyplot as plt
import seaborn as sns
from datasources.datagen import *
from datasources.batchloaders import *

import matplotlib.pyplot as plt
testing = BinClassDataGenerator(ds_len=1000, test_split=0.1, class_split_factor=0.3, data_dist=1000, num_features=2)

print(testing._x, testing._y)

plt.figure()
sns.set_style('white')
sns.scatterplot(testing._x[:,0], testing._x[:,1],
                hue=testing._y)

plt.show()
