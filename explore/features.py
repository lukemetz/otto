import pandas as pd
from fuel import config
import os
import numpy as np
import theano
from sklearn import preprocessing

base_path = os.path.join(config.data_path, "otto")
train =  pd.read_csv(os.path.join(base_path, 'train.csv'))

classes_str = train['target']
targets = np.zeros_like(classes_str.values, dtype="int32")
for i in range(1, 10):
    targets[classes_str.values == ("Class_%d"%i)] = i
features = train[["feat_%d"%x for x in range(1, 94)]].values.astype(theano.config.floatX)

print features

