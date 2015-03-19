import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset import OttoDataset
from tsne import bh_sne
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pylab as plt
import numpy as np

dataset = OttoDataset('train')
X = dataset.indexables[0].astype(np.float)
Y = dataset.indexables[1].astype(np.float)
n = 10000
X = X[0:n, :]
Y = Y[0:n]

X_2d = bh_sne(X, d=2)

print X_2d.shape
c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

for i in range(9):
    plt.scatter(X_2d[:,0][Y==i], X_2d[:, 1][Y==i], c=c[i])

plt.legend(["Class" + str(x+1) for x in range(0, 9)])
plt.show()
