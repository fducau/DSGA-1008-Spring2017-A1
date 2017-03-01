import torch
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data
import torch.nn as nn
import torch.nn.functional as F
import pickle
import matplotlib

cuda=True

def plot_2d_labeled(Q, data_loader):
    for X, y in data_loader:
        train_batch_size = 64
        X_dim = 784
        X.resize_(train_batch_size, X_dim)
        X, target = Variable(X), Variable(y)

        if cuda:
            X, target = X.cuda(), target.cuda()

        z_sample = Q(X)

        z_sample = np.array(z_sample.data.tolist())
        colors = matplotlib.colors.cnames.keys()
        cindex = y.numpy()
        color = [colors[i] for i in cindex]


        plt.scatter(z_sample[:,0], z_sample[:,1], c=color)

    plt.show()