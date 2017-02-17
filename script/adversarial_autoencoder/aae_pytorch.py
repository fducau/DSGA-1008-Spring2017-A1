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



mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 32
z_dim = 2
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 128
cnt = 0
lr = 1e-3


print('loading data!')
data_path = '../data/'
trainset_labeled = pickle.load(open(data_path + "train_labeled.p", "rb"))
validset = pickle.load(open(data_path + "validation.p", "rb"))

# Encoder
#Q = torch.nn.Sequential(
#    torch.nn.Linear(X_dim, h_dim),
#    torch.nn.ReLU(),
#    torch.nn.Linear(h_dim, z_dim)
#)
class Q_net(nn.Module):
    def __init__(self):
        super(Q_net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)

# Decoder
#P = torch.nn.Sequential(
#    torch.nn.Linear(z_dim, h_dim),
#    torch.nn.ReLU(),
#    torch.nn.Linear(h_dim, X_dim),
#    torch.nn.Sigmoid()
# )
class P_net(nn.Module):
    def __init__(self):
        super(P_net, self).__init__()
        self.lin1 = nn.Linear(z_dim, h_dim)
        self.lin2 = nn.Linear(h_dim, X_dim)
        # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(320, 50)
        # self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return F.sigmoid(x)
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 320)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = F.relu(self.fc2(x))
        # return F.log_softmax(x)


# Discriminator
# D = torch.nn.Sequential(
#     torch.nn.Linear(z_dim, h_dim),
#     torch.nn.ReLU(),
#     torch.nn.Linear(h_dim, 1),
#     torch.nn.Sigmoid()
# )

class D_net(nn.Module):
    def __init__(self):
        super(D_net, self).__init__()
        self.lin1 = nn.Linear(z_dim, h_dim)
        self.lin2 = nn.Linear(h_dim, 1)
        # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(320, 50)
        # self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return F.sigmoid(x)
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 320)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = F.relu(self.fc2(x))
        # return F.log_softmax(x)






def reset_grad():
    Q.zero_grad()
    P.zero_grad()
    D.zero_grad()


def sample_X(size, include_y=False):
    X, y = mnist.train.next_batch(size)
    X = Variable(torch.from_numpy(X))


    if include_y:
        y = np.argmax(y, axis=1).astype(np.int)
        y = Variable(torch.from_numpy(y))
        return X, y

    return X


Q = Q_net().cuda()
P = P_net().cuda()
D = D_net().cuda()

Q_solver = optim.Adam(Q.parameters(), lr=lr)
P_solver = optim.Adam(P.parameters(), lr=lr)
D_solver = optim.Adam(D.parameters(), lr=lr)

for it, (X, y) in enumerate(train_loader):
    X = sample_X(mb_size).cuda()
    #X.cuda()

    """ Reconstruction phase """
    z_sample = Q(X)
    X_sample = P(z_sample)

    recon_loss = F.binary_cross_entropy(X_sample, X)

    recon_loss.backward()
    P_solver.step()
    Q_solver.step()
    reset_grad()

    """ Regularization phase """
    # Discriminator
    z_real = Variable(torch.randn(mb_size, z_dim)).cuda()

    z_fake = Q(X)

    D_real = D(z_real)
    D_fake = D(z_fake)

    D_loss = -torch.mean(torch.log(D_real) + torch.log(1 - D_fake))

    D_loss.backward()
    D_solver.step()
    reset_grad()

    # Generator
    z_fake = Q(X)
    D_fake = D(z_fake)

    G_loss = -torch.mean(torch.log(D_fake))

    G_loss.backward()
    Q_solver.step()
    reset_grad()

    # Print and plot every now and then
    if it % 1000 == 0:
        cnt = it / 1000
        print('Iter-{}; D_loss: {:.4}; G_loss: {:.4}; recon_loss: {:.4}'
              .format(it, D_loss.data[0], G_loss.data[0], recon_loss.data[0]))

        samples = P(z_real)
        img = np.array(samples.data[0].tolist()).reshape(28,28)
        plt.imshow(img)
        
        plt.savefig('out/{}.png'
                    .format(str(cnt).zfill(3)), bbox_inches='tight')
        plt.close()



#        gs = gridspec.GridSpec(4, 4)
#        gs.update(wspace=0.05, hspace=0.05)
#        for i, sample in enumerate(samples):
#            ax = plt.subplot(gs[i])
#            plt.axis('off')
#            ax.set_xticklabels([])
#            ax.set_yticklabels([])
#            ax.set_aspect('equal')
#            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
#        if not os.path.exists('out/'):
#            os.makedirs('out/')
#        plt.savefig('out/{}.png'
#                    .format(str(cnt).zfill(3)), bbox_inches='tight')
#        cnt += 1
#        plt.close(fig)


