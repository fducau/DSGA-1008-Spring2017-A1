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


cuda = torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

mb_size = 32
z_dim = 2
X_dim = 784
y_dim = 10
h_dim = 128
cnt = 0
lr = 1e-3

train_batch_size = 64
val_batch_size = 64
epochs = 1000


print('loading data!')
data_path = './../data/'
trainset_labeled = pickle.load(open(data_path + "train_labeled.p", "rb"))
trainset_unlabeled = pickle.load(open(data_path + "train_unlabeled.p", "rb"))
validset = pickle.load(open(data_path + "validation.p", "rb"))


##################################
# Define Networks
##################################
# Encoder
class Q_net(nn.Module):
    def __init__(self):
        super(Q_net, self).__init__()
        self.lin1 = nn.Linear(X_dim, h_dim)
        self.lin2 = nn.Linear(h_dim, z_dim)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x

# Decoder
class P_net(nn.Module):
    def __init__(self):
        super(P_net, self).__init__()
        self.lin1 = nn.Linear(z_dim, h_dim)
        self.lin2 = nn.Linear(h_dim, X_dim)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return F.sigmoid(x)

# Discriminator
class D_net(nn.Module):
    def __init__(self):
        super(D_net, self).__init__()
        self.lin1 = nn.Linear(z_dim, h_dim)
        self.lin2 = nn.Linear(h_dim, 1)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return F.sigmoid(x)


def sample_X(size, include_y=False):
    X, y = mnist.train.next_batch(size)
    X = Variable(torch.from_numpy(X))

    if include_y:
        y = np.argmax(y, axis=1).astype(np.int)
        y = Variable(torch.from_numpy(y))
        return X, y

    return X

def reset_grads(models):
    for m in models:
        m.zero_grad()

def train(P, Q, D, P_solver, Q_solver, D_solver, data_loader):
    Q.train()
    P.train()
    D.train()

    for batch_idx, (X, target) in enumerate(data_loader):
        X.resize_(train_batch_size, X_dim)
        X, target = Variable(X), Variable(target)

        # X = sample_X(train_batch_size)

        if cuda:
            X, target = X.cuda(), target.cuda()

        # Reconstruction phase
        z_sample = Q(X)
        X_sample = P(z_sample)
    
        recon_loss = F.binary_cross_entropy(X_sample, X)
    
        recon_loss.backward()
        P_solver.step()
        Q_solver.step()
    
        P.zero_grad()
        Q.zero_grad()
        D.zero_grad()
        # reset_grads([P, Q, D])
    
        """ Regularization phase """
        # Discriminator
        z_real = Variable(torch.randn(train_batch_size, z_dim))
        if cuda:
            z_real = z_real.cuda()
    
        z_fake = Q(X)
    
        D_real = D(z_real)
        D_fake = D(z_fake)
    
        D_loss = -torch.mean(torch.log(D_real) + torch.log(1 - D_fake))
    
        D_loss.backward()
        D_solver.step()
    
        P.zero_grad()
        Q.zero_grad()
        D.zero_grad()
        # reset_grads([P, Q, D])
    
        # Generator
        z_fake = Q(X)
        D_fake = D(z_fake)
    
        G_loss = -torch.mean(torch.log(D_fake))
    
        G_loss.backward()
        Q_solver.step()
    
        P.zero_grad()
        Q.zero_grad()
        D.zero_grad()

        samples = P(z_real)
    # reset_grads([P, Q, D])
    return D_loss, G_loss, recon_loss, samples


##################################
# Create and initialize Networks
##################################
if cuda:
    Q = Q_net().cuda()
    P = P_net().cuda()
    D = D_net().cuda()
else:
    Q = Q_net()
    P = P_net()
    D = D_net()

Q_solver = optim.Adam(Q.parameters(), lr=lr)
P_solver = optim.Adam(P.parameters(), lr=lr)
D_solver = optim.Adam(D.parameters(), lr=lr)


##################################
# Data loaders
##################################
train_labeled_loader = torch.utils.data.DataLoader(trainset_labeled, batch_size=64, shuffle=True, **kwargs)
train_unlabeled_loader = torch.utils.data.DataLoader(trainset_unlabeled, batch_size=64, shuffle=True, **kwargs)

valid_loader = torch.utils.data.DataLoader(validset, batch_size=64, shuffle=True)





for epoch in range(epochs):
    D_loss, G_loss, recon_loss, samples = train(P, Q, D, P_solver, Q_solver, D_solver,
                                                train_labeled_loader)
    D_loss, G_loss, recon_loss, samples = train(P, Q, D, P_solver, Q_solver, D_solver,
                                                train_labeled_loader)


    # Print and plot every now and then
    if epoch % 100 == 0:

        print('Epoch-{}; D_loss: {:.4}; G_loss: {:.4}; recon_loss: {:.4}'
              .format(epoch, D_loss.data[0], G_loss.data[0], recon_loss.data[0]))


        img = np.array(samples.data[0].tolist()).reshape(28,28)
        plt.imshow(img, cmap='gray_r')

        plt.savefig('out/{}.png'
                    .format(str(epoch).zfill(3)), bbox_inches='tight')
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


