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
n_classes = 10
mb_size = 32
z_dim = 10
X_dim = 784
y_dim = 10
h_dim = 128
cnt = 0
lr = 0.001
momentum = 0.1

train_batch_size = 50
valid_batch_size = 50
epochs = 500


##################################
# Load data and create Data loaders
##################################

print('loading data!')
data_path = './../data/'
trainset_labeled = pickle.load(open(data_path + "train_labeled.p", "rb"))
trainset_unlabeled = pickle.load(open(data_path + "train_unlabeled.p", "rb"))
trainset_unlabeled.train_labels = torch.from_numpy(np.array([-1] * 47000))

validset = pickle.load(open(data_path + "validation.p", "rb"))


train_labeled_loader = torch.utils.data.DataLoader(trainset_labeled,
                                                   batch_size=train_batch_size,
                                                   shuffle=True, **kwargs)
train_unlabeled_loader = torch.utils.data.DataLoader(trainset_unlabeled,
                                                     batch_size=train_batch_size,
                                                     shuffle=True, **kwargs)

valid_loader = torch.utils.data.DataLoader(validset, batch_size=valid_batch_size, shuffle=True)


##################################
# Define Networks
##################################
# Encoder
class Q_net(nn.Module):
    def __init__(self):
        super(Q_net, self).__init__()
        self.lin1 = nn.Linear(X_dim, 1000)
        self.lin2 = nn.Linear(1000, 1000)
        self.lin3a = nn.Linear(1000, n_classes)
        self.lin3b = nn.Linear(1000, z_dim)
        #self.lin4a = nn.Linear(1000, n_classes)
        #self.lin4b = nn.Linear(1000, z_dim)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)

        x1 = self.lin3a(x)
        x1 = F.relu(x1)
        #x1 = self.lin4a(x1)
        #x1 = F.softmax(x1)

        x2 = self.lin3b(x)
        x2 = F.relu(x2)
        #x2 = self.lin4b(x2)
        #x2 = F.relu(x2)

        #x = torch.cat((x1,x2),1)

        return x1, x2

# Decoder
class P_net(nn.Module):
    def __init__(self):
        super(P_net, self).__init__()
        self.lin1 = nn.Linear(z_dim + 10, 1000)
        self.lin2 = nn.Linear(1000, X_dim)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return F.sigmoid(x)

# Discriminator
class D_net_cat(nn.Module):
    def __init__(self):
        super(D_net_cat, self).__init__()
        self.lin1 = nn.Linear(10, 1000)
        self.lin2 = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return F.sigmoid(x)

class D_net_gauss(nn.Module):
    def __init__(self):
        super(D_net_gauss, self).__init__()
        self.lin1 = nn.Linear(z_dim, 1000)
        self.lin2 = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return F.sigmoid(x)


def train(P, Q, D_cat, D_gauss,
          P_solver, Q_solver, D_cat_solver, D_gauss_solver,
          data_loader, labeled=False):

    Q.train()
    P.train()
    D_cat.train()
    D_gauss.train()

    for batch_idx, (X, target) in enumerate(data_loader):
        if batch_idx * data_loader.batch_size + data_loader.batch_size > data_loader.dataset.k:
            continue

        P.zero_grad()
        Q.zero_grad()
        D_cat.zero_grad()
        D_gauss.zero_grad()

        X = X * 0.3081 + 0.1307

        X.resize_(train_batch_size, X_dim)
        X, target = Variable(X), Variable(target)

        if cuda:
            X, target = X.cuda(), target.cuda()

        recon_loss = 'NA'
        #if not labeled:
            # Reconstruction phase
        z_sample = torch.cat(Q(X), 1)
        if cuda:
            z_sample = z_sample.cuda()
        #z_sample = Q(X)
        X_sample = P(z_sample)

        # Use epsilon to avoid log(0) case
        TINY = 1e-8
        recon_loss = F.binary_cross_entropy(X_sample + TINY, X + TINY)

        recon_loss.backward()
        P_solver.step()
        Q_solver.step()

        P.zero_grad()
        Q.zero_grad()
        D_cat.zero_grad()
        D_gauss.zero_grad()

        recon_loss = recon_loss.data[0]

        """ Regularization phase """
        # Discriminator
        #Change for sample_categorical
        z_real_cat = np.random.randint(0, 10, train_batch_size)
        z_real_cat = np.eye(n_classes)[z_real_cat].astype('float32')
        z_real_cat = torch.from_numpy(z_real_cat)
        z_real_cat = Variable(z_real_cat)
        ######
        if cuda:
            z_real_cat = z_real_cat.cuda()

        z_real_gauss = Variable(torch.randn(train_batch_size, z_dim))
        if cuda:
            z_real_gauss = z_real_gauss.cuda()

        z_fake_cat, z_fake_gauss = Q(X)

        D_real_cat = D_cat(z_real_cat)
        D_real_gauss = D_gauss(z_real_gauss)
        D_fake_cat = D_cat(z_fake_cat)
        D_fake_gauss = D_gauss(z_fake_gauss)

        D_loss_cat = -torch.mean(torch.log(D_real_cat + TINY) - torch.log(1 - D_fake_cat + TINY))
        D_loss_gauss = -torch.mean(torch.log(D_real_gauss + TINY) - torch.log(1 - D_fake_gauss + TINY))

        if D_loss_cat.data[0] > 15.0:
            D_loss_cat.data[0] = 15.

        if D_loss_gauss.data[0] > 15.:
            D_loss_gauss.data[0] = 15.

        D_loss = D_loss_cat + D_loss_gauss
        #D_loss = D_loss_gauss
        D_loss.backward()
        #if labeled:
        D_cat_solver.step()
        D_gauss_solver.step()

        P.zero_grad()
        Q.zero_grad()
        D_cat.zero_grad()
        D_gauss.zero_grad()

        # Generator

        z_fake_cat, z_fake_gauss = Q(X)

        D_fake_cat = D_cat(z_fake_cat)
        D_fake_gauss = D_gauss(z_fake_gauss)

        G_loss = - torch.mean(torch.log(D_fake_cat + TINY)) - torch.mean(torch.log(D_fake_gauss + TINY))
        #G_loss = - torch.mean(torch.log(D_fake_gauss))
        G_loss.backward()
        Q_solver.step()

        P.zero_grad()
        Q.zero_grad()
        D_cat.zero_grad()
        D_gauss.zero_grad()

        class_loss = float('nan')
        if labeled:
            pred = Q(X)[0]
            class_loss = F.cross_entropy(pred, target)
            class_loss.backward()
            Q_solver.step()

            P.zero_grad()
            Q.zero_grad()
            D_cat.zero_grad()
            D_gauss.zero_grad()

        z_sample = torch.cat(Q(X), 1)
        if cuda:
            z_sample = z_sample.cuda()

        samples = P(z_sample)
        xsample = X

        #D_loss_cat = D_loss_gauss
        #G_loss = D_loss_gauss
        #class_loss = D_loss_gauss
    return D_loss_cat, D_loss_gauss, G_loss, recon_loss, class_loss, (samples.data[0], xsample.data[0])

def report_loss(D_loss_cat, D_loss_gauss, G_loss, recon_loss, samples=None):
        print('Epoch-{}; D_loss_cat: {:.4}: D_loss_gauss: {:.4}; G_loss: {:.4}; recon_loss: {:.4}'
              .format(epoch, D_loss_cat.data[0], D_loss_gauss.data[0], 
                      G_loss.data[0], recon_loss))

        if samples is not None:
            img = np.array(samples[0].tolist()).reshape(28, 28)
            plt.imshow(img, cmap='hot')

            plt.savefig('out/{}.png'
                        .format(str(epoch).zfill(3)), bbox_inches='tight')

            img = np.array(samples[1].tolist()).reshape(28, 28)
            plt.imshow(img, cmap='hot')

            plt.savefig('out/{}_orig.png'
                        .format(str(epoch).zfill(3)), bbox_inches='tight')

            plt.close()

def create_latent(Q, loader):
    Q.eval()
    labels = []

    for batch_idx, (X, target) in enumerate(loader):

        X = X * 0.3081 + 0.1307
        X.resize_(loader.batch_size, X_dim)
        X, target = Variable(X), Variable(target)
        labels.extend(target.data.tolist())
        if cuda:
            X, target = X.cuda(), target.cuda()
        # Reconstruction phase
        z_sample = Q(X)
        if batch_idx > 0:
            z_values = np.concatenate((z_values, np.array(z_sample.data.tolist())))
        else:
            z_values = np.array(z_sample.data.tolist())
    labels = np.array(labels)

    return z_values, labels


##################################
# Create and initialize Networks
##################################
if cuda:
    Q = Q_net().cuda()
    P = P_net().cuda()
    D_cat = D_net_cat().cuda()
    D_gauss = D_net_gauss().cuda()
else:
    Q = Q_net()
    P = P_net()
    D_gauss = D_net_gauss()
    D_cat = D_net_cat()

Q_solver = optim.SGD(Q.parameters(), lr=0.01, momentum=0.9)
P_solver = optim.SGD(P.parameters(), lr=0.01, momentum=0.9)
D_gauss_solver = optim.SGD(D_gauss.parameters(), lr=0.001)
D_cat_solver = optim.SGD(D_cat.parameters(), lr=0.001)

for epoch in range(epochs):
    D_loss_cat_u, D_loss_gauss_u, G_loss_u, recon_loss_u, _, samples_u = train(P, Q, D_cat, D_gauss,
                                                                               P_solver, Q_solver,
                                                                               D_gauss_solver, D_cat_solver,
                                                                               train_unlabeled_loader)

    D_loss_cat, D_loss_gauss, G_loss, recon_loss, class_loss, samples = train(P, Q, D_cat, D_gauss,
                                                                              P_solver, Q_solver,
                                                                              D_gauss_solver, D_cat_solver,
                                                                              train_labeled_loader,
                                                                              labeled=True)
    #if epoch % 5 == 0:
    #    print('Epoch:{} - ReconLoss: {}'.format(epoch, recon_loss))

    # Print and plot every now and then
    if epoch % 5 == 0:
        print('Loss in UNLabeled')
        report_loss(D_loss_cat_u, D_loss_gauss_u, G_loss_u, recon_loss_u)
        print('Loss in Labeled')
        report_loss(D_loss_cat, D_loss_gauss, G_loss, recon_loss, samples)
        print('Classification loss: {}'.format(class_loss.data[0]))



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



MLP_epochs = 100
z_train, y_train = create_latent(Q, train_labeled_loader)
z_train_t = torch.from_numpy(z_train.astype('float32'))
y_train_t = torch.from_numpy(y_train)
z_trainset = torch.utils.data.TensorDataset(z_train_t, y_train_t)
z_trainset.k = z_train.shape[0]
z_train_loader = torch.utils.data.DataLoader(z_trainset, batch_size=50, shuffle=True)

z_val, y_val = create_latent(Q, valid_loader)
z_val_t = torch.from_numpy(z_val.astype('float32'))
y_val_t = torch.from_numpy(y_val)
z_validset = torch.utils.data.TensorDataset(z_val_t, y_val_t)
z_validset.k = z_val.shape[0]
z_valid_loader = torch.utils.data.DataLoader(z_validset, batch_size=50, shuffle=True)

MLP = MLP_net()
if cuda:
    MLP.cuda()
MLP_solver = optim.Adam(MLP.parameters(), lr=lr/10.)



for epoch in range(MLP_epochs):
    MLP_loss = train_MLP(MLP, z_train_loader, MLP_solver)
    if epoch % 5 == 0:
        print('MLP_loss: {:.3}'.format(MLP_loss.data[0]))

test_MLP(MLP, z_valid_loader)
test_MLP(MLP, z_train_loader)

