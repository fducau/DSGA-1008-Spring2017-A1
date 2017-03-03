import time
from utils import *
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import itertools
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
from torch.nn.modules.upsampling import UpsamplingNearest2d



cuda = torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
n_classes = 10
mb_size = 32
z_dim = 5
X_dim = 784
y_dim = 10
h_dim = 128
cnt = 0
lr = 0.001
momentum = 0.1
convolutional = True
train_batch_size = 50
valid_batch_size = 50


##################################
# Load data and create Data loaders
##################################

print('loading data!')
data_path = './../data/'
trainset_labeled = pickle.load(open(data_path + "train_labeled.p", "rb"))
trainset_unlabeled = pickle.load(open(data_path + "train_unlabeled.p", "rb"))
trainset_unlabeled.train_labels = torch.from_numpy(np.array([-1] * 47000))

validset = pickle.load(open(data_path + "validation.p", "rb"))

#print('Augmenting dataset!')
#augment_dataset(trainset_labeled, b=100, k=8)

#print('Augmented dataset to size: {}'.format(trainset_labeled.k)) 

train_labeled_loader = torch.utils.data.DataLoader(trainset_labeled,
                                                   batch_size=train_batch_size,
                                                   shuffle=True, **kwargs)

#train_augmented_loader = torch.utils.data.DataLoader(augmented,
#                                                   batch_size=train_batch_size,
#                                                   shuffle=True, **kwargs)

train_unlabeled_loader = torch.utils.data.DataLoader(trainset_unlabeled,
                                                     batch_size=783,
                                                     shuffle=True, **kwargs)

valid_loader = torch.utils.data.DataLoader(validset, batch_size=valid_batch_size, shuffle=True)

#N = 500
N = 320
##################################
# Define Networks
##################################
# Encoder
# Encoder
class Q_net(nn.Module):
    def __init__(self):
        super(Q_net, self).__init__()
        self.lin1 = nn.Linear(X_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3a = nn.Linear(N, N)
        self.lin3b = nn.Linear(N, z_dim)
        self.lin4a = nn.Linear(N, n_classes)
        #self.lin4b = nn.Linear(N, z_dim)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(x)

        x1 = self.lin3a(x)
        x1 = F.relu(x1)
        # x1 = F.dropout(x1, p=0.2, training=self.training)
        x1 = self.lin4a(x1)
        x1 = F.softmax(x1)

        x2 = self.lin3b(x)
        # x2 = F.tanh(x2)
        #x2 = self.lin4b(x2)
        #x2 = F.relu(x2)

        return x1, x2

class Q_net_conv(nn.Module):
    def __init__(self):
        super(Q_net_conv, self).__init__()
        self.conv1 = nn.Conv2d(1, 100, kernel_size=5)
        self.conv2 = nn.Conv2d(100, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.1)
        self.fc1 = nn.Linear(N, N)
        self.fc2 = nn.Linear(N, n_classes)
        self.fc3 = nn.Linear(N, z_dim)
        self.fc4 = nn.Linear(N, z_dim)

    def forward(self, x):
        x, id1 = F.max_pool2d(self.conv1(x), 2, stride=2, return_indices=True)
        self.pool1_idx = id1
        x = F.relu(x)
        x, id2 = F.max_pool2d(self.conv2_drop(self.conv2(x)), 2, stride=2, return_indices=True)
        self.pool2_idx = id2
        x = F.relu(x)
        x = x.view(-1, 320)

        x1 = F.relu(self.fc1(x))
        #x1 = F.dropout(x1, p=0.3, training=self.training)
        x1 = F.relu(self.fc2(x1))
        x1 = F.softmax(x1)

        x2 = self.fc3(x)
        # x2 = F.dropout(x2, training=self.training)
        # x2 = F.relu(self.fc4(x2))

        return x1, x2

# Decoder

class P_net_conv(nn.Module):
    def __init__(self):
        super(P_net_conv, self).__init__()
        self.lin1 = nn.Linear(z_dim + n_classes, N)
        self.conv1 = nn.Conv2d(75, 1, kernel_size=9)
        self.conv2 = nn.Conv2d(20, 75, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.upsample2 = UpsamplingNearest2d(scale_factor=4)
        self.upsample1 = UpsamplingNearest2d(scale_factor=3)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = x.view(-1, 20, 4, 4)
        x = self.upsample2(x)
        x = self.conv2(x)
        x = self.upsample1(x)
        x = self.conv1(x)
        x = F.sigmoid(x)

        return x

class P_net(nn.Module):
    def __init__(self):
        super(P_net, self).__init__()
        self.lin1 = nn.Linear(z_dim + 10, N)
        self.lin2 = nn.Linear(N, X_dim)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        # x = F.dropout(x, p=0., training=self.training)
        x = self.lin2(x)
        return F.sigmoid(x)

# Discriminator
class D_net_cat(nn.Module):
    def __init__(self):
        super(D_net_cat, self).__init__()
        self.lin1 = nn.Linear(10, N)
        self.lin2 = nn.Linear(N, 1)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        return F.sigmoid(x)

class D_net_gauss(nn.Module):
    def __init__(self):
        super(D_net_gauss, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, 1)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        return F.sigmoid(x)

def sample_categorical(batch_size, n_classes=10):
    cat = np.random.randint(0, 10, batch_size)
    cat = np.eye(n_classes)[cat].astype('float32')
    cat = torch.from_numpy(cat)
    return Variable(cat)

def get_categorical(labels, n_classes=10):
    cat = np.array(labels.data.tolist())
    cat = np.eye(n_classes)[cat].astype('float32')
    cat = torch.from_numpy(cat)
    return Variable(cat)

def pretrain(model, optimizer, epoch, train_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output[0], target)
        loss.backward()
        optimizer.step()
        if batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def pretest(model, epoch, valid_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in valid_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)[0]
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(valid_loader)  # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))


def train(P, Q, D_cat, D_gauss,
          P_solver, Q_solver, D_cat_solver, D_gauss_solver,
          train_labeled_loader, train_unlabeled_loader=None):

    Q.train()
    P.train()
    D_cat.train()
    D_gauss.train()

    if train_unlabeled_loader is None:
        train_unlabeled_loader = train_labeled_loader

    # for batch_idx, (X, target) in enumerate(data_loader):
    for (X_l, target_l), (X_u, target_u) in itertools.izip(train_labeled_loader, train_unlabeled_loader):
        #if batch_idx * data_loader.batch_size + data_loader.batch_size > data_loader.dataset.k:
        #    continue

        for X, target in [(X_u, target_u), (X_l, target_l)]:
            if target[0] == -1:
                labeled = False
            else:
                labeled = True

            P.zero_grad()
            Q.zero_grad()
            D_cat.zero_grad()
            D_gauss.zero_grad()
            X = X * 0.3081 + 0.1307
            if not convolutional:
                X.resize_(train_batch_size, X_dim)
            X, target = Variable(X), Variable(target)

            if cuda:
                X, target = X.cuda(), target.cuda()

            # Reconstruction phase
            if labeled:
                target_one_hot = get_categorical(target)
                if cuda:
                    target_one_hot = target_one_hot.cuda()

                _, z_gauss_sample = Q(X)
                z_sample = torch.cat((target_one_hot, z_gauss_sample), 1)
            else:
                z_sample = torch.cat(Q(X), 1)

            if cuda:
                z_sample = z_sample.cuda()
            X_sample = P(z_sample)

            # Use epsilon to avoid log(0) case
            TINY = 1e-8
            recon_loss = F.binary_cross_entropy(X_sample + TINY, X.resize(train_batch_size, X_dim) + TINY)

            # Placeholder for classification loss
            class_loss = Variable(torch.from_numpy(np.array([-1.])))
            if labeled:
                recon_loss.backward()
                P_solver.step()
                Q_solver.step()
                P.zero_grad()
                Q.zero_grad()
                D_cat.zero_grad()
                D_gauss.zero_grad()
                # Classification Loss

                pred = Q(X)[0]
                class_loss = F.cross_entropy(pred, target)
                class_loss = class_loss

                class_loss.backward()
                Q_solver.step()
                class_loss = class_loss

            else:
                recon_loss.backward()
                P_solver.step()
                Q_solver.step()

            P.zero_grad()
            Q.zero_grad()
            D_cat.zero_grad()
            D_gauss.zero_grad()
            """ Regularization phase """
            # Discriminator

            z_real_cat = sample_categorical(train_batch_size, n_classes=10)
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

            D_loss_cat = -torch.mean(torch.log(D_real_cat + TINY) + torch.log(1 - D_fake_cat + TINY))
            D_loss_gauss = -torch.mean(torch.log(D_real_gauss + TINY) + torch.log(1 - D_fake_gauss + TINY))

            if D_loss_cat.data[0] > 15.0:
                D_loss_cat.data[0] = 15.

            if D_loss_gauss.data[0] > 15.:
                D_loss_gauss.data[0] = 15.

            D_loss = D_loss_cat + D_loss_gauss
            D_loss = D_loss_gauss
            D_loss.backward()

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
            #G_loss = - torch.mean(torch.log(D_fake_gauss + TINY))
            G_loss = G_loss / 100.
            G_loss.backward()
            Q_solver.step()

            G_loss = G_loss * 100.

            P.zero_grad()
            Q.zero_grad()
            D_cat.zero_grad()
            D_gauss.zero_grad()

            z_sample = torch.cat(Q(X), 1)
            if cuda:
                z_sample = z_sample.cuda()

            samples = P(z_sample)
            xsample = X
            # D_loss_cat = recon_loss
            # D_loss_gauss = recon_loss
            # G_loss = recon_loss
            # class_loss = recon_loss
    return D_loss_cat, D_loss_gauss, G_loss, recon_loss, class_loss, (samples.data[0], xsample.data[0])

def report_loss(D_loss_cat, D_loss_gauss, G_loss, recon_loss, samples=None):
        print('Epoch-{}; D_loss_cat: {:.4}: D_loss_gauss: {:.4}; G_loss: {:.4}; recon_loss: {:.4}'
              .format(epoch, D_loss_cat.data[0], D_loss_gauss.data[0],
                      G_loss.data[0], recon_loss.data[0]))

        # if samples is not None:
            # img = np.array(samples[0].tolist()).reshape(28, 28)
            # plt.imshow(img, cmap='hot')

            # plt.savefig('out/{}.png'
            #             .format(str(epoch).zfill(3)), bbox_inches='tight')

            # img = np.array(samples[1].tolist()).reshape(28, 28)
            # plt.imshow(img, cmap='hot')

            # plt.savefig('out/{}_orig.png'
            #             .format(str(epoch).zfill(3)), bbox_inches='tight')

           # plt.close()

def create_latent(Q, loader):
    Q.eval()
    labels = []

    for batch_idx, (X, target) in enumerate(loader):

        X = X * 0.3081 + 0.1307
        # X.resize_(loader.batch_size, X_dim)
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

def predict_cat(Q, data_loader):
    Q.eval()
    labels = []
    test_loss = 0
    correct = 0

    for batch_idx, (X, target) in enumerate(data_loader):

        X = X * 0.3081 + 0.1307
        #X.resize_(data_loader.batch_size, X_dim)
        X, target = Variable(X), Variable(target)
        labels.extend(target.data.tolist())
        if cuda:
            X, target = X.cuda(), target.cuda()
        # Reconstruction phase
        output = Q(X)[0]

        test_loss += F.nll_loss(output, target).data[0]

        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(data_loader)
    print('\nAvg loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(
          test_loss, correct, len(data_loader.dataset),
          100. * correct / len(data_loader.dataset)))


##################################
# Create and initialize Networks
##################################
if cuda:
    if convolutional:
        Q = Q_net_conv().cuda()
        P = P_net_conv().cuda()
    else:
        Q = Q_net().cuda()
        P = P_net().cuda()

    D_cat = D_net_cat().cuda()
    D_gauss = D_net_gauss().cuda()
else:
    Q = Q_net()
    P = P_net()
    D_gauss = D_net_gauss()
    D_cat = D_net_cat()


Q_solver = optim.SGD(Q.parameters(), lr=0.1, momentum=0.9)
P_solver = optim.SGD(P.parameters(), lr=0.1, momentum=0.9)
D_gauss_solver = optim.SGD(D_gauss.parameters(), lr=0.001)
D_cat_solver = optim.SGD(D_cat.parameters(), lr=0.001)


train_labeled_loader = torch.utils.data.DataLoader(trainset_labeled,
                                                   batch_size=train_batch_size,
                                                   shuffle=True, **kwargs)
train_unlabeled_loader = torch.utils.data.DataLoader(trainset_unlabeled,
                                                     batch_size=train_batch_size,
                                                     shuffle=True, **kwargs)
valid_loader = torch.utils.data.DataLoader(validset, batch_size=valid_batch_size, shuffle=True)
epochs = 500
for epoch in range(epochs):
    D_loss_cat, D_loss_gauss, G_loss, recon_loss, class_loss, samples = train(P, Q, D_cat, D_gauss,
                                                                              P_solver, Q_solver,
                                                                              D_cat_solver, D_gauss_solver,
                                                                              train_labeled_loader,
                                                                              train_unlabeled_loader)
    if epoch % 10 == 0:
        report_loss(D_loss_cat, D_loss_gauss, G_loss, recon_loss, samples)
        print('Classification loss: {}'.format(class_loss.data[0]))

# pretrain_epochs = 50
# optimizer = optim.SGD(Q.parameters(), lr=0.001, momentum=0.4)
# for epoch in range(pretrain_epochs):
#     pretrain(Q, optimizer, epoch, train_labeled_loader)
#     pretest(Q, epoch, valid_loader)

train_labeled_loader = torch.utils.data.DataLoader(trainset_labeled,
                                                   batch_size=train_batch_size,
                                                   shuffle=True, **kwargs)
train_unlabeled_loader = torch.utils.data.DataLoader(trainset_unlabeled,
                                                     batch_size=train_batch_size,
                                                     shuffle=True, **kwargs)
valid_loader = torch.utils.data.DataLoader(validset, batch_size=valid_batch_size, shuffle=True)

epochs = 1000
train_start = time.time()
for epoch in range(epochs):
    D_loss_cat, D_loss_gauss, G_loss, recon_loss, class_loss, samples = train(P, Q, D_cat, D_gauss,
                                                                              P_solver, Q_solver,
                                                                              D_cat_solver, D_gauss_solver,
                                                                              train_labeled_loader,
                                                                              train_unlabeled_loader)
    if epoch % 10 == 0:
        report_loss(D_loss_cat, D_loss_gauss, G_loss, recon_loss, samples)
        print('Classification loss: {}'.format(class_loss.data[0]))
        print('Validation:')
        predict_cat(Q, valid_loader)


train_end = time.time()
predict_cat(Q, train_labeled_loader)
predict_cat(Q, valid_loader)
print('Total train time: {} seconds'.format(train_end - train_start))