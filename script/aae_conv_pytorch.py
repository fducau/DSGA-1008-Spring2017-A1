import time
from utils import *
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import nninit
import itertools
from vizualization import * 
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
z_dim = 10
X_dim = 784
y_dim = 10
cnt = 0
momentum = 0.1
convolutional = True
train_batch_size = 100
valid_batch_size = 100


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
#augment_dataset(trainset_labeled, b=100, k=2)

#print('Augmented dataset to size: {}'.format(trainset_labeled.k)) 

train_labeled_loader = torch.utils.data.DataLoader(trainset_labeled,
                                                   batch_size=train_batch_size,
                                                   shuffle=True, **kwargs)

#train_augmented_loader = torch.utils.data.DataLoader(augmented,
#                                                   batch_size=train_batch_size,
#                                                   shuffle=True, **kwargs)

train_unlabeled_loader = torch.utils.data.DataLoader(trainset_unlabeled,
                                                     batch_size=train_batch_size,
                                                     shuffle=True, **kwargs)

valid_loader = torch.utils.data.DataLoader(validset, batch_size=valid_batch_size, shuffle=True)


N = 640
##################################
# Define Networks
##################################
# Encoder
# Encoder
class Q_net(nn.Module):
    def __init__(self):
        super(Q_net, self).__init__()
        self.lin1 = nn.Linear(X_dim, N)
        nninit.kaiming_uniform(self.lin1.weight)
        self.lin2 = nn.Linear(N, N)
        nninit.kaiming_uniform(self.lin2.weight)
        # Categorical code
        self.lin3cat = nn.Linear(N, n_classes)
        nninit.kaiming_uniform(self.lin3cat.weight)
        # self.lin4cat = nn.Linear(N, n_classes)

        # Gaussian
        self.lin3gauss = nn.Linear(N, z_dim)        
        nninit.kaiming_uniform(self.lin3gauss.weight)
        # self.lin4gauss = nn.Linear(N, z_dim)

    def forward(self, x):
        x = self.lin1(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(x)

        xcat = F.softmax(self.lin3cat(x))
        xgauss = self.lin3gauss(x)

        return xcat, xgauss

class Q_net_conv(nn.Module):
    def __init__(self):
        super(Q_net_conv, self).__init__()
        self.conv1 = nn.Conv2d(1, 100, kernel_size=3)
        nninit.kaiming_uniform(self.conv1.weight)
        self.conv2 = nn.Conv2d(100, 80, kernel_size=3)
        self.conv3 = nn.Conv2d(80, 40, kernel_size=4)
        nninit.kaiming_uniform(self.conv2.weight)
        self.conv2_drop = nn.Dropout2d(p=0.01)

        self.lin1cat = nn.Linear(N, n_classes)
        nninit.kaiming_uniform(self.lin1cat.weight)
        self.lin2cat = nn.Linear(N, n_classes)
        self.lin1gauss = nn.Linear(N, z_dim)
        nninit.kaiming_uniform(self.lin1gauss.weight)
        self.lin2gauss = nn.Linear(N, z_dim)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2, stride=2)
        x = F.relu(x)
        x = self.conv2_drop(self.conv2(x))      
        x = F.relu(x)
        x = F.max_pool2d(self.conv3(x), 2, stride=2)
        x = F.relu(x)

        x = x.view(-1, N)

        xcat = F.relu(self.lin1cat(x))
        # x1 = F.dropout(x1, p=0.3, training=self.training)
        # xcat = F.relu(self.lin2cat(xcat))
        xcat = F.softmax(xcat)

        xgauss = self.lin1gauss(x)
        # xgauss = F.relu(xgauss)
        # xgauss = F.dropout(xgauss, training=self.training)
        # xgauss = self.lin2gauss(xgauss)

        return xcat, xgauss


# Decoder
class P_net_conv(nn.Module):
    def __init__(self):
        super(P_net_conv, self).__init__()
        self.lin1 = nn.Linear(z_dim + n_classes, N)
        nninit.kaiming_uniform(self.lin1.weight)
        self.conv1 = nn.Conv2d(100, 1, kernel_size=5)
        nninit.kaiming_uniform(self.conv1.weight)
        self.conv2 = nn.Conv2d(40, 100, kernel_size=5)
        nninit.kaiming_uniform(self.conv2.weight)
        self.conv2_drop = nn.Dropout2d(p=0.01)
        self.upsample2 = UpsamplingNearest2d(scale_factor=3)
        self.upsample1 = UpsamplingNearest2d(scale_factor=4)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = x.view(-1, 40, 4, 4)
        x = self.upsample2(x)
        x = self.conv2(x)
        x = self.upsample1(x)
        x = self.conv1(x)
        x = F.sigmoid(x)

        return x


class P_net(nn.Module):
    def __init__(self):
        super(P_net, self).__init__()
        self.lin1 = nn.Linear(z_dim + n_classes, N)
        nninit.kaiming_uniform(self.lin1.weight)
        self.lin2 = nn.Linear(N, N)
        nninit.kaiming_uniform(self.lin2.weight)
        self.lin3 = nn.Linear(N, X_dim)
        nninit.kaiming_uniform(self.lin3.weight)

    def forward(self, x):
        x = self.lin1(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(x)
        x = self.lin2(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin3(x)
        return F.sigmoid(x)


# Discriminator
class D_net_cat(nn.Module):
    def __init__(self):
        super(D_net_cat, self).__init__()
        self.lin1 = nn.Linear(10, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 1)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        return F.sigmoid(x)


class D_net_gauss(nn.Module):
    def __init__(self):
        super(D_net_gauss, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 1)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
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
          P_decoder, Q_encoder, Q_semi_supervised, Q_generator,
          D_cat_solver, D_gauss_solver,
          train_labeled_loader, train_unlabeled_loader):
    
    TINY = 1e-15
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


            # Load batch
            X = X * 0.3081 + 0.1307
            gaussian_length = train_batch_size * 28 * 28
            means = torch.from_numpy(np.array([0.] * gaussian_length).astype('float32'))
            stds = torch.from_numpy(np.array([0.5] * gaussian_length).astype('float32'))
            gaussian_noise = torch.normal(means, stds)
            gaussian_noise.resize_as_(X)

            X_noise = X + gaussian_noise
            if not convolutional:
                X.resize_(train_batch_size, X_dim)
                X_noise.resize_(train_batch_size, X_dim)

            X, target = Variable(X), Variable(target)
            X_noise = Variable(X_noise)
            if cuda:
                X, target = X.cuda(), target.cuda()
                X_noise = X_noise.cuda()

            # Init gradients
            P.zero_grad()
            Q.zero_grad()
            D_cat.zero_grad()
            D_gauss.zero_grad()

            # Reconstruction phase
            #if not labeled:
            z_sample = torch.cat(Q(X_noise), 1)
            X_sample = P(z_sample)

            recon_loss = F.binary_cross_entropy(X_sample + TINY, X.resize(train_batch_size, X_dim) + TINY)
            # mse_loss = torch.nn.MSELoss()
            # recon_loss = mse_loss(X_sample, X.resize(train_batch_size, X_dim))

            recon_loss.backward()
            P_decoder.step()
            Q_encoder.step()

            P.zero_grad()
            Q.zero_grad()
            D_cat.zero_grad()
            D_gauss.zero_grad()

            """ Regularization phase """
            # Discriminator
            Q.eval()
            z_real_cat = sample_categorical(train_batch_size, n_classes=10)
            z_real_gauss = Variable(torch.randn(train_batch_size, z_dim))
            if cuda:
                z_real_cat = z_real_cat.cuda()
                z_real_gauss = z_real_gauss.cuda()

            z_fake_cat, z_fake_gauss = Q(X)

            D_real_cat = D_cat(z_real_cat)
            D_real_gauss = D_gauss(z_real_gauss)
            D_fake_cat = D_cat(z_fake_cat)
            D_fake_gauss = D_gauss(z_fake_gauss)

            D_loss_cat = -torch.mean(torch.log(D_real_cat + TINY) + torch.log(1 - D_fake_cat + TINY))
            D_loss_gauss = -torch.mean(torch.log(D_real_gauss + TINY) + torch.log(1 - D_fake_gauss + TINY))

            D_loss = D_loss_cat + D_loss_gauss

            D_loss.backward()
            D_cat_solver.step()
            D_gauss_solver.step()

            P.zero_grad()
            Q.zero_grad()
            D_cat.zero_grad()
            D_gauss.zero_grad()

            # Generator
            Q.train()
            z_fake_cat, z_fake_gauss = Q(X)

            D_fake_cat = D_cat(z_fake_cat)
            D_fake_gauss = D_gauss(z_fake_gauss)

            G_loss = - torch.mean(torch.log(D_fake_cat + TINY)) - torch.mean(torch.log(D_fake_gauss + TINY))
            G_loss.backward()
            Q_generator.step()

            P.zero_grad()
            Q.zero_grad()
            D_cat.zero_grad()
            D_gauss.zero_grad()

                #pseudo_labels = predict_labels(Q, X)
                #pred, _ = Q(X)
                #class_loss = F.cross_entropy(pred, pseudo_labels)
                #class_loss.backward()
                #Q_pseudo.step()

                #P.zero_grad()
                #Q.zero_grad()
                #D_cat.zero_grad()
                #D_gauss.zero_grad()

            # Classification Loss
            if labeled:
                pred, _ = Q(X)
                class_loss = F.cross_entropy(pred, target)
                class_loss.backward()
                Q_semi_supervised.step()

                P.zero_grad()
                Q.zero_grad()
                D_cat.zero_grad()
                D_gauss.zero_grad()


            # D_loss_cat = recon_loss
            # D_loss_gauss = recon_loss
            # G_loss = recon_loss
            # class_loss = recon_loss
    return D_loss_cat, D_loss_gauss, G_loss, recon_loss, class_loss

def pretrain(Q, Q_semi_supervised, data_loader):
    for batch_idx, (X, target) in enumerate(data_loader):
        # Load batch
        X = X * 0.3081 + 0.1307
        gaussian_length = train_batch_size * 28 * 28
        means = torch.from_numpy(np.array([0.] * gaussian_length).astype('float32'))
        stds = torch.from_numpy(np.array([0.3] * gaussian_length).astype('float32'))
        gaussian_noise = torch.normal(means, stds)
        gaussian_noise.resize_as_(X)
        X_noise = X + gaussian_noise
        if not convolutional:
            X.resize_(train_batch_size, X_dim)
            X_noise.resize_(train_batch_size, X_dim)
        X, target = Variable(X), Variable(target)
        X_noise = Variable(X_noise)
        
        if cuda:
            X, target = X.cuda(), target.cuda()
            X_noise = X_noise.cuda()

        # Init gradients
        Q.zero_grad()
        pred, _ = Q(X)
        class_loss = F.cross_entropy(pred, target)
        class_loss.backward()
        Q_semi_supervised.step()

        Q.zero_grad()



def report_loss(D_loss_cat, D_loss_gauss, G_loss, recon_loss):
        print('Epoch-{}; D_loss_cat: {:.4}: D_loss_gauss: {:.4}; G_loss: {:.4}; recon_loss: {:.4}'
              .format(epoch, D_loss_cat.data[0], D_loss_gauss.data[0],
                      G_loss.data[0], recon_loss.data[0]))


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

def predict_labels(Q, X):
    Q.eval()
    output = Q(X)[0]
    pl = output.data.max(1)[1]
    pl.resize_(X.size()[0])
    return Variable(pl)


def predict_cat(Q, data_loader):
    Q.eval()
    labels = []
    test_loss = 0   
    correct = 0

    for batch_idx, (X, target) in enumerate(data_loader):
        X = X * 0.3081 + 0.1307
        if not convolutional:
            X.resize_(data_loader.batch_size, X_dim)
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
    return 100. * correct / len(data_loader.dataset)


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

if convolutional:
    gen_lr = 0.01
    semi_lr = 0.1
    reg_lr = 0.01
else:
    gen_lr = 0.0001
    semi_lr = 0.001
    reg_lr = 0.0001


if convolutional:
    P_decoder = optim.SGD(P.parameters(), lr=gen_lr, momentum=0.9)
    Q_encoder = optim.SGD(Q.parameters(), lr=gen_lr, momentum=0.9)
    
    Q_semi_supervised = optim.SGD(Q.parameters(), lr=semi_lr, momentum=0.9)
    
    
    Q_generator = optim.SGD(Q.parameters(), lr=reg_lr, momentum=0.1)
    D_gauss_solver = optim.SGD(D_gauss.parameters(), lr=reg_lr, momentum=0.1)
    D_cat_solver = optim.SGD(D_cat.parameters(), lr=reg_lr, momentum=0.1)

else:
    P_decoder = optim.Adam(P.parameters(), lr=gen_lr)
    Q_encoder = optim.Adam(Q.parameters(), lr=gen_lr)
    
    Q_semi_supervised = optim.Adam(Q.parameters(), lr=semi_lr)
    
    Q_generator = optim.Adam(Q.parameters(), lr=reg_lr)
    D_gauss_solver = optim.Adam(D_gauss.parameters(), lr=reg_lr)
    D_cat_solver = optim.Adam(D_cat.parameters(), lr=reg_lr)


train_labeled_loader = torch.utils.data.DataLoader(trainset_labeled,
                                                   batch_size=train_batch_size,
                                                   shuffle=True, **kwargs)
train_unlabeled_loader = torch.utils.data.DataLoader(trainset_unlabeled,
                                                     batch_size=train_batch_size,
                                                     shuffle=True, **kwargs)
valid_loader = torch.utils.data.DataLoader(validset, batch_size=valid_batch_size,
                                           shuffle=True)

epochs = 5000
train_start = time.time()
for epoch in range(epochs):
    if epoch == 500 or epoch == 1000:
        if not convolutional:
            P_decoder = optim.Adam(P.parameters(), lr=gen_lr/10.)
            Q_encoder = optim.Adam(Q.parameters(), lr=gen_lr/10.)
    
            Q_semi_supervised = optim.Adam(Q.parameters(), lr=semi_lr/10.)
    
            Q_generator = optim.Adam(Q.parameters(), lr=reg_lr/10.)
            D_gauss_solver = optim.Adam(D_gauss.parameters(), lr=reg_lr/10.)
            D_cat_solver = optim.Adam(D_cat.parameters(), lr=reg_lr/10.)
        else:
            P_decoder = optim.SGD(P.parameters(), lr=gen_lr, momentum=0.9)
            Q_encoder = optim.SGD(Q.parameters(), lr=gen_lr, momentum=0.9)
    
            Q_semi_supervised = optim.SGD(Q.parameters(), lr=semi_lr, momentum=0.5)
    
    
            Q_generator = optim.SGD(Q.parameters(), lr=reg_lr, momentum=0.1)
            D_gauss_solver = optim.SGD(D_gauss.parameters(), lr=reg_lr, momentum=0.1)
            D_cat_solver = optim.SGD(D_cat.parameters(), lr=reg_lr, momentum=0.1)

    D_loss_cat, D_loss_gauss, G_loss, recon_loss, class_loss = train(P, Q, D_cat,
                                                                     D_gauss, P_decoder,
                                                                     Q_encoder, Q_semi_supervised,
                                                                     Q_generator,
                                                                     D_cat_solver, D_gauss_solver,
                                                                     train_labeled_loader,
                                                                     train_unlabeled_loader)
    if epoch % 10 == 0:
        report_loss(D_loss_cat, D_loss_gauss, G_loss, recon_loss)
        print('Classification Loss: {:.3}'.format(class_loss.data[0]))
        print('Validation accuracy: {} %'.format(predict_cat(Q, valid_loader)))

train_end = time.time()
print predict_cat(Q, train_labeled_loader)
print predict_cat(Q, valid_loader)
print('Total train time: {} seconds'.format(train_end - train_start))
