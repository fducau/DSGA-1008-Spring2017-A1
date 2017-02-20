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
z_dim = 12
X_dim = 784
y_dim = 10
h_dim = 128
cnt = 0
lr = 1e-2
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

# Discriminator
class MLP_net(nn.Module):
    def __init__(self):
        super(MLP_net, self).__init__()
        self.lin1 = nn.Linear(z_dim, 10)
        self.lin2 = nn.Linear(10, 10)
        self.lin3 = nn.Linear(48, 24)
        self.lin4 = nn.Linear(24,10)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        #x = self.lin2(x)
        #x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        #x = self.lin3(x)
        #x = F.relu(x)
        #x = self.lin4(x)
        return F.log_softmax(x)



def sample_X(size, include_y=False):
    X, y = mnist.train.next_batch(size)
    X = Variable(torch.from_numpy(X))

    if include_y:
        y = np.argmax(y, axis=1).astype(np.int)
        y = Variable(torch.from_numpy(y))
        return X, y

    return X


def train_MLP(MLP, data_loader, MLP_solver):
    MLP.train()
    for batch_idx, (X, target) in enumerate(data_loader):
        if batch_idx * data_loader.batch_size + data_loader.batch_size > data_loader.dataset.k:
            continue
        MLP.zero_grad()
        X, target = Variable(X), Variable(target)
        if cuda:
            X, target = X.cuda(), target.cuda()
        target = target.resize(data_loader.batch_size)
        output = MLP(X)
        loss = F.nll_loss(output, target)
        loss.backward()
        MLP_solver.step()
    return loss

def test_MLP(MLP, data_loader):
    MLP.eval()
    test_loss = 0
    correct = 0
    for X, target in data_loader:
        X, target = Variable(X), Variable(target)
        if cuda:
            X, target = X.cuda(), target.cuda()
        target = target.resize(data_loader.batch_size)
        output = MLP(X)
        test_loss += F.nll_loss(output, target).data[0]

        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(data_loader)  # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))




def train(P, Q, D, P_solver, Q_solver, D_solver, data_loader, MLP=None, MLP_solver=None):
    Q.train()
    P.train()
    D.train()
    if MLP is not None:
        MLP.train()

    for batch_idx, (X, target) in enumerate(data_loader):
        if batch_idx * data_loader.batch_size + data_loader.batch_size > data_loader.dataset.k:
            continue

        P.zero_grad()
        Q.zero_grad()
        D.zero_grad()
        if MLP is not None:
            MLP.train()

        X = X * 0.3081 + 0.1307

        X.resize_(train_batch_size, X_dim)
        X, target = Variable(X), Variable(target)

        # Check if data has negative values and ignore
        if X.min().data[0] < 0:
            print 'Skipping bad input. '
            #raise ValueError
            continue

        #X = sample_X(train_batch_size)

        if cuda:
            X, target = X.cuda(), target.cuda()

        # Reconstruction phase
        z_sample = Q(X)
        X_sample = P(z_sample)

        #if X_sample.min().data[0] < 0:
        #    print 'breaked2'
        #    break

        # Use epsilon to avoid log(0) case
        epsilon = 1e-8
        recon_loss = F.binary_cross_entropy(X_sample + epsilon, X + epsilon)

        recon_loss.backward()
        P_solver.step()
        Q_solver.step()

        P.zero_grad()
        Q.zero_grad()
        D.zero_grad()
        if MLP is not None:
            MLP.train()
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
        if MLP is not None:
            MLP.train()
        # Generator
        z_fake = Q(X)
        D_fake = D(z_fake)

        G_loss = -torch.mean(torch.log(D_fake))

        G_loss.backward()
        Q_solver.step()

        P.zero_grad()
        Q.zero_grad()
        D.zero_grad()
        if MLP is not None:
            MLP.train()

        class_loss = float('nan')
        if MLP is not None:
            z_sample = Q(X)
            pred = MLP(z_sample)
            class_loss = F.nll_loss(pred, target)
            class_loss.backward()
            MLP_solver.step()
            Q_solver.step()

            P.zero_grad()
            Q.zero_grad()
            D.zero_grad()
            MLP.train()

        if D_loss.data[0] == float('nan'):
            print 'D_loss hurt'
            raise ValueError
        if G_loss.data[0] == float('nan'):
            print 'G_loss hurt'
            raise ValueError
        if recon_loss.data[0] == float('nan'):
            print 'recon_loss hurt'
            raise ValueError

        samples = P(z_sample)
        xsample = X
        # if batch_idx == 100:
        #     print('Epoch: {} D: {} G: {} R: {}'.format(epoch, D_loss.data[0], G_loss.data[0], recon_loss.data[0]))

    return D_loss, G_loss, recon_loss, class_loss, (samples.data[0], xsample.data[0])

def report_loss(D_loss, G_loss, recon_loss, samples=None):
        print('Epoch-{}; D_loss: {:.4}; G_loss: {:.4}; recon_loss: {:.4}'
              .format(epoch, D_loss.data[0], G_loss.data[0], recon_loss.data[0]))

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
    labels_onehot = np.zeros([loader.dataset.k, 10])
    labels_onehot[np.arange(loader.dataset.k), labels] = 1
    return z_values, labels

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

Q_solver = optim.Adam(Q.parameters(), lr=lr/100.)
P_solver = optim.Adam(P.parameters(), lr=lr/100.)
D_solver = optim.Adam(D.parameters(), lr=lr/100.)

MLP = MLP_net()
if cuda:
    MLP.cuda()
MLP_solver = optim.SGD(MLP.parameters(), lr=lr/50., momentum=0.4)

for epoch in range(epochs):
    D_loss, G_loss, recon_loss, class_loss, samples = train(P, Q, D, P_solver, Q_solver, D_solver,
                                                            train_labeled_loader,
                                                            MLP, MLP_solver)

    D_loss_u, G_loss_u, recon_loss_u, _, samples_u = train(P, Q, D, P_solver, Q_solver, D_solver,
                                                                    train_unlabeled_loader)

    # Print and plot every now and then
    if epoch % 5 == 0:
        print('Loss in Labeled')
        report_loss(D_loss, G_loss, recon_loss, samples)
        print('Classification loss: {}'.format(class_loss.data[0]))
        print('Loss in UNLabeled')
        report_loss(D_loss_u, G_loss_u, recon_loss_u)



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

