from __future__ import print_function
import scipy
import pickle
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import map_coordinates
from numpy.random import uniform
from torch.autograd import Variable


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--elastic-augment', type=bool, default=False, metavar='N',
                    help='Whether to augment dataset using elastic transformations')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout drop probability')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

params_filename = 'best_model.pkl'

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 100, kernel_size=5)
        self.conv2 = nn.Conv2d(100, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=args.dropout)

        self.fc1 = nn.Linear(320, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)


def train(model, optimizer, epoch, train_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def test(model, epoch, valid_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in valid_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(valid_loader)  # loss function already averages over batch size

    accuracy = 100. * correct / len(valid_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(valid_loader.dataset),
        accuracy))
    return accuracy


def save_model(model):
    print('Best model so far, saving it...')
    torch.save(model.state_dict(), params_filename)


def save_curves(curves):
    curves_filename = 'cnn_curves.pkl'
    pickle.dump(curves, open(curves_filename, 'w'))


def concat_sets(original, new_data, new_labels):
    new_data = np.array(new_data)
    new_labels = np.array(new_labels)
    data = np.concatenate((original.train_data.numpy(), new_data))
    labels = np.concatenate((original.train_labels.numpy(), new_labels))

    data = torch.from_numpy(data)
    labels = torch.from_numpy(labels)
    return data, labels


def augment_unlabeled(trainset_labeled, trainset_unlabeled, unlabeled_loader, i_augment, k=2):

    model = Net()
    if args.cuda:
        model.cuda()
    model.load_state_dict(torch.load(params_filename))

    augmented_data, augmented_labels = None, np.array([])
    model.eval()
    for i, (data, target) in enumerate(unlabeled_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        # if augmented_data is not None:
        #     augmented_data = np.concatenate((augmented_data, data.cpu().numpy()))
        # else:
        #     augmented_data = data.cpu().numpy()

        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        temp = output.data.max(1)[1].cpu().numpy().reshape(-1)
        augmented_labels = np.concatenate((augmented_labels, temp))
        if i >= k:
            break

    #augmented_data = augmented_data.reshape(augmented_data.shape[0], 28, 28)
    
    length = len(data) * (i+1)
    init = i_augment * length
    end = init + length
    augmented_data = trainset_unlabeled.train_data[init:end].numpy()
    augmented_labels = augmented_labels.astype(np.int64)
    data, labels = concat_sets(trainset_labeled, augmented_data, augmented_labels)
    trainset_labeled.train_data = data
    trainset_labeled.train_labels = labels
    trainset_labeled.k = len(data)


def augment_dataset(trainset_labeled, b=50, k=2, sigma=4, alpha=34):
    # Modifies trainset_labeled inline
    X = trainset_labeled.train_data
    Y = trainset_labeled.train_labels

    batches = [(X[i:i + b], Y[i:i + b]) for i in xrange(0, len(X), b)]

    augmented_data, augmented_labels = [], []
    for i in range(k):
        for img_batch, labels in batches:
            augmented_data.extend(elastic_transform(img_batch, sigma=4, alpha=34))
            augmented_labels.extend(labels)

    data, labels = concat_sets(trainset_labeled, augmented_data, augmented_labels)
    trainset_labeled.train_data = data
    trainset_labeled.train_labels = labels
    trainset_labeled.k = len(data)


def elastic_transform(img_batch, sigma=4, alpha=34):
    img_batch = img_batch.numpy()
    x_dim = img_batch.shape[1]
    y_dim = img_batch.shape[2]
    pos = np.array([[i, j] for i in range(x_dim) for j in range(y_dim)])
    pos = pos.transpose(1, 0).reshape(2, x_dim, y_dim)
    uniform_random_x = uniform(-1, 1, size=img_batch.shape[1:])
    uniform_random_y = uniform(-1, 1, size=img_batch.shape[1:])

    elastic_x = gaussian_filter(alpha * uniform_random_x,
                                sigma=sigma, mode='constant')
    elastic_y = gaussian_filter(alpha * uniform_random_y,
                                sigma=sigma, mode='constant')
    elastic_distortion_x = pos[0] + elastic_x
    elastic_distortion_y = pos[1] + elastic_y
    elastic = np.array([elastic_distortion_x, elastic_distortion_y])

    transformed = []
    batch_size = img_batch.shape[0]

    for i in range(batch_size):
        transformed.append(map_coordinates(img_batch[i], elastic, order=1,
                                           prefilter=False, mode='reflect'))
    return transformed


def main():
    print('loading data!')
    data_path = '../data/'
    trainset_labeled = pickle.load(open(data_path + "train_labeled.p", "rb"))
    validset = pickle.load(open(data_path + "validation.p", "rb"))
    trainset_unlabeled = pickle.load(open(data_path + "train_unlabeled.p", "rb"))
    trainset_unlabeled.train_labels = torch.from_numpy(np.array([-1] * 47000))

    if args.elastic_augment:
        print('Augmenting dataset!')
        augment_dataset(trainset_labeled, b=50, k=17)

        print('Augmented dataset to size: {}'.format(trainset_labeled.k))
        # filename = 'train_augmented.p'
        # print('Saving augmented dataset to {}{}'.format(data_path, filename))
        # output = open(data_path + filename, 'wb')
        # pickle.dump(trainset_labeled, output)

    train_loader = torch.utils.data.DataLoader(trainset_labeled, batch_size=64,
                                               shuffle=True, **kwargs)
    train_for_loss = torch.utils.data.DataLoader(trainset_labeled, batch_size=64,
                                                 shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=64,
                                               shuffle=True)
    unlabeled_loader = torch.utils.data.DataLoader(trainset_unlabeled, batch_size=100,
                                                   shuffle=False, **kwargs)

    model = Net()
    if args.cuda:
        model.cuda()

    curves = {}
    curves['train'] = []
    curves['valid'] = []

    best = float('-inf')
    not_learning = 0
    lr = args.lr
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)

    i_augment = 0
    for epoch in range(1, args.epochs + 1):

        if not_learning > 5:
            lr /= 10
            not_learning = 0
            print('Changing learning rate to {:.8f}'.format(lr))
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)

        # if epoch >= 40 and epoch % 10 == 0:
        #      print('Adding unlabeled data')
        #      augment_unlabeled(trainset_labeled, trainset_unlabeled, unlabeled_loader, i_augment, k=3)
        #      i_augment += 1
        #      train_loader = torch.utils.data.DataLoader(trainset_labeled, batch_size=64,
        #                                                 shuffle=True, **kwargs)
        #      lr = args.lr
        #      optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)

        train(model, optimizer, epoch, train_loader)
        accuracy_train = test(model, epoch, train_for_loss)
        accuracy_valid = test(model, epoch, valid_loader)

        if accuracy_valid > best:
            save_model(model)
            best = accuracy_valid
            not_learning = 0
        else:
            not_learning += 1

        curves['train'].append(accuracy_train)
        curves['valid'].append(accuracy_valid)

    save_curves(curves)


if __name__ == '__main__':
    main()
