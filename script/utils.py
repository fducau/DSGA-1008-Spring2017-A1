import torch
import numpy as np
from numpy.random import uniform
from scipy.ndimage import map_coordinates
from scipy.ndimage.filters import gaussian_filter


def augment_dataset(train_loader, trainset_labeled, k=2):
    for i in range(k - 1):
        augmented_data, augmented_labels = [], []
        for img_batch, labels in iter(train_loader):
            augmented_data.extend(elastic_transform(img_batch, sigma=4, alpha=34))
            augmented_labels.extend(labels)

        augmented_data = np.array(augmented_data)
        augmented_labels = np.array(augmented_labels)
        data = np.concatenate((trainset_labeled.train_data.numpy(), augmented_data))
        labels = np.concatenate((trainset_labeled.train_labels.numpy(), augmented_labels))
        trainset_labeled.train_data = torch.from_numpy(data)
        trainset_labeled.train_labels = torch.from_numpy(labels)
        trainset_labeled.k = data.shape[0]

    return trainset_labeled


def elastic_transform(img_batch, sigma=4, alpha=34):
    img_batch = img_batch.numpy()
    x_dim = img_batch.shape[2]
    y_dim = img_batch.shape[3]
    pos = np.array([[i, j] for i in range(x_dim) for j in range(y_dim)])
    pos = pos.transpose(1, 0).reshape(2, x_dim, y_dim)
    uniform_random_x = uniform(-1, 1, size=img_batch.shape[2:])
    uniform_random_y = uniform(-1, 1, size=img_batch.shape[2:])

    elastic_x = scipy.ndimage.filters.gaussian_filter(alpha * uniform_random_x,
                                sigma=sigma, mode='constant')
    elastic_y = scipy.ndimage.filters.gaussian_filter(alpha * uniform_random_y,
                                sigma=sigma, mode='constant')
    elastic_distortion_x = pos[0] + elastic_x
    elastic_distortion_y = pos[1] + elastic_y
    elastic = np.array([elastic_distortion_x, elastic_distortion_y])

    transformed = []
    batch_size = img_batch.shape[0]

    for i in range(batch_size):
        transformed.append(scipy.ndimage.map_coordinates(img_batch[i][-1, :, :],
                                           elastic, prefilter=False, mode='reflect'))
    return transformed
