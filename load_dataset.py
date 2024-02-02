from config import *
import os
import itertools
import numpy as np
from typing import Any, Callable, Optional, Tuple
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import torchvision.transforms as T
from torchvision.datasets import CIFAR100, CIFAR10, FashionMNIST, SVHN, ImageFolder, Caltech256, VisionDataset
import pdb

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, secondary_indices, primary_indices, batch_size, secondary_batch_size, unlabeled_size_limit=None):
        """
        :param primary_indices: unlabeled idxs
        :param secondary_indices: labeled idxs
        :param batch_size: total batch_size
        :param secondary_batch_size: labeled batch_size
        :param unlabeled_size_limit:
        """
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size # unlabeled batch_size
        self.unlabeled_size_limit = unlabeled_size_limit

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices, self.unlabeled_size_limit)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(secondary_iter, self.secondary_batch_size),
                    grouper(primary_iter, self.primary_batch_size))
        )

    def __len__(self):
        if self.unlabeled_size_limit is None:
            return len(self.primary_indices) // self.primary_batch_size
        else:
            return self.unlabeled_size_limit // self.primary_batch_size

def iterate_once(iterable, unlabeled_size_limit=None):
    if unlabeled_size_limit is None:
        return np.random.permutation(iterable)
    else:
        result = np.random.permutation(iterable)[:unlabeled_size_limit]
        return result

def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    args = [iter(iterable)] * n
    return zip(*args)

def get_img_num_per_cls(cls_num, imb_type, imb_factor):
    img_max = 5000
    img_num_per_cls = []
    if imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
    elif imb_type == 'step':
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max))
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max * imb_factor))
    else:
        img_num_per_cls.extend([int(img_max)] * cls_num)
    return img_num_per_cls

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


class MyDataset(Dataset):
    def __init__(self, dataset_name, train_flag, transf):
        self.dataset_name = dataset_name
        if self.dataset_name == "cifar10":
            self.cifar10 = CIFAR10('../data-local/cifar10', train=train_flag,
                                    download=True, transform=transf)
        if self.dataset_name == "cifar100":
            self.cifar100 = CIFAR100('../data-local/cifar100', train=train_flag,
                                    download=True, transform=transf)
        if self.dataset_name == "fashionmnist":
            self.fmnist = FashionMNIST('../data-local/fashionMNIST', train=train_flag,
                                    download=True, transform=transf)
        if self.dataset_name == "svhn":
            self.svhn = SVHN('../data-local/svhn', split="train",
                                    download=True, transform=transf)
        if self.dataset_name == 'caltech256':
            # self.caltech256 = Caltech256('../data-local/caltech256', train=train_flag, download=True, transform=transf)
            self.caltech256 = ImageFolder('../data-local/caltech256/by-image/train+val' , transf)
        if self.dataset_name == 'tiny_imagenet':
            self.tiny_imagenet = ImageFolder('../data-local/tiny-imagenet/by-image/train', transf)
        if self.dataset_name == "cifar10im":
            self.cifar10 = CIFAR10('../data-local/cifar10', train=train_flag,
                                    download=True, transform=transf)
            imbal_class_counts = [50, 5000] * 5
            targets = np.array(self.cifar10.targets)
            classes, class_counts = np.unique(targets, return_counts=True)
            nb_classes = len(classes)
            class_indices = [np.where(targets == i)[0] for i in range(nb_classes)]
            # Get imbalanced number of instances
            imbal_class_indices = [class_idx[:class_count] for class_idx, class_count in zip(class_indices, imbal_class_counts)]
            imbal_class_indices = np.hstack(imbal_class_indices)

            # Set target and data to dataset
            self.cifar10.targets = targets[imbal_class_indices]
            self.cifar10.data = self.cifar10.data[imbal_class_indices]
        if self.dataset_name == "cifar10imlong":
            self.cifar10 = CIFAR10('../data-local/cifar10', train=train_flag,
                                   download=True, transform=transf)

            targets = np.array(self.cifar10.targets)
            classes, class_counts = np.unique(targets, return_counts=True)
            nb_classes = len(classes)
            imbal_class_counts = get_img_num_per_cls(nb_classes, imb_type='exp', imb_factor=0.01)
            class_indices = [np.where(targets == i)[0] for i in range(nb_classes)]
            # Get imbalanced number of instances
            imbal_class_indices = [class_idx[:class_count] for class_idx, class_count in
                                   zip(class_indices, imbal_class_counts)]
            imbal_class_indices = np.hstack(imbal_class_indices)

            # Set target and data to dataset
            self.cifar10.targets = targets[imbal_class_indices]
            self.cifar10.data = self.cifar10.data[imbal_class_indices]
        if self.dataset_name == "cifar100im":
            self.cifar100 = CIFAR100('../data-local/cifar100', train=train_flag,
                                    download=True, transform=transf)
            imbal_class_counts = [5, 500] * 50
            targets = np.array(self.cifar100.targets)
            classes, class_counts = np.unique(targets, return_counts=True)
            nb_classes = len(classes)
            class_indices = [np.where(targets == i)[0] for i in range(nb_classes)]
            # Get imbalanced number of instances
            imbal_class_indices = [class_idx[:class_count] for class_idx, class_count in zip(class_indices, imbal_class_counts)]
            imbal_class_indices = np.hstack(imbal_class_indices)
            # Set target and data to dataset
            self.cifar100.targets = targets[imbal_class_indices]
            self.cifar100.data = self.cifar100.data[imbal_class_indices]

        if self.dataset_name == "cifar100imlong":
            self.cifar100 = CIFAR100('../data-local/cifar100', train=train_flag,
                                   download=True, transform=transf)

            targets = np.array(self.cifar100.targets)
            classes, class_counts = np.unique(targets, return_counts=True)
            nb_classes = len(classes)
            imbal_class_counts = get_img_num_per_cls(nb_classes, imb_type='exp', imb_factor=0.01)
            class_indices = [np.where(targets == i)[0] for i in range(nb_classes)]
            # Get imbalanced number of instances
            imbal_class_indices = [class_idx[:class_count] for class_idx, class_count in
                                   zip(class_indices, imbal_class_counts)]
            imbal_class_indices = np.hstack(imbal_class_indices)

            # Set target and data to dataset
            self.cifar100.targets = targets[imbal_class_indices]
            self.cifar100.data = self.cifar100.data[imbal_class_indices]

    def __getitem__(self, index):
        if self.dataset_name == "cifar10":
            data, target = self.cifar10[index]
        if self.dataset_name == "cifar10im":
            data, target = self.cifar10[index]
        if self.dataset_name == 'cifar10imlong':
            data, target = self.cifar10[index]
        if self.dataset_name == "cifar100":
            data, target = self.cifar100[index]
        if self.dataset_name == "cifar100im":
            data, target = self.cifar100[index]
        if self.dataset_name == "cifar100imlong":
            data, target = self.cifar100[index]
        if self.dataset_name == "fashionmnist":
            data, target = self.fmnist[index]
        if self.dataset_name == "svhn":
            data, target = self.svhn[index]
        if self.dataset_name == 'caltech256':
            data, target = self.caltech256[index]
        if self.dataset_name == 'tiny_imagenet':
            data, target = self.tiny_imagenet[index]
        return data, target, index

    def __len__(self):
        if self.dataset_name == "cifar10":
            return len(self.cifar10)
        elif self.dataset_name == "cifar10im":
            return len(self.cifar10)
        elif self.dataset_name == 'cifar10imlong':
            return len(self.cifar10)
        elif self.dataset_name == "cifar100":
            return len(self.cifar100)
        elif self.dataset_name == "cifar100im":
            return len(self.cifar100)
        elif self.dataset_name == "cifar100imlong":
            return len(self.cifar100)
        elif self.dataset_name == "fashionmnist":
            return len(self.fmnist)
        elif self.dataset_name == "svhn":
            return len(self.svhn)
        elif self.dataset_name == 'caltech256':
            return len(self.caltech256)
        elif self.dataset_name == 'tiny_imagenet':
            return len(self.tiny_imagenet)

class Transform_Aug:
    def __init__(self, train_transform, test_transform, args):
        self.args = args
        self.train_transform = train_transform
        self.test_transform = test_transform
    def __call__(self, inp):
        if self.args.use_semi_learning:
            return [self.train_transform(inp)[0] for _ in range(self.args.num_iter)] + [self.test_transform(inp)]
        else:
            return [self.train_transform(inp) for _ in range(self.args.num_iter)]  + [self.test_transform(inp)]


# Data
def load_dataset(dataset, args):
    if dataset == 'caltech256':
        train_transform = T.Compose([
            T.Resize(256),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            # T.Lambda(enhance),
            T.ToTensor(),
            T.Normalize([0.4850, 0.4560, 0.4060], [0.2290,  0.2240,  0.2250])
        ])
        test_transform = T.Compose([
            T.Resize(256),
            T.RandomCrop(224),
            T.ToTensor(),
            T.Normalize([0.4850, 0.4560, 0.4060], [0.2290,  0.2240,  0.2250])
            # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
        ])
    elif dataset == 'tiny_imagenet':
        train_transform = T.Compose([
            T.RandomRotation(10),
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.ToTensor(),
            T.Normalize([0.4850, 0.4560, 0.4060], [0.2290,  0.2240,  0.2250])
        ])
        test_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.4850, 0.4560, 0.4060], [0.2290,  0.2240,  0.2250])
        ])
    else:
        if args.use_semi_learning:
            train_transform = TransformTwice(T.Compose([
                T.RandomCrop(size=32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ]))
        else:
            train_transform = T.Compose([
                T.RandomCrop(size=32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]) # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
            ])
        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]) # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
        ])
    if args.method_type in ['ST-CoNAL-Aug', 'CSSAL']:
        aug_transform = Transform_Aug(train_transform, test_transform, args)

    if dataset == 'cifar10':
        data_train = CIFAR10('../data-local/cifar10', train=True, download=True, transform=train_transform)
        if args.method_type in ['ST-CoNAL-Aug', 'CSSAL']:
            data_unlabeled = MyDataset(dataset, True, aug_transform)
        else:
            data_unlabeled = MyDataset(dataset, True, test_transform)
        data_test  = CIFAR10('../data-local/cifar10', train=False, download=True, transform=test_transform)
        NUM_CLASSES = 10
        NUM_TRAIN = len(data_train)
        no_train = NUM_TRAIN
    elif dataset == 'cifar10im':
        data_train = CIFAR10('../data-local/cifar10', train=True, download=True, transform=train_transform)
        targets = np.array(data_train.targets)
        NUM_TRAIN = targets.shape[0]
        classes, class_counts = np.unique(targets, return_counts=True)
        nb_classes = len(classes)
        imb_class_counts = [50, 5000] * 5
        class_idxs = [np.where(targets == i)[0] for i in range(nb_classes)]
        imb_class_idx = [class_id[:class_count] for class_id, class_count in zip(class_idxs, imb_class_counts)]
        imb_class_idx = np.hstack(imb_class_idx)
        no_train = imb_class_idx.shape[0]
        data_train.targets = targets[imb_class_idx]
        data_train.data = data_train.data[imb_class_idx]
        if args.method_type in ['ST-CoNAL-Aug', 'CSSAL']:
            data_unlabeled = MyDataset(dataset, True, aug_transform)
        else:
            data_unlabeled = MyDataset(dataset, True, test_transform)
        data_test  = CIFAR10('../data-local/cifar10', train=False, download=True, transform=test_transform)
        NUM_TRAIN = len(data_train)
        NUM_CLASSES = 10
        no_train = NUM_TRAIN
    elif dataset == 'cifar10imlong':
        data_train = CIFAR10('../data-local/cifar10', train=True, download=True, transform=train_transform)
        targets = np.array(data_train.targets)
        NUM_TRAIN = targets.shape[0]
        classes, class_counts = np.unique(targets, return_counts=True)
        nb_classes = len(classes)
        imb_class_counts = get_img_num_per_cls(nb_classes, imb_type='exp', imb_factor=0.01)
        class_idxs = [np.where(targets == i)[0] for i in range(nb_classes)]
        imb_class_idx = [class_id[:class_count] for class_id, class_count in zip(class_idxs, imb_class_counts)]
        imb_class_idx = np.hstack(imb_class_idx)
        no_train = imb_class_idx.shape[0]
        data_train.targets = targets[imb_class_idx]
        data_train.data = data_train.data[imb_class_idx]
        if args.method_type in ['ST-CoNAL-Aug', 'CSSAL']:
            data_unlabeled = MyDataset(dataset, True, aug_transform)
        else:
            data_unlabeled = MyDataset(dataset, True, test_transform)
        data_test  = CIFAR10('../data-local/cifar10', train=False, download=True, transform=test_transform)
        NUM_TRAIN = len(data_train)
        NUM_CLASSES = 10
        no_train = NUM_TRAIN
    elif dataset == 'cifar100':
        data_train = CIFAR100('../data-local/cifar100', train=True, download=True, transform=train_transform)
        if args.method_type in ['ST-CoNAL-Aug', 'CSSAL']:
            data_unlabeled = MyDataset(dataset, True, aug_transform)
        else:
            data_unlabeled = MyDataset(dataset, True, test_transform)
        data_test  = CIFAR100('../data-local/cifar100', train=False, download=True, transform=test_transform)
        NUM_CLASSES = 100
        NUM_TRAIN = len(data_train)
        no_train = NUM_TRAIN
    elif dataset == 'cifar100im':
        data_train = CIFAR100('../data-local/cifar100', train=True, download=True, transform=train_transform)
        targets = np.array(data_train.targets)
        NUM_TRAIN = targets.shape[0]
        classes, class_counts = np.unique(targets, return_counts=True)
        nb_classes = len(classes)
        imb_class_counts = [5, 500] * 50
        class_idxs = [np.where(targets == i)[0] for i in range(nb_classes)]
        imb_class_idx = [class_id[:class_count] for class_id, class_count in zip(class_idxs, imb_class_counts)]
        imb_class_idx = np.hstack(imb_class_idx)
        no_train = imb_class_idx.shape[0]
        data_train.targets = targets[imb_class_idx]
        data_train.data = data_train.data[imb_class_idx]
        if args.method_type in ['ST-CoNAL-Aug', 'CSSAL']:
            data_unlabeled = MyDataset(dataset, True, aug_transform)
        else:
            data_unlabeled = MyDataset(dataset, True, test_transform)
        data_test  = CIFAR100('../data-local/cifar100', train=False, download=True, transform=test_transform)
        NUM_TRAIN = len(data_train)
        NUM_CLASSES = 100
    elif dataset == 'cifar100imlong':
        data_train = CIFAR100('../data-local/cifar100', train=True, download=True, transform=train_transform)
        targets = np.array(data_train.targets)
        NUM_TRAIN = targets.shape[0]
        classes, class_counts = np.unique(targets, return_counts=True)
        nb_classes = len(classes)
        imb_class_counts = get_img_num_per_cls(nb_classes, imb_type='exp', imb_factor=0.01)
        class_idxs = [np.where(targets == i)[0] for i in range(nb_classes)]
        imb_class_idx = [class_id[:class_count] for class_id, class_count in zip(class_idxs, imb_class_counts)]
        imb_class_idx = np.hstack(imb_class_idx)
        no_train = imb_class_idx.shape[0]
        data_train.targets = targets[imb_class_idx]
        data_train.data = data_train.data[imb_class_idx]
        if args.method_type in ['ST-CoNAL-Aug', 'CSSAL']:
            data_unlabeled = MyDataset(dataset, True, aug_transform)
        else:
            data_unlabeled = MyDataset(dataset, True, test_transform)
        data_test  = CIFAR100('../data-local/cifar100', train=False, download=True, transform=test_transform)
        NUM_TRAIN = len(data_train)
        NUM_CLASSES = 100
        no_train = NUM_TRAIN
    elif dataset == 'fashionmnist':
        data_train = FashionMNIST('../data-local/fashionMNIST', train=True, download=True,
                                    transform=T.Compose([T.ToTensor()]))
        data_unlabeled = MyDataset(dataset, True, T.Compose([T.ToTensor()]))
        data_test  = FashionMNIST('../data-local/fashionMNIST', train=False, download=True,
                                    transform=T.Compose([T.ToTensor()]))
        NUM_CLASSES = 10
        NUM_TRAIN = len(data_train)        
        no_train = NUM_TRAIN
    elif dataset == 'svhn':
        data_train = SVHN('../data-local/svhn', split='train', download=True,
                                    transform=T.Compose([T.ToTensor()]))
        data_unlabeled = MyDataset(dataset, True, T.Compose([T.ToTensor()]))
        data_test  = SVHN('../data-local/svhn', split='test', download=True,
                                    transform=T.Compose([T.ToTensor()]))
        NUM_CLASSES = 10
        NUM_TRAIN = len(data_train)        
        no_train = NUM_TRAIN
    elif dataset == 'tiny_imagenet':
        data_train = ImageFolder('../data-local/tiny-imagenet/by-image/train', train_transform)
        if args.method_type in ['ST-CoNAL-Aug', 'CSSAL']:
            data_unlabeled = MyDataset(dataset, True, aug_transform)
        else:
            data_unlabeled = MyDataset(dataset, True, test_transform)
        data_test = ImageFolder('../data-local/tiny-imagenet/by-image/val', test_transform)
        NUM_CLASSES = 200
        NUM_TRAIN = len(data_train)
        no_train = NUM_TRAIN
    elif dataset == 'caltech256':
        # train: 24660, test: 5119
        data_train = ImageFolder('../data-local/caltech256/by-image/train+val' , train_transform)
        if args.method_type in ['ST-CoNAL-Aug', 'CSSAL']:
            data_unlabeled = MyDataset(dataset, True, aug_transform)
        else:
            data_unlabeled = MyDataset(dataset, True, test_transform)
        data_test = ImageFolder('../data-local/caltech256/by-image/test', test_transform)
        NUM_CLASSES = 256
        NUM_TRAIN = len(data_train)
        no_train = NUM_TRAIN

    return data_train, data_unlabeled, data_test, NUM_CLASSES, no_train