import torch
from torch.utils.data import Subset, DataLoader
from torchvision import datasets
from torchvision import transforms
import numpy as np 
from PIL import Image

def cifar_dataloaders(data_dir, num_classes=10, AugMax=None, **AugMax_args):
    
    assert num_classes in [10, 100]
    CIFAR = datasets.CIFAR10 if num_classes == 10 else datasets.CIFAR100

    if AugMax is not None:
        train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4)])
        test_transform = transforms.Compose([transforms.ToTensor()])

        train_data = CIFAR(data_dir, train=True, transform=train_transform, download=True)
        test_data = CIFAR(data_dir, train=False, transform=test_transform, download=True)
        train_data = AugMax(train_data, test_transform, 
            mixture_width=AugMax_args['mixture_width'], mixture_depth=AugMax_args['mixture_depth'], aug_severity=AugMax_args['aug_severity'], 
        )

    else:
        train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor()])
        test_transform = transforms.Compose(
            [transforms.ToTensor()])

        train_data = CIFAR(data_dir, train=True, transform=train_transform, download=True)
        test_data = CIFAR(data_dir, train=False, transform=test_transform, download=True)
    
    return train_data, test_data

def cifar_random_affine_test_set(data_dir, num_classes=10):
    assert num_classes in [10, 100]
    CIFAR = datasets.CIFAR10 if num_classes == 10 else datasets.CIFAR100

    test_transform = transforms.Compose(
        [transforms.RandomAffine(degrees=30, translate=(3/32,3/32)), transforms.ToTensor()])

    test_data = CIFAR(data_dir, train=False, transform=test_transform, download=True)
    
    return test_data

def cifar_test_augmax_loader(data_dir, num_classes=10, test_batch_size=1000, num_workers=4, AugMax=None, **AugMax_args):
    '''
    AugMax on cifar test sets. This is not used in training or evaluation, just to see effectiveness of AugMax on test sets.
    '''
    assert num_classes in [10, 100]
    CIFAR = datasets.CIFAR10 if num_classes == 10 else datasets.CIFAR100

    test_transform = transforms.Compose([transforms.ToTensor()])

    test_data = CIFAR(data_dir, train=False, transform=None, download=True)

    test_data = AugMax(test_data, test_transform,
        mixture_width=AugMax_args['mixture_width'], mixture_depth=AugMax_args['mixture_depth'], aug_severity=AugMax_args['aug_severity']
    )

    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return test_loader

import os
from torchvision.datasets.utils import download_and_extract_archive, extract_archive

def cifar_c_testloader(corruption, data_dir, num_classes=10, 
    test_batch_size=100, num_workers=2):
    '''
    Returns:
        test_c_loader: corrupted testing set loader (original cifar10-C)
    CIFAR10-C has 50,000 test images. 
    The first 10,000 images in each .npy are of level 1 severity, and the last 10,000 are of level 5 severity.
    '''

    # # download:
    # url = 'https://zenodo.org/record/2535967/files/CIFAR-10-C.tar'
    # root_dir = data_dir
    # tgz_md5 = '56bf5dcef84df0e2308c6dcbcbbd8499'
    # if not os.path.exists(os.path.join(root_dir, 'CIFAR-10-C.tar')):
    #     download_and_extract_archive(url, root_dir, extract_root=root_dir, md5=tgz_md5)
    # elif not os.path.exists(os.path.join(root_dir, 'CIFAR-10-C')):
    #     extract_archive(os.path.join(root_dir, 'CIFAR-10-C.tar'), to_path=root_dir)

    if num_classes==10:
        CIFAR = datasets.CIFAR10
        base_c_path = os.path.join(data_dir, 'CIFAR-10-C')
    elif num_classes==100:
        CIFAR = datasets.CIFAR100
        base_c_path = os.path.join(data_dir, 'CIFAR-100-C')
    else:
        raise Exception('Wrong num_classes %d' % num_classes)
    
    # test set:
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_set = CIFAR(data_dir, train=False, transform=test_transform, download=False)
    
    # replace clean data with corrupted data:
    test_set.data = np.load(os.path.join(base_c_path, '%s.npy' % corruption))
    test_set.targets = torch.LongTensor(np.load(os.path.join(base_c_path, 'labels.npy')))
    print('loader for %s ready' % corruption)

    test_c_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return test_c_loader

def cifar10_1_testloader(data_dir, version_string='v4', test_batch_size=100, num_workers=2):
    '''
    Returns:
        cifar10_1_loader: data set loader of CIFAR10.1

    CIFAR10.1 has 2,000 test images. 
    '''

    filename = 'cifar10.1'
    if version_string == '':
        version_string = 'v7'
    if version_string in ['v4', 'v6', 'v7']:
        filename += '_' + version_string
    else:
        raise ValueError('Unknown dataset version "{}".'.format(version_string))
    label_filename = filename + '_labels.npy'
    imagedata_filename = filename + '_data.npy'
    label_filepath = os.path.abspath(os.path.join(data_dir, 'CIFAR-10.1/datasets', label_filename))
    imagedata_filepath = os.path.abspath(os.path.join(data_dir, 'CIFAR-10.1/datasets', imagedata_filename))
    print('Loading labels from file {}'.format(label_filepath))
    # assert pathlib.Path(label_filepath).is_file()
    labels = np.load(label_filepath)
    print('Loading image data from file {}'.format(imagedata_filepath))
    # assert pathlib.Path(imagedata_filepath).is_file()
    imagedata = np.load(imagedata_filepath)
    assert len(labels.shape) == 1
    assert len(imagedata.shape) == 4
    assert labels.shape[0] == imagedata.shape[0]
    assert imagedata.shape[1] == 32
    assert imagedata.shape[2] == 32
    assert imagedata.shape[3] == 3
    if version_string == 'v6' or version_string == 'v7':
        assert labels.shape[0] == 2000
    elif version_string == 'v4':
        assert labels.shape[0] == 2021
    
    # test set:
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_set = datasets.CIFAR10(data_dir, train=False, transform=test_transform, download=False)
    
    # replace clean data with corrupted data:
    test_set.data = imagedata
    test_set.targets = torch.LongTensor(labels)

    test_v2_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return test_v2_loader