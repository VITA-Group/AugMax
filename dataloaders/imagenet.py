'''
https://github.com/snu-mllab/PuzzleMix/blob/master/load_data.py
'''

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Subset, DataLoader
import numpy as np 
import os, shutil

def imagenet_dataloaders(data_dir, transform_train=True, AugMax=None, **AugMax_args):
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_root = os.path.join(data_dir, 'train')  # this is path to training images folder
    validation_root = os.path.join(data_dir, 'val')  # this is path to validation images folder
    print('Training images loading from %s' % train_root)
    print('Validation images loading from %s' % validation_root)
    
    preprocess = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    if AugMax is not None: 
        train_transform = transforms.Compose(
            [transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip()]) if transform_train else None
        test_transform = transforms.Compose(
            [transforms.Resize(256),
            transforms.CenterCrop(224),
            preprocess])

        train_data = datasets.ImageFolder(train_root, transform=train_transform)
        test_data = datasets.ImageFolder(validation_root, transform=test_transform)
        train_data = AugMax(train_data, preprocess, 
            mixture_width=AugMax_args['mixture_width'], mixture_depth=AugMax_args['mixture_depth'], aug_severity=AugMax_args['aug_severity'], 
        )

    else:
        train_transform = transforms.Compose(
            [transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            preprocess]) if transform_train else transforms.Compose(
            [transforms.Resize(256),
            transforms.CenterCrop(224),
            preprocess])
        test_transform = transforms.Compose(
            [transforms.Resize(256),
            transforms.CenterCrop(224),
            preprocess])

        train_data = datasets.ImageFolder(train_root, transform=train_transform)
        test_data = datasets.ImageFolder(validation_root, transform=test_transform)
    
    return train_data, test_data

def imagenet_deepaug_dataloaders(data_dir, transform_train=True, AugMax=None, **AugMax_args):
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_root = os.path.join(data_dir, 'train')  # this is path to training images folder
    print('Training images loading from %s' % train_root)
    
    preprocess = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    if AugMax is not None: 
        train_transform = transforms.Compose(
            [transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip()]) if transform_train else None

        train_data = datasets.ImageFolder(train_root, transform=train_transform)
        train_data = AugMax(train_data, preprocess, 
            mixture_width=AugMax_args['mixture_width'], mixture_depth=AugMax_args['mixture_depth'], aug_severity=AugMax_args['aug_severity'], 
        )

    else:
        train_transform = transforms.Compose(
            [transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            preprocess]) if transform_train else transforms.Compose(
            [transforms.Resize(256),
            transforms.CenterCrop(224),
            preprocess])

        train_data = datasets.ImageFolder(train_root, transform=train_transform)
    
    return train_data

def imagenet_c_testloader(corruption, severity, data_dir='/ssd1/haotao/ImageNet-C', 
    test_batch_size=1000, num_workers=4):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    preprocess = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    test_transform = transforms.Compose(
            [transforms.Resize(256),
            transforms.CenterCrop(224),
            preprocess])
    test_root = os.path.join(data_dir, corruption, str(severity))
    test_c_data = datasets.ImageFolder(test_root,transform=test_transform)
    test_c_loader = DataLoader(test_c_data, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return test_c_loader

def imagenet_v2_testloader(data_root_dir='/ssd2/haotao/', format_idx='b', test_batch_size=1000, num_workers=4):
    '''
    Args:
        set: string. choices=['a', 'b', 'c']. a: Threshold0.7, b: MatchedFrequency, c: TopImages
        num_classes: int. choices=[200, 1000]. 200: Load Tiny ImageNet-V2; 1000: Load ImageNet-V2.
    '''
    if format_idx == 'a':
        format_name = 'threshold0.7'
    elif format_idx == 'b':
        format_name = 'matched-frequency'


    data_dir = os.path.join(data_root_dir, 'imagenetv2-%s-format-val-wnid' % format_name)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    test_transform = transforms.Compose(
        [transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    print('Loading data from %s' % data_dir)

    test_v2_data = datasets.ImageFolder(data_dir, transform=test_transform)
    test_v2_loader = DataLoader(test_v2_data, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return test_v2_loader    