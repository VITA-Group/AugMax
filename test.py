import os, sys, argparse, time, random
from functools import partial
sys.path.append('./')
import numpy as np 
from advertorch.attacks import PGDAttack

import torch, torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader

from augmax_modules import augmentations
from models.cifar10.resnet_DuBIN import ResNet18_DuBIN
from models.cifar10.wideresnet_DuBIN import WRN40_DuBIN
from models.cifar10.resnext_DuBIN import ResNeXt29_DuBIN

from models.imagenet.resnet_DuBIN import ResNet18_DuBIN as INResNet18_DuBIN

from dataloaders.cifar10 import cifar_dataloaders, cifar_c_testloader, cifar10_1_testloader, cifar_random_affine_test_set
from dataloaders.tiny_imagenet import tiny_imagenet_dataloaders, tiny_imagenet_c_testloader
from dataloaders.imagenet import imagenet_dataloaders, imagenet_c_testloader, imagenet_v2_testloader

from utils.utils import *

parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier')
parser.add_argument('--gpu', default='0')
parser.add_argument('--cpus', type=int, default=4)
# dataset:
parser.add_argument('--dataset', '--ds', default='cifar10', choices=['cifar10', 'cifar100', 'tin', 'IN'], help='which dataset to use')
parser.add_argument('--data_root_path', '--drp', default='/ssd1/haotao/datasets/', help='(Tiny-)ImageNet dataset path.')
parser.add_argument('--v2_format', choices=['a', 'b', 'c'], default='b', help='Which format of ImageNet-V2 to use.')
parser.add_argument('--model', '--md', default='WRN40', choices=['ResNet18_DuBIN', 'WRN40_DuBIN', 'ResNeXt29_DuBIN'], help='which model to use')
parser.add_argument('--widen_factor', '--widen', default=2, type=int, help='which model to use')
# 
parser.add_argument('--test_batch_size', '--tb', type=int, default=1000)
parser.add_argument('--ckpt_path', default='')
parser.add_argument('--mode', default='clean', choices=['clean', 'c', 'v2', 'sta', 'all'], help='')
parser.add_argument('--k', default=10, type=int, help='worst-of-k spatial attack')
parser.add_argument('--save_root_path', '--srp', default='/ssd1/haotao')
args = parser.parse_args()
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

# model:
if args.dataset == 'IN':
    model_fn = INResNet18_DuBIN

else:
    if args.model == 'ResNet18_DuBIN':
        model_fn = ResNet18_DuBIN

    if args.model == 'WRN40_DuBIN':
        model_fn = partial(WRN40_DuBIN, widen_factor=args.widen_factor)

    if args.model == 'ResNeXt29_DuBIN':
        model_fn = ResNeXt29_DuBIN

if args.dataset in ['cifar10', 'cifar100']:
    num_classes=10 if args.dataset == 'cifar10' else 100
    init_stride = 1
elif args.dataset == 'tin':
    num_classes, init_stride = 200, 2
elif args.dataset == 'IN':
    num_classes, init_stride = 1000, None

if args.dataset == 'IN':
    model = model_fn().cuda()
else:
    model = model_fn(num_classes=num_classes, init_stride=init_stride).cuda()
model = torch.nn.DataParallel(model)

# load model:
ckpt = torch.load(os.path.join(args.save_root_path, 'AugMax_results', args.ckpt_path, 'best_SA.pth'))
model.load_state_dict(ckpt)        

# log file:
fp = open(os.path.join(args.save_root_path, 'AugMax_results', args.ckpt_path, 'test_results.txt'), 'a+')

## Test on CIFAR:
def val_cifar():
    _, val_data = cifar_dataloaders(data_dir=args.data_root_path, num_classes=num_classes, train_batch_size=256, test_batch_size=args.test_batch_size, num_workers=args.cpus, AugMax=None)
    test_loader = DataLoader(val_data, batch_size=args.test_batch_size, shuffle=False, num_workers=args.cpus, pin_memory=True)

    model.eval()
    ts = time.time()
    test_loss_meter, test_acc_meter = AverageMeter(), AverageMeter()
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.cuda(), targets.cuda()
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
            pred = logits.data.max(1)[1]
            acc = pred.eq(targets.data).float().mean()
            # append loss:
            test_loss_meter.append(loss.item())
            test_acc_meter.append(acc.item())
    print('clean test time: %.2fs' % (time.time()-ts))
    # test loss and acc of this epoch:
    test_loss = test_loss_meter.avg
    test_acc = test_acc_meter.avg

    # print:
    clean_str = 'clean: %.4f' % test_acc
    print(clean_str)
    fp.write(clean_str + '\n')
    fp.flush()

def val_cifar_worst_of_k_affine(K):
    '''
    Test model robustness against spatial transform attacks using worst-of-k method.
    '''
    model.eval()
    ts = time.time()
    test_loss_meter, test_acc_meter = AverageMeter(), AverageMeter()
    with torch.no_grad():        
        K_loss = torch.zeros((K, args.test_batch_size)).cuda()
        K_logits = torch.zeros((K, args.test_batch_size, num_classes)).cuda()
        for k in range(K):
            random.seed(k+1)
            val_data = cifar_random_affine_test_set(data_dir=args.data_root_path, num_classes=num_classes)
            test_loader = DataLoader(val_data, batch_size=args.test_batch_size, shuffle=False, num_workers=args.cpus, pin_memory=True)
            images, targets = next(iter(test_loader))
            images, targets = images.cuda(), targets.cuda()
            logits = model(images)
            loss = F.cross_entropy(logits, targets, reduction='none')
            # stack all losses:
            K_loss[k,:] = loss # shape=(K,N)
            K_logits[k,...] = logits
        # print('K_loss:', K_loss[:,0:3], K_loss.shape)
        adv_idx = torch.max(K_loss, dim=0).indices
        logits_adv = torch.zeros_like(logits).to(logits.device)
        for n in range(images.shape[0]):
            logits_adv[n] = K_logits[adv_idx[n],n,:]
        print('logits_adv:', logits_adv.shape)
        pred = logits_adv.data.max(1)[1]
        print('pred:', pred.shape)
        acc = pred.eq(targets.data).float().mean()
        # append loss:
        test_acc_meter.append(acc.item())
    print('worst of %d test time: %.2fs' % (K, time.time()-ts))
    # test loss and acc of this epoch:
    test_acc = test_acc_meter.avg

    # print:
    clean_str = 'worst of %d: %.4f' % (K, test_acc)
    print(clean_str)
    fp.write(clean_str + '\n')
    fp.flush()

def val_cifar_c():
    test_seen_c_loader_list = []
    for corruption in CORRUPTIONS:
        test_c_loader = cifar_c_testloader(corruption=corruption, data_dir=args.data_root_path, num_classes=num_classes, 
            test_batch_size=args.test_batch_size, num_workers=args.cpus)
        test_seen_c_loader_list.append(test_c_loader)

    # val corruption:
    print('evaluating corruptions...')
    test_c_losses, test_c_accs = [], []
    for corruption, test_c_loader in zip(CORRUPTIONS, test_seen_c_loader_list):
        test_c_batch_num = len(test_c_loader)
        print(test_c_batch_num) # each corruption has 10k * 5 images, each magnitude has 10k images
        ts = time.time()
        test_c_loss_meter, test_c_acc_meter = AverageMeter(), AverageMeter()
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(test_c_loader):
                images, targets = images.cuda(), targets.cuda()
                logits = model(images)
                loss = F.cross_entropy(logits, targets)
                pred = logits.data.max(1)[1]
                acc = pred.eq(targets.data).float().mean()
                # append loss:
                test_c_loss_meter.append(loss.item())
                test_c_acc_meter.append(acc.item())

        print('%s test time: %.2fs' % (corruption, time.time()-ts))
        # test loss and acc of each type of corruptions:
        test_c_losses.append(test_c_loss_meter.avg)
        test_c_accs.append(test_c_acc_meter.avg)

        # print
        corruption_str = '%s: %.4f' % (corruption, test_c_accs[-1])
        print(corruption_str)
        fp.write(corruption_str + '\n')
        fp.flush()
    # mean over 16 types of attacks:
    test_c_loss = np.mean(test_c_losses)
    test_c_acc = np.mean(test_c_accs)

    # print
    avg_str = 'corruption acc: (mean) %.4f' % (test_c_acc)
    print(avg_str)
    fp.write(avg_str + '\n')
    fp.flush()

def val_cifar10_1():
    '''
    Evaluate on cifar10.1
    '''
    test_v2_loader = cifar10_1_testloader(data_dir=os.path.join(args.data_root_path))

    model.eval()
    ts = time.time()
    test_loss_meter, test_acc_meter = AverageMeter(), AverageMeter()
    with torch.no_grad():
        for images, targets in test_v2_loader:
            images, targets = images.cuda(), targets.cuda()
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
            pred = logits.data.max(1)[1]
            acc = pred.eq(targets.data).float().mean()
            # append loss:
            test_loss_meter.append(loss.item())
            test_acc_meter.append(acc.item())
    print('cifar10.1 test time: %.2fs' % (time.time()-ts))
    # test loss and acc of this epoch:
    test_loss = test_loss_meter.avg
    test_acc = test_acc_meter.avg

    # print:
    clean_str = 'cifar10.1 test acc: %.4f' % test_acc
    print(clean_str)
    fp.write(clean_str + '\n')
    fp.flush()

## Test on Tiny-ImageNet:
ResNet18_c_CE_list = [
    0.8037, 0.7597, 0.7758, 0.8426, 0.8274, 
    0.7907, 0.8212, 0.7497, 0.7381, 0.7433, 
    0.6800, 0.8939, 0.7308, 0.6121, 0.6452
]

def find_mCE(target_model_c_CE, anchor_model_c_CE):
    '''
    Args:
        target_model_c_CE: np.ndarray. shape=(15). CE of each corruption type of the target model.
        anchor_model_c_CE: np.ndarray. shape=(15). CE of each corruption type of the anchor model (normally trained ResNet18 as default).
    '''
    assert len(target_model_c_CE) == 15 # a total of 15 types of corruptions
    mCE = 0
    for target_model_CE, anchor_model_CE in zip(target_model_c_CE, anchor_model_c_CE):
        mCE += target_model_CE/anchor_model_CE
    mCE /= len(target_model_c_CE)
    return mCE

def val_tin():
    _, val_data = tiny_imagenet_dataloaders(data_dir=os.path.join(args.data_root_path, 'tiny-imagenet-200'), AugMax=None)
    val_loader = DataLoader(val_data, batch_size=args.test_batch_size, shuffle=False, num_workers=args.cpus, pin_memory=True)

    model.eval()
    ts = time.time()
    test_loss_meter, test_acc_meter = AverageMeter(), AverageMeter()
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.cuda(), targets.cuda()
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
            pred = logits.data.max(1)[1]
            acc = pred.eq(targets.data).float().mean()
            # append loss:
            test_loss_meter.append(loss.item())
            test_acc_meter.append(acc.item())
    print('clean test time: %.2fs' % (time.time()-ts))
    # test loss and acc of this epoch:
    test_loss = test_loss_meter.avg
    test_acc = test_acc_meter.avg

    # print:
    clean_str = 'clean acc: %.4f' % test_acc
    print(clean_str)
    fp.write(clean_str + '\n')
    fp.flush()

def val_tin_c():
    test_seen_c_loader_list = []
    for corruption in CORRUPTIONS:
        test_seen_c_loader_list_c = []
        for severity in range(1,6):
            test_c_loader_c_s = tiny_imagenet_c_testloader(data_dir=os.path.join(args.data_root_path, 'TinyImageNet-C/Tiny-ImageNet-C'),
                corruption=corruption, severity=severity, 
                test_batch_size=args.test_batch_size, num_workers=args.cpus)
            test_seen_c_loader_list_c.append(test_c_loader_c_s)
        test_seen_c_loader_list.append(test_seen_c_loader_list_c)

    model.eval()
    # val corruption:
    print('evaluating corruptions...')
    test_CE_c_list = []
    for corruption, test_seen_c_loader_list_c in zip(CORRUPTIONS, test_seen_c_loader_list):
        test_c_CE_c_s_list = []
        ts = time.time()
        for severity in range(1,6):
            test_c_loader_c_s = test_seen_c_loader_list_c[severity-1]
            test_c_batch_num = len(test_c_loader_c_s)
            # print(test_c_batch_num) # each corruption has 10k * 5 images, each magnitude has 10k images
            test_c_loss_meter, test_c_CE_meter = AverageMeter(), AverageMeter()
            with torch.no_grad():
                for batch_idx, (images, targets) in enumerate(test_c_loader_c_s):
                    images, targets = images.cuda(), targets.cuda()
                    logits = model(images)
                    loss = F.cross_entropy(logits, targets)
                    pred = logits.data.max(1)[1]
                    CE = (~pred.eq(targets.data)).float().mean()
                    # append loss:
                    test_c_loss_meter.append(loss.item())
                    test_c_CE_meter.append(CE.item())
            
            # test loss and acc of each type of corruptions:
            test_c_CE_c_s = test_c_CE_meter.avg
            test_c_CE_c_s_list.append(test_c_CE_c_s)
        test_CE_c = np.mean(test_c_CE_c_s_list)
        test_CE_c_list.append(test_CE_c)

        # print
        print('%s test time: %.2fs' % (corruption, time.time()-ts))
        corruption_str = '%s CE: %.4f' % (corruption, test_CE_c)
        print(corruption_str)
        fp.write(corruption_str + '\n')
        fp.flush()
    # mean over 16 types of corruptions:
    test_c_acc = 1-np.mean(test_CE_c_list)
    # weighted mean over 16 types of corruptions:
    test_mCE = find_mCE(test_CE_c_list, anchor_model_c_CE=ResNet18_c_CE_list)

    # print
    avg_str = 'corruption acc: %.4f' % (test_c_acc)
    print(avg_str)
    fp.write(avg_str + '\n')
    mCE_str = 'mCE: %.4f' % test_mCE
    print(mCE_str)
    fp.write(mCE_str + '\n')
    fp.flush()

## Test on ImageNet:
def val_IN():
    _, val_data = imagenet_dataloaders(data_dir=os.path.join(args.data_root_path, 'imagenet'), AugMax=None)
    val_loader = DataLoader(val_data, batch_size=args.test_batch_size, shuffle=False, num_workers=args.cpus, pin_memory=True)

    model.eval()
    ts = time.time()
    test_loss_meter, test_acc_meter = AverageMeter(), AverageMeter()
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.cuda(), targets.cuda()
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
            pred = logits.data.max(1)[1]
            acc = pred.eq(targets.data).float().mean()
            # append loss:
            test_loss_meter.append(loss.item())
            test_acc_meter.append(acc.item())
    print('clean test time: %.2fs' % (time.time()-ts))
    # test loss and acc of this epoch:
    test_loss = test_loss_meter.avg
    test_acc = test_acc_meter.avg

    # print:
    clean_str = 'clean acc: %.4f' % test_acc
    print(clean_str)
    fp.write(clean_str + '\n')
    fp.flush()


AlexNet_ERR = [
    0.886428, 0.894468, 0.922640, 0.819880, 0.826268, 0.785948, 0.798360,
    0.866816, 0.826572, 0.819324, 0.564592, 0.853204, 0.646056, 0.717840,
    0.606500
]
def val_IN_c():
    test_seen_c_loader_list = []
    for corruption in CORRUPTIONS:
        test_seen_c_loader_list_c = []
        for severity in range(1,6):
            test_c_loader_c_s = imagenet_c_testloader(corruption=corruption, severity=severity, 
                data_dir=os.path.join(args.data_root_path, 'ImageNet-C'),
                test_batch_size=args.test_batch_size, num_workers=args.cpus)
            test_seen_c_loader_list_c.append(test_c_loader_c_s)
        test_seen_c_loader_list.append(test_seen_c_loader_list_c)

    model.eval()
    # val corruption:
    print('evaluating corruptions...')
    test_CE_c_list = []
    for corruption, test_seen_c_loader_list_c in zip(CORRUPTIONS, test_seen_c_loader_list):
        test_c_CE_c_s_list = []
        ts = time.time()
        for severity in range(1,6):
            test_c_loader_c_s = test_seen_c_loader_list_c[severity-1]
            test_c_batch_num = len(test_c_loader_c_s)
            # print(test_c_batch_num) # each corruption has 10k * 5 images, each magnitude has 10k images
            test_c_loss_meter, test_c_CE_meter = AverageMeter(), AverageMeter()
            with torch.no_grad():
                for batch_idx, (images, targets) in enumerate(test_c_loader_c_s):
                    images, targets = images.cuda(), targets.cuda()
                    logits = model(images)
                    loss = F.cross_entropy(logits, targets)
                    pred = logits.data.max(1)[1]
                    CE = (~pred.eq(targets.data)).float().mean()
                    # append loss:
                    test_c_loss_meter.append(loss.item())
                    test_c_CE_meter.append(CE.item())
            
            # test loss and acc of each type of corruptions:
            test_c_CE_c_s = test_c_CE_meter.avg
            test_c_CE_c_s_list.append(test_c_CE_c_s)
        test_CE_c = np.mean(test_c_CE_c_s_list)
        test_CE_c_list.append(test_CE_c)

        # print
        print('%s test time: %.2fs' % (corruption, time.time()-ts))
        corruption_str = '%s CE: %.4f' % (corruption, test_CE_c)
        print(corruption_str)
        fp.write(corruption_str + '\n')
        fp.flush()
    # mean over 16 types of corruptions:
    test_c_acc = 1-np.mean(test_CE_c_list)
    # weighted mean over 16 types of corruptions:
    test_mCE = find_mCE(test_CE_c_list, anchor_model_c_CE=AlexNet_ERR)

    # print
    avg_str = 'corruption acc: %.4f' % (test_c_acc)
    print(avg_str)
    fp.write(avg_str + '\n')
    mCE_str = 'mCE: %.4f' % test_mCE
    print(mCE_str)
    fp.write(mCE_str + '\n')
    fp.flush()

if __name__ == '__main__':

    model.apply(lambda m: setattr(m, 'route', 'M')) 

    if args.dataset in ['cifar10', 'cifar100']: 
        if args.mode in ['clean', 'all']:
            val_cifar()
        if args.mode in ['c', 'all']:
            val_cifar_c()
        if args.mode in ['v2']:
            val_cifar10_1()
        if args.mode in ['sta']:
            val_cifar_worst_of_k_affine(args.k)

    elif args.dataset == 'tin':
        if args.mode in ['clean', 'all']:
            val_tin()
        if args.mode in ['c', 'all']:
            val_tin_c()

    elif args.dataset == 'IN':
        if args.mode in ['clean', 'all']:
            val_IN()
        if args.mode in ['c', 'all']:
            val_IN_c()
                         