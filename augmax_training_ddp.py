'''
Training with AugMax data augmentation 
'''
import os, sys, argparse, time, socket
from functools import partial
sys.path.append('./')
import numpy as np 

import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.data import Subset, DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp

from augmax_modules import augmentations
from augmax_modules.augmax import AugMaxDataset, AugMaxModule, AugMixModule

from models.cifar10.resnet_DuBIN import ResNet18_DuBIN
from models.cifar10.wideresnet_DuBIN import WRN40_DuBIN
from models.cifar10.resnext_DuBIN import ResNeXt29_DuBIN

from models.imagenet.resnet_DuBIN import ResNet18_DuBIN as INResNet18_DuBIN
from models.imagenet.resnet_DuBIN import ResNet50_DuBIN as INResNet50_DuBIN

from dataloaders.cifar10 import cifar_dataloaders
from dataloaders.tiny_imagenet import tiny_imagenet_dataloaders, tiny_imagenet_deepaug_dataloaders
from dataloaders.imagenet import imagenet_dataloaders, imagenet_deepaug_dataloaders

from utils.utils import *
from utils.context import ctx_noparamgrad_and_eval
from utils.attacks import AugMaxAttack, FriendlyAugMaxAttack

parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier')
parser.add_argument('--gpu', default='0')
parser.add_argument('--num_workers', '--cpus', default=16, type=int)
# dataset:
parser.add_argument('--dataset', '--ds', default='cifar10', choices=['cifar10', 'cifar100', 'tin', 'IN'], help='which dataset to use')
parser.add_argument('--data_root_path', '--drp', help='ImageNet dataset path. (Only effective when using (Tiny) ImageNet)')
parser.add_argument('--model', '--md', default='WRN40', choices=['ResNet18', 'ResNet50', 'WRN40', 'ResNeXt29'], help='which model to use')
parser.add_argument('--widen_factor', '--widen', default=2, type=int, help='which model to use')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--decay_epochs', '--de', default=[100,150], nargs='+', type=int, help='milestones for multisteps lr decay')
parser.add_argument('--opt', default='sgd', choices=['sgd', 'adam'], help='which optimizer to use')
parser.add_argument('--decay', default='cos', choices=['cos', 'multisteps'], help='which lr decay method to use')
parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=256, help='Batch size.')
parser.add_argument('--test_batch_size', '--tb', type=int, default=1000)
parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
parser.add_argument('--wd', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# AugMix options
parser.add_argument('--mixture_width', default=3, help='Number of augmentation chains to mix per augmented example')
parser.add_argument('--mixture_depth', default=-1, help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
parser.add_argument('--aug_severity', default=3, help='Severity of base augmentation operators')
# augmax parameters:
parser.add_argument('--attacker', default='fat', choices=['pgd', 'fat'], help='If true, targeted attack')
parser.add_argument('--targeted', action='store_true', help='If true, targeted attack')
parser.add_argument('--alpha', type=float, default=0.1, help='attack step size')
parser.add_argument('--tau', type=int, default=1)
parser.add_argument('--steps', type=int, default=5)
parser.add_argument('--Lambda', type=float, default=10)
# others:
parser.add_argument('--deepaug', action='store_true', help='If true, use deep augmented training set. (Only works for TIN.)')
parser.add_argument('--resume', action='store_true', help='If true, resume from early stopped ckpt')
parser.add_argument('--save_root_path', '--srp', default='/ssd1/haotao/')
parser.add_argument('--ddp', action='store_true', help='If true, use distributed data parallel')
parser.add_argument('--ddp_backend', '--ddpbed', default='nccl', choices=['nccl', 'gloo', 'mpi'], help='If true, use distributed data parallel')
parser.add_argument('--num_nodes', default=1, type=int, help='Number of nodes')
parser.add_argument('--node_id', default=0, type=int, help='Node ID')
parser.add_argument('--dist_url', default='tcp://localhost:23456', type=str, help='url used to set up distributed training')
args = parser.parse_args()
# adjust learning rate:
if args.dataset == 'tin':
    args.lr *= args.batch_size / 256. # linearly scaled to batch size
    augmentations.IMAGE_SIZE = 64 # change imange size
elif args.dataset == 'IN':
    # args.cpus = 12
    args.lr *= args.batch_size / 256. # linearly scaled to batch size
    augmentations.IMAGE_SIZE = 224 # change imange size

# set CUDA:
if args.num_nodes == 1: # When using multiple nodes, we assume all gpus on each node are available.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu 

# select model_fn:
if args.dataset == 'IN':
    if args.model == 'ResNet18':
        model_fn = INResNet18_DuBIN
    elif args.model == 'ResNet50':
        model_fn = INResNet50_DuBIN
else:
    if args.model == 'ResNet18':
        model_fn = ResNet18_DuBIN
    elif args.model == 'WRN40':
        model_fn = WRN40_DuBIN
    elif args.model == 'ResNeXt29':
        model_fn = ResNeXt29_DuBIN

# mkdirs:
model_str = model_fn.__name__
if args.opt == 'sgd':
    opt_str = 'e%d-b%d_sgd-lr%s-m%s-wd%s' % (args.epochs, args.batch_size, args.lr, args.momentum, args.wd)
elif args.opt == 'adam':
    opt_str = 'e%d-b%d_adam-lr%s-wd%s' % (args.epochs, args.batch_size, args.lr, args.wd)
if args.decay == 'cos':
    decay_str = 'cos'
elif args.decay == 'multisteps':
    decay_str = 'multisteps-' + '-'.join(map(str, args.decay_epochs)) 
loss_str = 'Lambda%s' % args.Lambda
attack_str = ('%s-%s' % (args.attacker, args.tau) if args.attacker == 'fat' else args.attacker) + '-' + ('targeted' if args.targeted else 'untargeted') + '-%d-%s' % (args.steps, args.alpha)
if args.deepaug:
    dataset_str = '%s_deepaug' % args.dataset
    assert args.dataset in ['tin', 'IN']
else:
    dataset_str = args.dataset
save_folder = os.path.join(args.save_root_path, 'AugMax_results/augmax_training', dataset_str, model_str, '%s_%s_%s_%s' % (attack_str, loss_str, opt_str, decay_str))
create_dir(save_folder)
print('saving to %s' % save_folder)

def setup(rank, ngpus_per_node):
    # initialize the process group
    world_size = ngpus_per_node * args.num_nodes
    dist.init_process_group(args.ddp_backend, init_method=args.dist_url, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(gpu_id, ngpus_per_node):
    
    # get globale rank (thread id):
    rank = args.node_id * ngpus_per_node + gpu_id

    print(f"Running on rank {rank}.")

    if gpu_id == 0:
        print(args)

    # Initializes ddp:
    if args.ddp:
        setup(rank, ngpus_per_node)

    # intialize device:
    device = gpu_id if args.ddp else 'cuda'
    torch.backends.cudnn.benchmark = True # set cudnn.benchmark in each worker, as done in https://github.com/pytorch/examples/blob/b0649dcd638eb553238cdd994127fd40c8d9a93a/imagenet/main.py#L199

    # get batch size:
    train_batch_size = args.batch_size if not args.ddp else int(args.batch_size/ngpus_per_node/args.num_nodes)
    num_workers = args.num_workers if not args.ddp else int((args.num_workers+ngpus_per_node)/ngpus_per_node)

    # data loader:
    if args.dataset in ['cifar10', 'cifar100']:
        num_classes=10 if args.dataset == 'cifar10' else 100
        init_stride = 1
        train_data, val_data = cifar_dataloaders(data_dir=args.data_root_path, num_classes=num_classes,
            AugMax=AugMaxDataset, mixture_width=args.mixture_width, mixture_depth=args.mixture_depth, aug_severity=args.aug_severity
        )
    elif args.dataset == 'tin':
        num_classes, init_stride = 200, 2
        train_data, val_data = tiny_imagenet_dataloaders(data_dir=os.path.join(args.data_root_path, 'tiny-imagenet-200'),
            AugMax=AugMaxDataset, mixture_width=args.mixture_width, mixture_depth=args.mixture_depth, aug_severity=args.aug_severity
        )
        if args.deepaug:
            edsr_data = tiny_imagenet_deepaug_dataloaders(data_dir=os.path.join(args.data_root_path, 'tiny-imagenet-200-DeepAug-EDSR'),
                AugMax=AugMaxDataset, mixture_width=args.mixture_width, mixture_depth=args.mixture_depth, aug_severity=args.aug_severity
            )
            cae_data = tiny_imagenet_deepaug_dataloaders(data_dir=os.path.join(args.data_root_path, 'tiny-imagenet-200-DeepAug-CAE'),
                AugMax=AugMaxDataset, mixture_width=args.mixture_width, mixture_depth=args.mixture_depth, aug_severity=args.aug_severity
            )
            train_data = torch.utils.data.ConcatDataset([train_data, edsr_data, cae_data])
    elif args.dataset == 'IN':
        num_classes, init_stride = 1000, None
        train_data, val_data = imagenet_dataloaders(data_dir=os.path.join(args.data_root_path, 'imagenet'), 
            AugMax=AugMaxDataset, mixture_width=args.mixture_width, mixture_depth=args.mixture_depth, aug_severity=args.aug_severity
        )
        if args.deepaug:
            edsr_data = imagenet_deepaug_dataloaders(data_dir=os.path.join(args.data_root_path, 'imagenet-DeepAug-EDSR'), 
                AugMax=AugMaxDataset, mixture_width=args.mixture_width, mixture_depth=args.mixture_depth, aug_severity=args.aug_severity
            )
            cae_data = imagenet_deepaug_dataloaders(data_dir=os.path.join(args.data_root_path, 'imagenet-DeepAug-CAE'), 
                AugMax=AugMaxDataset, mixture_width=args.mixture_width, mixture_depth=args.mixture_depth, aug_severity=args.aug_severity
            )
            train_data = torch.utils.data.ConcatDataset([train_data, edsr_data, cae_data])
    if args.ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=(train_sampler is None), num_workers=num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(val_data, batch_size=args.test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # model:
    if args.dataset == 'IN':
        if args.model == 'WRN40':
            model = model_fn(widen_factor=args.widen_factor).to(device)
        else:
            model = model_fn().to(device)
    else:
        if args.model == 'WRN40':
            model = model_fn(num_classes=num_classes, init_stride=init_stride, widen_factor=args.widen_factor).to(device)
        else:
            model = model_fn(num_classes=num_classes, init_stride=init_stride).to(device)
    if args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id], broadcast_buffers=False, find_unused_parameters=False)
    else:
        model = torch.nn.DataParallel(model)

    # optimizer:
    if args.opt == 'sgd':
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    elif args.opt == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.decay == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    elif args.decay == 'multisteps':
        scheduler = lr_scheduler.MultiStepLR(optimizer, args.decay_epochs, gamma=0.1)

    # load ckpt:
    if args.resume:
        last_epoch, best_SA, training_loss, val_SA \
            = load_ckpt(model, optimizer, scheduler, os.path.join(save_folder, 'latest.pth'))
        start_epoch = last_epoch + 1
    else:
        start_epoch = 0
        best_SA = 0
        # training curve lists:
        training_loss, val_SA = [], []

    # attacker
    if args.attacker == 'pgd':
        attacker = AugMaxAttack(steps=args.steps, alpha=args.alpha, targeted=args.targeted)
    elif args.attacker == 'fat':
        attacker = FriendlyAugMaxAttack(steps=args.steps, alpha=args.alpha, tau=args.tau, targeted=args.targeted)
    augmix_model = AugMixModule(args.mixture_width, device=device)
    augmax_model = AugMaxModule(device=device)

    # train:
    for epoch in range(start_epoch, args.epochs):
        # reset sampler when using ddp:
        if args.ddp:
            train_sampler.set_epoch(epoch)
        fp = open(os.path.join(save_folder, 'train_log.txt'), 'a+')
        start_time = time.time()

        ## training:
        model.train()
        requires_grad_(model, True)
        accs, accs_augmax, losses = AverageMeter(), AverageMeter(), AverageMeter()
        for i, (images_tuples, labels) in enumerate(train_loader):

            # get batch:
            images_tuple = images_tuples[0]
            images_tuple = [images.to(device) for images in images_tuple]
            images_tuple_2 = images_tuples[1]
            images_tuple_2 = [images.to(device) for images in images_tuple_2]
            labels = labels.to(device)

            # switch to BN-A:
            if 'DuBN' in model_fn.__name__ or  'DuBIN' in model_fn.__name__:
                model.apply(lambda m: setattr(m, 'route', 'A')) # 

            # generate and forward aug images:
            with ctx_noparamgrad_and_eval(model):
                # generate augmax1:
                if args.targeted:
                    targets = torch.fmod(labels + torch.randint(low=1, high=num_classes, size=labels.size()).to(device), num_classes)
                    imgs_augmax_1, _, _ = attacker.attack(augmax_model, model, images_tuple, labels=labels, targets=targets, device=device)
                else:
                    imgs_augmax_1, _, _  = attacker.attack(augmax_model, model, images_tuple, labels=labels, device=device)
            # augmax image forward:
            logits_augmax_1 = model(imgs_augmax_1.detach())


            # switch to BN-M:
            if 'DuBN' in model_fn.__name__ or  'DuBIN' in model_fn.__name__:
                model.apply(lambda m: setattr(m, 'route', 'M')) # use main BN

            # generate augmix images:
            imgs_augmix_1 = augmix_model(images_tuple_2)
            logits_augmix_1 = model(imgs_augmix_1.detach())

            # logits for clean imgs:
            logits = model(images_tuple[0])

            # loss:
            loss_clean = F.cross_entropy(logits, labels)
            p_clean, p_aug1, p_aug2 = F.softmax(logits, dim=1), F.softmax(logits_augmax_1, dim=1), F.softmax(logits_augmix_1, dim=1)
            p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
            loss_cst = args.Lambda * (
                        F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                        F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                        F.kl_div(p_mixture, p_aug2, reduction='batchmean')
                        ) / 3.
            loss = loss_clean + loss_cst
            # update:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # metrics:
            accs.append((logits.argmax(1) == labels).float().mean().item())
            accs_augmax.append((logits_augmax_1.argmax(1) == labels).float().mean().item())
            losses.append(loss.item())

            if i % 50 == 0:
                train_str = 'Epoch %d-%d | Train | Loss: %.4f (%.4f, %.4f), SA: %.4f, RA: %.4f' % (epoch, i, losses.avg, loss_clean, loss_cst, accs.avg, accs_augmax.avg)
                if gpu_id == 0:
                    print(train_str)
        # lr schedualr update at the end of each epoch:
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()


        ## validation:
        if rank == 0:
            model.eval()
            requires_grad_(model, False)
            print(model.training)

            # eval_this_epoch = (epoch % 10 == 0) or (epoch>=int(0.7*args.epochs)) # boolean
            eval_this_epoch = True
            
            if eval_this_epoch:
                val_SAs = AverageMeter()
                if 'DuBN' in model_fn.__name__ or  'DuBIN' in model_fn.__name__: 
                    model.apply(lambda m: setattr(m, 'route', 'M')) # use main BN
                for i, (imgs, labels) in enumerate(val_loader):
                    imgs, labels = imgs.to(device), labels.to(device)
                    # logits for clean imgs:
                    logits = model(imgs)
                    val_SAs.append((logits.argmax(1) == labels).float().mean().item())

                val_str = 'Epoch %d | Validation | Time: %.4f | lr: %s | SA: %.4f' % (
                    epoch, (time.time()-start_time), current_lr, val_SAs.avg)
                print(val_str)
                fp.write(val_str + '\n')

            # save loss curve:
            training_loss.append(losses.avg)
            plt.plot(training_loss)
            plt.grid(True)
            plt.savefig(os.path.join(save_folder, 'training_loss.png'))
            plt.close()

            val_SA.append(val_SAs.avg) 
            plt.plot(val_SA, 'r')
            plt.grid(True)
            plt.savefig(os.path.join(save_folder, 'val_SA.png'))
            plt.close()

            # save pth:
            if eval_this_epoch:
                if val_SAs.avg >= best_SA:
                    best_SA = val_SAs.avg
                    torch.save(model.state_dict(), os.path.join(save_folder, 'best_SA.pth'))
            save_ckpt(epoch, model, optimizer, scheduler, best_SA, training_loss, val_SA, 
                os.path.join(save_folder, 'latest.pth'))

    # Clean up ddp:
    if args.ddp:
        cleanup()

if __name__ == '__main__':
    if args.ddp:
        ngpus_per_node = torch.cuda.device_count()
        torch.multiprocessing.spawn(train, args=(ngpus_per_node,), nprocs=ngpus_per_node, join=True)
    else:
        train(0, 0)