import os, sys, argparse, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
import numpy as np 
from skimage.io import imsave
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F

from augmax_modules import augmentations, corruptions_cifar, corruptions_IN

# data:
def aug(image, preprocess, mixture_depth, aug_severity):
	"""Perform augmentation operations on PIL.Images.

	Args:
		image: PIL.Image input image
		preprocess: Preprocessing function which should return a torch tensor.

	Returns:
		image_aug: Augmented image.
	"""
	aug_list = augmentations.augmentations

	image_aug = image.copy()
	depth = mixture_depth if mixture_depth > 0 else np.random.randint(1, 4)
	for _ in range(depth):
		op = np.random.choice(aug_list)
		image_aug = op(image_aug, aug_severity)
	
	image_aug = preprocess(image_aug)

	return image_aug

class AugMaxDataset(torch.utils.data.Dataset):
	"""Dataset wrapper to perform AugMix augmentation."""

	def __init__(self, dataset, preprocess, mixture_width=3, mixture_depth=-1, aug_severity=3, aug_fn=aug):
		self.dataset = dataset
		self.preprocess = preprocess
		self.mixture_width = mixture_width
		self.mixture_depth = mixture_depth
		self.aug_severity = aug_severity
		self.img_tuple_num = 2
		self.aug_fn = aug_fn

	def __getitem__(self, i):
		x, y = self.dataset[i]

		img_tuples = []
		for _ in range(self.img_tuple_num):
			img_tuple = [self.preprocess(x)]
			for _ in range(self.mixture_width):
				img_tuple.append(
					self.aug_fn(x, self.preprocess, self.mixture_depth, self.aug_severity)
				)
			# (original_imgs, aug_branch1_imgs, aug_branch2_imgs, aug_branch3_imgs)
			img_tuples.append(img_tuple)
		return img_tuples, y

	def __len__(self):
		return len(self.dataset)

class AugMaxModule(nn.Module):
	def __init__(self, device='cuda'):
		super(AugMaxModule, self).__init__()
		self.device = device

	def forward(self, xs, m, q):
		'''
		Args:
			xs: tuple of Tensors. len(x)=3. xs = (x_ori, x_aug1, x_aug2, x_aug3). x_ori.size()=(N,W,H,C)
			m: Tensor. m.size=(N)
			q: Tensor. q.size()=(N,3). w = softmax(q)
		'''
		
		x_ori = xs[0]
		w = torch.nn.functional.softmax(q, dim=1) # w.size()=(N,3)

		N = x_ori.size()[0]

		x_mix = torch.zeros_like(x_ori).to(self.device)
		for i, x_aug in enumerate(xs[1:]):
			wi = w[:,i].view((N,1,1,1)).expand_as(x_aug)
			x_mix += wi * x_aug 

		m = m.view((N,1,1,1)).expand_as(x_ori)
		x_mix = (1-m) * x_ori + m * x_mix

		return x_mix 

class AugMixModule(nn.Module):
	def __init__(self, mixture_width, device='cuda'):
		super(AugMixModule, self).__init__()

		self.mixture_width = mixture_width
		self.w_dist = torch.distributions.dirichlet.Dirichlet(torch.tensor([1.] * mixture_width)) # Dirichlet parameter must be float in pytorch
		self.m_dist = torch.distributions.beta.Beta(1, 1)

		self.device = device

	def forward(self, xs):
		'''
		Args:
			xs: tuple of Tensors. len(x)=3. xs = (x_ori, x_aug1, x_aug2, x_aug3). x_ori.size()=(N,W,H,C)
		'''
		# mixture_width = len(xs) - 1

		x_ori = xs[0]
		N = x_ori.size()[0]

		w = self.w_dist.sample([N]).to(self.device)
		m = self.m_dist.sample([N]).to(self.device)

		x_mix = torch.zeros_like(x_ori).to(self.device)
		for i, x_aug in enumerate(xs[1:]):
			wi = w[:,i].view((N,1,1,1)).expand_as(x_aug)
			x_mix += wi * x_aug 

		m = m.view((N,1,1,1)).expand_as(x_ori)
		x_mix = (1-m) * x_ori + m * x_mix

		return x_mix 
