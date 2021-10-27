import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

def linf_clamp(x, _min, _max):
	'''
	Inplace linf clamping on Tensor x.
	Args:
		x: Tensor. shape=(N,C,W,H)
		_min: Tensor with same shape as x.
		_max: Tensor with same shape as x.
	'''
	idx = x.data < _min
	x.data[idx] = _min[idx]
	idx = x.data > _max
	x.data[idx] = _max[idx]

	return x

class AugMaxAttack():
	def __init__(self, steps=10, alpha=0.1, loss_fn=None, targeted=False):
		'''
		Args:
			eps: float. noise bound.
			steps: int. PGD attack step number.
			alpha: float. step size for PGD attack.
			loss_fn: loss function which is maximized to generate adversarial images.
			targeted: bool. If Ture, do targeted attack.
		'''
		self.steps = steps
		self.alpha = alpha 
		self.targeted = targeted
		self.loss_fn = loss_fn if loss_fn else nn.CrossEntropyLoss(reduction="sum")    


	def attack(self, augmax_model, model, xs, labels=None, targets=None, noisy=False, grad_noise_level=0.1, random_level=False, device='cuda'):
		'''
		Args:
			xs: Tensor. Original images. size=(N,C,W,H)
			model: nn.Module. The model to be attacked.
			labels: Tensor. ground truth labels for x. size=(N,). Useful only under untargeted attack.
			targets: Tensor. target attack class for x. size=(N,). Useful only under targeted attack.
			noisy: bool. Whether 
			random_level: bool. Whether 
		Return:
			x_adv: Tensor. Adversarial images. size=(N,C,W,H)
		'''
		# 
		model.eval()

		# 
		mixture_width = len(xs) - 1
		N = xs[0].size()[0]

		# initialize m_adv:
		m_adv = torch.rand(N).to(device) # random initialize in [0,1)
		m_adv = torch.clamp(m_adv, 0, 1) # clamp to range [0,1)
		m_adv.requires_grad=True

		# initialize ws_adv:
		q_adv = torch.rand((N,mixture_width), requires_grad=True).to(device) # random initialize

		# initialize x_adv
		x_adv = augmax_model(xs, m_adv , q_adv)

		# attack step size
		if random_level:
			alpha = self.alpha * np.random.choice([1,0.5,0.2,0.1])
		else:
			alpha = self.alpha

		for t in range(self.steps):
			logits_adv = model(x_adv)
			if self.targeted:
				loss_adv = - self.loss_fn(logits_adv, targets)
			else: # untargeted attack
				loss_adv = self.loss_fn(logits_adv, labels)
			# grad:
			grad_m_adv, grad_q_adv = torch.autograd.grad(loss_adv, [m_adv, q_adv], only_inputs=True)
			# update m:
			m_adv.data.add_(alpha * torch.sign(grad_m_adv.data)) # gradient assend by Sign-SGD
			if noisy:
				m_adv.data.add_(grad_noise_level * alpha * torch.rand(grad_m_adv.data.size()).to(device))
			m_adv = torch.clamp(m_adv, 0, 1) # clamp to RGB range [0,1]
			# update w1:
			q_adv.data.add_(alpha * torch.sign(grad_q_adv.data)) # gradient assend by Sign-SGD
			if noisy:
				q_adv.data.add_(grad_noise_level * alpha * torch.rand(grad_q_adv.data.size()).to(device))
			# update x_adv:
			x_adv = augmax_model(xs, m_adv , q_adv)
			
		return x_adv, m_adv, q_adv


class FriendlyAugMaxAttack():
	def __init__(self, steps=10, alpha=0.1, tau=1, loss_fn=None, targeted=False):
		'''
		Args:
			eps: float. noise bound.
			steps: int. PGD attack step number.
			alpha: float. step size for PGD attack.
			loss_fn: loss function which is maximized to generate adversarial images.
			targeted: bool. If Ture, do targeted attack.
		'''
		self.steps = steps
		self.alpha = alpha 
		self.tau = tau
		self.targeted = targeted
		self.loss_fn = loss_fn if loss_fn else nn.CrossEntropyLoss(reduction="sum")    
	
	def attack(self, augmax_model, model, xs, labels, targets=None, verbose=False, noisy=False, grad_noise_level=0.1, random_level=False, device='cuda'):
		'''
		FAT (ICML'20)

		Args:
			xs: Tensor. Original images. size=(N,C,W,H)
			model: nn.Module. The model to be attacked.
			labels: Tensor. ground truth labels for x. size=(N,). Always useful to determine early stopping.
			targets: Tensor. target attack class for x. size=(N,). Useful only under targeted attack.
		Return:
			x_adv: Tensor. Adversarial images. size=(N,C,W,H)
		'''
		# 
		model.eval() # Parse by value (everything is parsed). Module is immutable variable; the original one won't be changed by this function. http://scipy-lectures.org/intro/language/functions.html#passing-by-value

		# 
		mixture_width = len(xs) - 1
		N = xs[0].size()[0]

		# initialize m_adv:
		m_adv = torch.rand(N).to(device) # random initialize in [0,1)
		m_adv = torch.clamp(m_adv, 0, 1) # clamp to range [0,1)
		m_adv.requires_grad=True

		# initialize ws_adv:
		q_adv = torch.rand((N,mixture_width), requires_grad=True).to(device) # random initialize

		# initialize x_adv
		x_adv = augmax_model(xs, m_adv , q_adv)

		# attack step size
		if random_level:
			alpha = self.alpha * np.random.choice([1,0.5,0.2,0.1])
		else:
			alpha = self.alpha

		# intialize counter
		if self.targeted:
			budget = (torch.ones(len(targets)) * self.tau).to(device)
		else:
			budget = (torch.ones(len(labels)) * self.tau).to(device)
			
		for t in range(self.steps):
			logits_adv = model(x_adv)
			pred = logits_adv.max(1, keepdim=True)[1]

			# Calculate the indexes of adversarial data those still needs to be iterated
			incorrect_sample_idx = (pred.squeeze() != labels)
			havent_used_up_budget_idx = (budget != 0)
			budget[torch.logical_and(incorrect_sample_idx, havent_used_up_budget_idx)] -= 1
			attack_sample_idx = (budget > 0).detach().cpu().numpy()
			if np.sum(attack_sample_idx) == 0: # all samples have used up their budgets
				if verbose:
					print('Early stopping at step %d' % t)
				break
			if self.targeted:
				loss_adv = - self.loss_fn(logits_adv, targets)
			else: # untargeted attack
				loss_adv = self.loss_fn(logits_adv, labels)
			# grad:
			grad_m_adv, grad_q_adv = torch.autograd.grad(loss_adv, [m_adv, q_adv], only_inputs=True)
			# mack out those:
			grad_m_adv.data[~attack_sample_idx].zero_()
			grad_q_adv.data[~attack_sample_idx].zero_()
			# update m:
			m_adv.data.add_(alpha * torch.sign(grad_m_adv.data)) # gradient assend by Sign-SGD
			if noisy:
				m_adv.data.add_(grad_noise_level * alpha * torch.rand(grad_m_adv.data.size()).to(device))
			m_adv = torch.clamp(m_adv, 0, 1) # clamp to RGB range [0,1]
			# update w1:
			q_adv.data.add_(alpha * torch.sign(grad_q_adv.data)) # gradient assend by Sign-SGD
			if noisy:
				q_adv.data.add_(grad_noise_level * alpha * torch.rand(grad_q_adv.data.size()).to(device))
			# update x_adv:
			x_adv = augmax_model(xs, m_adv, q_adv)
			
		return x_adv, m_adv, q_adv