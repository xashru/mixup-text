import torch
import numpy as np


use_cuda = True


def get_perm(x):
    """get random permutation"""
    batch_size = x.size()[1]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    return index


def mix_input(x, y, alpha):
    lam = np.random.beta(alpha, alpha)
    pos = int(round((1-lam) * x.shape[0]))
    index = get_perm(x)
    r = np.random.randint(2)
    if r == 0:
        x[:pos, :] = x[:pos, index]
    else:
        x[x.shape[0]-pos:, :] = x[x.shape[0]-pos:, index]
    return x, y, y[index], lam


def mix_input_front(x, y, alpha):
    lam = 0.5
    pos = int(round((1-lam) * x.shape[0]))
    index = get_perm(x)
    x[pos:, :] = x[:pos, index]
    return x, y, y[index], lam


def mix_none(x, y, alpha):
    return x, y, y, 1


MIXUP_METHODS = {
    'none': mix_none,
    'input': mix_input,
    'input-front': mix_input_front
}
