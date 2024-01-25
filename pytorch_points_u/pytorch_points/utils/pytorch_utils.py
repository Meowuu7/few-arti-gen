import torch
import numpy as np
import os
import warnings
from collections import OrderedDict
from ..misc import logger

saved_variables = {}
def save_grad(name):
    def hook(grad):
        saved_variables[name] = grad
    return hook

def check_values(tensor):
    """return true if tensor doesn't contain NaN or Inf"""
    return not (torch.any(torch.isnan(tensor)).item() or torch.any(torch.isinf(tensor)).item())

def linear_loss_weight(nepoch, epoch, max, init=0):
    """
    linearly vary scalar during training
    """
    return (max - init)/nepoch *epoch + init


def clamp_gradient(model, clip):
    for p in model.parameters():
        torch.nn.utils.clip_grad_value_(p, clip)

def clamp_gradient_norm(model, max_norm, norm_type=2):
    for p in model.parameters():
        torch.nn.utils.clip_grad_norm_(p, max_norm, norm_type=2)


def weights_init(m):
    """
    initialize the weighs of the network for Convolutional layers and batchnorm layers
    """
    if isinstance(m, (torch.nn.modules.conv._ConvNd, torch.nn.Linear)):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        torch.nn.init.constant_(m.bias, 0.0)
        torch.nn.init.constant_(m.weight, 1.0)

def save_network(net, directory, network_label, epoch_label=None, **kwargs):
    """
    save model to directory with name {network_label}_{epoch_label}.pth
    Args:
        net: pytorch model
        directory: output directory
        network_label: str
        epoch_label: convertible to str
        kwargs: additional value to be included
    """
    save_filename = "_".join((network_label, str(epoch_label))) + ".pth"
    save_path = os.path.join(directory, save_filename)
    merge_states = OrderedDict()
    merge_states["states"] = net.cpu().state_dict()
    for k in kwargs:
        merge_states[k] = kwargs[k]
    torch.save(merge_states, save_path)
    net = net.cuda()


def load_network(net, path):
    """
    load network parameters whose name exists in the pth file.
    return:
        INT trained step
    """
    # warnings.DeprecationWarning("load_network is deprecated. Use module.load_state_dict(strict=False) instead.")
    if isinstance(path, str):
        logger.info("loading network from {}".format(path))
        if path[-3:] == "pth":
            loaded_state = torch.load(path)
            if "states" in loaded_state:
                loaded_state = loaded_state["states"]
        else:
            loaded_state = np.load(path).item()
            if "states" in loaded_state:
                loaded_state = loaded_state["states"]
    elif isinstance(path, dict):
        loaded_state = path

    network = net.module if isinstance(
        net, torch.nn.DataParallel) else net

    missingkeys, unexpectedkeys = network.load_state_dict(loaded_state, strict=False)
    if len(missingkeys)>0:
        logger.warn("load_network {} missing keys".format(len(missingkeys)), "\n".join(missingkeys))
    if len(unexpectedkeys)>0:
        logger.warn("load_network {} unexpected keys".format(len(unexpectedkeys)), "\n".join(unexpectedkeys))

def fix_network_parameters(module):
    for param in module.parameters():
        param.requires_grad_(False)

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def tolerating_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    batch = [x for x in filter(lambda x: x is not None, batch)]
    return torch.utils.data.dataloader.default_collate(batch)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

class AverageValueMeter(object):
    """
    Slightly fancier than the standard AverageValueMeter
    """

    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def update(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.avg, self.std = np.nan, np.nan
        elif self.n == 1:
            self.avg = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.avg_old = self.avg
            self.m_s = 0.0
        else:
            self.avg = self.avg_old + (value - n * self.avg_old) / float(self.n)
            self.m_s += (value - self.avg_old) * (value - self.avg)
            self.avg_old = self.avg
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.avg, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.avg = np.nan
        self.avg_old = 0.0
        self.m_s = 0.0
        self.std = np.nan