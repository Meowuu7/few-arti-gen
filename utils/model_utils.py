from functools import total_ordering
from random import uniform
import torch


import torch.nn as nn
import numpy as np

import os


import math

def check_and_make_dir(dir_fn):
    if not os.path.exists(dir_fn):
        os.mkdir(dir_fn)

def set_bn_not_training(module):
    if isinstance(module, nn.ModuleList):
        for block in module:
            set_bn_not_training(block)
    elif isinstance(module, nn.Sequential):
        for block in module:
            if isinstance(block, nn.BatchNorm1d) or isinstance(block, nn.BatchNorm2d):
                block.is_training = False
    else:
        raise ValueError("Not recognized module to set not training!")

def set_grad_to_none(module):
    if isinstance(module, nn.ModuleList):
        for block in module:
            set_grad_to_none(block)
    elif isinstance(module, nn.Sequential):
        for block in module:
            for param in block.parameters():
                param.grad = None
    else:
        raise ValueError("Not recognized module to set not training!")


def init_weight(blocks):
    for module in blocks:
        if isinstance(module, nn.Sequential):
            for subm in module:
                if isinstance(subm, nn.Linear):
                    nn.init.xavier_uniform_(subm.weight)
                    nn.init.zeros_(subm.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)


def construct_conv1d_modules(mlp_dims, n_in, last_act=True, bn=True, others_bn=True):
    rt_module_list = nn.ModuleList()
    for i, dim in enumerate(mlp_dims):
        inc, ouc = n_in if i == 0 else mlp_dims[i-1], dim
        if i < len(mlp_dims) - 1 or (i == len(mlp_dims) - 1 and last_act):
            # if others_bn and ouc % 4 == 0:
            if others_bn: # and ouc % 4 == 0:
                blk = nn.Sequential(
                        nn.Conv1d(in_channels=inc, out_channels=ouc, kernel_size=(1,), stride=(1,), bias=True),
                        nn.BatchNorm1d(num_features=ouc, eps=1e-5, momentum=0.1),
                    # nn.GroupNorm(num_groups=4, num_channels=ouc),
                        nn.ReLU()
                    )
            else:
                blk = nn.Sequential(
                    nn.Conv1d(in_channels=inc, out_channels=ouc, kernel_size=(1,), stride=(1,), bias=True),
                    nn.ReLU()
                )
        # elif bn  and ouc % 4 == 0:
        elif bn: #  and ouc % 4 == 0:
            blk = nn.Sequential(
                nn.Conv1d(in_channels=inc, out_channels=ouc, kernel_size=(1,), stride=(1,), bias=True),
                nn.BatchNorm1d(num_features=ouc, eps=1e-5, momentum=0.1),
                # nn.GroupNorm(num_groups=4, num_channels=ouc),
            )
        else:
            blk = nn.Sequential(
                nn.Conv1d(in_channels=inc, out_channels=ouc, kernel_size=(1,), stride=(1,), bias=True),
            )
        rt_module_list.append(blk)
    init_weight(rt_module_list)
    return rt_module_list


def construct_conv_modules(mlp_dims, n_in, last_act=True, bn=True):
    rt_module_list = nn.ModuleList()
    for i, dim in enumerate(mlp_dims):
        inc, ouc = n_in if i == 0 else mlp_dims[i-1], dim
        # if (i < len(mlp_dims) - 1 or (i == len(mlp_dims) - 1 and last_act))  and ouc % 4 == 0:
        if (i < len(mlp_dims) - 1 or (i == len(mlp_dims) - 1 and last_act)): #  and ouc % 4 == 0:
            blk = nn.Sequential(
                    nn.Conv2d(in_channels=inc, out_channels=ouc, kernel_size=(1, 1), stride=(1, 1), bias=True),
                    nn.BatchNorm2d(num_features=ouc, eps=1e-5, momentum=0.1),
                # nn.GroupNorm(num_groups=4, num_channels=ouc),
                    nn.ReLU()
                )
        # elif bn  and ouc % 4 == 0:
        elif bn: #  and ouc % 4 == 0:
            blk = nn.Sequential(
                nn.Conv2d(in_channels=inc, out_channels=ouc, kernel_size=(1, 1), stride=(1, 1), bias=True),
                nn.BatchNorm2d(num_features=ouc, eps=1e-5, momentum=0.1),
                # nn.GroupNorm(num_groups=4, num_channels=ouc),
            )
        else:
            blk = nn.Sequential(
                nn.Conv2d(in_channels=inc, out_channels=ouc, kernel_size=(1, 1), stride=(1, 1), bias=True),
            )
        rt_module_list.append(blk)
    init_weight(rt_module_list)
    return rt_module_list


 # @staticmethod
def apply_module_with_conv2d_bn(x, module):
    x = x.transpose(2, 3).contiguous().transpose(1, 2).contiguous()
    # print(x.size())
    for layer in module:
        for sublayer in layer:
            x = sublayer(x.contiguous())
        x = x.float()
    x = torch.transpose(x, 1, 2).transpose(2, 3)
    return x

# @staticmethod
def apply_module_with_conv1d_bn(x, module):
    x = x.transpose(1, 2).contiguous()
    # print(x.size())
    for layer in module:
        for sublayer in layer:
            x = sublayer(x.contiguous())
        x = x.float()
    x = torch.transpose(x, 1, 2)
    return x


def initialize_model_modules(modules):
    for zz in modules:
        if isinstance(zz, nn.Sequential):
            for zzz in zz:
                if isinstance(zzz, nn.Conv2d):
                    torch.nn.init.xavier_uniform_(zzz.weight)
                    if zzz.bias is not None:
                        torch.nn.init.zeros_(zzz.bias)
        elif isinstance(zz, nn.Conv2d):
            torch.nn.init.xavier_uniform_(zz.weight)
            if zz.bias is not None:
                torch.nn.init.zeros_(zz.bias)


def calculate_model_nparams(model):
    total = sum([param.nelement() for param in model.parameters()])
    print(f"Number of parameters: {total / 1e6}M.")
    return total


def sample_pts_from_mesh(vertices, triangles, npoints=1024):
    # vertices: torch.Tensor: n_verts x 3
    # triangles: torch.Tensor: n_faces x 3
    sampled_pcts = [] ### tot smapled pts 
    # pts_to_seg_idx = []
    sub_val = torch.min(triangles).item()

    tot_areas = []
    with torch.no_grad():
        for i in range(triangles.size(0)):
            cur_triangle = triangles[i].cpu().tolist()
            cur_triangle = [int(vv) for vv in cur_triangle]
            v_a, v_b, v_c = cur_triangle
            v_a, v_b, v_c = vertices[v_a - sub_val], vertices[v_b - sub_val], vertices[v_c - sub_val]
            ab, ac = v_b - v_a, v_c - v_a
            cos_ab_ac = (torch.sum(ab * ac) / torch.clamp(torch.sqrt(torch.sum(ab ** 2)) * torch.sqrt(torch.sum(ac ** 2)), min=1e-9)).item()
            sin_ab_ac = math.sqrt(min(max(0., 1. - cos_ab_ac ** 2), 1.))
            ### current area ###
            cur_area = 0.5 * sin_ab_ac * torch.sqrt(torch.sum(ab ** 2)).item() * torch.sqrt(torch.sum(ac ** 2)).item()
            tot_areas.append(cur_area)
    tot_sum_area = sum(tot_areas)
    n_pts_per_faces = [int(cur_area / float(tot_sum_area) * npoints) + 1 for cur_area in tot_areas]
    for i_f in range(triangles.size(0)):
        cur_triangle_n_samples = n_pts_per_faces[i_f]
        cur_triangle = triangles[i_f].cpu().tolist()
        cur_triangle = [int(vv) for vv in cur_triangle]
        v_a, v_b, v_c = cur_triangle
        v_a, v_b, v_c = vertices[v_a - sub_val], vertices[v_b - sub_val], vertices[v_c - sub_val]
        unform_dist = torch.distributions.uniform.Uniform(low=torch.zeros((cur_triangle_n_samples, ), dtype=torch.float32).cuda(), high=torch.ones((cur_triangle_n_samples, ), dtype=torch.float32).cuda())
        tmp_x = unform_dist.sample().tolist()
        tmp_y = unform_dist.sample().tolist()
        for xx, yy in zip(tmp_x, tmp_y):
            sqrt_xx, sqrt_yy = math.sqrt(xx), math.sqrt(yy)
            aa = 1. - sqrt_xx
            bb = sqrt_xx * (1. - yy)
            cc = yy * sqrt_xx
            cur_pos = v_a * aa + v_b * bb + v_c * cc
            sampled_pcts.append(cur_pos.unsqueeze(0)) ### get the random sampled point ###
    sampled_pcts = torch.cat(sampled_pcts, dim=0)
    
    return sampled_pcts
