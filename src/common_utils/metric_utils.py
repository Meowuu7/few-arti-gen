"""Mesh data utilities."""
from re import I
# import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d  # pylint: disable=unused-import
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import networkx as nx
import numpy as np
import six
import os
import math
import torch
# from options.options import opt
# from polygen_torch.
# from utils.constants import MASK_GRID_VALIE

try:
    from torch_cluster import fps
except:
    pass

MAX_RANGE = 0.1
MIN_RANGE = -0.1

import open3d as o3d

from tqdm import tqdm

''' other things for evaluation --> mesh laplacian (cannot be defiend if assume no source shapes...) '''


def coverage_ratio(dataloader_test, generated_shps, cd_threshold):
  ### generated_shps: a list of [nn_pc x 3] shapes
  if isinstance(generated_shps, list):
    generated_shps = [shp.unsqueeze(0) for shp in generated_shps]
    generated_shps = torch.cat(generated_shps, dim=0) ### tot_n x nn_pc x 3 --> 1 x tot_n x nn_pc x 3 --> 1 x 
  tot_minn_cd = []
  
  for i, data in tqdm(enumerate(dataloader_test), total=len(dataloader_test), smoothing=0.9):
    dst_pc = data['tar_pc'] ### bsz x nn_pc x 3 -> bsz x 1 x nn_pc x 1 x 3
    bsz = dst_pc.size(0) 
    dists_dst_pc_generations = torch.sum((dst_pc.unsqueeze(1).unsqueeze(-2) - generated_shps.unsqueeze(0).unsqueeze(2)) ** 2, dim=-1) ### bsz x tot_n x nn_pc x nn_pc
    dists_dst_pc_generations, _ = torch.min(dists_dst_pc_generations, dim=-1) ### bsz x tot_n x nn_pc
    dists_pc_dst_generations, _ = torch.min(dists_dst_pc_generations, dim=-2)
    cd_dst_generations = (dists_dst_pc_generations.mean(-1) + dists_pc_dst_generations.mean(-1)) / 2. ### bsz x tot_n
    minn_cd_dst_generations, _ = torch.min(cd_dst_generations, dim=-1) 
    tot_minn_cd.append(minn_cd_dst_generations)
  tot_minn_cd = torch.cat(minn_cd_dst_generations, dim=0)
  indicators = (tot_minn_cd <= cd_threshold).float()
  cov = torch.sum(indicators).item() / float(indicators.size(0))
  avg_dists = torch.sum(indicators * tot_minn_cd).item() / max(float(torch.sum(indicators).item()), 1e-6)
  return cov, avg_dists
