# Copyright 2020 Deepmind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mesh data utilities."""
from math import sqrt
from platform import node
from tkinter import E
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d  # pylint: disable=unused-import
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import networkx as nx
import numpy as np
from scipy.misc import face
import six
from six.moves import range
from zmq import curve_keypair
# import tensorflow.compat.v1 as tf
# import tensorflow_probability as tfp
# tfd = tfp.distributions
import os
import torch

try:
    from torch_cluster import fps
except:
    pass

def save_obj_file(vertices, face_list, obj_fn, add_one=False):
  with open(obj_fn, "w") as wf:
    for i_v in range(vertices.shape[0]):
      cur_v_values = vertices[i_v]
      wf.write("v")
      for i_v_v in range(cur_v_values.shape[0]):
        wf.write(f" {float(cur_v_values[i_v_v].item())}")
      wf.write("\n")
    for i_f in range(len(face_list)):
      cur_face_idxes = face_list[i_f]
      wf.write("f")
      for cur_f_idx in range(len(cur_face_idxes)):
        wf.write(f" {cur_face_idxes[cur_f_idx] if not add_one else cur_face_idxes[cur_f_idx] + 1}")
      wf.write("\n")
    wf.close()

def mid_traverse_tree(tree):
  out = [tree['idx']]
  child_to_parent = {}
  if 'l' not in tree and 'r' not in tree:
    return out, {}
  else:
    if 'l' in tree:
      ltree = tree['l']
      lout, lctop = mid_traverse_tree(ltree)
      out = out + lout
      for node_idx in lctop:
        child_to_parent[node_idx] = lctop[node_idx]
      child_to_parent[ltree['idx']] = tree['idx']
    if 'r' in tree:
      rtree = tree['r']
      rout, rctop = mid_traverse_tree(rtree)
      out = out + rout
      for node_idx in rctop:
        child_to_parent[node_idx] = rctop[node_idx]
      child_to_parent[rtree['idx']] = tree['idx']
    return out, child_to_parent

def middle_traverse_tree(tree):
  # out = [tree['idx']]
  out = []
  child_to_parent = {}
  if 'l' not in tree and 'r' not in tree:
    out = [tree['idx']]
    return out, {}
  else:
    if 'l' in tree:
      ltree = tree['l']
      lout, lctop = middle_traverse_tree(ltree)
      out = out + lout
      for node_idx in lctop:
        child_to_parent[node_idx] = lctop[node_idx]
      child_to_parent[ltree['idx']] = tree['idx']
    out.append(tree['idx'])
    if 'r' in tree:
      rtree = tree['r']
      rout, rctop = middle_traverse_tree(rtree)
      out = out + rout
      for node_idx in rctop:
        child_to_parent[node_idx] = rctop[node_idx]
      child_to_parent[rtree['idx']] = tree['idx']
    return out, child_to_parent
    
# def middle_traverse_tree_with_joint(tree):
#   # out = [tree['idx']]
#   out = []
#   child_to_parent = {}
#   if 'l' not in tree and 'r' not in tree:
#     out = [tree['idx']]
#     return out, {}
#   else:
#     if 'l' in tree:
#       ltree = tree['l']
#       lout, lctop = middle_traverse_tree(ltree)
#       out = out + lout
#       for node_idx in lctop:
#         child_to_parent[node_idx] = lctop[node_idx]
#       child_to_parent[ltree['idx']] = tree['idx']
#     out.append(tree['idx'])
#     if 'r' in tree:
#       rtree = tree['r']
#       rout, rctop = middle_traverse_tree(rtree)
#       out = out + rout
#       for node_idx in rctop:
#         child_to_parent[node_idx] = rctop[node_idx]
#       child_to_parent[rtree['idx']] = tree['idx']
#     return out, child_to_parent
    



def farthest_point_sampling(pos: torch.FloatTensor, n_sampling: int):
    bz, N = pos.size(0), pos.size(1)
    feat_dim = pos.size(-1)
    device = pos.device
    sampling_ratio = float(n_sampling / N)
    pos_float = pos.float()

    batch = torch.arange(bz, dtype=torch.long).view(bz, 1).to(device)
    mult_one = torch.ones((N,), dtype=torch.long).view(1, N).to(device)

    batch = batch * mult_one
    batch = batch.view(-1)
    pos_float = pos_float.contiguous().view(-1, feat_dim).contiguous() # (bz x N, 3)
    # sampling_ratio = torch.tensor([sampling_ratio for _ in range(bz)], dtype=torch.float).to(device)
    # batch = torch.zeros((N, ), dtype=torch.long, device=device)
    sampled_idx = fps(pos_float, batch, ratio=sampling_ratio, random_start=False)
    # shape of sampled_idx?
    return sampled_idx



def random_shift(vertices, shift_factor=0.25):
  """Apply random shift to vertices."""
  max_shift_pos = tf.cast(255 - tf.reduce_max(vertices, axis=0), tf.float32)
  max_shift_pos = tf.maximum(max_shift_pos, 1e-9)

  max_shift_neg = tf.cast(tf.reduce_min(vertices, axis=0), tf.float32)
  max_shift_neg = tf.maximum(max_shift_neg, 1e-9)

  shift = tfd.TruncatedNormal(
      tf.zeros([1, 3]), shift_factor*255, -max_shift_neg,
      max_shift_pos).sample()
  shift = tf.cast(shift, tf.int32)
  vertices += shift
  return vertices


def make_vertex_model_dataset(ds, apply_random_shift=False):
  """Prepare dataset for vertex model training."""
  def _vertex_model_map_fn(example): # vertex_model_map_fn
    vertices = example['vertices']

    # Randomly shift vertices
    if apply_random_shift:
      vertices = random_shift(vertices)

    # Re-order vertex coordinates as (z, y, x). # vertices permute
    vertices_permuted = tf.stack(
        [vertices[:, 2], vertices[:, 1], vertices[:, 0]], axis=-1) # z, y, x 

    # Flatten quantized vertices, reindex starting from 1, and pad with a
    # zero stopping token.
    vertices_flat = tf.reshape(vertices_permuted, [-1]) # permuted and others? vertices_permuted
    example['vertices_flat'] = tf.pad(vertices_flat + 1, [[0, 1]]) # vertices_flat --- 

    # Create mask to indicate valid tokens after padding and batching.
    example['vertices_flat_mask'] = tf.ones_like( # mask # flat
        example['vertices_flat'], dtype=tf.float32) # vertices_flat
    return example
  return ds.map(_vertex_model_map_fn)


def make_vertex_model_dataset_part(ds, apply_random_shift=False):
  """Prepare dataset for vertex model training."""
  def _vertex_model_map_fn(example): # vertex_model_map_fn
    left_vertices = example['left_vertices']
    rgt_vertices = example['rgt_vertices']

    # Randomly shift vertices
    if apply_random_shift:
      left_vertices = random_shift(left_vertices) # left_vertices
      rgt_vertices = random_shift(rgt_vertices) # rgt_vertices

    # Re-order vertex coordinates as (z, y, x). # 
    left_vertices_permuted = tf.stack(
        [left_vertices[:, 2], left_vertices[:, 1], left_vertices[:, 0]], axis=-1)
    
    rgt_vertices_permuted = tf.stack(
        [rgt_vertices[:, 2], rgt_vertices[:, 1], rgt_vertices[:, 0]], axis=-1)

    # Flatten quantized vertices, reindex starting from 1, and pad with a
    # zero stopping token.
    left_vertices_flat = tf.reshape(left_vertices_permuted, [-1])
    rgt_vertices_flat = tf.reshape(rgt_vertices_permuted, [-1])
    example['left_vertices_flat'] = tf.pad(left_vertices_flat + 1, [[0, 1]])
    example['rgt_vertices_flat'] = tf.pad(rgt_vertices_flat + 1, [[0, 1]])

    # Create mask to indicate valid tokens after padding and batching.
    example['left_vertices_flat_mask'] = tf.ones_like(
        example['left_vertices_flat'], dtype=tf.float32)
    
    example['rgt_vertices_flat_mask'] = tf.ones_like(
        example['rgt_vertices_flat'], dtype=tf.float32)
    return example
  return ds.map(_vertex_model_map_fn)


def make_vertex_model_dataset_three_part(ds, apply_random_shift=False):
  """Prepare dataset for vertex model training."""
  def _vertex_model_map_fn(example): # vertex_model_map_fn
    left_vertices = example['left_vertices']
    rgt_vertices = example['rgt_vertices']
    base_vertices = example['base_vertices']

    # Randomly shift vertices
    if apply_random_shift:
      left_vertices = random_shift(left_vertices) # left_vertices
      rgt_vertices = random_shift(rgt_vertices) # rgt_vertices
      base_vertices = random_shift(base_vertices)

    # Re-order vertex coordinates as (z, y, x). # 
    left_vertices_permuted = tf.stack(
        [left_vertices[:, 2], left_vertices[:, 1], left_vertices[:, 0]], axis=-1)
    
    rgt_vertices_permuted = tf.stack(
        [rgt_vertices[:, 2], rgt_vertices[:, 1], rgt_vertices[:, 0]], axis=-1)
    
    base_vertices_permuted = tf.stack(
      [base_vertices[:, 2], base_vertices[:, 1], base_vertices[:, 0]], axis=-1
    )

    # Flatten quantized vertices, reindex starting from 1, and pad with a
    # zero stopping token.
    left_vertices_flat = tf.reshape(left_vertices_permuted, [-1])
    rgt_vertices_flat = tf.reshape(rgt_vertices_permuted, [-1])
    base_vertices_flat = tf.reshape(base_vertices_permuted, [-1])

    example['left_vertices_flat'] = tf.pad(left_vertices_flat + 1, [[0, 1]])
    example['rgt_vertices_flat'] = tf.pad(rgt_vertices_flat + 1, [[0, 1]])
    example['base_vertices_flat'] = tf.pad(base_vertices_flat + 1, [[0, 1]])

    # Create mask to indicate valid tokens after padding and batching.
    example['left_vertices_flat_mask'] = tf.ones_like(
        example['left_vertices_flat'], dtype=tf.float32)
    
    #### right vertices flat mask ####
    example['rgt_vertices_flat_mask'] = tf.ones_like(
        example['rgt_vertices_flat'], dtype=tf.float32)
    
    example['base_vertices_flat_mask'] = tf.ones_like(
        example['base_vertices_flat'], dtype=tf.float32)

    return example
  return ds.map(_vertex_model_map_fn)

def make_vertex_model_dataset_part_tree(ds, tree_traverse, child_to_parent, apply_random_shift=False):
  """Prepare dataset for vertex model training."""
  def _vertex_model_map_fn(example): # vertex_model_map_fn
    
    for node_idx in tree_traverse:
      cur_node_vertices = example[f'node_{node_idx}_vertices']
      if apply_random_shift: # random
        cur_node_vertices = random_shift(cur_node_vertices) # left_vertices
      cur_node_vertices_permuted = tf.stack(
        [cur_node_vertices[:, 2], cur_node_vertices[:, 1], cur_node_vertices[:, 0]], axis=-1
      )
      cur_node_vertices_flat = tf.reshape(cur_node_vertices_permuted, [-1])
      example[f'node_{node_idx}_vertices_flat'] = tf.pad(cur_node_vertices_flat + 1, [[0, 1]])
      example[f'node_{node_idx}_vertices_flat_mask'] = tf.ones_like(
        example[f'node_{node_idx}_vertices_flat'], dtype=tf.float32)
    return example
  return ds.map(_vertex_model_map_fn)


def make_vertex_model_dataset_pretext(ds, apply_random_shift=False):
  """Prepare dataset for vertex model training."""
  def _vertex_model_map_fn(example): # vertex_model_map_fn
    last_in_mask = example['last_in_mask']
    # last_in_mask = int(last_in_mask)
    vertices = example['vertices_masked']
    # vertices_vals, vertices_pad = vertices[:-1], vertices[-1:]
    # vertices_va

    vertices_ori = example['vertices']
    vertices_mask_identifier = example['vertices_mask_identifier']

    # Randomly shift vertices
    if apply_random_shift:
      vertices = random_shift(vertices)
      vertices_ori = random_shift(vertices_ori)

    # Re-order vertex coordinates as (z, y, x).
    # one vertex here
    # vertex model dataset

    # vertices_permuted = tf.stack(
    #   [vertices[:, 2], vertices[:, 1], vertices[:, 0]], axis=-1) # z, y, x 


    vertices_ori_permuted = tf.stack(
      [vertices_ori[:, 2], vertices_ori[:, 1], vertices_ori[:, 0]], axis=-1
    )
    # z, y, x

    # vertices_mask_identifier_permuted = tf.stack(
    #   [vertices_mask_identifier[:, 2], vertices_mask_identifier[:, 1], vertices_mask_identifier[:, 0]], axis=-1
    # )



    # coord_mask_ratio = 0.15

    # total_vertex_coordinates = int(tf.shape(vertices_permuted)[0])
    # nn_masked_coordinates = int(coord_mask_ratio * float(total_vertex_coordinates))
    # # todo: 0.5 total mask -- leave for future
    # print(total_vertex_coordinates, nn_masked_coordinates)
    # sampled_masked_coord_indices = np.random.choice(total_vertex_coordinates, size=nn_masked_coordinates, replace=False)
    
    # vertices_mask_identifier = np.zeros_like(vertices_permuted, dtype=np.float32)
    # vertices_mask_identifier[sampled_masked_coord_indices] = 1.
    # vertices_permuted[sampled_masked_coord_indices] = 2 ** discretization_bits
    
    # vertices_flat = vertices_permuted

    # Flatten quantized vertices, reindex starting from 1, and pad with a
    # zero stopping token. 

    # vertices_flat = tf.reshape(vertices_permuted, [-1])
    vertices_flat = vertices

    vertices_ori_flat = tf.reshape(vertices_ori_permuted, [-1])

    # vertices_mask_identifier_permuted_flat = tf.reshape(vertices_mask_identifier_permuted, [-1])
    vertices_mask_identifier_permuted_flat = vertices_mask_identifier

    # vertices_flat[sampled_masked_coord_indices] = 2 ** discretization_bits

    # vertices flat + stopping flag

    # example['vertices_flat'] = tf.pad(vertices_flat + 1, [[0, 1]]) # flat!
    # print('vertices_flat')
    example['vertices_flat'] = vertices_flat

    # vertices_mask_identifier_permuted_flat = tf.pad(vertices_mask_identifier_permuted_flat, [[0, 1]])

    # Create mask to indicate valid tokens after padding and batching.
    example['vertices_flat_mask'] = tf.ones_like( # mask # flat
        example['vertices_flat'], dtype=tf.float32) # vertices_flat
    example['vertices_ori_flat'] = tf.pad(vertices_ori_flat + 1, [[0, 1]]) # flat! # 
    
    example['vertices_mask_identifier'] = vertices_mask_identifier_permuted_flat

    return example
  return ds.map(_vertex_model_map_fn)


def make_vertex_model_dataset_with_mask(ds, apply_random_shift=False):
  """Prepare dataset for vertex model training."""
  def _vertex_model_map_fn(example): # vertex_model_map_fn
    # last_in_mask = example['last_in_mask']
    # last_in_mask = int(last_in_mask)
    vertices = example['vertices_masked']

    vertices_ori = example['vertices']
    vertices_mask_identifier = example['vertices_mask_identifier']

    # Randomly shift vertices
    # if apply_random_shift:
    #   vertices = random_shift(vertices)
    #   vertices_ori = random_shift(vertices_ori)

    vertices_ori_permuted = tf.stack(
      [vertices_ori[:, 2], vertices_ori[:, 1], vertices_ori[:, 0]], axis=-1
    )
    
    # vertices permuted...
    # vertices_flat = tf.reshape(vertices_permuted, [-1])
    vertices_flat = vertices

    vertices_ori_flat = tf.reshape(vertices_ori_permuted, [-1])

    # vertices_mask_identifier_permuted_flat = tf.reshape(vertices_mask_identifier_permuted, [-1])
    vertices_mask_identifier_permuted_flat = vertices_mask_identifier

    # vertices_flat[sampled_masked_coord_indices] = 2 ** discretization_bits

    # vertices flat + stopping flag

    # example['vertices_flat'] = tf.pad(vertices_flat + 1, [[0, 1]]) # flat!
    # print('vertices_flat')
    example['vertices_flat'] = vertices_flat # nn_verts + 1

    """ For finetuning tasks """
    # example['vertices_flat_for_pred'] = tf.pad(vertices_flat, [[0, 1]]) # nn_verts + 1
    example['vertices_flat_for_pred'] = vertices_flat # nn_verts + 1 # 
    

    # vertices_mask_identifier_permuted_flat = tf.pad(vertices_mask_identifier_permuted_flat, [[0, 1]])

    # Create mask to indicate valid tokens after padding and batching.
    example['vertices_flat_mask'] = tf.ones_like( # mask # flat
        example['vertices_flat'], dtype=tf.float32) # vertices_flat
    
    # example['vertices_flat_mask_for_pred'] = tf.pad(example['vertices_flat_mask'], [[0, 1]])

    example['vertices_flat_mask_for_pred'] = example['vertices_flat_mask']

    """ For pre-training task, [DUMB for STRT] + [Vertex Coords] + [DUMB for ending] """
    example['vertices_ori_flat'] = tf.pad(vertices_ori_flat + 1, [[1, 1]]) # flat!

    # example['vertices_flat'] = vertices_flat
    
    """ For pre-training task (mask identifier...), [MASK for STRT] + [MASKS for coords] + [MASK for ending] """
    example['vertices_mask_identifier'] = tf.pad(vertices_mask_identifier_permuted_flat, [[1, 0]]) # flat! # 

    # """ For pre-training task (mask identifier...) """
    # example['vertices_mask_identifier'] = vertices_mask_identifier_permuted_flat

    return example
  return ds.map(_vertex_model_map_fn)


def make_face_model_dataset(ds, apply_random_shift=False, shuffle_vertices=True, quantization_bits=8):
  """Prepare dataset for face model training."""
  def _face_model_map_fn(example):
    vertices = example['vertices']

    # Randomly shift vertices
    if apply_random_shift:
      vertices = random_shift(vertices)
    example['num_vertices'] = tf.shape(vertices)[0]

    # Optionally shuffle vertices and re-order faces to match
    if shuffle_vertices:
      permutation = tf.random_shuffle(tf.range(example['num_vertices']))
      vertices = tf.gather(vertices, permutation)
      face_permutation = tf.concat(
          [tf.constant([0, 1], dtype=tf.int32), tf.argsort(permutation) + 2],
          axis=0)
      example['faces'] = tf.cast(
          tf.gather(face_permutation, example['faces']), tf.int64)

    def _dequantize_verts(verts, n_bits):
      min_range = -0.5
      max_range = 0.5
      range_quantize = 2**n_bits - 1
      verts = tf.cast(verts, tf.float32)
      verts = verts * (max_range - min_range) / range_quantize + min_range
      return verts

    # Vertices are quantized. So convert to floats for input to face model
    example['vertices'] = _dequantize_verts(vertices, quantization_bits)
    example['vertices_mask'] = tf.ones_like(
        example['vertices'][..., 0], dtype=tf.float32)
    example['faces_mask'] = tf.ones_like(example['faces'], dtype=tf.float32)
    return example
  return ds.map(_face_model_map_fn)


# 
def make_face_model_dataset_part(
    ds, apply_random_shift=False, shuffle_vertices=True, quantization_bits=8):
  """Prepare dataset for face model training."""
  def _face_model_map_fn(example):
    # left vertices
    left_vertices = example['left_vertices']
    # right vertices
    rgt_vertices = example['rgt_vertices']

    # Randomly shift vertices
    if apply_random_shift:
      left_vertices = random_shift(left_vertices)
      rgt_vertices = random_shift(rgt_vertices)
    example['left_num_vertices'] = tf.shape(left_vertices)[0]
    example['rgt_num_vertices'] = tf.shape(rgt_vertices)[0]

    # Optionally shuffle vertices and re-order faces to match
    if shuffle_vertices:
      left_permutation = tf.random_shuffle(tf.range(example['left_num_vertices']))
      left_vertices = tf.gather(left_vertices, left_permutation)
      left_face_permutation = tf.concat(
          [tf.constant([0, 1], dtype=tf.int32), tf.argsort(left_permutation) + 2],
          axis=0)
      example['left_faces'] = tf.cast(
          tf.gather(left_face_permutation, example['left_faces']), tf.int64)
      
      rgt_permutation = tf.random_shuffle(tf.range(example['rgt_num_vertices']))
      rgt_vertices = tf.gather(rgt_vertices, rgt_permutation)
      # face permutation
      rgt_face_permutation = tf.concat(
          [tf.constant([0, 1], dtype=tf.int32), tf.argsort(rgt_permutation) + 2],
          axis=0)
      example['rgt_faces'] = tf.cast(
          tf.gather(rgt_face_permutation, example['rgt_faces']), tf.int64)

    def _dequantize_verts(verts, n_bits): # dequenti
      min_range = -0.5 # centrlized?
      max_range = 0.5
      range_quantize = 2**n_bits - 1
      verts = tf.cast(verts, tf.float32)
      # not a grid but just a quantized segment...
      verts = verts * (max_range - min_range) / range_quantize + min_range
      return verts

    # Vertices are quantized. So convert to floats for input to face model
    # 
    example['left_vertices'] = _dequantize_verts(left_vertices, quantization_bits)
    example['left_vertices_mask'] = tf.ones_like(
        example['left_vertices'][..., 0], dtype=tf.float32)
    example['left_faces_mask'] = tf.ones_like(example['left_faces'], dtype=tf.float32)

    # Vertices are quantized. So convert to floats for input to face model
    example['rgt_vertices'] = _dequantize_verts(rgt_vertices, quantization_bits)
    example['rgt_vertices_mask'] = tf.ones_like(
        example['rgt_vertices'][..., 0], dtype=tf.float32)
    example['rgt_faces_mask'] = tf.ones_like(example['rgt_faces'], dtype=tf.float32)
    return example
  return ds.map(_face_model_map_fn)


# 
def make_face_model_dataset_three_part(
    ds, apply_random_shift=False, shuffle_vertices=True, quantization_bits=8):
  """Prepare dataset for face model training."""
  def _face_model_map_fn(example):
    # left vertices
    left_vertices = example['left_vertices']
    # right vertices
    rgt_vertices = example['rgt_vertices']
    base_vertices = example['base_vertices']

    # Randomly shift vertices
    if apply_random_shift:
      left_vertices = random_shift(left_vertices)
      rgt_vertices = random_shift(rgt_vertices)
      base_vertices = random_shift(base_vertices)
    example['left_num_vertices'] = tf.shape(left_vertices)[0]
    example['rgt_num_vertices'] = tf.shape(rgt_vertices)[0]
    example['base_num_vertices'] = tf.shape(base_vertices)[0]

    # Optionally shuffle vertices and re-order faces to match
    if shuffle_vertices:
      left_permutation = tf.random_shuffle(tf.range(example['left_num_vertices']))
      left_vertices = tf.gather(left_vertices, left_permutation)
      left_face_permutation = tf.concat(
          [tf.constant([0, 1], dtype=tf.int32), tf.argsort(left_permutation) + 2],
          axis=0)
      example['left_faces'] = tf.cast(
          tf.gather(left_face_permutation, example['left_faces']), tf.int64)
      
      rgt_permutation = tf.random_shuffle(tf.range(example['rgt_num_vertices']))
      rgt_vertices = tf.gather(rgt_vertices, rgt_permutation)
      # face permutation
      rgt_face_permutation = tf.concat(
          [tf.constant([0, 1], dtype=tf.int32), tf.argsort(rgt_permutation) + 2],
          axis=0)
      example['rgt_faces'] = tf.cast(
          tf.gather(rgt_face_permutation, example['rgt_faces']), tf.int64)

      # permute vertices
      base_permutation = tf.random_shuffle(tf.range(example['base_num_vertices']))
      base_vertices = tf.gather(base_vertices, base_permutation)
      # face permutation
      base_face_permutation = tf.concat(
          [tf.constant([0, 1], dtype=tf.int32), tf.argsort(base_permutation) + 2],
          axis=0)
      example['base_faces'] = tf.cast(
          tf.gather(base_face_permutation, example['base_faces']), tf.int64)


    def _dequantize_verts(verts, n_bits): # dequenti
      min_range = -0.5 # centrlized?
      max_range = 0.5
      range_quantize = 2**n_bits - 1
      verts = tf.cast(verts, tf.float32)
      # not a grid but just a quantized segment...
      verts = verts * (max_range - min_range) / range_quantize + min_range
      return verts

    # Vertices are quantized. So convert to floats for input to face model
    # 
    example['left_vertices'] = _dequantize_verts(left_vertices, quantization_bits)
    example['left_vertices_mask'] = tf.ones_like(
        example['left_vertices'][..., 0], dtype=tf.float32)
    example['left_faces_mask'] = tf.ones_like(example['left_faces'], dtype=tf.float32)

    # Vertices are quantized. So convert to floats for input to face model
    example['rgt_vertices'] = _dequantize_verts(rgt_vertices, quantization_bits)
    example['rgt_vertices_mask'] = tf.ones_like(
        example['rgt_vertices'][..., 0], dtype=tf.float32)
    example['rgt_faces_mask'] = tf.ones_like(example['rgt_faces'], dtype=tf.float32)

    # dequantize vertices
    example['base_vertices'] = _dequantize_verts(base_vertices, quantization_bits)
    example['base_vertices_mask'] = tf.ones_like(
        example['base_vertices'][..., 0], dtype=tf.float32)
    example['base_faces_mask'] = tf.ones_like(example['base_faces'], dtype=tf.float32)
    return example
  return ds.map(_face_model_map_fn)


# 
def make_face_model_dataset_part_tree(
    ds, tree_traverse, child_to_parent, apply_random_shift=False, shuffle_vertices=True, quantization_bits=8):
  """Prepare dataset for face model training."""
  def _face_model_map_fn(example):

    def _dequantize_verts(verts, n_bits): # dequenti
      min_range = -0.5 # centrlized?
      max_range = 0.5
      range_quantize = 2**n_bits - 1
      verts = tf.cast(verts, tf.float32)
      # not a grid but just a quantized segment...
      verts = verts * (max_range - min_range) / range_quantize + min_range
      return verts

    
    # cur_node_vertices = example[f'node_{0}_vertices']
    # example[f'node_{0}_num_vertices'] = tf.shape(cur_node_vertices)[0]
    # example[f'node_{0}_vertices'] = _dequantize_verts(example[f'node_{0}_vertices'], quantization_bits)
    # example[f'node_{0}_vertices_mask'] = tf.ones_like(
    #   example[f'node_{0}_vertices'][..., 0], dtype=tf.float32)
    # example[f'node_{0}_faces_mask'] = tf.ones_like(example[f'node_{0}_faces'], dtype=tf.float32)

    # cur_node_vertices = example[f'node_{1}_vertices']
    # example[f'node_{1}_num_vertices'] = tf.shape(cur_node_vertices)[0]
    # example[f'node_{1}_vertices'] = _dequantize_verts(example[f'node_{1}_vertices'], quantization_bits)
    # example[f'node_{1}_vertices_mask'] = tf.ones_like(
    #   example[f'node_{1}_vertices'][..., 0], dtype=tf.float32)
    # example[f'node_{1}_faces_mask'] = tf.ones_like(example[f'node_{1}_faces'], dtype=tf.float32)

    # cur_node_vertices = example[f'node_{2}_vertices']
    # example[f'node_{2}_num_vertices'] = tf.shape(cur_node_vertices)[0]
    # example[f'node_{2}_vertices'] = _dequantize_verts(example[f'node_{2}_vertices'], quantization_bits)
    # example[f'node_{2}_vertices_mask'] = tf.ones_like(
    #   example[f'node_{2}_vertices'][..., 0], dtype=tf.float32)
    # example[f'node_{2}_faces_mask'] = tf.ones_like(example[f'node_{2}_faces'], dtype=tf.float32)

    

    for node_idx in tree_traverse: # 
    # for node_idx in range(3):
      print("current node_idx:", node_idx)
      cur_node_vertices = example[f'node_{node_idx}_vertices']
      if apply_random_shift:
        cur_node_vertices = random_shift(cur_node_vertices)
        # example[f'node_{node_idx}_vertices'] = random_shift(example[f'node_{node_idx}_vertices'])
      
      example[f'node_{node_idx}_num_vertices'] = tf.shape(cur_node_vertices)[0]
      # example[f'node_{node_idx}_num_vertices'] = tf.shape(example[f'node_{node_idx}_vertices'])[0]
      if shuffle_vertices:
        cur_node_permutation = tf.random_shuffle(tf.range(example[f'node_{node_idx}_num_vertices']))
        cur_node_vertices = tf.gather(cur_node_vertices, cur_node_permutation)
        cur_node_face_permutation = tf.concat(
            [tf.constant([0, 1], dtype=tf.int32), tf.argsort(cur_node_permutation) + 2],
            axis=0)
        example[f'node_{node_idx}_faces'] = tf.cast(
            tf.gather(cur_node_face_permutation, example[f'node_{node_idx}_faces']), tf.int64)
      # example[f'node_{node_idx}_vertices'] = _dequantize_verts(example[f'node_{node_idx}_vertices'], quantization_bits)

      example[f'node_{node_idx}_vertices'] = _dequantize_verts(cur_node_vertices, quantization_bits)
      example[f'node_{node_idx}_vertices_mask'] = tf.ones_like(
        example[f'node_{node_idx}_vertices'][..., 0], dtype=tf.float32)
      example[f'node_{node_idx}_faces_mask'] = tf.ones_like(example[f'node_{node_idx}_faces'], dtype=tf.float32)
    print(example)
    return example
  return ds.map(_face_model_map_fn)


def make_face_model_dataset_pretext(ds, apply_random_shift=False, shuffle_vertices=True, quantization_bits=8):
  """Prepare dataset for face model training."""
  def _face_model_map_fn(example):
    vertices = example['vertices']

    # Randomly shift vertices
    if apply_random_shift:
      vertices = random_shift(vertices)
    example['num_vertices'] = tf.shape(vertices)[0]

    # Optionally shuffle vertices and re-order faces to match
    # shuffle_vertices = False
    if shuffle_vertices:
      permutation = tf.random_shuffle(tf.range(example['num_vertices']))
      vertices = tf.gather(vertices, permutation)
      face_permutation = tf.concat(
          [tf.constant([0, 1, 2], dtype=tf.int32), tf.argsort(permutation) + 3],
          axis=0)
      example['faces'] = tf.cast(
          tf.gather(face_permutation, example['faces']), tf.int64)

    def _dequantize_verts(verts, n_bits): # _dequantize_verts
      min_range = -0.5
      max_range = 0.5
      range_quantize = 2**n_bits - 1
      verts = tf.cast(verts, tf.float32)
      verts = verts * (max_range - min_range) / range_quantize + min_range
      return verts

    # face_mask_ratio = 0.15
    # # todo: is it ok that we include the inter-face identifier and the stopping identifier into masking candidates?
    # total_face_indices = int(tf.shape(example['face'])[0])
    # nn_masked_faces = int(face_mask_ratio * float(total_face_indices))
    # # todo: 0.5 total mask -- leave for future
    # sampled_masked_face_indices = np.random.choice(total_face_indices, size=nn_masked_faces, replace=False)

    # face_mask_identifier = np.zeros_like(example['face'], dtype=np.float32)
    # example['face'][sampled_masked_face_indices] = max_seq_length
    # face_mask_identifier[sampled_masked_face_indices] = 1.

    # example['face_mask_identifier'] = face_mask_identifier

    # Vertices are quantized. So convert to floats for input to face model
    example['vertices'] = _dequantize_verts(vertices, quantization_bits)
    example['vertices_mask'] = tf.ones_like(example['vertices'][..., 0], dtype=tf.float32)
    example['faces_mask'] = tf.ones_like(example['faces'], dtype=tf.float32)
    return example
  return ds.map(_face_model_map_fn)



def make_face_model_dataset_with_mask(ds, apply_random_shift=False, shuffle_vertices=True, quantization_bits=8):
  """Prepare dataset for face model training."""
  def _face_model_map_fn(example):
    vertices = example['vertices']
    # Randomly shift vertices
    # if apply_random_shift:
    #   vertices = random_shift(vertices)

    example['num_vertices'] = tf.shape(vertices)[0]

    # Optionally shuffle vertices and re-order faces to match
    # shuffle_vertices = False

    # if shuffle_vertices:
    #   permutation = tf.random_shuffle(tf.range(example['num_vertices']))
    #   vertices = tf.gather(vertices, permutation)
    #   # 
    #   face_permutation = tf.concat(
    #       [tf.constant([0, 1, 2], dtype=tf.int32), tf.argsort(permutation) + 3],
    #       axis=0)
    #   example['faces'] = tf.cast(
    #       tf.gather(face_permutation, example['faces']), tf.int64)

    def _dequantize_verts(verts, n_bits): # _dequantize_verts
      min_range = -0.5
      max_range = 0.5
      range_quantize = 2**n_bits - 1
      verts = tf.cast(verts, tf.float32)
      verts = verts * (max_range - min_range) / range_quantize + min_range
      return verts

    # Vertices are quantized. So convert to floats for input to face model
    example['vertices'] = _dequantize_verts(vertices, quantization_bits)
    example['vertices_mask'] = tf.ones_like(example['vertices'][..., 0], dtype=tf.float32)
    # 
    example['faces_mask'] = tf.ones_like(example['faces'], dtype=tf.float32)
    

    face_mask_identifier = tf.pad(example['face_mask_identifier'], [[1, 0]]) # flat! # 
    example['face_mask_identifier'] = face_mask_identifier

    #### with the mask label ####
    example['faces_ori_for_pred'] = tf.pad(example['faces_ori'], [[0, 1]])
    example['faces_mask_for_pred'] = tf.pad(example['faces_mask'], [[0, 1]])
    #### with the mask label ####


    # example['faces_ori_for_pred'] = example['faces_ori']
    # example['faces_mask_for_pred'] = example['faces_mask']

    example['faces_ori'] = tf.pad(example['faces_ori'], [[1, 0]])

    return example
  return ds.map(_face_model_map_fn)



def read_obj_file(obj_file):
  """Read vertices and faces from already opened file."""
  vertex_list = []
  flat_vertices_list = []
  flat_vertices_indices = {}
  flat_triangles = []

  for line in obj_file:
    tokens = line.split()
    if not tokens:
      continue
    line_type = tokens[0]
    # We skip lines not starting with v or f.
    if line_type == 'v':
      vertex_list.append([float(x) for x in tokens[1:]])
    elif line_type == 'f':
      triangle = []
      for i in range(len(tokens) - 1):
        vertex_name = tokens[i + 1]
        if vertex_name in flat_vertices_indices:
          triangle.append(flat_vertices_indices[vertex_name])
          continue
        flat_vertex = []
        for index in six.ensure_str(vertex_name).split('/'):
          if not index:
            continue
          # obj triangle indices are 1 indexed, so subtract 1 here.
          flat_vertex += vertex_list[int(index) - 1]
        flat_vertex_index = len(flat_vertices_list)
        flat_vertices_list.append(flat_vertex)
        flat_vertices_indices[vertex_name] = flat_vertex_index
        triangle.append(flat_vertex_index)
      flat_triangles.append(triangle)

  return np.array(flat_vertices_list, dtype=np.float32), flat_triangles

def read_obj_file_ours(obj_fn, minus_one=False):
  vertices = []
  faces = []
  with open(obj_fn, "r") as rf:
    for line in rf:
      items = line.strip().split(" ")
      if items[0] == 'v':
        cur_verts = items[1:]
        cur_verts = [float(vv) for vv in cur_verts]
        vertices.append(cur_verts)
      elif items[0] == 'f':
        cur_faces = items[1:]
        cur_face_idxes = []
        for cur_f in cur_faces:
          try:
            cur_f_idx = int(cur_f.split("/")[0])
          except:
            cur_f_idx = int(cur_f.split("//")[0])
          cur_face_idxes.append(cur_f_idx if not minus_one else cur_f_idx - 1)
        faces.append(cur_face_idxes)
    rf.close()
  vertices = np.array(vertices, dtype=np.float)
  return vertices, faces


def read_obj(obj_path):
  """Open .obj file from the path provided and read vertices and faces."""

  with open(obj_path) as obj_file:
    return read_obj_file(obj_file)


def write_obj(vertices, faces, file_path, transpose=True, scale=1.):
  """Write vertices and faces to obj."""
  if transpose:
    vertices = vertices[:, [1, 2, 0]]
  vertices *= scale
  if faces is not None:
    if min(min(faces)) == 0:
      f_add = 1
    else:
      f_add = 0
  with open(file_path, 'w') as f:
    for v in vertices:
      f.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
    for face in faces:
      line = 'f'
      for i in face:
        line += ' {}'.format(i + f_add)
      line += '\n'
      f.write(line)

def quantize_angles(dir, n_bits=8):
  x, y, z = float(dir[0].item()), float(dir[1].item()), float(dir[2].item())
  xy_len = sqrt(x ** 2 + y ** 2)
  z_len = sqrt(z ** 2)
  xyz_len = sqrt(x ** 2 + y ** 2 + z ** 2)
  sin_beta = z / xyz_len
  sin_alpha = x / max(xy_len, 1e-9)
  
  min_range = -1.0
  max_range = 1.0
  range_quantize = 2 ** n_bits - 1
  sin_beta_quantize = (sin_beta - min_range) * range_quantize / (max_range - min_range)
  sin_alpha_quantize = (sin_alpha - min_range) * range_quantize / (max_range - min_range)
  alpha_beta_quantized_arr = np.array([sin_alpha_quantize, sin_beta_quantize], dtype=np.int32)
  return alpha_beta_quantized_arr


def dequantized_angles(quantized_angle, n_bits=8):
  min_range = -1.0
  max_range = 1.0
  range_quantize = 2**n_bits - 1
  quantized_angle = quantized_angle.astype('float32')
  quantized_angle = quantized_angle * (max_range - min_range) / range_quantize + min_range
  quantized_alpha, quantized_beta = quantized_angle[..., 0], quantized_angle[..., 1]
  z_val = quantized_beta
  xy_len = sqrt(1. - z_val ** 2)
  y_val = xy_len * quantized_alpha
  cos_alpha = sqrt(1. - quantized_alpha ** 2)
  x_val = xy_len * cos_alpha
  xyz_val = np.array([x_val, y_val, z_val], dtype=np.float32)
  return xyz_val
  
  

def quantize_verts(verts, n_bits=8):
  """Convert vertices in [-1., 1.] to discrete values in [0, n_bits**2 - 1]."""
  min_range = -0.5
  max_range = 0.5
  range_quantize = 2**n_bits - 1
  verts_quantize = (verts - min_range) * range_quantize / (
      max_range - min_range)
  return verts_quantize.astype('int32')


def dequantize_verts(verts, n_bits=8, add_noise=False):
  """Convert quantized vertices to floats."""
  min_range = -0.5
  max_range = 0.5
  range_quantize = 2**n_bits - 1
  verts = verts.astype('float32')
  verts = verts * (max_range - min_range) / range_quantize + min_range
  if add_noise:
    verts += np.random.uniform(size=verts.shape) * (1 / range_quantize)
  return verts


def face_to_cycles(face):
  """Find cycles in face."""
  g = nx.Graph()
  for v in range(len(face) - 1):
    g.add_edge(face[v], face[v + 1])
  g.add_edge(face[-1], face[0])
  return list(nx.cycle_basis(g))


def flatten_faces(faces, for_pretraining=False):
  """Converts from list of faces to flat face array with stopping indices."""
  if not faces:
    return np.array([0])
  else:
    if not for_pretraining:
      l = [f + [-1] for f in faces[:-1]]
      l += [faces[-1] + [-2]]
      ans = np.array([item for sublist in l for item in sublist]) + 2
    else:
      l = [f + [-2] for f in faces[:-1]]
      l += [faces[-1] + [-3]]
      ans = np.array([item for sublist in l for item in sublist]) + 3
    return ans  # pylint: disable=g-complex-comprehension


def unflatten_faces(flat_faces, for_pretraining=False):
  """Converts from flat face sequence to a list of separate faces."""
  if not for_pretraining:
    def group(seq):
      g = []
      for el in seq:
        if el == 0 or el == -1:
          yield g
          g = []
        else:
          g.append(el - 1)
      yield g
    outputs = list(group(flat_faces - 1))[:-1]
    # Remove empty faces
  else:
    print(f"her! unflatten!")
    def group(seq):
      g = []
      for el in seq:
        if el == -1 or el == -2 or el == 0: #
          yield g
          g = []
        else:
          g.append(el - 1)
      yield g
    outputs = list(group(flat_faces - 2))[:-1]
  return [o for o in outputs if len(o) > 2]

def post_process_meshes(vertices, faces):
  processed_vertices = []
  old_idx_to_new_idx = {}
  v_idx = 0
  for i_v in range(vertices.shape[0]):
    if i_v == 0:
      processed_vertices.append(vertices[i_v])
      old_idx_to_new_idx[i_v] = v_idx
      v_idx += 1
    else:
      cur_vert = vertices[i_v]
      prev_vert = processed_vertices[-1]
      cx, cy, cz = cur_vert.tolist()
      px, py, pz = prev_vert.tolist()
      if not ((cz > pz) or (cz == pz and cy > py) or (cz == pz and cy == py and cx > px)):
        continue
      else:
        processed_vertices.append(cur_vert)
        old_idx_to_new_idx[i_v] = v_idx
        v_idx += 1
  processed_vertices = np.array(processed_vertices, dtype=np.float32)
  ''' Finished processing vertices '''
  valid_faces = []
  valid_faces_minn_face_idx = []
  for i_f in range(len(faces)):
    if i_f == 0:
      valid_face = [vv for vv in faces[i_f] if vv in old_idx_to_new_idx]
      valid_face = [old_idx_to_new_idx[vv] for vv in valid_face]
      cur_face_np = np.array(valid_face, dtype=np.int32)
      # try: 
      cur_face_np = np.unique(cur_face_np, return_inverse=False)
      
      # cur_face_np
      cur_minn_face = np.min(cur_face_np)
      valid_faces.append(valid_face)
      valid_faces_minn_face_idx.append(cur_minn_face)
    else:
      valid_face = [vv for vv in faces[i_f] if vv in old_idx_to_new_idx]
      valid_face = [old_idx_to_new_idx[vv] for vv in valid_face]
      cur_face_np = np.array(valid_face, dtype=np.int32)
      cur_face_np = np.unique(cur_face_np, return_inverse=False)
      # cur_face_np
      cur_minn_face = np.min(cur_face_np)
      if cur_minn_face < valid_faces_minn_face_idx[-1]:
        continue
      valid_faces.append(valid_face)
      valid_faces_minn_face_idx.append(cur_minn_face)
  return processed_vertices, valid_faces
      


def center_vertices(vertices):
  """Translate the vertices so that bounding box is centered at zero."""
  vert_min = vertices.min(axis=0)
  vert_max = vertices.max(axis=0)
  vert_center = 0.5 * (vert_min + vert_max)
  return vertices - vert_center


def normalize_vertices_scale(vertices):
  """Scale the vertices so that the long diagonal of the bounding box is one."""
  vert_min = vertices.min(axis=0) # vert_min
  vert_max = vertices.max(axis=0) # vert_max
  extents = vert_max - vert_min # extents
  scale = np.sqrt(np.sum(extents**2)) # scale # 
  return vertices / scale

def augment_vertices_scale(vertices):
  scale_normalizing_factors = np.random.uniform(low=0.75, high=1.25, size=(3,))
  # min + (max - min) * scale_normalizing_factors (normalizing_factors)
  scale_normalizing_factors = np.reshape(scale_normalizing_factors, [1, 3])
  vertices = vertices * scale_normalizing_factors
  return vertices

# def
# warping function # construct the function
# 
def sample_gradients():
  # 
  sampled_gradients = np.random.normal(loc=0.0, scale=sqrt(0.5), size=(5,))
  # sampled_gradients: (5,)
  sampled_gradients = np.exp(sampled_gradients)
  # 
  return sampled_gradients

# 0.2
def warp_coord(sampled_gradients, val, reflect=False): # val from [0.0, 1.0] # from the 0.0 
  # assume single value as inputs
  grad_values = sampled_gradients.tolist()
  # mid_val
  mid_val = grad_values[0] * 0.2 + grad_values[1] * 0.2 + grad_values[2] * 0.1
  if reflect:
    grad_values[3] = grad_values[1]
    grad_values[4] = grad_values[0]

  # if not reflect:
  accum_val = 0.0
  for i_val in range(len(grad_values)):
    if val > 0.2 * (i_val + 1) and i_val < 4: # if i_val == 4, directly use the reamining length * corresponding gradient value
      accum_val += grad_values[i_val] * 0.2
    else:
      accum_val += grad_values[i_val] * (val - 0.2 * i_val)
      break
  return accum_val # modified value
    
  


def quantize_process_mesh(vertices, faces, tris=None, quantization_bits=8, unique_verts=True):
  """Quantize vertices, remove resulting duplicates and reindex faces."""
  # vertices, faces, _ = quantize_process_mesh(
      # vertices, faces, quantization_bits=quantization_bits)
  vertices = quantize_verts(vertices, quantization_bits)

  if unique_verts:
    vertices, inv = np.unique(vertices, axis=0, return_inverse=True)
  else:
    vertices = vertices
    inv = np.arange(0, vertices.shape[0], step=1, dtype=np.int32)


  

  # Sort vertices by z then y then x.
  sort_inds = np.lexsort(vertices.T)
  vertices = vertices[sort_inds]

  # Re-index faces and tris to re-ordered vertices.
  faces = [np.argsort(sort_inds)[inv[f]] for f in faces]
  if tris is not None:
    tris = np.array([np.argsort(sort_inds)[inv[t]] for t in tris])

  # Merging duplicate vertices and re-indexing the faces causes some faces to
  # contain loops (e.g [2, 3, 5, 2, 4]). Split these faces into distinct
  # sub-faces.
  sub_faces = []
  for f in faces:
    cliques = face_to_cycles(f)
    for c in cliques:
      c_length = len(c)
      # Only append faces with more than two verts.
      if c_length > 2:
        d = np.argmin(c)
        # Cyclically permute faces just that first index is the smallest.
        sub_faces.append([c[(d + i) % c_length] for i in range(c_length)])
  faces = sub_faces
  if tris is not None:
    tris = np.array([v for v in tris if len(set(v)) == len(v)])

  # Sort faces by lowest vertex indices. If two faces have the same lowest
  # index then sort by next lowest and so on.
  faces.sort(key=lambda f: tuple(sorted(f)))
  if tris is not None: # tris
    tris = tris.tolist()
    tris.sort(key=lambda f: tuple(sorted(f))) # sort
    tris = np.array(tris) # tris

  # After removing degenerate faces some vertices are now unreferenced.
  # Remove these.
  num_verts = vertices.shape[0]
  vert_connected = np.equal(
      np.arange(num_verts)[:, None], np.hstack(faces)[None]).any(axis=-1)
  vertices = vertices[vert_connected]

  # Re-index faces and tris to re-ordered vertices.
  vert_indices = (
      np.arange(num_verts) - np.cumsum(1 - vert_connected.astype('int')))
  faces = [vert_indices[f].tolist() for f in faces]
  if tris is not None:
    tris = np.array([vert_indices[t].tolist() for t in tris])

  return vertices, faces, tris


def process_mesh(vertices, faces, quantization_bits=8, max_seq_length=None, recenter_mesh=True, create_pretraining_mask=False, unique_verts=True, rt_unflatten_faces=False, pretraining=True): # 
  """Process mesh vertices and faces."""

  # Transpose so that z-axis is vertical.
  # transpose so that z-axis is vertical...
  #### todo: if we do not use the z-axis as the vertical axis? ####
  vertices = vertices[:, [2, 0, 1]] # [1, 2, 0] # z x y ---> z-up, y-up, x-up ---> 

  # Translate the vertices so that bounding box is centered at zero.
  # whether to recenter the mesh in the memory
  if recenter_mesh:
    # if shoudl recenter --- put in the center
    vertices = center_vertices(vertices)
    # Scale the vertices so that the long diagonal of the bounding box is equal
    # normalize scales
    vertices = normalize_vertices_scale(vertices) # normalize scale...; 

  # Quantize and sort vertices, remove resulting duplicates, sort and reindex
  # faces. quantize and sort vertices
  vertices, faces, _ = quantize_process_mesh( # process mesh
      vertices, faces, quantization_bits=quantization_bits, unique_verts=unique_verts)

  
  # nn_vertices = vertices.shape[0]

  # Flatten faces and add 'new face' = 1 and 'stop' = 0 tokens.
  if not rt_unflatten_faces:
    # else return flatten faces
    # whether flatten facess for the with-mask setting...
    faces = flatten_faces(faces, for_pretraining=create_pretraining_mask)
    # faces = flatten_faces(faces, for_pretraining=False)

  rt_dict = {
    'vertices': vertices,
    'faces': faces
  }

  if create_pretraining_mask:

    vertices_permuted = np.stack(
      [vertices[:, 2], vertices[:, 1], vertices[:, 0]], axis=-1
    )

    vertices_flatten = np.reshape(vertices_permuted + 1, [-1])


    coord_mask_ratio = 0.15
    total_vertex_coordinates = int(vertices_flatten.shape[0])
    nn_masked_coordinates = int(coord_mask_ratio * float(total_vertex_coordinates + 1))
    # todo: 0.5 total mask -- leave for future
    # print(total_vertex_coordinates, nn_masked_coordinates)
    sampled_masked_coord_indices = np.random.choice(total_vertex_coordinates + 1, size=nn_masked_coordinates, replace=False)

    sampled_masked_coord_indices = sampled_masked_coord_indices.tolist()
    sampled_masked_coord_indices_dict = {ii: 1 for ii in sampled_masked_coord_indices}
    last_in_mask = 1 if total_vertex_coordinates in sampled_masked_coord_indices_dict else 0
    sampled_masked_coord_indices = [ii for ii in sampled_masked_coord_indices_dict]
    sampled_masked_coord_indices = np.array(sampled_masked_coord_indices, dtype=np.int32)

    rt_dict['last_in_mask'] = last_in_mask

    # mask identifier
    vertices_mask_identifier = np.zeros_like(vertices_flatten, dtype=np.float32)
    
    # print(f"vertices_flatten: {vertices_flatten.shape}, ")
    vertices_mask_identifier = np.concatenate([vertices_mask_identifier, np.zeros((1,), dtype=np.float32)], axis=-1)

    if pretraining:
      vertices_mask_identifier[sampled_masked_coord_indices] = 1.
    else:
      vertices_mask_identifier[:] = 1.

    # vertices_flatten = np.concatenate
    # flatten + 1
    # vertices flatten and others..
    vertices_flatten = np.concatenate([vertices_flatten, np.zeros((1,), dtype=np.int32)], axis=-1)

    if pretraining:
      vertices_flatten[sampled_masked_coord_indices] = 2 ** quantization_bits + 1


    # vertices_masked = np.reshape(vertices_flatten, [vertices.shape[0], 3])
    vertices_masked = vertices_flatten
    # vertices_mask_identifier = np.reshape(vertices_mask_identifier, [vertices.shape[0], 3])

    # print(f"vertices_flatten: {vertices_flatten.shape}, vertices_masked: {vertices_masked.shape}, vertices_mask_identifier: {vertices_mask_identifier.shape}")
    rt_dict['vertices_masked'] = vertices_masked # masked! vertices masked...
    rt_dict['vertices_mask_identifier'] = vertices_mask_identifier

    assert max_seq_length is not None
    face_mask_ratio = 0.15

    # todo: is it ok that we include the inter-face identifier and the stopping identifier into masking candidates?
    # total_face_indices = int(tf.shape(faces)[0])

    faces_ori = np.ones_like(faces, dtype=np.int32)
    faces_ori[:] = faces[:]

    total_face_indices = int(faces.shape[0])
    nn_masked_faces = int(face_mask_ratio * float(total_face_indices))
    # todo: 0.5 total mask -- leave for future
    sampled_masked_face_indices = np.random.choice(total_face_indices, size=nn_masked_faces, replace=False)
    
    face_mask_identifier = np.zeros_like(faces, dtype=np.float32)
    # faces[sampled_masked_face_indices] = max_seq_length
    # masked faces
    if pretraining:
      faces[sampled_masked_face_indices] = 2 # total vertices + inter-face index + stopping index
      face_mask_identifier[sampled_masked_face_indices] = 1.
    else:
      face_mask_identifier[:] = 1.

    # print(f"mm_verts: {nn_vertices}, max face idx: {np.max(faces)}")

    
    rt_dict['face_mask_identifier'] = face_mask_identifier
    rt_dict['faces'] = faces
    rt_dict['faces_ori'] = faces_ori
    # vertices_flat = tf.reshape(vertices_permuted, [-1]) # permuted and others? vertices_permuted

    # vertices_flat[sampled_masked_coord_indices] = 2 ** discretization_bits

  # Discard degenerate meshes without faces.
  # return {
      # 'vertices': vertices,
      # 'faces': faces,
  # }
  return rt_dict


def load_process_mesh(mesh_obj_path, quantization_bits=8, recenter_mesh=True):
  """Load obj file and process."""
  # Load mesh
  vertices, faces = read_obj(mesh_obj_path)
  return process_mesh(vertices, faces, quantization_bits, recenter_mesh=recenter_mesh)


# mesh
def plot_meshes(mesh_list,
                ax_lims=0.3,
                fig_size=4,
                el=30,
                rot_start=120,
                vert_size=10,
                vert_alpha=0.75,
                n_cols=4, mesh_sv_fn=None):
  """Plots mesh data using matplotlib."""

  n_plot = len(mesh_list)
  n_cols = np.minimum(n_plot, n_cols)
  n_rows = np.ceil(n_plot / n_cols).astype('int')
  fig = plt.figure(figsize=(fig_size * n_cols, fig_size * n_rows))
  for p_inc, mesh in enumerate(mesh_list):

    for key in [
        'vertices', 'faces', 'vertices_conditional', 'pointcloud', 'class_name'
    ]:
      if key not in list(mesh.keys()):
        mesh[key] = None

    ax = fig.add_subplot(n_rows, n_cols, p_inc + 1, projection='3d')

    if mesh['faces'] is not None:
      if mesh['vertices_conditional'] is not None:
        face_verts = np.concatenate(
            [mesh['vertices_conditional'], mesh['vertices']], axis=0)
      else:
        face_verts = mesh['vertices'] # get face vertices...
      collection = []
      for f in mesh['faces']:
        collection.append(face_verts[f])
      plt_mesh = Poly3DCollection(collection)
      plt_mesh.set_edgecolor((0., 0., 0., 0.3))
      plt_mesh.set_facecolor((1, 0, 0, 0.2))
      ax.add_collection3d(plt_mesh)

    if mesh['vertices'] is not None:
      ax.scatter3D(
          mesh['vertices'][:, 0],
          mesh['vertices'][:, 1],
          mesh['vertices'][:, 2],
          lw=0.,
          s=vert_size,
          c='g',
          alpha=vert_alpha)

    if mesh['vertices_conditional'] is not None:
      ax.scatter3D(
          mesh['vertices_conditional'][:, 0],
          mesh['vertices_conditional'][:, 1],
          mesh['vertices_conditional'][:, 2],
          lw=0.,
          s=vert_size,
          c='b',
          alpha=vert_alpha)

    if mesh['pointcloud'] is not None:
      ax.scatter3D(
          mesh['pointcloud'][:, 0],
          mesh['pointcloud'][:, 1],
          mesh['pointcloud'][:, 2],
          lw=0.,
          s=2.5 * vert_size,
          c='b',
          alpha=1.)

    ax.set_xlim(-ax_lims, ax_lims)
    ax.set_ylim(-ax_lims, ax_lims)
    ax.set_zlim(-ax_lims, ax_lims)

    ax.view_init(el, rot_start)

    display_string = ''
    if mesh['faces'] is not None:
      display_string += 'Num. faces: {}\n'.format(len(collection))
    if mesh['vertices'] is not None:
      num_verts = mesh['vertices'].shape[0]
      if mesh['vertices_conditional'] is not None:
        num_verts += mesh['vertices_conditional'].shape[0]
      display_string += 'Num. verts: {}\n'.format(num_verts)
    if mesh['class_name'] is not None:
      display_string += 'Synset: {}'.format(mesh['class_name'])
    if mesh['pointcloud'] is not None:
      display_string += 'Num. pointcloud: {}\n'.format(
          mesh['pointcloud'].shape[0])
    # ax.text2D(0.05, 0.8, display_string, transform=ax.transAxes)
  plt.subplots_adjust(
      left=0., right=1., bottom=0., top=1., wspace=0.025, hspace=0.025)
  if mesh_sv_fn is not None:
    print("saving to ", mesh_sv_fn)
    # plt.show()
    try:
      fig.savefig(mesh_sv_fn, format="png")
    except:
      pass
    # plt.s
  else:
    plt.show()

def plot_sampled_meshes_new(v_sample, f_sample, sv_mesh_folder, cur_step=0, predict_joint=True, for_pretraining=False):
  # plot sampled meses
  if not os.path.exists(sv_mesh_folder):
    os.mkdir(sv_mesh_folder)

  part_vertex_samples = [v_sample]
  part_face_samples = [f_sample]

  # if predict_joint:
  #   pred_dir = v_sample['joint_dir']
  #   pred_pvp = v_sample['joint_pvp']
  #   print("pred_dir", pred_dir.shape, pred_dir)
  #   print("pred_pvp", pred_pvp.shape, pred_pvp)
  
  tot_n_samples = part_vertex_samples[0]['vertices'].shape[0]
  # tot_n_part = 2

  tot_mesh_list = []
  for i_p, (cur_part_v_samples_np, cur_part_f_samples_np) in enumerate(zip(part_vertex_samples, part_face_samples)):
      mesh_list = []  # mesh list
      for i_n in range(tot_n_samples):
          mesh_list.append(
              {
                  'vertices': cur_part_v_samples_np['vertices'][i_n][:cur_part_v_samples_np['num_vertices'][i_n]],
                  # num_vertices, faces
                  'faces': unflatten_faces( # unflat mesh
                      cur_part_f_samples_np['faces'][i_n][:cur_part_f_samples_np['num_face_indices'][i_n]], for_pretraining=for_pretraining)
              }
          )
      tot_mesh_list.append(mesh_list)
      # and write this obj file?
      # write_obj(vertices, faces, file_path, transpose=True, scale=1.):
      # write mesh objs
      for i_n in range(tot_n_samples):
        cur_mesh = mesh_list[i_n]
        cur_mesh_vertices, cur_mesh_faces = cur_mesh['vertices'], cur_mesh['faces']
        cur_mesh_sv_folder = os.path.join(sv_mesh_folder, f"training_step_{cur_step}_ins_{i_n}")
        os.makedirs(cur_mesh_sv_folder, exist_ok=True)
        cur_sv_mesh_fn = os.path.join(cur_mesh_sv_folder, "summary.obj")
        
        if cur_mesh_vertices.shape[0] > 0 and len(cur_mesh_faces) > 0:
          write_obj(cur_mesh_vertices, cur_mesh_faces, cur_sv_mesh_fn, transpose=False, scale=1.)  


def plot_sampled_meshes(v_sample, f_sample, sv_mesh_folder, cur_step=0, predict_joint=True):
  
  if not os.path.exists(sv_mesh_folder):
    os.mkdir(sv_mesh_folder)

  part_vertex_samples = [v_sample['left'], v_sample['rgt']]
  part_face_samples = [f_sample['left'], f_sample['rgt']]

  if predict_joint:
    pred_dir = v_sample['joint_dir']
    pred_pvp = v_sample['joint_pvp']
    print("pred_dir", pred_dir.shape, pred_dir)
    print("pred_pvp", pred_pvp.shape, pred_pvp)
  
  tot_n_samples = part_vertex_samples[0]['vertices'].shape[0]
  tot_n_part = 2


  tot_mesh_list = []
  for i_p, (cur_part_v_samples_np, cur_part_f_samples_np) in enumerate(zip(part_vertex_samples, part_face_samples)):
      mesh_list = []  # mesh list
      for i_n in range(tot_n_samples):
          mesh_list.append(
              {
                  'vertices': cur_part_v_samples_np['vertices'][i_n][:cur_part_v_samples_np['num_vertices'][i_n]],
                  # num_vertices, faces
                  'faces': unflatten_faces(
                      cur_part_f_samples_np['faces'][i_n][:cur_part_f_samples_np['num_face_indices'][i_n]])
              }
          )
          
      tot_mesh_list.append(mesh_list)
      # and write this obj file?
      # write_obj(vertices, faces, file_path, transpose=True, scale=1.):
      # write mesh objs
      for i_n in range(tot_n_samples):
        cur_mesh = mesh_list[i_n]
        cur_mesh_vertices, cur_mesh_faces = cur_mesh['vertices'], cur_mesh['faces']
        cur_mesh_sv_fn = os.path.join("./meshes", f"training_step_{cur_step}_part_{i_p}_ins_{i_n}.obj")
        if cur_mesh_vertices.shape[0] > 0 and len(cur_mesh_faces) > 0:
          write_obj(cur_mesh_vertices, cur_mesh_faces, cur_mesh_sv_fn, transpose=False, scale=1.)
        
      

  ###### plot mesh (predicted) ######
  tot_samples_mesh_dict = []
  for i_s in range(tot_n_samples):
      cur_s_tot_vertices = []
      cur_s_tot_faces = []
      cur_s_n_vertices = 0
      
      for i_p in range(tot_n_part):
          cur_s_cur_part_mesh_dict = tot_mesh_list[i_p][i_s]
          cur_s_cur_part_vertices, cur_s_cur_part_faces = cur_s_cur_part_mesh_dict['vertices'], \
                                                          cur_s_cur_part_mesh_dict['faces']
          cur_s_cur_part_new_faces = []
          for cur_s_cur_part_cur_face in cur_s_cur_part_faces:
              cur_s_cur_part_cur_new_face = [fid + cur_s_n_vertices for fid in cur_s_cur_part_cur_face]
              cur_s_cur_part_new_faces.append(cur_s_cur_part_cur_new_face)
          cur_s_n_vertices += cur_s_cur_part_vertices.shape[0]
          cur_s_tot_vertices.append(cur_s_cur_part_vertices)
          cur_s_tot_faces += cur_s_cur_part_new_faces

      cur_s_tot_vertices = np.concatenate(cur_s_tot_vertices, axis=0)
      cur_s_mesh_dict = {
          'vertices': cur_s_tot_vertices, 'faces': cur_s_tot_faces
      }
      tot_samples_mesh_dict.append(cur_s_mesh_dict)

  for i_s in range(tot_n_samples):
    cur_mesh = tot_samples_mesh_dict[i_s]
    cur_mesh_vertices, cur_mesh_faces = cur_mesh['vertices'], cur_mesh['faces']


    # cur_mesh_sv_fn = os.path.join(sv_mesh_folder, f"training_step_{cur_step}_ins_{i_s}.obj")
    # if cur_mesh_vertices.shape[0] > 0 and len(cur_mesh_faces) > 0:
    #   write_obj(cur_mesh_vertices, cur_mesh_faces, cur_mesh_sv_fn, transpose=False, scale=1.)

    ### save in a urdf-like manner ###
    cur_mesh_sv_folder = os.path.join(sv_mesh_folder, f"training_step_{cur_step}_ins_{i_s}")
    os.makedirs(cur_mesh_sv_folder, exist_ok=True)
    stat = {}
    for i_p in range(tot_n_part):
      cur_s_cur_part_mesh_dict = tot_mesh_list[i_p][i_s]
      cur_s_cur_part_vertices, cur_s_cur_part_faces = cur_s_cur_part_mesh_dict['vertices'], cur_s_cur_part_mesh_dict['faces']
      stat['link' + str(i_p + 1)] = "part" + str(i_p + 1) + ".obj"
      
      cur_s_cur_part_sv_fn = os.path.join(cur_mesh_sv_folder, "part" + str(i_p + 1) + ".obj")
      if cur_s_cur_part_vertices.shape[0] > 0 and len(cur_s_cur_part_faces) > 0:
        write_obj(cur_s_cur_part_vertices, cur_s_cur_part_faces, cur_s_cur_part_sv_fn, transpose=False, scale=1.)
    cur_s_summary_mesh_sv_fn = os.path.join(cur_mesh_sv_folder, "summary" + ".obj")
    if cur_mesh_vertices.shape[0] > 0 and len(cur_mesh_faces) > 0:
      write_obj(cur_mesh_vertices, cur_mesh_faces, cur_s_summary_mesh_sv_fn, transpose=False, scale=1.)
    if predict_joint:
      joint_stat = {'dir': pred_dir[i_s], 'pvp': pred_pvp[i_s]}
      stat['joint'] = joint_stat
    stat_sv_fn = os.path.join(cur_mesh_sv_folder, "stat.npy")
    np.save(stat_sv_fn, stat)
  ###### plot mesh (predicted) ######


  ###### plot mesh (translated) ######
  # tot_samples_mesh_dict = []
  # for i_s in range(tot_n_samples):
  #     cur_s_tot_vertices = []
  #     cur_s_tot_faces = []
  #     cur_s_n_vertices = 0
  #     cur_s_pred_pvp = pred_pvp[i_s]
      
  #     for i_p in range(tot_n_part):
  #         cur_s_cur_part_mesh_dict = tot_mesh_list[i_p][i_s]
  #         cur_s_cur_part_vertices, cur_s_cur_part_faces = cur_s_cur_part_mesh_dict['vertices'], \
  #                                                         cur_s_cur_part_mesh_dict['faces']
  #         cur_s_cur_part_new_faces = []
  #         for cur_s_cur_part_cur_face in cur_s_cur_part_faces:
  #             cur_s_cur_part_cur_new_face = [fid + cur_s_n_vertices for fid in cur_s_cur_part_cur_face]
  #             cur_s_cur_part_new_faces.append(cur_s_cur_part_cur_new_face)
  #         cur_s_n_vertices += cur_s_cur_part_vertices.shape[0]

  #         if i_p == 1:
  #           # min_rngs = cur_s_cur_part_vertices.min(1)
  #           # max_rngs = cur_s_cur_part_vertices.max(1)
  #           min_rngs = cur_s_cur_part_vertices.min(0)
  #           max_rngs = cur_s_cur_part_vertices.max(0)
  #           # shifted; cur_s_pred_pvp
  #           # shifted = np.array([0., cur_s_pred_pvp[1] - max_rngs[1], cur_s_pred_pvp[2] - min_rngs[2]], dtype=np.float)
  #           # shifted = np.reshape(shifted, [1, 3]) # 
  #           cur_s_pred_pvp = np.array([0., max_rngs[1], min_rngs[2]], dtype=np.float32)
  #           pvp_sample_pred_err = np.sum((cur_s_pred_pvp - pred_pvp[i_s]) ** 2)
  #           # print prediction err, pred pvp and real pvp
  #           print("cur_s, sample_pred_pvp_err:", pvp_sample_pred_err.item(), ";real val:", cur_s_pred_pvp, "; pred_val:", pred_pvp[i_s])

  #           # pred_pvp[i_s] = cur_s_pred_pvp
  #           shifted = np.zeros((1, 3), dtype=np.float32)
  #           cur_s_cur_part_vertices = cur_s_cur_part_vertices + shifted # shift vertices... # min_rngs
  #         # shifted
  #         cur_s_tot_vertices.append(cur_s_cur_part_vertices)
  #         cur_s_tot_faces += cur_s_cur_part_new_faces

  #     cur_s_tot_vertices = np.concatenate(cur_s_tot_vertices, axis=0)
  #     cur_s_mesh_dict = {
  #         'vertices': cur_s_tot_vertices, 'faces': cur_s_tot_faces
  #     }
  #     tot_samples_mesh_dict.append(cur_s_mesh_dict)

  # for i_s in range(tot_n_samples):
  #   cur_mesh = tot_samples_mesh_dict[i_s]
  #   cur_mesh_vertices, cur_mesh_faces = cur_mesh['vertices'], cur_mesh['faces']
  #   cur_mesh_sv_fn = os.path.join(sv_mesh_folder, f"training_step_{cur_step}_ins_{i_s}_shifted.obj")
  #   if cur_mesh_vertices.shape[0] > 0 and len(cur_mesh_faces) > 0:
  #     write_obj(cur_mesh_vertices, cur_mesh_faces, cur_mesh_sv_fn, transpose=False, scale=1.)
  ###### plot mesh (translated) ######



  ###### plot mesh (rotated) ######
  if predict_joint:
    from revolute_transform import revoluteTransform
    tot_samples_mesh_dict = []
    for i_s in range(tot_n_samples):
        cur_s_tot_vertices = []
        cur_s_tot_faces = []
        cur_s_n_vertices = 0
      
        cur_s_pred_dir = pred_dir[i_s]
        cur_s_pred_pvp = pred_pvp[i_s]
        print("current pred dir:", cur_s_pred_dir, "; current pred pvp:", cur_s_pred_pvp)
        cur_s_pred_dir = np.array([1.0, 0.0, 0.0], dtype=np.float)
        # cur_s_pred_pvp = cur_s_pred_pvp[[1, 2, 0]]

        for i_p in range(tot_n_part):
            cur_s_cur_part_mesh_dict = tot_mesh_list[i_p][i_s]
            cur_s_cur_part_vertices, cur_s_cur_part_faces = cur_s_cur_part_mesh_dict['vertices'], \
                                                            cur_s_cur_part_mesh_dict['faces']
                      
            if i_p == 1:
              cur_s_cur_part_vertices, _ = revoluteTransform(cur_s_cur_part_vertices, cur_s_pred_pvp, cur_s_pred_dir, -0.5 * np.pi) # reverse revolute vertices of the upper piece
              cur_s_cur_part_vertices = cur_s_cur_part_vertices[:, :3] # 
            cur_s_cur_part_new_faces = []
            for cur_s_cur_part_cur_face in cur_s_cur_part_faces:
                cur_s_cur_part_cur_new_face = [fid + cur_s_n_vertices for fid in cur_s_cur_part_cur_face]
                cur_s_cur_part_new_faces.append(cur_s_cur_part_cur_new_face)
            cur_s_n_vertices += cur_s_cur_part_vertices.shape[0]
            cur_s_tot_vertices.append(cur_s_cur_part_vertices)
            # print(f"i_s: {i_s}, i_p: {i_p}, n_vertices: {cur_s_cur_part_vertices.shape[0]}")
            cur_s_tot_faces += cur_s_cur_part_new_faces

        cur_s_tot_vertices = np.concatenate(cur_s_tot_vertices, axis=0)
        # print(f"i_s: {i_s}, n_cur_s_tot_vertices: {cur_s_tot_vertices.shape[0]}")
        cur_s_mesh_dict = {
            'vertices': cur_s_tot_vertices, 'faces': cur_s_tot_faces
        }
        tot_samples_mesh_dict.append(cur_s_mesh_dict)
    # plot_meshes(tot_samples_mesh_dict, ax_lims=0.5, mesh_sv_fn=f"./figs/training_step_{n}_part_{tot_n_part}_rot.png")  # plot the mesh;
    for i_s in range(tot_n_samples):
      cur_mesh = tot_samples_mesh_dict[i_s]
      cur_mesh_vertices, cur_mesh_faces = cur_mesh['vertices'], cur_mesh['faces']
      # rotated mesh fn
      cur_mesh_sv_fn = os.path.join(sv_mesh_folder, f"training_step_{cur_step}_ins_{i_s}_rot.obj")
      # write object in the file...
      if cur_mesh_vertices.shape[0] > 0 and len(cur_mesh_faces) > 0:
        write_obj(cur_mesh_vertices, cur_mesh_faces, cur_mesh_sv_fn, transpose=False, scale=1.)


def plot_sampled_meshes_three_parts(v_sample, f_sample, sv_mesh_folder, cur_step=0, predict_joint=True):
  
  if not os.path.exists(sv_mesh_folder):
    os.mkdir(sv_mesh_folder)

  part_vertex_samples = [v_sample['base'], v_sample['left'], v_sample['rgt']]
  part_face_samples = [f_sample['base'], f_sample['left'], f_sample['rgt']]

  if predict_joint:
    # pred_dir = v_sample['joint_dir']
    # pred_pvp = v_sample['joint_pvp']
    # print("pred_dir", pred_dir.shape, pred_dir)
    # print("pred_pvp", pred_pvp.shape, pred_pvp)
    pred_dir = v_sample['joint1_dir']
    pred_pvp = v_sample['joint1_pvp']

    pred_dir_rgt = v_sample['joint2_dir']
    pred_pvp_rgt = v_sample['joint2_pvp']
    print("pred_joint_dir_left", pred_dir)
    print('pred_joint_pvp_left', pred_pvp)

    print("pred_joint_dir_rgt", pred_dir_rgt)
    print('pred_joint_pvp_rgt', pred_pvp_rgt)

  
  
  tot_n_samples = part_vertex_samples[0]['vertices'].shape[0]
  tot_n_part = 3


  tot_mesh_list = []
  for i_p, (cur_part_v_samples_np, cur_part_f_samples_np) in enumerate(zip(part_vertex_samples, part_face_samples)):
      mesh_list = []  # mesh list
      for i_n in range(tot_n_samples):
          mesh_list.append(
              {
                  'vertices': cur_part_v_samples_np['vertices'][i_n][:cur_part_v_samples_np['num_vertices'][i_n]],
                  # num_vertices, faces
                  'faces': unflatten_faces(
                      cur_part_f_samples_np['faces'][i_n][:cur_part_f_samples_np['num_face_indices'][i_n]])
              }
          )
      tot_mesh_list.append(mesh_list)
      # and write this obj file?
      # write_obj(vertices, faces, file_path, transpose=True, scale=1.):
      # write mesh objs

      # for i_n in range(tot_n_samples):
      #   cur_mesh = mesh_list[i_n]
      #   cur_mesh_vertices, cur_mesh_faces = cur_mesh['vertices'], cur_mesh['faces']
      #   cur_mesh_sv_fn = os.path.join("./meshes", f"training_step_{cur_step}_part_{i_p}_ins_{i_n}.obj")
      #   if cur_mesh_vertices.shape[0] > 0 and len(cur_mesh_faces) > 0:
      #     write_obj(cur_mesh_vertices, cur_mesh_faces, cur_mesh_sv_fn, transpose=False, scale=1.)
        
      

  ###### plot mesh (predicted) ######
  tot_samples_mesh_dict = []
  for i_s in range(tot_n_samples):
      cur_s_tot_vertices = []
      cur_s_tot_faces = []
      cur_s_n_vertices = 0
      
      for i_p in range(tot_n_part):
          cur_s_cur_part_mesh_dict = tot_mesh_list[i_p][i_s]
          cur_s_cur_part_vertices, cur_s_cur_part_faces = cur_s_cur_part_mesh_dict['vertices'], \
                                                          cur_s_cur_part_mesh_dict['faces']
          cur_s_cur_part_new_faces = []
          for cur_s_cur_part_cur_face in cur_s_cur_part_faces:
              cur_s_cur_part_cur_new_face = [fid + cur_s_n_vertices for fid in cur_s_cur_part_cur_face]
              cur_s_cur_part_new_faces.append(cur_s_cur_part_cur_new_face)
          cur_s_n_vertices += cur_s_cur_part_vertices.shape[0]
          cur_s_tot_vertices.append(cur_s_cur_part_vertices)
          cur_s_tot_faces += cur_s_cur_part_new_faces

      cur_s_tot_vertices = np.concatenate(cur_s_tot_vertices, axis=0)
      cur_s_mesh_dict = {
          'vertices': cur_s_tot_vertices, 'faces': cur_s_tot_faces
      }
      tot_samples_mesh_dict.append(cur_s_mesh_dict)

  for i_s in range(tot_n_samples):
    cur_mesh = tot_samples_mesh_dict[i_s]
    cur_mesh_vertices, cur_mesh_faces = cur_mesh['vertices'], cur_mesh['faces']

    cur_s_pred_pvp_left = pred_pvp[i_s]
    cur_s_pred_pvp_rgt = pred_pvp_rgt[i_s]

    # base_vertices = cur_mesh_vertices[0]
    base_vertices = tot_mesh_list[0][i_s]['vertices']
# nerf-editing
    min_rngs = base_vertices.min(0)
    max_rngs = base_vertices.max(0)
    # shifted; cur_s_pred_pvp
    # shifted = np.array([0., cur_s_pred_pvp[1] - max_rngs[1], cur_s_pred_pvp[2] - min_rngs[2]], dtype=np.float)
    cur_s_real_pvp_rgt = np.array([0., max_rngs[1], min_rngs[2]], dtype=np.float32)
    cur_s_real_pvp_left = np.array([0., min_rngs[1], max_rngs[2]], dtype=np.float32)
    print(f"current real left pvp: {cur_s_real_pvp_left}, pred left pvp: {cur_s_pred_pvp_left}")
    print(f"current real rgt pvp: {cur_s_real_pvp_rgt}, pred rgt pvp: {cur_s_pred_pvp_rgt}")

    # cur_mesh_sv_fn = os.path.join(sv_mesh_folder, f"training_step_{cur_step}_ins_{i_s}.obj")
    # if cur_mesh_vertices.shape[0] > 0 and len(cur_mesh_faces) > 0:
    #   write_obj(cur_mesh_vertices, cur_mesh_faces, cur_mesh_sv_fn, transpose=False, scale=1.)

    ### save in a urdf-like manner ###
    cur_mesh_sv_folder = os.path.join(sv_mesh_folder, f"training_step_{cur_step}_ins_{i_s}")
    os.makedirs(cur_mesh_sv_folder, exist_ok=True)
    stat = {}
    for i_p in range(tot_n_part):
      cur_s_cur_part_mesh_dict = tot_mesh_list[i_p][i_s]
      cur_s_cur_part_vertices, cur_s_cur_part_faces = cur_s_cur_part_mesh_dict['vertices'], cur_s_cur_part_mesh_dict['faces']
      if i_p > 0:
        stat['link' + str(i_p)] = "part" + str(i_p) + ".obj"
      else:
        stat['base'] = 'base.obj'
      
      cur_s_cur_part_sv_fn = os.path.join(cur_mesh_sv_folder, "part" + str(i_p) + ".obj" if i_p > 0 else "base.obj")
      if cur_s_cur_part_vertices.shape[0] > 0 and len(cur_s_cur_part_faces) > 0:
        write_obj(cur_s_cur_part_vertices, cur_s_cur_part_faces, cur_s_cur_part_sv_fn, transpose=False, scale=1.)
    cur_s_summary_mesh_sv_fn = os.path.join(cur_mesh_sv_folder, "summary" + ".obj")
    if cur_mesh_vertices.shape[0] > 0 and len(cur_mesh_faces) > 0:
      write_obj(cur_mesh_vertices, cur_mesh_faces, cur_s_summary_mesh_sv_fn, transpose=False, scale=1.)
    if predict_joint:
      # joint_stat = {'dir': pred_dir[i_s], 'pvp': pred_pvp[i_s]}
      # stat['joint'] = joint_stat
      joint1_stat = {'dir': pred_dir[i_s], 'pvp': pred_pvp[i_s]}
      joint2_stat = {'dir': pred_dir_rgt[i_s], 'pvp': pred_pvp_rgt[i_s]}
      stat['joint'] = {
        'joint1': joint1_stat,
        'joint2': joint2_stat
      }
    stat_sv_fn = os.path.join(cur_mesh_sv_folder, "stat.npy")
    np.save(stat_sv_fn, stat)
  ###### plot mesh (predicted) ######


def merge_part_meshes(tot_mesh_list):
  tot_vertices = []
  tot_faces_list = []
  tot_n_vertices = 0
  for cur_mesh in tot_mesh_list:
    cur_vertices = cur_mesh['vertices']
    cur_faces = cur_mesh['faces']
    new_cur_faces = []
    for cur_face in cur_faces:
      new_cur_face = [f_idx + tot_n_vertices for f_idx in cur_face]
      new_cur_faces.append(new_cur_face)
    tot_vertices.append(cur_vertices)
    tot_n_vertices += cur_vertices.shape[0]
    tot_faces_list += new_cur_faces
  tot_vertices = np.concatenate(tot_vertices, axis=0)
  return tot_vertices, tot_faces_list


def merge_meshes(vertices_list, faces_list):
  tot_vertices = []
  tot_faces = []
  nn_verts = 0
  for cur_vertices, cur_faces in zip(vertices_list, faces_list):
    tot_vertices.append(cur_vertices)
    new_cur_faces = []
    for cur_face_idx in cur_faces:
      new_cur_face_idx = [vert_idx + nn_verts for vert_idx in cur_face_idx]
      new_cur_faces.append(new_cur_face_idx)
    nn_verts += cur_vertices.shape[0]
    tot_faces += new_cur_faces # get total-faces
  tot_vertices = np.concatenate(tot_vertices, axis=0)
  return tot_vertices, tot_faces



def plot_sampled_meshes_part_tree(v_sample, v_joint_sample, f_sample, sv_mesh_folder, n_samples, tree_traverse, child_to_parent, cur_step=0, predict_joint=True, post_process=False):
  
  if not os.path.exists(sv_mesh_folder):
    os.mkdir(sv_mesh_folder)

  for i_s in range(n_samples):
    cur_s_sv_folder = os.path.join(sv_mesh_folder, f"training_step_{cur_step}_ins_{i_s}")
    os.makedirs(cur_s_sv_folder, exist_ok=True)
    tot_meshes = []
    cur_s_joint_sample = {}
    for node_idx in tree_traverse:
      cur_node_v_samples = v_sample[f'node_{node_idx}']['vertices'][i_s]
      cur_node_v_samples_n = v_sample[f'node_{node_idx}']['num_vertices'][i_s]
      cur_node_f_samples = f_sample[f'node_{node_idx}']['faces'][i_s]
      cur_node_f_samples_indices = f_sample[f'node_{node_idx}']['num_face_indices'][i_s]
      
      if post_process:
        cur_verts, cur_faces = cur_node_v_samples[:cur_node_v_samples_n], unflatten_faces(
            cur_node_f_samples[:cur_node_f_samples_indices]
          )
        cur_verts, cur_faces = post_process_meshes(cur_verts, cur_faces)
        cur_mesh = {
          'vertices': cur_verts, 'faces': cur_faces
        }
      else:
        cur_mesh = {
          'vertices': cur_node_v_samples[:cur_node_v_samples_n],
          'faces': unflatten_faces(
            cur_node_f_samples[:cur_node_f_samples_indices]
          )
        }
      # meshes
      tot_meshes.append(cur_mesh)
      cur_s_cur_node_sv_fn = os.path.join(cur_s_sv_folder, "node" + str(node_idx) + ".obj")
      if cur_mesh['vertices'].shape[0] > 0 and len(cur_mesh['faces']) > 0:
        write_obj(cur_mesh['vertices'], cur_mesh['faces'], cur_s_cur_node_sv_fn, transpose=False, scale=1.)

    # s_joint_samples
    # save joints
    for node_idx in child_to_parent:
      cur_node_joint_dir = v_joint_sample[f'node_{node_idx}']['dir'][i_s]
      cur_node_joint_pvp = v_joint_sample[f'node_{node_idx}']['pvp'][i_s]
      cur_s_joint_sample[f'node_{node_idx}'] = {
        'dir': cur_node_joint_dir, 'pvp': cur_node_joint_pvp
      }

    tot_vertices, tot_faces = merge_part_meshes(tot_meshes)
    cur_s_summary_sv_fn = os.path.join(cur_s_sv_folder, "summary.obj")
    if tot_vertices.shape[0] > 0 and len(tot_faces) > 0:
      write_obj(tot_vertices, tot_faces, cur_s_summary_sv_fn, transpose=False, scale=1.)

    cur_s_stat_sv_fn = os.path.join(cur_s_sv_folder, "stat.npy")
    np.save(cur_s_stat_sv_fn, cur_s_joint_sample)

