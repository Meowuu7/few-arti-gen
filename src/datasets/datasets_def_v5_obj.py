from ctypes import util
from mimetypes import init
import torch
import numpy as np
import os
import random
# import utils.utils as utils
import utils
# from common_utils import dataset_utils

import time
### collect data from the dataset ###

# n_parts = 2
def my_collate(batch):
  
    # n_parts = len(batch[0]["src_ver"])
    # src_pc = [[] for _ in range(n_parts)]
    # src_name = [[] for _ in range(n_parts)]
    # tar_pc = [[ ] for _ in range(n_parts)]
    # tar_name = [[ ] for _ in range(n_parts)]
    # key_pts = [[ ] for _ in range(n_parts)]
    # w_pc = [[ ] for _ in range(n_parts)]
    # src_ver = [[ ] for _ in range(n_parts)]
    # src_face = [[ ] for _ in range(n_parts)]
    # tar_ver = [[ ] for _ in range(n_parts)]
    # tar_face = [[ ] for _ in range(n_parts)]
    # real_ver = [[ ] for _ in range(n_parts)]
    # real_face = [[ ] for _ in range(n_parts)]
    # w_mesh = [[ ] for _ in range(n_parts)]
    # src_edges = [[ ] for _ in range(n_parts)]
    # src_dofs = [[ ] for _ in range(n_parts)]
    # tar_edges = [[ ] for _ in range(n_parts)]
    # tar_dofs = [[ ] for _ in range(n_parts)]
    # tar_pc = [[ ] for _ in range(n_parts)]
    # tar_pc = [[ ] for _ in range(n_parts)]
    
    # cur_rt_dict['tar_scales'] = part_scale_info
    #     cur_rt_dict['tar_positions'] = part_position_info
    #     cur_rt_dict['tar_joint_dirs'] = joint_dir_info
    #     cur_rt_dict['tar_joint_pvps'] = joint_pvp_info
    #     cur_rt_dict['tar_extend'] = part_extend_info
        
    #     cur_rt_dict["joint_infos"] = part_joint_infos
    
    tar_scales =  []
    tar_positions = []
    tar_joint_dirs = []
    tar_joint_pvps = []
    tar_extend = [] ### tar_extend
    
    tot_obj_idxes = []
    
    
    tot_part_joint_infos = []

    for data in batch: 
      # "src_edges": src_edges, "src_dofs": src_dofs
      ### src_pc
        ###### obj-level information ######
        
        tar_scales.append(torch.from_numpy(data['tar_scales']).float().unsqueeze(0))
        tar_positions.append(torch.from_numpy(data['tar_positions']).float().unsqueeze(0))
        tar_joint_dirs.append(torch.from_numpy(data['tar_joint_dirs']).float().unsqueeze(0))
        tar_joint_pvps.append(torch.from_numpy(data['tar_joint_pvps']).float().unsqueeze(0))
        tar_extend.append(torch.from_numpy(data['tar_extend']).float().unsqueeze(0)) ### tar_extend and 
        
        tot_obj_idxes.append(data["idx"])
        
        # cur_obj_joint_info_tsr = []
        cur_obj_joint_infos = data["joint_infos"]
        cur_obj_joint_infos_tsr = []
        for part_joint_info in cur_obj_joint_infos:
          # print("axis of cur_part_joint_info", part_joint_info["axis"])
          for zz in ["dir", "center"]:
            # if isinstance(part_joint_info["axis"][zz], np.array):
            part_joint_info["axis"][zz] = torch.from_numpy(part_joint_info["axis"][zz]).float()
          cur_obj_joint_infos_tsr.append(part_joint_info)
        tot_part_joint_infos.append(cur_obj_joint_infos_tsr)
        
        # for i_p in range(n_parts):
        
        #   src_pc[i_p].append(torch.from_numpy(data["src_pc"][i_p]).unsqueeze(0).float())
        #   src_name[i_p].append(data["src_name"][i_p])
          
        #   tar_pc[i_p].append(torch.from_numpy(data["tar_pc"][i_p]).unsqueeze(0).float())
        #   tar_name[i_p].append(data["tar_name"][i_p])
          
        #   key_pts[i_p].append(torch.from_numpy(data["key_pts"][i_p]).unsqueeze(0).float())
        #   w_pc[i_p].append(torch.from_numpy(data["w_pc"][i_p]).unsqueeze(0).float())
          
        #   src_ver[i_p].append(torch.from_numpy(data["src_ver"][i_p]).float().cuda())
          
        #   src_face[i_p].append(torch.from_numpy(data["src_face"][i_p]))
          
        #   tar_ver[i_p].append(torch.from_numpy(data["tar_ver"][i_p]).float().cuda())
          
        #   tar_face[i_p].append(torch.from_numpy(data["tar_face"][i_p]))
        #   real_ver[i_p].append(torch.from_numpy(data["real_ver"][i_p]))
        #   real_face[i_p].append(torch.from_numpy(data["real_face"][i_p]))
        #   w_mesh[i_p].append(torch.from_numpy(data["w_mesh"][i_p]).float())
          
        #   # src_edges[i_p].append(torch.from_numpy(data['src_edges'][i_p]).long().unsqueeze(0).cuda()) #### src_edges
          # src_dofs[i_p].append(torch.from_numpy(data['src_dofs'][i_p]).float().unsqueeze(0).cuda()) #### vert_dofs 
          
          # tar_edges[i_p].append(torch.from_numpy(data['tar_edges'][i_p]).long().unsqueeze(0).cuda()) #### src_edges
          # tar_dofs[i_p].append(torch.from_numpy(data['tar_dofs'][i_p]).float().unsqueeze(0).cuda()) #### vert_dofs 
        
        
        
    tot_obj_idxes = torch.tensor(tot_obj_idxes, dtype=torch.long).long().cuda()
    ### obj-level infos ###
    tar_scales = torch.cat(tar_scales, dim=0).cuda()
    tar_positions = torch.cat(tar_positions, dim=0).cuda()
    tar_joint_dirs = torch.cat(tar_joint_dirs, dim=0).cuda()
    tar_joint_pvps = torch.cat(tar_joint_pvps, dim=0).cuda()
    tar_extend = torch.cat(tar_extend, dim=0).cuda() ### tar_extend ### tar_extend ###
    
    # print(f"tar_scales: {tar_scales.size()}, tar_positions: {tar_positions.size()}, tar_joint_dirs: {tar_joint_dirs.size()}, tar_joint_pvps: {tar_joint_pvps.size()}")

    # for i_p in range(n_parts):
      # src_pc[i_p] = torch.cat(src_pc[i_p]).cuda()
      # tar_pc[i_p] = torch.cat(tar_pc[i_p]).cuda()
      # key_pts[i_p] = torch.cat(key_pts[i_p]).cuda()
      # w_pc[i_p] = torch.cat(w_pc[i_p]).cuda()


    return {
      # "src_pc": src_pc, "src_name": src_name, "key_pts": key_pts,
      #       "tar_pc": tar_pc, "tar_name": tar_name, "w_pc": w_pc,
      #       "src_ver": src_ver, "src_face": src_face, "w_mesh": w_mesh,
      #       "tar_ver": tar_ver, "tar_face": tar_face,
      #       "real_ver": real_ver, "real_face": real_face,
      #       # "src_edges": src_edges, "src_dofs": src_dofs,
            # "tar_edges": tar_edges, "tar_dofs": tar_dofs,
            "tar_scales": tar_scales, "tar_positions": tar_positions, 
            "tar_joint_dirs": tar_joint_dirs, "tar_joint_pvps": tar_joint_pvps, "tar_extend": tar_extend, "joint_infos": tot_part_joint_infos, "idx": tot_obj_idxes
            }

import json


### 


import math

# joint_info = {
        # "type": cur_joint_type, 
        # "axis": {
        #   "dir": cur_joint_axis, "pvp": cur_joint_origin, "a": cur_joint_limit_a, "b": cur_joint_limit_b
        # },
      # }

def load_motion_infos(motion_attr_dict, part_nm_to_joint):
  cur_part_nm = motion_attr_dict['dof_name']
  cur_part_pvp = motion_attr_dict['center']
  cur_part_dir = motion_attr_dict['direction'] ### axis; center, dir
  cur_part_motion_type = motion_attr_dict["motion_type"]
  cur_part_pvp = np.array(cur_part_pvp)
  cur_part_dir = np.array(cur_part_dir)
  part_nm_to_joint[cur_part_nm] = (cur_part_pvp, cur_part_dir)
  if "children" in motion_attr_dict:
    cur_children_motion_attr_list = motion_attr_dict['children']
    for cur_child_motion_dict in cur_children_motion_attr_list: ### motion_attr_list...
      part_nm_to_joint = load_motion_infos(cur_child_motion_dict, part_nm_to_joint)
  return part_nm_to_joint


def load_joint_info_infos(motion_attr_dict, part_nm_to_joint):
  print(motion_attr_dict)
  cur_part_nm = motion_attr_dict['dof_name']
  cur_part_pvp = motion_attr_dict['center']
  cur_part_dir = motion_attr_dict['direction'] ### axis; center, dir
  cur_part_motion_type = motion_attr_dict["motion_type"]
  cur_part_motion_type = {"rotation": "revolute", "translation": "prismatic", "none": "none_motion"}[cur_part_motion_type]
  
  cur_part_pvp = np.array(cur_part_pvp)
  cur_part_dir = np.array(cur_part_dir)
  
  if cur_part_motion_type == "revolute":
    cur_part_limit_a, cur_part_limit_b = 0., 0.5 * math.pi
  else:
    cur_part_limit_a, cur_part_limit_b = 0., 1.
    
  axis_info = { "a": cur_part_limit_a, "b": cur_part_limit_b, "dir": cur_part_dir, "center": cur_part_pvp }
  cur_part_joint_info = {
    "type": cur_part_motion_type, "axis": axis_info
  }
  
  part_nm_to_joint[cur_part_nm] = cur_part_joint_info
  if "children" in motion_attr_dict:
    cur_children_motion_attr_list = motion_attr_dict['children']
    for cur_child_motion_dict in cur_children_motion_attr_list: ### motion_attr_list...
      part_nm_to_joint = load_joint_info_infos(cur_child_motion_dict, part_nm_to_joint)
  return part_nm_to_joint




def load_motion_infos_partnet_mobility(motion_attrs):
  #  ### axis; center, dir
  part_nm_to_joint = {}
  # print(motion_attrs)
  for k in motion_attrs:
    cur_part_axis = motion_attrs[k]['axis']
    if 'origin' in cur_part_axis:    
      cur_part_joint_center = cur_part_axis['origin']
      cur_part_joint_dir = cur_part_axis['direction']
      cur_part_joint_center = np.array(cur_part_joint_center, dtype=np.float32)
      cur_part_joint_dir = np.array(cur_part_joint_dir, dtype=np.float32)
      part_nm_to_joint[k] = (cur_part_joint_center, cur_part_joint_dir)
    else:
      cur_part_joint_center = np.array([0, 0., 0.], dtype=np.float32)
      cur_part_joint_dir = np.array([0., 0., 1.], dtype=np.float32)
      part_nm_to_joint[k] = (cur_part_joint_center, cur_part_joint_dir)
  return part_nm_to_joint
      

### perbsz, perpart
def load_joint_infos_partnet_mobility(motion_attrs):
  #  ### axis; center, dir
  part_nm_to_joint = {}
  # print(motion_attrs)
  print(motion_attrs)
  for k in motion_attrs:
    if "motion_type" in motion_attrs:
      cur_part_motion_type = motion_attrs[k]["motion_type"] if "motion_type" in motion_attrs[k] else "type"
      cur_part_axis = motion_attrs[k]['axis']
      if 'origin' in cur_part_axis:    
        cur_part_joint_center = cur_part_axis['origin']
        cur_part_joint_dir = cur_part_axis['direction']
        cur_part_motion_limit_info = motion_attrs[k]["limit"]
        
        cur_part_joint_center = np.array(cur_part_joint_center, dtype=np.float32)
        cur_part_joint_dir = np.array(cur_part_joint_dir, dtype=np.float32)
        limit_a, limit_b = cur_part_motion_limit_info["a"], cur_part_motion_limit_info["b"]
        
      else:
        cur_part_joint_center = np.array([0, 0., 0.], dtype=np.float32)
        cur_part_joint_dir = np.array([0., 0., 1.], dtype=np.float32)
        
        limit_a, limit_b = 0., 1.
      axis_info = { "a": limit_a, "b": limit_b, "dir": cur_part_joint_dir, "center": cur_part_joint_center }
      cur_part_joint_info = {
        "type": cur_part_motion_type, "axis": axis_info
      }
    else:
      cur_part_joint_info = motion_attrs[k] ### with type; and axis info ###
      cur_part_joint_info["axis"]["center"] = cur_part_joint_info["axis"]["pvp"]
    part_nm_to_joint[k] = cur_part_joint_info
      
  return part_nm_to_joint



def load_motion_infos_partnet_mobility_v2(motion_attrs):
  #  ### axis; center, dir
  part_nm_to_joint = {}
  # print(motion_attrs)
  for k in motion_attrs:
    cur_part_axis = motion_attrs[k]['axis']
    if 'pvp' in cur_part_axis:    
      cur_part_joint_center = cur_part_axis['pvp']
      cur_part_joint_dir = cur_part_axis['dir']
      cur_part_joint_center = np.array(cur_part_joint_center, dtype=np.float32)
      cur_part_joint_dir = np.array(cur_part_joint_dir, dtype=np.float32)
      part_nm_to_joint[k] = (cur_part_joint_center, cur_part_joint_dir)
    else:
      cur_part_joint_center = np.array([0, 0., 0.], dtype=np.float32)
      cur_part_joint_dir = np.array([0., 0., 1.], dtype=np.float32)
      part_nm_to_joint[k] = (cur_part_joint_center, cur_part_joint_dir)
  return part_nm_to_joint
      

### perbsz, perpart
def load_joint_infos_partnet_mobility_v2(motion_attrs):
  #  ### axis; center, dir
  part_nm_to_joint = {}
  # print(motion_attrs)
  for k in motion_attrs:
    
    cur_part_motion_type = motion_attrs[k]["type"] ## revolute or prismatic
    cur_part_axis = motion_attrs[k]['axis']
    if 'pvp' in cur_part_axis:    
      cur_part_joint_center = cur_part_axis['pvp']
      cur_part_joint_dir = cur_part_axis['dir']
      
      cur_part_joint_center = np.array(cur_part_joint_center, dtype=np.float32)
      cur_part_joint_dir = np.array(cur_part_joint_dir, dtype=np.float32)
      limit_a, limit_b = cur_part_axis["a"], cur_part_axis["b"]
    else:
      cur_part_joint_center = np.array([0, 0., 0.], dtype=np.float32)
      cur_part_joint_dir = np.array([0., 0., 1.], dtype=np.float32)
      
      limit_a, limit_b = 0., 1.
    axis_info = { "a": limit_a, "b": limit_b, "dir": cur_part_joint_dir, "center": cur_part_joint_center }
    cur_part_joint_info = {
      "type": cur_part_motion_type, "axis": axis_info
    }
    part_nm_to_joint[k] = cur_part_joint_info
      
  return part_nm_to_joint


  

def load_scale_position_joint_information_from_folder_partnet_mobility(inst_folder):
  part_objs = os.listdir(inst_folder)
  part_objs = [fn for fn in part_objs if fn.endswith(".obj")]
  part_nms = [fn.split(".")[0] for fn in part_objs]
  motion_attr_fn = "statistics.npy"
  
  #### for motiondataset ####
  # motion_attr_fn = os.path.join(inst_folder, motion_attr_fn) ### motion_attr_fn for the current instance
  
  # with open(motion_attr_fn, "r") as rf:
  #   motion_attrs = json.load(rf)

  motion_attrs = np.load(os.path.join(inst_folder, motion_attr_fn), allow_pickle=True).item()
  # print("motion_attrs", motion_attrs)
  # 'dof_name': 'dof_rootd', 'motion_type': 'none', 'center': [0, 0, 0], 'direction': [1, 0, 0], 'children': 
  
  if "_Style2" in inst_folder:
    part_nm_to_pvp_dir = load_motion_infos_partnet_mobility_v2(motion_attrs)
    part_nm_to_joint_info = load_joint_infos_partnet_mobility_v2(motion_attrs=motion_attrs)
  else:
    part_nm_to_pvp_dir = load_motion_infos_partnet_mobility(motion_attrs)
    part_nm_to_joint_info = load_joint_infos_partnet_mobility(motion_attrs=motion_attrs)
  
  part_nm_to_scale_center = {}
  part_nm_to_extend = {}
  part_nm_to_verts = {}
  for part_nm in part_nms:
    cur_part_obj_fn = f"{part_nm}.obj"
    cur_part_obj_fn = os.path.join(inst_folder, cur_part_obj_fn)
    cur_part_verts, cur_part_faces = utils.read_obj_file_ours(cur_part_obj_fn, sub_one=True)
    cur_part_center = utils.get_vertices_center(cur_part_verts) ### cur_part_verts ### vertices  center
    cur_part_scale = utils.get_vertices_scale(cur_part_verts) ### cur_part_verts
    ### cur-part_verts
    cur_part_verts_minn = cur_part_verts.min(axis=0)
    cur_part_verts_maxx = cur_part_verts.max(axis=0)
    cur_part_extend = cur_part_verts_maxx - cur_part_verts_minn
    ### the diagonal length ###
    part_nm_to_scale_center[part_nm] = (cur_part_center, cur_part_scale)
    part_nm_to_extend[part_nm] = cur_part_extend
    part_nm_to_verts[part_nm] = cur_part_verts
  return part_nm_to_scale_center, part_nm_to_pvp_dir, part_nm_to_extend, part_nm_to_verts, part_nm_to_joint_info



def load_scale_position_joint_information_from_folder(inst_folder):
  part_objs = os.listdir(inst_folder)
  part_objs = [fn for fn in part_objs if fn.endswith(".obj")]
  part_nms = [fn.split(".")[0] for fn in part_objs]
  motion_attr_fn = "motion_attributes.json"
  
  #### for motiondataset ####
  motion_attr_fn = os.path.join(inst_folder, motion_attr_fn) ### motion_attr_fn for the current instance
  with open(motion_attr_fn, "r") as rf:
    motion_attrs = json.load(rf)
  # print(motion_attrs)
  # 'dof_name': 'dof_rootd', 'motion_type': 'none', 'center': [0, 0, 0], 'direction': [1, 0, 0], 'children': 
  part_nm_to_pvp_dir = load_motion_infos(motion_attrs, {})
  
  part_nm_to_joint_info = load_joint_info_infos(motion_attrs, {})
  
  
  part_nm_to_scale_center = {}
  part_nm_to_extend = {}
  part_nm_to_verts = {}
  for part_nm in part_nms:
    cur_part_obj_fn = f"{part_nm}.obj"
    cur_part_obj_fn = os.path.join(inst_folder, cur_part_obj_fn)
    cur_part_verts, cur_part_faces = utils.read_obj_file_ours(cur_part_obj_fn, sub_one=True)
    cur_part_center = utils.get_vertices_center(cur_part_verts) ### cur_part_verts ### vertices  center
    cur_part_scale = utils.get_vertices_scale(cur_part_verts) ### cur_part_verts
    ### cur-part_verts
    cur_part_verts_minn = cur_part_verts.min(axis=0)
    cur_part_verts_maxx = cur_part_verts.max(axis=0)
    cur_part_extend = cur_part_verts_maxx - cur_part_verts_minn
    ### the diagonal length ###
    part_nm_to_scale_center[part_nm] = (cur_part_center, cur_part_scale)
    part_nm_to_extend[part_nm] = cur_part_extend
    part_nm_to_verts[part_nm] = cur_part_verts
    
  
  return part_nm_to_scale_center, part_nm_to_pvp_dir, part_nm_to_extend, part_nm_to_verts, part_nm_to_joint_info
 
 
def cd_loss_np(arr_a, arr_b):
  na  = arr_a.shape[0]
  nb = arr_b.shape[0]
  dist_a_b = np.sum((np.reshape(arr_a, (na, 1, 3)) - np.reshape(arr_b, (1, nb, 3))) ** 2, axis=-1) #### na x nb
  dist_a_to_b = np.min(dist_a_b, axis=-1).mean()
  dist_b_to_a = np.min(dist_a_b, axis=-2).mean()
  avg_dist_a_b = ((dist_a_to_b + dist_b_to_a) / 2).item() ### float of avg of distances
  return avg_dist_a_b



def merge_convexes(cvx_to_pts_a, cvx_to_pts_b, thres=0.001):
  ### cvx_idx: pts_arrry
  ## chamfer distances
  cvx_idx_to_parent = {}
  
  for c_a in cvx_to_pts_a:
    ca_pts = cvx_to_pts_a[c_a]
    for c_b in cvx_to_pts_a:
      if not c_b < c_a:
        continue
      cb_pts = cvx_to_pts_a[c_b]
      cd_a_b = cd_loss_np(ca_pts, cb_pts) ### a value
      if cd_a_b <= thres:
        cvx_idx_to_parent[c_a] = c_b
  



class ChairDataset(torch.utils.data.Dataset):
    def __init__(self, phase="train", data_dir="../data/chair", opt=None, use_shots=False):
        super().__init__()
        self.data_dir = data_dir
        self.phase = phase
        self.opt = opt
        self.load_meta = self.opt.load_meta
        self.num_sampled_pts = 4096
        self.n_shots = self.opt.n_shots
        # self.use_paired_data = self.opt.use_paired_data
        self.random_scaling = self.opt.random_scaling
        
        self.n_keypoints = self.opt.n_keypoints
        
        # self.cvx_to_verts_sufix = self.opt.cvx_to_verts_sufix
        
        self.n_parts = self.opt.n_parts ### obj_data_root_folder ### obj_data_root_folder ###
        self.obj_data_root_folder = self.opt.obj_data_root_folder
        print(f"Object data root folder: {self.obj_data_root_folder} with n_parts: {self.n_parts}") ### 
        
        
        # tot_insts = [int(inss_str.split("_")[-1]) for inss_str in self.dst_models[0]] ### inst idxes for intances
        tot_obj_inst_folders = os.listdir(self.obj_data_root_folder)
        tot_obj_inst_folders = [fn for fn in tot_obj_inst_folders if os.path.isdir(os.path.join(self.obj_data_root_folder, fn))] ### get obj file folders
        
        
        tot_obj_inst_folders = sorted(tot_obj_inst_folders)
        
        if use_shots:
          tot_obj_inst_folders = tot_obj_inst_folders[self.opt.n_shots:]
        
        ''' no shots for objs... ''' 
        # if self.opt.n_shots > 0:
        #   if phase == "train":
        #     tot_obj_inst_folders = tot_obj_inst_folders[:self.opt.n_shots]
        #   else:
        #     tot_obj_inst_folders = tot_obj_inst_folders[self.opt.n_shots:]
        
        # tot_obj_inst_folders = [tot_obj_inst_folders[ii] for ii in tot_insts] ### selected obj inst folders
        
        self.tot_obj_inst_folders = tot_obj_inst_folders
        
        print(f"tot_obj_inst_folders: {len(self.tot_obj_inst_folders)}")
        print(f"tot_obj_inst_folders: {self.tot_obj_inst_folders}")
        
        
        
    def load_data_from_model_indicator(self, rt_folder, model_indicator):
        ### read data from rt_folder and the model_indicator ###
        cur_model = model_indicator
        
        cur_keypoints_fn = cur_model + f"_manifold_tetra.mesh__sf.keypoints_{self.n_keypoints}.obj"
        cur_sampled_pts_fn = cur_model + "_manifold_tetra.mesh__sf.sampled_4096.obj"
        cur_surface_pts_fn = cur_model + "_manifold_tetra.mesh__sf.obj"
        cur_weights_sampled_fn = cur_model + f"_manifold_tetra.mesh__sf.keypoints_{self.n_keypoints}.weights.sampled_4096.txt"
        cur_weights_tot_fn = cur_model + f"_manifold_tetra.mesh__sf.keypoints_{self.n_keypoints}.weights.txt"
        
        cur_keypoints, _ =  utils.read_obj_file_ours(os.path.join(rt_folder, cur_keypoints_fn))
        cur_sampled, _ = utils.read_obj_file_ours(os.path.join(rt_folder, cur_sampled_pts_fn))
        cur_surface, cur_faces = utils.read_obj_file_ours(os.path.join(rt_folder, cur_surface_pts_fn), sub_one=True)
        cur_weights_sampled_keypoints = utils.load_txt(os.path.join(rt_folder, cur_weights_sampled_fn))
        cur_weights_mesh_keypoints = utils.load_txt(os.path.join(rt_folder, cur_weights_tot_fn))
        cur_faces = np.array(cur_faces, dtype=np.long) # .long()
        
        return cur_sampled, cur_keypoints, cur_surface, cur_faces, cur_weights_sampled_keypoints, cur_weights_mesh_keypoints


    def __len__(self):
        # return self.dst_n_models[0]
        return len(self.tot_obj_inst_folders)
        # return self.src_n_models * self.dst_n_models

    ### random extend scaling
    #### (pc - c) * (random_scale) / current_scale * ori_scale ---> bounding boxes should not change ####
    def apply_random_scaling(self, vertices, pc, keypoints):
        scale_normalizing_factors = np.random.uniform(low=0.75, high=1.25, size=(3,))
        # min + (max - min) * scale_normalizing_factors (normalizing_factors)
        scale_normalizing_factors = np.reshape(scale_normalizing_factors, [1, 3])
        #### vertices center of the vertices ####
        vertices_center = utils.get_vertices_center(vertices)
        vertices_center = np.reshape(vertices_center, (1, 3))

        centered_vertices = vertices - vertices_center
        centered_pc = pc - vertices_center ### centered_oc
        centered_keypoints = keypoints - vertices_center ### centered_keypoints

        vertices_scale = utils.get_vertices_scale(centered_vertices)

        scaled_vertices = centered_vertices * scale_normalizing_factors
        scaled_pc = centered_pc * scale_normalizing_factors
        scaled_keypoints = centered_keypoints * scale_normalizing_factors

        scaled_vertices = utils.normalize_vertices_scale(scaled_vertices) * vertices_scale
        scaled_pc = utils.normalize_vertices_scale(scaled_pc) * vertices_scale
        scaled_keypoints = utils.normalize_vertices_scale(scaled_keypoints) * vertices_scale
        return scaled_vertices, scaled_pc, scaled_keypoints
        
    def apply_random_scaling_obj(self, tot_verts, tot_pcs, tot_keypts):
        ### use the vertices as the bbox, scale and extend infos ### ## vertices scaling
        tot_obj_verts =  np.concatenate(tot_verts, axis=0) ### tot_obj_verts: n_tot_verts x 3 
        scale_normalizing_factors = np.random.uniform(low=0.75, high=1.25, size=(3,))
        # min + (max - min) * scale_normalizing_factors (normalizing_factors)
        scale_normalizing_factors = np.reshape(scale_normalizing_factors, [1, 3])
        
        #### vertices center of the vertices ####
        vertices_center = utils.get_vertices_center(tot_obj_verts)
        vertices_center = np.reshape(vertices_center, (1, 3))
        
        vertices_scale = utils.get_vertices_scale(tot_obj_verts)
    
    ### relative scaling factors for part vertices before anad after scaling for normalized parts and part scales ###
    
    def apply_random_scaling_obj(self, tot_obj_verts, joint_pvp_info): #### joint_pvp_info: 
      tot_vertices = np.concatenate(tot_obj_verts, axis=0)
      scale_normalizing_factors = np.random.uniform(low=0.75, high=1.25, size=(3,))
      # min + (max - min) * scale_normalizing_factors (normalizing_factors)
      scale_normalizing_factors = np.reshape(scale_normalizing_factors, [1, 3])
      
      vertices_center = utils.get_vertices_center(tot_vertices)
      vertices_center = np.reshape(vertices_center, (1, 3))
      vertices_scale = utils.get_vertices_scale(tot_vertices)
      
      ### (verts - center) * scale_factors / cur_scale * ori_scale
      scaled_part_vertices = []
      scaled_joint_pvp_info = []
      for cur_part_verts in tot_obj_verts:
        cur_s_part_verts = (cur_part_verts - vertices_center) * scale_normalizing_factors
        scaled_part_vertices.append(cur_s_part_verts)
      scaled_joint_pvp_info = (joint_pvp_info - vertices_center) * scale_normalizing_factors
      
      
      tot_scaled_part_vertices = np.concatenate(scaled_part_vertices, axis=0)
      tot_scaled_cur_scale = utils.get_vertices_scale(tot_scaled_part_vertices)
      for i_p, cur_part_verts in enumerate(scaled_part_vertices):
        cur_s_part_verts = cur_part_verts / tot_scaled_cur_scale * vertices_scale
        scaled_part_vertices[i_p] = cur_s_part_verts
      
      scaled_joint_pvp_info = scaled_joint_pvp_info / tot_scaled_cur_scale * vertices_scale ### scaled_pvp_info ###
      return scaled_part_vertices, scaled_joint_pvp_info
    
    def transform_part_normalized_infos(self, tot_verts, tot_pcs, tot_keypts, tot_obj_verts, scaled_part_vertices):
      s_tot_verts = []
      s_tot_pcs = []
      s_tot_keypts = []
      for i_p in range(len(tot_verts)):
        cur_p_verts, cur_p_pcs, cur_p_keypts = tot_verts[i_p], tot_pcs[i_p], tot_keypts[i_p]
        cur_p_obj_verts, cur_p_scaled_obj_verts = tot_obj_verts[i_p], scaled_part_vertices[i_p]
        cur_scaled_p_extents = utils.get_vertices_extends(cur_p_scaled_obj_verts)
        cur_p_extents = utils.get_vertices_extends(cur_p_obj_verts)
        cur_scaled_p_extents = cur_scaled_p_extents / np.clip(cur_p_extents, a_min=1e-7, a_max=9999999.0)
        
        cur_p_ori_scale = utils.get_vertices_scale(cur_p_verts)
        
        cur_p_verts = cur_p_verts * cur_scaled_p_extents
        cur_p_pcs = cur_p_pcs * cur_scaled_p_extents
        cur_p_keypts = cur_p_keypts * cur_scaled_p_extents
        
        cur_p_cur_scale = utils.get_vertices_scale(cur_p_verts)
        
        cur_p_verts = cur_p_verts / cur_p_cur_scale * cur_p_ori_scale
        cur_p_pcs = cur_p_pcs / cur_p_cur_scale * cur_p_ori_scale
        cur_p_keypts = cur_p_keypts / cur_p_cur_scale * cur_p_ori_scale
        
        
        s_tot_verts.append(cur_p_verts)
        s_tot_pcs.append(cur_p_pcs)
        s_tot_keypts.append(cur_p_keypts)
        
      return s_tot_verts, s_tot_pcs, s_tot_keypts
      # 
      
      
    def get_cvx_pts_structure(self, idx):
      
      tot_cvx_to_bbox_center = []
      tot_cvx_to_pts =[]
      for i_p, cur_dst_folder in enumerate(self.dst_folder): 
        cur_dst_model_nm = self.dst_models[i_p][idx] ### model_nm
        cur_dst_cvx_to_verts_fn = os.path.join(cur_dst_folder, cur_dst_model_nm + self.cvx_to_verts_sufix)
        
        if not os.path.exists(cur_dst_cvx_to_verts_fn):
          cur_dst_cvx_to_verts_fn = os.path.join(cur_dst_folder, cur_dst_model_nm + "_manifold_cvx_to_verts.npy")
          
        cur_dst_cvx_to_verts = np.load(cur_dst_cvx_to_verts_fn, allow_pickle=True).item()
        
        cvx_to_pts = {}
        cur_cvx_to_bbox_center = {}
        for cvx in cur_dst_cvx_to_verts:
          cur_cvx_verts = cur_dst_cvx_to_verts[cvx] ### cvx to verts dict
          cur_cvx_verts = torch.from_numpy(cur_cvx_verts).float()
          cvx_to_pts[cvx] = cur_cvx_verts
          cvx_minn, _ = torch.min(cur_cvx_verts, dim=0, keepdim=True)
          cvx_maxx, _ = torch.max(cur_cvx_verts, dim=0, keepdim=True) ### 
          cvx_center = (cvx_minn + cvx_maxx) * 0.5
          cvx_enxtents = torch.sum((cvx_maxx - cvx_minn) ** 2, dim=-1, keepdim=True)
          cvx_enxtents = torch.sqrt(cvx_enxtents) ### 1 x 1
          cur_cvx_to_bbox_center[cvx] = (cvx_center, cvx_enxtents)
        tot_cvx_to_bbox_center.append(cur_cvx_to_bbox_center) #### for i-th part
        tot_cvx_to_pts.append(cvx_to_pts)
      return tot_cvx_to_bbox_center, tot_cvx_to_pts
    
    
    def __getitem__(self, idx): ## assembling parameters...
        # idxes = [random.choice(range(cur_src_n_models * cur_dst_n_models)) for cur_src_n_models, cur_dst_n_models in zip(self.src_n_models, self.dst_n_models)]
        
        # idx = int(time.time()) % len(self.tot_obj_inst_folders)
        # 
        # idx = 
        # dst_idx = idx
        
        #### source idxes ####
        # src_idxes = [random.choice(range(cur_src_n_models)) for cur_src_n_models in self.src_n_models]
        
        # src_idxes = [idx for _ in range(len(self.src_n_models))]
        
        # if self.use_paired_data:
        ###### src_idxes for each part ###
        ### src idxes ###
        
        # src_idxes = [idx if idx < self.src_n_models[i_p] else random.choice(range(self.src_n_models[i_p]))  for i_p in range(len(self.src_n_models)) ]
        
        # else:
        #   # src_idxes = [idx for _ in range(len(self.src_n_models))]
        #   src_idxes = [random.choice(range(cur_src_n_models)) for cur_src_n_models in self.src_n_models]
        
        
        # print(f"idx: {idx}, obj_folder_nm: {self.tot_obj_inst_folders[idx]}, src_model: {self.src_models[0][idx]}, dst_model: {self.dst_models[0][idx]}") 
        ## tot_obj_inst_folders --> obj_inst_folders ##
        ### tot_obj_inst_folders --> obj_inst_idx ###
        cur_obj_inst_folder = self.tot_obj_inst_folders[idx]
        cur_obj_inst_folder = os.path.join(self.obj_data_root_folder, cur_obj_inst_folder) ### cur_obj_inst_folder for the isntinformations
        ##### part-nm-to-extend #####
        
        
        
        # motion_dataset_attr_fn = "motion_attributes.json"
        
        
        # if os.path.exists(os.path.join(cur_obj_inst_folder, motion_dataset_attr_fn)):
        
        
        # part_nm_to_scale_center, part_nm_to_pvp_dir, part_nm_to_extend, part_nm_to_verts, part_nm_to_joint_info = load_scale_position_joint_information_from_folder(cur_obj_inst_folder) 
        # part_order = ["dof_rootd_Aa001_r", "dof_rootd_Aa002_r", "none_motion"]
        
        if "SAPIEN" in self.obj_data_root_folder:
          part_nm_to_scale_center, part_nm_to_pvp_dir, part_nm_to_extend, part_nm_to_verts, part_nm_to_joint_info = load_scale_position_joint_information_from_folder_partnet_mobility(cur_obj_inst_folder) 
          if "Oven" in self.obj_data_root_folder:
            tot_obj_fns = os.listdir(cur_obj_inst_folder)
            tot_obj_fns = [fn for fn in tot_obj_fns if fn.endswith(".obj")]
            tot_obj_idxes = [int(fn.split(".")[0].split("_")[-1]) for fn in tot_obj_fns]
            tot_obj_idxes = sorted(tot_obj_idxes)
            part_order = [f"link_{tot_obj_idxes[0]}", f"link_{tot_obj_idxes[-1]}"]
          elif "Trash" in self.obj_data_root_folder:
            sorted_part_nms = sorted(list(part_nm_to_scale_center.keys()))
            part_order = sorted_part_nms[-2:]
            print(f"here, part_order: {part_order}")
            
            # print(par)
          else:
            if self.opt.n_parts == 4:
              part_order = ["link_0", "link_1", "link_2", "link_3"]
            elif self.opt.n_parts == 3:
              part_order = ["link_0", "link_1", "link_2"]
            elif self.opt.n_parts == 2:
              # part_order = ["link_1", "link_2"]
              
              part_order = ["link_0", "link_1"]
            else:
              part_order = ["link_2"]
        elif "Shape2Motion" in self.obj_data_root_folder or "MotionDataset" in self.obj_data_root_folder:
          ### part_nm_to_scale_center: to 
          part_nm_to_scale_center, part_nm_to_pvp_dir, part_nm_to_extend, part_nm_to_verts, part_nm_to_joint_info = load_scale_position_joint_information_from_folder(cur_obj_inst_folder) 
          # part_order = ["link_0", "link_1", "link_2"]
          # /data/datasets/genn/Shape2Motion_Deform_key_256/eyeglasses/dof_rootd_Aa001_r
          # part_order = ["link_1", "link_2"] for 
          part_order = ["dof_rootd_Aa001_r", "none_motion"]
        else:
          raise ValueError("Unrecognized dataset type...")
          
        
        
        ## part_nm_to_joint_info --> part_nm_to_joint_info... 
        print(f"part_nm_to_joint_info: {part_nm_to_joint_info}")

        
        part_scale_info = []
        part_position_info = []
        part_extend_info = []
        
        joint_pvp_info = []
        joint_dir_info = []
        
        part_obj_verts_info = []
        
        part_joint_infos = []
        
        for i_p, part_nm in enumerate(part_order):
          cur_part_center, cur_part_scale = part_nm_to_scale_center[part_nm]
          #### part_nm, center, part_scale of t he current part... ####
          # print(f"i_p: {i_p}, part_nm: {part_nm}, cur_part_center: {cur_part_center}, cur_part_scale: {cur_part_scale}")
          part_position_info.append(np.reshape(cur_part_center, (1, 3)))
          part_scale_info.append(cur_part_scale)
          ### part_scale_info, scaling_info ###
          
          cur_part_extend = part_nm_to_extend[part_nm]
          part_extend_info.append(np.reshape(cur_part_extend, (1, 3))) #### part_extend_info
          
          part_obj_verts_info.append(part_nm_to_verts[part_nm])


          if not part_nm in part_nm_to_joint_info:
            cur_part_joint_info = {
              "type": "none_motion", 
              "axis": {
                "a": 0., "b": 1., "dir": np.array([0., 0., 1.], dtype=np.float32), "center": np.zeros((3,), dtype=np.float32)
              }
            }
            part_nm_to_joint_info[part_nm] = cur_part_joint_info
          part_joint_infos.append(part_nm_to_joint_info[part_nm])
        
        
        ### cur part pvp, part dir ###
        for i_p, part_nm in enumerate(part_order[:-1]):
          if part_nm not in part_nm_to_pvp_dir:
            joint_pvp_info.append(np.zeros((1, 3), dtype=np.float32))
            joint_dir_info.append(np.zeros((1, 3), dtype=np.float32))
          else:
            cur_part_pvp, cur_part_dir = part_nm_to_pvp_dir[part_nm]
            joint_pvp_info.append(np.reshape(cur_part_pvp, (1, 3)))
            joint_dir_info.append(np.reshape(cur_part_dir, (1, 3)))
        joint_pvp_info = np.concatenate(joint_pvp_info, axis=0)
        joint_dir_info = np.concatenate(joint_dir_info, axis=0)
        part_extend_info = np.concatenate(part_extend_info, axis=0)
        part_scale_info = np.array(part_scale_info)
        if len(part_scale_info.shape) == 1:
          part_scale_info = np.reshape(part_scale_info, (part_scale_info.shape[0], 1))
        part_position_info = np.concatenate(part_position_info, axis=0) 
        ### part_position_info ### 


        cur_rt_dict = {}
        cur_rt_dict['tar_scales'] = part_scale_info
        cur_rt_dict['tar_positions'] = part_position_info
        cur_rt_dict['tar_joint_dirs'] = joint_dir_info
        cur_rt_dict['tar_joint_pvps'] = joint_pvp_info
        cur_rt_dict['tar_extend'] = part_extend_info
        
        cur_rt_dict["joint_infos"] = part_joint_infos
        cur_rt_dict["idx"] = idx
        
        
        
        ### rt_dict ###
        # tot_rt_dict.append(
        #   cur_rt_dict
        # )
        ### rt_dict ###
        return cur_rt_dict
