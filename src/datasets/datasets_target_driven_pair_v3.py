from ctypes import util
from mimetypes import init
import torch
import numpy as np
import os
import random
import utils
from common_utils import dataset_utils

### collect data from the dataset ###

# n_parts = 2
def my_collate(batch):

    n_parts = len(batch[0]["src_ver"])
    src_pc = [[] for _ in range(n_parts)]
    src_name = [[] for _ in range(n_parts)]
    tar_pc = [[ ] for _ in range(n_parts)]
    tar_name = [[ ] for _ in range(n_parts)]
    key_pts = [[ ] for _ in range(n_parts)]
    w_pc = [[ ] for _ in range(n_parts)]
    src_ver = [[ ] for _ in range(n_parts)]
    src_face = [[ ] for _ in range(n_parts)]
    tar_ver = [[ ] for _ in range(n_parts)]
    tar_face = [[ ] for _ in range(n_parts)]
    real_ver = [[ ] for _ in range(n_parts)]
    real_face = [[ ] for _ in range(n_parts)]
    w_mesh = [[ ] for _ in range(n_parts)]
    src_edges = [[ ] for _ in range(n_parts)]
    src_dofs = [[ ] for _ in range(n_parts)]
    tar_edges = [[ ] for _ in range(n_parts)]
    tar_dofs = [[ ] for _ in range(n_parts)]
    # tar_pc = [[ ] for _ in range(n_parts)]
    # tar_pc = [[ ] for _ in range(n_parts)]
    tar_scales =  []
    tar_positions = []
    tar_joint_dirs = []
    tar_joint_pvps = []
    tar_extend = [] ### tar_extend
    
    
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
        
        for i_p in range(n_parts):
        
          src_pc[i_p].append(torch.from_numpy(data["src_pc"][i_p]).unsqueeze(0).float())
          src_name[i_p].append(data["src_name"][i_p])
          
          tar_pc[i_p].append(torch.from_numpy(data["tar_pc"][i_p]).unsqueeze(0).float())
          tar_name[i_p].append(data["tar_name"][i_p])
          
          key_pts[i_p].append(torch.from_numpy(data["key_pts"][i_p]).unsqueeze(0).float())
          w_pc[i_p].append(torch.from_numpy(data["w_pc"][i_p]).unsqueeze(0).float())
          
          src_ver[i_p].append(torch.from_numpy(data["src_ver"][i_p]).float().cuda())
          
          src_face[i_p].append(torch.from_numpy(data["src_face"][i_p]))
          
          tar_ver[i_p].append(torch.from_numpy(data["tar_ver"][i_p]).float().cuda())
          
          tar_face[i_p].append(torch.from_numpy(data["tar_face"][i_p]))
          real_ver[i_p].append(torch.from_numpy(data["real_ver"][i_p]))
          real_face[i_p].append(torch.from_numpy(data["real_face"][i_p]))
          w_mesh[i_p].append(torch.from_numpy(data["w_mesh"][i_p]).float())
          
          src_edges[i_p].append(torch.from_numpy(data['src_edges'][i_p]).long().unsqueeze(0).cuda()) #### src_edges
          src_dofs[i_p].append(torch.from_numpy(data['src_dofs'][i_p]).float().unsqueeze(0).cuda()) #### vert_dofs 
          
          tar_edges[i_p].append(torch.from_numpy(data['tar_edges'][i_p]).long().unsqueeze(0).cuda()) #### src_edges
          tar_dofs[i_p].append(torch.from_numpy(data['tar_dofs'][i_p]).float().unsqueeze(0).cuda()) #### vert_dofs 
        
        
        
  
    ### obj-level infos ###
    tar_scales = torch.cat(tar_scales, dim=0).cuda()
    tar_positions = torch.cat(tar_positions, dim=0).cuda()
    tar_joint_dirs = torch.cat(tar_joint_dirs, dim=0).cuda()
    tar_joint_pvps = torch.cat(tar_joint_pvps, dim=0).cuda()
    tar_extend = torch.cat(tar_extend, dim=0).cuda() ### tar_extend ### tar_extend ###
    
    # print(f"tar_scales: {tar_scales.size()}, tar_positions: {tar_positions.size()}, tar_joint_dirs: {tar_joint_dirs.size()}, tar_joint_pvps: {tar_joint_pvps.size()}")

    for i_p in range(n_parts):
      src_pc[i_p] = torch.cat(src_pc[i_p]).cuda()
      tar_pc[i_p] = torch.cat(tar_pc[i_p]).cuda()
      key_pts[i_p] = torch.cat(key_pts[i_p]).cuda()
      w_pc[i_p] = torch.cat(w_pc[i_p]).cuda()


    return {"src_pc": src_pc, "src_name": src_name, "key_pts": key_pts,
            "tar_pc": tar_pc, "tar_name": tar_name, "w_pc": w_pc,
            "src_ver": src_ver, "src_face": src_face, "w_mesh": w_mesh,
            "tar_ver": tar_ver, "tar_face": tar_face,
            "real_ver": real_ver, "real_face": real_face,
            "src_edges": src_edges, "src_dofs": src_dofs,
            "tar_edges": tar_edges, "tar_dofs": tar_dofs,
            "tar_scales": tar_scales, "tar_positions": tar_positions, 
            "tar_joint_dirs": tar_joint_dirs, "tar_joint_pvps": tar_joint_pvps, "tar_extend": tar_extend, "joint_infos": tot_part_joint_infos
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
  cur_part_nm = motion_attr_dict['dof_name']
  cur_part_pvp = motion_attr_dict['center']
  cur_part_dir = motion_attr_dict['direction'] ### axis; center, dir
  cur_part_motion_type = motion_attr_dict["motion_type"]
  cur_part_motion_type = {"rotation": "revolute", "translation": "prismatic"}[cur_part_motion_type]
  
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
      part_nm_to_joint = load_motion_infos(cur_child_motion_dict, part_nm_to_joint)
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
    def __init__(self, phase="train", data_dir="../data/chair", opt=None):
        super().__init__()
        self.data_dir = data_dir
        self.phase = phase
        self.opt = opt
        self.load_meta = self.opt.load_meta
        self.num_sampled_pts = 4096
        self.use_paired_data = self.opt.use_paired_data
        self.random_scaling = self.opt.random_scaling
        
        self.n_keypoints = self.opt.n_keypoints
        
        
        self.n_parts = self.opt.n_parts
        self.obj_data_root_folder = self.opt.obj_data_root_folder
        print(f"Object data root folder: {self.obj_data_root_folder} with n_parts: {self.n_parts}") ### 
        
        
        # src_folder = os.path.join(self.data_dir, "src")
        
        #### if we can use any shape as the source shape for deformation? ####
        # if not self.use_paired_data: ### not use paired data...
        #     src_folder = os.path.join(self.data_dir, "dst")
        # else:
        #     src_folder = os.path.join(self.data_dir, "src")
        
        self.data_dir = data_dir ### a list of data_dirs for all parts
        
        self.src_folder_nm = self.opt.src_folder_nm
        self.dst_folder_nm = self.opt.dst_folder_nm
        
        if ";" in self.src_folder_nm:
          self.src_folder_nm = self.src_folder_nm.split(";")
        else:
          self.src_folder_nm = [self.src_folder_nm for _ in range(self.n_parts)]
          
        
        
        # src_folder = [os.path.join(cur_data_dir, "src") for cur_data_dir in self.data_dir] #### src folders
        # dst_folder = [os.path.join(cur_data_dir, "dst") for cur_data_dir in self.data_dir] #### dst folders 
        
        src_folder = [os.path.join(cur_data_dir, cur_src_folder_nm) for cur_data_dir, cur_src_folder_nm in zip(self.data_dir, self.src_folder_nm)]
        dst_folder = [os.path.join(cur_data_dir, self.dst_folder_nm) for cur_data_dir in self.data_dir]
        
        src_folder = dst_folder
        
        # src_folder = os.path.join(self.data_dir, "src")
        # dst_folder = os.path.join(self.data_dir, "dst") ### folder_nm
        
        ### src folders and dst folders ###
        self.src_folder = src_folder
        self.dst_folder = dst_folder
        
        
        
        # with open(os.path.join(self.data_dir, "all.txt")) as f:
        #     lines = f.readlines()
        #     self.models = [line.rstrip() for line in lines]
        
        ### all src models ###
        self.src_models = []
        for cur_src_folder in self.src_folder:
          with open(os.path.join(cur_src_folder, "all.txt"), "r") as rf:
            lines = rf.readlines()
            cur_src_models = [line.rstrip() for line in lines]
            self.src_models.append(cur_src_models)
        
        self.dst_models = []
        minn_dst_nn = 99999
        minn_dst_models_list = []
        for cur_dst_folder in self.dst_folder:
          with open(os.path.join(cur_dst_folder, "all.txt"), "r") as rf:
            lines = rf.readlines()
            cur_dst_models = [line.rstrip() for line in lines]
            self.dst_models.append(cur_dst_models) ### current dst_models 
            if len(cur_dst_models) < minn_dst_nn:
              minn_dst_nn = len(cur_dst_models)
              minn_dst_models_list = cur_dst_models
        self.dst_models = [minn_dst_models_list for _ in range(len(self.dst_models))]
        
        ### inst_0, 
        tot_insts = [int(inss_str.split("_")[-1]) for inss_str in self.dst_models[0]] ### inst idxes for intances
        tot_obj_inst_folders = os.listdir(self.obj_data_root_folder)
        tot_obj_inst_folders = [fn for fn in tot_obj_inst_folders if os.path.isdir(os.path.join(self.obj_data_root_folder, fn))] ### get obj file folders
        
        
        tot_obj_inst_folders = sorted(tot_obj_inst_folders)
        tot_obj_inst_folders = [tot_obj_inst_folders[ii] for ii in tot_insts] ### selected obj inst folders
        
        self.tot_obj_inst_folders = tot_obj_inst_folders
        
        
        if self.n_parts == 4:
          self.part_order = ["drawer0", "drawer1", "drawer2", "cabinet_frame3"]
        elif self.n_parts == 3:
          self.part_order = ["link_0", "link_1", "link_2"]
        elif self.n_parts == 5:
          self.part_order = ["link_0", "link_1", "link_2", "link_3", "link_4"]
        else:
          self.part_order = ["drawer0", "cabinet_frame3"]
        
        valid_inst_idxes = []
        for i_inst, cur_inst_idx in enumerate(tot_insts):
          cur_inst_obj_inst_folder = self.tot_obj_inst_folders[i_inst]
          cur_inst_obj_inst_folder = os.path.join(self.obj_data_root_folder, cur_inst_obj_inst_folder)
          part_nm_to_scale_center, part_nm_to_pvp_dir, part_nm_to_extend, part_nm_to_verts, part_nm_to_joint_info = load_scale_position_joint_information_from_folder_partnet_mobility(cur_inst_obj_inst_folder) 
          
          valid = True
          for part_nm in self.part_order[:-1]:
            if part_nm not in part_nm_to_pvp_dir:
              valid = False
              break
          if valid:
            valid_inst_idxes.append(i_inst)
        
        self.src_models = [[self.src_models[i_p][ii] for ii in valid_inst_idxes] for i_p in range(len(self.src_models))]
        self.dst_models = [[self.dst_models[i_p][ii] for ii in valid_inst_idxes] for i_p in range(len(self.dst_models))]
        self.tot_obj_inst_folders = [self.tot_obj_inst_folders[ii] for ii in valid_inst_idxes]

            
        # with open(os.path.join(self.src_folder, "all.txt"), "r") as rf:
        #   lines = rf.readlines()
        #   self.src_models = [line.rstrip() for line in lines]
        
        # with open(os.path.join(self.dst_folder, "all.txt"), "r") as rf:
        #   lines = rf.readlines()
        #   self.dst_models = [line.rstrip() for line in lines]
        
        # print(f"Meta-infos loaded with src_models: {len(self.src_models)}, dst_models: {len(self.dst_models)}")
        
        # tot_n_models = len(self.models)
        # 
        
        # self.src_n_models = len(self.src_models)
        # self.dst_n_models = len(self.dst_models)
        
        #### src_n_models ####
        self.src_n_models = [len(cur_src_models) for cur_src_models in self.src_models]
        #### dst_n_models ####
        self.dst_n_models = [len(cur_dst_models) for cur_dst_models in self.dst_models]

        # print(f"src_n_models: {self.src_n_models}, dst_n_models: {self.dst_n_models}")
        ### keypoint 256: gt_step_0_subd_0_manifold_tetra.mesh__sf.keypoints_256 
        ### sampled 4096: gt_step_0_subd_0_manifold_tetra.mesh__sf.sampled_4096
        ### surface points: gt_step_0_subd_0_manifold_tetra.mesh__sf
        ### weights sampled 4096 -> 256: gt_step_0_subd_0_manifold_tetra.mesh__sf.keypoints_256.weights.sampled_4096
        ### weights tot pts -> 256: gt_step_0_subd_0_manifold_tetra.mesh__sf.keypoints_256.weights


        self.src_pc = [[None for _ in range(cur_src_n_models)] for cur_src_n_models in self.src_n_models]
        self.src_key_pts = [[None for _ in range(cur_src_n_models)] for cur_src_n_models in self.src_n_models]
        self.src_mesh_vertices = [[None for _ in range(cur_src_n_models)] for cur_src_n_models in self.src_n_models]
        self.src_mesh_faces = [[None for _ in range(cur_src_n_models)] for cur_src_n_models in self.src_n_models]
        self.src_w_pc = [[None for _ in range(cur_src_n_models)] for cur_src_n_models in self.src_n_models]
        self.src_w_mesh = [[None for _ in range(cur_src_n_models)] for cur_src_n_models in self.src_n_models]
        
        self.dst_pc = [[None for _ in range(cur_dst_n_models)] for cur_dst_n_models in self.dst_n_models]
        self.dst_key_pts = [[None for _ in range(cur_dst_n_models)] for cur_dst_n_models in self.dst_n_models]
        self.dst_mesh_vertices = [[None for _ in range(cur_dst_n_models)] for cur_dst_n_models in self.dst_n_models]
        self.dst_mesh_faces = [[None for _ in range(cur_dst_n_models)] for cur_dst_n_models in self.dst_n_models]
        self.dst_w_pc = [[None for _ in range(cur_dst_n_models)] for cur_dst_n_models in self.dst_n_models]
        self.dst_w_mesh = [[None for _ in range(cur_dst_n_models)] for cur_dst_n_models in self.dst_n_models]
        
        
        
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
        return self.dst_n_models[0]
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
    
    
    def __getitem__(self, idx): ## assembling parameters...
        # idxes = [random.choice(range(cur_src_n_models * cur_dst_n_models)) for cur_src_n_models, cur_dst_n_models in zip(self.src_n_models, self.dst_n_models)]
        dst_idx = idx
        
        #### source idxes ####
        # src_idxes = [random.choice(range(cur_src_n_models)) for cur_src_n_models in self.src_n_models]
        
        # src_idxes = [idx for _ in range(len(self.src_n_models))]
        
        if self.use_paired_data:
          ###### src_idxes for each part ###
          src_idxes = [idx if idx < self.src_n_models[i_p] else random.choice(range(self.src_n_models[i_p]))  for i_p in range(len(self.src_n_models)) ]
        else:
          # src_idxes = [idx for _ in range(len(self.src_n_models))]
          src_idxes = [random.choice(range(cur_src_n_models)) for cur_src_n_models in self.src_n_models]
        
        
        # print(f"idx: {idx}, obj_folder_nm: {self.tot_obj_inst_folders[idx]}, src_model: {self.src_models[0][idx]}, dst_model: {self.dst_models[0][idx]}")
        cur_obj_inst_folder = self.tot_obj_inst_folders[idx]
        cur_obj_inst_folder = os.path.join(self.obj_data_root_folder, cur_obj_inst_folder) ### cur_obj_inst_folder for the isntinformations
        ##### part-nm-to-extend #####
        
        
        
        motion_dataset_attr_fn = "motion_attributes.json"
        if os.path.exists(os.path.join(cur_obj_inst_folder, motion_dataset_attr_fn)):
          part_nm_to_scale_center, part_nm_to_pvp_dir, part_nm_to_extend, part_nm_to_verts, part_nm_to_joint_info = load_scale_position_joint_information_from_folder(cur_obj_inst_folder) 
          part_order = ["dof_rootd_Aa001_r", "none_motion"]
        else: ### part_nm_to_scale_center... ###
          # print(f"Loading... for idx: {idx}")
          part_nm_to_scale_center, part_nm_to_pvp_dir, part_nm_to_extend, part_nm_to_verts, part_nm_to_joint_info = load_scale_position_joint_information_from_folder_partnet_mobility(cur_obj_inst_folder) 
          if self.n_parts == 4:
            part_order = ["drawer0", "drawer1", "drawer2", "cabinet_frame3"]
          elif self.n_parts == 3:
            part_order = ["link_0", "link_1", "link_2"]
          elif self.n_parts == 5:
            part_order = ["link_0", "link_1", "link_2", "link_3", "link_4"]
          else:
            part_order = ["drawer0", "cabinet_frame3"]
          # part_order = ["drawer0", "cabinet_frame3"]
          # part_order = ["drawer0", "drawer1", "drawer2", "cabinet_frame3"]
        
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
          
    #       axis_info = { "a": limit_a, "b": limit_b, "dir": cur_part_joint_dir, "center": cur_part_joint_center }
    # cur_part_joint_info = {
    #   "type": cur_part_motion_type, "axis": axis_info
    # }
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
        
        # print(part_position_info.shape, joint_pvp_info.shape, joint_dir_info.shape)
        
        
        
        # tot_rt_dict = []
        
        # cur_rt_dict = {"src_name": src_name, "tar_name": tar_name,
        #         "src_pc": src_pc, "tar_pc": tar_pc,
        #         "src_ver": src_vertices, "src_face": src_faces,
        #         "tar_ver": dst_vertices, "tar_face": dst_faces,
        #         "real_ver": dst_vertices, "real_face": dst_faces,
        #         "key_pts": src_key_pts, "w_mesh": src_w_mesh, "w_pc": src_w_pc,
        #         "src_edges": src_edges, "src_dofs": src_dofs,
        #         "tar_edges": tar_edges, "tar_dofs": tar_dofs,
        #         }
        
        ### and other strategy to enforce the constraint ###
        
        tot_src_name, tot_tar_name, tot_src_pc, tot_tar_pc, tot_src_ver, tot_src_face, tot_tar_ver, tot_tar_face, tot_real_ver, tot_real_face, tot_key_pts, tot_w_mesh, tot_w_pc, tot_src_edges, tot_src_dofs, tot_tar_edges, tot_tar_dofs = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        
        tot_tar_keypts = []
      
        # idx = random.choice(range(self.src_n_models * self.dst_n_models))
        
        for i_p, (idxx, cur_src_n_models, cur_dst_n_models) in enumerate(zip(src_idxes, self.src_n_models, self.dst_n_models)):
          # src_idx = idxx // cur_dst_n_models
          # dst_idx = idxx % cur_dst_n_models
          
          src_idx = src_idxes[i_p]
          dst_idx = dst_idx
          
          # print(f"i_p: {i_p}, idx: {idxx}, cur_src_n_models: {cur_src_n_models}, cur_dst_n_models: {cur_dst_n_models}, src_idx:{src_idx}, dst_idx: {dst_idx}")
          
          ### src_pc is None ###
          if self.src_pc[i_p][src_idx] is None:
            src_pc, src_key_pts, src_vertices, src_faces, src_w_pc, src_w_mesh = self.load_data_from_model_indicator(self.src_folder[i_p], self.src_models[i_p][src_idx])
            self.src_pc[i_p][src_idx] = src_pc
            self.src_key_pts[i_p][src_idx] = src_key_pts
            self.src_mesh_vertices[i_p][src_idx] = src_vertices
            self.src_mesh_faces[i_p][src_idx] = src_faces
            self.src_w_pc[i_p][src_idx] = src_w_pc
            self.src_w_mesh[i_p][src_idx] = src_w_mesh
          while not (self.src_w_pc[i_p][src_idx].shape[1] == self.n_keypoints and self.src_w_pc[i_p][src_idx].shape[0] == self.num_sampled_pts):
            src_idx = random.choice(range(len(self.src_pc[i_p]))) #### src_pc ---> 
            if self.src_pc[i_p][src_idx] is None:
              src_pc, src_key_pts, src_vertices, src_faces, src_w_pc, src_w_mesh = self.load_data_from_model_indicator(self.src_folder[i_p], self.src_models[i_p][src_idx])
              self.src_pc[i_p][src_idx] = src_pc
              self.src_key_pts[i_p][src_idx] = src_key_pts
              self.src_mesh_vertices[i_p][src_idx] = src_vertices
              self.src_mesh_faces[i_p][src_idx] = src_faces
              self.src_w_pc[i_p][src_idx] = src_w_pc
              self.src_w_mesh[i_p][src_idx] = src_w_mesh
              
          if self.dst_pc[i_p][dst_idx] is None:
            tar_pc, dst_key_pts, dst_vertices, dst_faces, dst_w_pc, dst_w_mesh = self.load_data_from_model_indicator(self.dst_folder[i_p], self.dst_models[i_p][dst_idx])
            self.dst_pc[i_p][dst_idx] = tar_pc
            self.dst_key_pts[i_p][dst_idx] = dst_key_pts
            self.dst_mesh_vertices[i_p][dst_idx] = dst_vertices
            self.dst_mesh_faces[i_p][dst_idx] = dst_faces
            self.dst_w_pc[i_p][dst_idx] = dst_w_pc
            self.dst_w_mesh[i_p][dst_idx] = dst_w_mesh

          src_pc, src_key_pts, src_vertices, src_faces, src_w_pc, src_w_mesh = self.src_pc[i_p][src_idx], self.src_key_pts[i_p][src_idx], self.src_mesh_vertices[i_p][src_idx], self.src_mesh_faces[i_p][src_idx], self.src_w_pc[i_p][src_idx], self.src_w_mesh[i_p][src_idx]
          
          tar_pc, dst_key_pts, dst_vertices, dst_faces, dst_w_pc, dst_w_mesh = self.dst_pc[i_p][dst_idx], self.dst_key_pts[i_p][dst_idx], self.dst_mesh_vertices[i_p][dst_idx], self.dst_mesh_faces[i_p][dst_idx], self.dst_w_pc[i_p][dst_idx], self.dst_w_mesh[i_p][dst_idx]
          
          src_edges, src_dofs = dataset_utils.get_edges_from_faces(src_vertices, src_faces)
          tar_edges, tar_dofs = dataset_utils.get_edges_from_faces(dst_vertices, dst_faces) ### get edges from faces...
          
          # if self.random_scaling:
          #     dst_vertices, tar_pc, dst_key_pts = self.apply_random_scaling(dst_vertices, tar_pc, dst_key_pts)
        
          src_name = f"src_{src_idx}"
          tar_name = f"dst_{dst_idx}"
          
          tot_src_name.append(src_name)
          tot_tar_name.append(tar_name)
          tot_src_pc.append(src_pc)
          tot_tar_pc.append(tar_pc)
          tot_src_ver.append(src_vertices)
          tot_src_face.append(src_faces)
          tot_tar_ver.append(dst_vertices)
          tot_tar_face.append(dst_faces)
          tot_real_ver.append(dst_vertices)
          tot_real_face.append(dst_faces)
          tot_key_pts.append(src_key_pts)
          tot_w_mesh.append(src_w_mesh)
          tot_w_pc.append(src_w_pc)
          tot_src_edges.append(src_edges)
          tot_src_dofs.append(src_dofs)
          tot_tar_edges.append(tar_edges)
          tot_tar_dofs.append(tar_dofs)
          
          tot_tar_keypts.append(dst_key_pts)
        
        if self.random_scaling:
          part_scale_info = []
          part_position_info = []
          part_extend_info  = []
          s_part_vertices, joint_pvp_info = self.apply_random_scaling_obj( part_obj_verts_info, joint_pvp_info)
          tot_tar_ver, tot_tar_pc, tot_tar_keypts = self.transform_part_normalized_infos(tot_tar_ver, tot_tar_pc, tot_tar_keypts, part_obj_verts_info, s_part_vertices)
          
          for i_p, cur_part_s_verts in enumerate(s_part_vertices):
            cur_part_center = utils.get_vertices_center(cur_part_s_verts) ### cur_part_verts ### vertices  center
            cur_part_scale = utils.get_vertices_scale(cur_part_s_verts) ### cur_part_verts
            ### cur-part_verts
            cur_part_verts_minn = cur_part_s_verts.min(axis=0)
            cur_part_verts_maxx = cur_part_s_verts.max(axis=0)
            cur_part_extend = cur_part_verts_maxx - cur_part_verts_minn
            ### the diagonal length ###
            part_position_info.append(np.reshape(cur_part_center, (1, 3)))
            part_scale_info.append(cur_part_scale)
            ### part_scale_info, scaling_info ###
            
            cur_part_extend = part_nm_to_extend[part_nm]
            part_extend_info.append(np.reshape(cur_part_extend, (1, 3))) #### part_extend_info
            
          part_extend_info = np.concatenate(part_extend_info, axis=0)
          part_scale_info = np.array(part_scale_info)
          if len(part_scale_info.shape) == 1:
            part_scale_info = np.reshape(part_scale_info, (part_scale_info.shape[0], 1))
          part_position_info = np.concatenate(part_position_info, axis=0) 
          
        cur_rt_dict = {"src_name": tot_src_name, "tar_name": tot_tar_name,
              "src_pc": tot_src_pc, "tar_pc": tot_tar_pc,
              "src_ver": tot_src_ver, "src_face": tot_src_face,
              "tar_ver": tot_tar_ver, "tar_face": tot_tar_face,
              "real_ver": tot_tar_ver, "real_face": tot_tar_face,
              "key_pts": tot_key_pts, "w_mesh": tot_w_mesh, "w_pc": tot_w_pc,
              "src_edges": tot_src_edges, "src_dofs": tot_src_dofs,
              "tar_edges": tot_tar_edges, "tar_dofs": tot_tar_dofs,
              }

        cur_rt_dict['tar_scales'] = part_scale_info
        cur_rt_dict['tar_positions'] = part_position_info
        cur_rt_dict['tar_joint_dirs'] = joint_dir_info
        cur_rt_dict['tar_joint_pvps'] = joint_pvp_info
        cur_rt_dict['tar_extend'] = part_extend_info
        
        cur_rt_dict["joint_infos"] = part_joint_infos
        
        
        
        ### rt_dict ###
        # tot_rt_dict.append(
        #   cur_rt_dict
        # )
        ### rt_dict ###
        return cur_rt_dict
