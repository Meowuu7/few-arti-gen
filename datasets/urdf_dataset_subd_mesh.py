import enum
import os

import os.path
import json
from tkinter import E
import numpy as np
import math
import sys
import torch

from torch.utils import data
# from utils.data_utils_torch import dequantize_verts
from utils.constants import ENDING_XYZ, PART_SEP_GRID_VALUE, MASK_GRID_VALIE
# import data_utils_torch as data_utils
import utils.data_utils_torch  as data_utils
import utils.dataset_utils as dataset_utils

from options.options import opt


# design the

class URDFDataset(data.Dataset):
    def __init__(self, root_folder, quantization_bits=8, nn_max_vertices=8900, nn_max_faces=8900,
                 nn_max_permite_vertices=400, nn_max_permite_faces=2000, category_name="eyeglasses", instance_nns=[],
                 part_names=None, category_nm_to_part_nm=None, is_training=False, padding_to_same=False, split=None, 
                 ):
        super(URDFDataset, self).__init__()

        ##### urdfj
        # self.root_folder = root_folder
        # self.part_names = part_names # part_names # part_names
        # self.statistic_folder = statistic_folder
        self.category_name = category_name  # category # category_name
        self.quantization_bits = quantization_bits
        self.nn_max_vertices = nn_max_vertices + 1
        self.nn_max_faces = nn_max_faces
        self.instance_nns = instance_nns
        self.is_training = is_training
        self.mode = opt.common.exp_mode
        self.apply_random_flipping = opt.dataset.apply_random_flipping
        self.random_scaling = opt.dataset.apply_random_scaling
        self.ar_subd_idx = opt.dataset.ar_subd_idx 



        self.apply_random_shift = opt.dataset.apply_random_shift
        self.category_part_indicator = opt.dataset.category_part_indicator
        self.max_num_grids = opt.vertex_model.max_num_grids
        self.grid_size = opt.vertex_model.grid_size
        self.split = split

        self.use_part = opt.dataset.use_part

        self.st_subd_idx = opt.dataset.st_subd_idx
        print(f"Start subdivision index: {self.st_subd_idx}")

        self.subdn = opt.dataset.subdn
        # debug = opt.model.debug
        # datasest_name = opt.dataset.dataset_name

        self.nn_vertices_predict_ratio = opt.dataset.nn_vertices_predict_ratio
        self.cut_vertices = opt.model.cut_vertices
        self.context_window = opt.model.context_window
        self.remove_du = not opt.dataset.not_remove_du
        self.data_type = opt.dataset.data_type
        self.load_meta = opt.dataset.load_meta
        self.use_inst = opt.dataset.use_inst
        self.balance_classes = opt.dataset.balance_classes

        #### Whether to use context window as the limit of number of grids ####
        self.use_context_window_as_max = opt.dataset.use_context_window_as_max
        self.nn_context_part = 1

        category_part_indicator_to_idxes = {}

        print(f"Use context window as max: {self.use_context_window_as_max}")
        if self.use_context_window_as_max:
          self.max_num_grids = self.context_window

        # class 
        # context window 
        # 

        self.num_parts = opt.dataset.num_parts
        self.num_objects = opt.dataset.num_objects

        # context_design_strategy = opt.model.context_design_strategy

        # samples_list = os.listdir(self.root_folder)
        # samples_list = [fn for fn in samples_list if os.path.isdir(os.path.join(self.root_folder, fn))]

        # print(f"number of samples: {len(samples_list)}")

        # if datasest_name in ["MotionDataset", "PolyGen_Samples"]:
        #     mesh_dicts, part_tree = dataset_utils.get_mesh_dict_list(self.root_folder, samples_list,
        #                                                              valid_indices=self.instance_nns)

        if not isinstance(category_name, list):
            category_name = [category_name]  # category name

        mesh_dicts = []
        mesh_dict_idx_to_category_name = {}
        category_name_to_mesh_dict_idxes = {}
        category_part_indicator_to_mesh_dict_idxes = {}
        for cur_cat_nm in category_name: # category na
            ###### root folder of the current category ######
            cur_root_folder = os.path.join(opt.dataset.dataset_root_path, opt.dataset.dataset_name, cur_cat_nm)

            print(f"Loading part objs from folder {cur_root_folder}...") # load part objs # load part objs
            if category_nm_to_part_nm is None:
                if cur_cat_nm == "eyeglasses":
                    part_names = ["none_motion"]
                else:
                    part_names = ["dof_rootd_Aa001_r", "dof_rootd_Aa002_r"]
            else:
                part_names = category_nm_to_part_nm[cur_cat_nm]

            read_meta_info_func = dataset_utils.get_mesh_dict_part_subd_mesh_meta_info
            cur_cat_mesh_dicts, part_tree, part_nm_to_mesh_dict_idxes = read_meta_info_func(
              cur_root_folder, part_names=part_names,
              ret_part_name_to_mesh_idx=True, remove_du=self.remove_du, use_inst=self.use_inst, split=self.split)
            print(f"Pretraining dataset, start loading data for the category {cur_cat_nm} with parts {part_names}. Split: {self.split}. ")

            # if self.data_type == 'binvox':
            #     if not self.load_meta: ##### load_meta
            #         cur_cat_mesh_dicts, part_tree, part_nm_to_mesh_dict_idxes = dataset_utils.get_mesh_dict_list_multi_part_vox(
            #             cur_root_folder, part_names=part_names,
            #             ret_part_name_to_mesh_idx=True, remove_du=self.remove_du)
            #     else:
            #       # get_mesh_dict_list_obj_vox_meta_info
            #         # cur_cat_mesh_dicts, part_tree, part_nm_to_mesh_dict_idxes = dataset_utils.get_mesh_dict_list_multi_part_vox_meta_info(
            #         #     cur_root_folder, part_names=part_names,
            #         #     ret_part_name_to_mesh_idx=True, remove_du=self.remove_du, use_inst=self.use_inst)
            #         # use_part
            #         read_meta_info_func = dataset_utils.get_mesh_dict_list_obj_vox_meta_info
            #         if self.use_part:
            #           read_meta_info_func = dataset_utils.get_mesh_dict_part_vox_meta_info
            #         cur_cat_mesh_dicts, part_tree, part_nm_to_mesh_dict_idxes = read_meta_info_func(
            #             cur_root_folder, part_names=part_names,
            #             ret_part_name_to_mesh_idx=True, remove_du=self.remove_du, use_inst=self.use_inst, split=self.split)
            # else:
            #   #### mesh as voxes
            #     cur_cat_mesh_dicts, part_tree, part_nm_to_mesh_dict_idxes = dataset_utils.get_mesh_dict_list_multi_part_mesh_vertices(
            #         cur_root_folder, part_names=part_names,
            #         ret_part_name_to_mesh_idx=True, remove_du=self.remove_du)
            print(f"Dataset loaded with {len(cur_cat_mesh_dicts)} instances for category {cur_cat_nm} with parts {part_names}. Split: {self.split}.") # categories and parts

            # get_mesh_dict_list_multi_part_mesh_vertices

            # the second part generation sequence
            # relative positional embedding 
            # sample context window from the whole training dataset
            # 

            cur_mesh_dict_len = len(mesh_dicts)
            cur_cat_nn = 0
            for cur_cat_cur_part_nm in part_nm_to_mesh_dict_idxes: # part_nm to mesh_dict_idxes # part_nm_to_mesh_dict_idxes
                new_mesh_dict_idxes = []
                for iii in part_nm_to_mesh_dict_idxes[cur_cat_cur_part_nm]:
                    new_mesh_dict_idxes.append(iii + cur_mesh_dict_len)
                    category_part_nm_indicator = f"{cur_cat_nm}-{cur_cat_cur_part_nm}"
                    mesh_dict_idx_to_category_name[iii + cur_mesh_dict_len] = category_part_nm_indicator
                    # cur_cat_nn += 1
                    if category_part_nm_indicator not in category_part_indicator_to_idxes:
                        category_part_indicator_to_idxes[category_part_nm_indicator] = len(
                            category_part_indicator_to_idxes)
                    # mesh_dicts.append(cur_cat_mesh_dicts[iii]) # it should be iii...
                category_part_indicator_to_mesh_dict_idxes[category_part_nm_indicator] = new_mesh_dict_idxes
                part_nm_to_mesh_dict_idxes[cur_cat_cur_part_nm] = new_mesh_dict_idxes
                # part_nm_to_mesh

            category_name_to_mesh_dict_idxes[cur_cat_nm] = part_nm_to_mesh_dict_idxes
            # cur_category_mesh_dict_idxes = range(cur_mesh_dict_len, cur_mesh_dict_len + len(cur_cat_mesh_dicts))
            # mesh_dicts += cur_cat_mesh_dicts

            # cur_sample_vertices, nn_face_indices = cur_cat_mesh_dicts['vertices'], cur_cat_mesh_dicts['faces']
            # if not (cur_sample_vertices.shape[0] <= nn_max_permite_vertices and nn_face_indices <= nn_max_permite_faces):
            #     continue
            mesh_dicts += cur_cat_mesh_dicts
        
        # category_name_to_mesh_dict_idxes --> mesh dict idxes
        self.category_name_to_mesh_dict_idxes = category_name_to_mesh_dict_idxes
        self.mesh_dict_idx_to_category_name = mesh_dict_idx_to_category_name
        self.category_part_indicator_to_idxes = category_part_indicator_to_idxes
        self.category_part_indicator_to_mesh_dict_idxes = category_part_indicator_to_mesh_dict_idxes

        if self.balance_classes:
            self.category_name_to_mesh_dict_idxes, self.balanced_mesh_dicts = dataset_utils.balance_class_idxes(self.category_part_indicator_to_mesh_dict_idxes)

        print(f"number of valid samples: {len(mesh_dicts)}")

        tot_n_mesh = len(mesh_dicts) if not self.balance_classes else len(self.balanced_mesh_dicts) #### balanced mesh dicts
        self.mesh_dicts = mesh_dicts
        self.mesh_idx_to_part_pts_idxes = {}

        self.dataset_len = tot_n_mesh

    def __len__(self):
        return self.dataset_len


    def augment_vertices_scale(self, vertices):
        scale_normalizing_factors = np.random.uniform(low=0.75, high=1.25, size=(3,))
        # min + (max - min) * scale_normalizing_factors (normalizing_factors)
        scale_normalizing_factors = np.reshape(scale_normalizing_factors, [1, 3])
        minn_verts = np.min(vertices, axis=0, keepdims=True)
        vertices = minn_verts + (vertices - minn_verts) * scale_normalizing_factors # scale 
        # vertices = vertices * scale_normalizing_factors
        return vertices

    def apply_random_scaling(self, vertices):
        dequan_verts = data_utils.dequantize_verts(vertices, n_bits=self.quantization_bits)
        normalized_verts = dequan_verts + 0.5
        
        # ar
        normalized_verts = self.augment_vertices_scale(normalized_verts)
        normalized_verts = data_utils.center_vertices(normalized_verts)
        normalized_verts = data_utils.normalize_vertices_scale(normalized_verts)
        scaled_verts = data_utils.quantize_verts(normalized_verts, self.quantization_bits)

        # normalized_verts = normalized_verts + 0.5

        # sigma = 0.25
        # x_warp_gradient_sampled = np.random.normal(loc=0.0, scale=math.sqrt(sigma), size=(5,))
        # y_warp_gradient_sampled = np.random.normal(loc=0.0, scale=math.sqrt(sigma), size=(5,))
        # z_warp_gradient_sampled = np.random.normal(loc=0.0, scale=math.sqrt(sigma), size=(5,))

        # x_warp_gradient_sampled = np.exp(x_warp_gradient_sampled)
        # y_warp_gradient_sampled = np.exp(y_warp_gradient_sampled)
        # z_warp_gradient_sampled = np.exp(z_warp_gradient_sampled)

        # # x_reflect = True
        # x_reflect = False
        # y_reflect = True
        # z_reflect = True

        # scaled_verts = []
        # for i_v in range(normalized_verts.shape[0]):
        #     cur_v_xyz = normalized_verts[i_v].tolist()
        #     cur_x, cur_y, cur_z = float(cur_v_xyz[0]), float(cur_v_xyz[1]), float(cur_v_xyz[2])
        #     scaled_x = data_utils.warp_coord(x_warp_gradient_sampled, cur_x, reflect=x_reflect)
        #     scaled_y = data_utils.warp_coord(y_warp_gradient_sampled, cur_y, reflect=y_reflect)
        #     scaled_z = data_utils.warp_coord(z_warp_gradient_sampled, cur_z, reflect=z_reflect)
        #     scaled_verts.append([scaled_x, scaled_y, scaled_z])
        # scaled_verts = np.array(scaled_verts, dtype=np.float)  # n x 3
        # scaled_verts = data_utils.center_vertices(scaled_verts)
        # # normalize_vertices_scale --
        # scaled_verts = data_utils.normalize_vertices_scale(scaled_verts)
        # scaled_verts = data_utils.quantize_verts(scaled_verts, self.quantization_bits)
        return scaled_verts

    def convert_to_grid_xyz(self, vertices, pts_idxes=None):
      # vertices: n_verts
      # n_max_grids x (grid_size x grid_size) --> grid points information
      # n_max_grids x 3 --> grid coordinates information
      # grid_size of the vertex_model
      #### meta info and ####
      grid_size = opt.vertex_model.grid_size

      if pts_idxes is None:
        pts_idxes = [np.arange(0, vertices.shape[0])]
      
      all_grid_xyzs = []
      all_grid_pts = []

      if not self.is_training:
        pts_idxes = pts_idxes[2:]
      
      for i_part, cur_part_pts_idxes in enumerate(pts_idxes):
        if not self.is_training and i_part >= self.nn_context_part:
          break
        cur_part_vertices = vertices[cur_part_pts_idxes]
        self.grid_xyz_to_points = {}
        for i_v in range(cur_part_vertices.size(0)): # vertices
          cur_vert_xyz = cur_part_vertices[i_v].long().tolist()
          cur_x, cur_y, cur_z = cur_vert_xyz # vert xyz
          cur_grid_x, cur_grid_y, cur_grid_z = cur_x // grid_size, cur_y // grid_size, cur_z // grid_size
          cur_x, cur_y, cur_z = cur_x % grid_size, cur_y % grid_size, cur_z % grid_size
          if (cur_grid_x, cur_grid_y, cur_grid_z) not in self.grid_xyz_to_points:
            cur_grid_pts = torch.zeros((grid_size, grid_size, grid_size), dtype=torch.long)
            cur_grid_pts[cur_x, cur_y, cur_z] = 1
            self.grid_xyz_to_points[(cur_grid_x, cur_grid_y, cur_grid_z)] = cur_grid_pts
          else:
            self.grid_xyz_to_points[(cur_grid_x, cur_grid_y, cur_grid_z)][cur_x, cur_y, cur_z] = 1
        tot_grid_pts = []
        tot_grid_xyzs = []
        sorted_grid_points = sorted(self.grid_xyz_to_points.items(), key=lambda ii: ii[0], reverse=False)
        # tot_grid_pts: nn_grids x (grid_pts)
        for cur_item in sorted_grid_points:
          tot_grid_pts.append(cur_item[1].unsqueeze(0))
          tot_grid_xyzs.append(cur_item[0]) # sorted xyzs
        tot_grid_xyzs = torch.tensor(tot_grid_xyzs, dtype=torch.long) # n_grids x 3 --> grid xyzs
        tot_grid_pts = torch.cat(tot_grid_pts, dim=0) # n_grids x (grid_size x grid_size x grid_size)

        all_grid_xyzs.append(tot_grid_xyzs)
        all_grid_pts.append(tot_grid_pts)
      tot_grid_pts = torch.cat(all_grid_pts, dim=0)
      tot_grid_xyzs = torch.cat(all_grid_xyzs, dim=0)
      # if not self.is_training:
      #   tot_grid_pts = tot_grid_pts[:10] # first 10 pts
      #   tot_grid_xyzs = tot_grid_xyzs[:10] # first 10 xyzs
      return tot_grid_pts, tot_grid_xyzs

    def convert_to_grid_xyz_vox_list(self, vertices, pts_idxes=None):
      # vertices: n_verts
      # n_max_grids x (grid_size x grid_size) --> grid points information
      # n_max_grids x 3 --> grid coordinates information
      # grid_size of the vertex_model
      grid_size = opt.vertex_model.grid_size

      if pts_idxes is None:
        ##### get total vertices idxes #####
        pts_idxes = [np.arange(0, cur_vertices.shape[0]) for cur_vertices in vertices]
        # pts_idxes = [np.arange(0, vertices.shape[0]) ]
      
      all_grid_xyzs = []
      all_grid_pts = []
      all_grid_content = []

      sep_part_grid_xyz = torch.tensor(ENDING_XYZ, dtype=torch.long).unsqueeze(0) ##### 1 x 3
      sep_part_grid_content = torch.tensor([PART_SEP_GRID_VALUE], dtype=torch.long)

      # if not self.is_training:
      #   pts_idxes = pts_idxes[2:]
      
      for i_part, (cur_obj_vertices, cur_part_pts_idxes) in enumerate(zip(vertices, pts_idxes)):
        if not self.is_training and i_part >= self.nn_context_part:
          break
        # cur_part_vertices = cur_part_vertices[cur_part_pts_idxes]
        cur_part_vertices = cur_obj_vertices[cur_part_pts_idxes]
        self.grid_xyz_to_points = {}
        for i_v in range(cur_part_vertices.size(0)): # vertices
          cur_vert_xyz = cur_part_vertices[i_v].long().tolist()
          cur_x, cur_y, cur_z = cur_vert_xyz # vert xyz
          cur_grid_x, cur_grid_y, cur_grid_z = cur_x // grid_size, cur_y // grid_size, cur_z // grid_size
          cur_x, cur_y, cur_z = cur_x % grid_size, cur_y % grid_size, cur_z % grid_size
          if (cur_grid_x, cur_grid_y, cur_grid_z) not in self.grid_xyz_to_points:
            cur_grid_pts = torch.zeros((grid_size, grid_size, grid_size), dtype=torch.long)
            cur_grid_pts[cur_x, cur_y, cur_z] = 1
            self.grid_xyz_to_points[(cur_grid_x, cur_grid_y, cur_grid_z)] = cur_grid_pts
          else:
            self.grid_xyz_to_points[(cur_grid_x, cur_grid_y, cur_grid_z)][cur_x, cur_y, cur_z] = 1
        tot_grid_pts = []
        tot_grid_xyzs = []
        sorted_grid_points = sorted(self.grid_xyz_to_points.items(), key=lambda ii: ii[0], reverse=False)
        # tot_grid_pts: nn_grids x (grid_pts) # grid_pts
        for cur_item in sorted_grid_points:
          tot_grid_pts.append(cur_item[1].unsqueeze(0))
          tot_grid_xyzs.append(cur_item[0]) # sorted xyzs
        tot_grid_xyzs = torch.tensor(tot_grid_xyzs, dtype=torch.long) # n_grids x 3 --> grid xyzs
        tot_grid_pts = torch.cat(tot_grid_pts, dim=0) # n_grids x (grid_size x grid_size x grid_size)

        tot_grid_content = self.convert_grid_values_to_grid_content(tot_grid_pts) ### n_grids
        tot_grid_content = torch.cat([tot_grid_content, sep_part_grid_content], dim=0) ### n_grids + 1
        
        tot_grid_xyzs = torch.cat([tot_grid_xyzs, sep_part_grid_xyz], dim=0) ##### n_grids x 3

        all_grid_xyzs.append(tot_grid_xyzs)
        all_grid_pts.append(tot_grid_pts)
        all_grid_content.append(tot_grid_content) # n_grids + 1
      tot_grid_pts = torch.cat(all_grid_pts, dim=0)
      tot_grid_xyzs = torch.cat(all_grid_xyzs, dim=0)
      tot_grid_content = torch.cat(all_grid_content, dim=0)
      # if not self.is_training:
      #   tot_grid_pts = tot_grid_pts[:10]
      #   tot_grid_xyzs = tot_grid_xyzs[:10]
      return tot_grid_pts, tot_grid_xyzs, tot_grid_content


    def convert_grid_values_to_grid_content(self, grid_values):
        # grid_values: nn_grids x gs x gs x gs # 0/1 values
        grid_content = grid_values.contiguous().view(grid_values.size(0), -1).contiguous() # nn_grids x (gs x gs x gs)
        # mult_factors = reversed([2 ** i for i in range(self.grid_size ** 3)])
        mult_factors = [2 ** i for i in range(self.grid_size ** 3 - 1, -1, -1)]

        # 
        # mult_factors = np.array(mult_factors, dtype=np.long)
        # print(f"mult_factors: {mult_factors}")
        mult_factors = torch.tensor(mult_factors, dtype=torch.long).long()
        # grid_content = np.sum(grid_content * np.reshape(mult_factors, (1, self.grid_size ** 3)), axis=-1) # nn_grids
        # grid_content = grid_content.astype(np.long)
        grid_content = torch.sum(grid_content * mult_factors.unsqueeze(0), dim=-1).long()
        return grid_content
    
    def sample_bfs_component(self, selected_vert, faces):
      vert_idx_to_adj_verts = {}
      for i_f, cur_f in enumerate(faces):
        # for i0, v0 in enumerate(cur_f):
        for i0 in range(len(cur_f)):
          v0 = cur_f[i0] - 1
          i1 = (i0 + 1) % len(cur_f)
          v1 = cur_f[i1] - 1
          if v0 not in vert_idx_to_adj_verts:
            vert_idx_to_adj_verts[v0] = {v1: 1}
          else:
            vert_idx_to_adj_verts[v0][v1] = 1
          if v1 not in vert_idx_to_adj_verts:
            vert_idx_to_adj_verts[v1] = {v0: 1}
          else:
            vert_idx_to_adj_verts[v1][v0] = 1
      vert_idx_to_visited = {} # whether visisted here # 
      vis_que = [selected_vert]
      vert_idx_to_visited[selected_vert] = 1
      visited = [selected_vert]
      while len(vis_que) > 0 and len(visited) < self.max_num_grids:
        cur_frnt_vert = vis_que[0]
        vis_que.pop(0)
        if cur_frnt_vert in vert_idx_to_adj_verts:
          cur_frnt_vert_adjs = vert_idx_to_adj_verts[cur_frnt_vert]
          for adj_vert in cur_frnt_vert_adjs:
            if adj_vert not in vert_idx_to_visited:
              vert_idx_to_visited[adj_vert] = 1
              vis_que.append(adj_vert)
              visited.append(adj_vert)
      if len(visited) >= self.max_num_grids:
        visited = visited[: self.max_num_grids - 1]
      return visited

    def select_faces_via_verts(self, selected_verts, faces):
      if not isinstance(selected_verts, list):
        selected_verts = selected_verts.tolist()
      # selected_verts_dict = {ii: 1 for ii in selected_verts}
      old_idx_to_new_idx = {v + 1: ii + 1 for ii, v in enumerate(selected_verts)} ####### v + 1: ii + 1 --> for the selected_verts
      new_faces = []
      for i_f, cur_f in enumerate(faces):
        cur_new_f = []
        valid = True
        for cur_v in cur_f:
          if cur_v in old_idx_to_new_idx:
            cur_new_f.append(old_idx_to_new_idx[cur_v])
          else:
            valid  = False
            break
        if valid:
          new_faces.append(cur_new_f)
      return new_faces 
      
    ##### sample connected componets #####
    def sample_connected_component(self, vertices,  faces, prev_verts, prev_faces, nn_verts=None):
      if nn_verts is None:
        nn_verts = self.max_num_grids
      tot_nn_verts = vertices.shape[0] # tot_nn_verts
      ##### vert_idx #####
      selected_vert_idx = np.random.randint(tot_nn_verts, size=(1,), dtype=np.long).item() # an int number
      # faces
      selected_bfs_verts = self.sample_bfs_component(selected_vert_idx, faces)
      selected_bfs_verts = sorted(selected_bfs_verts, reverse=False) # reverse
      selected_prev_bfs_verts = [vv for vv in selected_bfs_verts if  vv < prev_verts.shape[0]]
      # selected_prev_bfs_verts

      selected_bfs_verts = np.array(selected_bfs_verts, dtype=np.long)
      selected_bfs_faces = self.select_faces_via_verts(selected_bfs_verts, faces) # ##### selecte faces via verts #####
      
      selected_prev_bfs_verts = np.array(selected_prev_bfs_verts, dtype=np.long)
      selected_bfs_prev_faces = self.select_faces_via_verts(selected_prev_bfs_verts, prev_faces)
      
      # selected_vertices = vertices[selected_bfs_verts]
      # selected_prev_vertices = prev_verts[selected_prev_bfs_v]
      
      
      return selected_bfs_verts, selected_bfs_faces, selected_prev_bfs_verts, selected_bfs_prev_faces

    def recenter_vertices(self, verts_gt, verts_upsample):
      # dequan_verts_gt, dequan_verts_upsample = data_utils.dequantize_verts(verts_gt, n_bits=self.quantization_bits), data_utils.dequantize_verts(verts_upsample, n_bits=self.quantization_bits)

      dequan_verts_gt = verts_gt
      dequan_verts_upsample = verts_upsample
      center_dequan_verts_gt = data_utils.get_vertices_center(dequan_verts_gt)

      dequan_verts_gt = dequan_verts_gt - center_dequan_verts_gt
      dequan_verts_upsample = dequan_verts_upsample - center_dequan_verts_gt

      verts_gt = dequan_verts_gt
      verts_upsample = dequan_verts_upsample

      # verts_gt = data_utils.quantize_verts(dequan_verts_gt, n_bits=self.quantization_bits)
      # verts_upsample = data_utils.quantize_verts(dequan_verts_upsample, n_bits=self.quantization_bits)
      return verts_gt, verts_upsample

    # TODO: preprocess: 1) remove duplicated vertices; 2) reorder vertices; 3) quantize vertices
    def permute_vertices(self, verts):
      verts_permuted = torch.cat(
        [verts[..., 2].unsqueeze(-1), verts[..., 1].unsqueeze(-1), verts[..., 0].unsqueeze(-1)], dim=-1
      )
      return verts_permuted

    ### 

    def get_edges_from_faces(self, faces):
      # face_list
      if not isinstance(faces, list):
        faces = faces.tolist()
      edges_to_exist = {}
      edge_fr, edge_to = [], []
      for i_f, cur_f in enumerate(faces):
        for i0, v0 in enumerate(cur_f):
          i1 = (i0 + 1) % len(cur_f)
          v1 = cur_f[i1]
          edge_pair = (v0, v1) if v0 < v1 else (v1, v0)
          if edge_pair not in edges_to_exist:
            edges_to_exist[edge_pair] = 1
            edge_fr += [v0, v1]
            edge_to += [v1, v0]
      edges = torch.tensor([edge_fr, edge_to], dtype=torch.long) # edge_fr, edge_to; 2 x nn_edges
      return edges

    def compute_dof_from_edges(self, vertices, edges):
      vertices_dof = torch.zeros((vertices.shape[0], ), dtype=torch.long)
      for i_edge in range(edges.size(1)):
        cur_edge = edges[:, i_edge].tolist()
        idx_fr, idx_to = cur_edge
        vertices_dof[idx_fr - 1] += 1
      return vertices_dof

    ''' Load data with random axises flipping and random vertices shifting  '''
    def __getitem__(self, item):
        ###### num objects ######
        ###### num_objects ######
        # if not self.is_training:
        #   item = 0
        if self.balance_classes:
            item = self.balanced_mesh_dicts[item] # get new mesh dict index of the current item

        items = [item]
        # if self.num_objects > 1 and self.is_training: ##### num_objects and training flag #####
        #   cur_category_part_indicator = self.mesh_dict_idx_to_category_name[item]
        #   cur_category_part_other_instances = self.category_part_indicator_to_mesh_dict_idxes[cur_category_part_indicator]
        #   ###### num_objects ######
        #   selected_other_instances = np.random.choice(len(cur_category_part_other_instances), size=(self.num_objects - 1,), replace=True).tolist()
        #   selected_other_instances = [cur_category_part_other_instances[ii] for ii in selected_other_instances]


        for cur_item in items:
          if not isinstance(self.mesh_dicts[cur_item], dict):
            # self.mesh_dicts[item] = dataset_utils.read_binvox_to_pts(self.mesh_dicts[item])
            # mesh_list = self.mesh_dicts[cur_item]
            # print(f"Read mesh dicts from {self.mesh_dicts[cur_item]}.")
            self.mesh_dicts[cur_item] = dataset_utils.read_subd_meshes(self.mesh_dicts[cur_item])
            # self.mesh_dicts[cur_item], pts_idxes  = dataset_utils.read_binvox_list_to_pts(self.mesh_dicts[cur_item])
            # self.mesh_idx_to_part_pts_idxes[cur_item] = pts_idxes # read binvox
          # else:
          if not isinstance(self.mesh_dicts[0], dict):
            self.mesh_dicts[0] = dataset_utils.read_subd_meshes(self.mesh_dicts[0])
              
          #     pts_idxes = self.mesh_idx_to_part_pts_idxes[cur_item]

        # category_part_nm indicat  or... # indicator
        category_part_nm_indicator = self.mesh_dict_idx_to_category_name[item]
        category_part_idx = self.category_part_indicator_to_idxes[category_part_nm_indicator]

        cur_item_mesh_dict = self.mesh_dicts[item]
        # grid xyz coordinates and grid contents

        # subd_0_upsample, subd_1_gt
        # subd_1_upsample, subd_2_gt
        # faces (the last subdivision)
        # selecte a vertex; a connected component; and others
        # 'vertices': vertices, 'upsampled_vertices': up_verts, 'faces': faces
        #### map from upsampled_vertices to nex_subd_gt vertices (conditioned on currrent vertices); current vertices
        rt_dict = {}
        for i_subd in range(self.subdn - 1): # subd_upsample
          cur_subd_upsample = cur_item_mesh_dict[i_subd]['upsampled_vertices'] # upsample vertices and faces # expanding process ---> for the connected component
          nex_subd_gt = cur_item_mesh_dict[i_subd + 1]['vertices'] # for the current subd
          cur_subd = cur_item_mesh_dict[i_subd]['vertices']
          cur_subd_faces = cur_item_mesh_dict[i_subd]['faces']
          #### nex_subd_faces --> nex_subd_faces
          nex_subd_faces = cur_item_mesh_dict[i_subd + 1]['faces']
          cur_subd_half_flaps = cur_item_mesh_dict[i_subd]['half_flaps']
          cur_subd_half_flaps_mid_points_idx = cur_item_mesh_dict[i_subd]['half_flaps_mid_points_idx']
          cur_subd_fake_upsample = cur_item_mesh_dict[i_subd]['fake_upsampled_vertices']
          cur_subd_fake_upsample_faces = cur_item_mesh_dict[i_subd]['fake_upsampled_faces']


          # nex_subd_edges = 

          # if i_subd == 0:
          #   cur_subd_upsample = self.mesh_dicts[0][i_subd]['upsampled_vertices']
          #   cur_subd = self.mesh_dicts[0][i_subd]['vertices'] # #### 0-th mesh's vertices
          #   cur_subd_faces = self.mesh_dicts[0][i_subd]['faces'] # #### 0-th mesh's faces

          if cur_subd_upsample.shape[0] >= self.max_num_grids and self.is_training: ##### selected_prev_bfs_verts: 
            selected_bfs_verts, selected_bfs_faces, selected_prev_bfs_verts, selected_bfs_prev_faces = self.sample_connected_component(nex_subd_gt,  nex_subd_faces, cur_subd, cur_subd_faces, nn_verts=None)
            cur_subd_upsample = cur_subd_upsample[selected_bfs_verts]
            nex_subd_gt = nex_subd_gt[selected_bfs_verts] ##### bfs_verts
            cur_subd = cur_subd[selected_prev_bfs_verts]
            nex_subd_gt, cur_subd_upsample = self.recenter_vertices(nex_subd_gt, cur_subd_upsample)
            cur_subd, _ = self.recenter_vertices(cur_subd, cur_subd)
            
            cur_subd_faces = selected_bfs_prev_faces
            nex_subd_faces = selected_bfs_faces

            rt_dict[f'subd_{i_subd}_upsample_pts_indices'] = selected_bfs_verts

          ######## cur_subd_upsample ########
          cur_subd_upsample = torch.from_numpy(cur_subd_upsample).float() # .long() # upsampled vertices
          nex_subd_gt = torch.from_numpy(nex_subd_gt).float() # .long()
          cur_subd = torch.from_numpy(cur_subd).float() # .long()
          cur_subd_faces = torch.tensor(cur_subd_faces, dtype=torch.long) # .long()
          nex_subd_faces = torch.tensor(nex_subd_faces, dtype=torch.long)
          cur_subd_fake_upsample = torch.from_numpy(cur_subd_fake_upsample).float()
          cur_subd_fake_upsample_faces = torch.tensor(cur_subd_fake_upsample_faces, dtype=torch.long)

          cur_subd_edges = self.get_edges_from_faces(cur_subd_faces) # 2 x n_edges
          nex_subd_edges = self.get_edges_from_faces(nex_subd_faces)

          cur_subd_half_flaps = torch.from_numpy(cur_subd_half_flaps).long()
          cur_subd_half_flaps_mid_points_idx = torch.from_numpy(cur_subd_half_flaps_mid_points_idx).long()

          # print(f"cur instnace subd_upsample: {cur_subd_upsample[:10]}, cur instance nex_subd_gt: {nex_subd_gt[:10]}")

          # minn_verts_nn = min(cur_subd_upsample.size(0), nex_subd_gt.size(0))
          # cur_subd_upsample = cur_subd_upsample[:minn_verts_nn]
          # nex_subd_gt = nex_subd_gt[:minn_verts_nn]

          # print(f"i_subd: {i_subd}, cur_subd_upsample: {cur_subd_upsample.size()}, nex_subd_gt: {nex_subd_gt.size()}")

          cur_subd_upsample = self.permute_vertices(cur_subd_upsample)
          nex_subd_gt = self.permute_vertices(nex_subd_gt) # nex_subd_gt # 
          cur_subd = self.permute_vertices(cur_subd)
          cur_subd_fake_upsample = self.permute_vertices(cur_subd_fake_upsample)
          
          cur_subd_fake_upsample_edges = self.get_edges_from_faces(cur_subd_fake_upsample_faces)

          ###### write upsampled obj and gt-obj ######
          # if i_subd == self.subdn - 2:
          #   upsample_sv_fn = f"item_{item}_last_subd_upsample.obj"
          #   dequan_upsample = data_utils.dequantize_verts(cur_subd_upsample.numpy(), n_bits=self.quantization_bits)
          #   data_utils.write_obj(dequan_upsample, nex_subd_faces.tolist(), upsample_sv_fn, transpose=True, scale=1.)
          #   nex_gt_sv_fn = f"item_{item}_last_subd_gt.obj"
          #   dequan_gt = data_utils.dequantize_verts(nex_subd_gt.numpy(), n_bits=self.quantization_bits)
          #   data_utils.write_obj(dequan_gt, nex_subd_faces.tolist(), nex_gt_sv_fn, transpose=True, scale=1.)
          
          rt_dict[f'subd_{i_subd}_upsample'] = cur_subd_upsample # z, y, x
          rt_dict[f'subd_{i_subd + 1}_gt'] = nex_subd_gt
          rt_dict[f'subd_{i_subd}'] = cur_subd
          rt_dict[f'subd_{i_subd}_faces'] = cur_subd_faces
          rt_dict[f'subd_{i_subd}_edges'] = cur_subd_edges
          rt_dict[f'subd_{i_subd}_upsample_edges'] = nex_subd_edges
          rt_dict[f'subd_{i_subd}_half_flaps'] = cur_subd_half_flaps
          rt_dict[f'subd_{i_subd}_half_flaps_mid_points_idx'] = cur_subd_half_flaps_mid_points_idx
          rt_dict[f'subd_{i_subd}_fake_upsample'] = cur_subd_fake_upsample
          rt_dict[f'subd_{i_subd}_fake_upsample_faces'] = cur_subd_fake_upsample_faces
          rt_dict[f'subd_{i_subd}_fake_upsample_edges'] = cur_subd_fake_upsample_edges
          # rt_dict[f'subd_{}']
          # #### add selected_bfs_verts to the rt_dict ####
          # rt_dict[f'subd_{i_subd}_upsample_pts_indices'] = selected_bfs_verts
          # # # Re-order vertex coordinates as (z, y, x).
          # vertices_permuted = torch.cat(
          #     [vertices[..., 2].unsqueeze(-1), vertices[..., 1].unsqueeze(-1), vertices[..., 0].unsqueeze(-1)], dim=-1
          # )
          if i_subd == self.subdn - 2:
            # nex_subd_faces = cur_item_mesh_dict[i_subd + 1]['faces']
            nex_subd_faces = nex_subd_faces
            # nex_subd_faces = torch.tensor(nex_subd_faces, dtype=torch.long)
            rt_dict['faces'] = nex_subd_faces
            rt_dict[f'subd_{i_subd + 1}_faces'] = nex_subd_faces
            rt_dict[f'subd_{i_subd + 1}'] = nex_subd_gt
            nex_subd_edges = self.get_edges_from_faces(nex_subd_faces)
            rt_dict[f'subd_{i_subd + 1}_edges'] = nex_subd_edges #### edges; embed edges and others with center point indicator and context point indicator

            nex_subd_fake_upsample = cur_item_mesh_dict[i_subd + 1]['fake_upsampled_vertices']
            nex_subd_fake_upsample = torch.from_numpy(nex_subd_fake_upsample).float()
            nex_subd_fake_upsample = self.permute_vertices(nex_subd_fake_upsample)
            rt_dict[f'subd_{i_subd + 1}_fake_upsample'] = nex_subd_fake_upsample ### fake_upsample of the next level

            cur_subd_fake_upsample_faces = cur_item_mesh_dict[i_subd]['fake_upsampled_faces']
            cur_subd_fake_upsample_edges = self.get_edges_from_faces(cur_subd_fake_upsample_faces)
            nex_subd_dof = self.compute_dof_from_edges(cur_subd_fake_upsample, cur_subd_fake_upsample_edges)
            rt_dict[f'subd_{i_subd + 1}_dof'] = nex_subd_dof

            nex_subd_fake_upsample_faces = cur_item_mesh_dict[i_subd + 1]['fake_upsampled_faces']
            nex_subd_fake_upsample_faces = torch.tensor(nex_subd_fake_upsample_faces, dtype=torch.long) # .long()
            rt_dict[f'subd_{i_subd + 1}_fake_upsample_faces'] = nex_subd_fake_upsample_faces ### fake_upsample of the next level

            
            


        ar_mesh_dict = cur_item_mesh_dict['ar']
        ar_vertices, ar_faces, ar_faces_unflatten = ar_mesh_dict['vertices'], ar_mesh_dict['faces'], ar_mesh_dict['faces_unflatten']
        ar_vertices = data_utils.center_vertices(ar_vertices)
        ar_vertices = data_utils.normalize_vertices_scale(ar_vertices) # ### normalize_vertices_scale

        ar_quant_vertices = data_utils.quantize_verts(ar_vertices, n_bits=self.quantization_bits) # n_verts x 3
        ar_quant_vertices = torch.from_numpy(ar_quant_vertices).long()

        # cur_subd_mesh_sv_fn = "ar_subd_mesh.obj"
        # data_utils.write_obj(ar_vertices, None, cur_subd_mesh_sv_fn, transpose=True, scale=1.)
        
        ar_vertices = torch.from_numpy(ar_vertices).float() ## nn_verts x 3

        ar_stop_vertices = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float32).unsqueeze(0)
        ar_vertices = torch.cat(
          [ar_vertices, ar_stop_vertices], dim=0
        ) ## (nn_verts + 1) x 3

        ar_stop_vertices = torch.tensor([0, 0, 0], dtype=torch.long).unsqueeze(0)
        ar_quant_vertices = torch.cat(
          [ar_quant_vertices + 1, ar_stop_vertices], dim=0 #### (nn_verts + 1) x 3
        )

        ar_faces = torch.tensor(ar_faces, dtype=torch.long)


        # vertices_mask = torch.ones((ar_vertices.size(0) - 1, ), dtype=torch.float32) ## nn_verts,
        vertices_mask = torch.cat(
          [torch.ones((ar_vertices.size(0) - 1,), dtype=torch.float32), torch.zeros((1,), dtype=torch.float32)], dim=0
        )

        faces_mask = torch.ones((ar_faces.size(0), ), dtype=torch.float32)
        

        rt_dict['vertices'] = ar_vertices
        # rt_dict['quant_vertices'] = data_utils.
        # rt_dict['vertices_faces'] =     ar_vertices[:-1, :] ##### nn_verts x 3
        rt_dict['vertices_mask'] = vertices_mask ## (nn_verts + 1,) --> vertices_mask
        rt_dict['quant_vertices'] = ar_quant_vertices

        rt_dict['faces'] = ar_faces ## (nn_faces_flatten,)
        rt_dict['faces_unflatten'] = torch.tensor(ar_faces_unflatten, dtype=torch.long) ## nn_faces x 3 --> triangle faces
        rt_dict['faces_mask'] = faces_mask
        

        rt_dict['class_label'] = torch.zeros((1,), dtype=torch.long)
        # print(f"rt_dict_keys: {rt_dict.keys()}")

        # tot_vertices = []
        # tot_pts_idxes = []

        # for cur_item in items:
        #   # make vertices; long vertices # 
        #   cur_item_mesh_dict = self.mesh_dicts[cur_item]
        #   cur_item_pts_idxes = self.mesh_idx_to_part_pts_idxes[cur_item]

        #   if not self.is_training:
            
        #     # all_parts_nn_pts = [ptss.shape[0] for ptss in cur_item_pts_idxes]
        #     # print(f"number of parts: {len(cur_item_pts_idxes)}, all_parts_nn_pts: {all_parts_nn_pts}")
        #     cur_item_pts_idxes = cur_item_pts_idxes[2 if len(cur_item_pts_idxes) >= 3 else 0]

        #     # print(f"noly use pts_idxes of the thrid part with number of points: {cur_item_pts_idxes.shape}")
        #   else:
        #     cur_item_pts_idxes = np.concatenate(cur_item_pts_idxes, axis=0)
          
        #   vertices = torch.from_numpy(cur_item_mesh_dict['vertices']).long() ##### vertices of the current mesh item
        #   if self.random_scaling > 0.5:
        #     ###### scaled vertices and vertices ######
        #     scaled_vertices = self.apply_random_scaling(vertices.numpy())
        #     vertices = torch.from_numpy(scaled_vertices).long()
        #   if self.apply_random_shift:
        #     vertices = data_utils.random_shift(vertices=vertices, shift_factor=0.25)
        #   if  self.apply_random_flipping:
        #     ###### randomly flip xyz-coordiantes / permute xyz coordiantes ######
        #     permutation_idxes = np.random.randint(low=0, high=6, size=(1,)).item()
        #     rgt_idx = permutation_idxes % 3
        #     ##### ramains idxes #####
        #     remains_idxes = [xx for xx in range(3) if xx != rgt_idx]
        #     mid_idx = remains_idxes[permutation_idxes // 3]
        #     lft_idx = [xx for xx in range(3) if xx != rgt_idx and xx != mid_idx][0]
        #     permutation_idxes = [lft_idx, mid_idx, rgt_idx]
        #     vertices = vertices[:, permutation_idxes]

        #   # Re-order vertex coordinates as (z, y, x).
        #   vertices_permuted = torch.cat(
        #       [vertices[..., 2].unsqueeze(-1), vertices[..., 1].unsqueeze(-1), vertices[..., 0].unsqueeze(-1)], dim=-1
        #   )
        #   ##### vertices #####
        #   tot_vertices.append(vertices_permuted)
        #   tot_pts_idxes.append(cur_item_pts_idxes)
        
        # # faces = torch.from_numpy(cur_item_mesh_dict['faces']).long()
        # if not self.category_part_indicator:  # add category part indicator
        #     class_label = torch.tensor([cur_item_mesh_dict['class_label']], dtype=torch.long)
        # else:
        #     class_label = torch.tensor([category_part_idx], dtype=torch.long)

        # ###### vertices_permuted ######
        

        # # grid_size = opt.vertex_model.grid_size

        # ######## tot_vertices, tot_pts_idxes ########
        # # grid_pts: n_grids x grid_size x grid_size x grid_size; grid_xyzs: n_grids x 3
        # grid_pts, grid_xyzs, grid_content = self.convert_to_grid_xyz_vox_list(tot_vertices, tot_pts_idxes) # grid_pts: n_grids x ... ##### pts_idxes!
        # # grid_xyzs_flat = grid_xyzs.contiguous().view(-1).contiguous() # (n_grids x 3, )
        # # grid_pts_flat = grid_pts.contiguous().view(grid_pts.size(0), grid_size ** 3).contiguous() # n_grids x (grid_size ** 3)

        # # grid_content = self.convert_grid_values_to_grid_content(grid_pts) # nn_grids # content!
 
        # grid_pos = torch.arange(start=0, end=grid_content.size(0), step=1, dtype=torch.long) # grid pos

        # nn_grids = grid_content.size(0)
        # # print(f"nn_grids: {nn_grids}, max_num_grids: {self.max_num_grids}, grid_size: {self.grid_size}")

        # # grid_pts_mask = torch.ones((grid_pts.size(0), self.grid_size, self.grid_size, self.grid_size), dtype=torch.float32)

        # # grid_content_mask = torch.ones((grid_content.size(0), ), dtype=torch.float32)
        # # grid_xyzs_mask = torch.ones((grid_xyzs.size(0), 3), dtype=torch.float32)

        
        # if nn_grids >= self.max_num_grids:
        #   cut_grid_st_pos = np.random.randint(low=0, high=nn_grids - self.max_num_grids + 2, size=(1,)).item()
        #   # grid_pts = grid_pts[cut_grid_st_pos: cut_grid_st_pos + self.max_num_grids - 1] # grid_pts
        #   grid_xyzs = grid_xyzs[cut_grid_st_pos: cut_grid_st_pos + self.max_num_grids - 1]
        #   grid_content = grid_content[cut_grid_st_pos: cut_grid_st_pos + self.max_num_grids - 1]
        
        #   ###### whether to use grid_pos ######
        #   if self.use_context_window_as_max:
        #     grid_pos = grid_pos[ :self.max_num_grids - 1] # start from zero
        #   else:
        #     grid_pos = grid_pos[cut_grid_st_pos: cut_grid_st_pos + self.max_num_grids - 1] # max_num_grids
        
        #   # grid_content = grid_content[cut_grid_st_pos: cut_grid_st_pos + self.max_num_grids - 1] # grid_content: xxx

        #   grid_xyzs_mask = torch.ones((grid_xyzs.size(0), 3), dtype=torch.float32)
        #   # grid_pts_mask

        #   # print(f"cutting nn_grids: {nn_grids}, max_num_grids: {self.max_num_grids}, sampled cut_grid_st_pos: {cut_grid_st_pos}")
        #   # no extra mask is needed here ######
        #   if self.is_training:
        #     if cut_grid_st_pos == nn_grids - self.max_num_grids + 1:
        #       grid_xyzs_mask = torch.cat(
        #         [grid_xyzs_mask, torch.tensor([[0, 0, 0]], dtype=torch.float32), torch.zeros((self.max_num_grids - grid_xyzs.size(0) - 1, 3), dtype=torch.float32)], dim=0
        #       )
        #     else:
        #       grid_xyzs_mask = torch.cat(
        #         [grid_xyzs_mask, torch.tensor([[0, 0, 0]], dtype=torch.float32), torch.zeros((self.max_num_grids - grid_xyzs.size(0) - 1, 3), dtype=torch.float32)], dim=0
        #       )
        # else:
        #   grid_xyzs_mask = torch.ones((grid_xyzs.size(0), 3), dtype=torch.float32)
        #   if self.is_training:
        #     grid_xyzs_mask = torch.cat(
        #       [grid_xyzs_mask, torch.tensor([[0, 0, 0]], dtype=torch.float32), torch.zeros((self.max_num_grids - grid_xyzs.size(0) - 1, 3), dtype=torch.float32)], dim=0
        #     )

        # # grid_pts_mask = torch.ones((grid_pts.size(0), self.grid_size, self.grid_size, self.grid_size), dtype=torch.float32)

        # grid_content_mask = torch.ones((grid_content.size(0), ), dtype=torch.float32)
        

        # # grid_xyzs_mask = torch.cat(
        # #   [torch.ones((grid_xyzs.size(0) + 1, 3), dtype=torch.float32), torch.zeros((self.max_num_grids - grid_xyzs.size(0) - 1, 3), dtype=torch.float32)], dim=0
        # # )

        # ##### grid_xyzs_mask #####
        # # grid_xyzs_mask = torch.cat(
        # #   [torch.ones((grid_xyzs.size(0), 3), dtype=torch.float32), torch.tensor([[1, 0, 0]], dtype=torch.float32), torch.zeros((self.max_num_grids - grid_xyzs.size(0) - 1, 3), dtype=torch.float32)], dim=0
        # # )
        # ##### grid_xyzs_mask #####

        # grid_xyzs = grid_xyzs + 1

        # if self.is_training:
        #   # grid_pts_mask = torch.cat(
        #   #   [grid_pts_mask, torch.zeros((self.max_num_grids - grid_xyzs.size(0), self.grid_size, self.grid_size, self.grid_size), dtype=torch.float32)], dim=0
        #   # )

        #   grid_content_mask = torch.cat(
        #       [grid_content_mask, torch.zeros((self.max_num_grids - grid_content.size(0), ), dtype=torch.float32)], dim=0
        #   )

        #   # grid size should < max_num_grids --- should leave one grid position for blank masking
        #   # grid position for dummy grids --- self.max_num_grids
        #   # grid_xyzs: n_grids x 3
        #   grid_xyzs = torch.cat(
        #     [grid_xyzs, torch.zeros((self.max_num_grids - grid_xyzs.size(0), 3), dtype=torch.long)], dim=0
        #   )
        #   # ### grid_pts: self.max_num_grids x gs x gs x gs
        #   # grid_pts = torch.cat(
        #   #   [grid_pts, torch.zeros((self.max_num_grids - grid_pts.size(0), self.grid_size, self.grid_size, self.grid_size), dtype=torch.long)], dim=0
        #   # )

        #   grid_content = torch.cat(
        #     [grid_content, torch.zeros((self.max_num_grids - grid_content.size(0), ), dtype=torch.long) + MASK_GRID_VALIE + 3], dim=0
        #   )

        #   # grid_pos
        #   # grid_pos = torch.cat(
        #   #   [grid_pos, torch.full((self.max_num_grids - grid_pos.size(0),), fill_value=self.max_num_grids, dtype=torch.long)], dim=0
        #   # )

        #   grid_pos = torch.cat(
        #     [grid_pos, torch.zeros((self.max_num_grids - grid_pos.size(0), ), dtype=torch.long)], dim=0
        #   )

        

        

        # # vertices_flat
        # vertices_flat = vertices_permuted.contiguous().view(-1).contiguous()

        # #### ? ####
        # # vertices_flat_masked, vertices_mask_identifier, vertices_mask_permut = self._generate_vertices_mask(vertices_flat)
        # #### ? ####

        # # vertices_mask_permut = torch.argsort(vertices_mask_identifier, dim=0, descending=True) # identifier
        # # 1111 s should be first...

        # nn_vertices = vertices_flat.size(0)

        # # nn_faces = faces.size(0)  # faces

        # ''' minn vertex index value is 1 here, so + 1 can get real vertex indices for faces '''
        # vertices_flat = torch.cat(
        #     [vertices_flat + 1, torch.zeros((self.nn_max_vertices * 3 - vertices_flat.size(0),), dtype=torch.long)],
        #     dim=-1
        # )

        # ''' Permutation '''
        # permut = np.random.permutation(nn_vertices + 1)
        # permut = torch.from_numpy(permut).long() # seq_len
        # res_permut = np.array(range(nn_vertices + 1, self.nn_max_vertices * 3), dtype=np.int32)
        # res_permut = torch.from_numpy(res_permut).long()
        # permut = torch.cat(
        #   [permut, res_permut], dim=0
        # )
        # ''' Permutation '''

        # ''' Permutation '''
        # permut = np.random.permutation(nn_vertices)
        # permut = torch.from_numpy(permut).long() # seq_len
        # res_permut = np.array(range(nn_vertices, self.nn_max_vertices * 3), dtype=np.int32)
        # res_permut = torch.from_numpy(res_permut).long()
        # permut = torch.cat(
        #   [permut, res_permut], dim=0
        # )
        # ''' Permutation '''

        # # dequantized vertices
        # real_nn_vertices = vertices.size(0)

        # nn_vertices_predict = int(real_nn_vertices * self.nn_vertices_predict_ratio)
        # nn_vertices_masked = real_nn_vertices - nn_vertices_predict

        # vertices_flat_mask = torch.cat(
        #     [torch.zeros((nn_vertices_masked * 3, ), dtype=torch.float32),
        #      torch.ones((nn_vertices_predict * 3 + 1,), dtype=torch.float32),
        #      torch.zeros((self.nn_max_vertices * 3 - nn_vertices - 1), dtype=torch.float32)], dim=-1
        # )

        ##### cut vetices to fit in the context window #####
        # if self.context_window > 0 and self.cut_vertices > 0 and real_nn_vertices > self.context_window // 3:
        #     print("cutting vertices")
        #     context_nn_vertices = self.context_window // 3 # get 
        #     # if real_nn_vertices > context_nn_vertices:
        #     context_st_vertices = np.random.randint(low=0, high=real_nn_vertices - context_nn_vertices, size=(1,), dtype=np.int32).item()
        #     cur_context_window = self.context_window
        #     if context_st_vertices == real_nn_vertices - context_nn_vertices - 1:
        #         cur_context_window = cur_context_window + 1
        #     context_vertices_flat = vertices_flat[context_st_vertices * 3: context_st_vertices * 3 + cur_context_window]
        #     context_vertices_flat_mask = vertices_flat_mask[context_st_vertices * 3: context_st_vertices * 3 + cur_context_window]
        #     context_vertices_flat = torch.cat(
        #         [context_vertices_flat, torch.zeros((self.nn_max_vertices * 3 - context_vertices_flat.size(0),), dtype=torch.long)],
        #         dim=-1
        #     )
        #     context_vertices_flat_mask = torch.cat(
        #         # context veritces flat mask...
        #         [context_vertices_flat_mask, torch.zeros((self.nn_max_vertices * 3 - context_vertices_flat_mask.size(0),), dtype=torch.float32)], dim=-1
        #     )
        #     vertices_flat = context_vertices_flat # context vertices flat
        #     vertices_flat_mask = context_vertices_flat_mask
        ##### cut vetices to fit in the context window #####
            

        # vertices position encoding + others is still ok OR 3D convolution


        # start_sampling_idx = np.random.randint(0, real_nn_vertices // 2, (1,))
        # start_sampling_idx = torch.from_numpy(start_sampling_idx).long()

        # start_sampling_idx = np.random.randint(nn_vertices // 4, nn_vertices // 2, (1,))
        # start_sampling_idx = torch.from_numpy(start_sampling_idx).long()

        # if self.mode == "sampling":
        #   # sampling_function =
        #   #### TODO a more genral version, current one is for sampling only ####
        #   # permut, start_sampling_idx, vertices_flat = self.generate_interpolate_mask(vertices=vertices[:vertices.size(0) // 2, :])
        #   ##### middle parts #####
        #   # permut, start_sampling_idx, vertices_flat = self.generate_scaled_verts_mask(vertices=vertices)
        #   # permut, start_sampling_idx, vertices_flat = self.generate_scaled_verts_mask_low_part(vertices=vertices)
        #   # permut, start_sampling_idx, vertices_flat = self.generate_scaled_verts_mask_high_part(vertices=vertices)
        #   permut, start_sampling_idx, vertices_flat = self.generate_scaled_verts_mask_high_part(vertices=vertices[vertices.size(0) // 2:])
        #   permut = torch.cat(
        #     [permut, torch.arange(permut.size(0), self.nn_max_vertices * 3, dtype=torch.long)], dim=0
        #   )
        #   start_sampling_idx = torch.tensor([start_sampling_idx], dtype=torch.long)
        #   vertices_flat = torch.cat(
        #       [vertices_flat, torch.zeros((self.nn_max_vertices * 3 - vertices_flat.size(0),), dtype=torch.long)], dim=-1
        #   )
        #   # vertices_flat_mask = torch.cat(
        #   #     [torch.ones((nn_vertices * 2 + 1,), dtype=torch.float32), torch.zeros((self.nn_max_vertices * 3 - nn_vertices * 2 - 1), dtype=torch.float32)], dim=-1
        #   # )
        #   #### TODO a more genral version, current one is for sampling only ####

        # vertices
        # vertices = torch.cat(  #
        #     [vertices, torch.zeros((self.nn_max_vertices - real_nn_vertices, 3), dtype=torch.long)], dim=0
        # )

        # vertices_mask = torch.cat(  #
        #     [torch.ones((real_nn_vertices,), dtype=torch.float32),
        #      torch.zeros((self.nn_max_vertices - real_nn_vertices,), dtype=torch.float32)], dim=0
        # )

        # # print(f"item: {item}, nn_left_faces: {nn_left_faces}, nn_rgt_faces: {nn_rgt_faces}")
        # faces_mask = torch.cat(
        #     [torch.ones((nn_faces,), dtype=torch.float32),
        #      torch.zeros((self.nn_max_faces - nn_faces), dtype=torch.float32)], dim=-1
        # )

        # # faces and faces mask
        # faces = torch.cat(
        #     [faces, torch.zeros((self.nn_max_faces - nn_faces), dtype=torch.long)], dim=-1
        # )

        # rt_dict = {
        #     # 'vertices_flat_mask': vertices_flat_mask,
        #     # 'vertices_flat': vertices_flat,
        #     # 'faces_mask': faces_mask,
        #     'vertices': vertices,
        #     # 'vertices_mask': vertices_mask,
        #     # 'faces': faces,
        #     'class_label': class_label,
        #     # 'grid_xyzs_flat': grid_xyzs_flat,
        #     'grid_xyzs': grid_xyzs,
        #     'grid_xyzs_mask': grid_xyzs_mask,
        #     # 'grid_pts_flat': grid_pts_flat,
        #     # 'grid_content': grid_pts,
        #     # 'grid_content_mask': grid_pts_mask, ### only need contents...
        #     'grid_pos': grid_pos,
        #     'grid_content_vocab': grid_content,
        #     'grid_content_vocab_mask': grid_content_mask
            
        #     # 'vertices_flat_masked': vertices_flat_masked,
        #     # 'vertices_mask_identifier': vertices_mask_identifier,
        #     # 'vertices_permutation': permut,
        #     # 'start_sampling_idx': start_sampling_idx,
        #     # 'vertices_masked_permutation': vertices_mask_permut
        # }
        rt_dict = rt_dict

        return rt_dict

