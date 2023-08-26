import os

import os.path
import json
from tkinter import E
import numpy as np
import math
import sys
import torch

from torch.utils import data
from utils.constants import ENDING_GRID_VALUE, ENDING_XYZ, PART_SEP_GRID_VALUE, PART_SEP_XYZ
# import data_utils_torch as data_utils
import utils.data_utils_torch  as data_utils
import utils.dataset_utils as dataset_utils

from options.options import opt


# design the

class URDFDataset(data.Dataset):
    def __init__(self, root_folder, quantization_bits=8, nn_max_vertices=8900, nn_max_faces=8900,
                 nn_max_permite_vertices=400, nn_max_permite_faces=2000, category_name="eyeglasses", instance_nns=[],
                 part_names=None, category_nm_to_part_nm=None, is_training=False, split=None
                 ):
        super(URDFDataset, self).__init__()

        self.root_folder = root_folder
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

        self.split = split



        self.apply_random_shift = opt.dataset.apply_random_shift
        self.category_part_indicator = opt.dataset.category_part_indicator
        self.max_num_grids = opt.vertex_model.max_num_grids
        self.grid_size = opt.vertex_model.grid_size

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
        self.use_context_window_as_max = opt.dataset.use_context_window_as_max
        self.nn_context_part = 1


        category_part_indicator_to_idxes = {}

        print(f"Use context window as max: {self.use_context_window_as_max}")
        if self.use_context_window_as_max:
          self.max_num_grids = self.context_window

        # class 
        # context window 
        # 

        # context_design_strategy = opt.model.context_design_strategy

        samples_list = os.listdir(self.root_folder)
        samples_list = [fn for fn in samples_list if os.path.isdir(os.path.join(self.root_folder, fn))]

        print(f"number of samples: {len(samples_list)}")

        # if datasest_name in ["MotionDataset", "PolyGen_Samples"]:
        #     mesh_dicts, part_tree = dataset_utils.get_mesh_dict_list(self.root_folder, samples_list,
        #                                                              valid_indices=self.instance_nns)
        # else:

        if not isinstance(category_name, list):
            category_name = [category_name]  # category name

        mesh_dicts = []
        mesh_dict_idx_to_category_name = {}
        category_name_to_mesh_dict_idxes = {}
        category_part_indicator_to_mesh_dict_idxes = {}
        for cur_cat_nm in category_name: # category na
            cur_root_folder = os.path.join(opt.dataset.dataset_root_path, opt.dataset.dataset_name, cur_cat_nm)
            print(f"Loading part objs from folder {cur_root_folder}...") # load part objs # load part objs
            if category_nm_to_part_nm is None:
                if cur_cat_nm == "eyeglasses":
                    part_names = ["none_motion"]
                else:
                    part_names = ["dof_rootd_Aa001_r", "dof_rootd_Aa002_r"]
            else:
                part_names = category_nm_to_part_nm[cur_cat_nm]

            print(f"Pretraining dataset, start loading data for the category {cur_cat_nm} with parts {part_names}.")
            if self.data_type == 'binvox':
                if not self.load_meta:
                    cur_cat_mesh_dicts, part_tree, part_nm_to_mesh_dict_idxes = dataset_utils.get_mesh_dict_list_multi_part_vox(
                        cur_root_folder, part_names=part_names,
                        ret_part_name_to_mesh_idx=True, remove_du=self.remove_du)
                else:
                  # get_mesh_dict_list_obj_vox_meta_info
                    # cur_cat_mesh_dicts, part_tree, part_nm_to_mesh_dict_idxes = dataset_utils.get_mesh_dict_list_multi_part_vox_meta_info(
                    #     cur_root_folder, part_names=part_names,
                    #     ret_part_name_to_mesh_idx=True, remove_du=self.remove_du, use_inst=self.use_inst)
                    ####### get meta info for the current split's data #########
                    cur_cat_mesh_dicts, part_tree, part_nm_to_mesh_dict_idxes = dataset_utils.get_mesh_dict_list_obj_vox_meta_info(
                        cur_root_folder, part_names=part_names,
                        ret_part_name_to_mesh_idx=True, remove_du=self.remove_du, use_inst=self.use_inst, split=self.split)
            else:
                cur_cat_mesh_dicts, part_tree, part_nm_to_mesh_dict_idxes = dataset_utils.get_mesh_dict_list_multi_part_mesh_vertices(
                    cur_root_folder, part_names=part_names,
                    ret_part_name_to_mesh_idx=True, remove_du=self.remove_du)
            print(f"Dataset loaded with {len(cur_cat_mesh_dicts)} instances for category {cur_cat_nm} with parts {part_names}.") # categories and parts

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
            

        print(f"numbers of valid samples: {len(mesh_dicts)}")

        tot_n_mesh = len(mesh_dicts) if not self.balance_classes else len(self.balanced_mesh_dicts)
        self.mesh_dicts = mesh_dicts
        self.mesh_idx_to_part_pts_idxes = {}

        self.dataset_len = tot_n_mesh

    def __len__(self):
        return self.dataset_len


    def convert_to_grid_xyz(self, vertices, pts_idxes=None):
      # vertices: n_verts
      # n_max_grids x (grid_size x grid_size) --> grid points information
      # n_max_grids x 3 --> grid coordinates information
      # input should be permuted vertices
      grid_size = opt.vertex_model.grid_size

      if pts_idxes is None:
        pts_idxes = [np.arange(0, vertices.shape[0])]
      
      all_grid_xyzs = []
      all_grid_pts = []
      all_grid_content_mask = []

      # part_sep_grid_pts = data_utils.convert_grid_content_to_grid_pts(PART_SEP_GRID_VALUE, grid_size=self.grid_size)
      part_sep_grid_pts = torch.tensor([PART_SEP_GRID_VALUE], dtype=torch.long)
      
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
        tot_grid_content_mask = []
        sorted_grid_points = sorted(self.grid_xyz_to_points.items(), key=lambda ii: ii[0], reverse=False)
        # tot_grid_pts: nn_grids x (grid_pts)
        for cur_item in sorted_grid_points:
          # print(f"grid pts: {self.convert_grid_values_to_grid_content(cur_item[1].unsqueeze(0)).size()}")
          tot_grid_pts.append(self.convert_grid_values_to_grid_content(cur_item[1].unsqueeze(0)))
          tot_grid_xyzs.append(cur_item[0]) # sorted xyzs
          tot_grid_content_mask.append(torch.ones((1,), dtype=torch.float32))
        
        if i_part < len(pts_idxes) - 1:
          cur_part_sep_xyzs = tuple(PART_SEP_XYZ)
          tot_grid_xyzs.append(cur_part_sep_xyzs)
          # print(part_sep_grid_pts.unsqueeze(0).size())
          tot_grid_pts.append(part_sep_grid_pts)
          tot_grid_content_mask.append(torch.zeros((1,), dtype=torch.float32))

          # cur_part_sep_pts 

        tot_grid_xyzs = torch.tensor(tot_grid_xyzs, dtype=torch.long) # n_grids x 3 --> grid xyzs
        tot_grid_pts = torch.cat(tot_grid_pts, dim=0) # n_grids x (grid_size x grid_size x grid_size)
        tot_grid_content_mask =torch.cat(tot_grid_content_mask, dim=0)
        

        all_grid_xyzs.append(tot_grid_xyzs)
        all_grid_pts.append(tot_grid_pts)
        all_grid_content_mask.append(tot_grid_content_mask)
      
      ending_grid_xyzs = torch.tensor(ENDING_XYZ, dtype=torch.long)
      all_grid_xyzs.append(ending_grid_xyzs.unsqueeze(0))
      # ending_grid_pts = data_utils.convert_grid_content_to_grid_pts(ENDING_GRID_VALUE, grid_size=self.grid_size)
      ending_grid_pts = torch.tensor([ENDING_GRID_VALUE], dtype=torch.long)
      all_grid_pts.append(ending_grid_pts)

      all_grid_content_mask.append(torch.zeros((1,), dtype=torch.float32))

      # all_grid_pts.append(torch.)
      tot_grid_pts = torch.cat(all_grid_pts, dim=0)
      tot_grid_xyzs = torch.cat(all_grid_xyzs, dim=0)
      tot_grid_content_mask = torch.cat(all_grid_content_mask, dim=0)
      #   tot_grid_pts = tot_grid_pts[:10]
      #   tot_grid_xyzs = tot_grid_xyzs[:10]
      return tot_grid_pts, tot_grid_xyzs, tot_grid_content_mask

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
      

    ''' Load data with random axises flipping and random vertices shifting  '''
    def __getitem__(self, item):
        
        # if not self.is_training:
        #   item = 0
        if self.balance_classes:
            item = self.balanced_mesh_dicts[item] # get new mesh dict index of the current item

        if not isinstance(self.mesh_dicts[item], dict):
            # self.mesh_dicts[item] = dataset_utils.read_binvox_to_pts(self.mesh_dicts[item])
            self.mesh_dicts[item], pts_idxes  = dataset_utils.read_binvox_list_to_pts(self.mesh_dicts[item])
            self.mesh_idx_to_part_pts_idxes[item] = pts_idxes # read binvox
        else:
            pts_idxes = self.mesh_idx_to_part_pts_idxes[item]

        # category_part_nm indicator...
        category_part_nm_indicator = self.mesh_dict_idx_to_category_name[item]
        category_part_idx = self.category_part_indicator_to_idxes[category_part_nm_indicator]

        cur_item_mesh_dict = self.mesh_dicts[item]
        # grid xyz coordinates and grid contents

        # make vertices; long vertices # 
        vertices = torch.from_numpy(cur_item_mesh_dict['vertices']).long()
        # if self.random_scaling > 0.5:
        #     # print("Applying random scaling...")
        #     scaled_vertices = self.apply_random_scaling(vertices.numpy())
        #     vertices = torch.from_numpy(scaled_vertices).long()
        if self.apply_random_shift:
            vertices = data_utils.random_shift(vertices=vertices, shift_factor=0.25)
        # faces = torch.from_numpy(cur_item_mesh_dict['faces']).long()
        if not self.category_part_indicator:  # add class label
            class_label = torch.tensor([cur_item_mesh_dict['class_label']], dtype=torch.long)
        else:
            # print(f"current class label: {category_part_idx}, category_part_nm_indicator: {category_part_nm_indicator}")
            class_label = torch.tensor([category_part_idx], dtype=torch.long)


        # Re-order vertex coordinates as (z, y, x).
        vertices_permuted = torch.cat(
            [vertices[..., 2].unsqueeze(-1), vertices[..., 1].unsqueeze(-1), vertices[..., 0].unsqueeze(-1)], dim=-1
        )

        # grid_size = opt.vertex_model.grid_size

        # grid_pts: n_grids x grid_size x grid_size x grid_size; grid_xyzs: n_grids x 3
        grid_pts, grid_xyzs, grid_content_mask = self.convert_to_grid_xyz(vertices_permuted, pts_idxes) # grid_pts: n_grids x ... ##### pts_idxes!
        # grid_xyzs_flat = grid_xyzs.contiguous().view(-1).contiguous() # (n_grids x 3, )
        # grid_pts_flat = grid_pts.contiguous().view(grid_pts.size(0), grid_size ** 3).contiguous() # n_grids x (grid_size ** 3)
        # grid_pts_flat

        # grid_content = self.convert_grid_values_to_grid_content(grid_pts) # nn_grids # content!
        grid_content = grid_pts
 
        grid_pos = torch.arange(start=0, end=grid_pts.size(0), step=1, dtype=torch.long) # grid pos

        nn_grids = grid_content.size(0)
        # print(f"nn_grids: {nn_grids}, max_num_grids: {self.max_num_grids}, grid_size: {self.grid_size}")

        # grid_pts_mask = torch.ones((grid_pts.size(0), self.grid_size, self.grid_size, self.grid_size), dtype=torch.float32)

        # grid_content_mask = torch.ones((grid_content.size(0), ), dtype=torch.float32)
        # grid_xyzs_mask = torch.ones((grid_xyzs.size(0), 3), dtype=torch.float32)

        
        if nn_grids >= self.max_num_grids:
          cut_grid_st_pos = np.random.randint(low=0, high=nn_grids - self.max_num_grids + 2, size=(1,)).item()
          # grid_pts = grid_pts[cut_grid_st_pos: cut_grid_st_pos + self.max_num_grids - 1] # grid_pts
          grid_xyzs = grid_xyzs[cut_grid_st_pos: cut_grid_st_pos + self.max_num_grids - 1]
          grid_pos = grid_pos[cut_grid_st_pos: cut_grid_st_pos + self.max_num_grids - 1] # max_num_grids
          grid_content = grid_content[cut_grid_st_pos: cut_grid_st_pos + self.max_num_grids - 1] # grid_content: xxx
          grid_content_mask = grid_content_mask[cut_grid_st_pos: cut_grid_st_pos + self.max_num_grids - 1]

          grid_xyzs_mask = torch.ones((grid_xyzs.size(0), 3), dtype=torch.float32)
          # grid_pts_mask
          ##### grid_pts_mask 

          # print(f"cutting nn_grids: {nn_grids}, max_num_grids: {self.max_num_grids}, sampled cut_grid_st_pos: {cut_grid_st_pos}")
          # 
          if self.is_training:
            if cut_grid_st_pos == nn_grids - self.max_num_grids + 1:
              # grid_xyzs_mask = torch.cat(
              #   [grid_xyzs_mask, torch.tensor([[1, 0, 0]], dtype=torch.float32), torch.zeros((self.max_num_grids - grid_xyzs.size(0) - 1, 3), dtype=torch.float32)], dim=0
              # )
              grid_xyzs_mask = torch.cat(
                [grid_xyzs_mask, torch.tensor([[0, 0, 0]], dtype=torch.float32), torch.zeros((self.max_num_grids - grid_xyzs.size(0) - 1, 3), dtype=torch.float32)], dim=0
              )
            else:
              grid_xyzs_mask = torch.cat(
                [grid_xyzs_mask, torch.tensor([[0, 0, 0]], dtype=torch.float32), torch.zeros((self.max_num_grids - grid_xyzs.size(0) - 1, 3), dtype=torch.float32)], dim=0
              )
        else:
          grid_xyzs_mask = torch.ones((grid_xyzs.size(0), 3), dtype=torch.float32)
          if self.is_training:
            # grid_xyzs_mask = torch.cat(
            #   [grid_xyzs_mask, torch.tensor([[1, 0, 0]], dtype=torch.float32), torch.zeros((self.max_num_grids - grid_xyzs.size(0) - 1, 3), dtype=torch.float32)], dim=0
            # )
            grid_xyzs_mask = torch.cat(
              [grid_xyzs_mask, torch.tensor([[0, 0, 0]], dtype=torch.float32), torch.zeros((self.max_num_grids - grid_xyzs.size(0) - 1, 3), dtype=torch.float32)], dim=0
            )

        # grid_pts_mask = torch.ones((grid_pts.size(0), self.grid_size, self.grid_size, self.grid_size), dtype=torch.float32)

        # grid_content_mask = torch.ones((grid_content.size(0), ), dtype=torch.float32)
        

        # grid_xyzs_mask = torch.cat(
        #   [torch.ones((grid_xyzs.size(0) + 1, 3), dtype=torch.float32), torch.zeros((self.max_num_grids - grid_xyzs.size(0) - 1, 3), dtype=torch.float32)], dim=0
        # )

        ##### grid_xyzs_mask #####
        # grid_xyzs_mask = torch.cat(
        #   [torch.ones((grid_xyzs.size(0), 3), dtype=torch.float32), torch.tensor([[1, 0, 0]], dtype=torch.float32), torch.zeros((self.max_num_grids - grid_xyzs.size(0) - 1, 3), dtype=torch.float32)], dim=0
        # )
        ##### grid_xyzs_mask #####

        grid_xyzs = grid_xyzs + 1

        if self.is_training:
          # grid_pts_mask = torch.cat(
            # [grid_pts_mask, torch.zeros((self.max_num_grids - grid_xyzs.size(0), self.grid_size, self.grid_size, self.grid_size), dtype=torch.float32)], dim=0
          # )

          grid_content_mask = torch.cat(
              [grid_content_mask, torch.zeros((self.max_num_grids - grid_content.size(0), ), dtype=torch.float32)], dim=0
          )

          # grid size should < max_num_grids --- should leave one grid position for blank masking
          # grid position for dummy grids --- self.max_num_grids
          # grid_xyzs: n_grids x 3
          grid_xyzs = torch.cat(
            [grid_xyzs, torch.zeros((self.max_num_grids - grid_xyzs.size(0), 3), dtype=torch.long)], dim=0
          )
          # ### grid_pts: self.max_num_grids x gs x gs x gs
          # grid_pts = torch.cat(
          #   [grid_pts, torch.zeros((self.max_num_grids - grid_pts.size(0), self.grid_size, self.grid_size, self.grid_size), dtype=torch.long)], dim=0
          # )

          grid_content = torch.cat(
            [grid_content, torch.zeros((self.max_num_grids - grid_content.size(0), ), dtype=torch.long) + (2 ** (self.grid_size ** 3))], dim=0
          )

          # grid_pos
          # grid_pos = torch.cat(
          #   [grid_pos, torch.full((self.max_num_grids - grid_pos.size(0),), fill_value=self.max_num_grids, dtype=torch.long)], dim=0
          # )

          grid_pos = torch.cat(
            [grid_pos, torch.arange(grid_pos.size(0), self.max_num_grids, dtype=torch.long)], dim=0
          )

        

        

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

        rt_dict = {
            # 'vertices_flat_mask': vertices_flat_mask,
            # 'vertices_flat': vertices_flat,
            # 'faces_mask': faces_mask,
            'vertices': vertices,
            # 'vertices_mask': vertices_mask,
            # 'faces': faces,
            'class_label': class_label,
            # 'grid_xyzs_flat': grid_xyzs_flat,
            'grid_xyzs': grid_xyzs,
            'grid_xyzs_mask': grid_xyzs_mask,
            # 'grid_pts_flat': grid_pts_flat,
            # 'grid_content': grid_pts,
            # 'grid_content_mask': grid_pts_mask,
            'grid_pos': grid_pos,
            'grid_content_vocab': grid_content,
            'grid_content_vocab_mask': grid_content_mask
            
            # 'vertices_flat_masked': vertices_flat_masked,
            # 'vertices_mask_identifier': vertices_mask_identifier,
            # 'vertices_permutation': permut,
            # 'start_sampling_idx': start_sampling_idx,
            # 'vertices_masked_permutation': vertices_mask_permut
        }

        return rt_dict
