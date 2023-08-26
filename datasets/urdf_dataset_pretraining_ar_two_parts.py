import os

import os.path
import json
import numpy as np
import math
import sys
import torch

from torch.utils import data
# import data_utils_torch as data_utils
import utils.data_utils_torch  as data_utils
import utils.dataset_utils as dataset_utils

from options.options import opt

# design the 

class URDFDataset(data.Dataset):
    def __init__(self, root_folder, quantization_bits=8, nn_max_vertices=8900, nn_max_faces=8900, nn_max_permite_vertices=400, nn_max_permite_faces=2000, category_name="eyeglasses", instance_nns=[], part_names=None, category_nm_to_part_nm=None, is_training=False
    ):
        super(URDFDataset, self).__init__()
        
        self.root_folder = root_folder
        # self.part_names = part_names # part names
        # self.statistic_folder = statistic_folder
        self.category_name = category_name # category 
        self.quantization_bits = quantization_bits 
        self.nn_max_vertices = nn_max_vertices + 1
        self.nn_max_faces = nn_max_faces
        self.instance_nns = instance_nns
        self.is_training = is_training
        self.mode = opt.common.exp_mode

        debug = opt.model.debug
        datasest_name = opt.dataset.dataset_name
        self.apply_random_shift = opt.dataset.apply_random_shift
        self.apply_random_flipping = opt.dataset.apply_random_flipping
        self.category_part_indicator = opt.dataset.category_part_indicator

        # context_design_strategy = opt.model.context_design_strategy

        samples_list = os.listdir(self.root_folder)
        samples_list = [fn for fn in samples_list if os.path.isdir(os.path.join(self.root_folder, fn))]

        print(f"number of samples: {len(samples_list)}")

        category_part_indicator_to_idxes = {}

        if datasest_name in ["MotionDataset", "PolyGen_Samples"]:
          mesh_dicts, part_tree = dataset_utils.get_mesh_dict_list(self.root_folder, samples_list, valid_indices=self.instance_nns)
        else:

          if not isinstance(category_name, list):
            category_name = [category_name] # category name
          
          mesh_dict_idx_to_category_name = {}
          category_name_to_mesh_dict_idxes = {}
          mesh_dicts = []
          for cur_cat_nm in category_name:
            cur_root_folder = os.path.join(opt.dataset.dataset_root_path, opt.dataset.dataset_name, cur_cat_nm)
            if category_nm_to_part_nm is None:
              if cur_cat_nm == "eyeglasses":
                part_names = ["none_motion"]
              else:
                part_names=["dof_rootd_Aa001_r", "dof_rootd_Aa002_r"]
            else:
              part_names = category_nm_to_part_nm[cur_cat_nm]
            
            print(f"Pretraining dataset, start loading data for the category {cur_cat_nm} with parts {part_names}.")
            cur_cat_mesh_dicts, part_tree, part_nm_to_mesh_dict_idxes = dataset_utils.get_mesh_dict_list_multi_part_part_first(cur_root_folder, part_names=part_names, valid_indices=self.instance_nns if cur_cat_nm == "eyeglasses" else None, category_name=cur_cat_nm, ret_part_name_to_mesh_idx=True)

            cur_mesh_dict_len = len(mesh_dicts)
            for cur_cat_cur_part_nm in part_nm_to_mesh_dict_idxes:
              new_mesh_dict_idxes = []
              for iii in part_nm_to_mesh_dict_idxes[cur_cat_cur_part_nm]:
                new_mesh_dict_idxes.append(iii + cur_mesh_dict_len)
                category_part_nm_indicator = f"{cur_cat_nm}-{cur_cat_cur_part_nm}"
                mesh_dict_idx_to_category_name[iii + cur_mesh_dict_len] = category_part_nm_indicator
                if category_part_nm_indicator not in category_part_indicator_to_idxes:
                  category_part_indicator_to_idxes[category_part_nm_indicator] = len(category_part_indicator_to_idxes)
              part_nm_to_mesh_dict_idxes[cur_cat_cur_part_nm] = new_mesh_dict_idxes
              # part_nm_to_mesh

            category_name_to_mesh_dict_idxes[cur_cat_nm] = part_nm_to_mesh_dict_idxes
            # cur_category_mesh_dict_idxes = range(cur_mesh_dict_len, cur_mesh_dict_len + len(cur_cat_mesh_dicts))
            mesh_dicts += cur_cat_mesh_dicts
            # category_name_to_mesh_dict_idxes[cur_cat_nm] = cur_category_mesh_dict_idxes
            # for cur_mesh_idx in cur_category_mesh_dict_idxes:
            #   mesh_dict_idx_to_category_name[cur_mesh_idx] = cur_cat_nm
          self.category_name_to_mesh_dict_idxes = category_name_to_mesh_dict_idxes
          self.mesh_dict_idx_to_category_name = mesh_dict_idx_to_category_name
          self.category_part_indicator_to_idxes = category_part_indicator_to_idxes

        print(f"numbers of valid samples: {len(mesh_dicts)}")

        tot_n_mesh = len(mesh_dicts)
        self.mesh_dicts = mesh_dicts

        self.dataset_len = tot_n_mesh

    def __len__(self):
        return self.dataset_len

    # gneerate vertices mask...
    def _generate_vertices_mask_bak(self, vertices_flat):
      mask_ratio_low = opt.dataset.mask_low_ratio
      mask_ratio_high = opt.dataset.mask_high_ratio
      tot_n = vertices_flat.size(-1) + 1 # 
      # mask_ratio = np.random.uniform(low=0.05, high=0.06, size=(1,)).item()
      mask_ratio = np.random.uniform(low=mask_ratio_low, high=mask_ratio_high, size=(1,)).item()
      mask_ratio = float(mask_ratio) # ratio
      cur_nn_masks = int(mask_ratio * tot_n)
      cur_nn_masks = cur_nn_masks if cur_nn_masks > 1 else 1
      vertices_flat_masked = vertices_flat.clone()
      # vertices_flat_masked = np.zeros_like(vertices_flat)
      # vertices_flat_masked[:] = vertices_flat[:]
      # NNNN
      ### + 1!!!
      vertices_flat_masked = torch.cat([vertices_flat_masked + 1, torch.zeros((1, ), dtype=torch.long)], dim=-1)

      sampled_masked_coord_indices = np.random.choice(tot_n, size=cur_nn_masks, replace=False)
      sampled_masked_coord_indices = torch.from_numpy(sampled_masked_coord_indices).long() # maksed coord 
      vertices_flat_masked[sampled_masked_coord_indices] = 2 ** self.quantization_bits + 1
      vertices_mask_identifier = torch.zeros_like(vertices_flat_masked).float()
      vertices_mask_identifier[sampled_masked_coord_indices] = 1.
      
      vertices_flat_masked = torch.cat([vertices_flat_masked, torch.zeros((self.nn_max_vertices * 3 - vertices_flat_masked.size(0)), dtype=torch.long)], dim=-1)

      vertices_mask_permut = torch.argsort(vertices_mask_identifier, dim=0, descending=True) # identifier

      vertices_mask_identifier = torch.cat([vertices_mask_identifier, torch.zeros((self.nn_max_vertices * 3 - vertices_mask_identifier.size(0)), dtype=torch.float32)], dim=-1)

      res_permut = np.array(range(vertices_mask_permut.size(0), self.nn_max_vertices * 3), dtype=np.int32)
      res_permut = torch.from_numpy(res_permut).long()
      vertices_mask_permut = torch.cat([vertices_mask_permut, res_permut], dim=-1)

      return vertices_flat_masked, vertices_mask_identifier, vertices_mask_permut
    
    # gneerate vertices mask...
    def _generate_vertices_mask(self, vertices_flat):
      # mask_ratio_low = opt.dataset.mask_low_ratio
      # mask_ratio_high = opt.dataset.mask_high_ratio
      tot_n = vertices_flat.size(-1) + 1 # 
      # mask_ratio = np.random.uniform(low=0.05, high=0.06, size=(1,)).item()
      # mask_ratio = np.random.uniform(low=mask_ratio_low, high=mask_ratio_high, size=(1,)).item()
      # mask_ratio = float(mask_ratio) # ratio
      # cur_nn_masks = int(mask_ratio * tot_n)
      # cur_nn_masks = cur_nn_masks if cur_nn_masks > 1 else 1
      vertices_flat_masked = vertices_flat.clone()
      # vertices_flat_masked = np.zeros_like(vertices_flat)
      # vertices_flat_masked[:] = vertices_flat[:]
      # NNNN
      ### + 1!!!
      # vertices_flat_masked = torch.cat([vertices_flat_masked + 1, torch.zeros((1, ), dtype=torch.long)], dim=-1)

      # sampled_masked_coord_indices = np.random.choice(tot_n, size=cur_nn_masks, replace=False)
      # sampled_masked_coord_indices = torch.from_numpy(sampled_masked_coord_indices).long() # maksed coord 
      # vertices_flat_masked[sampled_masked_coord_indices] = 2 ** self.quantization_bits + 1
      # vertices_mask_identifier = torch.zeros_like(vertices_flat_masked).float()
      # vertices_mask_identifier[sampled_masked_coord_indices] = 1.
      
      # vertices_flat_masked = torch.cat([vertices_flat_masked, torch.zeros((self.nn_max_vertices * 3 - vertices_flat_masked.size(0)), dtype=torch.long)], dim=-1)

      vertices_mask_permut = torch.argsort(vertices_mask_identifier, dim=0, descending=True) # identifier

      vertices_mask_identifier = torch.cat([vertices_mask_identifier, torch.zeros((self.nn_max_vertices * 3 - vertices_mask_identifier.size(0)), dtype=torch.float32)], dim=-1)

      res_permut = np.array(range(vertices_mask_permut.size(0), self.nn_max_vertices * 3), dtype=np.int32)
      res_permut = torch.from_numpy(res_permut).long()
      vertices_mask_permut = torch.cat([vertices_mask_permut, res_permut], dim=-1)

      return vertices_flat_masked, vertices_mask_identifier, vertices_mask_permut
    
    # generate interpolation mask...
    def generate_interpolate_mask(self, vertices):
      # TODO a more general version -- the current one is only suitable for shapes with small number of vertices
      nn_verts = vertices.size(0)
      two_times_verts_permutation = torch.zeros((nn_verts * 2, 3), dtype=torch.long)
      vertical_orders = torch.arange(0, nn_verts * 2, dtype=torch.long) * 3 ## verti
      horizental_orders = torch.tensor([0, 1, 2], dtype=torch.long)
      two_times_verts_permutation = two_times_verts_permutation + vertical_orders.unsqueeze(-1) + horizental_orders.unsqueeze(0)
      # (nn_verts * 2) x 3
      current_permutation = two_times_verts_permutation[::2, :]
      interpolation_permutation = two_times_verts_permutation[1::2, :]
      if opt.model.debug:
        print("current permutation:", current_permutation[:5])
        print("interpolation permutation:", interpolation_permutation[:5])
      two_times_flat_permutation = torch.cat(
        [current_permutation.contiguous().view(-1).contiguous(),
        interpolation_permutation.contiguous().view(-1).contiguous()], dim=0
      )
      start_sampling_idx = nn_verts * 3

      dequan_vertices = data_utils.dequantize_verts(vertices.numpy(), self.quantization_bits)
      dequan_vertices = data_utils.center_vertices(dequan_vertices)
      quan_vertices = data_utils.quantize_verts(dequan_vertices, self.quantization_bits)
      quan_vertices = torch.from_numpy(quan_vertices).long()
      
      vertices_permuted = torch.cat(
          [quan_vertices[..., 2].unsqueeze(-1), quan_vertices[..., 1].unsqueeze(-1), quan_vertices[..., 0].unsqueeze(-1)], dim=-1
      ) + 1
      vertices_permuted_two_times = torch.zeros((vertices_permuted.size(0) * 2, 3), dtype=torch.long)
      for i_vert in range(vertices_permuted.size(0)):
        vertices_permuted_two_times[i_vert * 2, :] = vertices_permuted[i_vert, :]
      vertices_permuted_two_times = vertices_permuted_two_times.contiguous().view(-1).contiguous()

      return two_times_flat_permutation, start_sampling_idx, vertices_permuted_two_times
      
    def generate_scaled_verts_mask(self, vertices):
      nn_verts = vertices.size(0)
      dequan_vertices = data_utils.dequantize_verts(vertices.numpy()) # dequan_vertices
      dequan_vertices = data_utils.center_vertices(dequan_vertices)
      vertices_scaling = np.array([1.0, 1.0, 0.33], dtype=np.float32)
      # minn_z = np.min()
      dequan_vertices = dequan_vertices * np.reshape(vertices_scaling, (1, 3))
      dequan_vertices_min = np.min(dequan_vertices, axis=0)[-1].item()

      # dequan_vertices = dequan_vertices + np.array([[0.0, 0.0, -0.5 - dequan_vertices_min]], dtype=np.float32)

      quantized_vertices = data_utils.quantize_verts(dequan_vertices, n_bits=self.quantization_bits)
      quantized_vertices = torch.from_numpy(quantized_vertices).long()
      vertices_permuted = torch.cat(
          [quantized_vertices[..., 2].unsqueeze(-1), quantized_vertices[..., 1].unsqueeze(-1), quantized_vertices[..., 0].unsqueeze(-1)], dim=-1
      ) + 1
      vertices_permuted_flat = vertices_permuted.contiguous().view(-1).contiguous()
      nn_verts_flat = vertices_permuted_flat.size(-1)
      ori_permutation = torch.arange(0, nn_verts_flat).long()
      vertices_permutation = torch.cat(
        [ori_permutation + nn_verts_flat, ori_permutation, ori_permutation + nn_verts_flat * 2], dim=0
      )
      # vertices_permutation = torch.cat(
      #   [ori_permutation, ori_permutation + nn_verts_flat, ori_permutation + nn_verts_flat * 2], dim=0
      # )
      start_sampling_idx = nn_verts_flat
      vertices_permuted_flat = torch.cat(
        [torch.zeros_like(vertices_permuted_flat), vertices_permuted_flat, torch.zeros_like(vertices_permuted_flat)], dim=0
      )
      # vertices_permuted_flat = torch.cat(
      #   [vertices_permuted_flat, torch.zeros_like(vertices_permuted_flat), torch.zeros_like(vertices_permuted_flat)], dim=0
      # )
      return vertices_permutation, start_sampling_idx, vertices_permuted_flat # return vertices permutations, sampling idxes, and permuted flat...

    def generate_scaled_verts_mask_low_part(self, vertices):
      nn_verts = vertices.size(0)
      dequan_vertices = data_utils.dequantize_verts(vertices.numpy()) # dequan_vertices
      dequan_vertices = data_utils.center_vertices(dequan_vertices)
      vertices_scaling = np.array([1.0, 1.0, 0.33], dtype=np.float32)
      # minn_z = np.min()
      dequan_vertices = dequan_vertices * np.reshape(vertices_scaling, (1, 3))
      dequan_vertices_min = np.min(dequan_vertices, axis=0)[-1].item()

      dequan_vertices = dequan_vertices + np.array([[0.0, 0.0, -0.5 - dequan_vertices_min]], dtype=np.float32)
      
      quantized_vertices = data_utils.quantize_verts(dequan_vertices, n_bits=self.quantization_bits)
      quantized_vertices = torch.from_numpy(quantized_vertices).long()
      vertices_permuted = torch.cat(
          [quantized_vertices[..., 2].unsqueeze(-1), quantized_vertices[..., 1].unsqueeze(-1), quantized_vertices[..., 0].unsqueeze(-1)], dim=-1
      ) + 1
      vertices_permuted_flat = vertices_permuted.contiguous().view(-1).contiguous()
      nn_verts_flat = vertices_permuted_flat.size(-1)
      ori_permutation = torch.arange(0, nn_verts_flat).long()
      # vertices_permutation = torch.cat(
      #   [ori_permutation + nn_verts_flat, ori_permutation, ori_permutation + nn_verts_flat * 2], dim=0
      # )
      vertices_permutation = torch.cat(
        [ori_permutation, ori_permutation + nn_verts_flat, ori_permutation + nn_verts_flat * 2], dim=0
      )
      start_sampling_idx = nn_verts_flat
      # vertices_permuted_flat = torch.cat(
      #   [torch.zeros_like(vertices_permuted_flat), vertices_permuted_flat, torch.zeros_like(vertices_permuted_flat)], dim=0
      # )
      vertices_permuted_flat = torch.cat(
        [vertices_permuted_flat, torch.zeros_like(vertices_permuted_flat), torch.zeros_like(vertices_permuted_flat)], dim=0
      )
      return vertices_permutation, start_sampling_idx, vertices_permuted_flat # return vertices permutations, sampling idxes, and permuted flat...

    def generate_scaled_verts_mask_high_part(self, vertices):
      nn_verts = vertices.size(0)
      dequan_vertices = data_utils.dequantize_verts(vertices.numpy()) # dequan_vertices
      dequan_vertices = data_utils.center_vertices(dequan_vertices)
      vertices_scaling = np.array([1.0, 1.0, 1.0], dtype=np.float32)
      # minn_z = np.min()
      dequan_vertices = dequan_vertices * np.reshape(vertices_scaling, (1, 3))
      dequan_vertices_max = np.max(dequan_vertices, axis=0)[-1].item()

      dequan_vertices = dequan_vertices + np.array([[0.0, 0.0, 0.5 - dequan_vertices_max]], dtype=np.float32)
      
      quantized_vertices = data_utils.quantize_verts(dequan_vertices, n_bits=self.quantization_bits)
      quantized_vertices = torch.from_numpy(quantized_vertices).long()
      vertices_permuted = torch.cat(
          [quantized_vertices[..., 2].unsqueeze(-1), quantized_vertices[..., 1].unsqueeze(-1), quantized_vertices[..., 0].unsqueeze(-1)], dim=-1
      ) + 1
      vertices_permuted_flat = vertices_permuted.contiguous().view(-1).contiguous()
      nn_verts_flat = vertices_permuted_flat.size(-1)
      ori_permutation = torch.arange(0, nn_verts_flat).long()
      # vertices_permutation = torch.cat(
      #   [ori_permutation + nn_verts_flat, ori_permutation, ori_permutation + nn_verts_flat * 2], dim=0
      # )
      vertices_permutation = torch.cat(
        [ori_permutation  + nn_verts_flat * 2, ori_permutation, ori_permutation + nn_verts_flat], dim=0
      )
      start_sampling_idx = nn_verts_flat
      # vertices_permuted_flat = torch.cat(
      #   [torch.zeros_like(vertices_permuted_flat), vertices_permuted_flat, torch.zeros_like(vertices_permuted_flat)], dim=0
      # )
      vertices_permuted_flat = torch.cat(
        [torch.zeros_like(vertices_permuted_flat), torch.zeros_like(vertices_permuted_flat), vertices_permuted_flat], dim=0
      )
      return vertices_permutation, start_sampling_idx, vertices_permuted_flat # return vertices permutations, sampling idxes, and permuted flat...

    def _scale_vertices_to_up(self, vertices):
      nn_verts = vertices.size(0)
      dequan_vertices = data_utils.dequantize_verts(vertices.numpy()) # dequan_vertices
      dequan_vertices = data_utils.center_vertices(dequan_vertices)
      vertices_scaling = np.array([1.0, 1.0, 0.45], dtype=np.float32)
      # minn_z = np.min()
      dequan_vertices = dequan_vertices * np.reshape(vertices_scaling, (1, 3))
      dequan_vertices_max = np.max(dequan_vertices, axis=0)[-1].item()

      dequan_vertices = dequan_vertices + np.array([[0.0, 0.0, 0.5 - dequan_vertices_max]], dtype=np.float32)
      
      quantized_vertices = data_utils.quantize_verts(dequan_vertices, n_bits=self.quantization_bits)
      quantized_vertices = torch.from_numpy(quantized_vertices).long()
      return quantized_vertices
    
    def _scale_vertices_to_down(self, vertices):
      nn_verts = vertices.size(0)
      # dequantize_verts()
      dequan_vertices = data_utils.dequantize_verts(vertices.numpy()) # dequan_vertices
      dequan_vertices = data_utils.center_vertices(dequan_vertices)
      vertices_scaling = np.array([1.0, 1.0, 0.45], dtype=np.float32)
      # minn_z = np.min()
      dequan_vertices = dequan_vertices * np.reshape(vertices_scaling, (1, 3))
      dequan_vertices_min = np.min(dequan_vertices, axis=0)[-1].item()

      dequan_vertices = dequan_vertices + np.array([[0.0, 0.0, -0.5 - dequan_vertices_min]], dtype=np.float32)
      
      quantized_vertices = data_utils.quantize_verts(dequan_vertices, n_bits=self.quantization_bits)
      quantized_vertices = torch.from_numpy(quantized_vertices).long()
      return quantized_vertices
    

    def __getitem__(self, item):


        cur_item_category_name = self.mesh_dict_idx_to_category_name[item]

        # category_part_nm_indicator = self.mesh_dict_idx_to_category_name[item]
        category_part_idx = self.category_part_indicator_to_idxes[cur_item_category_name]
        cur_item_category_name, cur_item_part_nm = cur_item_category_name.split("-")
        # category_name_to_mesh_dict_idxes --- fine the same part ####
        # #### cur_item_part_nm and cur_item_category_name ---- fine the same part #### #
        cur_category_item_idxes = self.category_name_to_mesh_dict_idxes[cur_item_category_name][cur_item_part_nm]
        
        # another item idx
        cur_sampled_another_item_idx = np.random.randint(0, len(cur_category_item_idxes), (1,)).item()
        cur_another_mesh_dict = cur_category_item_idxes[cur_sampled_another_item_idx]
        cur_another_mesh_dict = self.mesh_dicts[cur_another_mesh_dict]

        cur_item_mesh_dict = self.mesh_dicts[item]

        ############ Left part vertex model data construction ############
        # make vertices
        vertices = torch.from_numpy(cur_item_mesh_dict['vertices']).long()

        if self.apply_random_shift: # shift vertices
          vertices = data_utils.random_shift(vertices, shift_factor=0.25)

        # vertices = self._scale_vertices_to_up(vertices) # should not scale to high...
        faces = torch.from_numpy(cur_item_mesh_dict['faces']).long()
        # class_label = torch.tensor([cur_item_mesh_dict['class_label']], dtype=torch.long)

        if not self.category_part_indicator: # add class label
          class_label = torch.tensor([cur_item_mesh_dict['class_label']], dtype=torch.long)
        else:
          class_label = torch.tensor([category_part_idx], dtype=torch.long)

        # Re-order vertex coordinates as (z, y, x).
        vertices_permuted = torch.cat(
            [vertices[..., 2].unsqueeze(-1), vertices[..., 1].unsqueeze(-1), vertices[..., 0].unsqueeze(-1)], dim=-1
        )

        vertices_flat = vertices_permuted.contiguous().view(-1).contiguous()

        nn_vertices = vertices_flat.size(0)

        nn_faces = faces.size(0) # faces

        ''' minn vertex index value is 1 here, so + 1 can get real vertex indices for faces ''' 
        vertices_flat = torch.cat(
            [vertices_flat + 1, torch.zeros((self.nn_max_vertices * 3 - vertices_flat.size(0),), dtype=torch.long)], dim=-1
        )

        # ''' Permutation '''
        # # permut = np.random.permutation(nn_vertices)
        # # permut = torch.from_numpy(permut).long() # seq_len
        # # res_permut = np.array(range(nn_vertices, self.nn_max_vertices * 3), dtype=np.int32)
        # # res_permut = torch.from_numpy(res_permut).long()
        # # permut = torch.cat(
        # #   [permut, res_permut], dim=0
        # # )
        # permut = torch.arange(0, self.nn_max_vertices * 3, dtype=torch.long)
        # ''' Permutation '''
        
        vertices_flat_mask = torch.cat(
            [torch.ones((nn_vertices + 1,), dtype=torch.float32), torch.zeros((self.nn_max_vertices * 3 - nn_vertices - 1), dtype=torch.float32)], dim=-1
        )

        # dequantized vertices
        real_nn_vertices = vertices.size(0)

        # start_sampling_idx = np.random.randint(0, real_nn_vertices // 2, (1,))
        # start_sampling_idx = torch.from_numpy(start_sampling_idx).long()

        ########## Left part vertex model data construction ############
        # start_sampling_idx = np.random.randint(nn_vertices // 4, nn_vertices // 2, (1,))
        # start_sampling_idx = torch.from_numpy(start_sampling_idx).long()
        ########## Left part vertex model data construction ############
        # 



        ########## Left part vertex model data construction ############
        # make vertices
        vertices_other = torch.from_numpy(cur_another_mesh_dict['vertices']).long()
        if self.apply_random_shift:
          vertices_other = data_utils.random_shift(vertices_other, shift_factor=0.25)
        # vertices_other = self._scale_vertices_to_down(vertices_other)
        faces_other = torch.from_numpy(cur_another_mesh_dict['faces']).long()
        # class_label_other = torch.tensor([cur_another_mesh_dict['class_label']], dtype=torch.long)

        if not self.category_part_indicator: # add class label
          class_label_other = torch.tensor([cur_item_mesh_dict['class_label']], dtype=torch.long)
        else:
          class_label_other = torch.tensor([category_part_idx], dtype=torch.long)

        # Re-order vertex coordinates as (z, y, x). # vertices 
        vertices_permuted_other = torch.cat(
            [vertices_other[..., 2].unsqueeze(-1), vertices_other[..., 1].unsqueeze(-1), vertices_other[..., 0].unsqueeze(-1)], dim=-1
        )

        vertices_flat_other = vertices_permuted_other.contiguous().view(-1).contiguous()

        nn_vertices_other = vertices_flat_other.size(0)

        nn_faces_other = faces_other.size(0) # faces

        ''' minn vertex index value is 1 here, so + 1 can get real vertex indices for faces ''' 
        vertices_flat_other = torch.cat(
            [vertices_flat_other + 1, torch.zeros((self.nn_max_vertices * 3 - vertices_flat_other.size(0),), dtype=torch.long)], dim=-1
        )

        # ''' Permutation '''
        # # permut = np.random.permutation(nn_vertices)
        # # permut = torch.from_numpy(permut).long() # seq_len
        # # res_permut = np.array(range(nn_vertices, self.nn_max_vertices * 3), dtype=np.int32)
        # # res_permut = torch.from_numpy(res_permut).long()
        # # permut = torch.cat(
        # #   [permut, res_permut], dim=0
        # # )
        # permut = torch.arange(0, self.nn_max_vertices * 3, dtype=torch.long)
        # ''' Permutation '''
        
        vertices_flat_mask_other = torch.cat(
            [torch.ones((nn_vertices_other + 1,), dtype=torch.float32), torch.zeros((self.nn_max_vertices * 3 - nn_vertices_other - 1), dtype=torch.float32)], dim=-1
        )

        # dequantized vertices
        real_nn_vertices_other = vertices_other.size(0)

        # start_sampling_idx = np.random.randint(0, real_nn_vertices // 2, (1,))
        # start_sampling_idx = torch.from_numpy(start_sampling_idx).long()

        # start_sampling_idx_other = np.random.randint(nn_vertices_other // 4, nn_vertices_other // 2, (1,))
        # start_sampling_idx_other = torch.from_numpy(start_sampling_idx_other).long()
        # ########## Left part vertex model data construction ############

        
        # vertices 
        vertices = torch.cat( # 
            [vertices, torch.zeros((self.nn_max_vertices - real_nn_vertices, 3), dtype=torch.long)], dim=0
        )
        
        vertices_mask = torch.cat( # 
            [torch.ones((real_nn_vertices,), dtype=torch.float32), torch.zeros((self.nn_max_vertices - real_nn_vertices,), dtype=torch.float32)], dim=0
        )
        ###### 


        faces_mask = torch.cat(
            [torch.ones((nn_faces,), dtype=torch.float32), torch.zeros((self.nn_max_faces - nn_faces), dtype=torch.float32)], dim=-1
        )
        
        # faces and faces mask
        # left faces mask
        faces = torch.cat(
            [faces, torch.zeros((self.nn_max_faces - nn_faces), dtype=torch.long)], dim=-1
        )


        # vertices 
        vertices_other = torch.cat( # 
            [vertices_other, torch.zeros((self.nn_max_vertices - real_nn_vertices_other, 3), dtype=torch.long)], dim=0
        )
        
        vertices_mask_other = torch.cat( # 
            [torch.ones((real_nn_vertices_other,), dtype=torch.float32), torch.zeros((self.nn_max_vertices - real_nn_vertices_other,), dtype=torch.float32)], dim=0
        )
        ###### 


        faces_mask_other = torch.cat(
            [torch.ones((nn_faces_other,), dtype=torch.float32), torch.zeros((self.nn_max_faces - nn_faces_other), dtype=torch.float32)], dim=-1
        )
        
        # faces and faces mask
        # left faces mask
        faces_other = torch.cat(
            [faces_other, torch.zeros((self.nn_max_faces - nn_faces_other), dtype=torch.long)], dim=-1
        )

        nn_vertices_lft = torch.tensor([nn_vertices], dtype=torch.long)
        nn_vertices_rgt = torch.tensor([nn_vertices_other], dtype=torch.long)

        lft_rt_dict = {
          'vertices_flat_mask': vertices_flat_mask,
          'vertices_flat': vertices_flat, # flat...
          # 'vertices_flat_mask_': vertices_flat_mask_other,
          # 'vertices_flat_rgt': vertices_flat_other, # flat...
          'faces_mask': faces_mask,
          'vertices': vertices,
          'vertices_mask': vertices_mask,
          'faces': faces,
          'class_label': class_label,
          'num_vertices': nn_vertices_lft,
        }

        rgt_rt_dict = {
          'vertices_flat_mask': vertices_flat_mask_other,
          'vertices_flat': vertices_flat_other, # flat...
          # 'vertices_flat_mask_': vertices_flat_mask_other,
          # 'vertices_flat_rgt': vertices_flat_other, # flat...
          'faces_mask': faces_mask_other,
          'vertices': vertices_other,
          'vertices_mask': vertices_mask_other,
          'faces': faces_other,
          'class_label': class_label_other,
          'num_vertices': nn_vertices_rgt,
        }

        rt_dict = {
          'lft': lft_rt_dict,
          'rgt': rgt_rt_dict,
          'class_label': class_label
        }

        return rt_dict

