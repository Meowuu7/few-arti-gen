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
    def __init__(self, root_folder, quantization_bits=8, nn_max_vertices=8900, nn_max_faces=8900, nn_max_permite_vertices=400, nn_max_permite_faces=2000, category_name="eyeglasses", instance_nns=[], part_names=None, category_nm_to_part_nm=None
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

        debug = opt.model.debug
        datasest_name = opt.dataset.dataset_name

        samples_list = os.listdir(self.root_folder)
        samples_list = [fn for fn in samples_list if os.path.isdir(os.path.join(self.root_folder, fn))]

        print(f"number of samples: {len(samples_list)}")

        if datasest_name in ["MotionDataset", "PolyGen_Samples"]:
          # mesh dict list
          mesh_dicts, part_tree = dataset_utils.get_mesh_dict_list(self.root_folder, samples_list, valid_indices=self.instance_nns)
        else:
          # train_valid_indices = ["%.4d" % iii for iii in range(1, 4)]

        #   mesh_dicts, part_tree = dataset_utils.get_mesh_dict_list_multi_part(self.root_folder, part_names=["none_motion"], valid_indices=self.instance_nns)

          # if category_name == "eyeglasses":
          #   part_names = ["none_motion"]
          # else:
          #   part_names=["dof_rootd_Aa001_r", "dof_rootd_Aa002_r"]

          # mesh_dicts, part_tree = dataset_utils.get_mesh_dict_list_multi_part_part_first(self.root_folder, part_names=part_names, valid_indices=self.instance_nns)


          if not isinstance(category_name, list):
            category_name = [category_name] # category name
          
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
            
            print(f"Pretraining dataset, start loading paired data for the category {cur_cat_nm} with parts {part_names}.")
            cur_cat_mesh_dicts, part_tree = dataset_utils.get_mesh_dict_list_multi_part_part_first_pc_cond(cur_root_folder, part_names=part_names, valid_indices=self.instance_nns if cur_cat_nm == "eyeglasses" else None, category_name=cur_cat_nm)

            # cur_sample_vertices, nn_face_indices = cur_cat_mesh_dicts['vertices'], cur_cat_mesh_dicts['faces']
            # if not (cur_sample_vertices.shape[0] <= nn_max_permite_vertices and nn_face_indices <= nn_max_permite_faces):
            #     continue 
            mesh_dicts += cur_cat_mesh_dicts

        # mesh_dicts = []

        # for i_s, fn in enumerate(samples_list): # samples list and summary
        #     if debug and i_s >= 200:
        #       break
        #     cur_s_mesh_fn = os.path.join(self.root_folder, fn, "summary.obj")
        #     if not os.path.exists(cur_s_mesh_fn):
        #       continue
        #     cur_sample_vertices, cur_sample_faces = data_utils.read_obj(cur_s_mesh_fn) # mesh fn
        #     nn_face_indices = sum([len(sub_face) for sub_face in cur_sample_faces])

        #     # print()
        #     if not (cur_sample_vertices.shape[0] <= nn_max_permite_vertices and nn_face_indices <= nn_max_permite_faces):
        #         continue 
        #     ### centralize vertices ###
        #     ins_vert_min = cur_sample_vertices.min(axis=0)
        #     ins_vert_max = cur_sample_vertices.max(axis=0)
        #     ins_vert_center = 0.5 * (ins_vert_min + ins_vert_max)
        #     cur_sample_vertices = cur_sample_vertices - ins_vert_center
        #     ins_vert_min = cur_sample_vertices.min(axis=0)
        #     ins_vert_max = cur_sample_vertices.max(axis=0)
        #     ins_extents = ins_vert_max - ins_vert_min
        #     ins_scale = np.sqrt(np.sum(ins_extents**2))
        #     cur_sample_vertices = cur_sample_vertices / ins_scale

        #     ### get sampling mesh dict ###
        #     cur_sample_mesh_dict = data_utils.process_mesh(cur_sample_vertices, cur_sample_faces, quantization_bits=quantization_bits, recenter_mesh=True)

        #     cur_sample_mesh_dict['class_label'] = 0
        #     mesh_dict = cur_sample_mesh_dict

        #     mesh_dicts.append(mesh_dict)

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
      

    def __getitem__(self, item):

        cur_item_mesh_dict = self.mesh_dicts[item]

        vertices = torch.from_numpy(cur_item_mesh_dict['vertices']).long()
        faces = torch.from_numpy(cur_item_mesh_dict['faces']).long()
        class_label = torch.tensor([cur_item_mesh_dict['class_label']], dtype=torch.long)
        sampled_pts = torch.from_numpy(cur_item_mesh_dict['points']).float()

        # Re-order vertex coordinates as (z, y, x).
        vertices_permuted = torch.cat(
            [vertices[..., 2].unsqueeze(-1), vertices[..., 1].unsqueeze(-1), vertices[..., 0].unsqueeze(-1)], dim=-1
        )

        vertices_flat = vertices_permuted.contiguous().view(-1).contiguous()

        #### ?####
        # vertices_flat_masked, vertices_mask_identifier, vertices_mask_permut = self._generate_vertices_mask(vertices_flat)
        #### ?####


        # vertices_mask_permut = torch.argsort(vertices_mask_identifier, dim=0, descending=True) # identifier
        # 1111 s should be first...
        

        nn_vertices = vertices_flat.size(0)

        nn_faces = faces.size(0) # faces

        ''' minn vertex index value is 1 here, so + 1 can get real vertex indices for faces ''' 
        vertices_flat = torch.cat(
            [vertices_flat + 1, torch.zeros((self.nn_max_vertices * 3 - vertices_flat.size(0),), dtype=torch.long)], dim=-1
        )

        # ''' Permutation '''
        # permut = np.random.permutation(nn_vertices + 1)
        # permut = torch.from_numpy(permut).long() # seq_len
        # res_permut = np.array(range(nn_vertices + 1, self.nn_max_vertices * 3), dtype=np.int32)
        # res_permut = torch.from_numpy(res_permut).long()
        # permut = torch.cat(
        #   [permut, res_permut], dim=0
        # )
        # ''' Permutation '''

        ''' Permutation '''
        permut = np.random.permutation(nn_vertices)
        permut = torch.from_numpy(permut).long() # seq_len
        res_permut = np.array(range(nn_vertices, self.nn_max_vertices * 3), dtype=np.int32)
        res_permut = torch.from_numpy(res_permut).long()
        permut = torch.cat(
          [permut, res_permut], dim=0
        )
        ''' Permutation '''

        
        vertices_flat_mask = torch.cat(
            [torch.ones((nn_vertices + 1,), dtype=torch.float32), torch.zeros((self.nn_max_vertices * 3 - nn_vertices - 1), dtype=torch.float32)], dim=-1
        )

        # dequantized vertices
        real_nn_vertices = vertices.size(0)

        # start_sampling_idx = np.random.randint(0, real_nn_vertices // 2, (1,))
        # start_sampling_idx = torch.from_numpy(start_sampling_idx).long()

        start_sampling_idx = np.random.randint(nn_vertices // 4, nn_vertices // 2, (1,))
        start_sampling_idx = torch.from_numpy(start_sampling_idx).long()

        # #### TODO a more genral version, current one is for sampling only ####
        # # permut, start_sampling_idx, vertices_flat = self.generate_interpolate_mask(vertices=vertices[:vertices.size(0) // 2, :])
        # permut, start_sampling_idx, vertices_flat = self.generate_scaled_verts_mask(vertices=vertices)
        # permut = torch.cat(
        #   [permut, torch.arange(permut.size(0), self.nn_max_vertices * 3, dtype=torch.long)], dim=0
        # )
        # start_sampling_idx = torch.tensor([start_sampling_idx], dtype=torch.long)
        # vertices_flat = torch.cat(
        #     [vertices_flat, torch.zeros((self.nn_max_vertices * 3 - vertices_flat.size(0),), dtype=torch.long)], dim=-1
        # )
        # # vertices_flat_mask = torch.cat(
        # #     [torch.ones((nn_vertices * 2 + 1,), dtype=torch.float32), torch.zeros((self.nn_max_vertices * 3 - nn_vertices * 2 - 1), dtype=torch.float32)], dim=-1
        # # )
        # #### TODO a more genral version, current one is for sampling only ####


        pts_fps_idx = data_utils.farthest_point_sampling(sampled_pts.unsqueeze(0), n_sampling=512)
        sampled_pts = sampled_pts[pts_fps_idx]

        
        # vertices 
        vertices = torch.cat( # 
            [vertices, torch.zeros((self.nn_max_vertices - real_nn_vertices, 3), dtype=torch.long)], dim=0
        )
        
        vertices_mask = torch.cat( # 
            [torch.ones((real_nn_vertices,), dtype=torch.float32), torch.zeros((self.nn_max_vertices - real_nn_vertices,), dtype=torch.float32)], dim=0
        )

        # print(f"item: {item}, nn_left_faces: {nn_left_faces}, nn_rgt_faces: {nn_rgt_faces}")
        # left faces mask
        faces_mask = torch.cat(
            [torch.ones((nn_faces,), dtype=torch.float32), torch.zeros((self.nn_max_faces - nn_faces), dtype=torch.float32)], dim=-1
        )
        
        # faces and faces mask
        # left faces mask
        faces = torch.cat(
            [faces, torch.zeros((self.nn_max_faces - nn_faces), dtype=torch.long)], dim=-1
        )

        rt_dict = {
            'vertices_flat_mask': vertices_flat_mask,
            'vertices_flat': vertices_flat, # flat...
            'faces_mask': faces_mask,
            'vertices': vertices,
            'vertices_mask': vertices_mask,
            'faces': faces,
            'class_label': class_label,
            # 'vertices_flat_masked': vertices_flat_masked,
            # 'vertices_mask_identifier': vertices_mask_identifier,
            'vertices_permutation': permut,
            'start_sampling_idx': start_sampling_idx,
            'sampled_pts': sampled_pts
            # 'vertices_masked_permutation': vertices_mask_permut
        }

        return rt_dict

