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


        if datasest_name in ["MotionDataset", "PolyGen_Samples"]:
          samples_list = os.listdir(self.root_folder)
          samples_list = [fn for fn in samples_list if os.path.isdir(os.path.join(self.root_folder, fn))]

          print(f"number of samples: {len(samples_list)}")

          mesh_dicts, part_tree = dataset_utils.get_mesh_dict_list(self.root_folder, samples_list, valid_indices=self.instance_nns)
        else:
          # train_valid_indices = ["%.4d" % iii for iii in range(1, 4)]
          # mesh_dicts, part_tree = dataset_utils.get_mesh_dict_list_multi_part(self.root_folder, part_names=["none_motion"], valid_indices=self.instance_nns)

          if not isinstance(category_name, list):
            category_name = [category_name]
          
          mesh_dicts = []
          for cur_cat_nm in category_name:

            cur_root_folder = os.path.join(opt.dataset.dataset_root_path, opt.dataset.dataset_name, cur_cat_nm)
            if category_nm_to_part_nm is None:
              if category_name == "eyeglasses":
                part_names = ["none_motion"]
              else:
                part_names=["dof_rootd_Aa001_r", "dof_rootd_Aa002_r"]
            else:
              part_names = category_nm_to_part_nm[cur_cat_nm]
            

            cur_cat_mesh_dicts, part_tree = dataset_utils.get_mesh_dict_list_multi_part_part_first(cur_root_folder, part_names=part_names, valid_indices=self.instance_nns, category_name=cur_cat_nm)
            
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
    def _generate_vertices_mask(self, vertices_flat):

      vertices_flat_permut = torch.arange(0, vertices_flat.size(0), dtype=torch.long)
      nn_vertices_no_flat = vertices_flat.size(0) // 3
      vertices_flat_permut = torch.cat(
        [vertices_flat_permut[:3], vertices_flat_permut[-3:], vertices_flat_permut[3: -3]], dim=0
      )
      vertices_mask_identifier = torch.zeros_like(vertices_flat, dtype=torch.float32)
      vertices_mask_identifier[3: -3] = 1. # for prediction

      vertices_flat_masked = torch.zeros_like(vertices_flat)
      vertices_flat_masked[:3] = vertices_flat[:3] + 1
      vertices_flat_masked[-3:] = vertices_flat[-3:] + 1
      vertices_flat_masked = torch.cat([vertices_flat_masked, torch.zeros((self.nn_max_vertices * 3 - vertices_flat_masked.size(0)), dtype=torch.long)], dim=0)

      

      return vertices_flat_permut, vertices_mask_identifier, vertices_flat_masked
      
      


    def __getitem__(self, item):

        cur_item_mesh_dict = self.mesh_dicts[item]

        vertices = torch.from_numpy(cur_item_mesh_dict['vertices']).long()
        faces = torch.from_numpy(cur_item_mesh_dict['faces']).long()
        class_label = torch.tensor([cur_item_mesh_dict['class_label']], dtype=torch.long)

        # Re-order vertex coordinates as (z, y, x).
        vertices_permuted = torch.cat(
            [vertices[..., 2].unsqueeze(-1), vertices[..., 1].unsqueeze(-1), vertices[..., 0].unsqueeze(-1)], dim=-1
        )

        vertices_flat = vertices_permuted.contiguous().view(-1).contiguous()

        # masked vertices permutation... --- 
        # sample the context vertices for other vertices' generation
        # vertices_flat_masked, vertices_mask_identifier, vertices_mask_permut = self._generate_vertices_mask(vertices_flat)
        # vertices_mask_permut = torch.argsort(vertices_mask_identifier, dim=0, descending=True) # identifier
        # 1111 s should be first...

        permut, vertices_mask_identifier, vertices_flat_masked = self._generate_vertices_mask(vertices_flat)
        

        nn_vertices = vertices_flat.size(0)

        nn_faces = faces.size(0) # faces

        ''' minn vertex index value is 1 here, so + 1 can get real vertex indices for faces ''' 
        # vertices_flat
        vertices_flat = torch.cat(
            [vertices_flat + 1, torch.zeros((self.nn_max_vertices * 3 - vertices_flat.size(0),), dtype=torch.long)], dim=-1
        )

        ''' Permutation '''
        # permut = np.random.permutation(nn_vertices + 1)
        # permut = torch.from_numpy(permut).long() # seq_len
        res_permut = np.array(range(nn_vertices, self.nn_max_vertices * 3), dtype=np.int32)
        res_permut = torch.from_numpy(res_permut).long()
        permut = torch.cat(
          [permut, res_permut], dim=0
        )
        ''' Permutation '''

        
        # vertices_flat_mask = torch.cat(
            # [torch.ones((nn_vertices + 1,), dtype=torch.float32), torch.zeros((self.nn_max_vertices * 3 - nn_vertices - 1), dtype=torch.float32)], dim=-1
        # )

        # get
        vertices_flat_mask = torch.cat( # identifier + the ending symbol
          [vertices_mask_identifier, torch.ones((1,), dtype=torch.float32), torch.zeros((self.nn_max_vertices * 3 - nn_vertices - 1), dtype=torch.float32)], dim=-1
        )

        # vertices mask identifier
        # mask identifier
        # vertices_mask_identifier = torch.cat(
        #   [1. - vertices_mask_identifier, torch.ones((1,), dtype=torch.float32), torch.zeros((self.nn_max_vertices * 3 - nn_vertices - 1), dtype=torch.float32)], dim=-1
        # )

        vertices_mask_identifier = torch.cat(
          [1. - vertices_mask_identifier, torch.zeros((1,), dtype=torch.float32), torch.zeros((self.nn_max_vertices * 3 - nn_vertices - 1), dtype=torch.float32)], dim=-1
        )


        ##### Process data for face model inputs #####
        # dequantized vertices
        real_nn_vertices = vertices.size(0)
        # vertices 
        vertices = torch.cat( # 
            [vertices, torch.zeros((self.nn_max_vertices - real_nn_vertices, 3), dtype=torch.long)], dim=0
        )
        
        vertices_mask = torch.cat( # 
            [torch.ones((real_nn_vertices,), dtype=torch.float32), torch.zeros((self.nn_max_vertices - real_nn_vertices,), dtype=torch.float32)], dim=0
        )
        
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
            'vertices_flat_masked': vertices_flat_masked,
            'vertices_mask_identifier': vertices_mask_identifier,
            'vertices_permutation': permut,
            'vertices_masked_permutation': permut
        }

        return rt_dict

