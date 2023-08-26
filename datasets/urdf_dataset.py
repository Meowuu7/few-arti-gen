import os

import os.path
import json
import numpy as np
import math
import sys
import torch

from torch.utils import data
import utils.data_utils_torch as data_utils
from options.options import opt



class URDFDataset(data.Dataset):
    def __init__(self, root_folder, quantization_bits=8, nn_max_vertices=8900, nn_max_faces=8900, nn_max_permite_vertices=400, nn_max_permite_faces=2000, category_name="eyeglasses",
    ):
        super(URDFDataset, self).__init__()
        self.root_folder = root_folder
        # self.part_names = part_names # part names
        # self.statistic_folder = statistic_folder
        self.category_name = category_name # category 
        self.quantization_bits = quantization_bits 
        self.nn_max_vertices = nn_max_vertices + 1
        self.nn_max_faces = nn_max_faces
        self.apply_random_shift = opt.dataset.apply_random_shift


        samples_list = os.listdir(self.root_folder)
        samples_list = [fn for fn in samples_list if os.path.isdir(os.path.join(self.root_folder, fn))]

        print(f"number of samples: {len(samples_list)}")

        mesh_dicts = []

        for i_s, fn in enumerate(samples_list):
            cur_s_mesh_fn = os.path.join(self.root_folder, fn, "summary.obj")
            if not os.path.exists(cur_s_mesh_fn):
                continue
            cur_sample_vertices, cur_sample_faces = data_utils.read_obj(cur_s_mesh_fn)
            nn_face_indices = sum([len(sub_face) for sub_face in cur_sample_faces])

            # print()
            if not (cur_sample_vertices.shape[0] <= nn_max_permite_vertices and nn_face_indices <= nn_max_permite_faces):
                continue 
            ### centralize vertices ###
            ins_vert_min = cur_sample_vertices.min(axis=0)
            ins_vert_max = cur_sample_vertices.max(axis=0)
            ins_vert_center = 0.5 * (ins_vert_min + ins_vert_max)
            cur_sample_vertices = cur_sample_vertices - ins_vert_center
            ins_vert_min = cur_sample_vertices.min(axis=0)
            ins_vert_max = cur_sample_vertices.max(axis=0)
            ins_extents = ins_vert_max - ins_vert_min
            ins_scale = np.sqrt(np.sum(ins_extents**2))
            cur_sample_vertices = cur_sample_vertices / ins_scale

            ### get sampling mesh dict ###
            cur_sample_mesh_dict = data_utils.process_mesh(cur_sample_vertices, cur_sample_faces, quantization_bits=quantization_bits, recenter_mesh=True)

            cur_sample_mesh_dict['class_label'] = 0
            mesh_dict = cur_sample_mesh_dict

            mesh_dicts.append(mesh_dict)

        print(f"numbers of valid samples: {len(mesh_dicts)}")

        tot_n_mesh = len(mesh_dicts)
        self.mesh_dicts = mesh_dicts

        self.dataset_len = tot_n_mesh

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):

        cur_item_mesh_dict = self.mesh_dicts[item]

        vertices = torch.from_numpy(cur_item_mesh_dict['vertices']).long()
        if self.apply_random_shift:
            vertices = data_utils.random_shift(vertices)
        faces = torch.from_numpy(cur_item_mesh_dict['faces']).long()
        class_label = torch.tensor([cur_item_mesh_dict['class_label']], dtype=torch.long)

        # Re-order vertex coordinates as (z, y, x).
        vertices_permuted = torch.cat(
            [vertices[..., 2].unsqueeze(-1), vertices[..., 1].unsqueeze(-1), vertices[..., 0].unsqueeze(-1)], dim=-1
        )

        vertices_flat = vertices_permuted.contiguous().view(-1).contiguous()

        nn_vertices = vertices_flat.size(0)

        nn_faces = faces.size(0)

        ''' minn vertex index value is 1 here, so + 1 can get real vertex indices for faces ''' 
        vertices_flat = torch.cat(
            [vertices_flat + 1, torch.zeros((self.nn_max_vertices * 3 - vertices_flat.size(0),), dtype=torch.long)], dim=-1
        )
        
        vertices_flat_mask = torch.cat(
            [torch.ones((nn_vertices + 1,), dtype=torch.float32), torch.zeros((self.nn_max_vertices * 3 - nn_vertices - 1), dtype=torch.float32)], dim=-1
        )

        # dequantized vertices
        real_nn_vertices = vertices.size(0)
        vertices = torch.cat( # 
            [vertices, torch.zeros((self.nn_max_vertices - real_nn_vertices, 3), dtype=torch.long)], dim=0
        )
        
        vertices_mask = torch.cat(
            [torch.ones((real_nn_vertices,), dtype=torch.float32), torch.zeros((self.nn_max_vertices - real_nn_vertices,), dtype=torch.float32)], dim=0
        )

        # print(f"item: {item}, nn_left_faces: {nn_left_faces}, nn_rgt_faces: {nn_rgt_faces}")
        # left faces mask
        faces_mask = torch.cat(
            [torch.ones((nn_faces,), dtype=torch.float32), torch.zeros((self.nn_max_faces - nn_faces), dtype=torch.float32)], dim=-1
        )

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
        }

        # for k in rt_dict:
        #     print(item, k, rt_dict[k].size())

        return rt_dict

