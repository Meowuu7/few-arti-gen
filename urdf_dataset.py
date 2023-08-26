import os

import os.path
import json
import numpy as np
import math
import sys
import torch

from torch.utils import data
import data_utils_torch as data_utils


class URDFDatasetTwoParts(data.Dataset):
    def __init__(self, root_folder, part_names, statistic_folder, n_max_instance, quantization_bits=8, nn_max_vertices=8900, nn_max_faces=8900, category_name="eyeglasses",
    ):
        super(URDFDatasetTwoParts, self).__init__()
        self.root_folder = root_folder
        self.part_names = part_names
        self.statistic_folder = statistic_folder
        self.category_name = category_name
        self.quantization_bits = quantization_bits
        self.nn_max_vertices = nn_max_vertices
        self.nn_max_faces = nn_max_faces

        # part instance names of the current category
        self.cat_folder_name_list = []
        for part_nm in self.part_names:
            cur_cat_cur_part_folder = os.path.join(self.root_folder, part_nm)
            cur_cat_cur_part_ins_list = os.listdir(cur_cat_cur_part_folder)
            cur_cat_cur_part_ins_list = [fn for fn in cur_cat_cur_part_ins_list if fn.endswith(".obj")]
            cur_cat_cur_part_ins_list = sorted(cur_cat_cur_part_ins_list, reverse=False)
            cur_cat_cur_part_ins_list = cur_cat_cur_part_ins_list[:n_max_instance]
            # add current part's instance name list to the category part instance name list
            self.cat_folder_name_list.append(cur_cat_cur_part_ins_list)

        self.stat_folder_name_list = os.listdir(statistic_folder)
        self.stat_folder_name_list = [fn for fn in self.stat_folder_name_list if fn.endswith(".npy")]
        self.stat_folder_name_list = sorted(self.stat_folder_name_list)
        
        # self.stat_folder_name_list = sorted(statistic_folder)

        self.stat_folder_name_list = self.stat_folder_name_list[:n_max_instance]
        while len(self.stat_folder_name_list) < n_max_instance:
            self.stat_folder_name_list = self.stat_folder_name_list + self.stat_folder_name_list
        self.stat_folder_name_list = self.stat_folder_name_list[:n_max_instance]

        ### n_part; n_mesh
        tot_n_part = len(self.cat_folder_name_list)  # number of parts of this category
        tot_n_mesh = len(self.cat_folder_name_list[0])  # number of instance of this category/dataset
        tot_part_ex_list = [[] for i_part in range(tot_n_part)]  # tot_part_ex_list
        # tot_part_ex_list;
        stat_list = []
        # statistic list for statistic folder
        for i_ins in range(tot_n_mesh):
            ins_vertices = []
            ins_faces = []
            for i_p in range(tot_n_part):
                cur_part_mesh_fn = os.path.join(root_folder, self.part_names[i_p], self.cat_folder_name_list[i_p][i_ins])
                cur_part_vertices, cur_part_faces = data_utils.read_obj(cur_part_mesh_fn)
                ins_vertices.append(cur_part_vertices)
                ins_faces.append(cur_part_faces)
            ### get instance vertices ###
            cur_ins_vertices = np.concatenate(ins_vertices, axis=0)
            ### get vertices' center and scale for further recentering and normalizing ###
            ins_vert_min = cur_ins_vertices.min(axis=0)  # vert_min
            ins_vert_max = cur_ins_vertices.max(axis=0)  # vert_max
            ins_vert_center = 0.5 * (ins_vert_min + ins_vert_max)
            cur_ins_vertices = cur_ins_vertices - ins_vert_center
            ins_vert_min = cur_ins_vertices.min(axis=0)  # vert_min
            ins_vert_max = cur_ins_vertices.max(axis=0)  # vert_max
            ins_extents = ins_vert_max - ins_vert_min  # extents
            ins_scale = np.sqrt(np.sum(ins_extents ** 2))  # scale #

            # instance statistic folder name list
            cur_ins_stat_fn = os.path.join(self.statistic_folder, self.stat_folder_name_list[i_ins])
            cur_ins_stat = np.load(cur_ins_stat_fn, allow_pickle=True).item()
            # stat_list.append(cur_ins_stat)

            for i_p in range(tot_n_part):
                cur_part_vertices, cur_part_faces = ins_vertices[i_p], ins_faces[i_p]
                cur_part_vertices = cur_part_vertices - ins_vert_center
                cur_part_vertices = cur_part_vertices / ins_scale
                mesh_dict = data_utils.process_mesh(cur_part_vertices, cur_part_faces,
                                                    quantization_bits=self.quantization_bits, recenter_mesh=False)
                mesh_dict['class_label'] = 0
                tot_part_ex_list[i_p].append(mesh_dict)
            # Get pivot point and instance vertices center
            cur_pvp = cur_ins_stat['pvp']
            cur_pvp = cur_pvp - ins_vert_center
            cur_pvp = cur_pvp / ins_scale
            cur_pvp = cur_pvp[[2, 0, 1]]
            print("ins name:", self.cat_folder_name_list[0][i_ins], "current ins:", i_ins, "current pvp:", cur_pvp,
                  "ins_vert_center:", ins_vert_center)
            cur_ins_stat['pvp'] = cur_pvp
            stat_list.append(cur_ins_stat)

        self.stat_list = stat_list  # statistic list
        self.tot_part_ex_list = tot_part_ex_list  # statistic ex list

        ### total object mesh dictionary list ###
        tot_obj_ex_list = []
        for i_mesh in range(tot_n_mesh):
            left_part_dict, rgt_part_dict = self.tot_part_ex_list[0][i_mesh], self.tot_part_ex_list[1][i_mesh]
            stat_dict = self.stat_list[i_mesh]
            obj_mesh_dict = {
                'left_vertices': left_part_dict['vertices'],
                'left_faces': left_part_dict['faces'], # left faces
                'rgt_vertices': rgt_part_dict['vertices'],
                'rgt_faces': rgt_part_dict['faces'],
                'class_label': 0,
                'dir': stat_dict['dir'],
                'pvp': stat_dict['pvp']
            }

            tot_obj_ex_list.append(obj_mesh_dict)  # total obj_ex_list
            ##### Get mesh for the object with part information #####
        self.tot_obj_ex_list = tot_obj_ex_list
        ### total object mesh dictionary list ###

        self.dataset_len = tot_n_mesh

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):

        cur_item_mesh_dict = self.tot_obj_ex_list[item]

        left_vertices = torch.from_numpy(cur_item_mesh_dict['left_vertices']).long()
        rgt_vertices = torch.from_numpy(cur_item_mesh_dict['rgt_vertices']).long()
        left_faces = torch.from_numpy(cur_item_mesh_dict['left_faces']).long()
        rgt_faces = torch.from_numpy(cur_item_mesh_dict['rgt_faces']).long()
        class_label = torch.tensor([cur_item_mesh_dict['class_label']], dtype=torch.long)
        dir = torch.from_numpy(cur_item_mesh_dict['dir']).float()
        pvp = torch.from_numpy(cur_item_mesh_dict['pvp']).float()

        # Re-order vertex coordinates as (z, y, x).
        left_vertices_permuted = torch.cat(
            [left_vertices[..., 2].unsqueeze(-1), left_vertices[..., 1].unsqueeze(-1), left_vertices[..., 0].unsqueeze(-1)], dim=-1
        )
        rgt_vertices_permuted = torch.cat(
            [rgt_vertices[..., 2].unsqueeze(-1), rgt_vertices[..., 1].unsqueeze(-1), rgt_vertices[..., 0].unsqueeze(-1)], dim=-1
        )
        left_vertices_flat = left_vertices_permuted.contiguous().view(-1).contiguous()
        rgt_vertices_flat = rgt_vertices_permuted.contiguous().view(-1).contiguous()

        nn_left_vertices = left_vertices_flat.size(0)
        nn_rgt_vertices = rgt_vertices_flat.size(0)

        nn_left_faces = left_faces.size(0)
        nn_rgt_faces = rgt_faces.size(0)

        # print(f"item: {item}, nn_left_vertices: {nn_left_vertices}, nn_rgt_vertices: {nn_rgt_vertices}")

        # vertex indices... 

        # print(f"max left vertices flat: {torch.max(left_vertices_flat)}, min left vertices flat: {torch.min(left_vertices_flat)}")

        # print(f"max rgt vertices flat: {torch.max(rgt_vertices_flat)}, min rgt vertices flat: {torch.min(rgt_vertices_flat)}")


        ''' minn vertex index value is 1 here, so + 1 can get real vertex indices for faces ''' 
        left_vertices_flat = torch.cat(
            [left_vertices_flat + 1, torch.zeros((self.nn_max_vertices - left_vertices_flat.size(0),), dtype=torch.long)], dim=-1
        )
        rgt_vertices_flat = torch.cat(
            [rgt_vertices_flat + 1, torch.zeros((self.nn_max_vertices - rgt_vertices_flat.size(0)), dtype=torch.long)], dim=-1
        )

        
        left_vertices_flat_mask = torch.cat(
            [torch.ones((nn_left_vertices + 1,), dtype=torch.float32), torch.zeros((self.nn_max_vertices - nn_left_vertices - 1), dtype=torch.float32)], dim=-1
        )
        rgt_vertices_flat_mask = torch.cat(
            [torch.ones((nn_rgt_vertices + 1), dtype=torch.float32), torch.zeros((self.nn_max_vertices  - nn_rgt_vertices - 1), dtype=torch.float32)], dim=-1
        )

        real_nn_max_vertices = self.nn_max_vertices // 3
        real_nn_left_vertices = left_vertices.size(0)
        left_vertices = torch.cat(
            [left_vertices + 1, torch.zeros((real_nn_max_vertices - real_nn_left_vertices, 3), dtype=torch.long)], dim=0
        )
        # real_nn_r
        real_nn_rgt_vertices = rgt_vertices.size(0)
        rgt_vertices = torch.cat(
            [rgt_vertices + 1, torch.zeros((real_nn_max_vertices - real_nn_rgt_vertices, 3), dtype=torch.long)], dim=0
        )
        # 
        left_vertices_mask = torch.cat(
            [torch.ones((real_nn_left_vertices,), dtype=torch.float32), torch.zeros((real_nn_max_vertices - real_nn_left_vertices,), dtype=torch.float32)], dim=0
        )
        rgt_vertices_mask = torch.cat(
            [torch.ones((real_nn_rgt_vertices,), dtype=torch.float32), torch.zeros((real_nn_max_vertices - real_nn_rgt_vertices), dtype=torch.float32)], dim=0
        )

        # print(f"item: {item}, nn_left_faces: {nn_left_faces}, nn_rgt_faces: {nn_rgt_faces}")
        # left faces mask
        left_faces_mask = torch.cat(
            [torch.ones((nn_left_faces,), dtype=torch.float32), torch.zeros((self.nn_max_faces - nn_left_faces), dtype=torch.float32)], dim=-1
        )
        rgt_faces_mask = torch.cat(
            [torch.ones((nn_rgt_faces,), dtype=torch.float32), torch.zeros((self.nn_max_faces - nn_rgt_faces), dtype=torch.float32)], dim=-1
        )

        # print(f"item: {item}, max of left faces: {torch.max(left_faces).item()}, min of left faces: {torch.min(left_faces).item()}")
        # print(f"item: {item}, max of rgt faces: {torch.max(rgt_faces).item()}, min of rgt faces: {torch.min(rgt_faces).item()}")

        # left faces mask
        left_faces = torch.cat(
            [left_faces, torch.zeros((self.nn_max_faces - nn_left_faces), dtype=torch.long)], dim=-1
        )
        rgt_faces = torch.cat(
            [rgt_faces, torch.zeros((self.nn_max_faces - nn_rgt_faces), dtype=torch.long)], dim=-1
        )

        

        rt_dict = {
            'left_vertices_flat_mask': left_vertices_flat_mask,
            'rgt_vertices_flat_mask': rgt_vertices_flat_mask,
            'left_vertices_flat': left_vertices_flat,
            'rgt_vertices_flat': rgt_vertices_flat,
            'left_faces_mask': left_faces_mask,
            'rgt_faces_mask': rgt_faces_mask,
            'left_vertices': left_vertices,
            'rgt_vertices': rgt_vertices,
            'left_vertices_mask': left_vertices_mask,
            'rgt_vertices_mask': rgt_vertices_mask,
            'left_faces': left_faces,
            'rgt_faces': rgt_faces,
            'class_label': class_label,
            'dir': dir,
            'pvp': pvp
        }

        # for k in rt_dict:
        #     print(item, k, rt_dict[k].size())

        return rt_dict



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

