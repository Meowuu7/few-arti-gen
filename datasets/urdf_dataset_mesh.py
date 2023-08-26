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

        self.padding_nn_vertices = self.nn_max_vertices
        self.padding_nn_faces = self.nn_max_faces + self.nn_max_faces // 3 + 300 ### 


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
        print("Here!")
        for cur_cat_nm in category_name: # category na
            ###### root folder of the current category ######
            cur_root_folder = os.path.join(opt.dataset.dataset_root_path, opt.dataset.dataset_name, cur_cat_nm)

            print(f"Loading part objs from folder {cur_root_folder}...") # load part objs # load part objs
            # if category_nm_to_part_nm is None:
            #     if cur_cat_nm == "eyeglasses":
            #         part_names = ["none_motion"]
            #     else:
            #         part_names = ["dof_rootd_Aa001_r", "dof_rootd_Aa002_r"]
            # else:
            #     part_names = category_nm_to_part_nm[cur_cat_nm]

            read_meta_info_func = dataset_utils.get_mesh_meta_info
            cur_cat_mesh_dicts, part_tree, part_nm_to_mesh_dict_idxes = read_meta_info_func(
              cur_root_folder, split=self.split)
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

        # if self.balance_classes:
        #     self.category_name_to_mesh_dict_idxes, self.balanced_mesh_dicts = dataset_utils.balance_class_idxes(self.category_part_indicator_to_mesh_dict_idxes)

        print(f"number of valid samples: {len(mesh_dicts)}")

        self.balance_classes = False

        tot_n_mesh = len(mesh_dicts) # if not self.balance_classes else len(self.balanced_mesh_dicts) #### balanced mesh dicts
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
      dequan_verts_gt, dequan_verts_upsample = data_utils.dequantize_verts(verts_gt, n_bits=self.quantization_bits), data_utils.dequantize_verts(verts_upsample, n_bits=self.quantization_bits)
      center_dequan_verts_gt = data_utils.get_vertices_center(dequan_verts_gt)
      dequan_verts_gt = dequan_verts_gt - center_dequan_verts_gt
      dequan_verts_upsample = dequan_verts_upsample - center_dequan_verts_gt
      verts_gt = data_utils.quantize_verts(dequan_verts_gt, n_bits=self.quantization_bits)
      verts_upsample = data_utils.quantize_verts(dequan_verts_upsample, n_bits=self.quantization_bits)
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

    ''' Load data with random axises flipping and random vertices shifting  '''
    def __getitem__(self, item):
        ###### num objects ######
        ###### num_objects ######
        # if not self.is_training:
        #   item = 0
        if self.balance_classes:
            item = self.balanced_mesh_dicts[item] # get new mesh dict index of the current item

        items = [item] ### items 

        for cur_item in items:
          if not isinstance(self.mesh_dicts[cur_item], dict):
            ### read mesh from obj file ####
            self.mesh_dicts[cur_item] = data_utils.read_mesh_from_obj_file(self.mesh_dicts[cur_item], quantization_bits=self.quantization_bits, recenter_mesh=True, remove_du=False)
              
          #     pts_idxes = self.mesh_idx_to_part_pts_idxes[cur_item]

        # category_part_nm indicat  or... # indicator
        # category_part_nm_indicator = self.mesh_dict_idx_to_category_name[item]
        # category_part_idx = self.category_part_indicator_to_idxes[category_part_nm_indicator]

        cur_item_mesh_dict = self.mesh_dicts[item]
        # grid xyz coordinates and grid contents


        #### 
        dequant_vertices = cur_item_mesh_dict['dequant_vertices'] # dequant
        vertices = cur_item_mesh_dict['vertices']
        faces = cur_item_mesh_dict['faces']
        # unflatten_faces = cur_item_mesh_dict['unflatten_faces']

        dequant_vertices = torch.from_numpy(dequant_vertices).float() # dequant_vertices
        vertices = torch.from_numpy(vertices).long()
        faces = torch.from_numpy(faces).long()
        # unflatten_faces = torch.from_numpy(unflatten_faces).long()
        
        # nn_vertices = vertices.size(0)
        # nn_flatten_faces = faces.size(0)
        # # nn_unflatten_faces = unflatten_faces.size(0)

        # # vertices = vertices + 1

        # nn_max_vertices = self.nn_max_vertices
        # nn_max_faces = self.nn_max_faces
        
        # dequant_vertices = self.permute_vertices(dequant_vertices)
        vertices = self.permute_vertices(vertices)
        dequant_vertices = self.permute_vertices(dequant_vertices) ### permute vertices here...
        
        ar_quant_vertices = vertices
        ar_vertices = dequant_vertices
        ar_faces = faces
        # ar_faces_unflatten = unflatten_faces
        

        # ar_quant_vertices = data_utils.quantize_verts(ar_vertices, n_bits=self.quantization_bits) # n_verts x 3
        # ar_quant_vertices = torch.from_numpy(ar_quant_vertices).long()

        # cur_subd_mesh_sv_fn = "ar_subd_mesh.obj"
        # data_utils.write_obj(ar_vertices, None, cur_subd_mesh_sv_fn, transpose=True, scale=1.)
        
        # ar_vertices = torch.from_numpy(ar_vertices).float() ## nn_verts x 3

        nn_valid_vertices = ar_vertices.size(0)
        ar_stop_vertices = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32).unsqueeze(0)
        ar_vertices = torch.cat(
          [ar_vertices, ar_stop_vertices], dim=0
        ) ## (nn_verts + 1) x 3

        #### ar_vertices: padding_nn_vertices x 3 ####
        ar_vertices = torch.cat(
          [ar_vertices, torch.zeros((self.padding_nn_vertices - ar_vertices.size(0), 3), dtype=torch.float32)], dim=0
        )

        ar_stop_vertices = torch.tensor([0, 0, 0], dtype=torch.long).unsqueeze(0)
        ar_quant_vertices = torch.cat(
          [ar_quant_vertices + 1, ar_stop_vertices], dim=0 #### (nn_verts + 1) x 3
        )

        #### ar_quant_verticeS: padding_nn_vertices x 3 ####
        #### ar_quant_vertices ####
        ar_quant_vertices = torch.cat(
          [ ar_quant_vertices, torch.zeros((self.padding_nn_vertices - ar_quant_vertices.size(0), 3), dtype=torch.long) ], dim=0
        )

        # ar_faces = torch.tensor(ar_faces, dtype=torch.long)


        # vertices_mask = torch.ones((ar_vertices.size(0) - 1, ), dtype=torch.float32) ## nn_verts,
        vertices_mask = torch.cat(
          [torch.ones((nn_valid_vertices, ), dtype=torch.float32), torch.zeros((self.padding_nn_vertices - nn_valid_vertices,), dtype=torch.float32)], dim=0
        ) 


        # quant_vertices_mask
        quant_vertices_mask = torch.cat(
          [torch.ones((nn_valid_vertices, 3), dtype=torch.float32), torch.tensor([1., 0., 0.], dtype=torch.float32).unsqueeze(0), torch.zeros((self.padding_nn_vertices - nn_valid_vertices - 1, 3), dtype=torch.float32)], dim=0
        )



        nn_valid_faces = ar_faces.size(0)
        ar_faces = torch.cat(
          [ar_faces, torch.zeros((self.padding_nn_faces - nn_valid_faces, ), dtype=torch.long)], dim=0
        )
        # faces_mask = torch.ones((ar_faces.size(0), ), dtype=torch.float32)
        faces_mask = torch.cat( #### 
          [torch.ones((nn_valid_faces, ), dtype=torch.float32), torch.zeros((self.padding_nn_faces - nn_valid_faces, ), dtype=torch.float32)], dim=0
        )

        
        

        rt_dict = {}
        rt_dict['vertices'] = ar_vertices #### v
        # rt_dict['quant_vertices'] = data_utils.
        # rt_dict['vertices_faces'] =     ar_vertices[:-1, :] ##### nn_verts x 3
        rt_dict['vertices_mask'] = vertices_mask ## (nn_verts + 1,) --> vertices_mask
        rt_dict['quant_vertices'] = ar_quant_vertices
        rt_dict['quant_vertices_mask'] = quant_vertices_mask

        rt_dict['faces'] = ar_faces ## (nn_faces_flatten,)
        # rt_dict['faces_unflatten'] = torch.tensor(ar_faces_unflatten, dtype=torch.long) ## nn_faces x 3 --> triangle faces
        rt_dict['faces_mask'] = faces_mask
        

        rt_dict['class_label'] = torch.zeros((1,), dtype=torch.long)
        # print(f"rt_dict_keys: {rt_dict.keys()}")
 

        rt_dict = rt_dict

        return rt_dict

