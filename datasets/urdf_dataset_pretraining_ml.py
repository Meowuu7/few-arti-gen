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
from utils.constants import *



class URDFDataset(data.Dataset):
    def __init__(self, root_folder, quantization_bits=8, nn_max_vertices=8900, nn_max_faces=8900,
                 nn_max_permite_vertices=400, nn_max_permite_faces=2000, category_name="eyeglasses", instance_nns=[],
                 part_names=None, category_nm_to_part_nm=None, is_training=False, split=None
                 ):
        super(URDFDataset, self).__init__()

        self.root_folder = root_folder
        # self.part_names = part_names # part names
        # self.statistic_folder = statistic_folder # statistic_folder
        self.category_name = category_name  # category
        self.quantization_bits = quantization_bits
        self.nn_max_vertices = nn_max_vertices + 1
        self.nn_max_faces = nn_max_faces
        self.instance_nns = instance_nns
        self.is_training = is_training
        self.mode = opt.common.exp_mode
        self.apply_random_flipping = opt.dataset.apply_random_flipping
        self.random_scaling = opt.dataset.apply_random_scaling

        self.apply_random_shift = opt.dataset.apply_random_shift
        self.category_part_indicator = opt.dataset.category_part_indicator

        debug = opt.model.debug
        datasest_name = opt.dataset.dataset_name

        self.nn_vertices_predict_ratio = opt.dataset.nn_vertices_predict_ratio
        self.cut_vertices = opt.model.cut_vertices
        self.context_window = opt.model.context_window
        self.remove_du = not opt.dataset.not_remove_du
        self.split = split

        print(f"Current split: {self.split}")


        category_part_indicator_to_idxes = {}

        # context_design_strategy = opt.model.context_design_strategy

        samples_list = os.listdir(self.root_folder)
        samples_list = [fn for fn in samples_list if os.path.isdir(os.path.join(self.root_folder, fn))]

        print(f"number of samples: {len(samples_list)}")

        if datasest_name in ["MotionDataset", "PolyGen_Samples"]:
            mesh_dicts, part_tree = dataset_utils.get_mesh_dict_list(self.root_folder, samples_list,
                                                                     valid_indices=self.instance_nns)
        else:
            if not isinstance(category_name, list):
                category_name = [category_name]  # category name

            mesh_dicts = []
            mesh_dict_idx_to_category_name = {}
            category_name_to_mesh_dict_idxes = {}
            for cur_cat_nm in category_name:
                cur_root_folder = os.path.join(opt.dataset.dataset_root_path, opt.dataset.dataset_name, cur_cat_nm)
                print(f"Loading part objs from folder {cur_root_folder}...") # load part objs
                if category_nm_to_part_nm is None:
                    if cur_cat_nm == "eyeglasses":
                        part_names = ["none_motion"]
                    else:
                        part_names = ["dof_rootd_Aa001_r", "dof_rootd_Aa002_r"]
                else:
                    part_names = category_nm_to_part_nm[cur_cat_nm]

                print(f"Pretraining dataset, start loading data for the category {cur_cat_nm} with parts {part_names}.")
                cur_cat_mesh_dicts, part_tree, part_nm_to_mesh_dict_idxes = dataset_utils.get_mesh_dict_list_multi_part_part_first(
                    cur_root_folder, part_names=part_names,
                    valid_indices=self.instance_nns if cur_cat_nm == "eyeglasses" else None, category_name=cur_cat_nm,
                    ret_part_name_to_mesh_idx=True, remove_du=self.remove_du, split=self.split)

                cur_mesh_dict_len = len(mesh_dicts)
                cur_cat_nn = 0
                for cur_cat_cur_part_nm in part_nm_to_mesh_dict_idxes:
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

                    part_nm_to_mesh_dict_idxes[cur_cat_cur_part_nm] = new_mesh_dict_idxes
                    # part_nm_to_mesh

                category_name_to_mesh_dict_idxes[cur_cat_nm] = part_nm_to_mesh_dict_idxes
                # cur_category_mesh_dict_idxes = range(cur_mesh_dict_len, cur_mesh_dict_len + len(cur_cat_mesh_dicts))
                # mesh_dicts += cur_cat_mesh_dicts

                # cur_sample_vertices, nn_face_indices = cur_cat_mesh_dicts['vertices'], cur_cat_mesh_dicts['faces']
                # if not (cur_sample_vertices.shape[0] <= nn_max_permite_vertices and nn_face_indices <= nn_max_permite_faces):
                #     continue
                mesh_dicts += cur_cat_mesh_dicts

            self.category_name_to_mesh_dict_idxes = category_name_to_mesh_dict_idxes
            self.mesh_dict_idx_to_category_name = mesh_dict_idx_to_category_name
            self.category_part_indicator_to_idxes = category_part_indicator_to_idxes
            

        print(f"numbers of valid samples: {len(mesh_dicts)}")

        tot_n_mesh = len(mesh_dicts)
        self.mesh_dicts = mesh_dicts

        self.dataset_len = tot_n_mesh

    def __len__(self):
        return self.dataset_len

    def _generate_vertices_mask_ml(self, vertices_flat):
      nn_vertices_flat = vertices_flat.size(0)
      nn_vertices_unflat = nn_vertices_flat // 3
      vertices_unflat = vertices_flat.contiguous().view(nn_vertices_unflat, 3).contiguous()
      mask_ratio_low = opt.dataset.mask_low_ratio
      mask_ratio_high = opt.dataset.mask_high_ratio
      mask_ratio = np.random.uniform(low=mask_ratio_low, high=mask_ratio_high, size=(1,)).item()
      mask_nn_unflat = int(mask_ratio * nn_vertices_unflat)
      mask_nn_unflat = max(mask_nn_unflat, 1)
      
      sampled_mask_vertices_indices = np.random.choice(nn_vertices_unflat, size=mask_nn_unflat, replace=False)

    #   sampled_mask_vertices_indices = np.arange(nn_vertices_unflat // 2, nn_vertices_unflat, dtype=np.long)
      sampled_mask_vertices_indices = torch.from_numpy(sampled_mask_vertices_indices).long()  # maksed coord

      mask_vertices_unflat = vertices_unflat.clone()
      mask_vertices_unflat[sampled_mask_vertices_indices, :] = MASK_COORD_VALUE # 2 ** self.quantization_bits # ** 2
      mask_vertices_unflat_indicator = torch.zeros_like(mask_vertices_unflat).float()
      mask_vertices_unflat_indicator[sampled_mask_vertices_indices, :] = 1.
      mask_vertices_flat = mask_vertices_unflat.contiguous().view(-1).contiguous()
      mask_vertices_flat_indicator = mask_vertices_unflat_indicator.contiguous().view(-1).contiguous()
      return mask_vertices_flat, mask_vertices_flat_indicator
      
      

    # gneerate vertices mask...
    def _generate_vertices_mask_bak(self, vertices_flat):
        mask_ratio_low = opt.dataset.mask_low_ratio
        mask_ratio_high = opt.dataset.mask_high_ratio
        tot_n = vertices_flat.size(-1) + 1  #
        # mask_ratio = np.random.uniform(low=0.05, high=0.06, size=(1,)).item()
        mask_ratio = np.random.uniform(low=mask_ratio_low, high=mask_ratio_high, size=(1,)).item()
        mask_ratio = float(mask_ratio)  # ratio
        cur_nn_masks = int(mask_ratio * tot_n)
        cur_nn_masks = cur_nn_masks if cur_nn_masks > 1 else 1
        vertices_flat_masked = vertices_flat.clone()
        # vertices_flat_masked = np.zeros_like(vertices_flat)
        # vertices_flat_masked[:] = vertices_flat[:]
        # NNNN
        ### + 1!!!
        vertices_flat_masked = torch.cat([vertices_flat_masked + 1, torch.zeros((1,), dtype=torch.long)], dim=-1)

        sampled_masked_coord_indices = np.random.choice(tot_n, size=cur_nn_masks, replace=False)
        sampled_masked_coord_indices = torch.from_numpy(sampled_masked_coord_indices).long()  # maksed coord
        vertices_flat_masked[sampled_masked_coord_indices] = 2 ** self.quantization_bits + 1
        vertices_mask_identifier = torch.zeros_like(vertices_flat_masked).float()
        vertices_mask_identifier[sampled_masked_coord_indices] = 1.

        vertices_flat_masked = torch.cat([vertices_flat_masked,
                                          torch.zeros((self.nn_max_vertices * 3 - vertices_flat_masked.size(0)),
                                                      dtype=torch.long)], dim=-1)

        vertices_mask_permut = torch.argsort(vertices_mask_identifier, dim=0, descending=True)  # identifier

        vertices_mask_identifier = torch.cat([vertices_mask_identifier,
                                              torch.zeros((self.nn_max_vertices * 3 - vertices_mask_identifier.size(0)),
                                                          dtype=torch.float32)], dim=-1)

        res_permut = np.array(range(vertices_mask_permut.size(0), self.nn_max_vertices * 3), dtype=np.int32)
        res_permut = torch.from_numpy(res_permut).long()
        vertices_mask_permut = torch.cat([vertices_mask_permut, res_permut], dim=-1)

        return vertices_flat_masked, vertices_mask_identifier, vertices_mask_permut

    # gneerate vertices mask...
    def _generate_vertices_mask(self, vertices_flat):
        # mask_ratio_low = opt.dataset.mask_low_ratio
        # mask_ratio_high = opt.dataset.mask_high_ratio
        tot_n = vertices_flat.size(-1) + 1  #
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

        vertices_mask_permut = torch.argsort(vertices_mask_identifier, dim=0, descending=True)  # identifier

        vertices_mask_identifier = torch.cat([vertices_mask_identifier,
                                              torch.zeros((self.nn_max_vertices * 3 - vertices_mask_identifier.size(0)),
                                                          dtype=torch.float32)], dim=-1)

        res_permut = np.array(range(vertices_mask_permut.size(0), self.nn_max_vertices * 3), dtype=np.int32)
        res_permut = torch.from_numpy(res_permut).long()
        vertices_mask_permut = torch.cat([vertices_mask_permut, res_permut], dim=-1)

        return vertices_flat_masked, vertices_mask_identifier, vertices_mask_permut

    # generate interpolation mask...
    def generate_interpolate_mask(self, vertices):
        # TODO a more general version -- the current one is only suitable for shapes with small number of vertices
        nn_verts = vertices.size(0)
        two_times_verts_permutation = torch.zeros((nn_verts * 2, 3), dtype=torch.long)
        vertical_orders = torch.arange(0, nn_verts * 2, dtype=torch.long) * 3  ## verti
        horizental_orders = torch.tensor([0, 1, 2], dtype=torch.long)
        two_times_verts_permutation = two_times_verts_permutation + vertical_orders.unsqueeze(
            -1) + horizental_orders.unsqueeze(0)
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
            [quan_vertices[..., 2].unsqueeze(-1), quan_vertices[..., 1].unsqueeze(-1),
             quan_vertices[..., 0].unsqueeze(-1)], dim=-1
        ) + 1
        vertices_permuted_two_times = torch.zeros((vertices_permuted.size(0) * 2, 3), dtype=torch.long)
        for i_vert in range(vertices_permuted.size(0)):
            vertices_permuted_two_times[i_vert * 2, :] = vertices_permuted[i_vert, :]
        vertices_permuted_two_times = vertices_permuted_two_times.contiguous().view(-1).contiguous()

        return two_times_flat_permutation, start_sampling_idx, vertices_permuted_two_times

    def generate_scaled_verts_mask(self, vertices):
        nn_verts = vertices.size(0)
        dequan_vertices = data_utils.dequantize_verts(vertices.numpy())  # dequan_vertices
        dequan_vertices = data_utils.center_vertices(dequan_vertices)
        vertices_scaling = np.array([1.0, 1.0, 0.33], dtype=np.float32)
        # minn_z = np.min()
        dequan_vertices = dequan_vertices * np.reshape(vertices_scaling, (1, 3))
        dequan_vertices_min = np.min(dequan_vertices, axis=0)[-1].item()

        # dequan_vertices = dequan_vertices + np.array([[0.0, 0.0, -0.5 - dequan_vertices_min]], dtype=np.float32)

        quantized_vertices = data_utils.quantize_verts(dequan_vertices, n_bits=self.quantization_bits)
        quantized_vertices = torch.from_numpy(quantized_vertices).long()
        vertices_permuted = torch.cat(
            [quantized_vertices[..., 2].unsqueeze(-1), quantized_vertices[..., 1].unsqueeze(-1),
             quantized_vertices[..., 0].unsqueeze(-1)], dim=-1
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
            [torch.zeros_like(vertices_permuted_flat), vertices_permuted_flat,
             torch.zeros_like(vertices_permuted_flat)], dim=0
        )
        # vertices_permuted_flat = torch.cat(
        #   [vertices_permuted_flat, torch.zeros_like(vertices_permuted_flat), torch.zeros_like(vertices_permuted_flat)], dim=0
        # )
        return vertices_permutation, start_sampling_idx, vertices_permuted_flat  # return vertices permutations, sampling idxes, and permuted flat...

    def generate_scaled_verts_mask_low_part(self, vertices):
        nn_verts = vertices.size(0)
        dequan_vertices = data_utils.dequantize_verts(vertices.numpy())  # dequan_vertices
        dequan_vertices = data_utils.center_vertices(dequan_vertices)
        vertices_scaling = np.array([1.0, 1.0, 0.33], dtype=np.float32)
        # minn_z = np.min()
        dequan_vertices = dequan_vertices * np.reshape(vertices_scaling, (1, 3))
        dequan_vertices_min = np.min(dequan_vertices, axis=0)[-1].item()

        dequan_vertices = dequan_vertices + np.array([[0.0, 0.0, -0.5 - dequan_vertices_min]], dtype=np.float32)

        quantized_vertices = data_utils.quantize_verts(dequan_vertices, n_bits=self.quantization_bits)
        quantized_vertices = torch.from_numpy(quantized_vertices).long()
        vertices_permuted = torch.cat(
            [quantized_vertices[..., 2].unsqueeze(-1), quantized_vertices[..., 1].unsqueeze(-1),
             quantized_vertices[..., 0].unsqueeze(-1)], dim=-1
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
            [vertices_permuted_flat, torch.zeros_like(vertices_permuted_flat),
             torch.zeros_like(vertices_permuted_flat)], dim=0
        )
        return vertices_permutation, start_sampling_idx, vertices_permuted_flat  # return vertices permutations, sampling idxes, and permuted flat...

    def generate_scaled_verts_mask_high_part(self, vertices):
        nn_verts = vertices.size(0)
        dequan_vertices = data_utils.dequantize_verts(vertices.numpy())  # dequan_vertices
        dequan_vertices = data_utils.center_vertices(dequan_vertices)
        vertices_scaling = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        # minn_z = np.min()
        dequan_vertices = dequan_vertices * np.reshape(vertices_scaling, (1, 3))
        dequan_vertices_max = np.max(dequan_vertices, axis=0)[-1].item()

        dequan_vertices = dequan_vertices + np.array([[0.0, 0.0, 0.5 - dequan_vertices_max]], dtype=np.float32)

        quantized_vertices = data_utils.quantize_verts(dequan_vertices, n_bits=self.quantization_bits)
        quantized_vertices = torch.from_numpy(quantized_vertices).long()
        vertices_permuted = torch.cat(
            [quantized_vertices[..., 2].unsqueeze(-1), quantized_vertices[..., 1].unsqueeze(-1),
             quantized_vertices[..., 0].unsqueeze(-1)], dim=-1
        ) + 1
        vertices_permuted_flat = vertices_permuted.contiguous().view(-1).contiguous()
        nn_verts_flat = vertices_permuted_flat.size(-1)
        ori_permutation = torch.arange(0, nn_verts_flat).long()
        # vertices_permutation = torch.cat(
        #   [ori_permutation + nn_verts_flat, ori_permutation, ori_permutation + nn_verts_flat * 2], dim=0
        # )
        vertices_permutation = torch.cat(
            [ori_permutation + nn_verts_flat * 2, ori_permutation, ori_permutation + nn_verts_flat], dim=0
        )
        start_sampling_idx = nn_verts_flat
        # vertices_permuted_flat = torch.cat(
        #   [torch.zeros_like(vertices_permuted_flat), vertices_permuted_flat, torch.zeros_like(vertices_permuted_flat)], dim=0
        # )
        vertices_permuted_flat = torch.cat(
            [torch.zeros_like(vertices_permuted_flat), torch.zeros_like(vertices_permuted_flat),
             vertices_permuted_flat], dim=0
        )
        return vertices_permutation, start_sampling_idx, vertices_permuted_flat  # return vertices permutations, sampling idxes, and permuted flat...

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
        # normalized_verts = self.augment_vertices_scale(normalized_verts)
        # normalized_verts = data_utils.center_vertices(normalized_verts)
        # normalized_verts = data_utils.normalize_vertices_scale(normalized_verts)
        # scaled_verts = data_utils.quantize_verts(normalized_verts, self.quantization_bits)

        # normalized_verts = normalized_verts + 0.5

        sigma = 0.25
        x_warp_gradient_sampled = np.random.normal(loc=0.0, scale=math.sqrt(sigma), size=(5,))
        y_warp_gradient_sampled = np.random.normal(loc=0.0, scale=math.sqrt(sigma), size=(5,))
        z_warp_gradient_sampled = np.random.normal(loc=0.0, scale=math.sqrt(sigma), size=(5,))

        x_warp_gradient_sampled = np.exp(x_warp_gradient_sampled)
        y_warp_gradient_sampled = np.exp(y_warp_gradient_sampled)
        z_warp_gradient_sampled = np.exp(z_warp_gradient_sampled)

        # x_reflect = True
        x_reflect = False
        y_reflect = True
        z_reflect = True

        scaled_verts = []
        for i_v in range(normalized_verts.shape[0]):
            cur_v_xyz = normalized_verts[i_v].tolist()
            cur_x, cur_y, cur_z = float(cur_v_xyz[0]), float(cur_v_xyz[1]), float(cur_v_xyz[2])
            scaled_x = data_utils.warp_coord(x_warp_gradient_sampled, cur_x, reflect=x_reflect)
            scaled_y = data_utils.warp_coord(y_warp_gradient_sampled, cur_y, reflect=y_reflect)
            scaled_z = data_utils.warp_coord(z_warp_gradient_sampled, cur_z, reflect=z_reflect)
            scaled_verts.append([scaled_x, scaled_y, scaled_z])
        scaled_verts = np.array(scaled_verts, dtype=np.float)  # n x 3
        scaled_verts = data_utils.center_vertices(scaled_verts)
        # normalize_vertices_scale --
        scaled_verts = data_utils.normalize_vertices_scale(scaled_verts)
        scaled_verts = data_utils.quantize_verts(scaled_verts, self.quantization_bits)
        return scaled_verts

    ''' Load data with random axises flipping and random vertices shifting  '''

    def __getitem__(self, item):

        # category_part_nm indicator...
        category_part_nm_indicator = self.mesh_dict_idx_to_category_name[item]
        category_part_idx = self.category_part_indicator_to_idxes[category_part_nm_indicator]

        cur_item_mesh_dict = self.mesh_dicts[item]

        # make vertices
        vertices = torch.from_numpy(cur_item_mesh_dict['vertices']).long()
        if self.random_scaling > 0.5:
            # print("Applying random scaling...")
            scaled_vertices = self.apply_random_scaling(vertices.numpy())
            vertices = torch.from_numpy(scaled_vertices).long()
        if self.apply_random_shift:
            vertices = data_utils.random_shift(vertices=vertices, shift_factor=0.25)
        faces = torch.from_numpy(cur_item_mesh_dict['faces']).long()
        if not self.category_part_indicator:  # add class label
            class_label = torch.tensor([cur_item_mesh_dict['class_label']], dtype=torch.long)
        else:
            class_label = torch.tensor([category_part_idx], dtype=torch.long)

        # if self.is_training and self.apply_random_flipping:
        #     permutation_idxes = np.random.randint(low=0, high=6, size=(1,)).item()
        #     rgt_idx = permutation_idxes % 3
        #     remains_idxes = [xx for xx in range(3) if xx != rgt_idx]
        #     mid_idx = remains_idxes[permutation_idxes // 3]
        #     lft_idx = [xx for xx in range(3) if xx != rgt_idx and xx != mid_idx][0]
        #     permutation_idxes = [lft_idx, mid_idx, rgt_idx]
        #     vertices = vertices[:, permutation_idxes]

        # Re-order vertex coordinates as (z, y, x).
        vertices_permuted = torch.cat(
            [vertices[..., 2].unsqueeze(-1), vertices[..., 1].unsqueeze(-1), vertices[..., 0].unsqueeze(-1)], dim=-1
        )

        vertices_flat = vertices_permuted.contiguous().view(-1).contiguous()

        mask_vertices_flat, mask_vertices_flat_indicator = self._generate_vertices_mask_ml(vertices_flat)
        


        nn_vertices = vertices_flat.size(0)

        nn_faces = faces.size(0)  # faces

        ''' minn vertex index value is 1 here, so + 1 can get real vertex indices for faces '''
        vertices_flat = torch.cat(
            [vertices_flat + 1, torch.zeros((self.nn_max_vertices * 3 - vertices_flat.size(0),), dtype=torch.long)],
            dim=-1
        )

        mask_vertices_flat = torch.cat( # masked vertices
            [mask_vertices_flat + 1, torch.zeros((self.nn_max_vertices * 3 - mask_vertices_flat.size(0),), dtype=torch.long)],
            dim=-1
        )

        mask_vertices_flat_indicator = torch.cat(
            [mask_vertices_flat_indicator, torch.zeros((self.nn_max_vertices * 3 - mask_vertices_flat_indicator.size(0),), dtype=torch.float32)],
            dim=-1
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

        # ''' Permutation '''
        # permut = np.random.permutation(nn_vertices)
        # permut = torch.from_numpy(permut).long() # seq_len
        # res_permut = np.array(range(nn_vertices, self.nn_max_vertices * 3), dtype=np.int32)
        # res_permut = torch.from_numpy(res_permut).long()
        # permut = torch.cat(
        #   [permut, res_permut], dim=0
        # )
        # ''' Permutation '''

        # dequantized vertices
        real_nn_vertices = vertices.size(0)

        nn_vertices_predict = int(real_nn_vertices * self.nn_vertices_predict_ratio)
        nn_vertices_masked = real_nn_vertices - nn_vertices_predict

        vertices_flat_mask = torch.cat(
            [torch.zeros((nn_vertices_masked * 3, ), dtype=torch.float32),
             torch.ones((nn_vertices_predict * 3 + 1,), dtype=torch.float32),
             torch.zeros((self.nn_max_vertices * 3 - nn_vertices - 1), dtype=torch.float32)], dim=-1
        )

        if self.context_window > 0 and self.cut_vertices > 0 and real_nn_vertices > self.context_window // 3:
            print("cutting vertices")
            context_nn_vertices = self.context_window // 3 # get 
            # if real_nn_vertices > context_nn_vertices:
            context_st_vertices = np.random.randint(low=0, high=real_nn_vertices - context_nn_vertices, size=(1,), dtype=np.int32).item()
            cur_context_window = self.context_window
            if context_st_vertices == real_nn_vertices - context_nn_vertices - 1:
                cur_context_window = cur_context_window + 1
            context_vertices_flat = vertices_flat[context_st_vertices * 3: context_st_vertices * 3 + cur_context_window]
            context_vertices_flat_mask = vertices_flat_mask[context_st_vertices * 3: context_st_vertices * 3 + cur_context_window]
            context_vertices_flat = torch.cat(
                [context_vertices_flat, torch.zeros((self.nn_max_vertices * 3 - context_vertices_flat.size(0),), dtype=torch.long)],
                dim=-1
            )
            context_vertices_flat_mask = torch.cat(
                # context veritces flat mask...
                [context_vertices_flat_mask, torch.zeros((self.nn_max_vertices * 3 - context_vertices_flat_mask.size(0),), dtype=torch.float32)], dim=-1
            )
            vertices_flat = context_vertices_flat # context vertices flat
            vertices_flat_mask = context_vertices_flat_mask
            

        

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
        vertices = torch.cat(  #
            [vertices, torch.zeros((self.nn_max_vertices - real_nn_vertices, 3), dtype=torch.long)], dim=0
        )

        vertices_mask = torch.cat(  #
            [torch.ones((real_nn_vertices,), dtype=torch.float32),
             torch.zeros((self.nn_max_vertices - real_nn_vertices,), dtype=torch.float32)], dim=0
        )

        # print(f"item: {item}, nn_left_faces: {nn_left_faces}, nn_rgt_faces: {nn_rgt_faces}")
        faces_mask = torch.cat(
            [torch.ones((nn_faces,), dtype=torch.float32),
             torch.zeros((self.nn_max_faces - nn_faces), dtype=torch.float32)], dim=-1
        )

        # faces and faces mask
        faces = torch.cat(
            [faces, torch.zeros((self.nn_max_faces - nn_faces), dtype=torch.long)], dim=-1
        )

        rt_dict = {
            'vertices_flat_mask': vertices_flat_mask,
            'mask_vertices_flat': mask_vertices_flat,
            'mask_vertices_flat_indicator': mask_vertices_flat_indicator,
            'vertices_flat': vertices_flat,  # flat...
            'faces_mask': faces_mask,
            'vertices': vertices,
            'vertices_mask': vertices_mask,
            'faces': faces,
            'class_label': class_label,
            # 'vertices_flat_masked': vertices_flat_masked,
            # 'vertices_mask_identifier': vertices_mask_identifier,
            # 'vertices_permutation': permut,
            # 'start_sampling_idx': start_sampling_idx
            # 'vertices_masked_permutation': vertices_mask_permut
        }

        return rt_dict

