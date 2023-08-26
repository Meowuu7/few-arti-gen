# from glob import glob
from inspect import getmodule
import os
from unicodedata import category

from utils.data_utils_torch import dequantize_verts
# import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import os.path
import json
import numpy as np
import math
import sys
import torch

from torch.utils import data
import utils.data_utils_torch as data_utils
from utils.trainer_utils import *
# from datasets.urdf_dataset_pretraining_xl import URDFDataset as URDFDataset_xl
# from datasets.urdf_dataset_pretraining_xl import URDFDataset as URDFDataset_xl
from datasets.urdf_dataset_pretraining_ar_grid_vox import URDFDataset as URDFDataset_ar
from datasets.urdf_dataset_pretraining_ar_grid_vox_multi_part import URDFDataset as URDFDataset_ar_multi_part
from datasets.urdf_dataset_pretraining_ar_grid_vox_obj import URDFDataset as URDFDataset_ar_obj
from datasets.urdf_dataset_pretraining_ar_grid_vox_corpus import URDFDataset as URDFDataset_ar_corpus
from datasets.urdf_dataset_pretraining_ar_grid_vox_obj_corpus import URDFDataset as URDFDataset_ar_obj_corpus
from datasets.urdf_dataset_subd_mesh import URDFDataset as URDFDataset_subd_mesh
from datasets.urdf_dataset_mesh import URDFDataset as URDFDataset_mesh
# from datasets.urdf_dataset_pretraining_cus_perm_xl import URDFDataset as URDFDataset_cus_perm_xl
# from datasets.urdf_dataset_pretraining_cus_perm_xl_2 import URDFDataset as URDFDataset_cus_perm_xl
# from modules.modules_pretraining_grid_v2 import VertexModel
from modules.modules_pretraining_ar_subd_meshes import VertexModel, FaceModel
# from modules.modules_[re
# from modules.modules_pretraining_xl import VertexModel, FaceModel
# from modules.modules_pretraining_xl_2 import VertexModel, FaceModel
from utils.model_utils import *


from tqdm import tqdm
import time
import torch.optim as optim
from options.options import opt
from utils.constants import *


epoch_counter = 0
test_epoch_counter = 0
# use_cuda = True
step_counter = 0

# nn_max_vertices = 2000
# nn_max_faces = 2000

# todo: add logger to beautify the logging
### inputs data: ordered vertics + faces

def optimize(data, vertex_model, vertex_optimizer):
    # global use_cuda
    global step_counter

    vertex_model.module.train()
    face_model.module.train()

    for k in data:
        data[k] = data[k].cuda(non_blocking=True)
        # print(f"key: {k}, shape: {data[k].size()}")

    # try:
    # vertex_model_pred_dict
    # pred_dist_grid_xyz, pred_dist_grid_values = vertex_model(data)
    pred_subd_to_gt_dist = vertex_model(data) # i_subd: distribution

    torch.cuda.empty_cache()
    # print("After vertex model forwarding")s

    pred_face_dist = face_model(data)
    all_pred_loss = []
    all_reg_sigma_loss = []
    # face_model_pred_dict = face_model(data) # 
    tot_prediction_loss = 0.

    # for i_subd in pred_subd_to_gt_dist:
    cur_subd_gt_dist = pred_subd_to_gt_dist # [i_subd]
    # pretraining_ar_subd_meshes
    if isinstance(cur_subd_gt_dist, torch.distributions.Categorical):
        cur_subd_xyz_gt = data[f'subd_{i_subd + 1}_gt'] # bsz x n_verts x 3
        cur_subd_pred_loss = -torch.sum(
            cur_subd_gt_dist.log_prob(cur_subd_xyz_gt), dim=-1
        )
        cur_subd_pred_loss = cur_subd_pred_loss.sum(-1)
        tot_prediction_loss += cur_subd_pred_loss
        all_pred_loss.append(cur_subd_pred_loss.mean().detach().item())
    elif isinstance(cur_subd_gt_dist, torch.distributions.normal.Normal):
        cur_subd_xyz_gt = data[f'subd_{i_subd + 1}_gt']
        cur_subd_xyz_gt = dequantize_verts(cur_subd_xyz_gt.detach().cpu().numpy(), n_bits=quantization_bits)
        cur_subd_xyz_gt = torch.from_numpy(cur_subd_xyz_gt).float().cuda()
        cur_subd_xyz_upsample = data[f'subd_{i_subd}_upsample']
        cur_subd_xyz_upsample = dequantize_verts(cur_subd_xyz_upsample.detach().cpu().numpy(), n_bits=quantization_bits)
        cur_subd_xyz_upsample = torch.from_numpy(cur_subd_xyz_upsample).float().cuda()
        offset_xyz = cur_subd_xyz_gt - cur_subd_xyz_upsample
        cur_subd_pred_loss = -torch.sum(
            cur_subd_gt_dist.log_prob(offset_xyz), dim=-1
        )
        cur_subd_pred_loss = cur_subd_pred_loss.sum(-1)
        tot_prediction_loss += cur_subd_pred_loss
        all_pred_loss.append(cur_subd_pred_loss.mean().detach().item())
    # elif isinstance(cur_subd_gt_dist, torch.Tensor):
    elif isinstance(cur_subd_gt_dist, dict):
        delta_xyz, mu_xyz, sigma_xyz = cur_subd_gt_dist['delta_xyz'], cur_subd_gt_dist['mu_xyz'], cur_subd_gt_dist['sigma_xyz']
        # delta_xyz = cur_subd_gt_dist
        # cur_subd_xyz_gt = data[f'subd_{i_subd + 1}_gt']
        cur_subd_xyz_gt = data[f'vertices'] # bsz x n_grids x 3
        # cur_subd_xyz_gt = dequantize_verts(cur_subd_xyz_gt.detach().cpu().numpy(), n_bits=quantization_bits)
        # cur_subd_xyz_gt = torch.from_numpy(cur_subd_xyz_gt).float().cuda()
        # cur_subd_xyz_upsample = data[f'subd_{i_subd}_upsample']
        # cur_subd_xyz_upsample = dequantize_verts(cur_subd_xyz_upsample.detach().cpu().numpy(), n_bits=quantization_bits)
        # cur_subd_xyz_upsample = torch.from_numpy(cur_subd_xyz_upsample).float().cuda()

        offset_xyz = cur_subd_xyz_gt # - cur_subd_xyz_upsample
        cur_subd_xyz_dist = delta_xyz
        vertex_prediction_loss = -torch.sum(
          cur_subd_xyz_dist.log_prob(data['quant_vertices']) * data['quant_vertices_mask'], dim=-1 #### no face mask 
        )
        vertex_prediction_loss = torch.sum(vertex_prediction_loss)

        xyz_dist_loss = vertex_prediction_loss

        # print("mean of delta_xyz", torch.sum(torch.abs(delta_xyz), dim=-1).mean().item())

        # pred_subd_xyz = cur_subd_xyz_upsample + delta_xyz # bsz x n1 x 3
        # dist_pred_gt = torch.sum((pred_subd_xyz.unsqueeze(2) - cur_subd_xyz_gt.unsqueeze(1)) ** 2, dim=-1)
        # minn_pred_to_gt, _ = torch.min(dist_pred_gt, dim=-1)
        # minn_gt_to_pred, _ = torch.min(dist_pred_gt, dim=-2)
        # xyz_dist_loss = (minn_pred_to_gt.sum(-1) + minn_gt_to_pred.sum(-1))


        # xyz_dist_loss += torch.sqrt(torch.sum((offset_xyz - delta_xyz) ** 2, dim=-1)).sum(-1)

        # xyz_dist_loss = (torch.sum((offset_xyz - delta_xyz) ** 2, dim=-1)).mean(-1)

        ##### reg_sigma xxxx #####
        # reg_sigma = torch.sum(sigma_xyz ** 2, dim=-1).mean(-1) # bsz ##### want larger sigma value? ### less 
        # xyz_dist_loss = xyz_dist_loss - reg_sigma

        # xyz_dist_loss = xyz_dist_loss

        ##### reg_sigma and a simple ###### # reg_sigma
        all_reg_sigma_loss.append(0 )

        tot_prediction_loss += xyz_dist_loss
        # all_pred_loss.append(xyz_dist_loss.mean().detach().item())
        
    else:
        raise ValueError(f"Unrecognized distribution!")

    # tot_prediction_loss /= len(pred_subd_to_gt_dist)

    # face_prediction_loss = []
    if isinstance(pred_face_dist, torch.distributions.Categorical):
        # face prediction loss # predict face dict
      face_prediction_loss = -torch.sum(
          pred_face_dist.log_prob(data['faces']) * data['faces_mask'] , dim=-1 #### no face mask 
      ) 
      face_prediction_loss = torch.sum(face_prediction_loss)
    else:
      raise ValueError(f"Unrecognized distribution for faces prediction!")

    
    if _use_multi_gpu:
      torch.distributed.barrier() 
      # tot_prediction_loss; 
      tot_prediction_loss = reduce_mean(tot_prediction_loss, opt.nprocs)
      face_prediction_loss = reduce_mean(face_prediction_loss, opt.nprocs)
      # all_pred_loss = [reduce_mean(tmp_loss, opt.nprocs) for tmp_loss in all_pred_loss]


    # # grid_xyz_prediction_loss: bsz x grid_length x 3
    # grid_xyz_prediction_loss = -torch.sum( # ientifiers
    #     pred_dist_grid_xyz.log_prob(data['grid_xyzs']) * data['grid_xyzs_mask'], dim=-1
    # ) 
    # grid_xyz_prediction_loss = grid_xyz_prediction_loss.sum(-1)
    # # grid_xyz_prediction_loss = grid_xyz_prediction_loss.
    
    # # values of pred_dist_grid_values: bsz x grid_length
    # grid_values_prediction_loss = -pred_dist_grid_values.log_prob(data['grid_content_vocab']) * data['grid_content_vocab_mask'] 
    # # -torch.sum( # ientifiers
    # #     pred_dist_grid_values.log_prob(data['grid_content']) * data['grid_content_mask'], dim=-1
    # # ) 
    # grid_values_prediction_loss = grid_values_prediction_loss.sum(-1) # .sum(-1).sum(-1).sum(-1)

    # # # vertices loss and face loss
    # # vertex_prediction_loss = -torch.sum( # ientifiers
    # #     vertex_model_pred_dict.log_prob(data['vertices_flat']) * data['vertices_flat_mask'], dim=-1
    # # ) 
    # vertex_prediction_loss = torch.mean(grid_xyz_prediction_loss + grid_values_prediction_loss) # grid xyz/values predictions


    # face prediction loss
    # face_prediction_loss = -torch.sum(
    #     face_model_pred_dict.log_prob(data['faces']) * data['faces_mask'], dim=-1
    # ) 

    ###
    # face_prediction_loss = torch.sum(face_prediction_loss)

    torch.cuda.empty_cache()
    loss = tot_prediction_loss + face_prediction_loss 
    torch.cuda.empty_cache()
    # face_optimizer.zero_grad() # zero_grad
    vertex_optimizer.zero_grad()
    face_optimizer.zero_grad()
    torch.cuda.empty_cache()
    loss.backward()
    torch.cuda.empty_cache()
    # face_optimizer.step()
    vertex_optimizer.step()
    face_optimizer.step()
    torch.cuda.empty_cache()
    # except:
    #     print(f"Something bad happended, continuing...")

    if step_counter % check_step == 0:
        #### all_reg_sigma_loss ####
      print(f"Step {step_counter}")
      print(f"Vertex prediction loss: {tot_prediction_loss.mean().item()}, face_loss: {face_prediction_loss.mean().item()} Avg loss: {tot_prediction_loss.item()}")
      # print(f"Face Loss {face_prediction_loss.item()}")
    step_counter += 1

    return



def step(train_dataset, vertex_model, vertex_optimizer, batch_size, dataset_iter):
    global epoch_counter
    try:
        data = next(dataset_iter)
        if data['class_label'].size(0) < batch_size:
            raise StopIteration
    except StopIteration:
        epoch_counter += 1
        # print("[DataLoader]: At Epoch %d!" % epoch_counter)
        # reset dataset iterator
        dataset_iter = iter(train_dataset)
        # get data for the next iteration
        data = next(dataset_iter)
    optimize(data, vertex_model, vertex_optimizer)
    return dataset_iter


def step_sample(test_dataset, vertex_model, test_dataset_iter, cur_iter=0):
    global test_epoch_counter # 
    try:
        data = next(test_dataset_iter)
        if data['class_label'].size(0) < batch_size:
            raise StopIteration
    except StopIteration:
        test_epoch_counter += 1
        # print("[Test DataLoader]: At Epoch %d!" % test_epoch_counter)
        # reset dataset iterator
        test_dataset_iter = iter(test_dataset)
        # get data for the next iteration
        data = next(test_dataset_iter)
    sample(data, vertex_model, cur_step=cur_iter)
    return test_dataset_iter


def sample(test_data, vertex_model, cur_step=0):
    # global nn_max_vertices
    # global nn_max_faces

    ### subd meshes ###

    vertex_model.module.eval()
    face_model.module.eval()

    with torch.no_grad():

        for k in test_data:
          test_data[k] = test_data[k].cuda(non_blocking=True)
        # print(f"key: {k}, shape: {data[k].size()}")
        # construct sampling context
        
        bsz = test_data['class_label'].size(0)

        sampled_vertices = vertex_model.module.sample(bsz, context=test_data)
        
        sampled_faces = face_model.module.sample(context=sampled_vertices)

        # vertices = sampled_vertices['vertices']
        # faces = sampled_faces['faces']
        # vertices_mask = sampled_vertices['vertices_mask']

        v_sample = {}
        for k in sampled_vertices:
          v_sample[k] = sampled_vertices[k].detach().cpu().numpy()
        f_sample = {}
        for k in sampled_faces:
          f_sample[k] = sampled_faces[k].detach().cpu().numpy()

        data_utils.plot_sampled_ar_subd_meshes(v_sample, f_sample, sv_mesh_folder, cur_step=cur_step)


        # if opt.dataset.subdn_test == 2:
        #     # try:
        #     # vertex_model_pred_dict
        #     # pred_dist_grid_xyz, pred_dist_grid_values = vertex_model(data)
        #     pred_subd_to_gt_dist = vertex_model(test_data) # i_subd: distribution
        #     # pred_subd_to_gt_dist = vertex_model.sample(1, context=test_data) # i_subd: distribution
        #     subd_idx = opt.dataset.subdn - 1
        #     ####### gt_dist #######
        #     if isinstance(pred_subd_to_gt_dist, dict) and f'subd_{subd_idx}_gt' in pred_subd_to_gt_dist:
        #         pred_last_subd_verts = pred_subd_to_gt_dist[f'subd_{subd_idx}_gt']
        #     else:
        #         pred_last_subd_dist = pred_subd_to_gt_dist[opt.dataset.subdn - 2] ##### 3 - 2 --> predicted dist...
        #         # pred_last_subd_verts = pred_last_subd_dist.sample() # bsz x n_vert x 3
        #         if isinstance(pred_last_subd_dist, torch.distributions.normal.Normal):
        #             pred_last_subd_dist = pred_subd_to_gt_dist[opt.dataset.subdn - 2] ##### 3 - 2 --> predicted dist...
        #             pred_last_subd_verts = pred_last_subd_dist.sample() # bsz x n_vert x 3

        #             cur_subd_upsample = test_data[f'subd_{opt.dataset.subdn - 2}_upsample'].detach().cpu().numpy()
        #             cur_subd_upsample = dequantize_verts(cur_subd_upsample, quantization_bits)
        #             pred_last_subd_verts = cur_subd_upsample + pred_last_subd_verts.detach().cpu().numpy()
        #             # pred_last_subd_verts = cur_subd_upsample ####
        #         elif isinstance(pred_last_subd_dist, torch.distributions.Categorical):
        #             pred_last_subd_dist = pred_subd_to_gt_dist[opt.dataset.subdn - 2] ##### 3 - 2 --> predicted dist...
        #             pred_last_subd_verts = pred_last_subd_dist.sample() # bsz x n_vert x 3
        #             ##### current upsampled data #####
        #             cur_subd_upsample = test_data[f'subd_{opt.dataset.subdn - 2}_upsample'].detach().cpu().numpy()
        #             cur_subd_upsample = dequantize_verts(cur_subd_upsample, quantization_bits)
        #             pred_last_subd_verts = pred_last_subd_verts.detach().cpu().numpy()
        #             pred_last_subd_verts = data_utils.dequantize_verts(pred_last_subd_verts, n_bits=opt.model.quantization_bits) 
        #         elif isinstance(pred_last_subd_dist, torch.Tensor):
        #             cur_subd_upsample = test_data[f'subd_{opt.dataset.subdn - 2}_upsample'].detach().cpu().numpy()
        #             cur_subd_upsample = dequantize_verts(cur_subd_upsample, quantization_bits)

        #             pred_last_subd_verts =  pred_subd_to_gt_dist[opt.dataset.subdn - 2]
        #             pred_last_subd_verts = cur_subd_upsample + pred_last_subd_verts.detach().cpu().numpy()
        #         else:
        #             raise ValueError(f"Unrecognized distribution!")

        #     last_subd_faces = test_data[f'subd_{subd_idx}_faces'].detach().cpu().numpy().tolist()
        #     for i_bsz in range(pred_last_subd_verts.shape[0]):
        #         cur_vertices = pred_last_subd_verts[i_bsz]
        #         cur_faces = last_subd_faces[i_bsz]
        #         # cur_vertices = data_utils.dequantize_verts(cur_vertices, n_bits=opt.model.quantization_bits) # bsz x n_verts x 3
        #         cur_mesh_sv_fn = os.path.join(sv_mesh_folder, f"sample_step_{cur_step}.obj")
        #         print(f"Saving {cur_step}-th sample to {cur_mesh_sv_fn}.")
        #         data_utils.write_obj(cur_vertices, cur_faces, cur_mesh_sv_fn, transpose=True, scale=1.)

        #         cur_subd_upsample_vertices = cur_subd_upsample[i_bsz]
        #         cur_mesh_sv_fn = os.path.join(sv_mesh_folder, f"sample_step_{cur_step}_context.obj")
        #         print(f"Saving {cur_step}-th upsample to {cur_mesh_sv_fn}.")
        #         data_utils.write_obj(cur_subd_upsample_vertices, cur_faces, cur_mesh_sv_fn, transpose=True, scale=1.)
        # else:
        #     sampled_batch = vertex_model.sample_forward(test_data, adapter_modules=None)
        #     # for i_subd in range(opt)
        #     subdn = opt.dataset.subdn 
        #     for i_subd in range(subdn): #### save all subd data ####
        #         cur_subd_verts = sampled_batch[f'subd_{i_subd}']
        #         cur_subd_faces = test_data[f'subd_{i_subd}_faces']
        #         cur_subd_verts = cur_subd_verts.detach().cpu().numpy()
        #         cur_subd_verts = dequantize_verts(cur_subd_verts, quantization_bits)
        #         for i_bsz in range(cur_subd_verts.shape[0]):
        #             cur_bsz_subd_verts = cur_subd_verts[i_bsz]
        #             cur_bsz_subd_faces = cur_subd_faces[i_bsz].detach().cpu().numpy().tolist() #### face_list
        #             cur_bsz_subd_mesh_sv_fn = os.path.join(sv_mesh_folder, f"sampled_step_{cur_step}_subd_{i_subd}.obj")
        #             print(f"Saving {cur_step}-th {i_subd}-th subd mesh to {cur_bsz_subd_mesh_sv_fn}.")
        #             data_utils.write_obj(cur_bsz_subd_verts, cur_bsz_subd_faces, cur_bsz_subd_mesh_sv_fn, transpose=True, scale=1.)




def save_ckpts(vertex_model, face_model, cur_step):
    vertex_model_sv_folder = os.path.join(sv_ckpts_folder, "vertex_model")
    os.makedirs(vertex_model_sv_folder, exist_ok=True)
    # face_model_sv_folder = os.path.join(sv_ckpts_folder, "face_model")
    # os.makedirs(face_model_sv_folder, exist_ok=True)
    model_sv_path = f'{cur_step}.pth'
    vertex_model_sv_pth = os.path.join(vertex_model_sv_folder, model_sv_path)
    vertex_model_params = vertex_model.state_dict()
    torch.save(vertex_model_params, vertex_model_sv_pth) 

    face_model_sv_folder = os.path.join(sv_ckpts_folder, "face_model")
    os.makedirs(face_model_sv_folder, exist_ok=True)
    #### face_model_sv_...
    face_model_sv_pth = os.path.join(face_model_sv_folder, model_sv_path)
    face_model_params = face_model.state_dict()
    torch.save(face_model_params, face_model_sv_pth)

    # model_sv_path = f'{cur_step}.pth'
    # face_model_sv_pth = os.path.join(face_model_sv_folder, model_sv_path)
    # face_model_params = face_model.state_dict()
    # torch.save(face_model_params, face_model_sv_pth) # face model path...
    print(f"Vertex model at step {cur_step} saved to {vertex_model_sv_pth}.")
    # print(f"Face model at step {cur_step} saved to {face_model_sv_pth}.")



# train iter
def train_iter(train_dataset, test_dataset, vertex_model, vertex_optimizer, training_steps, batch_size):
    # dataset_iter
    # iter dataset
    dataset_iter = iter(train_dataset)
    test_dataset_iter = iter(test_dataset)
    for i in range(training_steps):
        torch.cuda.empty_cache()
        dataset_iter = step(train_dataset, vertex_model, vertex_optimizer, batch_size, dataset_iter)
        torch.cuda.empty_cache()
        if i % check_step == 0 and local_rank == 0: # then sample from the model and save
            # sample(num_samples=batch_size, test_dataset=test_dataset, vertex_model=vertex_model, face_model=face_model, class_label=0, cur_step=i)
            test_dataset_iter = step_sample(test_dataset, vertex_model, test_dataset_iter, cur_iter=i)
            torch.cuda.empty_cache()
            save_ckpts(vertex_model.module, face_model.module, cur_step=i)
            # save_ckpts(vertex_model.module, cur_step=i)
            torch.cuda.empty_cache()

# train iter
def test_iter(test_dataset, vertex_model, vertex_optimizer, training_steps, batch_size):
    # dataset_iter
    # iter dataset
    # dataset_iter = iter(test_dataset)
    test_dataset_iter = iter(test_dataset)
    for i in range(training_steps):
        # dataset_iter = step(train_dataset, vertex_model, vertex_optimizer, batch_size, dataset_iter)
        # if i % check_step == 0: # then sample from the model and save
        # sample(num_samples=batch_size, test_dataset=test_dataset, vertex_model=vertex_model, face_model=face_model, class_label=0, cur_step=i)
        test_dataset_iter = step_sample(test_dataset, vertex_model, test_dataset_iter, cur_iter=i)
        # save_ckpts(vertex_model, cur_step=i)


# sometiems more easier than eyeglasses?
def get_model_descriptor():
    motion_dataset_nm = opt.dataset.dataset_name
    # motion_dataset_nm
    # motion_cat = "eyeglasses"
    motion_cat = opt.dataset.category
    batch_size = opt.loss.batch_size
    # nn 
    nn_max_vertices = opt.vertex_model.max_vertices
    nn_max_faces = opt.face_model.max_faces
    nn_max_permite_vertices = opt.dataset.max_permit_vertices
    nn_max_permite_faces = opt.dataset.max_permit_faces
    # mask_low_ratio = opt.dataset.mask_low_ratio
    # mask_high_ratio = opt.dataset.mask_high_ratio
    cus_perm_xl = opt.dataset.cus_perm_xl # whether to use the customized permutation for the permutation based auto-regressive model
    cross = opt.model.cross
    exp_flag = opt.common.exp_flag
    exp_mode = opt.common.exp_mode
    use_light_weight = opt.model.use_light_weight
    apply_random_shift = opt.dataset.apply_random_shift
    apply_random_flipping = opt.dataset.apply_random_flipping
    category_part_cat = opt.dataset.category_part_indicator
    face_class_conditional = opt.face_model.face_class_conditional
    random_scaling = opt.dataset.apply_random_scaling
    specify_light_weight = opt.model.specify_light_weight
    nn_vertices_predict_ratio = opt.dataset.nn_vertices_predict_ratio
    use_eyeglasses_frame = opt.dataset.use_eyeglasses_frame
    context_window = opt.model.context_window
    grid_size = opt.vertex_model.grid_size
    max_num_grids = opt.vertex_model.max_num_grids
    prefix_key_len = opt.model.prefix_key_len
    prefix_value_len = opt.model.prefix_value_len
    # set model descriptor
    model_descriptor = f"{exp_mode}_ar_subd_mesh_{exp_flag}_indi_{category_part_cat}_ctx_{context_window}_gs_{grid_size}_ngrids_{max_num_grids}_nobj_{opt.dataset.num_objects}_scale_{random_scaling}_{motion_dataset_nm}_bsz_{batch_size}_max_verts_{nn_max_vertices}_max_permit_verts_{nn_max_permite_vertices}_key_{prefix_key_len}"
    return model_descriptor


def safe_load_ckpt_common(model, state_dicts):
    ori_dict = state_dicts
    part_dict = dict()
    model_dict = model.state_dict()
    tot_params_n = 0
    for k in ori_dict:
        if k in model_dict:
            v = ori_dict[k]
            part_dict[k] = v
            tot_params_n += 1
    model_dict.update(part_dict)
    model.load_state_dict(model_dict)
    # self.logger.log('Setup', f"Resume glb-backbone finished!! Total number of parameters: {tot_params_n}.")
    #


def resume_from_ckpts(vertex_model, vertex_model_path):

    vertex_model_state_dicts = torch.load(vertex_model_path, map_location="cpu")
    # face_model_state_dicts = torch.load(face_model_path, map_location="cpu")

    safe_load_ckpt_common(vertex_model, vertex_model_state_dicts)
    # safe_load_ckpt_common(face_model, face_model_state_dicts)


    # vertex_model.load_state_dict(vertex_model_state_dicts)
    # face_model.load_state_dict(face_model_state_dicts)
    print(f"Vertex model loaded from {vertex_model_path}.")
    # print(f"Face model loaded from {face_model_path}.")



if __name__=='__main__':
    #### motion dataset name #####
    #### motion dataset name #####

    opt.nprocs = torch.cuda.device_count()
    print(f"Using {opt.nprocs} GPUs.")

    local_rank, _use_multi_gpu = ddp_init()


    # motion_dataset_nm = "MotionDataset"
    motion_dataset_nm = opt.dataset.dataset_name
    # motion_dataset_nm
    # motion_cat = "eyeglasses"
    motion_cat = opt.dataset.category

    dataset_root_path = opt.dataset.dataset_root_path # root path
    debug = opt.model.debug

    print(f"Debug: {debug}")
    
    # if THU_LAB_FLAG in dataset_root_path:
    opt.dataset.lab = "thu"
    # else:
    #     opt.dataset.lab = "meg"
    
    # root_folder for the dataset
    root_folder = os.path.join(dataset_root_path, motion_dataset_nm, motion_cat)
    ### root folder ###
    dataset_root_folder = os.path.join(dataset_root_path, motion_dataset_nm)
    
    # process recenter mesh # recenter mesh
    # process_recenter_mesh = True

    recenter_mesh = opt.model.recenter_mesh
    process_recenter_mesh = opt.model.recenter_mesh

    # quantization_bits = 8
    quantization_bits = opt.model.quantization_bits

    batch_size = opt.loss.batch_size
    num_thread = 10
    plot_sample_step = 2000

    # nn_max_permite_vertices = 400
    # nn_max_permite_faces = 2000

    nn_max_permite_vertices = opt.dataset.max_permit_vertices
    nn_max_permite_faces = opt.dataset.max_permit_faces

    # nn_max_vertices = 400
    # nn_max_faces = 2000
    nn_max_vertices = opt.vertex_model.max_vertices
    nn_max_faces = opt.face_model.max_faces

    # cus_perm_xl = opt.dataset.cus_perm_xl
    # corss = opt.model.cross

    model_descriptor = get_model_descriptor()

    # if motion_cat == "eyeglasses":
    train_ins_nns = None
    test_ins_nns = None
    # else:
    #   train_ins_nns = [str(iii) for iii in range(3000)]
    #   test_ins_nns = [str(iii) for iii in range(3000, 4001)]

    # if not cus_perm_xl:
    # URDFDataset = URDFDataset_ar
    # else:
    #     URDFDataset = URDFDataset_cus_perm_xl

    if opt.dataset.ar_object:
        # URDFDataset = URDFDataset_ar_obj
        if opt.dataset.num_objects > 1:
            URDFDataset = URDFDataset_ar_obj_corpus
        else:
            URDFDataset = URDFDataset_ar_obj
    else:
        # if not cus_perm_xl:
        if opt.dataset.num_parts > 1:
            # URDFDataset = URDFDataset_ar_multi_part
            URDFDataset = URDFDataset_ar_corpus 
        else:
            #### URDFDataset_ar ####
            URDFDataset = URDFDataset_ar
    
    # URDFDataset = URDFDataset_subd_mesh
    URDFDataset = URDFDataset_mesh

    ##### multi parts not 
    multi_parts_not_ar_object = opt.dataset.num_parts > 1 and (not opt.dataset.ar_object)

    debug = opt.model.debug

    if not debug:

    #   #### for train dataset ####
    #   category_name = ["bunny_6_reformed",]
    #   category_names_to_part_names = {
    #       "bunny_6_reformed": None, 
    #   }
    #   category_names = list(category_names_to_part_names.keys())
    #   #### for train dataset ####

      #### for partnet and shapenet #### 
    #   category_name = os.listdir(dataset_root_folder)
      # category_name = ["data_v0_deci_0.2"]
      # category_name = ["eyeglasses"]
      category_name = ["subd0"] # ["data_v0_deci_0.2"] ####
      category_name = [fn for fn in category_name if os.path.isdir(os.path.join(dataset_root_folder, fn))]
      category_names_to_part_names = {
        cat_nm: None for cat_nm in category_name
      }
      category_names = list(category_names_to_part_names.keys())
      #### for partnet and shapenet ####

      # #### for train dataset ####
      # category_name = ["pen_1_reformed",]
      # category_names_to_part_names = {
      #     "pen_1_reformed": None, 
      # }
      # category_names = list(category_names_to_part_names.keys())
      # #### for train dataset ####

    else:

      #### for train dataset ####
      category_name = ["eyeglasses",]
      category_names_to_part_names = {
          "eyeglasses": ["none_motion"], 
      }
      category_names = list(category_names_to_part_names.keys())
      #### for train dataset ####

    print(f"motion cat: {motion_cat}")

    dataset_train = URDFDataset(root_folder=root_folder, quantization_bits=quantization_bits, nn_max_vertices=nn_max_vertices, nn_max_faces=nn_max_faces, nn_max_permite_vertices=nn_max_permite_vertices, nn_max_permite_faces=nn_max_permite_faces, category_name=category_names, instance_nns=train_ins_nns, part_names=None, category_nm_to_part_nm=category_names_to_part_names, is_training=True, split='train')

    # ##### category part inidcator to idxes #####
    dataset_train_category_part_indicator_to_idxes = dataset_train.category_part_indicator_to_idxes


    if _use_multi_gpu:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_train)
        dataset_train = torch.utils.data.DataLoader(dataset_train,
                                                  batch_size=batch_size,
                                                  sampler=train_sampler,
                                                  num_workers=num_thread)
    else:
        # shuffle = False
        # dataet for train...
        dataset_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_thread)
    

    # dataset_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_thread)
    
    if not debug:
    #   #### for train dataset ####
    #   category_name = ["bunny_6_reformed",]
    #   category_names_to_part_names = {
    #       "bunny_6_reformed": None, 
    #   }
    #   category_names = list(category_names_to_part_names.keys())
    #   #### for train dataset ####

      #### for partnet and shapenet ####
    #   category_name = os.listdir(dataset_root_folder)
      # category_name = ["eyeglasses"] # ["data_v0_deci_0.2"] ####
      category_name = ["subd0"] # ["data_v0_deci_0.2"] ####
      ### category_name ###
      category_name = [fn for fn in category_name if os.path.isdir(os.path.join(dataset_root_folder, fn))]
      category_names_to_part_names = {
        cat_nm: None for cat_nm in category_name
      }
      category_names = list(category_names_to_part_names.keys())
      #### for partnet and shapenet ####

      # #### for train dataset ####
      # category_name = ["pen_1_reformed",]
      # category_names_to_part_names = {
      #     "pen_1_reformed": None, 
      # }
      # category_names = list(category_names_to_part_names.key s())
      # #### for train dataset ####
    else:
        #### for train dataset ####
        category_name = ["eyeglasses"]
        category_names_to_part_names = {
            "eyeglasses": ["dof_rootd_Aa001_r"]
        }
        category_names = list(category_names_to_part_names.keys())
        #### for train dataset ####

    dataset_test = URDFDataset(root_folder=root_folder, quantization_bits=quantization_bits, nn_max_vertices=nn_max_vertices, nn_max_faces=nn_max_faces, nn_max_permite_vertices=nn_max_permite_vertices, nn_max_permite_faces=nn_max_permite_faces, category_name=category_name, instance_nns=test_ins_nns, part_names=None, category_nm_to_part_nm=category_names_to_part_names, is_training=False, split='validation') # we 
    ##### dataset test #####
    dataset_test_category_part_indicator_to_idxes = dataset_test.category_part_indicator_to_idxes


    if _use_multi_gpu:
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_test)
        dataset_test = torch.utils.data.DataLoader(dataset_test,
                                                  batch_size=batch_size,
                                                  sampler=test_sampler,
                                                  num_workers=num_thread)
    else:
        # shuffle = False
        # dataet for train...
        dataset_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_thread)
    

    # shuffle = False
    # dataet for train...
    # dataset_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_thread)


    ##### Model configs ##### ### specify light weight
    if opt.model.specify_light_weight:
        vertex_module_config, face_module_config = get_model_configs(use_light_weight=True)
    else:
        vertex_module_config, face_module_config = get_model_configs()
    
    #### vertex
    vertex_module_config['use_multi_gpu'] = True
    vertex_model = VertexModel(**vertex_module_config)
    face_model = FaceModel(**face_module_config)

    vertex_model_path = opt.model.vertex_model_path
    face_model_path = opt.model.face_model_path

    if len(vertex_model_path) > 0:
        resume_from_ckpts(vertex_model, vertex_model_path)
    if len(face_model_path) > 0:
      resume_from_ckpts(face_model, face_model_path)

    #### calculate model nparams ####
    vertex_model_nparams = calculate_model_nparams(vertex_model) # model parameters
    print(f"Number of parameters in vertex_model: {vertex_model_nparams / 1e6}M.")

    if _use_multi_gpu:
      vertex_model = nn.SyncBatchNorm.convert_sync_batchnorm(vertex_model)
      vertex_model = vertex_model.cuda(local_rank)

      vertex_model = torch.nn.parallel.DistributedDataParallel(vertex_model, device_ids=[local_rank], find_unused_parameters=False)
    else: 
      vertex_model = vertex_model.cuda()

    # vertex_model = vertex_model.cuda()
    # face_model = face_model.cuda()


    ###### face model nparams ######
    face_model_nparams = calculate_model_nparams(face_model)
    print(f"Number of parameters in face_model: {face_model_nparams / 1e6}M.")

    if _use_multi_gpu:
      face_model = nn.SyncBatchNorm.convert_sync_batchnorm(face_model)
      face_model = face_model.cuda(local_rank)

      face_model = torch.nn.parallel.DistributedDataParallel(face_model, device_ids=[local_rank], find_unused_parameters=False)
    else: 
      face_model = face_model.cuda()

    # face_model = face_model.cuda()

    print("Dataset constructed!")

    # Optimization settings
    # learning_rate = 5e-4
    # training_steps = 2000 * 500
    # check_step = 5

    learning_rate = opt.loss.learning_rate
    training_steps = opt.loss.training_steps
    check_step = opt.loss.check_step

    #### Setup optimizers ####
    # vertex_optimizer = optim.Adam(vertex_model.parameters(), lr=learning_rate)
    # face_optimizer = optim.Adam(face_model.parameters(), lr=learning_rate)
    #### Setup optimizers ####

    if _use_multi_gpu:
        #### Setup optimizers ####
      vertex_optimizer = optim.Adam(vertex_model.module.parameters(), lr=learning_rate)
      #### Setup optimizers ####
    else:
      #### Setup optimizers ####
      vertex_optimizer = optim.Adam(vertex_model.parameters(), lr=learning_rate)    

    
    if _use_multi_gpu:
        #### Setup optimizers ####
      face_optimizer = optim.Adam(face_model.module.parameters(), lr=learning_rate)
      #### Setup optimizers ####
    else:
      #### Setup optimizers ####
      face_optimizer = optim.Adam(face_model.parameters(), lr=learning_rate)    

    # if os.path.exists("/nas/datasets/gen"):
    # if opt.dataset.lab == "thu": # save samples
    sv_mesh_folder = os.path.join("/nas/datasets/gen", "samples")
    os.makedirs(sv_mesh_folder, exist_ok=True)
    # sv_mesh_folder = os.path.join(sv_mesh_folder, "tables_large_model")
    sv_mesh_folder = os.path.join(sv_mesh_folder, model_descriptor)
    os.makedirs(sv_mesh_folder, exist_ok=True)

    sv_ckpts_folder = os.path.join("/nas/datasets/gen", "ckpts")
    os.makedirs(sv_ckpts_folder, exist_ok=True)
    sv_ckpts_folder = os.path.join(sv_ckpts_folder, model_descriptor)
    os.makedirs(sv_ckpts_folder, exist_ok=True)


    ### save dataset train related information ###
    dataset_train_part_indicator_to_idx_fn = os.path.join(sv_ckpts_folder, "dataset_train_part_indicator_to_idx.npy")
    np.save(dataset_train_part_indicator_to_idx_fn, dataset_train_category_part_indicator_to_idxes)
    ### save dataset test related information ###
    dataset_test_part_indicator_to_idx_fn = os.path.join(sv_ckpts_folder, "dataset_test_part_indicator_to_idx.npy")
    np.save(dataset_test_part_indicator_to_idx_fn, dataset_test_category_part_indicator_to_idxes)

    # train iters...
    # train_iter(dataset_train, dataset_test, vertex_model, face_model, vertex_optimizer, face_optimizer, training_steps, batch_size)
    print(f"exp_mode: {opt.common.exp_mode}")
    if opt.common.exp_mode == 'sampling':
        print("in test iter!")
        # train_iter(dataset_train, dataset_test, vertex_model, vertex_optimizer, training_steps, batch_size)
        test_iter(dataset_test, vertex_model, vertex_optimizer, training_steps, batch_size)
    else:
        train_iter(dataset_train, dataset_test, vertex_model, vertex_optimizer, training_steps, batch_size)
