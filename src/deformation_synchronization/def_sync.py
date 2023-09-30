# import datasets
# Shape as Points
# from email.policy import default
# import datasets_target_driven as datasets

# import datasets_target_driven_v2 as datasets_v2
# import datasets_target_driven_v3 as datasets_v3
import src.datasets.datasets_def_v5 as datasets_v3
import src.datasets.datasets_def_v5_obj as dataset_obj
# from laplacian import mesh_laplacian_raw
import torch
import numpy as np
# import visdom
# import network
# import network_target_driven_prob as network
# import network_target_driven_paired_data_hier_deform as network_paired_data
# import network_target_driven_paired_data_hier_deform_v2 as network_paired_data_codebook
# import network_target_driven_paired_data_hier_deform_v6 as network_paired_data
import src.networks.network_with_cages as network_paired_data
# import network_with_cages as network_paired_data
from utils import normalie_pc_bbox_batched, save_obj_file, weights_init
import argparse
import shutil
import logging
from tqdm import tqdm
import os
from pathlib import Path
import datetime
# import losses

from src.common_utils.losses import chamfer_distance_raw as chamfer_distance


import src.common_utils.data_utils_torch as data_utils

# from network import model as glb_deform_model

from torch import autograd


import utils
from torch_batch_svd import svd
# from render import render_mesh
from torch.autograd import Variable
# from torchvision.utils import save_image

# from losses import laplacian_loss, normal_loss, normal_loss_raw
import random
from src.networks.neuralcages_networks import deform_with_MVC

import src.common_utils.collision_sim as collision_sim


#### dataset for multiple parts ####
def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=42,
                        help='Batch Size during training')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers for data loader')
    parser.add_argument('--epoch',  default=30, type=int, help='Epoch to run')
    parser.add_argument('--learning_rate', default=0.0001,
                        type=float, help='Initial learning rate')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float,
                        default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--step_size', type=int,  default=5,
                        help='Decay step for lr decay')
    parser.add_argument('--lr_decay', type=float,
                        default=0.5, help='Decay rate for lr decay')
    parser.add_argument('--gpu', type=str, default='0,1,2', help='GPU to use') 
    parser.add_argument('--display', type=int, default=10,
                        help='number of iteration per display')
    parser.add_argument('--env_name', type=str,
                        default='train_chair', help='enviroment name of visdom')
    parser.add_argument('--num_basis', type=int, default=15,
                        help='number of basis vectors')
    parser.add_argument('--num_keypoints', type=int, default=256,
                        help='number of basis vectors')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='number of basis vectors')
    parser.add_argument('--with-discriminator', default=False, action='store_true',
                        help='whether to use discriminator')
    parser.add_argument('--kl-weight', type=float,
                        default=0.001, help='Weigth for KL divergence')
    parser.add_argument('--exp_flag', type=str,
                        default='exp', help='enviroment name of visdom')
    parser.add_argument('--data_dir', type=str,
                        default="/home/xueyi/gen/DeepMetaHandles/data/toy_eye_tot", help='data dir for pair-wise deformation data...')
    parser.add_argument('--use-prob', default=False, action='store_true',
                        help='whether to use discriminator')
    parser.add_argument('--load_meta', default=False, action='store_true',
                        help='whether to use discriminator')
    parser.add_argument('--n_keypoints', type=int, default=256,
                        help='number of basis vectors')
    parser.add_argument('--tar_basis', type=int, default=1,
                        help='number of basis vectors')
    parser.add_argument('--use_paired_data', default=False, action='store_true',
                        help='number of basis vectors')
    parser.add_argument('--pred_type', type=str,
                        default='basis', help='enviroment name of visdom')
    # coef_multiplier
    parser.add_argument('--coef_multiplier', type=float,
                        default=0.1, help='Decay rate for lr decay')
    parser.add_argument('--with_discern_loss', default=False, action='store_true',
                        help='number of basis vectors')
    parser.add_argument('--random_scaling', default=False, action='store_true',
                        help='number of basis vectors')
    parser.add_argument('--use_pointnet2', default=False, action='store_true',
                        help='number of basis vectors')
    parser.add_argument('--use_graphconv', default=False, action='store_true',
                        help='number of basis vectors')
    parser.add_argument('--use_pp_tar_out_feat', default=False, action='store_true',
                        help='number of basis vectors')
    parser.add_argument('--sv_ckpt_freq', type=int, default=100,
                        help='number of basis vectors')
    parser.add_argument('--neighbouring_k', type=int, default=128,
                        help='number of basis vectors')
    # n_layers ## checkpoints
    parser.add_argument('--net_path', type=str,
                        default="", help='data dir for pair-wise deformation data...')
    parser.add_argument('--net_glb_path', type=str,
                        default="", help='data dir for pair-wise deformation data...')
    parser.add_argument('--n_layers', type=int, default=3,
                        help='number of basis vectors')
    parser.add_argument('--dataset_name', type=str,
                        default='v1', help='enviroment name of visdom')
    parser.add_argument('--n_samples', type=int, default=1024,
                        help='number of basis vectors')
    parser.add_argument('--symmetry_axis', type=int, default=1,
                        help='number of basis vectors')
    parser.add_argument('--local_handle', default=False, action='store_true',
                        help='number of basis vectors')
    parser.add_argument('--hier_stage', type=int, default=0,
                        help='number of basis vectors')
    parser.add_argument('--with_cat_deform', type=int, default=1,
                        help='number of basis vectors')
    parser.add_argument('--wo_keypts_abs', default=False, action='store_true',
                        help='number of basis vectors')
    parser.add_argument('--test_only', default=False, action='store_true',
                        help='number of basis vectors')
    parser.add_argument('--use_prob_src', default=False, action='store_true',
                        help='number of basis vectors')
    parser.add_argument('--use_codebook', default=False, action='store_true',
                        help='number of basis vectors') # with_recon
    parser.add_argument('--with_recon', default=False, action='store_true',
                        help='number of basis vectors')
    parser.add_argument('--def_verts', default=False, action='store_true',
                        help='number of basis vectors')
    parser.add_argument('--lap_lambda', type=float,
                        default=0.2, help='Decay rate for lr decay')
    parser.add_argument('--sym_lambda', type=float,
                        default=1.0, help='Decay rate for lr decay')
    parser.add_argument('--use_cvx_feats', default=False, action='store_true',
                        help='number of basis vectors')
    parser.add_argument('--use_def_pc', default=False, action='store_true',
                        help='number of basis vectors')
    parser.add_argument('--use_trans_range', default=False, action='store_true',
                        help='number of basis vectors')
    parser.add_argument('--use_delta_prob', default=False, action='store_true',
                        help='number of basis vectors')
    parser.add_argument('--use_cond_vae', default=False, action='store_true',
                        help='number of basis vectors')
    parser.add_argument('--use_cond_vae_cvx_feats', default=False, action='store_true',
                        help='number of basis vectors')
    parser.add_argument('--use_vae_opt', type=str,
                        default='cvx', help='enviroment name of visdom')
    parser.add_argument('--pred_positions', default=False, action='store_true',
                        help='number of basis vectors')
    parser.add_argument('--src_folder_fn', type=str,
                        default='src', help='enviroment name of visdom')
    parser.add_argument('--dst_folder_fn', type=str,
                        default='dst', help='enviroment name of visdom')
    parser.add_argument('--decode_cond_pos', default=False, action='store_true',
                        help='number of basis vectors')
    parser.add_argument('--use_recon_as_cd', default=False, action='store_true',
                        help='number of basis vectors')
    parser.add_argument('--cond_tar_pc', default=False, action='store_true',
                        help='number of basis vectors')
    parser.add_argument('--recon_cond_tar', default=False, action='store_true',
                        help='number of basis vectors')
    parser.add_argument('--recon_cond', type=str,
                        default='cvx', help='enviroment name of visdom')
    parser.add_argument('--cvx_dim', type=int, default=-1,
                        help='number of basis vectors')
    parser.add_argument('--only_tar', default=False, action='store_true',
                        help='number of basis vectors')
    parser.add_argument('--test_recon', default=False, action='store_true',
                        help='number of basis vectors')  
    parser.add_argument('--recon_normalized_pts', default=False, action='store_true',  
                        help='number of basis vectors') 
    parser.add_argument('--tar_n_samples', type=int, default=1024,
                        help='number of basis vectors') 
    parser.add_argument('--debug', default=False, action='store_true',  
                        help='number of basis vectors') 
    parser.add_argument('--n_pts_per_convex', type=int, default=256,
                        help='number of basis vectors') 
    parser.add_argument('--cvx_to_pts_sufix', type=str, default="_cvx_to_verts_cdim_128.npy",
                        help='number of basis vectors') 
    parser.add_argument('--ckpt', type=str, default="", 
                        help='number of basis vectors') 
    parser.add_argument('--with_glb', default=False, action='store_true', 
                        help='number of basis vectors') 
    parser.add_argument('--one_shp', default=False, action='store_true', 
                        help='number of basis vectors') 
    parser.add_argument('--with_dis_loss', default=False, action='store_true', 
                        help='number of basis vectors')
    parser.add_argument('--only_src_cvx', default=False, action='store_true', 
                        help='number of basis vectors')
    parser.add_argument('--def_version', type=str, default="v3", 
                        help='number of basis vectors') 
    parser.add_argument('--dis_factor', type=float,
                        default=2.0, help='Decay rate for lr decay')
    parser.add_argument('--pred_offset', default=False, action='store_true', 
                    help='number of basis vectors') 
    parser.add_argument('--small_tar_nn', default=False, action='store_true', 
                    help='number of basis vectors') 
    parser.add_argument("--gravity_weight", type=float, help="center of cage == center of shape", default=0)
    parser.add_argument('--template', type=str, default="./data/sphere_V42_F80.off", 
                        help='number of basis vectors')  
    parser.add_argument('--glb_template', type=str, default="/home/xueyi/gen/DeepMetaHandles/data/sphere_template_2.obj", 
                        help='number of basis vectors') 
    parser.add_argument('--n_fold', type=int, default=1,
                        help='number of basis vectors')
    parser.add_argument("--normalization", type=str, choices=["batch", "instance", "none"], default="none")
    parser.add_argument("--concat_prim", action="store_true", help="concatenate template coordinate in every layer of decoder")
    parser.add_argument("--by_part", action="store_true", help="concatenate template coordinate in every layer of decoder")
    parser.add_argument("--use_gt_cages", action="store_true", help="concatenate template coordinate in every layer of decoder")
    parser.add_argument("--bais_per_vert", action="store_true", help="concatenate template coordinate in every layer of decoder")
    parser.add_argument("--optimize_coef", action="store_true", help="concatenate template coordinate in every layer of decoder")
    parser.add_argument('--cd_weight', type=float,
                        default=1.0, help='Decay rate for lr decay')
    parser.add_argument("--optimize_basis", action="store_true", help="concatenate template coordinate in every layer of decoder")
    parser.add_argument('--rnd_sample_nn', type=int, default=0,
                        help='number of basis vectors')
    parser.add_argument("--cvx_folder_fn", type=str, default="")
    parser.add_argument("--obj_data_root_folder", type=str, default="")
    parser.add_argument('--n_parts', type=int, default=2,
                        help='number of basis vectors') 
    parser.add_argument("--cvx_to_verts_sufix", type=str, default="")
    parser.add_argument('--n_shots', type=int, default=-1,
                        help='number of basis vectors')
    parser.add_argument("--new_glb_template", type=str, default="")
    parser.add_argument('--src_index', type=int, default=-1,
                        help='number of basis vectors') 
    parser.add_argument('--cvx_list_filter', default=False, action='store_true',
                    help='number of basis vectors') 
    return parser.parse_args() 



opt = parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

if "ShapeNet" in opt.data_dir:
    opt.symmetry_axis = 0
else:
    opt.symmetry_axis = 1

experiment_dir = Path('./log/')
experiment_dir.mkdir(exist_ok=True)

if opt.log_dir is None:
    log_name = str(datetime.datetime.now().strftime(
        '%m-%d_%H-%M_')) + opt.env_name
    experiment_dir = experiment_dir.joinpath(log_name)
else:
    experiment_dir = experiment_dir.joinpath(opt.log_dir)
experiment_dir.mkdir(exist_ok=True)


checkpoints_dir = "./ckpts"
os.makedirs(checkpoints_dir, exist_ok=True)



log_dir = experiment_dir.joinpath('logs/')
log_dir.mkdir(exist_ok=True)
if opt.log_dir is None:
    shutil.copy('src/datasets_target_driven.py', str(experiment_dir))
    shutil.copy('src/losses.py', str(experiment_dir))
    shutil.copy('src/network.py', str(experiment_dir))
    shutil.copy('src/pointnet_utils.py', str(experiment_dir))
    shutil.copy('src/train_target_driven.py', str(experiment_dir))
    shutil.copy('src/utils.py', str(experiment_dir))
    shutil.copy('src/laplacian.py', str(experiment_dir))



def log_string(str):
    logger.info(str)
    print(str)


logger = logging.getLogger("Model")
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('%s/log.txt' % log_dir)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
log_string('PARAMETER ...')
log_string(opt)



data_dir = opt.data_dir
src_folder_fns = opt.src_folder_fn.split(";")
dst_folder_fns = opt.dst_folder_fn.split(";")

### part-data-dirs ###
part_data_dirs = opt.data_dir.split(";")
part_cvx_to_pts_sufix = opt.cvx_to_pts_sufix.split(";")
part_dataset_train = []
part_dataloader_train = []

# datasets_obj
# n_keypoints = [64, 16]
n_keypoints = [64, 64, 64, 64]

for i_part, cur_part_dir in enumerate(part_data_dirs):
  cur_part_cvx_to_pts_sufix = part_cvx_to_pts_sufix[i_part] ## i_parts 
  ### cur_part_dataset_train ###
  print(f"Loading from {cur_part_dir}")
  
  cur_part_dataset_train = datasets_v3.ChairDataset("train", cur_part_dir, split="train", opt=opt, cvx_to_pts_sufix=cur_part_cvx_to_pts_sufix, n_keypoints=n_keypoints[i_part], src_folder_fn=src_folder_fns[i_part], dst_folder_fn=dst_folder_fns[i_part])
  
  part_dataset_train.append(cur_part_dataset_train)
  
  cur_part_dataloader_train = torch.utils.data.DataLoader(cur_part_dataset_train, collate_fn=datasets_v3.my_collate, drop_last=True, shuffle=True, batch_size=opt.batch_size, num_workers=opt.num_workers, worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))
  part_dataloader_train.append(cur_part_dataloader_train)



if ";" in opt.glb_template:
    part_glb_template = opt.glb_template.split(";")
else:
    part_glb_template = [opt.glb_template for _ in range(opt.n_parts)]
    
part_networks = []
# part_networksD = []
for i_part in range(len(part_data_dirs)):
  ## ImageRender
  cur_part_glb_template = part_glb_template[i_part]
  opt.glb_template = cur_part_glb_template
  net = network_paired_data.model(opt.num_basis, opt=opt).cuda()
  part_networks.append(net)



def safe_load_ckpt_common(model, state_dicts, weight_flag=None):
    ori_dict = state_dicts
    part_dict = dict()
    model_dict = model.state_dict()
    tot_params_n = 0
    for k in ori_dict:
        if k in model_dict and ( (weight_flag is not None and weight_flag in k) or weight_flag is None):
            v = ori_dict[k]
            part_dict[k] = v.clone()
            tot_params_n += 1
    model_dict.update(part_dict)
    model.load_state_dict(model_dict)
    print(f"Resumed with {tot_params_n} parameters...")




# Object dataset and data_dir # ### data_dir
data_dir = opt.data_dir ### train, data_dir, opts ###
dataset_train = dataset_obj.ChairDataset("train", data_dir, opt=opt)


opt.train_batch_size = 1
dataloader_train = torch.utils.data.DataLoader(dataset_train, collate_fn=dataset_obj.my_collate, shuffle=True, batch_size=opt.train_batch_size, num_workers=0, worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))


    
log_string('No existing model, starting training from scratch...')
start_epoch = 0
cd_curve = []
sym_curve = []
lap_curve = []
nor_curve = []
tot_curve = []
g_curve = []
d_curve = []


part_net_optimizers = []
# part_netD_opti

part_net_paths = opt.net_path.split(";")

for i_part, cur_net_path in enumerate(part_net_paths):
    part_networks[i_part] = part_networks[i_part].apply(weights_init)
    # part_networksD[i_part] = part_networksD[i_part].apply(weights_init)
    ###### net_state_dict xxx ######
    # net_state_dict = torch.load(opt.net_path, map_location="cpu", )
    print(f"Loading from {cur_net_path}")
    net_state_dict = torch.load(cur_net_path, map_location='cpu')
    # safe_load_ckpt_common(net, net_state_dict, weight_flag=weight_flag)
    # safe_load_ckpt_common(net, net_state_dict) ### laod net path ###
    safe_load_ckpt_common(part_networks[i_part], net_state_dict) ### laod net path ###

    part_networks[i_part].glb_cages.setup_template(opt.new_glb_template)
    
    cur_part_net_optimizer = torch.optim.Adam(
        part_networks[i_part].parameters(),
        lr=opt.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=opt.decay_rate)
    part_net_optimizers.append(cur_part_net_optimizer) ### part_net_optimizer ###


def get_def_pcs_by_cages(tot_bsz_part_cages, part_pc, tot_part_basis, tot_bsz_coefs, part_cvx_pts_idxes, cur_data_dir):
    # part_cvx_pts_idxes:list of cvx pts idxes 
    ### part_cvx_pts_idxes: the list of convex pts ### ### idx 
    ''' Instance level synchronization '''
    
    # part_cages: bsz x nn_cage_pts x 3 for each cage in part_cages
    # part_def_cages: bsz x nn_cage_pts x 3 for each cage in part_def_cages
    # coefs: bsz x n_basis #
    # coefs: bsz x n_basis # 
    
    
    
    ## src_pc_cvx_indicator: bsz x nn_pts x nn_cvx ##
    ## 
    
    obj_idx  = 0
    part_cages = tot_bsz_part_cages[obj_idx]
    coefs = tot_bsz_coefs[obj_idx]
    part_basis = tot_part_basis[obj_idx]
    part_basis = part_basis.squeeze()# .unsqueeze(0)
    coefs = coefs.squeeze().unsqueeze(0)
    
    # print()
    
    nn_cvx = len(part_cages) ## 
    # part_cages  
    nn_cage_pts = part_cages[0].size(1)
    # part_cages = part_cages[obj_idx]
    cur_obj_cvx_pts_idxes = part_cvx_pts_idxes[obj_idx] ## list of cvx idexs
    
    cur_obj_src_nn_pts = part_pc.size(1)
    cur_obj_pc_offsets = torch.zeros((1, cur_obj_src_nn_pts, 3), dtype=torch.float32).cuda()
    cur_obj_pc_add_nns = torch.zeros((1, cur_obj_src_nn_pts), dtype=torch.float32).cuda()
    for i_cvx in range(nn_cvx):
        # cur_src_pc_cvx_indicator = src_pc_cvx_indicator[:, :, i_cvx]
        # cur_cvx_nn = int(cur_src_pc_cvx_indicator.)
        cur_cvx_cage = part_cages[i_cvx] # bsz x nn_cages_pts x 3 --> 
        cur_cvx_basis = part_basis[i_cvx].unsqueeze(0) ## sorted basis ### bsz x n_basis x (n_pts x 3)
        # print(f"part_basis: {part_basis.size()}, coefs: {coefs.size()}")
        # print(f"coefs_requires_grad: {coefs.requires_grad}, cur_cvx_basis: {cur_cvx_basis.size()}, coefs: {coefs.size()}")
        # print()
        cur_cvx_cage_offset = torch.sum((cur_cvx_basis * coefs.unsqueeze(-1)), dim=1) #### bsz x (n_pts x 3)
        n_bsz, _ = cur_cvx_cage_offset.size(0), cur_cvx_cage_offset.size(1)
        
        cur_cvx_cage_offset = cur_cvx_cage_offset.contiguous().view(n_bsz, nn_cage_pts, -1) ### bsz x nn_cage_pts x 3 --> offset of the cage
        ### offset pts ###
        # print(f"cur_cvx_cage: {cur_cvx_cage.size()}, cur_cvx_cage_offset: {cur_cvx_cage_offset.size()}")
        cur_cvx_def_cage = cur_cvx_cage + cur_cvx_cage_offset ## bsz x nn_cage_pts x 3 --> def_cage
        ### get cvx pts idxes from obj cvx pts idxes ###
        cur_cvx_pts_idxes = cur_obj_cvx_pts_idxes[i_cvx]
        cur_cvx_ori_pts = part_pc[obj_idx, cur_cvx_pts_idxes].unsqueeze(0) ## 1 x nn_cvx_pts x 3 
        ## should use the part-network for deformation ##
        
        cur_cvx_ori_pts, cur_cvx_center, cur_cvx_scale = utils.normalie_pc_bbox_batched(cur_cvx_ori_pts, rt_stats=True) ### 1 x nn_cvx_pts x 3 ### 
        # print(f"cur_cvx_cage: {cur_cvx_cage.size()}, cur_cvx_def_cage: {cur_cvx_def_cage.size()}, cur_cvx_ori_pts: {cur_cvx_ori_pts.size()}")
        cur_cvx_def_pts, rnd_weights, rnd_weights_unnormed = deform_with_MVC(cur_cvx_cage,
                                                                         cur_cvx_def_cage,
                                                                         net.cages.cage_network.template_faces.expand(1,-1,-1),
                                                                         cur_cvx_ori_pts,
                                                                         verbose=True)
        cur_cvx_def_offset = cur_cvx_def_pts - cur_cvx_ori_pts ### ori_pts 
        cur_cvx_def_offset = cur_cvx_def_offset * cur_cvx_scale ### scale of the cvx pts 
        # print(f"cur_cvx_def_offset: {cur_cvx_def_offset.size()}, cur_cvx_pts_idxes: {cur_cvx_pts_idxes.size()}")
        cur_obj_pc_offsets[:, cur_cvx_pts_idxes] += cur_cvx_def_offset
        cur_obj_pc_add_nns[:, cur_cvx_pts_idxes] += 1
    cur_obj_pc_offsets  = cur_obj_pc_offsets / torch.clamp(cur_obj_pc_add_nns.unsqueeze(-1), min=1e-3) 
    cur_obj_def_pc = part_pc + cur_obj_pc_offsets ### bsz x nn_pts x 3 --> deformed pts ###
    cage_of_part_pcs = net.glb_cages.cage_network.get_cages(part_pc[0: 0 + 1])
    cage_of_def_part_pcs = net.glb_cages.cage_network.get_cages(cur_obj_def_pc) ### cage of def pcs 
    
    # print(f"cage_of_def_part_pcs_requires_grad: {cage_of_def_part_pcs.requires_grad}")
    
        
    
    ### eyeglasses and scaling factors ###
    if "eyeglasses" in cur_data_dir and "none_motion" in cur_data_dir:
        extents_mult = torch.tensor([1.0, 0.7, 1.0], dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(0)
    elif "SAPIEN" in cur_data_dir:
        extents_mult = torch.tensor([0.8, 1.0, 0.8], dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(0)
    else:
        extents_mult = torch.tensor([0.8, 0.8, 0.8], dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(0)
    # print(f"merged_cages: {merged_cages.size()}, merged_new_cages: {merged_new_cages.size()}, merged_faces: {merged_faces.size()}")
    ## merged_def
    ### to the current cage ###
    
    # cur_glb_rnd_cage = cur_bsz_glb_tot_rnd_new_cages[i_s]
    # print(f"for cages: net.glb_cages.cage_network.template_faces{net.glb_cages.cage_network.template_faces.size()}, cage_of_def_part_pcs: {cage_of_def_part_pcs.size()}, cage_of_part_pcs: {cage_of_part_pcs.size()}")
    cur_part_def_pcs, _, _ = deform_with_MVC(
        cage_of_part_pcs,
        cage_of_def_part_pcs, ## use this cage for deformation ##
        net.glb_cages.cage_network.template_faces.expand(-1,-1,-1), ### deform with pcsj
        # src_pc[i_bsz: i_bsz + 1] * extents_mult, # .transpose(1,2).contiguous(),
        part_pc * extents_mult,
        # weights=cur_bsz_cage_weights_tot,
        verbose=True,)
    cur_part_def_pcs /= extents_mult
    return cur_part_def_pcs, cur_obj_def_pc, cage_of_part_pcs, cage_of_def_part_pcs ### def pcs of the current part 



def get_assembled_pcs(tot_def_pcs, tot_mesh_faces, tot_scale_infos, tot_position_infos):
    assembled_pcs = []
    assembled_verts = []
    assembled_faces = []
    assembled_detection_pts = []
    n_pts_detection = 512 # 
    sim_weight = 1e-4
    assembled_det_vs = []
    ## farthest_point_sampling --> 
    for i_p in range(opt.n_parts): ### total number of parts ###
        # cur_part_registered_pc: nn_pts x 3 ---> total sampled pcs ###
        # cur_part_def_mesh_pc = tot_def_pcs[i_p]
        # cur_part_sampled_pc_fps_idx = data_utils.farthest_point_sampling(cur_part_def_mesh_pc, n_sampling=n_pts_detection)
        # cur_part_registered_pc = cur_part_def_mesh_pc[:, cur_part_sampled_pc_fps_idx]
        # cur_part_sampled_mesh = tot_recon_meshes[i_p]
            
        cur_part_v = tot_def_pcs[i_p]
        cur_part_f = tot_mesh_faces[i_p]
            
        cur_part_cur_bsz_scale = tot_scale_infos[obj_idx, i_p]
        cur_part_cur_bsz_pos_offset = tot_position_infos[obj_idx, i_p]
    
        # to target scale #
        # cur_part_v = normalie_pc_bbox_batched(cur_part_v.unsqueeze(0)).squeeze(0)
        cur_part_v = normalie_pc_bbox_batched(cur_part_v)
        ###### target scale ######
        # cur_part_v = data_utils.scale_vertices_to_target_scale(cur_part_v, cur_part_cur_bsz_scale.unsqueeze(0)).squeeze(0)
        cur_part_v = data_utils.scale_vertices_to_target_scale(cur_part_v, cur_part_cur_bsz_scale.unsqueeze(0)).squeeze(0)
        ###### target scale ######

        cur_part_v = cur_part_v + cur_part_cur_bsz_pos_offset.unsqueeze(0) ### offset info for assembling 
        # cur_part_assembled_verts.append(cur_part_v)
        cur_part_v = cur_part_v.unsqueeze(0)
        
        # cur_part_assembled_faces.append(cur_part_f)
        
        # print(f"cur_part_cur_bsz_v: {cur_part_v.size()}") ## 
        if cur_part_v.size(1) > n_pts_detection:
            # print(f"cur_part_cur_bsz_v: {cur_part_v.size()}")
            sampled_cur_part_verts_fps_idx = data_utils.farthest_point_sampling(cur_part_v, n_sampling=n_pts_detection)
            cur_part_det_v = cur_part_v[0, sampled_cur_part_verts_fps_idx]
        else:
            cur_part_det_v = cur_part_v[0]
            
        # cur_part_v = 
        # print(f"maxx_cur_part_v: {torch.max(cur_part_v, dim=1)}, minn_cur_part_v: {torch.min(cur_part_v, dim=1)}, maxx_cur_part_det_v: {torch.max(cur_part_det_v, dim=0)}, minn_cur_part_det_v: {torch.min(cur_part_det_v, dim=0)}")
        cur_part_v = cur_part_v[0]
            
        assembled_det_vs.append(cur_part_det_v)
        assembled_detection_pts.append(cur_part_det_v)
        assembled_verts.append(cur_part_v)
        assembled_faces.append(cur_part_f)
    return assembled_det_vs, assembled_detection_pts, assembled_verts, assembled_faces



### part def pc ###
LEARNING_RATE_CLIP = 1e-7
MOMENTUM_ORIGINAL = 0.1
MOMENTUM_DECCAY = 0.5
MOMENTUM_DECCAY_STEP = opt.step_size

current_epoch = 0
adversarial_loss = torch.nn.BCELoss()



if opt.test_only:
    opt.epoch = start_epoch + 1


##### add exp_flag to the samples dir's name #####
use_prob_indicator = 1 if opt.use_prob else 0 #### use_prob..
use_prob_src_indicator = 1 if opt.use_prob_src else 0



model_flag = f"obj_nd2_{opt.exp_flag}_{opt.learning_rate}_kl_{opt.kl_weight}_num_basis_{opt.num_basis}_n_keypoints_{opt.n_keypoints}_"
# samples_dir = f"{opt.exp_flag}_tar_d_glb2_prob_samples_lr_{opt.learning_rate}_lap_{lap_lambda}_rng_{rng_coeff}_kl_{opt.kl_weight}_use_prob_{use_prob_indicator}_discern_loss_{opt.with_discern_loss}_caling_{opt.random_scaling}_tar_basis_{opt.tar_basis}_num_basis_{opt.num_basis}_n_keypoints_{opt.n_keypoints}_neighbouring_k_{opt.neighbouring_k}_n_layers_{opt.n_layers}_use_pointnet2_{opt.use_pointnet2}_use_pp_feat_{opt.use_pp_tar_out_feat}"


samples_dir = model_flag
samples_root_dir = "./samples"
os.makedirs(samples_root_dir, exist_ok=True)
samples_dir = os.path.join(samples_root_dir, samples_dir)
os.makedirs(samples_dir, exist_ok=True)



tot_inst = 0

tot_collision_losses = []
glb_net = None
# for epoch in range(start_epoch, opt.epoch):
    # log_string('Epoch %d (%d/%s):' % (current_epoch + 1, epoch + 1, opt.epoch))
    # '''Adjust learning rate and BN momentum'''

    # lr = max(opt.learning_rate * (opt.lr_decay **
    #         (epoch // opt.step_size)), LEARNING_RATE_CLIP)
    # log_string('Learning rate:%f' % lr)

    # for cur_optimizer in part_net_optimizers:
    #     for param_group in cur_optimizer.param_groups:
    #         param_group['lr'] = lr

    # momentum = MOMENTUM_ORIGINAL * \
    #     (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
            
    # if opt.test_only or momentum < 0.01:
        # momentum = 0.01

    # print('BN momentum updated to: %f' % momentum)
    # if not len(opt.net_path) > 0:

    # momentum = 0.01
    
for part_net in part_networks:
    # part_net = part_net.apply(lambda x: bn_momentum_adjust(x, momentum))
    # part_net = part_net.apply(lambda x: bn_momentum_adjust(x, momentum))
    # part_net.train()
    part_net.eval()

buf = {"cd" : [], "cage_shift": [], "tar_cage_shift": [], "tot": [], "tar_extents": [], "tar_offset": [], "orth": [], "sp": [], "svd": [], "cat_offset": []
        } 

src_names = []
tar_names = []

### from object to parts ###
tot_n_data_obj = len(part_dataset_train[0])

data_idx_to_part_idx_to_S_mtx_z = {}
for src_idx in range(tot_n_data_obj):
    part_idx_to_S_mtx_z = {}
    for i_part, cur_part_dataset in enumerate(part_dataset_train):
        # cur_src_tot_bases, cur_src_tot_def_coefs #
        cur_src_tot_bases = []
        cur_src_tot_def_coefs = []
        for dst_idx in range(tot_n_data_obj):
            if dst_idx == src_idx:
                continue
            cur_part_rt_dict = cur_part_dataset.__getitem__(src_idx, dst_idx) ## def_cages_obj... ### 8 shots ###
            cur_part_rt_dict = datasets_v3.my_collate([cur_part_rt_dict])
            # nbsz = 1
            cur_part_network = part_networks[i_part]
            src_pc = cur_part_rt_dict["src_pc"]
            tar_pc = cur_part_rt_dict["tar_pc"]
            key_pts = cur_part_rt_dict["key_pts"]
            key_pts = cur_part_rt_dict["tar_pc"]
            w_mesh = cur_part_rt_dict["w_mesh"]
            w_pc = cur_part_rt_dict['w_pc']
            src_verts = cur_part_rt_dict['src_ver']
            tar_verts = cur_part_rt_dict['tar_ver']
            src_faces = cur_part_rt_dict['src_face']
            tar_faces = cur_part_rt_dict['tar_face']
            src_cvx_to_pts = cur_part_rt_dict['src_cvx_to_pts']
            dst_cvx_to_pts = cur_part_rt_dict['dst_cvx_to_pts']
            # print(f"src_pc: {src_pc.size()}, tar_pc: {tar_pc.size()}")
            
            # rt_dict = cur_part_network.forward7(src_pc, tar_pc, key_pts, key_pts, w_pc,  src_verts, src_cvx_to_pts , tar_verts, dst_cvx_to_pts)

            mesh_rt_dict = cur_part_network.forward7(src_verts[0].unsqueeze(0), tar_verts[0].unsqueeze(0), key_pts, key_pts, w_pc,  src_verts, src_cvx_to_pts, tar_verts, dst_cvx_to_pts)
            # def cvx to p ts # 
            
            cur_part_cur_dst_cage_basis = cur_part_rt_dict["tot_cage_basis"]
            cur_part_cur_dst_cage_coefs = cur_part_rt_dict["tot_cage_coefs"]
            # cur_src_tot_bases, cur_src_tot_def_coefs #
            cur_src_tot_bases.append(cur_part_cur_dst_cage_basis[0])
            cur_src_tot_def_coefs.append(cur_part_cur_dst_cage_coefs[0])
        # tot_data_N x tot_cvx x nn_bases x 3  # B_cms 
        # tot_data_N x tot_cvx x nn_bases # y_cms
        # 
        B_cms = cur_src_tot_bases[0] # tot_cvx x nn_bases x nn_pts x 3 
        N_cvx, N_bases, N_pts = B_cms.size(0), B_cms.size(1), B_cms.size(2)
        B_cms = B_cms.contiguous().view(N_cvx, N_bases, -1).contiguous().permute(0, 2, 1).contiguous() # 
        y_cms = torch.stack(cur_src_tot_def_coefs, dim=0) ### tot_data_N x tot_cvx x nn_bases  #
        
        S_cms = torch.eye(N_bases).to(B_cms.device).unsqueeze(0).repeat(N_cvx, 1, 1).contiguous() # N_cvx x N_bases x N_bases
        z = torch.mean(y_cms, dim=1) # tot_data_N x nn_bases 
        N_data = y_cms.size(0)
        # initialize 
        N_tot_iters = 100
        for i_iter in range(N_tot_iters):
            new_S_cms = []
            for i_cvx in range(N_cvx):
                cur_cvx_y = y_cms[:, i_cvx, :] # tot_data_N x N_bases
                # https://pytorch.org/docs/stable/generated/torch.linalg.svd.html#torch.linalg.svd
                u, sigma, vt = torch.linalg.svd(z)
                u_m, sigma_m, vt_m = torch.linalg.svd(cur_cvx_y)
                # U[:, :3] @ torch.diag(S) @ Vh
                cur_cvx_S = u_m @ torch.diag(sigma_m) @ vt_m @ vt.transpose(1, 0) @ torch.inv(torch.diag(sigma)) @ u.transpose(1, 0)
                new_S_cms.append(cur_cvx_S)
            new_S_cms = torch.cat(new_S_cms, dim=0) ### tot_cvx x N_bases x N_base 
            S_cms = new_S_cms
            
            new_z = []
            for i_n in range(N_data):
                cur_data_def_coefs = []
                for i_cvx in range(N_cvx):
                    new_z_by_cvx = torch.linalg.lstsq(S_cms[i_cvx], y_cms[i_n, i_cvx].unsqueeze(-1)).solution
                    new_z_by_cvx = new_z_by_cvx.squeeze(-1) # n_basis
                    cur_data_def_coefs.append(new_z_by_cvx)
                cur_data_def_coefs = torch.stack(cur_data_def_coefs, dim=0) # N_cvx x N_basis
                cur_data_def_coefs = torch.mean(cur_data_def_coefs, dim=0)
                new_z.append(cur_data_def_coefs)
            new_z = torch.stack(new_z, dim=0) # N_data x N_basis
            z = new_z
            
        part_idx_to_S_mtx_z[i_part] = {'S_cms': S_cms.detach().cpu().numpy(), 'z': z.detach().cpu().numpy()}
    data_idx_to_part_idx_to_S_mtx_z[src_idx] = part_idx_to_S_mtx_z

data_idx_to_part_idx_to_S_mtx_z_sv_path = "./data/data_idx_to_part_idx_to_S_mtx_z.npy"
np.save(data_idx_to_part_idx_to_S_mtx_z_sv_path, data_idx_to_part_idx_to_S_mtx_z)

    
# for i, data in tqdm(enumerate(dataloader_train), total=len(dataloader_train), smoothing=0.9): ### dataloader_train and dataset_train
#     # src_pc = data['src_pc']
#     # tar_pc = data['tar_pc']
#     print(f"Total avg collision losses: {sum(tot_collision_losses) / max(len(tot_collision_losses), 1)}!")
#     tot_scale_infos = data['tar_scales'] #### bsz x n_parts x 1 --> the scale
#     tot_position_infos = data['tar_positions'] #### bsz x n_parts x 3 ---> parts position information ---> the offset information ### tar positions ###
#     tot_joint_infos = data["joint_infos"] ### randomly get one instance and one part...
    
#     obj_idxes = data["idx"] ### obj_idxes --> obj_idxes
    
#     # print(f"obj_idxes: {obj_idxes}")
    
#     # object structure ##
#     obj_idx = 0
#     tot_part_cvx_to_bbox_center = []
#     tot_part_cvx_to_pc  = []
#     tot_part_cvx_to_cvx_pts = []
#     tot_part_shp_pts = []
    
#     # cur_data_idx = int(obj_idxes[obj_idx].item())
#     # ori_data_idx = random.choice(range(len(part_dataset_train[0])))

#     # src_idx = i ### source index ###
#     # dst_idx = random.choice(range(tot_n_data_obj))
    
#     src_idx = random.choice(range(len(part_dataset_train[0]))) ## src_n_models ###
#     dst_idx = int(obj_idxes[obj_idx].item())
#     tot_part_rt_dict = []
#     tot_def_pcs = []
#     tot_mesh_faces = []
#     ### GET obj-level info ###
#     tot_part_coefs = []
#     tot_part_cages_ori = []
#     tot_part_src_pcs = []
#     tot_part_sorted_basis = []
#     tot_part_cvx_pts_idxes = []
#     tot_part_cvx_nn_info = []
    
#     tot_rnd_def_pcs = [[] for i_rnd in range(opt.rnd_sample_nn)]
    
#     # src_idx = 1 # 0
#     src_idx = opt.src_index if opt.src_index >= 0 else src_idx
#     print(f"src_idx: {src_idx}, dst_idx: {dst_idx}")
    
#     for i_part, cur_part_dataset in enumerate(part_dataset_train):
#         cur_part_rt_dict = cur_part_dataset.__getitem__(src_idx, dst_idx) ## def_cages_obj... ### 8 shots ###
#         cur_part_rt_dict = datasets_v3.my_collate([cur_part_rt_dict])
#         cur_part_network = part_networks[i_part]
#         src_pc = cur_part_rt_dict["src_pc"]
#         tar_pc = cur_part_rt_dict["tar_pc"]
#         key_pts = cur_part_rt_dict["key_pts"]
#         key_pts = cur_part_rt_dict["tar_pc"]
#         w_mesh = cur_part_rt_dict["w_mesh"]
#         w_pc = cur_part_rt_dict['w_pc']
#         src_verts = cur_part_rt_dict['src_ver']
#         tar_verts = cur_part_rt_dict['tar_ver']
#         src_faces = cur_part_rt_dict['src_face']
#         tar_faces = cur_part_rt_dict['tar_face']
#         src_cvx_to_pts = cur_part_rt_dict['src_cvx_to_pts']
#         dst_cvx_to_pts = cur_part_rt_dict['dst_cvx_to_pts']
#         # print(f"src_pc: {src_pc.size()}, tar_pc: {tar_pc.size()}")

#         # rt_dict = cur_part_network.forward7(src_pc, tar_pc, key_pts, key_pts, w_pc,  src_verts, src_cvx_to_pts , tar_verts, dst_cvx_to_pts)

#         mesh_rt_dict = cur_part_network.forward7(src_verts[0].unsqueeze(0), tar_verts[0].unsqueeze(0), key_pts, key_pts, w_pc,  src_verts, src_cvx_to_pts, tar_verts, dst_cvx_to_pts)
#         # def cvx to p ts # 
        
        
        
        
        
#         ### sorted_basis --> 
#         cur_part_sorted_basis = mesh_rt_dict["tot_sorted_basis"] ### sorted_basis: bsz x n_basis x (n_keypts x 3) 
#         # ### bsz x n_basis ### #
#         cur_part_avg_coefs = mesh_rt_dict["tot_avg_coefs"]
#         ### sorted_basis, avg_coefs ### cage-pts: 
#         cur_part_cage = mesh_rt_dict["cage"] ### part_cage: bsz x 
#         ### part, cage of this part --> cage-of-merged-cages; cage-of-def-cages; ###
#         ### cage of the deformed cages ###
#         tot_cage_of_merged_cages = mesh_rt_dict["tot_cage_of_merged_cages"] ## cage of merged cages ### list of merged cages --> [] --> for each item: bsz x nn_merged_cage_pts x 3 --> 
#         # tot_cage_of_merged_cages 
#         ### def part pts by cages, average defed pts, get cage of defed pts, def all pts from the merged cage and defed cage ###
        
#         ### part_cages: a list of ori_cvx_cages --> ori cages of convexes; 
#         ### part_pc: bsz x nn_src_pc x 3 --> source pc
#         ### part_basis: a list of convex basis in shape bsz x nn_basis x (nn_pts x 3)
#         ### coefs: an array of shape bsz x nn_basis
#         ### part_cvx_pts_idxes: a list of convex pts idxes -> list in the part level -> list in the convex level
#         # def get_def_pcs_by_cages(part_cages, part_pc, part_basis, coefs, part_cvx_pts_idxes)
#         tot_bsz_cages_ori = mesh_rt_dict["tot_bsz_cages_ori"] ### ori cages; convex cages list in the obj level; 
#         # part_pc = src_pc
#         part_pc = src_verts[0].unsqueeze(0)
#         tot_sorted_basis = mesh_rt_dict["tot_sorted_basis"] ### tot_sorted_basis --> list fo sorted basis 
#         tot_bsz_coefs = mesh_rt_dict["tot_avg_coefs"] ###  avg coefs ### 
#         tot_cvx_pts_idxes = mesh_rt_dict["tot_cvx_pts_idxes"] ### for each bsz?
        
#         tot_bsz_coefs = [cur_coef.requires_grad_(True) for cur_coef in tot_bsz_coefs]
#         tot_part_cvx_nn_info.append(len(tot_bsz_coefs))
        
#         tot_part_coefs += tot_bsz_coefs
        
#         tot_part_cages_ori.append(tot_bsz_cages_ori)
#         tot_part_src_pcs.append(part_pc)
#         tot_part_sorted_basis.append(tot_sorted_basis)
#         tot_part_cvx_pts_idxes.append(tot_cvx_pts_idxes)
        
#         def_verts = mesh_rt_dict["merged_deformed"]
#         def_mesh_sv_fn = os.path.join(samples_dir, f"train_ep_{epoch}_iter_{i}_bsz_{0}_def_mesh.obj")
#         save_obj_file(def_verts.detach().cpu().numpy()[0], src_faces[0].detach().cpu().numpy().tolist(), def_mesh_sv_fn, add_one=True)
#         # print("Saveing to ")
        
#         ### def_part_pc ###
#         def_part_pc, cur_obj_def_pc, cage_of_part_pcs, cage_of_def_part_pcs = get_def_pcs_by_cages(tot_bsz_cages_ori, part_pc, tot_sorted_basis, tot_bsz_coefs, tot_cvx_pts_idxes, part_data_dirs[i_part]) 
        
#         # def_verts = mesh_rt_dict["merged_deformed"]
#         def_mesh_sv_fn = os.path.join(samples_dir, f"train_ep_{epoch}_iter_{i}_bsz_{0}_def_mesh_def1.obj")
#         save_obj_file(def_part_pc.detach().cpu().numpy()[0], src_faces[0].detach().cpu().numpy().tolist(), def_mesh_sv_fn, add_one=True)
#         # print("Saveing to ")
        
        
#         # def_verts = mesh_rt_dict["merged_deformed"]
#         def_mesh_sv_fn = os.path.join(samples_dir, f"train_ep_{epoch}_iter_{i}_bsz_{0}_def_mesh_def1_def_pc.obj")
#         save_obj_file(cur_obj_def_pc.detach().cpu().numpy()[0], src_faces[0].detach().cpu().numpy().tolist(), def_mesh_sv_fn, add_one=True)
#         # print("Saveing to ")
        
#          # def_verts = mesh_rt_dict["merged_deformed"]
#         # def_mesh_sv_fn = os.path.join(samples_dir, f"train_ep_{epoch}_iter_{i}_bsz_{0}_def_mesh_def1_cage_of_merged_pc.obj")
#         # save_obj_file(cage_of_part_pcs.detach().cpu().numpy()[0], src_faces[0].detach().cpu().numpy().tolist(), def_mesh_sv_fn, add_one=True)
#         # # print("Saveing to ")
        
#         cage_face_list = net.glb_cages.cage_network.template_faces.detach().cpu().numpy()[0].tolist()
#         cur_bsz_cage_of_merged_cages = cage_of_part_pcs[0].detach().cpu().numpy() 
#         cur_bsz_merged_cages_sv_fn = os.path.join(samples_dir, f"train_ep_{epoch}_iter_{i}_bsz_{0}_cage_of_merged_cages.obj")
#         save_obj_file(cur_bsz_cage_of_merged_cages, cage_face_list, cur_bsz_merged_cages_sv_fn, add_one=True)
        
        
#         # print(f"Saving to {cur_bsz_merged_cages_sv_fn} with net.glb_cages.cage_network.template_faces: {net.glb_cages.cage_network.template_faces.size()}, verts: {cage_of_def_part_pcs.size()}")
#         cage_face_list = net.glb_cages.cage_network.template_faces.detach().cpu().numpy()[0].tolist()
#         cur_bsz_cage_of_merged_cages = cage_of_def_part_pcs[0].detach().cpu().numpy() 
#         cur_bsz_merged_cages_sv_fn = os.path.join(samples_dir, f"train_ep_{epoch}_iter_{i}_bsz_{0}_cage_of_merged_cages_2.obj")
#         save_obj_file(cur_bsz_cage_of_merged_cages, cage_face_list, cur_bsz_merged_cages_sv_fn, add_one=True)
                    
                    
#         mesh_tot_tot_sampled_merged_rnd_def_pcs = mesh_rt_dict["tot_tot_sampled_merged_rnd_def_pcs"]
#         for i_s in range(len(mesh_tot_tot_sampled_merged_rnd_def_pcs)):
#             cur_mesh_sampled_pc = mesh_tot_tot_sampled_merged_rnd_def_pcs[i_s]# [obj_idx]
#             tot_rnd_def_pcs[i_s].append(cur_mesh_sampled_pc)
        
        
#         # tot_def_pcs.append(defs_part_pc)
#         tot_def_pcs.append(def_verts)
#         # tot_def_pcs.append(src_verts[0].unsqueeze(0))
#         cur_src_face = src_faces[0] ### current source face 
#         # # print(f"cur_src_face: {cur_src_face.size()}")
#         tot_mesh_faces.append(cur_src_face) ## cur_src_face: n_faces x 3 
        
#         # tot_def_pcs.append(tar_verts[0].unsqueeze(0))
#         # cur_src_face = tar_faces[0] ### current source face 
#         # # print(f"cur_src_face: {cur_src_face.size()}")
#         # tot_mesh_faces.append(cur_src_face) ## cur_src_face: n_faces x 3 
        
    
    
    
#     # Loop and steps
#     assembled_pcs = []
#     tot_rnd_assembled_verts = [[] for i_s in range(opt.rnd_sample_nn)]
#     assembled_verts = []
#     assembled_faces = []
#     assembled_detection_pts = []
#     n_pts_detection = 512 # 
#     sim_weight = 1e-4
#     assembled_det_vs = []

#     for i_p in range(opt.n_parts): ### total number of parts ###
#         # cur_part_registered_pc: nn_pts x 3 ---> total sampled pcs ###
#         # cur_part_def_mesh_pc = tot_def_pcs[i_p]
#         # cur_part_sampled_pc_fps_idx = data_utils.farthest_point_sampling(cur_part_def_mesh_pc, n_sampling=n_pts_detection)
#         # cur_part_registered_pc = cur_part_def_mesh_pc[:, cur_part_sampled_pc_fps_idx]
#         # cur_part_sampled_mesh = tot_recon_meshes[i_p]
            
#         # cur_part_assembled_verts = [] ### the lisst of assembled part vertices of each shape
#         # cur_part_assembled_faces = []
#         # cur_part_assembled_detection_verts = []
#         # # for i_bsz in range(len(cur_part_sampled_mesh)):
#         #   cur_part_cur_bsz_mesh = cur_part_sampled_mesh[i_bsz]
#         # cur_part_v = cur_part_sampled_mesh["v"] ### n_mesh_verts x 3
#         # cur_part_f = cur_part_sampled_mesh["f"] ### n_mesh_verts x 3
#         cur_part_v = tot_def_pcs[i_p]
#         cur_part_f = tot_mesh_faces[i_p]
            
#         cur_part_cur_bsz_scale = tot_scale_infos[obj_idx, i_p]
#         cur_part_cur_bsz_pos_offset = tot_position_infos[obj_idx, i_p]
    
#         # to target scale #
#         # cur_part_v = normalie_pc_bbox_batched(cur_part_v.unsqueeze(0)).squeeze(0)
#         cur_part_v = normalie_pc_bbox_batched(cur_part_v)
#         ###### target scale ######
#         # cur_part_v = data_utils.scale_vertices_to_target_scale(cur_part_v, cur_part_cur_bsz_scale.unsqueeze(0)).squeeze(0)
#         cur_part_v = data_utils.scale_vertices_to_target_scale(cur_part_v, cur_part_cur_bsz_scale.unsqueeze(0)).squeeze(0)
#         ###### target scale ######

#         cur_part_v = cur_part_v + cur_part_cur_bsz_pos_offset.unsqueeze(0) ### offset info for assembling 
#         # cur_part_assembled_verts.append(cur_part_v)
#         cur_part_v = cur_part_v.unsqueeze(0)
        
#         # cur_part_assembled_faces.append(cur_part_f)
        
#         # print(f"cur_part_cur_bsz_v: {cur_part_v.size()}") ## 
#         if cur_part_v.size(1) > n_pts_detection:
#             # print(f"cur_part_cur_bsz_v: {cur_part_v.size()}")
#             sampled_cur_part_verts_fps_idx = data_utils.farthest_point_sampling(cur_part_v, n_sampling=n_pts_detection)
#             cur_part_det_v = cur_part_v[0, sampled_cur_part_verts_fps_idx]
#         else:
#             cur_part_det_v = cur_part_v[0]
            
#         # cur_part_v = 
#         # print(f"maxx_cur_part_v: {torch.max(cur_part_v, dim=1)}, minn_cur_part_v: {torch.min(cur_part_v, dim=1)}, maxx_cur_part_det_v: {torch.max(cur_part_det_v, dim=0)}, minn_cur_part_det_v: {torch.min(cur_part_det_v, dim=0)}")
#         cur_part_v = cur_part_v[0]
#         assembled_det_vs.append(cur_part_det_v)
#         assembled_detection_pts.append(cur_part_det_v)
#         assembled_verts.append(cur_part_v)
#         assembled_faces.append(cur_part_f)
        
#         # for i_s in range(opt.rnd_sample_nn):
#         #     cur_sample_verts = tot_rnd_def_pcs[i_s][i_p]
#         #     cur_sample_verts = normalie_pc_bbox_batched(cur_sample_verts)
#         #     cur_sample_verts = data_utils.scale_vertices_to_target_scale(cur_sample_verts, cur_part_cur_bsz_scale.unsqueeze(0)).squeeze(0)
#         #     cur_sample_verts = cur_sample_verts + cur_part_cur_bsz_pos_offset.unsqueeze(0)
#         #     cur_sample_verts = cur_sample_verts.unsqueeze(0)
#         #     cur_sample_verts = cur_sample_verts[0]
#         #     tot_rnd_assembled_verts[i_s].append(cur_sample_verts)
        
#     cur_obj_folder = os.path.join(samples_dir, f"obj_{obj_idxes[obj_idx].detach().item()}_" + str(datetime.datetime.now().strftime(
#     '%m-%d_%H-%M_')) )
#     os.makedirs(cur_obj_folder, exist_ok=True)
#     for i_p in range(len(assembled_verts)):
#         cur_part_verts = assembled_verts[i_p]
#         cur_part_faces = assembled_faces[i_p]
#         # exp_f
#         cat_info = opt.exp_flag[:3]
#         # cur_step_cur_part_mesh_sv_fn = os.path.join(cur_obj_folder, f"eps_{epoch}_part_{i_p}_{cat_info}.obj")
#         cur_step_cur_part_mesh_sv_fn = os.path.join(cur_obj_folder, f"eps_{tot_inst}_part_{i_p}_{cat_info}.obj")
#         save_obj_file(vertices=cur_part_verts.detach().cpu().numpy(), face_list=cur_part_faces.detach().cpu().numpy().tolist(), obj_fn=cur_step_cur_part_mesh_sv_fn, add_one=True)

#         # for i_s in range(opt.rnd_sample_nn):
#         #     cur_sample_verts = tot_rnd_assembled_verts[i_s][i_p]
#         #     cur_part_faces = assembled_faces[i_p]
#         #     cur_step_cur_part_mesh_sv_fn = os.path.join(cur_obj_folder, f"eps_{epoch}_part_{i_p}_i_s_{i_s}.obj")
#         #     save_obj_file(vertices=cur_sample_verts.detach().cpu().numpy(), face_list=cur_part_faces.detach().cpu().numpy().tolist(), obj_fn=cur_step_cur_part_mesh_sv_fn, add_one=True)
        
#     verts_list_np = [cur_part_verts.detach().cpu().numpy() for cur_part_verts in assembled_verts]
#     faces_list_np = [cur_part_faces.detach().cpu().numpy().tolist() for cur_part_faces in assembled_faces]
#     cur_obj_verts, cur_obj_faces = utils.merge_meshes(verts_list_np, faces_list_np)
#     # cur_step_cur_obj_mesh_sv_fn = os.path.join(cur_obj_folder, f"eps_{epoch}_{cat_info}.obj")
#     cur_step_cur_obj_mesh_sv_fn = os.path.join(cur_obj_folder, f"eps_{tot_inst}_{cat_info}.obj")
#     save_obj_file(vertices=cur_obj_verts, face_list=cur_obj_faces, obj_fn=cur_step_cur_obj_mesh_sv_fn, add_one=True)
    
    
    
    
#     joint_infos_np = data_utils.joint_infos_to_numpy(tot_joint_infos[obj_idx])
#     # joint_infos_sv_fn = os.path.join(cur_obj_folder, f"joint_infos_eps_{epoch}_{cat_info}.npy")
#     joint_infos_sv_fn = os.path.join(cur_obj_folder, f"joint_infos_eps_{tot_inst}_{cat_info}.npy")
    
#     tot_inst += 1
#     np.save(joint_infos_sv_fn, joint_infos_np)
#     print(f"Current obj saved -> {cur_obj_folder}")
    
#     # for i_s in range(opt.rnd_sample_nn):
#     #     verts_list_np = [cur_part_verts.detach().cpu().numpy() for cur_part_verts in tot_rnd_assembled_verts[i_s]]
#     #     faces_list_np = [cur_part_faces.detach().cpu().numpy().tolist() for cur_part_faces in assembled_faces]
#     #     cur_obj_verts, cur_obj_faces = utils.merge_meshes(verts_list_np, faces_list_np)
#     #     cur_step_cur_obj_mesh_sv_fn = os.path.join(cur_obj_folder, f"eps_{epoch}_i_s_{i_s}.obj")
#     #     save_obj_file(vertices=cur_obj_verts, face_list=cur_obj_faces, obj_fn=cur_step_cur_obj_mesh_sv_fn, add_one=True)
    
    
    
#     ''' Forward articulation simulation ''' 
    
#     last_moving_part = None
#     selected_state_idx = None
    
#     # last_moving_part = [0 for _ in range(n_bsz)]
    
#     n_steps = 2
#     for i_step in range(n_steps):
#         ### tot_part_cvx_to_sampled_pts ---> for backpropagation
#         tot_collision_loss = 0.
#         ### syn meshes --> assembled vertices and faces ### for each part 
#         # no batched 
#         syn_meshes = [[cur_part_v.squeeze(0).requires_grad_(True), cur_part_f.squeeze(0)] for (cur_part_v, cur_part_f) in zip(assembled_verts, assembled_faces)]
#         # cur_bsz_syn_meshes = [cur_bsz_assembled_v, cur_bsz_assembled_f]
#         joints = tot_joint_infos[obj_idx]
#         for part_joint in joints:
#             for k in ["dir", "center"]:
#                 # if isinstance(part_joint["axis"][k], torch.Tensor):
#                 if isinstance(part_joint["axis"][k], torch.Tensor):
#                     part_joint["axis"][k] = part_joint["axis"][k].cuda()
#                 else:
#                     part_joint["axis"][k] = torch.from_numpy(part_joint["axis"][k]).float().cuda()
            
#         collision_loss_func = collision_sim.collision_loss_joint_structure_with_rest_pose_rnd_detect
        
#         #### collision_loss_func --> for th collision loss ####
#         # n_sim_steps = 100
#         # n_sim_steps = args.train_n_sim_steps
#         n_sim_steps = 50
#         # n_sim_steps = 10
        
#         # cur_bsz_v_shps = [cur_v.size() for cur_v in cur_bsz_assembled_v]
#         # cur_bsz_det_pts_shps = [cur_v.size() for cur_v in cur_bsz_assembled_detection_pts]
#         # cur_bsz_f_shps = [cur_f.size() for cur_f in cur_bsz_assembled_f]
#         # print(f"v-shps: {cur_bsz_v_shps}, pts-shps: {cur_bsz_det_pts_shps}, f-shps: {cur_bsz_f_shps}")
#         print(f"Start collision detection...")
#         #### mesh_pts_sequenc; stateverts ### 
#         collision_loss, keypts_sequence, mesh_pts_sequence, cur_state_verts, last_moving_part, selected_state_idx = collision_loss_func(syn_meshes, assembled_detection_pts, joints, n_sim_steps, selected_moving_part_idx=last_moving_part, selected_state_idx=selected_state_idx, fix_other_parts=True)
        
#         tot_collision_losses.append(collision_loss.mean().item())
#         ''' TODO: save pts / verts seuqneces for collision detection '''
        
#         # print(f"mesh pts sequences: {len(mesh_pts_sequence)}")
#         # for i_sim_step in range(len(mesh_pts_sequence)):
#         #   for i_p in range(n_part):
#         #     if i_p == last_moving_part[i_bsz]:
#         #       cur_bsz_cur_step_cur_sim_part_obj = (mesh_pts_sequence[i_sim_step], cur_bsz_assembled_f[i_p])
#         #       cur_bsz_cur_step_cur_sim_part_obj_sv_fn = os.path.join(sv_samples_folder_nm, f"loop_{i_step}_sim_step_{i_sim_step}_bsz_{i_bsz}_part_{i_p}.obj")
#         #       save_obj_file(cur_bsz_cur_step_cur_sim_part_obj[0].detach().cpu().numpy(), cur_bsz_cur_step_cur_sim_part_obj[1].detach().cpu().numpy().tolist(), cur_bsz_cur_step_cur_sim_part_obj_sv_fn, True)
#         #     else:
#         #       cur_bsz_cur_step_cur_sim_part_obj = (cur_bsz_assembled_v[i_p], cur_bsz_assembled_f[i_p])
#         #       cur_bsz_cur_step_cur_sim_part_obj_sv_fn = os.path.join(sv_samples_folder_nm, f"loop_{i_step}_sim_step_{i_sim_step}_bsz_{i_bsz}_part_{i_p}.obj")
#         #       save_obj_file(cur_bsz_cur_step_cur_sim_part_obj[0].detach().cpu().numpy(), cur_bsz_cur_step_cur_sim_part_obj[1].detach().cpu().numpy().tolist(), cur_bsz_cur_step_cur_sim_part_obj_sv_fn, True)
#         # print(f"Svaing to {cur_bsz_cur_step_cur_sim_part_obj_sv_fn}.")

#         n_bsz = 1
#         tot_collision_loss += collision_loss
#         collision_loss = tot_collision_loss / float(n_bsz)
    
#         ### guide the z's sampling process? ### 
        
        

#         ## flat tot_part_cvx_sampled_pts? ---> how to flat them? 
#         print(f"Start flatting")
#         # tot_obj_zs_flatten, tot_obj_zs_nn = flat_zs(tot_part_zs)
        
#         # tot_obj_xTs_flatten, tot_obj_xTs_nn = flat_zs(tot_part_x_Ts)
#         # # flat_list, part_bsz_cvx_nns = flat_tot_part_cvx_sampled_pts(tot_part_cvx_sampled_pts)
#         # print(f"After flatting with flatten list: {len(tot_obj_zs_flatten)}")
    
    
#         # tot_part_cvx_sampled_pts
#         try:
#             grad_coefs = autograd.grad(
#                 outputs=collision_loss,
#                 # inputs=tot_part_cvx_sampled_pts,
#                 # inputs=tot_obj_zs_flatten,
#                 inputs=tot_part_coefs, ### a list of coefs ## a list of arr in shape bsz x n_basis
#                 grad_outputs=torch.ones_like(collision_loss),
#                 create_graph=True,
#                 retain_graph=True,
#                 allow_unused=True
#             ) # [0]
#         except:
#             grad_coefs = [None for _ in range(len(tot_part_coefs))]
#         cons = collision_loss
#         # for i, cur_pts in enumerate(tot_obj_zs_flatten):
#         for i_coef, cur_coefs in enumerate(tot_part_coefs):
#             # cur_basis: bsz x n_basis x n_dim
#             # cons : B * 1; grad: B * num_particles * dimension
#             grad_cur_coefs = grad_coefs[i_coef]
            
#             if grad_cur_coefs is not None:
#                 # print(grad_cur_pts)
#                 print(f"here! grad size: {grad_cur_coefs.size()}")
#                 delta_cur_coefs = -1.0 * grad_cur_coefs * sim_weight # * 10
#                 # print(f"delta_cur_basis: {delta_cur_basis}")
#                 nex_coefs = cur_coefs + delta_cur_coefs
#             else:
#                 nex_coefs = cur_coefs
#             #### nexx_basis ####
#             # nexx_basis = torch.zeros(nex_basis.size(), dtype=torch.float32).cuda()
#             # nexx_basis[:, :, :] = nex_basis.data[:, :, :]
#             # # nexx_basis.
#             # # nex_bases.append(nex_basis.clone().requires_grad_(True))
#             # tot_obj_zs_flatten[i] = nex_pts.clone().requires_grad_(True)
#             # tot_obj_xTs_flatten[i_pts] = nex_pts.clone().requires_grad_(True)
#             tot_part_coefs[i_coef] = nex_coefs.clone().requires_grad_(True) ### part_coefs 
#         print(f"step: {i_step}, collision_loss: {collision_loss.mean().item()}") ### collision loss

#         real_tot_part_coefs = []
#         cvx_st_idx = 0
#         for i_part in range(len(tot_part_cvx_nn_info)):
#             real_tot_part_coefs.append(tot_part_coefs[cvx_st_idx: cvx_st_idx + tot_part_cvx_nn_info[i_part]])
#             # real_tot_part_coefs 
#             cvx_st_idx += tot_part_cvx_nn_info[i_part]

#         tot_def_pcs = []
        
#         for i_part in range(opt.n_parts):

#             # cur_part_cages_ori = tot_part_coefs[i_part]
#             cur_part_cages_ori = tot_part_cages_ori[i_part]
#             part_pc = tot_part_src_pcs[i_part]
#             cur_part_sorted_basis = tot_part_sorted_basis[i_part]
#             cur_part_bsz_coefs = real_tot_part_coefs[i_part]
#             cur_part_cvx_pts_idxes = tot_part_cvx_pts_idxes[i_part]
            
#         #     tot_bsz_cages_ori = mesh_rt_dict["tot_bsz_cages_ori"] ### ori cages; convex cages list in the obj level; 
#         # # part_pc = src_pc
#         # part_pc = src_verts[0].unsqueeze(0)
#         # tot_sorted_basis = mesh_rt_dict["tot_sorted_basis"] ### tot_sorted_basis --> list fo sorted basis 
#         # tot_bsz_coefs = mesh_rt_dict["tot_avg_coefs"] ###  avg coefs ### 
#         # tot_cvx_pts_idxes = mesh_rt_dict["tot_cvx_pts_idxes"] ### for each bsz?
        
#         # tot_bsz_coefs = [cur_coef.requires_grad_(True) for cur_coef in tot_bsz_coefs]
#         # tot_part_cvx_nn_info.append(len(tot_bsz_coefs))
        
#         # tot_part_coefs += tot_bsz_coefs
        
#         # tot_part_cages_ori.append(tot_bsz_cages_ori)
#         # tot_part_src_pcs.append(part_pc)
#         # tot_part_sorted_basis.append(tot_sorted_basis)
#         # tot_part_cvx_pts_idxes.append(tot_cvx_pts_idxes)
        
            
            
#             ### def_part_pc ### --> def_pcs by cages ####
#             def_part_pc, cur_obj_def_pc, cage_of_part_pcs, cage_of_def_part_pcs = get_def_pcs_by_cages(cur_part_cages_ori, part_pc, cur_part_sorted_basis, cur_part_bsz_coefs, cur_part_cvx_pts_idxes, part_data_dirs[i_part]) 
            
#             tot_def_pcs.append(def_part_pc)
#             # cur_src_face = src_faces[0] ### current source face 
#             # tot_mesh_faces.append(cur_src_face) ## cur_src_face: n_faces x 3 
        
        
        
#         assembled_det_vs, assembled_detection_pts, assembled_verts, assembled_faces = get_assembled_pcs(tot_def_pcs, tot_mesh_faces, tot_scale_infos, tot_position_infos)
#         # assemble parts
#         # tot_part_samples, tot_part_samples_mesh, tot_part_zs, tot_part_x_Ts = get_assemble_pts_zs(zs_unflatten, xTs_unflatten, part_models,  part_cvx_encoders, part_sap_models, part_sap_model_trainers, tot_part_train_datasets, tot_part_cvx_to_pc, tot_part_cvx_to_cvx_pts, tot_part_cvx_to_bbox_center, last_step=i_step == n_steps - 1)
        
    
#         # assembled_pcs, assembled_verts, assembled_detection_pts = assemble_pts(tot_part_samples, tot_part_samples_mesh, tot_scale_infos, tot_position_infos)
#         # assembled_faces = [cur_part_mesh['f'] for cur_part_mesh in tot_part_samples_mesh]
#         # for i_p in range(len(assembled_verts)):
#         #   if args.fixed_part_idx is not None and i_p == args.fixed_part_idx:
#         #     assembled_verts[i_p] = data["tar_ver"][i_p]
    
#         # print(f"Saveing objs...")
#         for i_p in range(len(assembled_verts)):
#             cur_part_verts = assembled_verts[i_p]
#             cur_part_faces = assembled_faces[i_p]
#             cur_step_cur_part_mesh_sv_fn = os.path.join(samples_dir, f"eps_{epoch}_bsz_{i}_step_{i_step}_part_{i_p}.obj")
#             save_obj_file(vertices=cur_part_verts.detach().cpu().numpy(), face_list=cur_part_faces.detach().cpu().numpy().tolist(), obj_fn=cur_step_cur_part_mesh_sv_fn, add_one=True)
        
#         verts_list_np = [cur_part_verts.detach().cpu().numpy() for cur_part_verts in assembled_verts]
#         faces_list_np = [cur_part_faces.detach().cpu().numpy().tolist() for cur_part_faces in assembled_faces]
#         cur_obj_verts, cur_obj_faces = utils.merge_meshes(verts_list_np, faces_list_np)
#         cur_step_cur_obj_mesh_sv_fn = os.path.join(samples_dir, f"eps_{epoch}_bsz_{i}_step_{i_step}.obj")
#         save_obj_file(vertices=cur_obj_verts, face_list=cur_obj_faces, obj_fn=cur_step_cur_obj_mesh_sv_fn, add_one=True)
#         print(f"Current step mesh saved to {cur_step_cur_obj_mesh_sv_fn}...")
    
    
#         torch.cuda.empty_cache()
        