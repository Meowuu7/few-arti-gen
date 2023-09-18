import src.datasets.datasets_def_v5 as datasets_v3
import torch
import numpy as np
import src.networks.network_target_driven_prob as network
import src.networks.network_with_cages as network_paired_data
from src.common_utils.utils import normalie_pc_bbox_batched, save_obj_file, weights_init, weights_init1
import argparse
import shutil
import logging
from tqdm import tqdm
import os
from pathlib import Path
import datetime
# from src.common_utils.losses import chamfer_distance_raw as chamfer_distance
from src.common_utils.evaluation_metrics import *
import common_utils.data_utils_torch as data_utils
from src.networks.network import model as glb_deform_model



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
                        default="./data/toy_eye_tot", help='data dir for pair-wise deformation data...')
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
                        help='number of basis vectors') #
    parser.add_argument('--use_def_pc', default=False, action='store_true',
                        help='number of basis vectors') #
    parser.add_argument('--use_trans_range', default=False, action='store_true',
                        help='number of basis vectors') #
    parser.add_argument('--use_delta_prob', default=False, action='store_true',
                        help='number of basis vectors') # 
    parser.add_argument('--use_cond_vae', default=False, action='store_true',
                        help='number of basis vectors') #
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
                        help='number of basis vectors') # cond_tar_pc
    parser.add_argument('--cond_tar_pc', default=False, action='store_true',
                        help='number of basis vectors') # recon_cond_tar
    parser.add_argument('--recon_cond_tar', default=False, action='store_true',
                        help='number of basis vectors')  ## recon_cond
    parser.add_argument('--recon_cond', type=str,
                        default='cvx', help='enviroment name of visdom') 
    parser.add_argument('--cvx_dim', type=int, default=-1,
                        help='number of basis vectors') 
    parser.add_argument('--only_tar', default=False, action='store_true',
                        help='number of basis vectors')  
    parser.add_argument('--test_recon', default=False, action='store_true',
                        help='number of basis vectors') 
    parser.add_argument('--recon_normalized_pts', default=False, action='store_true', ### 
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
    # with_image_loss
    parser.add_argument('--with_image_loss', default=False, action='store_true',  
                    help='number of basis vectors')
    parser.add_argument('--rnd_sample_nn', type=int, default=0, 
                        help='number of basis vectors') 
    parser.add_argument("--cvx_folder_fn", type=str, default="")
    parser.add_argument("--glb_template", type=str, default="") 
    parser.add_argument("--gen_pts_path", type=str, default="") 
    parser.add_argument("--ref_pts_path", type=str, default="") 
    parser.add_argument('--use_train_pcs', default=False, action='store_true', 
                    help='number of basis vectors')
    parser.add_argument('--n_shots', type=int, default=5,
                        help='number of basis vectors')
    parser.add_argument('--n_samples_per_ref', type=int, default=40,
                        help='number of basis vectors') 
    parser.add_argument('--cvx_list_filter', default=False, action='store_true',
                    help='number of basis vectors') 
    parser.add_argument('--src_data_dir', type=str,
                        default="", help='data dir for pair-wise deformation data...')
    parser.add_argument('--dst_data_dir', type=str,
                        default="", help='data dir for pair-wise deformation data...')
    parser.add_argument('--src_n_keypoints', type=int, default=256,
                        help='number of basis vectors')
    parser.add_argument('--dst_n_keypoints', type=int, default=256,
                        help='number of basis vectors')
    parser.add_argument('--src_cvx_to_pts_sufix', type=str, default="_cvx_to_verts_cdim_128.npy",
                        help='number of basis vectors') 
    parser.add_argument('--dst_cvx_to_pts_sufix', type=str, default="_cvx_to_verts_cdim_128.npy",
                        help='number of basis vectors') 
    return parser.parse_args() 


opt = parse_args()



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


##### add exp_flag to the samples dir's name #####
use_prob_indicator = 1 if opt.use_prob else 0
use_prob_src_indicator = 1 if opt.use_prob_src else 0



model_flag = f"{opt.exp_flag}_{opt.learning_rate}_kl_{opt.kl_weight}_use_prob_{use_prob_indicator}_src_prob_{use_prob_src_indicator}_scaling_{opt.random_scaling}_tar_basis_{opt.tar_basis}_num_basis_{opt.num_basis}_n_keypoints_{opt.n_keypoints}_neighbouring_k_{opt.neighbouring_k}"


samples_dir = model_flag
samples_root_dir = "./samples"
samples_root_dir = os.path.join(samples_root_dir, "few_arti_gen")
os.makedirs(samples_root_dir, exist_ok=True)
samples_dir = os.path.join(samples_root_dir, samples_dir)
os.makedirs(samples_dir, exist_ok=True)

data_dir = opt.data_dir


dataset_train = datasets_v3.ChairDataset("train", data_dir, split="train", opt=opt)

#### collect fn #####
dataloader_train = torch.utils.data.DataLoader(dataset_train, collate_fn=datasets_v3.my_collate, drop_last=True, shuffle=True, batch_size=opt.batch_size, num_workers=opt.num_workers, worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))


dataset_test = datasets_v3.ChairDataset("test", data_dir, split="test", opt=opt)

#### collect fn #####
dataloader_test = torch.utils.data.DataLoader(dataset_test, collate_fn=datasets_v3.my_collate, drop_last=True, shuffle=False, batch_size=opt.batch_size, num_workers=opt.num_workers, worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))


nn_com_pts = 4096
mesh_sample_npoints = 1

tot_ref_pts = []

if len(opt.ref_pts_path) > 0:
    tot_ref_pts = np.load(opt.ref_pts_path, allow_pickle=True)
    tot_ref_pts = torch.from_numpy(tot_ref_pts).float()
else:
    ''' Ref pcs '''
    for tst_idx in range(len(dataset_test)):
        cur_tst_info_dict = dataset_test.get_objs_info_via_idxess(tst_idx)
        verts, faces = cur_tst_info_dict["vertices"], cur_tst_info_dict["faces"]
        faces_list = faces.tolist()
        sampled_pcts = data_utils.sample_pts_from_mesh(verts, faces_list, npoints=mesh_sample_npoints)
        print(f"tst_idx: {tst_idx}, ori_nn_sampled_pts: {sampled_pcts.shape}")
        sampled_pcts = torch.from_numpy(sampled_pcts).float().cuda() ## nn_pts x 3
        sampled_pcts_fps_idx = data_utils.farthest_point_sampling(sampled_pcts.unsqueeze(0), n_sampling=nn_com_pts) ### nn_pts x 3
        sampled_pcts = sampled_pcts[sampled_pcts_fps_idx].cpu()
        tot_ref_pts.append(sampled_pcts.unsqueeze(0))

    tot_ref_pts = torch.cat(tot_ref_pts, dim=0)
print(f"Reference points: {tot_ref_pts.size()}")

ref_pts_sv_fn = os.path.join(samples_dir, "tot_ref_pts.npy")
np.save(ref_pts_sv_fn, tot_ref_pts.detach().cpu().numpy())

print(f"Reference pts saved to {ref_pts_sv_fn}...")

## ImageRender
net = network_paired_data.model(opt.num_basis, opt=opt).cuda()

netD = network.Discriminator().cuda() 


image_render = network_paired_data.ImageRender()





if opt.with_glb: ### 
    glb_net = glb_deform_model(opt.num_basis, opt=opt).cuda()
else:
    glb_net = None




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



    
log_string('No existing model, starting training from scratch...')
start_epoch = 0
cd_curve = []
sym_curve = []
lap_curve = []
nor_curve = []
tot_curve = []
g_curve = []
d_curve = []
net = net.apply(weights_init)
netD = netD.apply(weights_init1)

### net_path: net_path --> net_path... ###
if len(opt.net_path) > 0:
    ###### net_state_dict xxx ######
    net_state_dict = torch.load(opt.net_path, map_location="cpu", )
    # safe_load_ckpt_common(net, net_state_dict, weight_flag=weight_flag)
    safe_load_ckpt_common(net, net_state_dict) ### laod net path ###
    


### net_path: net_path --> net_path... ###
if len(opt.net_glb_path) > 0:
    ###### net_state_dict xxx ######
    net_state_dict = torch.load(opt.net_glb_path, map_location="cpu", )
    # safe_load_ckpt_common(net, net_state_dict, weight_flag=weight_flag)
    safe_load_ckpt_common(glb_net, net_state_dict) ### laod net path ###



### global loss?


if opt.test_only:
    opt.epoch = start_epoch + 1


glb_net = None



sampled_pts = []


epoch = 0


if glb_net is not None:
    glb_net.train()
    net.train()
    netD.train()
else:
    net.train()
    netD.train()



buf = {"cd" : [], "cage_shift": [], "tar_cage_shift": [], "tot": [], "tar_extents": [], "tar_offset": [], "orth": [], "sp": [], "svd": [], "cat_offset": []
        } 

src_names = []
tar_names = []


if len(opt.gen_pts_path) > 0:
    sampled_pts = np.load(opt.gen_pts_path, allow_pickle=True)
    sampled_pts = torch.from_numpy(sampled_pts).float()
    print(f'Loaded from {opt.gen_pts_path} with shape: {sampled_pts.size()}')
elif opt.use_train_pcs:
    for tst_idx in range(len(dataset_train)):
        cur_tst_info_dict = dataset_train.get_objs_info_via_idxess(tst_idx)
        verts, faces = cur_tst_info_dict["vertices"], cur_tst_info_dict["faces"]
        faces_list = faces.tolist()
        sampled_pcts = data_utils.sample_pts_from_mesh(verts, faces_list, npoints=mesh_sample_npoints)
        print(f"train_idx: {tst_idx}, ori_nn_sampled_pts: {sampled_pcts.shape}")
        sampled_pcts = torch.from_numpy(sampled_pcts).float().cuda() ## nn_pts x 3
        sampled_pcts_fps_idx = data_utils.farthest_point_sampling(sampled_pcts.unsqueeze(0), n_sampling=nn_com_pts) ### nn_pts x 3
        sampled_pcts = sampled_pcts[sampled_pcts_fps_idx].cpu()
        sampled_pts += [sampled_pcts.unsqueeze(0) for _ in range(12)]
    sampled_pts = torch.cat(sampled_pts, dim=0)
    print(f"Sampled_pts: {sampled_pts.size()}")
else:
    ## should have a 
    for i_sample in range(opt.n_samples_per_ref):
      #### for the sample index ####
      print(f"i_sample:  {i_sample + 1} / {opt.n_samples_per_ref}")
      cur_iter_sampled_pts = []
      
      cur_sampled_pts_iter_fn = os.path.join(samples_dir, f"sampled_pts_iter_{i_sample}.npy")
      if os.path.exists(cur_sampled_pts_iter_fn):
          cur_iter_sampled_pts = np.load(cur_sampled_pts_iter_fn, allow_pickle=True)
          cur_iter_sampled_pts = torch.from_numpy(cur_iter_sampled_pts).float() # .cuda()
          print(f"Loaded from {cur_sampled_pts_iter_fn} for sampled pts with shape {cur_iter_sampled_pts.size()}")
          sampled_pts.append(cur_iter_sampled_pts)
          continue
      for i, data in tqdm(enumerate(dataloader_train), total=len(dataloader_train), smoothing=0.9):
          src_pc = data['src_pc']
          tar_pc = data['tar_pc']
          
          # src_pc = tar_pc
          key_pts = data['key_pts']
          
          dst_key_pts = data['dst_key_pts']
          # key_pts = dst_key_pts
          
          w_mesh = data["w_mesh"]
          w_pc = data['w_pc']
          src_verts = data['src_ver']
          tar_verts = data['tar_ver']
          src_faces = data['src_face']
          tar_faces = data['tar_face']
          src_cvx_to_pts = data['src_cvx_to_pts']
          dst_cvx_to_pts = data['dst_cvx_to_pts']
          real_pc = data["real_pc"]
          real_vertices = data["real_vertices"]
          real_w_mesh = data["real_w_mesh"]
          real_cvx_to_pts = data["real_cvx_to_pts"] ### real_cvx_to_pts and others ###
          
          ### forward
          mesh_rt_dict = net.forward7(src_verts[0].unsqueeze(0), tar_verts[0].unsqueeze(0), key_pts, dst_key_pts, w_pc,  src_verts,src_cvx_to_pts , tar_verts, dst_cvx_to_pts)
          rt_dict = mesh_rt_dict
          
          tot_bsz_assembled_def_cvx_pts = mesh_rt_dict["tot_bsz_assembled_def_cvx_pts"]
          
          cd_loss, mesh_cd_loss = rt_dict["cd_loss"], mesh_rt_dict["cd_loss"]
          loss = (cd_loss + mesh_cd_loss) / 2.
          extents_loss, mesh_extents_loss = rt_dict["extents_loss"], mesh_rt_dict["extents_loss"]
          offset_loss, mesh_offset_loss =  rt_dict["offset_loss"], mesh_rt_dict["offset_loss"]
          
          cat_offset_loss, mesh_cat_offset_loss = rt_dict["cat_offset_loss"], mesh_rt_dict["cat_offset_loss"] ### offset losses ###
          
          tot_tot_sampled_merged_rnd_def_pcs = rt_dict["tot_tot_sampled_merged_rnd_def_pcs"]
          mesh_tot_tot_sampled_merged_rnd_def_pcs = mesh_rt_dict["tot_tot_sampled_merged_rnd_def_pcs"]
          
          extents_loss = (extents_loss + mesh_extents_loss) / 2.
          offset_loss = (offset_loss + mesh_offset_loss) / 2.
          cat_offset_loss = (cat_offset_loss + mesh_cat_offset_loss) / 2. ### cat_offset_losses ###
          
          buf["cd"].append(loss.detach().cpu().numpy())
          buf["tar_extents"].append(extents_loss.detach().cpu().numpy())
          buf["tar_offset"].append(offset_loss.detach().cpu().numpy())
          buf["cat_offset"].append(cat_offset_loss.detach().cpu().numpy()) ### add cat offset loss 
          
          # tot_cage_basis
          tot_cage_basis = rt_dict["tot_cage_basis"]
          tot_cage_basis_np = [[tot_cage_basis[i_bsz][i_cvx].detach().cpu().numpy() for i_cvx in range(len(tot_cage_basis[i_bsz]))] for i_bsz in range(len(tot_cage_basis)) ]
          
          tot_cage = rt_dict["cage"]
          tot_cage = [[tot_cage[i_bsz][i_cvx].detach().cpu().numpy() for i_cvx in range(len(tot_cage[i_bsz]))] for i_bsz in range(len(tot_cage))]
          
          tot_mesh_cage = mesh_rt_dict["cage"]
          tot_mesh_cage = [[tot_mesh_cage[i_bsz][i_cvx].detach().cpu().numpy() for i_cvx in range(len(tot_mesh_cage[i_bsz]))] for i_bsz in range(len(tot_mesh_cage))]
          
          tot_mesh_cage_basis = rt_dict["tot_cage_basis"]
          tot_mesh_cage_basis_np = [[tot_mesh_cage_basis[i_bsz][i_cvx].detach().cpu().numpy() for i_cvx in range(len(tot_mesh_cage_basis[i_bsz]))] for i_bsz in range(len(tot_mesh_cage_basis)) ]
          
          
          
          tot_cage_coefs = rt_dict["tot_cage_coefs"]
          tot_cage_coefs_np = [[tot_cage_coefs[i_bsz][i_cvx].detach().cpu().numpy() for i_cvx in range(len(tot_cage_coefs[i_bsz]))] for i_bsz in range(len(tot_cage_coefs)) ]
          
          
          coefs_argsort = rt_dict["coefs_argsort"]
          coefs_argsort = coefs_argsort.detach().cpu().numpy()
          
          cage_face = rt_dict["cage_face"]
          cage_face = [[cage_face[i_bsz][i_cvx].detach().cpu().numpy() for i_cvx in range(len(cage_face[i_bsz]))] for i_bsz in range(len(cage_face))]
          
          tot_bsz_assembled_def_cvx_pts_np = [[cvx_arr.detach().cpu().numpy() for cvx_arr in cur_bsz_tot_def_cvx_arr] for cur_bsz_tot_def_cvx_arr in tot_bsz_assembled_def_cvx_pts]
          
          
          ### all for basis sv dict ###
          
          basis_sv_dict = {
              "tot_cage": tot_cage, "tot_mesh_cage": tot_mesh_cage, "tot_cage_basis": tot_cage_basis_np, "tot_mesh_cage_basis": tot_mesh_cage_basis_np, "cage_face": cage_face, 
              "coefs_argsort": coefs_argsort, "tot_cage_coefs": tot_cage_coefs_np, "assembled_def_cvx_pts": tot_bsz_assembled_def_cvx_pts_np ### assembled-def-pcs 
          }
          
        #   tot_bsz_assembled_def_cvx_pts_np = [[cvx_arr.detach().cpu().numpy() for cvx_arr in cur_bsz_tot_def_cvx_arr] for cur_bsz_tot_def_cvx_arr in tot_bsz_assembled_def_cvx_pts]
          

          # "tot_cage_basis": tot_cage_basis,
          # "tot_cage_coefs": tot_cage_coefs
          tot_basis, tot_coefs = rt_dict["tot_cage_basis"], rt_dict["tot_cage_coefs"]
          tot_basis_mesh, tot_coefs_mesh = mesh_rt_dict["tot_cage_basis"], mesh_rt_dict["tot_cage_coefs"]
          
          tot_ortho_loss, tot_svd_loss, tot_sp_loss = losses.basis_reg_losses(tot_basis, tot_coefs)
          mesh_tot_ortho_loss, mesh_tot_svd_loss, mesh_tot_sp_loss = losses.basis_reg_losses(tot_basis_mesh, tot_coefs_mesh)
          
          tot_ortho_loss = (tot_ortho_loss + mesh_tot_ortho_loss) / 2.
          tot_svd_loss = (tot_svd_loss + mesh_tot_svd_loss) / 2.
          tot_sp_loss = (tot_sp_loss + mesh_tot_sp_loss) / 2.
          
          
          # train_def_cages
          if opt.with_dis_loss:  
              disent_loss =  tot_ortho_loss * 0.0001 +  tot_sp_loss * 0.0001 
          else:
              disent_loss = torch.zeros((1, ), dtype=torch.float32).cuda().mean()
              
          # tot_loss = loss + extents_loss * opt.gravity_weight + offset_loss + disent_loss
          tot_loss = loss * opt.cd_weight + extents_loss * opt.gravity_weight  + offset_loss + disent_loss + cat_offset_loss
          buf["tot"].append(tot_loss.detach().cpu().numpy())
          buf["orth"].append(tot_ortho_loss.detach().cpu().numpy())
          buf["sp"].append(tot_sp_loss.detach().cpu().numpy())
          buf["svd"].append(tot_svd_loss.detach().cpu().numpy())
          

          ##### opt.display ####
          #### dispalye samples ###
          if epoch % opt.display == 0:
          #     "cage": tot_cages, 
          # "new_cage": tot_new_cages,
          # "cage_face": tot_cages_faces,
              cages = rt_dict["cage"]
              new_cages = rt_dict["new_cage"]
              cage_faces = rt_dict["cage_face"]
              rnd_new_cages = rt_dict["rnd_new_cage"]
              
              tot_rnd_def_pcs = mesh_rt_dict["tot_rnd_tot_def_pcs"]
              
              
              for i_bsz in range(len(cages)):
                  for i_cvx in range(len(cages[0])):
                      cur_cvx_cage = cages[i_bsz][i_cvx]
                      cur_cvx_new_cage = new_cages[i_bsz][i_cvx]
                      cur_cvx_cage_faces = cage_faces[i_bsz][i_cvx]
                      cur_cvx_rnd_new_cagee = rnd_new_cages[i_bsz][i_cvx]
                      
              
                      cages_sv_fn = os.path.join(samples_dir, f"train_ep_{epoch}_iter_{i}_bsz_{i_bsz}_cvx_{i_cvx}_cages.obj")
                      save_obj_file(cur_cvx_cage.detach().cpu().numpy()[0], cur_cvx_cage_faces.detach().cpu().numpy()[0].tolist(), cages_sv_fn, add_one=True)
          
                      def_cages_sv_fn = os.path.join(samples_dir, f"train_ep_{epoch}_iter_{i}_bsz_{i_bsz}_cvx_{i_cvx}_def_cages.obj")
                      save_obj_file(cur_cvx_new_cage.detach().cpu().numpy()[0], cur_cvx_cage_faces.detach().cpu().numpy()[0].tolist(), def_cages_sv_fn, add_one=True)
                      
                      rnd_def_cages_sv_fn = os.path.join(samples_dir, f"train_ep_{epoch}_iter_{i}_bsz_{i_bsz}_cvx_{i_cvx}_rnd_def_cages.obj")
                      save_obj_file(cur_cvx_rnd_new_cagee.detach().cpu().numpy()[0], cur_cvx_cage_faces.detach().cpu().numpy()[0].tolist(), rnd_def_cages_sv_fn, add_one=True)
          
                  cur_bsz_src_verts = src_verts[i_bsz].detach().cpu().numpy()
                  cur_bsz_src_face = src_faces[i_bsz].detach().cpu().numpy().tolist()
                  cur_bsz_src_mesh_sv_fn = os.path.join(samples_dir, f"train_ep_{epoch}_iter_{i}_bsz_{i_bsz}_i_s_{i_sample}_src_mesh.obj")
                  save_obj_file(cur_bsz_src_verts, cur_bsz_src_face, cur_bsz_src_mesh_sv_fn, add_one=True)
                  
                  cur_bsz_tar_verts = tar_verts[i_bsz].detach().cpu().numpy()
                  cur_bsz_tar_face = tar_faces[i_bsz].detach().cpu().numpy().tolist()
                  cur_bsz_tar_mesh_sv_fn = os.path.join(samples_dir, f"train_ep_{epoch}_iter_{i}_bsz_{i_bsz}_i_s_{i_sample}_tar_mesh.obj")
                  save_obj_file(cur_bsz_tar_verts, cur_bsz_tar_face, cur_bsz_tar_mesh_sv_fn, add_one=True)
                  
                  def_verts = mesh_rt_dict["deformed"]
                  def_mesh_sv_fn = os.path.join(samples_dir, f"train_ep_{epoch}_iter_{i}_bsz_{i_bsz}_i_s_{i_sample}_def_mesh.obj")
                  save_obj_file(def_verts.detach().cpu().numpy()[i_bsz], src_faces[0].detach().cpu().numpy().tolist(), def_mesh_sv_fn, add_one=True)
                  
                #   rnd_def_verts = mesh_rt_dict["rnd_deformed"]
                #   rnd_def_mesh_sv_fn = os.path.join(samples_dir, f"train_ep_{epoch}_iter_{i}_bsz_{i_bsz}_rnd_def_mesh.obj")
                #   save_obj_file(rnd_def_verts.detach().cpu().numpy()[i_bsz], src_faces[0].detach().cpu().numpy().tolist(), rnd_def_mesh_sv_fn, add_one=True)
                  
                  merged_def_verts = mesh_rt_dict["merged_deformed"]
                  def_mesh_sv_fn = os.path.join(samples_dir, f"train_ep_{epoch}_iter_{i}_bsz_{i_bsz}_i_s_{i_sample}_merged_def_mesh.obj")
                  save_obj_file(merged_def_verts.detach().cpu().numpy()[i_bsz], src_faces[0].detach().cpu().numpy().tolist(), def_mesh_sv_fn, add_one=True)
                  
                  
                  s_verts = merged_def_verts.detach().cpu().numpy()[i_bsz]
                  s_faces = src_faces[0].detach().cpu().numpy().tolist()
                  
                  s_sampled_pts = data_utils.sample_pts_from_mesh(s_verts, s_faces, npoints=mesh_sample_npoints, minus_one=False)
                  s_sampled_pts = data_utils.fps_fr_numpy(s_sampled_pts, n_sampling=nn_com_pts)
                  sampled_pts.append(s_sampled_pts.unsqueeze(0))
                  cur_iter_sampled_pts.append(s_sampled_pts.unsqueeze(0))
                  
                  pure_merged_def_verts = mesh_rt_dict["pure_merged_deformed"]
                  def_mesh_sv_fn = os.path.join(samples_dir, f"train_ep_{epoch}_iter_{i}_bsz_{i_bsz}_i_s_{i_sample}_src_merged_def_mesh.obj")
                  save_obj_file(pure_merged_def_verts.detach().cpu().numpy()[i_bsz], src_faces[0].detach().cpu().numpy().tolist(), def_mesh_sv_fn, add_one=True)
                  
                  s_verts = pure_merged_def_verts.detach().cpu().numpy()[i_bsz]
                  s_faces = src_faces[0].detach().cpu().numpy().tolist()
                  
                  s_sampled_pts = data_utils.sample_pts_from_mesh(s_verts, s_faces, npoints=mesh_sample_npoints, minus_one=False)
                  s_sampled_pts = data_utils.fps_fr_numpy(s_sampled_pts, n_sampling=nn_com_pts)
                  sampled_pts.append(s_sampled_pts.unsqueeze(0))
                  cur_iter_sampled_pts.append(s_sampled_pts.unsqueeze(0))
                  
              #### save basis dict ####
              sv_basis_dict_fn = os.path.join(samples_dir, f"train_ep_{epoch}_iter_{i}_i_s_{i_sample}_basis_dict.npy")
              np.save(sv_basis_dict_fn, basis_sv_dict)

              print(f"Svaing to : {def_mesh_sv_fn}")

              #### log all losses ####
              log_string(" tot %f, cd %f, gravity %f, tar_cage_shift %f, tar_extents %f, tar_offset %f, orth %f, svd %f, sp %f, cat_offset %f" %
                  (
                  np.mean(buf['tot']), np.mean(buf["cd"]), np.mean(buf["cage_shift"]), np.mean(buf["tar_cage_shift"]), np.mean(buf["tar_extents"]), np.mean(buf["tar_offset"]), np.mean(buf["orth"]), np.mean(buf["svd"]), np.mean(buf["sp"]),  np.mean(buf["cat_offset"])
                  )
              )
        
      cur_iter_sampled_pts = torch.cat(cur_iter_sampled_pts, dim=0)
      cur_iter_sampled_pts_sv_fn = os.path.join(samples_dir, f"sampled_pts_iter_{i_sample}.npy")
      np.save(cur_iter_sampled_pts_sv_fn, cur_iter_sampled_pts.detach().cpu().numpy())
      print(f"{i_sample}-th sampled pts saved to {cur_iter_sampled_pts_sv_fn}...")

    sampled_pts = torch.cat(sampled_pts, dim=0)
    print(f"Sampled_pts: {sampled_pts.size()}")
    # np.save(os.path.join(samples_dir, "tot_sampled_pts.npy"), sampled_pts.detach().cpu().numpy())



sampled_pts_sv_fn = os.path.join(samples_dir, "tot_sampled_pts.npy")
np.save(sampled_pts_sv_fn, sampled_pts.detach().cpu().numpy())

print(f"Sampled pts saved to {sampled_pts_sv_fn}...")


ref_pcs = tot_ref_pts.cuda()
gen_pcs = sampled_pts.cuda()




gen_pcs = normalie_pc_bbox_batched(gen_pcs)
ref_pcs = normalie_pc_bbox_batched(ref_pcs)


def normalize_batched_unit_cube(pc):
    shift = torch.mean(pc, dim=1, keepdim=True)
    scale = torch.std(pc.contiguous().view(pc.size(0), -1), dim=-1).unsqueeze(-1).unsqueeze(-1)
    pc = (pc - shift) / scale
    return pc
# Compute metrics
results = {}
with torch.no_grad():
    # No normalization
    normed_gen_pcs = gen_pcs
    normed_ref_pcs = ref_pcs
    
    # results = compute_all_metrics(normed_gen_pcs.cpu(), normed_ref_pcs.cpu(), args.batch_size)
    results = compute_all_metrics(normed_gen_pcs, normed_ref_pcs, opt.batch_size)
    results = {k:v.item() for k, v in results.items()}
    
    for k, v in results.items():
        logger.info('%s: %.12f' % (k, v))
    
    normed_gen_pcs = normalize_batched_unit_cube(gen_pcs)
    normed_ref_pcs = normalize_batched_unit_cube(ref_pcs)
    normed_gen_pcs = normalize_batched_unit_cube(normed_gen_pcs)
    normed_ref_pcs = normalize_batched_unit_cube(normed_ref_pcs)
    
    jsd = jsd_between_point_cloud_sets(normed_gen_pcs.cpu().numpy(), normed_ref_pcs.cpu().numpy())
    results['jsd'] = jsd

for k, v in results.items():
    logger.info('%s: %.12f' % (k, v))


