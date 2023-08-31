from multiprocessing.sharedctypes import Value
from platform import java_ver
import torch
import torch.nn.functional as F
import torch.nn as nn
# from pointnet_utils import pointnet_encoder

from pointnet_utils import pointnet_encoder, PointFlowEncoder


# from losses import chamfer_distance
from losses import chamfer_distance_raw as chamfer_distance
import utils
from pointnet2 import PointnetPP
import edge_propagation
import model_utils
from common_utils.data_utils_torch import farthest_point_sampling, batched_index_select
from common_utils.data_utils_torch import compute_normals_o3d, get_vals_via_nearest_neighbours
from scipy.optimize import linear_sum_assignment
import numpy as np

from training.networks_get3d import DMTETSynthesisNetwork

from diffusion_model import GaussianVAE

from pvcnn.modules.pvconv import PVConv

class Discriminator(nn.Module): ### 
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.leakyrelu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.leakyrelu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(32, 32, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.leakyrelu3 = nn.LeakyReLU(0.2, inplace=True)

        self.fc = nn.Linear(8192, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.leakyrelu1(self.bn1(self.conv1(x)))
        x = self.leakyrelu2(self.bn2(self.conv2(x)))
        x = self.leakyrelu3(self.bn3(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.sigmoid(self.fc(x))
        return x



class model(nn.Module):
    def __init__(self, num_basis, opt=None):
        super(model, self).__init__()
        print("for network", opt)
        
        # self.cat_deform_net = deform_model(num_basis=num_basis, opt=opt) ## cat_deform_net... ##
        self.opt = opt
        self.use_prob = self.opt.use_prob
        self.tar_basis = self.opt.tar_basis
        self.coef_multiplier = self.opt.coef_multiplier
        self.n_layers = self.opt.n_layers
        
        self.num_basis = num_basis
        # self.pred_offset = self.opt.pred_offset ### pred_offset ###
        self.pred_type = self.opt.pred_type ### neighbouring_k ###
        self.neighbouring_k = opt.neighbouring_k ### neighbouring_k ###
        
        self.n_samples = opt.n_samples
        self.symmetry_axis = opt.symmetry_axis ### symmetry_axis ###
        
        self.wo_keypts_abs = opt.wo_keypts_abs
        
        print(f"Using symmetry_axis: {self.symmetry_axis}")
        
        print(f"prediction type: {self.pred_type}")
        self.use_pointnet2 = self.opt.use_pointnet2
        print(f"whether to use pointnet2: {self.use_pointnet2}")
        self.use_graphconv = self.opt.use_graphconv
        print(f"whether to use graphconv: {self.use_graphconv}")
        
        #### pp_tar_out_feat ####
        self.use_pp_tar_out_feat = self.opt.use_pp_tar_out_feat
        self.use_prob_src = self.opt.use_prob_src
        
        self.use_cvx_feats = self.opt.use_cvx_feats
        
        self.use_def_pc = self.opt.use_def_pc
        
        self.use_trans_range = self.opt.use_trans_range
        self.use_delta_prob = self.opt.use_delta_prob
        self.use_cond_vae = self.opt.use_cond_vae
        self.use_vae_opt = self.opt.use_vae_opt #### vae_option --> 
        self.pred_positions = self.opt.pred_positions
        
        self.cond_tar_pc = self.opt.cond_tar_pc
        # self.recon_cond_tar = self.opt.recon_cond_tar
        self.recon_cond = self.opt.recon_cond
        
        print(f"self.recon_cond: {self.recon_cond}")
        
        self.use_recon_as_cd = self.opt.use_recon_as_cd
        # self.use_vae_opt = 
        print(f"vae option: {self.use_vae_opt}")
        
        self.use_cond_vae_cvx_feats = self.opt.use_cond_vae_cvx_feats
        print(f"whether to use pp_tar_out_feat: {self.use_pp_tar_out_feat}")
        
        self.neighbouring_tar_k = 8
        self.neighbouring_tar_k = 1
        
        self.decode_cond_pos = self.opt.decode_cond_pos ### whether cond pos for decoding
        
        in_channels = 3
        kernel_size = 5
        feat_dim = 512
        self.feat_dim = feat_dim
        
        self.pvconv = PVConv(3, feat_dim, kernel_size, 100, with_se=False, normalize=True, eps=0)
        
        if not self.use_graphconv:
            if self.use_pointnet2:
                # self.pointnet = PointnetPP(in_feat_dim=3, use_light_weight=False, args=opt)
                # self.tar_pointnet = PointnetPP(in_feat_dim=3, use_light_weight=False, args=opt)
                self.pp_out_dim = 128
                
                self.pointnet = edge_propagation.KNNFeatureBlock(k=self.neighbouring_k, encode_feat_dim=self.pp_out_dim)
                self.tar_pointnet = edge_propagation.KNNFeatureBlock(k=self.neighbouring_k, encode_feat_dim=self.pp_out_dim)
                self.pp_out_dim = 128
                
                self.prob_pointnet = edge_propagation.KNNFeatureBlock(k=self.neighbouring_k, encode_feat_dim=self.pp_out_dim)
                self.tar_prob_pointnet = edge_propagation.KNNFeatureBlock(k=self.neighbouring_k, encode_feat_dim=self.pp_out_dim)
                self.pp_out_dim = 128
                
                ### neighbourign_k for flow net encoding ----> encode as a whole --> decode for each convex; condition on [each convex's flow latent vector, convex features, current ptss cvx-relative features] for the ground-truth flow decoding ---> encode flows as latent values 
                if self.use_vae_opt in ["flow", "flow_cvx", "diffusion", "flow_pc"]:
                    self.flow_net_in_dim = 6 if self.use_vae_opt in ["flow", "flow_cvx", "flow_pc"] else 3
                    if self.cond_tar_pc:
                        self.flow_net_in_dim = 3
                    self.flow_encoding_net = edge_propagation.KNNFeatureBlock(k=self.neighbouring_k, encode_feat_dim=self.pp_out_dim, in_dim=self.flow_net_in_dim)
                
                self.flow_encoding_net_out_dim = 512
                self.pp_out_dim = 128
                self.flow_encoding_net = PointFlowEncoder(zdim=self.flow_encoding_net_out_dim, input_dim=3, use_deterministic_encoder=False)
                # self.pp_out_dim = 128 + 1024
            else: ### pp_out_dim...
                self.pointnet = pointnet_encoder()
                self.tar_pointnet = pointnet_encoder()
                self.pp_out_dim = 2883
        else:
            self.graph_conv_net = edge_propagation.GraphConvNet(n_layers=self.n_layers)
            self.tar_graph_conv_net = edge_propagation.GraphConvNet(n_layers=self.n_layers)
            self.pp_out_dim = 128 ### 128
            

        if self.use_vae_opt == "diffusion":
            class dummy:
                def __init__(self):
                    self.latent_dim = 64
                    self.beta_1 = 1e-4
                    self.beta_T = 0.005
                    self.sched_mode = 'linear'
                    self.residual = True
                    self.num_steps = 20
            diffusion_args = dummy()
            self.gaussian_diffusion = GaussianVAE(args=diffusion_args)
        
        
        
        # src point feature 2883 * N
        self.conv11 = torch.nn.Conv1d(self.pp_out_dim, 128, 1)
        self.conv12 = torch.nn.Conv1d(128, 128, 1)
        self.conv13 = torch.nn.Conv1d(128, 128, 1)
        self.bn11 = nn.BatchNorm1d(128)
        self.bn12 = nn.BatchNorm1d(128)
        self.bn13 = nn.BatchNorm1d(128)
        
        self.tar_conv11 = torch.nn.Conv1d(self.pp_out_dim, self.pp_out_dim, 1)
        self.tar_conv12 = torch.nn.Conv1d(128, 128, 1)
        self.tar_conv13 = torch.nn.Conv1d(128, 128, 1)
        self.tar_bn11 = nn.BatchNorm1d(128)
        self.tar_bn12 = nn.BatchNorm1d(128)
        self.tar_bn13 = nn.BatchNorm1d(128)
        
        
        self.tar_conv11 = torch.nn.Conv1d(self.pp_out_dim, self.pp_out_dim, 1)
        self.tar_conv12 = torch.nn.Conv1d(128, 128, 1)
        self.tar_conv13 = torch.nn.Conv1d(128, 128, 1)
        self.tar_bn11 = nn.BatchNorm1d(128)
        self.tar_bn12 = nn.BatchNorm1d(128)
        self.tar_bn13 = nn.BatchNorm1d(128)
        
        self.glb_feat_out_net = nn.Sequential(
             torch.nn.Conv1d(self.feat_dim, self.feat_dim, 1), nn.ReLU(), 
             torch.nn.Conv1d(self.feat_dim, self.feat_dim, 1), nn.ReLU(), 
             torch.nn.Conv1d(self.feat_dim, self.feat_dim, 1)
        )
        
        self.prob_src_out_conv_net = nn.Sequential(
            *[torch.nn.Conv1d(self.pp_out_dim, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
              torch.nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
              torch.nn.Conv1d(128, 128, 1), # nn.BatchNorm1d(128)
              ]
        )
        
        self.prob_tar_out_conv_net = nn.Sequential(
            *[torch.nn.Conv1d(self.pp_out_dim, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
              torch.nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
              torch.nn.Conv1d(128, 128, 1),#  nn.BatchNorm1d(128)
              ]
        )
        
        # if self.use_vae_opt == "flow" or self.use_vae_opt == "flow_cvx" or self.use_vae_opt == "diffusion": ### pp_flow_out conv_net ###
        if len(self.use_vae_opt) > 0:
            self.flow_out_conv_net = nn.Sequential(
                *[torch.nn.Conv1d(self.pp_out_dim, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                torch.nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                torch.nn.Conv1d(128, 128, 1),#  nn.BatchNorm1d(128)
                ]
            )
        
        
        # w_dim,  # Intermediate latent (W) dimensionality.
            # img_resolution,  # Output image resolution.
            # img_channels,  # Number of color channels.
            # device='cuda',
            # data_camera_mode='carla',
            # geometry_type='normal',
            # tet_res=64,  # Resolution for tetrahedron grid
            # render_type='neural_render',  # neural type
            # use_tri_plane=False,
            # n_views=2,
            # tri_plane_resolution=128,
            # deformation_multiplier=2.0,
            # feat_channel=128,
            # mlp_latent_channel=256,
            # dmtet_scale=1.8,
            # inference_noise_mode='random',
            # one_3d_generator=False,
            # **block_kwargs,  # Arguments for SynthesisBlock.
        # if self.use_vae_opt == "implicit": 
        tet_res = 256
        tet_res = 100
        self.dmt_syn_net = DMTETSynthesisNetwork( ### 
        #   w_dim=128, 
          w_dim=feat_dim,
          # w_dim=512,
          img_resolution=256, img_channels=3, device='cuda', geometry_type="conv3d", use_tri_plane=True, tet_res=tet_res, #  one_3d_generator=True # num_ws=1
        )
        
        self.num_ws = 7
        
        ### 1) consume features;

        # key point feature K (64 + 3 + 1) * N
        self.conv21 = torch.nn.Conv1d(68, 64, 1)
        self.conv22 = torch.nn.Conv1d(64, 64, 1)
        self.bn21 = nn.BatchNorm1d(64)
        self.bn22 = nn.BatchNorm1d(64)
        
        self.tar_conv21 = torch.nn.Conv1d(128, 64, 1)
        self.tar_conv22 = torch.nn.Conv1d(64, 64, 1)
        self.tar_bn21 = nn.BatchNorm1d(64)
        self.tar_bn22 = nn.BatchNorm1d(64)
        
        self.prob_tar_recon_net = nn.Sequential(
            torch.nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            torch.nn.Conv1d(128, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            torch.nn.Conv1d(64, 64, 1), # nn.BatchNorm1d(64),
        )
        
        #### for conv ####
        # self.prob_conv21 = torch.nn.Conv1d(128, 128, 1)
        # self.prob_conv22 = torch.nn.Conv1d(128, 128, 1)
        # self.prob_bn21 = nn.BatchNorm1d(128)
        # self.prob_bn22 = nn.BatchNorm1d(128)
        
        # self.prob_conv = nn.Sequential(
        #     *[self.prob_conv21, self.prob_bn21, nn.ReLU(), self.prob_conv22, self.prob_bn22]
        # )
        
        self.prob_conv = nn.Sequential(
            *[torch.nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
              torch.nn.Conv1d(256, 512, 1), nn.BatchNorm1d(512), nn.ReLU(), 
              torch.nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256), nn.ReLU(), 
              torch.nn.Conv1d(256, 128, 1), #  nn.BatchNorm1d(128), 
              ]
        )

        keypts_trans_nn_feats = 64 + 3 if not self.wo_keypts_abs else 64
        # basis feature K 64
        # self.conv31 = torch.nn.Conv1d(64 + 3, 256, 1)
        self.conv31 = torch.nn.Conv1d(keypts_trans_nn_feats, 256, 1)
        self.conv32 = torch.nn.Conv1d(256, 512, 1)
        self.conv33 = torch.nn.Conv1d(512, self.num_basis * 3, 1)
        self.bn31 = nn.BatchNorm1d(256)
        self.bn32 = nn.BatchNorm1d(512)
        
        
        
        if self.use_cond_vae:
            self.basis_pred_nn_in_dim = 128 + 128 + 3
            self.normal_dists_nn_in_dim =  128 + 128 + 3 + 3
        else:
            self.basis_pred_nn_in_dim = 128 + 128 + 3
            self.normal_dists_nn_in_dim =  128 + 128 + 3
        # self.basis_pred_nn_mid_dim = 
        self.basis_pred_conv_nets = nn.Sequential(
          torch.nn.Conv1d(self.basis_pred_nn_in_dim, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
          torch.nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
          torch.nn.Conv1d(128, 3, 1), 
        )
        
        ### range prediction and coef prediction ###
        self.coef_pred_conv_nets = nn.Sequential(
            torch.nn.Conv1d(self.basis_pred_nn_in_dim, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            torch.nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            torch.nn.Conv1d(128, 3, 1), 
        )
        
        
        ### mus and sigmas prediction conv nets ###
        self.normal_dists_statistics_pred_conv_nets = nn.Sequential(
            torch.nn.Conv1d(self.normal_dists_nn_in_dim, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            torch.nn.Conv1d(256, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            torch.nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            torch.nn.Conv1d(128, 6, 1),  ### 3 xj
        )
        
        self.cvx_feats_z_dim = 64
        ### mus and sigmas prediction conv nets ###
        self.normal_dists_statistics_pred_conv_nets_cvx_net = nn.Sequential(
            torch.nn.Conv1d(128 + 128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            torch.nn.Conv1d(256, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            torch.nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            torch.nn.Conv1d(128, self.cvx_feats_z_dim * 2, 1),  ### 3 xj
        )
        
        # if self.use_vae_opt == "flow" or self.use_vae_opt == "flow_cvx" or self.use_vae_opt == "diffusion" or self.use_vae_opt == "flow_pc":
        if len(self.use_vae_opt) > 0:
            self.flow_feats_z_dim = 64 
            self.flow_stats_net_in_dim = 128 + 128 ### if with convex feats
            if self.use_vae_opt == "flow_pc":
                self.cvx_feats_z_dim = 128
            self.normal_dists_statistics_pred_conv_nets_flow_net = nn.Sequential(
                torch.nn.Conv1d(self.flow_stats_net_in_dim, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
                torch.nn.Conv1d(256, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
                torch.nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                torch.nn.Conv1d(128, self.cvx_feats_z_dim * 2, 1),  ### ### z_j ### use_jvae_opt == flow --> flow_latent, src_cvx_feats, tar_cvx_feats, relative positions
            ) ### sample flow cond and sampl
        
        
        n_pts_per_convex = 256
        if self.use_vae_opt == "flow_pc": ### pc_recon_net...
            ###### recon_cond_tar #######
            # if self.recon_cond_tar:
            #     self.pc_recon_net_in_dim = 128 + 128
            # else:
            #     self.pc_recon_net_in_dim = 128
            ###### recon_cond_tar #######
            if self.recon_cond == "cvx":
                self.pc_recon_net_in_dim = 128 + 128
            elif self.recon_cond == "bbox":
                self.pc_recon_net_in_dim = 128 + 3
            else:
                self.pc_recon_net_in_dim = 128
            self.pc_recon_net = nn.Sequential(
                torch.nn.Conv1d(self.pc_recon_net_in_dim, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(), ### treat convexes as batches ...
                torch.nn.Conv1d(1024, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(),
                # torch.nn.Conv1d(512, 512, 1), nn.BatchNorm1d(512), nn.ReLU(),
                torch.nn.Conv1d(1024, n_pts_per_convex * 3, 1) ### pc reconstruction network...
            )
        
        if self.use_cond_vae:
            self.delta_pts_recon_net_in_dim = 128 + 128 + 3 + 3 ### cat with other dims
        elif self.use_cond_vae_cvx_feats:
            self.delta_pts_recon_net_in_dim = 64 + 3
        else:
            self.delta_pts_recon_net_in_dim = 6
        
        if self.use_vae_opt == "flow" or self.use_vae_opt == "flow_cvx":
            if self.decode_cond_pos:
                self.delta_pts_recon_net_in_dim = self.flow_feats_z_dim + 128 + 128 + 3 ### flow latent dim, cvx feats dim, cvx dim, relative position dim
            else:
                self.delta_pts_recon_net_in_dim = self.flow_feats_z_dim + 128 + 128
        
        
        if (self.use_cond_vae or self.use_cond_vae_cvx_feats or self.use_vae_opt == "flow" or self.use_vae_opt == "flow_cvx") and self.use_trans_range:
            self.delta_pts_recon_net_out_dim = 6
        else:
            self.delta_pts_recon_net_out_dim = 3
        ### feature channel should be in the second dimension ###
        self.delta_pts_recon_net = nn.Sequential(
            torch.nn.Conv1d(self.delta_pts_recon_net_in_dim, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            torch.nn.Conv1d(256, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            torch.nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            torch.nn.Conv1d(128, self.delta_pts_recon_net_out_dim, 1),  ### 3 xj
        )

        # key point feature with target K (2048 + 64 + 3) ### key points 
        if self.use_pp_tar_out_feat:
            self.coeff_pred_in_dim = 64 + 64
        else:
            self.coeff_pred_in_dim = 2048 + 64 + 3
            
        # self.conv41 = torch.nn.Conv1d(2048 + 64 + 3, 256, 1)
        self.conv41 = torch.nn.Conv1d(self.coeff_pred_in_dim, 256, 1)
        self.conv42 = torch.nn.Conv1d(256, 128, 1)
        self.conv43 = torch.nn.Conv1d(128, 128, 1)
        self.bn41 = nn.BatchNorm1d(256)
        self.bn42 = nn.BatchNorm1d(128)
        self.bn43 = nn.BatchNorm1d(128)

        # coef feature 15 (128 + 3 + 3) K
        self.conv51 = torch.nn.Conv1d(128 + 3 + 3, 256, 1)
        self.conv52 = torch.nn.Conv1d(256, 128, 1)
        self.conv53 = torch.nn.Conv1d(128, 128, 1)
        self.bn51 = nn.BatchNorm1d(256)
        self.bn52 = nn.BatchNorm1d(128)
        self.bn53 = nn.BatchNorm1d(128)

        # coef feature 15 128
        self.conv61 = torch.nn.Conv1d(128 + 2, 64, 1)
        self.conv62 = torch.nn.Conv1d(64, 32, 1)
        self.conv63 = torch.nn.Conv1d(32, 1, 1)
        self.bn61 = nn.BatchNorm1d(64)
        self.bn62 = nn.BatchNorm1d(32)

        self.conv71 = torch.nn.Conv1d(64 + 3 + 3, 32, 1)
        self.conv72 = torch.nn.Conv1d(32, 16, 1)
        self.conv73 = torch.nn.Conv1d(16, 2, 1)
        self.bn71 = nn.BatchNorm1d(32)
        self.bn72 = nn.BatchNorm1d(16)

        self.sigmoid = nn.Sigmoid()


    
    def get_correspondances(self, src_keypts, tar_keypts, src_verts, src_faces, tar_verts, tar_faces):
        ##### 1. their coordiantes direction  
        ### indexes as correspondances ###
        bsz = src_keypts.size(0)
        tot_keypts_correspondences = []
        for i_bsz in range(bsz):
            cur_src_keypts, cur_tar_keypts = src_keypts[i_bsz], tar_keypts[i_bsz]
            cur_src_verts, cur_src_faces = src_verts[i_bsz], src_faces[i_bsz]
            cur_tar_verts, cur_tar_faces = tar_verts[i_bsz], tar_faces[i_bsz]
            cur_src_verts_normals = compute_normals_o3d(cur_src_verts, cur_src_faces)
            cur_tar_verts_normals = compute_normals_o3d(cur_tar_verts, cur_tar_faces)
            cur_src_keypts_normals = get_vals_via_nearest_neighbours(src_keypts, cur_src_verts, cur_src_verts_normals)
            cur_tar_keypts_normals = get_vals_via_nearest_neighbours(tar_keypts, cur_tar_verts, cur_tar_verts_normals)
            
            
            pos_dist_keypts = torch.sum((cur_src_keypts.unsqueeze(-2) - cur_tar_keypts.unsqueeze(0)) ** 2, dim=-1) ### n_src x n_tar (keypts)
            normal_dists_keypts = torch.sum((cur_src_keypts_normals.unsqueeze(-2) * cur_tar_keypts_normals.unsqueeze(0)) , dim=-1) ### n_src x n_tar (keypts)
            
            pos_dist_keypts_src_src = torch.sum((cur_src_keypts.unsqueeze(-2) - cur_src_keypts.unsqueeze(0)) ** 2, dim=-1) ### n_src x n_tar (keypts)
            self_arange = torch.arnage(cur_src_keypts.size(0), dtype=torch.long).cuda()
            pos_dist_keypts_src_src[self_arange.unsqueeze(-1) == self_arange.unsqueeze(0)] = 999999.0
            minn_nei_keypts_dists = torch.min(pos_dist_keypts_src_src, dim=-1)[0]
            wei_lambda = minn_nei_keypts_dists.mean().item()
            ##### dists_val: n_keypts x n_keypts ####
            dists_val = pos_dist_keypts + wei_lambda * normal_dists_keypts
            
            row_ind, col_ind = linear_sum_assignment(dists_val.detach().cpu().numpy())
            row_ind = torch.from_numpy(row_ind).long().cuda()
            col_ind = torch.from_numpy(col_ind).long().cuda()
            
            row_arg_sort_pos = torch.argsort(row_ind, descending=False)
            corresponding_col_ind = batched_index_select(values=col_ind, indices=row_arg_sort_pos, dim=0) ## (n_pos,) ### in the cuda device 
            tot_keypts_correspondences.append(corresponding_col_ind.unsqueeze(0))
        tot_keypts_correspondences = torch.cat(tot_keypts_correspondences, dim=0)
        return tot_keypts_correspondences
            
    
    def get_correspondances_coordinates(self, src_keypts, tar_keypts):
        ##### 1. their coordiantes direction  
        ### indexes as correspondances ###
        bsz = src_keypts.size(0)
        tot_keypts_correspondences = []
        tot_keypts_self_correspondences = []
        for i_bsz in range(bsz):
            cur_src_keypts, cur_tar_keypts = src_keypts[i_bsz], tar_keypts[i_bsz]
            # cur_src_verts, cur_src_faces = src_verts[i_bsz], src_faces[i_bsz]
            # cur_tar_verts, cur_tar_faces = tar_verts[i_bsz], tar_faces[i_bsz]
            # cur_src_verts_normals = compute_normals_o3d(cur_src_verts, cur_src_faces)
            # cur_tar_verts_normals = compute_normals_o3d(cur_tar_verts, cur_tar_faces)
            # cur_src_keypts_normals = get_vals_via_nearest_neighbours(src_keypts, cur_src_verts, cur_src_verts_normals)
            # cur_tar_keypts_normals = get_vals_via_nearest_neighbours(tar_keypts, cur_tar_verts, cur_tar_verts_normals)
            
            
            pos_dist_keypts = torch.sum((cur_src_keypts.unsqueeze(-2) - cur_tar_keypts.unsqueeze(0)) ** 2, dim=-1) ### n_src x n_tar (keypts)
            # normal_dists_keypts = torch.sum((cur_src_keypts_normals.unsqueeze(-2) * cur_tar_keypts_normals.unsqueeze(0)) , dim=-1) ### n_src x n_tar (keypts)
            
            # pos_dist_keypts_src_src = torch.sum((cur_src_keypts.unsqueeze(-2) - cur_src_keypts.unsqueeze(0)) ** 2, dim=-1) ### n_src x n_tar (keypts)
            # self_arange = torch.arnage(cur_src_keypts.size(0), dtype=torch.long).cuda()
            # pos_dist_keypts_src_src[self_arange.unsqueeze(-1) == self_arange.unsqueeze(0)] = 999999.0
            # minn_nei_keypts_dists = torch.min(pos_dist_keypts_src_src, dim=-1)[0]
            # wei_lambda = minn_nei_keypts_dists.mean().item()
            ##### dists_val: n_keypts x n_keypts ####
            dists_val = pos_dist_keypts # + wei_lambda * normal_dists_keypts
            
            row_ind, col_ind = linear_sum_assignment(dists_val.detach().cpu().numpy())
            # print(f"row_ind: {row_ind.shape}, col_ind: {col_ind.shape}")
            row_ind = torch.from_numpy(row_ind).long().cuda() ## row_ind: 
            col_ind = torch.from_numpy(col_ind).long().cuda()
            
            row_arg_sort_pos = torch.argsort(row_ind, descending=False)
            corresponding_col_ind = batched_index_select(values=col_ind, indices=row_arg_sort_pos, dim=0) ## (n_pos,) ### in the cuda device 
            tot_keypts_correspondences.append(corresponding_col_ind.unsqueeze(0))
            
            corresponding_row_ind = batched_index_select(values=row_ind, indices=row_arg_sort_pos, dim=0)
            tot_keypts_self_correspondences.append(corresponding_row_ind.unsqueeze(0))
        tot_keypts_correspondences = torch.cat(tot_keypts_correspondences, dim=0) ## tot_keypts_correspondences: bsz x n_pts
        tot_keypts_self_correspondences = torch.cat(tot_keypts_self_correspondences, dim=0)
        return tot_keypts_correspondences, tot_keypts_self_correspondences
    
    ### conditional deformation? 
    ### conditional deformation ---> conditional 
    ### deformation base 

    def forward(self, src_pc, tar_pc, key_pts, dst_key_pts, w_pc,  src_verts, src_convex_pts, src_faces, src_edges, src_dofs, tar_verts, dst_convex_pts, tar_faces, tar_edges, tar_dofs, deform_net=None):
        ### pont net or pvcnns for the feature extraction ###
        
        ### src_convex_pts ---- only use convexes with the same idx in those two shapes ###
        ### src_convex_pts: bsz x n_cvxes x n_pts_per_convex 
        ### src_keypts: bsz x n_keypts x 3
        #### B, N, _
        #### add src_faces, tar_faces for src_meshes and tar_meshes #### 
        #### network target driven ####
        
        # tar_pc_mean = torch.mean(tar_pc, dim=1, keepdim=True)
        # tar_pc_scale = torch.std(tar_pc.view(tar_pc.size(0), -1), dim=-1, keepdim=True).unsqueeze(-1)
        # tar_pc = (tar_pc - tar_pc_mean) / tar_pc_scale
        
        # maxx_tar_pc = torch.max(tar_pc, dim=1, keepdim=True)[0]
        # minn_tar_pc = torch.min(tar_pc, dim=1, keepdim=True)[0]
        # extents = maxx_tar_pc - minn_tar_pc
        
        ### voxels for diffusion model training... ### 
        ## 
        
        
        ### hierarchical deform
        # tar_pc = tar_pc / torch.clamp(extents, min=1e-6) 
        
        
        scale_factor = torch.tensor([2.0, 0.6, 1.0], dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(0)
        scale_factor_v = torch.tensor([1.0, 0.7, 1.0], dtype=torch.float32).cuda().unsqueeze(0)
        tar_pc = tar_pc * scale_factor
        
        
        
        B, N, _ = src_pc.shape
        
        ### convex pts and ### convex pts and ### deformation offset 
        
        ### src_pc: bsz x n_pc x 3
        ### tar_pc: bsz x 
        
        maxx_src_pc, _ = torch.max(src_pc, dim=1)
        minn_src_pc, _ = torch.min(src_pc, dim=1)
        maxx_src_cvx_pts, _ = torch.max(src_convex_pts, dim=2)
        minn_src_cvx_pts, _ = torch.min(src_convex_pts, dim=2)
        maxx_src_cvx_pts = maxx_src_cvx_pts.max(1)[0]
        minn_src_cvx_pts = minn_src_cvx_pts.min(1)[0]
        extends_src_pc = maxx_src_pc - minn_src_pc
        extends_src_cvx_pts = maxx_src_cvx_pts - minn_src_cvx_pts
        scale_src_pc = torch.sqrt(torch.sum(extends_src_pc ** 2, dim=-1))
        scale_src_cvx_pts = torch.sqrt(torch.sum(extends_src_cvx_pts ** 2, dim=-1))
        
        
        
        maxx_dst_pc, _ = torch.max(tar_pc, dim=1)
        minn_dst_pc, _ = torch.min(tar_pc, dim=1)
        maxx_dst_cvx_pts, _ = torch.max(dst_convex_pts, dim=2)
        minn_dst_cvx_pts, _ = torch.min(dst_convex_pts, dim=2)
        maxx_dst_cvx_pts = maxx_dst_cvx_pts.max(1)[0]
        minn_dst_cvx_pts = minn_dst_cvx_pts.min(1)[0]
        extends_dst_pc = maxx_dst_pc - minn_dst_pc ### dst_pc, dst_pc...
        extends_dst_cvx_pts = maxx_dst_cvx_pts - minn_dst_cvx_pts
        scale_dst_pc = torch.sqrt(torch.sum(extends_dst_pc ** 2, dim=-1))
        scale_dst_cvx_pts = torch.sqrt(torch.sum(extends_dst_cvx_pts ** 2, dim=-1))
        
        dst_convex_pts = dst_convex_pts / scale_dst_cvx_pts * scale_dst_pc
        
        # src_convex_pts =
        src_convex_pts = src_convex_pts / scale_src_cvx_pts * scale_src_pc
        
        
        src_convex_pts = dst_convex_pts
        
        # print(f"maxx_src_pc: {maxx_src_pc}, minn_src_pc: {minn_src_pc}, maxx_src_cvx_pts: {maxx_src_cvx_pts}, minn_src_cvx_pts: {minn_src_cvx_pts}, scale_src_pc: {scale_src_pc}, scale_src_cvx_pts: {scale_src_cvx_pts}")
        
        # if self.opt.hier_stage == 1:
        #     with torch.no_grad(): ## space deformation net
        #         deform_net = deform_net # if deform_net is not None else self.cat_deform_net ## external deformation net ##
        #         if deform_net is not None:
        #             def_key_pts, def_pc, cat_cd_loss, cat_basis, cat_coef, cat_tot_sampled_def_key_pts, cat_sym_loss, coef_range, loss_prior, src_pp_topk_idxes, tar_pp_topk_idxes = \
        #             deform_net(src_pc, tar_pc, key_pts, dst_key_pts, w_pc,  src_verts, src_edges, src_dofs, tar_verts, tar_edges, tar_dofs)
        #         else:
        def_key_pts = key_pts
        def_pc = src_pc
        # else:
        #     deform_net = deform_net if deform_net is not None else self.cat_deform_net
            
        #     def_key_pts, def_pc, cat_cd_loss, cat_basis, cat_coef, cat_tot_sampled_def_key_pts, cat_sym_loss, coef_range, loss_prior, src_pp_topk_idxes, tar_pp_topk_idxes = \
        #     deform_net(src_pc, tar_pc, key_pts, dst_key_pts, w_pc,  src_verts, src_edges, src_dofs, tar_verts, tar_edges, tar_dofs)

        if self.opt.hier_stage == 1:
            def_key_pts = def_key_pts.detach()
            def_pc = def_pc.detach()
        
        bsz = src_pc.size(0)
        n_samples = 1024
        n_samples = 512
        n_cvx, n_cvx_pts = src_convex_pts.size(1), src_convex_pts.size(2)
        # 1024
        
        ''' convex pts; convex features '''
        if self.use_cvx_feats: ### 
            flat_src_convex_pts = src_convex_pts.contiguous().view(bsz * n_cvx, n_cvx_pts, 3)

            fps_src_idx = farthest_point_sampling(pos=flat_src_convex_pts[:, :, :3], n_sampling=n_samples) 
            src_convex_pts = src_convex_pts.contiguous().view(bsz * n_cvx * n_cvx_pts, 3)[fps_src_idx].contiguous().view(bsz * n_cvx, n_samples, 3).contiguous().view(bsz, n_cvx, n_samples, 3).contiguous()
            
            ### flat_dst_convex_pts: flat_dst_convex_pts
            flat_dst_convex_pts = dst_convex_pts.contiguous().view(bsz * n_cvx, n_cvx_pts, 3)
            fps_dst_idx = farthest_point_sampling(pos=flat_dst_convex_pts[:, :, :3], n_sampling=n_samples) 
            dst_convex_pts = dst_convex_pts.contiguous().view(bsz * n_cvx * n_cvx_pts, 3)[fps_dst_idx].contiguous().view(bsz * n_cvx, n_samples, 3).contiguous().view(bsz, n_cvx, n_samples, 3).contiguous()
            
            n_cvx_pts = dst_convex_pts.size(2)
        ''' convex pts; convex features '''
        
        cat_def_key_pts = def_key_pts.clone()
        cat_def_pc = def_pc.clone()
          
        src_pc = def_pc
        key_pts = def_key_pts
        
        
        nn_pc = src_pc.size(1)
        n_pc_samples = 512
        fps_src_pc_idx = farthest_point_sampling(pos=src_pc[:, :, :3], n_sampling=n_pc_samples) 
        sampled_src_pc = src_pc.view(bsz * src_pc.size(1), 3).contiguous()[fps_src_pc_idx].contiguous().view(bsz, n_pc_samples, 3).contiguous() ### n_pc_samples
        
        nn_pc = tar_pc.size(1)
        # n_pc_samples = 512
        fps_tar_pc_idx = farthest_point_sampling(pos=tar_pc[:, :, :3], n_sampling=n_pc_samples) 
        sampled_tar_pc = tar_pc.view(bsz * tar_pc.size(1), 3).contiguous()[fps_tar_pc_idx].contiguous().view(bsz, n_pc_samples, 3).contiguous() ### n_pc_samples


        #### pc --> keypts --> verts ####
        src_dist_keypts_cvx = torch.sum((key_pts.unsqueeze(2).unsqueeze(2) - src_convex_pts.unsqueeze(1)) ** 2, dim=-1)
        minn_src_dist_keypts_cvx, _ = torch.min(src_dist_keypts_cvx, dim=-1) ## bsz x n_keypts x n_cvx
        keypts_cvx_dist, keypts_cvx_idx = torch.min(minn_src_dist_keypts_cvx, dim=-1) ## bsz x n_keypts ### keypts_cvx_idx: (bsz x n_pts)
        
        
        #### pc --> keypts --> verts ####
        src_dist_pc_cvx = torch.sum((sampled_src_pc.unsqueeze(2).unsqueeze(2) - src_convex_pts.unsqueeze(1)) ** 2, dim=-1)
        minn_src_dist_pc_cvx, _ = torch.min(src_dist_pc_cvx, dim=-1) ## bsz x n_keypts x n_cvx
        _, pc_cvx_idx = torch.min(minn_src_dist_pc_cvx, dim=-1) ## bsz x n_keypts ### keypts_cvx_idx: (bsz x n_pts)
        ### for each batch; for each convex; deformed pts; target pts 
        
        ''' TODO: how to calculate pc-pc correspondences is also a question '''
        ## tar_pc: bsz x n_pts x 3
        # dist_src_pc_tar_pc = torch.sum((src_pc.unsqueeze(2) - tar_pc.unsqueeze(1)) ** 2, dim=-1) #### bsz x n_pts x n_pts ####
        # _, src_pc_minn_tar_pc_idx = torch.min(dist_src_pc_tar_pc, dim=-1) ### bsz x n_pts
        
        
        # ''' unsample '''
        src_pc_minn_tar_pc_idx, _ = self.get_correspondances_coordinates(sampled_src_pc, sampled_tar_pc)
        src_pc_tar_pc = batched_index_select(values=sampled_tar_pc, indices=src_pc_minn_tar_pc_idx, dim=1) ### bsz x n_pts x 
        src_pc_offset = src_pc_tar_pc - sampled_src_pc ### src_pc_tar_pc ###
        
        
        # dist_src_keypts_tar_pc = torch.sum((key_pts.unsqueeze(2) - tar_pc.unsqueeze(1)) ** 2, dim=-1)
        # _, src_keypts_minn_tar_pc_idx = torch.min(dist_src_keypts_tar_pc, dim=-1) ### bsz x n_keypts
        
        src_keypts_minn_tar_pc_idx, _ = self.get_correspondances_coordinates(key_pts, sampled_tar_pc) ### src_keypts_tar_pcs...
        src_keypts_tar_pc = batched_index_select(values=sampled_tar_pc,  indices=src_keypts_minn_tar_pc_idx, dim=1) ### bsz x n_keypts x 3 ### src_keypts_tar_pc
        src_keypts_offset = src_keypts_tar_pc - key_pts
        
        
        ### src_keypts_minn
        
        #### use convex features ####
        if self.use_cvx_feats:
            # src_convex_pts = src_convex_pts.contiguous().view(bsz, n_cvx, n_cvx_pts, 3).contiguous()
            # dst_convex_pts = dst_convex_pts.contiguous().view(bsz, n_cvx, n_cvx_pts, 3).contiguous()
            # print(f"src_cvx_out: {src_cvx_out.size()}, tar_cvx_out: {tar_cvx_out.size()}")
            
            # # print(f"src_convex_pts: {src_convex_pts.size()}, dst_convex_pts: {dst_convex_pts.size()}") ### use_cvx_feats
            src_convex_pts = src_convex_pts.contiguous().view(bsz * n_cvx, n_cvx_pts, 3).contiguous()
            dst_convex_pts = dst_convex_pts.contiguous().view(bsz * n_cvx, n_cvx_pts, 3).contiguous()

            src_cvx_out, _ = self.pointnet(src_convex_pts)
            tar_cvx_out, _ = self.tar_pointnet(dst_convex_pts)
            
            src_cvx_out = src_cvx_out.max(dim=-1)[0].unsqueeze(-1)
            tar_cvx_out = tar_cvx_out.max(dim=-1)[0].unsqueeze(-1)


            src_cvx_out = F.relu(self.bn11(self.conv11(src_cvx_out)))
            src_cvx_out = F.relu(self.bn12(self.conv12(src_cvx_out)))
            src_cvx_out = self.bn13(self.conv13(src_cvx_out)) ### src cvx 

            tar_cvx_out = F.relu(self.tar_bn11(self.tar_conv11(tar_cvx_out)))
            tar_cvx_out = F.relu(self.tar_bn12(self.tar_conv12(tar_cvx_out)))
            tar_cvx_out = self.tar_bn13(self.tar_conv13(tar_cvx_out)) ## tar_out: bsz x n_cvx x dim 
            
            src_convex_pts = src_convex_pts.contiguous().view(bsz, n_cvx, n_cvx_pts, 3).contiguous()
            dst_convex_pts = dst_convex_pts.contiguous().view(bsz, n_cvx, n_cvx_pts, 3).contiguous()
        else:
            ''' Filter convexes via keypts '''         ### unique cvx idxes ###
            unique_cvx_idxes = torch.unique(keypts_cvx_idx.view(-1).contiguous())
            cvx_pts_indicators = torch.zeros((n_cvx,), dtype=torch.float32).cuda()
            cvx_pts_indicators[unique_cvx_idxes] = 1. ### (n_cvx, ) ### unique cvx idxes... ###
            ''' Filter convexes via keypts ''' 
            
            ''' Get pp-features for source shape and target shape ''' 
            # src_out: bsz x dim x n_pc;  ### src_pc; tar_pc ###
            src_out, _ = self.pointnet(src_pc) ### (bsz x n_cvx) x n_cvx_pts x 3 ---> (bsz x n_cvx) x n_cvx_pts x dim
            tar_out, _ = self.tar_pointnet(tar_pc) 
            ''' Get pp-features for source shape and target shape ''' 
            
            #### src_out: bsz x dim x n_pc
            #### tar_out: bsz x dim x n_pc
            
            ### keypts: bsz x n_keypts x 3; convex: bsz x n_cvx x n_cvx_pts x 3
            ### src_dist_keypts_cvx: bsz x n_keypts x n_cvx x n_cvx_pts
            
            ''' Get src_dist_pc_cvx_indicators ''' 
            ### src_pc: bsz x n_pc x 3; src_convex_pts: bsz x n_cvx x n_cvx_pts x 3
            src_dist_pc_cvx = torch.sum((src_pc.unsqueeze(2).unsqueeze(2) - src_convex_pts.unsqueeze(1)) ** 2, dim=-1) ### bsz x n_pc x n_cvx x n_cvx_pts  ### bsz x n_pc x n_cvx  ### bsz x n_pc
            src_dist_pc_cvx, _ = torch.min(src_dist_pc_cvx, dim=-1)
            src_dist_pc_cvx = src_dist_pc_cvx + (1. - cvx_pts_indicators.unsqueeze(0).unsqueeze(0)) * 999999.0
            src_dist_pc_cvx_minn, src_dist_pc_cvx_minn_idxes = torch.min(src_dist_pc_cvx, dim=-1)
            
            src_dist_pc_cvx_indicators = torch.zeros((bsz, N, n_cvx), dtype=torch.float32).cuda()
            src_dist_pc_cvx_indicators[:, :, src_dist_pc_cvx_minn_idxes] = 1.
            ''' Get src_dist_pc_cvx_indicators ''' 

            ''' Get dst_dist_pc_cvx_indicators '''
            ### src_pc: bsz x n_pc x 3; src_convex_pts: bsz x n_cvx x n_cvx_pts x 3
            tar_dist_pc_cvx = torch.sum((tar_pc.unsqueeze(2).unsqueeze(2) - dst_convex_pts.unsqueeze(1)) ** 2, dim=-1) ### bsz x n_pc x n_cvx x n_cvx_pts
            tar_dist_pc_cvx, _ = torch.min(tar_dist_pc_cvx, dim=-1) ### bsz x n_pc x n_cvx ### bsz x n_pc x n_cvx
            tar_dist_pc_cvx = tar_dist_pc_cvx + (1. - cvx_pts_indicators.unsqueeze(0).unsqueeze(0)) * 999999.0
            tar_dist_pc_cvx_minn, tar_dist_pc_cvx_minn_idxes = torch.min(tar_dist_pc_cvx, dim=-1) ### bsz x n_pc

            tar_dist_pc_cvx_indicators = torch.zeros((bsz, N, n_cvx), dtype=torch.float32).cuda() ### N, n_cvx 
            tar_dist_pc_cvx_indicators[:, :, tar_dist_pc_cvx_minn_idxes] = 1. ### tar
            ''' Get dst_dist_pc_cvx_indicators '''

            ### src_out: bsz x dim x N ###
            src_out_expanded = src_out.contiguous().unsqueeze(-1).repeat(1, 1, 1, n_cvx).contiguous()
            tar_out_expanded = tar_out.contiguous().unsqueeze(-1).repeat(1, 1, 1, n_cvx).contiguous()
            
            ### src_out_cvx: bsz x dim x n_cvx
            src_cvx_out, _ = torch.max((src_out_expanded - (1. - src_dist_pc_cvx_indicators) * 999999.0), dim=2)
            tar_cvx_out, _ = torch.max((tar_out_expanded - (1. - tar_dist_pc_cvx_indicators) * 999999.0), dim=2)
            
            ## src_cvx_out ---> src_features
            src_cvx_out = src_cvx_out.contiguous().transpose(-1, -2).contiguous().view(bsz * n_cvx, -1).unsqueeze(-1)
            ## tar_cvx_out ---> tar_features
            tar_cvx_out = tar_cvx_out.contiguous().transpose(-1, -2).contiguous().view(bsz * n_cvx, -1).unsqueeze(-1)
            
            src_cvx_out = F.relu(self.bn11(self.conv11(src_cvx_out)))
            src_cvx_out = F.relu(self.bn12(self.conv12(src_cvx_out)))
            src_cvx_out = self.bn13(self.conv13(src_cvx_out)) ### src cvx 

            tar_cvx_out = F.relu(self.tar_bn11(self.tar_conv11(tar_cvx_out)))
            tar_cvx_out = F.relu(self.tar_bn12(self.tar_conv12(tar_cvx_out)))
            tar_cvx_out = self.tar_bn13(self.tar_conv13(tar_cvx_out)) ## tar_out: bsz x n_cvx x dim 


        ### tar_cvx_out ###
        # n_samples = self.n_samples
        # n_samples = 2048 ## convex features & other features concat

        N = src_verts[0].size(0)

        ''' Get convexes features from convex points '''
        # # print(f"src_convex_pts: {src_convex_pts.size()}, dst_convex_pts: {dst_convex_pts.size()}")
        
        # src_convex_pts = src_convex_pts.contiguous().view(bsz * n_cvx, n_cvx_pts, 3).contiguous()
        # dst_convex_pts = dst_convex_pts.contiguous().view(bsz * n_cvx, n_cvx_pts, 3).contiguous()
        
        # src_cvx_out, _ = self.pointnet(src_convex_pts) ### (bsz x n_cvx) x n_cvx_pts x 3 ---> (bsz x n_cvx) x n_cvx_pts x dim
        # tar_cvx_out, _ = self.tar_pointnet(dst_convex_pts)
        
        # src_cvx_out = src_cvx_out.max(dim=-1)[0].unsqueeze(-1)
        # tar_cvx_out = tar_cvx_out.max(dim=-1)[0].unsqueeze(-1)
        ''' Get convexes features from convex points '''
        
        ## cvx_flow_outs ##


        src_cvx_out = src_cvx_out.view(bsz, n_cvx, -1).contiguous()
        tar_cvx_out = tar_cvx_out.view(bsz, n_cvx, -1).contiguous()
        
        
        
        ## bsz x n_keypts x dim; src_cvx_out xxx keypts_cvx_idx ---> bsz x n_keypts x dim
        keypts_cvx_feats = batched_index_select(values=src_cvx_out, indices=keypts_cvx_idx, dim=1)  ### key_pts_cvx 
        ## bsz x n_keypts x dim; tar_cvx_out xxx keypts_cvx_idx ---> bsz x n_keypts x dim
        keypts_dst_cvx_feats = batched_index_select(values=tar_cvx_out, indices=keypts_cvx_idx, dim=1 )
        
        pc_cvx_feats = batched_index_select(values=src_cvx_out, indices=pc_cvx_idx, dim=1)
        pc_dst_cvx_feats = batched_index_select(values=tar_cvx_out, indices=pc_cvx_idx, dim=1 )
        
        
        ''' Keypts to cvx features '''
        keypts_tot_cvx_feats = torch.cat(
          [keypts_cvx_feats, keypts_dst_cvx_feats], dim=-1 ### bsz x n_keypts x (dim + dim)
        )
        src_cvx_maxx_coords, _ = torch.max(src_convex_pts, dim=2)
        src_cvx_minn_coords, _ = torch.min(src_convex_pts, dim=2)
        src_cvx_center = (src_cvx_maxx_coords + src_cvx_minn_coords) / 2. #### cvx_maxx_coords, cvx_minn_coords
        # src_cvx_center = src_cvx_center.view(bsz, n_cvx, 3) ### bsz x n_cvx x 3
        ### key_pts: bsz x n_keypts x 3 ###
        src_key_cvx_center = batched_index_select(values=src_cvx_center, indices=keypts_cvx_idx, dim=1) ### bsz x n_keypts x 3
        rel_src_key_cvx_center = key_pts - src_key_cvx_center ### bsz x n_keypts x 3

        keypts_tot_cvx_feats = torch.cat(
          [keypts_tot_cvx_feats, rel_src_key_cvx_center], dim=-1 ### bsz x n_keypts x (dim + dim + 3)
        )
        ### bsz x (in_feat_dim) x n_keyptss
        keypts_basis = self.basis_pred_conv_nets(keypts_tot_cvx_feats.view(bsz * keypts_tot_cvx_feats.size(1), -1).unsqueeze(-1))
        keypts_basis = keypts_basis.squeeze(-1).view(bsz, -1, 3).contiguous()
        
        
        
        flow_feats_in = src_pc_offset ### flow_feats_in ### ### sampled_s 
            ## sampled_src_pc: bsz x n_sampled x 3; pc_cvx: bsz x n_cvx x n_pts x 3
            ### flow_pc_cvx xxx 
        ''' dist determined by shp-pts-to-cvx-pts distances '''
        minn_dist_pc_to_cvx_pts = utils.get_avg_minn_shp_pts_to_cvx_pts_batch(tar_pc, dst_convex_pts) ### bsz
        dist_sampled_pc_cvx_pts = torch.sum((sampled_tar_pc.unsqueeze(2).unsqueeze(2) - dst_convex_pts.unsqueeze(1)) ** 2, dim=-1) 
        dist_sampled_pc_cvx_pts = torch.sqrt(dist_sampled_pc_cvx_pts) ### bsz x n_sampled_pts x n_cvx x n_cvx_pts
        dist_sampled_pc_cvx, _ = torch.min(dist_sampled_pc_cvx_pts, dim=-1) ### bsz x n_sampled x n_cvx ### 
        pc_cvx_indicator = (dist_sampled_pc_cvx <= minn_dist_pc_to_cvx_pts.unsqueeze(-1).unsqueeze(-1) * 2).float()
                
              
        #### src_convex_pts, src_convex_pts ####
        bsz, n_cvx = src_convex_pts.size(0), src_convex_pts.size(1)
        # print(f"bsz: {bsz}, n_cvx: {n_cvx}")
        tot_recon_cvx_cd = []
        tot_recon_tot_cd = []
        tot_cvx_recon_pts = []
        rnd_tot_cvx_recon_pts = []
        tot_loss_prior = []
        tot_loss_entropy = []
        tot_loss_log_z = []
        
        tot_cvx_vs = [[] for i_bsz in range(bsz)]
        tot_cvx_fs = [[] for i_bsz in range(bsz)]
        tot_cvx_to_seg_pts = []
        
        pc_cvx_indicator[:, :, 0] = 1.
        pc_cvx_indicator[:, :, 1:] = 0.
        
        
        
        obj_pc = tar_pc # b x n_pts x 3
        
        ''' point net featuers '''
        # obj_out, _ = self.pointnet(obj_pc) ### obj_out features 
        ''' point net features '''

        obj_pc_features = obj_pc - torch.mean(obj_pc, dim=1, keepdim=True) ### b x n_pts x 3 ---> obj_pc --> 
        obj_pc_features = obj_pc_features.contiguous().transpose(-1, -2).contiguous() ### b x 3 x n_pts 
        fused_features, voxel_coords = self.pvconv((obj_pc_features, obj_pc.contiguous().transpose(-1, -2).contiguous())) ### contiguous() transpose() view()  ### obj
        ## fused_features: b x dim x n_pts 
        obj_out = fused_features
        
        
        obj_out = obj_out.max(dim=-1)[0].unsqueeze(-1) ## bsz x dim x 1

        obj_out = self.glb_feat_out_net(obj_out)

        # obj_out = F.relu((self.conv11(obj_out)))
        # obj_out = F.relu((self.conv12(obj_out)))
        # obj_out = self.conv13(obj_out)
        
        obj_out = obj_out.squeeze(-1)

        
        # print(obj_out.size())
        
        # bsz_to_recon_pts
        for i_bsz in range(bsz):
            cur_bsz_tot_recon_pts = []
            cur_bsz_chamfer_dist = []
            cur_bsz_cvx_recon_pts = []
            # rnd_cur_bsz_cvx_recon_pts = []
            # cur_bsz_loss_prior = []
            # cur_bsz_loss_entropy = []
            # cur_bsz_loss_log_z = []
            cur_cvx_to_seg_pts = {}
            
            
            cur_bsz_cvx_out = obj_out[i_bsz : i_bsz + 1, :].unsqueeze(1).repeat(1, self.num_ws + 4 + 1, 1)
            
            # v_list, f_list, sdf, deformation, v_deformed, sdf_reg_loss = self.dmt_syn_net.get_geometry_prediction(cur_bsz_cvx_out, sdf_feature=None, position=tar_pc) ### get geometry prediction
            v_list, f_list, sdf, deformation, v_deformed, sdf_reg_loss = self.dmt_syn_net.get_geometry_prediction(cur_bsz_cvx_out, sdf_feature=None) ### get geometry prediction

            cur_v = v_list[0]
            cur_f = f_list[0]
            
            cur_v = cur_v * scale_factor_v
            
            # print(f"maxx_tar_pc: {torch.max(tar_pc[i_bsz], dim=0)[0]}, minn_tar_pc: {torch.min(tar_pc[i_bsz], dim=0)[0]}, maxx_cur_v: {torch.max(cur_v, dim=0)[0]}, minn_cur_v: {torch.min(cur_v, dim=0)[0]}")
            
            # print(f"cur_v: {cur_v.size()}, cur_f: {cur_f.size()}")
            # cur_bsz_scale = utils.get_vertices_scale_torch_no_batch(obj_pc[i_bsz])
            # cur_bsz_bbox_maxx, _ = torch.max(obj_pc[i_bsz], dim=0, keepdim=True)
            # cur_bsz_bbox_minn, _ = torch.min(obj_pc[i_bsz], dim=0, keepdim=True)
            # cur_bsz_bbox_center = (cur_bsz_bbox_maxx + cur_bsz_bbox_minn)  / 2.

            # cur_v = utils.normalie_pc_bbox_batched(cur_v.unsqueeze(0)).squeeze(0)
            
            # print()
                
            # cur_v = cur_v * cur_bsz_scale
            
            # cur_v = cur_v + cur_bsz_bbox_center
            
            # cur_v = cur_v
            
            
            # denorm_v = (cur_v * tar_pc_scale[i_bsz]) + tar_pc_mean[i_bsz]
            # denorm_v = cur_v * extents[i_bsz]
            tot_cvx_vs[i_bsz].append(cur_v)
            # tot_cvx_vs[i_bsz].append(denorm_v)
            tot_cvx_fs[i_bsz].append(cur_f)
            
        
            
            cur_v_fps = farthest_point_sampling(cur_v.unsqueeze(0), n_sampling=src_pc.size(1))
            tot_sampled_pts = cur_v[cur_v_fps]
            
            
            cur_bsz_tot_recon_pts = [tot_sampled_pts]

            # tot_sdf_reg_loss += sdf_reg_loss
            
            
            # cur_bsz_cvx_cd = chamfer_distance(obj_pc[i_bsz].unsqueeze(0), tot_sampled_pts.unsqueeze(0)) ### 
            cur_bsz_cvx_cd = chamfer_distance(obj_pc[i_bsz].unsqueeze(0), cur_v.unsqueeze(0)) ### 
            
            
            cur_bsz_chamfer_dist.append(cur_bsz_cvx_cd) 
            
            cur_cvx_to_seg_pts[0] = tot_sampled_pts.detach().cpu().numpy()
    
    
            # for i_cvx in range(n_cvx):
            #     cur_bsz_cur_cvx_pts = src_convex_pts[i_bsz, i_cvx] ### n_cvx_ptss x 3 
            #     cur_bsz_cur_cvx_pc_indicators = pc_cvx_indicator[i_bsz, :, i_cvx]
            #     cur_bsz_cvx_nn_pc = int(cur_bsz_cur_cvx_pc_indicators.sum().item())
            #     # print(f"cur_bsz_cvx_nn_pc: {cur_bsz_cvx_nn_pc}")
            #     if cur_bsz_cvx_nn_pc > 0:
                  
            #         cur_bsz_cvx_sorted_pc_idxes = torch.argsort(cur_bsz_cur_cvx_pc_indicators, dim=-1, descending=True)
            #         cur_bsz_cvx_pc_idxes = cur_bsz_cvx_sorted_pc_idxes[: cur_bsz_cvx_nn_pc] ### nn_pc 
            #         cur_bsz_cvx_pc = sampled_tar_pc[i_bsz, cur_bsz_cvx_pc_idxes] ### sampled_tar_pc: bsz x nn_tot_pts x 3 ---> nn_sampled_pts x 3
                    
            #         cur_cvx_to_seg_pts[i_cvx] = cur_bsz_cvx_pc.detach().cpu().numpy()
                    
            #         cur_bsz_cvx_pc = cur_bsz_cvx_pc.unsqueeze(0) ### 1 x nn_sampled_pts x 3 ### cur_bsz_cvx_pc- -> 1 x n_pts x 3 --> bsz pc
            #         # print(f"cur_bsz_cvx_nn_pc: {cur_bsz_cvx_pc.size()}")
                    
                    
                    
            #         # cur_bsz_cvx_pc = utils.normalie_pc_bbox_batched(cur_bsz_cvx_pc) ### normalize for encoding
                    
                    
            #         cur_bsz_cvx_out_mus, cur_bsz_cvx_out_vs = self.flow_encoding_net(cur_bsz_cvx_pc) ### no flow as inputs ### 
            #         ### cur_bsz_cvxout: bsz x dim x n_pts 
                    
            #         # print(f"cur_bsz_cvx_out_mus: {cur_bsz_cvx_out_mus.size()}")
                    
            #         cur_bsz_cvx_out = cur_bsz_cvx_out_mus
                    
            #         cur_bsz_cvx_out = tar_cvx_out[i_bsz, i_cvx].unsqueeze(0)
                      
                      ### 
                    ### minn dist from one to another
                    
            #         ''' VAE ''' 
            #         # cur_bsz_cvx_out_z = utils.reparameterize_gaussian(mean=cur_bsz_cvx_out_mus, logvar=cur_bsz_cvx_out_vs) ### cvx_flow_z: bsz x dim x n_cvx
            #         # pc_loss_prior = utils.kld_loss(cur_bsz_cvx_out_mus, cur_bsz_cvx_out_vs).mean() ### pc_prior_loss ## kld_loss ### add pc prior loss ... 
                    
            #         # cur_bsz_loss_prior.append(pc_loss_prior)
            #         # # cur_bsz_cvx_out, _ = torch.max(cur_bsz_cvx_out, dim=-1)
                    
            #         # cur_bsz_cvx_out = cur_bsz_cvx_out_z
            #         # cur_bsz_cvx_out = self.flow_out_conv_net(cur_bsz_cvx_out.unsqueeze(-1)) ## 1 x dim
                    
            #         ### cur_bsz_cvx_out: bsz x dim
            #         # print(f"encoded features: {cur_bsz_cvx_out}")
                    
            #         #### cvx_flow_out ####
            #         cur_bsz_cvx_out = cur_bsz_cvx_out.contiguous().unsqueeze(1).repeat(1, self.num_ws + 4 + 1, 1)
                    
            #         v_list, f_list, sdf, deformation, v_deformed, sdf_reg_loss = self.dmt_syn_net.get_geometry_prediction(cur_bsz_cvx_out, sdf_feature=None)


            #         cur_v = v_list[0]
            #         cur_f = f_list[0]
            #         # print(f"cur_v: {cur_v.size()}, cur_f: {cur_f.size()}")
            #         cur_bsz_cvx_pts = dst_convex_pts[i_bsz, i_cvx] ### n_pts x 3
            #         cur_bsz_cvx_dst_scale = utils.get_vertices_scale_torch_no_batch(cur_bsz_cvx_pts)

            #         cur_v = utils.normalie_pc_bbox_batched(cur_v.unsqueeze(0)).squeeze(0)
                      
            #         cur_v = cur_v * cur_bsz_cvx_dst_scale
                    
            #         cur_v = cur_v + src_cvx_center[i_bsz, i_cvx].unsqueeze(0)
                    
            #         tot_cvx_vs[i_bsz].append(cur_v)
            #         tot_cvx_fs[i_bsz].append(cur_f)
                    
            #         tot_cur_cvx_pts = cur_v
                    
            #         cur_cvx_sampled_pts_fps = farthest_point_sampling(tot_cur_cvx_pts.unsqueeze(0), n_sampling=2048)
            #         tot_cur_cvx_pts = tot_cur_cvx_pts[cur_cvx_sampled_pts_fps]
                    
                    
            #         cur_bsz_tot_recon_pts.append(tot_cur_cvx_pts)

            #         # tot_sdf_reg_loss += sdf_reg_loss
                    
            #         cur_bsz_cvx_cd = chamfer_distance(sampled_tar_pc[i_bsz, cur_bsz_cvx_pc_idxes].unsqueeze(0), tot_cur_cvx_pts.unsqueeze(0)) ### 
                    
                    
            #         cur_bsz_chamfer_dist.append(cur_bsz_cvx_cd) 
                    
                    
                    
                    
                    
                    
            cur_bsz_chamfer_dist = sum(cur_bsz_chamfer_dist) / max(1e-6, float(len(cur_bsz_chamfer_dist)))
            # cur_bsz_loss_prior = sum(cur_bsz_loss_prior) / max(1e-6, float(len(cur_bsz_loss_prior)))
            # cur_bsz_loss_entropy = sum(cur_bsz_loss_entropy) / max(1e-6, float(len(cur_bsz_loss_entropy)))
            # cur_bsz_loss_log_z = sum(cur_bsz_loss_log_z) / max(1e-6, float(len(cur_bsz_loss_log_z)))
            # tot_loss_prior.append(cur_bsz_loss_prior)
            # tot_loss_entropy.append(cur_bsz_loss_entropy)
            # tot_loss_log_z.append(cur_bsz_loss_log_z)
            
            tot_cvx_to_seg_pts.append(cur_cvx_to_seg_pts)
            
            if len(cur_bsz_tot_recon_pts) > 0:
                cur_bsz_tot_recon_pts = torch.cat(cur_bsz_tot_recon_pts, dim=0) ## bsz x n_recon_pts x 3 ---> 
                # cur_bsz_cd  = chamfer_distance(cur_bsz_tot_recon_pts.unsqueeze(0), sampled_tar_pc[i_bsz].unsqueeze(0))
                
                cur_bsz_cd  = chamfer_distance(cur_bsz_tot_recon_pts.unsqueeze(0), tar_pc[i_bsz].unsqueeze(0)) ### tar_pc
                
                tot_recon_cvx_cd.append(cur_bsz_chamfer_dist)
                tot_recon_tot_cd.append(cur_bsz_cd)
                
            # else:
            #     ## sampled_tar_pc[i_bsz]: n_sampled_pts x 3 ##
            #     tot_cvx_recon_pts.append(sampled_tar_pc[i_bsz].unsqueeze(0).unsqueeze(0))
            #     rnd_tot_cvx_recon_pts.append(sampled_tar_pc[i_bsz].unsqueeze(0).unsqueeze(0))
            
        tot_recon_cvx_cd = sum(tot_recon_cvx_cd) / max(1e-6, float(len(tot_recon_cvx_cd)))
        tot_recon_tot_cd = sum(tot_recon_tot_cd) / max(1e-6, float(len(tot_recon_tot_cd)))
        ### tot_cvx_recon_pt
        # if len(tot_cvx_recon_pts) > 0:
        #     #### 
        #     tot_cvx_recon_pts = torch.cat(tot_cvx_recon_pts, dim=0) ### bsz xn_cvx x n_pts x 3
        #     cvx_recon_pts = tot_cvx_recon_pts   
            
        #     rnd_tot_cvx_recon_pts = torch.cat(rnd_tot_cvx_recon_pts, dim=0)
        #     rnd_cvx_recon_pts = rnd_tot_cvx_recon_pts
        
        # cd_loss =  tot_recon_cvx_cd.mean() + tot_recon_tot_cd.mean()
        # cd_loss =  tot_recon_cvx_cd.mean() # + tot_recon_tot_cd.mean()
        cd_loss =   tot_recon_tot_cd.mean()
        # cd_loss =   tot_recon_tot_cd.mean()
        # if cd_loss.item() > 10.:
        #     cd_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean()
        
        # cd_loss = torch.clamp(cd_loss, max=torch.tensor([10.], dtype=torch.float32).cuda().mean())
        
        # cd_loss  = cd_loss * 1e-4
        
        
        # pc_loss_prior = sum(tot_loss_prior) / max(1e-6, float(len(tot_loss_prior))) 

        loss_prior =  torch.zeros((1,), dtype=torch.float32).cuda().mean()
        
        
        loss_log_pz = torch.zeros((1,), dtype=torch.float32).cuda().mean()
        loss_entropy = torch.zeros((1,), dtype=torch.float32).cuda().mean()


        rt_dict = {
          "cd_loss": cd_loss, 
          # "tot_sdf_reg_loss": tot_sdf_reg_loss, 
          # "tot_recon_meshes": tot_recon_meshes, 
          # "cvx_recon_pts": cvx_recon_pts, 
          "recon_vs": tot_cvx_vs,
          "recon_fs": tot_cvx_fs,
          "dst_convex_pts": dst_convex_pts, 
          "loss_prior": loss_prior, 
          "loss_log_pz": loss_log_pz, 
          "loss_entropy": loss_entropy, 
          "tot_cvx_to_seg_pts": tot_cvx_to_seg_pts
          # "rnd_tot_recon_meshes": tot_rnd_recon_meshes
        }
        


        return rt_dict

