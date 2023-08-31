from multiprocessing.sharedctypes import Value
from platform import java_ver
import torch
import torch.nn.functional as F
import torch.nn as nn
from pointnet_utils import pointnet_encoder, PointFlowEncoder
# from losses import chamfer_distance
from losses import chamfer_distance_raw as chamfer_distance
import utils
from pointnet2 import PointnetPP
import edge_propagation

from common_utils.data_utils_torch import farthest_point_sampling, batched_index_select
from common_utils.data_utils_torch import compute_normals_o3d, get_vals_via_nearest_neighbours
from scipy.optimize import linear_sum_assignment
import numpy as np

# from diffusion_model import GaussianVAE

from vae_gaussian_ori import GaussianVAE

from PointFlow.models.networks import PointFlow
from PointFlow.args import get_args
from pvcnn.modules.pvconv import PVConv

from network import model as glb_deform_model


### 
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
        
        self.pointnet = pointnet_encoder()
        self.tar_pointnet = pointnet_encoder()
        
        self.feat_dim = 2883
        self.feat_dim = 64
        self.feat_dim = 512
        
        self.dis_factor = opt.dis_factor
        
        # self.pvconv = PVConv(3, self.feat_dim, 5,  64, with_se=False, normalize=True, eps=0)
        # self.tar_pvconv = PVConv(3, self.feat_dim, 5,  64, with_se=False, normalize=True, eps=0)
        
        # (self, zdim, input_dim=3, use_deterministic_encoder=False):
        self.pn_enc = PointFlowEncoder(zdim=self.feat_dim, input_dim=3, use_deterministic_encoder=True)
        self.tar_pn_enc = PointFlowEncoder(zdim=self.feat_dim, input_dim=3, use_deterministic_encoder=True)
        
        
        self.glb_pn_enc = PointFlowEncoder(zdim=self.feat_dim, input_dim=3, use_deterministic_encoder=True)
        self.glb_tar_pn_enc = PointFlowEncoder(zdim=self.feat_dim, input_dim=3, use_deterministic_encoder=True)
        
        # self.glb_pn_enc = PVConv(3, self.feat_dim, 5,  64, with_se=False, normalize=True, eps=0)
        # self.glb_tar_pn_enc = PVConv(3, self.feat_dim, 5,  64, with_se=False, normalize=True, eps=0)
        
        
        self.src_ppfeats_out_net = nn.Sequential(
             torch.nn.Conv1d(self.feat_dim * 2, self.feat_dim, 1), nn.ReLU(), 
             torch.nn.Conv1d(self.feat_dim, self.feat_dim, 1), nn.ReLU(), 
             torch.nn.Conv1d(self.feat_dim, self.feat_dim, 1)
        )
        
        
        self.src_glb_feat_out_net = nn.Sequential(
             torch.nn.Conv1d(self.feat_dim, self.feat_dim, 1), nn.ReLU(), 
             torch.nn.Conv1d(self.feat_dim, self.feat_dim, 1), nn.ReLU(), 
             torch.nn.Conv1d(self.feat_dim, self.feat_dim, 1)
        )
        
        self.tar_glb_feat_out_net = nn.Sequential(
             torch.nn.Conv1d(self.feat_dim, self.feat_dim, 1), nn.ReLU(), 
             torch.nn.Conv1d(self.feat_dim, self.feat_dim, 1), nn.ReLU(), 
             torch.nn.Conv1d(self.feat_dim, self.feat_dim, 1)
        )
        
        self.glb_src_glb_feat_out_net = nn.Sequential(
             torch.nn.Conv1d(self.feat_dim, self.feat_dim, 1), nn.ReLU(), 
             torch.nn.Conv1d(self.feat_dim, self.feat_dim, 1), nn.ReLU(), 
             torch.nn.Conv1d(self.feat_dim, self.feat_dim, 1)
        )
        
        self.glb_tar_glb_feat_out_net = nn.Sequential(
             torch.nn.Conv1d(self.feat_dim, self.feat_dim, 1), nn.ReLU(), 
             torch.nn.Conv1d(self.feat_dim, self.feat_dim, 1), nn.ReLU(), 
             torch.nn.Conv1d(self.feat_dim, self.feat_dim, 1)
        )
        
        
        self.num_basis = num_basis
        
        self.coef_multiplier = opt.coef_multiplier
        
        self.pred_type = opt.pred_type
        
        

        # src point feature 2883 * N
        # self.conv11 = torch.nn.Conv1d(2883, 128, 1)
        self.conv11 = torch.nn.Conv1d(self.feat_dim, 128, 1)
        self.conv12 = torch.nn.Conv1d(128, 64, 1)
        self.conv13 = torch.nn.Conv1d(64, 64, 1)
        self.bn11 = nn.BatchNorm1d(128)
        self.bn12 = nn.BatchNorm1d(64)
        self.bn13 = nn.BatchNorm1d(64)

        # key point feature K (64 + 3 + 1) * N
        self.conv21_in_feats = 68 if opt.def_version == "v4" else 67
        self.conv21 = torch.nn.Conv1d(self.conv21_in_feats, 64, 1)
        # self.conv21 = torch.nn.Conv1d(67, 64, 1)
        self.conv22 = torch.nn.Conv1d(64, 64, 1)
        self.bn21 = nn.BatchNorm1d(64)
        self.bn22 = nn.BatchNorm1d(64)

        # basis feature K 64
        
        self.conv31 = torch.nn.Conv1d(64 + 3, 256, 1)
        self.conv32 = torch.nn.Conv1d(256, 512, 1)
        self.conv33 = torch.nn.Conv1d(512, self.num_basis * 3, 1)
        self.bn31 = nn.BatchNorm1d(256)
        self.bn32 = nn.BatchNorm1d(512)

        # key point feature with target K (2048 + 64 + 3)
        # self.conv41 = torch.nn.Conv1d(2048 + 64 + 3, 256, 1)
        
        self.conv41 = torch.nn.Conv1d(self.feat_dim + self.feat_dim + 64 + 3, 256, 1)
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


        coef_pred_out_dim = 3 if self.pred_type == "offset" else 1
        coef_pred_in_dim = 128 if self.pred_type == "offset" else 128 + 2
        # coef feature 15 128
        self.conv61 = torch.nn.Conv1d(coef_pred_in_dim, 64, 1)
        self.conv62 = torch.nn.Conv1d(64, 32, 1)
        self.conv63 = torch.nn.Conv1d(32, coef_pred_out_dim, 1)
        self.bn61 = nn.BatchNorm1d(64)
        self.bn62 = nn.BatchNorm1d(32)

        self.conv71 = torch.nn.Conv1d(64 + 3 + 3, 32, 1)
        self.conv72 = torch.nn.Conv1d(32, 16, 1)
        self.conv73 = torch.nn.Conv1d(16, 2, 1)
        self.bn71 = nn.BatchNorm1d(32)
        self.bn72 = nn.BatchNorm1d(16)

        self.sigmoid = nn.Sigmoid()
        
        # self.glb_deform_model = glb_deform_model(num_basis=self.num_basis, opt=opt)


    ## local chamfer distances, global chamfer distances? 
    def get_minn_distances_base_pc_cvx_pts(self, base_pc, cvx_pts): 
      ### base_pc: bsz x nn_pts x 3; cvx_pts: bsz x n_cvx x nn_cvx_pts x 3 ###
      dists_base_pc_cvx_pts = torch.sum((base_pc.unsqueeze(-2).unsqueeze(-2) - cvx_pts.unsqueeze(1)) ** 2, dim=-1) 
      dists_base_pc_cvx_pts = torch.sqrt(dists_base_pc_cvx_pts) ### bsz x nn_pts x n_cvx x nn_cvx_pts 
      dists_base_pc_cvx_pts, _ = torch.min(dists_base_pc_cvx_pts, dim=-1) ### bsz x nn_pts x nn_cvx  ### minn_dist_to_cvx
      dists_base_pc_cvx_pts, _ = torch.min(dists_base_pc_cvx_pts, dim=-1) ### bsz x nn_pts --> dists_base_pc_cvx_pts ### bsz x nn_pts 
      avg_dists_base_pc_cvx_pts = torch.mean(dists_base_pc_cvx_pts, dim=-1) ### bsz
      return avg_dists_base_pc_cvx_pts #### average distance between pc and cvx pts ####


    def get_base_pc_cvx_indicator(self, base_pc, cvx_pts, minn_dist_base_to_cvx_pts): ### minn_dist_base_to_cvx_pts -> a value
      dists_base_pc_cvx_pts = torch.sum((base_pc.unsqueeze(-2).unsqueeze(-2) - cvx_pts.unsqueeze(1)) ** 2, dim=-1) 
      dists_base_pc_cvx_pts = torch.sqrt(dists_base_pc_cvx_pts) ### bsz x nn_pts x n_cvx x nn_cvx_pts 
      dists_base_pc_cvx_pts, _ = torch.min(dists_base_pc_cvx_pts, dim=-1) ### bsz x nn_pts x nn_cvx  ### minn_dist_tocvx
    #   coef = 2.0
      coef = 2.5
      coef = self.dis_factor
      base_pc_cvx_indicators = (dists_base_pc_cvx_pts <= minn_dist_base_to_cvx_pts.unsqueeze(-1).unsqueeze(-1) * coef).float() ### indicator: bsz x nn_pts x nn_cvx --> nn_pts, nn_cvx 
      return base_pc_cvx_indicators ### base
    
    # def get_pts_idx_from_indicator(self, pc_cvx_indicator): 
    #   ### bsz x nn_pts x 
      
    ### basis and coefficients for mesh deformation ###
    ### basis 
    def forward(self, src_pc, tar_pc, key_pts, dst_key_pts, w_pc,  src_verts, src_convex_pts, tar_verts, dst_convex_pts):
        ### src_pc, tar_pc;
        ### src_convex_pts: 
        ### src_convex_pts ---- only use convexes with the same idx in those two shapes ###
        ### src_convex_pts: bsz x n_cvxes x n_pts_per_convex 
        ### src_keypts: bsz x n_keypts x 3
        #### B, N, _ = src
        #### add src_faces, tar_faces for src_meshes and tar_meshes #### 
        #### network target driven ####
        B, N, _ = src_pc.shape
        nn_cvx = src_convex_pts.size(1) ### bsz x nn_cvx 
        
        src_convex_pts = src_convex_pts * 2.
        dst_convex_pts = dst_convex_pts * 2.
        
        # print(f"maxx_src: {torch.max(src_pc)}, minn_src: {torch.min(src_pc)}, maxx src_convex_pts: {torch.max(src_convex_pts)}, minn src_convex_pts: {torch.min(src_convex_pts)}, ")
        
        ###  src ### 
        avg_src_pc_cvx = self.get_minn_distances_base_pc_cvx_pts(src_pc, src_convex_pts) ### 
        ### bsz x nn_pts x nn_cvx --> []
        src_pc_cvx_indicator = self.get_base_pc_cvx_indicator(src_pc, src_convex_pts, avg_src_pc_cvx) ### 
        src_keypts_cvx_indicator = self.get_base_pc_cvx_indicator(key_pts, src_convex_pts, avg_src_pc_cvx) ### 
        
        avg_dst_pc_cvx = self.get_minn_distances_base_pc_cvx_pts(tar_pc, dst_convex_pts)
        tar_pc_cvx_indicator = self.get_base_pc_cvx_indicator(tar_pc, dst_convex_pts, avg_dst_pc_cvx) ### dst_pc_cvx
        
        # tot_keypts_basis_offset = torch.zeros(()) ### 
        
        
        
        tot_keypts_basis_offset = torch.zeros_like(key_pts) ## bsz x nn_keypts x 3
        
        tot_keypts_add_nn = torch.zeros_like(key_pts)[..., 0] ## bsz x nn_keypts x 3
        
        tot_rnd_tot_keypts_basis_offset = [ torch.zeros_like(key_pts) for _ in range(10)]
        tot_rnd_tot_keypts_add_nn = [torch.zeros_like(key_pts)[..., 0]  for _ in range(10)]
        
        tot_cd_loss = 0.
        
        src_cvx_pc_pts = []
        tar_cvx_pc_pts = []
        src_cvx_def_pc_pts = []
        src_cvx_def_keypts_pts = []
        
        # print(f"B: {B}, nn_cvx: {nn_cvx}")
        for i_cvx in range(nn_cvx): 
          ### get the keypts offset ##
          cur_src_pc_cvx_indicator = src_pc_cvx_indicator[:, :, i_cvx] ### bsz x nn_pts 
          cur_src_keypts_cvx_indicator = src_keypts_cvx_indicator[:, :, i_cvx] ## bsz x nn_keypts 
          cur_tar_pc_cvx_indicator = tar_pc_cvx_indicator[:, :, i_cvx]
          
          cur_pc_nn = torch.sum(cur_src_pc_cvx_indicator, dim=-1).long() ## bsz
          pc_idx_argsort = torch.argsort(cur_src_pc_cvx_indicator, dim=-1, descending=True) ### bsz x 
          cur_keypts_nn = torch.sum(cur_src_keypts_cvx_indicator, dim=-1).long()
          keypts_idx_argsort = torch.argsort(cur_src_keypts_cvx_indicator, dim=-1, descending=True)
          
          cur_tar_pc_nn = torch.sum(cur_tar_pc_cvx_indicator, dim=-1).long() ## bsz
          tar_pc_idx_argsort = torch.argsort(cur_tar_pc_cvx_indicator, dim=-1, descending=True) ### bsz x 
          cur_src_cvx_pc_pts = []
          cur_tar_cvx_pc_pts = []
          
          cur_cvx_def_pc_pts = []
          cur_cvx_def_keypts_pts = []
          
          for i_bsz in range(B):
            cur_bsz_pc_nn = int(cur_pc_nn[i_bsz].item())
            cur_bsz_keypts_nn = int(cur_keypts_nn[i_bsz].item()) ### keypts_nn
            cur_bsz_tar_pc_nn = int(cur_tar_pc_nn[i_bsz].item()) ### 
            
            # print(f"cur_bsz_pc_nn: {cur_bsz_pc_nn}, cur_bsz_keypts_nn: {cur_bsz_keypts_nn}, cur_bsz_tar_pc_nn: {cur_bsz_tar_pc_nn}")
            if cur_bsz_pc_nn <= 1 or cur_bsz_keypts_nn <= 1 or cur_bsz_tar_pc_nn <= 1:
              continue
            cur_bsz_pc_idx = pc_idx_argsort[i_bsz, :cur_bsz_pc_nn]
            cur_bsz_keypts_idx = keypts_idx_argsort[i_bsz, :cur_bsz_keypts_nn] 
            cur_bsz_tar_pc_idx = tar_pc_idx_argsort[i_bsz, : cur_bsz_tar_pc_nn]
            
            ### w_pc: 
            cur_bsz_cur_cvx_w_pc = w_pc[i_bsz, cur_bsz_pc_idx, :] ## nn_cvx_pc x nn_keypts ##
            cur_bsz_cur_cvx_w_pc = cur_bsz_cur_cvx_w_pc[:, cur_bsz_keypts_idx] ## nn_cvx_pc x nn_cvx_keypts ##
            cur_bsz_cur_cvx_w_pc = cur_bsz_cur_cvx_w_pc.unsqueeze(0) ## 1 x nn_cvx_pc x nn_cvx_keypts
            cur_bsz_cur_cvx_w_pc2 = cur_bsz_cur_cvx_w_pc.transpose(2, 1).unsqueeze(2)
            
            
            # cur_src_cvx_pc_pt
            
            
            cur_bsz_pc = src_pc[i_bsz, cur_bsz_pc_idx, :].unsqueeze(0)
            cur_bsz_keypts = key_pts[i_bsz, cur_bsz_keypts_idx, :].unsqueeze(0)
            cur_bsz_tar_pc = tar_pc[i_bsz, cur_bsz_tar_pc_idx, :].unsqueeze(0) #### tar_pc_idx
            
            
            
           
            
            cur_bsz_pc = cur_bsz_pc - torch.mean(cur_bsz_pc, dim=1, keepdim=True)
            cur_bsz_tar_pc = cur_bsz_tar_pc - torch.mean(cur_bsz_tar_pc, dim=1, keepdim=True)
            cur_bsz_keypts = cur_bsz_keypts - torch.mean(cur_bsz_keypts, dim=1, keepdim=True)
            
            cur_src_cvx_pc_pts.append(cur_bsz_pc.detach().cpu().numpy())
            cur_tar_cvx_pc_pts.append(cur_bsz_tar_pc.detach().cpu().numpy())
            
            cur_bsz_keypts1 = cur_bsz_keypts.unsqueeze(-1).expand(-1, -1, -1, cur_bsz_pc_nn)
            
            cur_bsz_pc1 = cur_bsz_pc.unsqueeze(-1).expand(-1, -1, -1, cur_bsz_pc_nn)
            
            src_feats, src_glb_feats = self.pointnet(cur_bsz_pc, return_global=False) ## bsz x 
            ### 
            tar_feats, tar_glb_feats = self.tar_pointnet(cur_bsz_tar_pc, return_global=False) ### tar_feats 
            
            src_feats = F.relu(self.bn11(self.conv11(src_feats)))
            src_feats = F.relu(self.bn12(self.conv12(src_feats)))
            src_feats = F.relu(self.bn13(self.conv13(src_feats)))
        ### 
            
            ### 
            src_feats = src_feats.unsqueeze(1).expand(-1, cur_bsz_keypts_nn, -1, -1) 
            
            
            cur_bsz_keypts_feats = torch.cat(
              [src_feats, cur_bsz_cur_cvx_w_pc2, cur_bsz_keypts1], dim=2
            ).view(1 * cur_bsz_keypts_nn, -1, cur_bsz_pc_nn)
            
            
            cur_bsz_src_pc_feats = torch.cat( ### 
              [src_feats, cur_bsz_pc1], dim=2
            ).view(1 * cur_bsz_keypts_nn, -1, cur_bsz_pc_nn)
            
            # print(f"src_feats: {src_feats.size()}, cur_bsz_cur_cvx_w_pc: {cur_bsz_cur_cvx_w_pc.size()}, cur_bsz_keypts1: {cur_bsz_keypts1.size()}")
            
            
            cur_bsz_keypts_feats = F.relu(self.bn21(self.conv21(cur_bsz_keypts_feats)))
            cur_bsz_keypts_feats = self.bn22(self.conv22(cur_bsz_keypts_feats))
            
            ### cur_bsz_keypts_feats: bsz x nn_feats
            cur_bsz_keypts_feats = torch.max(cur_bsz_keypts_feats, 2, keepdim=True)[0] 
            key_fea = cur_bsz_keypts_feats.view(1 * cur_bsz_keypts_nn, -1, 1) ### 
            net = torch.cat([key_fea, cur_bsz_keypts.view(1 * cur_bsz_keypts_nn, 3, 1)], 1) 
            
            
            net = F.relu(self.bn31(self.conv31(net))) #### net
            net = F.relu(self.bn32(self.conv32(net)))
            basis = self.conv33(net).view(1, cur_bsz_keypts_nn * 3, self.num_basis).transpose(1, 2)
            basis = basis / basis.norm(p=2, dim=-1, keepdim=True)  ### basis norm 
            
            key_fea_range = key_fea.view(
                1, cur_bsz_keypts_nn, -1, 1).expand(-1, -1, -1, self.num_basis).transpose(1, 3)
            key_pts_range = cur_bsz_keypts.view(
                1, cur_bsz_keypts_nn, 3, 1).expand(-1, -1, -1, self.num_basis).transpose(1, 3)
            basis_range = basis.view(1, self.num_basis, cur_bsz_keypts_nn, 3).transpose(2, 3)
            
            
            
            ### get range
            coef_range = torch.cat([key_fea_range, key_pts_range, basis_range], 2).view(
                1 * self.num_basis, -1, cur_bsz_keypts_nn)
            coef_range = F.relu(self.bn71(self.conv71(coef_range)))
            coef_range = F.relu(self.bn72(self.conv72(coef_range)))
            coef_range = self.conv73(coef_range)
            coef_range = torch.max(coef_range, 2, keepdim=True)[0]
            # coef_range = coef_range.view(B, self.num_basis, 2) * 0.02 ### 
            # coef_range = coef_range.view(B, self.num_basis, 2) * 0.1 
            # coef_range = coef_range.view(1, self.num_basis, 2) * 0.2
            # coef_range = coef_range.view(1, self.num_basis, 2) * 0.1
            coef_range = coef_range.view(1, self.num_basis, 2) * 0.05
            # coef_range = coef_range.view(B, self.num_basis, 2) * 0.2
            coef_range[:, :, 0] = coef_range[:, :, 0] * -1

            src_tar = torch.cat([src_glb_feats, tar_glb_feats], 1).unsqueeze(
                1).expand(-1, cur_bsz_keypts_nn, -1).reshape(1 * cur_bsz_keypts_nn, -1, 1)     
            
            
            
            ### to the source shape ###
            ### keypoint features, srouce target fused features, and keypoint coordiantes ###
            ### key_feature; source_target
            key_fea = torch.cat([key_fea, src_tar, cur_bsz_keypts.view(1 * cur_bsz_keypts_nn, 3, 1)], 1)
            key_fea = F.relu(self.bn41(self.conv41(key_fea))) ### key_fea
            key_fea = F.relu(self.bn42(self.conv42(key_fea))) ### key_fea
            key_fea = F.relu(self.bn43(self.conv43(key_fea))) ### 

            key_fea = key_fea.view(1, cur_bsz_keypts_nn, 128).transpose(
                1, 2).unsqueeze(1).expand(-1, self.num_basis, -1, -1)
            cur_bsz_keypts2 = cur_bsz_keypts.view(1, cur_bsz_keypts_nn, 3).transpose(  ### key_pts2;
                1, 2).unsqueeze(1).expand(-1, self.num_basis, -1, -1)
            basis1 = basis.view(1, self.num_basis, cur_bsz_keypts_nn, 3).transpose(2, 3)

            net = torch.cat([key_fea, basis1, cur_bsz_keypts2], 2).view(
                1 * self.num_basis, -1, cur_bsz_keypts_nn)

            net = F.relu(self.bn51(self.conv51(net)))
            net = F.relu(self.bn52(self.conv52(net)))
            net = self.bn53(self.conv53(net))

            net = torch.max(net, 2, keepdim=True)[0]
            net = net.view(1 * self.num_basis, -1, 1)

            net = torch.cat([net, coef_range.view(1 * self.num_basis, 2, 1)], 1)
            net = F.relu(self.bn61(self.conv61(net)))
            net = F.relu(self.bn62(self.conv62(net)))
            coef = self.sigmoid(self.conv63(net)).view(B, self.num_basis)

            coef = (coef * coef_range[:, :, 0] + (1 - coef)
                    * coef_range[:, :, 1]).view(1, 1, self.num_basis)
            
            cur_bsz_cur_cvx_key_pts_offset = torch.bmm(coef, basis).view(1, cur_bsz_keypts_nn, 3)
            ### def_key_pts ###
            
            
            cur_bsz_cur_cvx_def_key_pts = cur_bsz_keypts + cur_bsz_cur_cvx_key_pts_offset
            
            cur_bsz_cur_cvx_def_pc = torch.bmm(cur_bsz_cur_cvx_w_pc, cur_bsz_cur_cvx_def_key_pts)
            
            cd_loss = chamfer_distance(cur_bsz_cur_cvx_def_pc, cur_bsz_tar_pc)
            tot_cd_loss += cd_loss
            # tot_keypts_basis_offset = torch.zeros_like(key_pts) ## bsz x nn_keypts x 3
            # 
            # tot_keypts_basis_offset: bsz x nn_keypts x 3
            tot_keypts_basis_offset[i_bsz, cur_bsz_keypts_idx, :] += cur_bsz_cur_cvx_key_pts_offset[0] 
            tot_keypts_add_nn[i_bsz, cur_bsz_keypts_idx] += 1
            
            ## cd_loss = chamfer_distance(def_pc, tar_pc)
            
        #     cur_cvx_def_pc_pts = []
        #   cur_cvx_def_keypts_pts = []
            cur_cvx_def_pc_pts.append(cur_bsz_cur_cvx_def_pc.detach().cpu().numpy())
            cur_cvx_def_keypts_pts.append(cur_bsz_cur_cvx_def_key_pts.detach().cpu().numpy())
            
            
            
            
            for i_s in range(len(tot_rnd_tot_keypts_basis_offset)):
            
                ratio = torch.rand((1, self.num_basis)).cuda()
                sample_coef = (ratio * coef_range[:, :, 0] + (1 - ratio)
                            * coef_range[:, :, 1]).view(1, 1, self.num_basis)
                sample_def_key_pts_offset =  \
                    torch.bmm(sample_coef, basis).view(1, cur_bsz_keypts_nn, 3)
                tot_rnd_tot_keypts_basis_offset[i_s][i_bsz, cur_bsz_keypts_idx, :] += sample_def_key_pts_offset[0] 
                tot_rnd_tot_keypts_add_nn[i_s][i_bsz, cur_bsz_keypts_idx] += 1

        
          src_cvx_pc_pts.append(cur_src_cvx_pc_pts)
          tar_cvx_pc_pts.append(cur_tar_cvx_pc_pts)
          
          src_cvx_def_pc_pts.append(cur_cvx_def_pc_pts)
          src_cvx_def_keypts_pts.append(cur_cvx_def_keypts_pts)
            

        tot_rnd_def_key_pts = []
        tot_rnd_def_pc = []
        for i_s in range(len(tot_rnd_tot_keypts_basis_offset)):
            rnd_def_key_pts = key_pts + tot_rnd_tot_keypts_basis_offset[i_s] / torch.clamp(tot_rnd_tot_keypts_add_nn[i_s].unsqueeze(-1), min=1e-6)
            tot_rnd_def_key_pts.append(rnd_def_key_pts)
            rnd_def_pc = torch.bmm(w_pc, rnd_def_key_pts)
            tot_rnd_def_pc.append(rnd_def_pc)
        
        def_key_pts = key_pts + tot_keypts_basis_offset / torch.clamp(tot_keypts_add_nn.unsqueeze(-1), min=1e-6)
        def_pc = torch.bmm(w_pc, def_key_pts)
        # cd_loss = chamfer_distance(def_pc, tar_pc)
        
        cd_loss = tot_cd_loss / nn_cvx
        
        rt_dict = {
          "cd_loss": cd_loss, 
          "def_key_pts": def_key_pts, 
          "def_pc": def_pc,
          "tot_rnd_def_key_pts": tot_rnd_def_key_pts,
          "tot_rnd_def_pc": tot_rnd_def_pc,
          "src_cvx_pc_pts": src_cvx_pc_pts,
          "tar_cvx_pc_pts": tar_cvx_pc_pts,
          "src_cvx_def_pc_pts": src_cvx_def_pc_pts,
          "src_cvx_def_keypts_pts": src_cvx_def_keypts_pts
        }
        
        return rt_dict
    
    def forward2(self, src_pc, tar_pc, key_pts, dst_key_pts, w_pc,  src_verts, src_convex_pts, tar_verts, dst_convex_pts):
        ### src_pc, tar_pc;
        ### src_convex_pts: 
        ### src_convex_pts ---- only use convexes with the same idx in those two shapes ###
        ### src_convex_pts: bsz x n_cvxes x n_pts_per_convex 
        ### src_keypts: bsz x n_keypts x 3
        #### B, N, _ = src
        #### add src_faces, tar_faces for src_meshes and tar_meshes #### 
        #### network target driven ####
        B, N, _ = src_pc.shape
        nn_cvx = src_convex_pts.size(1) ### bsz x nn_cvx 
        
        src_convex_pts = src_convex_pts * 2.
        dst_convex_pts = dst_convex_pts * 2.
        
        # w_pc: bsz x n_pc x n_keypts
        # print(f"w_pc: {w_pc.size()}, w_pc_sum: {torch.sum(w_pc, dim=-1)[0, :2]}")
        # print(f"maxx_src: {torch.max(src_pc)}, minn_src: {torch.min(src_pc)}, maxx src_convex_pts: {torch.max(src_convex_pts)}, minn src_convex_pts: {torch.min(src_convex_pts)}, ")
        
        ###  src ### 
        avg_src_pc_cvx = self.get_minn_distances_base_pc_cvx_pts(src_pc, src_convex_pts) ### 
        ### bsz x nn_pts x nn_cvx --> []
        src_pc_cvx_indicator = self.get_base_pc_cvx_indicator(src_pc, src_convex_pts, avg_src_pc_cvx) ### 
        src_keypts_cvx_indicator = self.get_base_pc_cvx_indicator(key_pts, src_convex_pts, avg_src_pc_cvx) ### 
        
        avg_dst_pc_cvx = self.get_minn_distances_base_pc_cvx_pts(tar_pc, dst_convex_pts)
        tar_pc_cvx_indicator = self.get_base_pc_cvx_indicator(tar_pc, dst_convex_pts, avg_dst_pc_cvx) ### dst_pc_cvx
        
        # tot_keypts_basis_offset = torch.zeros(()) ### 
        
        
        
        tot_keypts_basis_offset = torch.zeros_like(key_pts) ## bsz x nn_keypts x 3
        
        tot_keypts_add_nn = torch.zeros_like(key_pts)[..., 0] ## bsz x nn_keypts x 3
        
        tot_rnd_tot_keypts_basis_offset = [ torch.zeros_like(key_pts) for _ in range(10)]
        tot_rnd_tot_keypts_add_nn = [torch.zeros_like(key_pts)[..., 0]  for _ in range(10)]
        
        tot_cd_loss = 0.
        
        src_cvx_pc_pts = []
        tar_cvx_pc_pts = []
        src_cvx_def_pc_pts = []
        src_cvx_def_keypts_pts = []
        
        # print(f"B: {B}, nn_cvx: {nn_cvx}")
        for i_cvx in range(nn_cvx): 
          ### get the keypts offset ##
          cur_src_pc_cvx_indicator = src_pc_cvx_indicator[:, :, i_cvx] ### bsz x nn_pts 
          cur_src_keypts_cvx_indicator = src_keypts_cvx_indicator[:, :, i_cvx] ## bsz x nn_keypts 
          cur_tar_pc_cvx_indicator = tar_pc_cvx_indicator[:, :, i_cvx]
          
          cur_pc_nn = torch.sum(cur_src_pc_cvx_indicator, dim=-1).long() ## bsz x nn_pc
          
          
          
          pc_idx_argsort = torch.argsort(cur_src_pc_cvx_indicator, dim=-1, descending=True) ### bsz x 
          cur_keypts_nn = torch.sum(cur_src_keypts_cvx_indicator, dim=-1).long()
          keypts_idx_argsort = torch.argsort(cur_src_keypts_cvx_indicator, dim=-1, descending=True)
          
          cur_tar_pc_nn = torch.sum(cur_tar_pc_cvx_indicator, dim=-1).long() ## bsz
          tar_pc_idx_argsort = torch.argsort(cur_tar_pc_cvx_indicator, dim=-1, descending=True) ### bsz x 
          cur_src_cvx_pc_pts = []
          cur_tar_cvx_pc_pts = []
          
          cur_cvx_def_pc_pts = []
          cur_cvx_def_keypts_pts = []
          
          for i_bsz in range(B):
            cur_bsz_pc_nn = int(cur_pc_nn[i_bsz].item())
            cur_bsz_keypts_nn = int(cur_keypts_nn[i_bsz].item()) ### keypts_nn
            cur_bsz_tar_pc_nn = int(cur_tar_pc_nn[i_bsz].item()) ### 
            
            
            
            # print(f"cur_bsz_pc_nn: {cur_bsz_pc_nn}, cur_bsz_keypts_nn: {cur_bsz_keypts_nn}, cur_bsz_tar_pc_nn: {cur_bsz_tar_pc_nn}")
            if cur_bsz_pc_nn <= 1 or cur_bsz_keypts_nn <= 1 or cur_bsz_tar_pc_nn <= 1:
              continue
            cur_bsz_pc_idx = pc_idx_argsort[i_bsz, :cur_bsz_pc_nn]
            cur_bsz_keypts_idx = keypts_idx_argsort[i_bsz, :cur_bsz_keypts_nn] 
            cur_bsz_tar_pc_idx = tar_pc_idx_argsort[i_bsz, : cur_bsz_tar_pc_nn]
            
            
            cur_keypts_w_pc = w_pc[:, :, cur_bsz_keypts_idx]
            cur_keypts_w_pc_sum = torch.sum(cur_keypts_w_pc, dim=-1)
            cur_src_pc_cvx_indicator[cur_keypts_w_pc_sum < 1e-6] = 0.
            cur_pc_nn = torch.sum(cur_src_pc_cvx_indicator, dim=-1).long() ## bsz x nn_pc
            cur_bsz_pc_nn = int(cur_pc_nn[i_bsz].item())
            if cur_bsz_pc_nn <= 1 :
                continue
            pc_idx_argsort = torch.argsort(cur_src_pc_cvx_indicator, dim=-1, descending=True) ### bsz x 
            cur_bsz_pc_idx = pc_idx_argsort[i_bsz, :cur_bsz_pc_nn]
            
            
            ### w_pc: 
            cur_bsz_cur_cvx_w_pc = w_pc[i_bsz, cur_bsz_pc_idx, :] ## nn_cvx_pc x nn_keypts ##
            cur_bsz_cur_cvx_w_pc = cur_bsz_cur_cvx_w_pc[:, cur_bsz_keypts_idx] ## nn_cvx_pc x nn_cvx_keypts ##
            cur_bsz_cur_cvx_w_pc = cur_bsz_cur_cvx_w_pc.unsqueeze(0) ## 1 x nn_cvx_pc x nn_cvx_keypts
            
            
            
            cur_bsz_cur_cvx_w_pc = cur_bsz_cur_cvx_w_pc / torch.clamp(torch.sum(cur_bsz_cur_cvx_w_pc, dim=-1, keepdim=True), min=1e-96) ### 1 x nn_cvx_pc x 1
            cur_bsz_cur_cvx_w_pc2 = cur_bsz_cur_cvx_w_pc.transpose(2, 1).unsqueeze(2) 
            
            
            # cur_src_cvx_pc_pt
            
            
            cur_bsz_pc = src_pc[i_bsz, cur_bsz_pc_idx, :].unsqueeze(0)
            cur_bsz_keypts = key_pts[i_bsz, cur_bsz_keypts_idx, :].unsqueeze(0)
            cur_bsz_tar_pc = tar_pc[i_bsz, cur_bsz_tar_pc_idx, :].unsqueeze(0) #### tar_pc_idx
            
            
            
           
            
            cur_bsz_pc = cur_bsz_pc - torch.mean(cur_bsz_pc, dim=1, keepdim=True)
            cur_bsz_tar_pc = cur_bsz_tar_pc - torch.mean(cur_bsz_tar_pc, dim=1, keepdim=True)
            cur_bsz_keypts = cur_bsz_keypts - torch.mean(cur_bsz_keypts, dim=1, keepdim=True)
            
            cur_src_cvx_pc_pts.append(cur_bsz_pc.detach().cpu().numpy())
            cur_tar_cvx_pc_pts.append(cur_bsz_tar_pc.detach().cpu().numpy())
            
            cur_bsz_keypts1 = cur_bsz_keypts.unsqueeze(-1).expand(-1, -1, -1, cur_bsz_pc_nn)
            
            cur_bsz_pc1 = cur_bsz_pc.unsqueeze(-1).expand(-1, -1, -1, cur_bsz_pc_nn)
            
            
            ''' pointnet '''
            # src_feats, src_glb_feats = self.pointnet(cur_bsz_pc, return_global=False) ## bsz x 
            # ### 
            # tar_feats, tar_glb_feats = self.tar_pointnet(cur_bsz_tar_pc, return_global=False) ### tar_feats 
            ''' pointnet '''
            
            
            
            ''' pvcnn '''
            fused_features, voxel_coords = self.pvconv((cur_bsz_pc.contiguous().transpose(-1, -2).contiguous(), cur_bsz_pc.contiguous().transpose(-1, -2).contiguous())) ### contiguous() transpose() view()  ### obj
            ## fused_features: b x dim x n_pts 
            src_feats = fused_features
            
            
            src_glb_feats = src_feats.max(dim=-1)[0].unsqueeze(-1) ## bsz x dim x 1

            src_glb_feats = self.src_glb_feat_out_net(src_glb_feats).squeeze(-1)


            tar_fused_features, voxel_coords = self.tar_pvconv((cur_bsz_tar_pc.contiguous().transpose(-1, -2).contiguous(), cur_bsz_tar_pc.contiguous().transpose(-1, -2).contiguous())) ### contiguous() transpose() view()  ### obj
            ## fused_features: b x dim x n_pts 
            tar_feats = tar_fused_features
            
            
            tar_glb_feats = tar_feats.max(dim=-1)[0].unsqueeze(-1) ## bsz x dim x 1

            tar_glb_feats = self.tar_glb_feat_out_net(tar_glb_feats).squeeze(-1)      
            
            # print(f"src_feats: {src_feats.size()}, tar_glb_feats: {tar_glb_feats.size()}, src_glb_feats: {src_glb_feats.size()}")
                  
            
            
            src_feats = F.relu(self.bn11(self.conv11(src_feats)))
            src_feats = F.relu(self.bn12(self.conv12(src_feats)))
            src_feats = F.relu(self.bn13(self.conv13(src_feats)))
        ### 
            
            ### 
            src_feats = src_feats.unsqueeze(1).expand(-1, cur_bsz_pc_nn, -1, -1) 
            
            
            # cur_bsz_keypts_feats = torch.cat(
            #   [src_feats, cur_bsz_cur_cvx_w_pc2, cur_bsz_keypts1], dim=2
            # ).view(1 * cur_bsz_keypts_nn, -1, cur_bsz_pc_nn)
            
            
            cur_bsz_src_pc_feats = torch.cat( ### 
              [src_feats, cur_bsz_pc1], dim=2
            ).view(1 * cur_bsz_pc_nn, -1, cur_bsz_pc_nn)
            
            # print(f"src_feats: {src_feats.size()}, cur_bsz_cur_cvx_w_pc: {cur_bsz_cur_cvx_w_pc.size()}, cur_bsz_keypts1: {cur_bsz_keypts1.size()}")
            
            
            cur_bsz_src_pc_feats = F.relu(self.bn21(self.conv21(cur_bsz_src_pc_feats)))
            cur_bsz_src_pc_feats = self.bn22(self.conv22(cur_bsz_src_pc_feats))
            
            ### cur_bsz_keypts_feats: bsz x nn_feats
            cur_bsz_src_pc_feats = torch.max(cur_bsz_src_pc_feats, 2, keepdim=True)[0] 
            key_fea = cur_bsz_src_pc_feats.view(1 * cur_bsz_pc_nn, -1, 1) ### 
            net = torch.cat([key_fea, cur_bsz_pc.view(1 * cur_bsz_pc_nn, 3, 1)], 1) 
            
            
            net = F.relu(self.bn31(self.conv31(net))) #### net
            net = F.relu(self.bn32(self.conv32(net)))
            basis = self.conv33(net).view(1, cur_bsz_pc_nn * 3, self.num_basis).transpose(1, 2)
            basis = basis / basis.norm(p=2, dim=-1, keepdim=True)  ### basis norm 
            
            key_fea_range = key_fea.view(
                1, cur_bsz_pc_nn, -1, 1).expand(-1, -1, -1, self.num_basis).transpose(1, 3)
            key_pts_range = cur_bsz_pc.view(
                1, cur_bsz_pc_nn, 3, 1).expand(-1, -1, -1, self.num_basis).transpose(1, 3)
            basis_range = basis.view(1, self.num_basis, cur_bsz_pc_nn, 3).transpose(2, 3)
            
            
            
            ### get range
            coef_range = torch.cat([key_fea_range, key_pts_range, basis_range], 2).view(
                1 * self.num_basis, -1, cur_bsz_pc_nn)
            coef_range = F.relu(self.bn71(self.conv71(coef_range)))
            coef_range = F.relu(self.bn72(self.conv72(coef_range)))
            coef_range = self.conv73(coef_range)
            coef_range = torch.max(coef_range, 2, keepdim=True)[0]
            # coef_range = coef_range.view(B, self.num_basis, 2) * 0.02 ### 
            # coef_range = coef_range.view(B, self.num_basis, 2) * 0.1 
            coef_range = coef_range.view(1, self.num_basis, 2) * 0.2
            # coef_range = coef_range.view(1, self.num_basis, 2) * 0.1
            # coef_range = coef_range.view(1, self.num_basis, 2) * 0.05
            # coef_range = coef_range.view(B, self.num_basis, 2) * 0.2
            coef_range[:, :, 0] = coef_range[:, :, 0] * -1

            src_tar = torch.cat([src_glb_feats, tar_glb_feats], 1).unsqueeze(
                1).expand(-1, cur_bsz_pc_nn, -1).reshape(1 * cur_bsz_pc_nn, -1, 1)     
            
            
            
            ### to the source shape ###
            ### keypoint features, srouce target fused features, and keypoint coordiantes ###
            ### key_feature; source_target
            key_fea = torch.cat([key_fea, src_tar, cur_bsz_pc.view(1 * cur_bsz_pc_nn, 3, 1)], 1)
            # print(f"key_fea: {key_fea.size()}")
            key_fea = F.relu(self.bn41(self.conv41(key_fea))) ### key_fea
            key_fea = F.relu(self.bn42(self.conv42(key_fea))) ### key_fea
            key_fea = F.relu(self.bn43(self.conv43(key_fea))) ### 

            key_fea = key_fea.view(1, cur_bsz_pc_nn, 128).transpose(
                1, 2).unsqueeze(1).expand(-1, self.num_basis, -1, -1)
            cur_bsz_keypts2 = cur_bsz_pc.view(1, cur_bsz_pc_nn, 3).transpose(  ### key_pts2;
                1, 2).unsqueeze(1).expand(-1, self.num_basis, -1, -1)
            basis1 = basis.view(1, self.num_basis, cur_bsz_pc_nn, 3).transpose(2, 3)

            net = torch.cat([key_fea, basis1, cur_bsz_keypts2], 2).view(
                1 * self.num_basis, -1, cur_bsz_pc_nn)

            net = F.relu(self.bn51(self.conv51(net)))
            net = F.relu(self.bn52(self.conv52(net)))
            net = self.bn53(self.conv53(net))

            net = torch.max(net, 2, keepdim=True)[0]
            net = net.view(1 * self.num_basis, -1, 1)

            net = torch.cat([net, coef_range.view(1 * self.num_basis, 2, 1)], 1)
            net = F.relu(self.bn61(self.conv61(net)))
            net = F.relu(self.bn62(self.conv62(net)))
            coef = self.sigmoid(self.conv63(net)).view(B, self.num_basis)

            coef = (coef * coef_range[:, :, 0] + (1 - coef)
                    * coef_range[:, :, 1]).view(1, 1, self.num_basis)
            
            cur_bsz_cur_cvx_key_pts_offset = torch.bmm(coef, basis).view(1, cur_bsz_pc_nn, 3)
            ### def_key_pts ###
            
            
            cur_bsz_cur_cvx_def_key_pts = cur_bsz_pc + cur_bsz_cur_cvx_key_pts_offset
            
            # 
            dists_cur_bsz_keypts_pc = torch.sum((cur_bsz_keypts.unsqueeze(-2) - cur_bsz_pc.unsqueeze(1)) ** 2, dim=-1) 
            dists_cur_bsz_keypts_pc, dists_cur_bsz_keypts_pc_minn_idx = torch.min(dists_cur_bsz_keypts_pc, dim=-1) ## 1 x nn_keypts
            cur_bsz_cur_cvx_def_key_pts = batched_index_select(cur_bsz_cur_cvx_def_key_pts, indices=dists_cur_bsz_keypts_pc_minn_idx, dim=1) ## 1 x nn_keypts x 3 --> 
            cur_bsz_cur_cvx_key_pts_offset = batched_index_select(cur_bsz_cur_cvx_key_pts_offset, indices=dists_cur_bsz_keypts_pc_minn_idx, dim=1)
            
            # print(f"dists_cur_bsz_keypts_pc: {dists_cur_bsz_keypts_pc.size()}, dists_cur_bsz_keypts_pc_minn_idx: {dists_cur_bsz_keypts_pc_minn_idx.size()}, cur_bsz_cur_cvx_def_key_pts: {cur_bsz_cur_cvx_def_key_pts.size()}")
            cur_bsz_cur_cvx_def_pc = torch.bmm(cur_bsz_cur_cvx_w_pc, cur_bsz_cur_cvx_def_key_pts)
            
            cd_loss = chamfer_distance(cur_bsz_cur_cvx_def_pc, cur_bsz_tar_pc)
            tot_cd_loss += cd_loss
            # tot_keypts_basis_offset = torch.zeros_like(key_pts) ## bsz x nn_keypts x 3
            # 
            # tot_keypts_basis_offset: bsz x nn_keypts x 3
            tot_keypts_basis_offset[i_bsz, cur_bsz_keypts_idx, :] += cur_bsz_cur_cvx_key_pts_offset[0] 
            tot_keypts_add_nn[i_bsz, cur_bsz_keypts_idx] += 1
            
            ## cd_loss = chamfer_distance(def_pc, tar_pc)
            
        #     cur_cvx_def_pc_pts = []
        #   cur_cvx_def_keypts_pts = []
            cur_cvx_def_pc_pts.append(cur_bsz_cur_cvx_def_pc.detach().cpu().numpy())
            cur_cvx_def_keypts_pts.append(cur_bsz_cur_cvx_def_key_pts.detach().cpu().numpy())
            
            
            
            
            for i_s in range(len(tot_rnd_tot_keypts_basis_offset)):
            
                ratio = torch.rand((1, self.num_basis)).cuda()
                sample_coef = (ratio * coef_range[:, :, 0] + (1 - ratio)
                            * coef_range[:, :, 1]).view(1, 1, self.num_basis)
                sample_def_key_pts_offset =  \
                    torch.bmm(sample_coef, basis).view(1, cur_bsz_pc_nn, 3)
                sample_def_key_pts_offset = batched_index_select(sample_def_key_pts_offset, indices=dists_cur_bsz_keypts_pc_minn_idx, dim=1)
                tot_rnd_tot_keypts_basis_offset[i_s][i_bsz, cur_bsz_keypts_idx, :] += sample_def_key_pts_offset[0] 
                tot_rnd_tot_keypts_add_nn[i_s][i_bsz, cur_bsz_keypts_idx] += 1

        
          src_cvx_pc_pts.append(cur_src_cvx_pc_pts)
          tar_cvx_pc_pts.append(cur_tar_cvx_pc_pts)
          
          src_cvx_def_pc_pts.append(cur_cvx_def_pc_pts)
          src_cvx_def_keypts_pts.append(cur_cvx_def_keypts_pts)
            

        tot_rnd_def_key_pts = []
        tot_rnd_def_pc = []
        for i_s in range(len(tot_rnd_tot_keypts_basis_offset)):
            rnd_def_key_pts = key_pts + tot_rnd_tot_keypts_basis_offset[i_s] / torch.clamp(tot_rnd_tot_keypts_add_nn[i_s].unsqueeze(-1), min=1e-6)
            tot_rnd_def_key_pts.append(rnd_def_key_pts)
            rnd_def_pc = torch.bmm(w_pc, rnd_def_key_pts)
            tot_rnd_def_pc.append(rnd_def_pc)
        
        def_key_pts = key_pts + tot_keypts_basis_offset / torch.clamp(tot_keypts_add_nn.unsqueeze(-1), min=1e-6)
        def_pc = torch.bmm(w_pc, def_key_pts)
        # cd_loss = chamfer_distance(def_pc, tar_pc)
        
        cd_loss = tot_cd_loss / nn_cvx
        
        rt_dict = {
          "cd_loss": cd_loss, 
          "def_key_pts": def_key_pts, 
          "def_pc": def_pc,
          "tot_rnd_def_key_pts": tot_rnd_def_key_pts,
          "tot_rnd_def_pc": tot_rnd_def_pc,
          "src_cvx_pc_pts": src_cvx_pc_pts,
          "tar_cvx_pc_pts": tar_cvx_pc_pts,
          "src_cvx_def_pc_pts": src_cvx_def_pc_pts,
          "src_cvx_def_keypts_pts": src_cvx_def_keypts_pts
        }
        
        return rt_dict
    
    
    
    
    def forward3(self, src_pc, tar_pc, key_pts, dst_key_pts, w_pc,  src_verts, src_convex_pts, tar_verts, dst_convex_pts, glb_net=None):
        ### src_pc, tar_pc;
        ### src_convex_pts: 
        ### src_convex_pts ---- only use convexes with the same idx in those two shapes ###
        ### src_convex_pts: bsz x n_cvxes x n_pts_per_convex  ### forward3
        ### src_keypts: bsz x n_keypts x 3
        #### B, N, _ = src
        #### add src_faces, tar_faces for src_meshes and tar_meshes #### 
        #### network target driven ####
        B, N, _ = src_pc.shape
        nn_cvx = src_convex_pts.size(1) ### bsz x nn_cvx 
        
        src_convex_pts = src_convex_pts * 2.
        dst_convex_pts = dst_convex_pts * 2.
        
        # w_pc: bsz x n_pc x n_keypts
        # print(f"w_pc: {w_pc.size()}, w_pc_sum: {torch.sum(w_pc, dim=-1)[0, :2]}")
        # print(f"maxx_src: {torch.max(src_pc)}, minn_src: {torch.min(src_pc)}, maxx src_convex_pts: {torch.max(src_convex_pts)}, minn src_convex_pts: {torch.min(src_convex_pts)}, ")
        
        ###  src ### 
        avg_src_pc_cvx = self.get_minn_distances_base_pc_cvx_pts(src_pc, src_convex_pts) ### 
        ### bsz x nn_pts x nn_cvx --> []
        src_pc_cvx_indicator = self.get_base_pc_cvx_indicator(src_pc, src_convex_pts, avg_src_pc_cvx) ### 
        src_keypts_cvx_indicator = self.get_base_pc_cvx_indicator(key_pts, src_convex_pts, avg_src_pc_cvx) ### 
        
        avg_dst_pc_cvx = self.get_minn_distances_base_pc_cvx_pts(tar_pc, dst_convex_pts)
        tar_pc_cvx_indicator = self.get_base_pc_cvx_indicator(tar_pc, dst_convex_pts, avg_dst_pc_cvx) ### base-pc-cvx
        
        # tot_keypts_basis_offset = torch.zeros(()) ### 
        
        tot_basis = []
        tot_coef_ranges = []
        tot_coefs = []
        
        
        
        tot_keypts_basis_offset = torch.zeros_like(key_pts) ## bsz x nn_keypts x 3
        
        tot_keypts_add_nn = torch.zeros_like(key_pts)[..., 0] ## bsz x nn_keypts x 3
        
        tot_rnd_tot_keypts_basis_offset = [ torch.zeros_like(key_pts) for _ in range(10)]
        tot_rnd_tot_keypts_add_nn = [ torch.zeros_like(key_pts)[..., 0]  for _ in range(10)]
        
        tot_cd_loss = 0.
        
        # maxx_src_pc_nn = 0
        
        minn_keypts_nn = 255
        
        # minn_keypts_nn = 128
        
        minn_keypts_nn = 64
        
        minn_keypts_nn = 32
        minn_keypts_nn = 22
        minn_keypts_nn = 3
        
        sum_w_weight_thres = 3e-1
        
        src_cvx_pc_pts = []
        tar_cvx_pc_pts = []
        src_cvx_def_pc_pts = []
        src_cvx_def_keypts_pts = []
        
        cvx_nn_to_loss = {}
        
        # print(f"B: {B}, nn_cvx: {nn_cvx}")
        for i_cvx in range(nn_cvx):  ### nn_cvxes 
          ### get the keypts offset ##
          cur_src_pc_cvx_indicator = src_pc_cvx_indicator[:, :, i_cvx] ### bsz x nn_pts 
          cur_src_keypts_cvx_indicator = src_keypts_cvx_indicator[:, :, i_cvx] ## bsz x nn_keypts 
          cur_tar_pc_cvx_indicator = tar_pc_cvx_indicator[:, :, i_cvx]
          
          cur_pc_nn = torch.sum(cur_src_pc_cvx_indicator, dim=-1).long() ## bsz x nn_pc
          
          
          #### 
          pc_idx_argsort = torch.argsort(cur_src_pc_cvx_indicator, dim=-1, descending=True) ### bsz x 
          cur_keypts_nn = torch.sum(cur_src_keypts_cvx_indicator, dim=-1).long()
          keypts_idx_argsort = torch.argsort(cur_src_keypts_cvx_indicator, dim=-1, descending=True)
          
          cur_tar_pc_nn = torch.sum(cur_tar_pc_cvx_indicator, dim=-1).long() ## bsz
          tar_pc_idx_argsort = torch.argsort(cur_tar_pc_cvx_indicator, dim=-1, descending=True) ### bsz x 
          cur_src_cvx_pc_pts = []
          cur_tar_cvx_pc_pts = []
          
          cur_cvx_def_pc_pts = []
          cur_cvx_def_keypts_pts = []
          
          cur_cvx_tot_basis = []
          cur_cvx_tot_coef_ranges = []
          cur_cvx_tot_coefs = []
          
          
          
          for i_bsz in range(B):
            cur_bsz_pc_nn = int(cur_pc_nn[i_bsz].item())
            cur_bsz_keypts_nn = int(cur_keypts_nn[i_bsz].item()) ### keypts_nn
            cur_bsz_tar_pc_nn = int(cur_tar_pc_nn[i_bsz].item()) ### 
            
            
            
            # print(f"cur_bsz_pc_nn: {cur_bsz_pc_nn}, cur_bsz_keypts_nn: {cur_bsz_keypts_nn}, cur_bsz_tar_pc_nn: {cur_bsz_tar_pc_nn}")
            if cur_bsz_pc_nn <= 100 or cur_bsz_keypts_nn <= minn_keypts_nn or cur_bsz_tar_pc_nn <= 100:
              continue
            cur_bsz_pc_idx = pc_idx_argsort[i_bsz, :cur_bsz_pc_nn]
            cur_bsz_keypts_idx = keypts_idx_argsort[i_bsz, :cur_bsz_keypts_nn] 
            cur_bsz_tar_pc_idx = tar_pc_idx_argsort[i_bsz, : cur_bsz_tar_pc_nn]
            
            
            cur_keypts_w_pc = w_pc[:, :, cur_bsz_keypts_idx]
            # cur_keypts_w_pc_sum = torch.abs(torch.sum((cur_keypts_w_pc), dim=-1))
            cur_keypts_w_pc_sum = (torch.sum((cur_keypts_w_pc), dim=-1))
            cur_src_pc_cvx_indicator[cur_keypts_w_pc_sum < sum_w_weight_thres] = 0.
            cur_pc_nn = torch.sum(cur_src_pc_cvx_indicator, dim=-1).long() ## bsz x nn_pc
            cur_bsz_pc_nn = int(cur_pc_nn[i_bsz].item())
            if cur_bsz_pc_nn <= 100 :
                continue
            pc_idx_argsort = torch.argsort(cur_src_pc_cvx_indicator, dim=-1, descending=True) ### bsz x 
            cur_bsz_pc_idx = pc_idx_argsort[i_bsz, :cur_bsz_pc_nn]
            
            
            ### w_pc: 
            cur_bsz_cur_cvx_w_pc = w_pc[i_bsz, cur_bsz_pc_idx, :] ## nn_cvx_pc x nn_keypts ##
            cur_bsz_cur_cvx_w_pc = cur_bsz_cur_cvx_w_pc[:, cur_bsz_keypts_idx] ## nn_cvx_pc x nn_cvx_keypts ##
            cur_bsz_cur_cvx_w_pc = cur_bsz_cur_cvx_w_pc.unsqueeze(0) ## 1 x nn_cvx_pc x nn_cvx_keypts
            
            
            ### cur_bsz_cur_cvx_w_pc --> 
            cur_bsz_cur_cvx_w_pc = cur_bsz_cur_cvx_w_pc / torch.sum((cur_bsz_cur_cvx_w_pc), dim=-1, keepdim=True)
            # torch.clamp(torch.sum((cur_bsz_cur_cvx_w_pc), dim=-1, keepdim=True), min=sum_w_weight_thres) ### 1 x nn_cvx_pc x 1
            # cur_bsz_cur_cvx_w_pc[cur_bsz_cur_cvx_w_pc > 1.] = 1.
            cur_bsz_cur_cvx_w_pc2 = cur_bsz_cur_cvx_w_pc.transpose(2, 1).unsqueeze(2) 
            
            
            # cur_src_cvx_pc_pt
            
            
            cur_bsz_pc = src_pc[i_bsz, cur_bsz_pc_idx, :].unsqueeze(0)
            cur_bsz_keypts = key_pts[i_bsz, cur_bsz_keypts_idx, :].unsqueeze(0)
            cur_bsz_tar_pc = tar_pc[i_bsz, cur_bsz_tar_pc_idx, :].unsqueeze(0) #### tar_pc_idx
            
            
            
           
            
            cur_bsz_pc = cur_bsz_pc - torch.mean(cur_bsz_pc, dim=1, keepdim=True)
            cur_bsz_tar_pc = cur_bsz_tar_pc - torch.mean(cur_bsz_tar_pc, dim=1, keepdim=True)
            cur_bsz_keypts = cur_bsz_keypts - torch.mean(cur_bsz_keypts, dim=1, keepdim=True)
            
            
            # cur_bsz_pc, src_center, src_scale = utils.normalie_pc_bbox_batched(pc=cur_bsz_pc, rt_stats=True)
            # cur_bsz_keypts = (cur_bsz_keypts - src_center) / torch.clamp(src_scale, min=1e-6)
            # cur_bsz_tar_pc = utils.normalie_pc_bbox_batched(cur_bsz_tar_pc, rt_stats=False)
            
            
            
            cur_src_cvx_pc_pts.append(cur_bsz_pc.detach().cpu().numpy())
            cur_tar_cvx_pc_pts.append(cur_bsz_tar_pc.detach().cpu().numpy())
            
            cur_bsz_keypts1 = cur_bsz_keypts.unsqueeze(-1).expand(-1, -1, -1, cur_bsz_pc_nn)
            
            cur_bsz_pc1 = cur_bsz_pc.unsqueeze(-1).expand(-1, -1, -1, cur_bsz_pc_nn)
            
            
            ''' pointnet '''
            # src_feats, src_glb_feats = self.pointnet(cur_bsz_pc, return_global=False) ## bsz x 
            # ### 
            # tar_feats, tar_glb_feats = self.tar_pointnet(cur_bsz_tar_pc, return_global=False) ### tar_feats 
            ''' pointnet '''
            
            
            
            ''' pvcnn '''
            # fused_features, voxel_coords = self.pvconv((cur_bsz_pc.contiguous().transpose(-1, -2).contiguous(), cur_bsz_pc.contiguous().transpose(-1, -2).contiguous())) ### contiguous() transpose() view()  ### obj
            # ## fused_features: b x dim x n_pts 
            
            fused_features, _ = self.pn_enc(cur_bsz_pc)
            
            
            
            
            src_feats = fused_features
            
            
            src_glb_feats = src_feats.max(dim=-1)[0].unsqueeze(-1) ## bsz x dim x 1
            
            
            src_feats = torch.cat([src_feats, src_glb_feats.expand(-1, -1, src_feats.size(-1))], dim=1)
            src_feats = self.src_ppfeats_out_net(src_feats) ### for global features 

            src_glb_feats = self.src_glb_feat_out_net(src_glb_feats).squeeze(-1) #### src_glb_feats


            # tar_fused_features, voxel_coords = self.tar_pvconv((cur_bsz_tar_pc.contiguous().transpose(-1, -2).contiguous(), cur_bsz_tar_pc.contiguous().transpose(-1, -2).contiguous())) ### contiguous() transpose() view()  ### obj
            # ## fused_features: b x dim x n_pts 
            
            tar_fused_features, _ = self.tar_pn_enc(cur_bsz_tar_pc)
            tar_feats = tar_fused_features
            
            
            tar_glb_feats = tar_feats.max(dim=-1)[0].unsqueeze(-1) ## bsz x dim x 1

            tar_glb_feats = self.tar_glb_feat_out_net(tar_glb_feats).squeeze(-1)      
            
            # print(f"src_feats: {src_feats.size()}, tar_glb_feats: {tar_glb_feats.size()}, src_glb_feats: {src_glb_feats.size()}")
            
            
            src_feats = F.relu(self.bn11(self.conv11(src_feats)))
            src_feats = F.relu(self.bn12(self.conv12(src_feats)))
            src_feats = F.relu(self.bn13(self.conv13(src_feats)))
        ### 
            
            ### 
            src_feats = src_feats.unsqueeze(1).expand(-1, cur_bsz_keypts_nn, -1, -1) 
            
            
            # cur_bsz_keypts_feats = torch.cat(
            #   [src_feats, cur_bsz_cur_cvx_w_pc2, cur_bsz_keypts1], dim=2
            # ).view(1 * cur_bsz_keypts_nn, -1, cur_bsz_pc_nn)
            
            
            cur_bsz_src_pc_feats = torch.cat( ### 
              [src_feats, cur_bsz_keypts1], dim=2
            ).view(1 * cur_bsz_keypts_nn, -1, cur_bsz_pc_nn)
            
            # print(f"src_feats: {src_feats.size()}, cur_bsz_cur_cvx_w_pc: {cur_bsz_cur_cvx_w_pc.size()}, cur_bsz_keypts1: {cur_bsz_keypts1.size()}")
            
            '''  '''
            # cur_bsz_src_pc_feats = F.relu(self.bn21(self.conv21(cur_bsz_src_pc_feats)))
            # cur_bsz_src_pc_feats = self.bn22(self.conv22(cur_bsz_src_pc_feats))
            
            cur_bsz_src_pc_feats = F.relu((self.conv21(cur_bsz_src_pc_feats)))
            cur_bsz_src_pc_feats = (self.conv22(cur_bsz_src_pc_feats))
            
            ### cur_bsz_keypts_feats: bsz x nn_feats
            cur_bsz_src_pc_feats = torch.max(cur_bsz_src_pc_feats, 2, keepdim=True)[0] 
            key_fea = cur_bsz_src_pc_feats.view(1 * cur_bsz_keypts_nn, -1, 1) ### 
            net = torch.cat([key_fea, cur_bsz_keypts.view(1 * cur_bsz_keypts_nn, 3, 1)], 1) 
            
            
            net = F.relu(self.bn31(self.conv31(net))) #### net
            net = F.relu(self.bn32(self.conv32(net)))
            basis = self.conv33(net).view(1, cur_bsz_keypts_nn * 3, self.num_basis).transpose(1, 2)
            basis = basis / basis.norm(p=2, dim=-1, keepdim=True)  ### basis norm 
            
            key_fea_range = key_fea.view(
                1, cur_bsz_keypts_nn, -1, 1).expand(-1, -1, -1, self.num_basis).transpose(1, 3)
            key_pts_range = cur_bsz_keypts.view(
                1, cur_bsz_keypts_nn, 3, 1).expand(-1, -1, -1, self.num_basis).transpose(1, 3)
            basis_range = basis.view(1, self.num_basis, cur_bsz_keypts_nn, 3).transpose(2, 3)
            
            
            
            ### get range
            coef_range = torch.cat([key_fea_range, key_pts_range, basis_range], 2).view(
                1 * self.num_basis, -1, cur_bsz_keypts_nn)
            coef_range = F.relu(self.bn71(self.conv71(coef_range)))
            coef_range = F.relu(self.bn72(self.conv72(coef_range)))
            coef_range = self.conv73(coef_range)
            coef_range = torch.max(coef_range, 2, keepdim=True)[0]
            # coef_range = coef_range.view(B, self.num_basis, 2) * 0.02 ### 
            # coef_range = coef_range.view(B, self.num_basis, 2) * 0.1 
            # coef_range = coef_range.view(1, self.num_basis, 2) * 0.2
            # coef_range = coef_range.view(1, self.num_basis, 2) * 0.1
            # coef_range = coef_range.view(1, self.num_basis, 2) * 0.05
            # coef_range = coef_range.view(B, self.num_basis, 2) * 0.2 # coef_multiplier
            coef_range = coef_range.view(1, self.num_basis, 2) * self.coef_multiplier ### coef-multiplier
            coef_range[:, :, 0] = coef_range[:, :, 0] * -1

            src_tar = torch.cat([src_glb_feats, tar_glb_feats], 1).unsqueeze(
                1).expand(-1, cur_bsz_keypts_nn, -1).reshape(1 * cur_bsz_keypts_nn, -1, 1)     
            
            
            
            ### to the source shape ###
            ### keypoint features, srouce target fused features, and keypoint coordiantes ###
            ### key_feature; source_target
            key_fea = torch.cat([key_fea, src_tar, cur_bsz_keypts.view(1 * cur_bsz_keypts_nn, 3, 1)], 1)
            # print(f"key_fea: {key_fea.size()}")
            key_fea = F.relu(self.bn41(self.conv41(key_fea))) ### key_fea
            key_fea = F.relu(self.bn42(self.conv42(key_fea))) ### key_fea
            key_fea = F.relu(self.bn43(self.conv43(key_fea))) ### 

            key_fea = key_fea.view(1, cur_bsz_keypts_nn, 128).transpose(
                1, 2).unsqueeze(1).expand(-1, self.num_basis, -1, -1)
            cur_bsz_keypts2 = cur_bsz_keypts.view(1, cur_bsz_keypts_nn, 3).transpose(  ### key_pts2;
                1, 2).unsqueeze(1).expand(-1, self.num_basis, -1, -1)
            basis1 = basis.view(1, self.num_basis, cur_bsz_keypts_nn, 3).transpose(2, 3)

            net = torch.cat([key_fea, basis1, cur_bsz_keypts2], 2).view( ### keypoint features, basis, coefs, --- global feautre -> fo rcoefs 
                1 * self.num_basis, -1, cur_bsz_keypts_nn)

            net = F.relu(self.bn51(self.conv51(net)))
            net = F.relu(self.bn52(self.conv52(net)))
            net = self.bn53(self.conv53(net))

            net = torch.max(net, 2, keepdim=True)[0]
            net = net.view(1 * self.num_basis, -1, 1)

            net = torch.cat([net, coef_range.view(1 * self.num_basis, 2, 1)], 1)
            net = F.relu(self.bn61(self.conv61(net)))
            net = F.relu(self.bn62(self.conv62(net)))
            coef = self.sigmoid(self.conv63(net)).view(1, self.num_basis)

            coef = (coef * coef_range[:, :, 0] + (1 - coef)
                    * coef_range[:, :, 1]).view(1, 1, self.num_basis)
            
            cur_cvx_tot_basis.append(basis) ### 1 x n_basis x (n_keypts x 3)
            cur_cvx_tot_coef_ranges.append(coef_range)
            
            cur_cvx_tot_coefs.append(coef)
            
            cur_bsz_cur_cvx_key_pts_offset = torch.bmm(coef, basis).view(1, cur_bsz_keypts_nn, 3)
            ### def_key_pts ###
            
            
            cur_bsz_cur_cvx_def_key_pts = cur_bsz_keypts + cur_bsz_cur_cvx_key_pts_offset
            
            
            # dists_cur_bsz_keypts_pc = torch.sum((cur_bsz_keypts.unsqueeze(-2) - cur_bsz_pc.unsqueeze(1)) ** 2, dim=-1) 
            # dists_cur_bsz_keypts_pc, dists_cur_bsz_keypts_pc_minn_idx = torch.min(dists_cur_bsz_keypts_pc, dim=-1) ## 1 x nn_keypts
            # cur_bsz_cur_cvx_def_key_pts = batched_index_select(cur_bsz_cur_cvx_def_key_pts, indices=dists_cur_bsz_keypts_pc_minn_idx, dim=1) ## 1 x nn_keypts x 3 --> 
            # cur_bsz_cur_cvx_key_pts_offset = batched_index_select(cur_bsz_cur_cvx_key_pts_offset, indices=dists_cur_bsz_keypts_pc_minn_idx, dim=1)
            
            # print(f"dists_cur_bsz_keypts_pc: {dists_cur_bsz_keypts_pc.size()}, dists_cur_bsz_keypts_pc_minn_idx: {dists_cur_bsz_keypts_pc_minn_idx.size()}, cur_bsz_cur_cvx_def_key_pts: {cur_bsz_cur_cvx_def_key_pts.size()}")
            cur_bsz_cur_cvx_def_pc = torch.bmm(cur_bsz_cur_cvx_w_pc, cur_bsz_cur_cvx_def_key_pts)
            
            cd_loss = chamfer_distance(cur_bsz_cur_cvx_def_pc, cur_bsz_tar_pc)
            tot_cd_loss += cd_loss
            
            cvx_nn_to_loss[cur_bsz_pc_nn] = cd_loss
            
            # tot_keypts_basis_offset = torch.zeros_like(key_pts) ## bsz x nn_keypts x 3
            # 
            
            # if cur_bsz_pc_nn > maxx_src_pc_nn:
            #     maxx_src_pc_nn = cur_bsz_pc_nn
            #     tot_cd_loss = cd_loss
            
            
            # tot_keypts_basis_offset: bsz x nn_keypts x 3
            tot_keypts_basis_offset[i_bsz, cur_bsz_keypts_idx, :] += cur_bsz_cur_cvx_key_pts_offset[0]  # * src_scale[0]
            tot_keypts_add_nn[i_bsz, cur_bsz_keypts_idx] += 1
            
            # (cur_bsz_keypts - src_center) / torch.clamp(src_scale, min=1e-6)
            
            ## cd_loss = chamfer_distance(def_pc, tar_pc)
            
        #     cur_cvx_def_pc_pts = []
        #   cur_cvx_def_keypts_pts = []
            cur_cvx_def_pc_pts.append(cur_bsz_cur_cvx_def_pc.detach().cpu().numpy())
            cur_cvx_def_keypts_pts.append(cur_bsz_cur_cvx_def_key_pts.detach().cpu().numpy())
            
            
            
            
            for i_s in range(len(tot_rnd_tot_keypts_basis_offset)):
            
                ratio = torch.rand((1, self.num_basis)).cuda()
                sample_coef = (ratio * coef_range[:, :, 0] + (1 - ratio)
                            * coef_range[:, :, 1]).view(1, 1, self.num_basis)
                sample_def_key_pts_offset =  \
                    torch.bmm(sample_coef, basis).view(1, cur_bsz_keypts_nn, 3)
                # sample_def_key_pts_offset = batched_index_select(sample_def_key_pts_offset, indices=dists_cur_bsz_keypts_pc_minn_idx, dim=1)
                tot_rnd_tot_keypts_basis_offset[i_s][i_bsz, cur_bsz_keypts_idx, :] += sample_def_key_pts_offset[0] #  * src_scale[0]
                tot_rnd_tot_keypts_add_nn[i_s][i_bsz, cur_bsz_keypts_idx] += 1

        
          src_cvx_pc_pts.append(cur_src_cvx_pc_pts)
          tar_cvx_pc_pts.append(cur_tar_cvx_pc_pts)
          
          src_cvx_def_pc_pts.append(cur_cvx_def_pc_pts)
          src_cvx_def_keypts_pts.append(cur_cvx_def_keypts_pts)
          
          tot_basis.append(cur_cvx_tot_basis)
          tot_coef_ranges.append(cur_cvx_tot_coef_ranges)
          tot_coefs.append(cur_cvx_tot_coefs)
            

        tot_rnd_def_key_pts = []
        tot_rnd_def_pc = []
        
        if len(cvx_nn_to_loss) > 0:
            sorted_cvx_nn_to_loss = sorted(cvx_nn_to_loss.items(), key=lambda ii: ii[0], reverse=True)
            
            sorted_cvx_nn_to_loss = [ii[1] for ii in sorted_cvx_nn_to_loss[:7]]
            
            cd_loss = sum(sorted_cvx_nn_to_loss) / float(len(sorted_cvx_nn_to_loss))
        else:
            cd_loss  = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            
        
        for i_s in range(len(tot_rnd_tot_keypts_basis_offset)):
            rnd_def_key_pts = key_pts + tot_rnd_tot_keypts_basis_offset[i_s] / torch.clamp(tot_rnd_tot_keypts_add_nn[i_s].unsqueeze(-1), min=1.)
            tot_rnd_def_key_pts.append(rnd_def_key_pts)
            rnd_def_pc = torch.bmm(w_pc, rnd_def_key_pts)
            tot_rnd_def_pc.append(rnd_def_pc)
        
        def_key_pts = key_pts + tot_keypts_basis_offset / torch.clamp(tot_keypts_add_nn.unsqueeze(-1), min=1.)
        def_pc = torch.bmm(w_pc, def_key_pts)
        # cd_loss = chamfer_distance(def_pc, tar_pc)
        
        # cd_loss = tot_cd_loss / nn_cvx
        if len(tot_basis[0]) > 0:
            basis_np = torch.cat(tot_basis[0], dim=0).detach().cpu().numpy()
            np.save("tot_basis_fr3.npy", basis_np)
        
        
        
        if glb_net is not None:
            # glb_def_key_pts, glb_def_pc, glb_cd_loss, glb_basis, glb_coef, glb_sample_def_key_pts, glb_sym_loss,  glb_coef_range, tot_sampled_def_keypts = glb_net(
            #     def_pc, tar_pc, def_key_pts, w_pc
            # )
            
            cd_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            
            glb_def_key_pts, glb_def_pc, glb_cd_loss, glb_basis, glb_coef, glb_sample_def_key_pts, glb_sym_loss,  glb_coef_range, tot_sampled_def_keypts = glb_net(
                src_pc, tar_pc, key_pts, w_pc ### w_pc and others...
            )
        
            rt_dict = {
                "cd_loss": cd_loss, 
                "def_key_pts": def_key_pts, 
                "def_pc": def_pc,
                "tot_rnd_def_key_pts": tot_rnd_def_key_pts,
                "tot_rnd_def_pc": tot_rnd_def_pc,
                "src_cvx_pc_pts": src_cvx_pc_pts,
                "tar_cvx_pc_pts": tar_cvx_pc_pts,
                "src_cvx_def_pc_pts": src_cvx_def_pc_pts,
                "src_cvx_def_keypts_pts": src_cvx_def_keypts_pts, 
                "tot_basis": tot_basis, 
                "tot_coefs": tot_coefs,
                "tot_coef_ranges": tot_coef_ranges,
                "glb_def_key_pts": glb_def_key_pts,
                "glb_def_pc": glb_def_pc,
                "glb_cd_loss": glb_cd_loss, 
                "glb_basis": glb_basis, 
                "glb_coef": glb_coef, 
                "glb_coef_range": glb_coef_range,
                #   "tot_sampled_def_keypts": tot_sampled_def_keypts
            }
        else:
            rt_dict = {
                "cd_loss": cd_loss, 
                "def_key_pts": def_key_pts, 
                "def_pc": def_pc,
                "tot_rnd_def_key_pts": tot_rnd_def_key_pts,
                "tot_rnd_def_pc": tot_rnd_def_pc,
                "src_cvx_pc_pts": src_cvx_pc_pts,
                "tar_cvx_pc_pts": tar_cvx_pc_pts,
                "src_cvx_def_pc_pts": src_cvx_def_pc_pts,
                "src_cvx_def_keypts_pts": src_cvx_def_keypts_pts, 
                "tot_basis": tot_basis, 
                "tot_coefs": tot_coefs,
                "tot_coef_ranges": tot_coef_ranges,
            }
            
        
        return rt_dict
   
   
     
    def sample3(self, src_pc, tar_pc, key_pts, dst_key_pts, w_pc,  src_verts, src_convex_pts, tar_verts, dst_convex_pts, glb_net=None):
        ### src_pc, tar_pc;
        ### src_convex_pts: 
        ### src_convex_pts ---- only use convexes with the same idx in those two shapes ###
        ### src_convex_pts: bsz x n_cvxes x n_pts_per_convex  ### forward3
        ### src_keypts: bsz x n_keypts x 3
        #### B, N, _ = src
        #### add src_faces, tar_faces for src_meshes and tar_meshes #### 
        #### network target driven ####
        B, N, _ = src_pc.shape
        nn_cvx = src_convex_pts.size(1) ### bsz x nn_cvx 
        
        src_convex_pts = src_convex_pts * 2.
        dst_convex_pts = dst_convex_pts * 2.
        
        # w_pc: bsz x n_pc x n_keypts
        # print(f"w_pc: {w_pc.size()}, w_pc_sum: {torch.sum(w_pc, dim=-1)[0, :2]}")
        # print(f"maxx_src: {torch.max(src_pc)}, minn_src: {torch.min(src_pc)}, maxx src_convex_pts: {torch.max(src_convex_pts)}, minn src_convex_pts: {torch.min(src_convex_pts)}, ")
        
        ###  src ### 
        avg_src_pc_cvx = self.get_minn_distances_base_pc_cvx_pts(src_pc, src_convex_pts) ### 
        ### bsz x nn_pts x nn_cvx --> []
        src_pc_cvx_indicator = self.get_base_pc_cvx_indicator(src_pc, src_convex_pts, avg_src_pc_cvx) ### 
        src_keypts_cvx_indicator = self.get_base_pc_cvx_indicator(key_pts, src_convex_pts, avg_src_pc_cvx) ### 
        
        avg_dst_pc_cvx = self.get_minn_distances_base_pc_cvx_pts(tar_pc, dst_convex_pts) ### dst_convex_pts...
        tar_pc_cvx_indicator = self.get_base_pc_cvx_indicator(tar_pc, dst_convex_pts, avg_dst_pc_cvx) ### base-pc-cvx
        
        # tot_keypts_basis_offset = torch.zeros(()) ### 
        
        tot_basis = []
        tot_coef_ranges = []
        
        
        
        tot_keypts_basis_offset = torch.zeros_like(key_pts) ## bsz x nn_keypts x 3
        
        tot_keypts_add_nn = torch.zeros_like(key_pts)[..., 0] ## bsz x nn_keypts x 3
        
        tot_rnd_tot_keypts_basis_offset = [ torch.zeros_like(key_pts) for _ in range(10)]
        tot_rnd_tot_keypts_add_nn = [torch.zeros_like(key_pts)[..., 0]  for _ in range(10)]
        
        tot_cd_loss = 0.
        
        # maxx_src_pc_nn = 0
        
        minn_keypts_nn = 255
        
        # minn_keypts_nn = 128
        
        minn_keypts_nn = 64
        minn_keypts_nn = 32
        
        sum_w_weight_thres = 3e-1
        
        src_cvx_pc_pts = []
        tar_cvx_pc_pts = []
        src_cvx_def_pc_pts = []
        src_cvx_def_keypts_pts = []
        
        cvx_nn_to_loss = {}
        
        # print(f"B: {B}, nn_cvx: {nn_cvx}")
        for i_cvx in range(nn_cvx):  ### nn_cvxes 
          ### get the keypts offset ##
          cur_src_pc_cvx_indicator = src_pc_cvx_indicator[:, :, i_cvx] ### bsz x nn_pts 
          cur_src_keypts_cvx_indicator = src_keypts_cvx_indicator[:, :, i_cvx] ## bsz x nn_keypts 
          cur_tar_pc_cvx_indicator = tar_pc_cvx_indicator[:, :, i_cvx]
          
          cur_pc_nn = torch.sum(cur_src_pc_cvx_indicator, dim=-1).long() ## bsz x nn_pc
          
          
          
          pc_idx_argsort = torch.argsort(cur_src_pc_cvx_indicator, dim=-1, descending=True) ### bsz x 
          cur_keypts_nn = torch.sum(cur_src_keypts_cvx_indicator, dim=-1).long()
          keypts_idx_argsort = torch.argsort(cur_src_keypts_cvx_indicator, dim=-1, descending=True)
          
          cur_tar_pc_nn = torch.sum(cur_tar_pc_cvx_indicator, dim=-1).long() ## bsz
          tar_pc_idx_argsort = torch.argsort(cur_tar_pc_cvx_indicator, dim=-1, descending=True) ### bsz x ; tar_pc_indicators 
          cur_src_cvx_pc_pts = []
          cur_tar_cvx_pc_pts = []
          
          cur_cvx_def_pc_pts = []
          cur_cvx_def_keypts_pts = []
          
          cur_cvx_tot_basis = []
          cur_cvx_tot_coef_ranges = []
          
          
          
          for i_bsz in range(B):
            cur_bsz_pc_nn = int(cur_pc_nn[i_bsz].item())
            cur_bsz_keypts_nn = int(cur_keypts_nn[i_bsz].item()) ### keypts_nn
            cur_bsz_tar_pc_nn = int(cur_tar_pc_nn[i_bsz].item()) ### 
            
            
            
            # print(f"cur_bsz_pc_nn: {cur_bsz_pc_nn}, cur_bsz_keypts_nn: {cur_bsz_keypts_nn}, cur_bsz_tar_pc_nn: {cur_bsz_tar_pc_nn}")
            if cur_bsz_pc_nn <= 100 or cur_bsz_keypts_nn <= minn_keypts_nn: ### no demand on tar_pc ###
              continue
            cur_bsz_pc_idx = pc_idx_argsort[i_bsz, :cur_bsz_pc_nn]
            cur_bsz_keypts_idx = keypts_idx_argsort[i_bsz, :cur_bsz_keypts_nn] 
            cur_bsz_tar_pc_idx = tar_pc_idx_argsort[i_bsz, : cur_bsz_tar_pc_nn]
            
            
            cur_keypts_w_pc = w_pc[:, :, cur_bsz_keypts_idx]
            # cur_keypts_w_pc_sum = torch.abs(torch.sum((cur_keypts_w_pc), dim=-1))
            cur_keypts_w_pc_sum = (torch.sum((cur_keypts_w_pc), dim=-1))
            cur_src_pc_cvx_indicator[cur_keypts_w_pc_sum < sum_w_weight_thres] = 0.
            cur_pc_nn = torch.sum(cur_src_pc_cvx_indicator, dim=-1).long() ## bsz x nn_pc
            cur_bsz_pc_nn = int(cur_pc_nn[i_bsz].item())
            if cur_bsz_pc_nn <= 100 :
                continue
            pc_idx_argsort = torch.argsort(cur_src_pc_cvx_indicator, dim=-1, descending=True) ### bsz x 
            cur_bsz_pc_idx = pc_idx_argsort[i_bsz, :cur_bsz_pc_nn]
            
            
            ### w_pc: 
            cur_bsz_cur_cvx_w_pc = w_pc[i_bsz, cur_bsz_pc_idx, :] ## nn_cvx_pc x nn_keypts ##
            cur_bsz_cur_cvx_w_pc = cur_bsz_cur_cvx_w_pc[:, cur_bsz_keypts_idx] ## nn_cvx_pc x nn_cvx_keypts ##
            cur_bsz_cur_cvx_w_pc = cur_bsz_cur_cvx_w_pc.unsqueeze(0) ## 1 x nn_cvx_pc x nn_cvx_keypts
            
            
            ### cur_bsz_cur_cvx_w_pc --> 
            cur_bsz_cur_cvx_w_pc = cur_bsz_cur_cvx_w_pc / torch.sum((cur_bsz_cur_cvx_w_pc), dim=-1, keepdim=True)
            # torch.clamp(torch.sum((cur_bsz_cur_cvx_w_pc), dim=-1, keepdim=True), min=sum_w_weight_thres) ### 1 x nn_cvx_pc x 1
            # cur_bsz_cur_cvx_w_pc[cur_bsz_cur_cvx_w_pc > 1.] = 1.
            cur_bsz_cur_cvx_w_pc2 = cur_bsz_cur_cvx_w_pc.transpose(2, 1).unsqueeze(2) 
            
            
            # cur_src_cvx_pc_pt
            
            
            cur_bsz_pc = src_pc[i_bsz, cur_bsz_pc_idx, :].unsqueeze(0)
            cur_bsz_keypts = key_pts[i_bsz, cur_bsz_keypts_idx, :].unsqueeze(0)
            cur_bsz_tar_pc = tar_pc[i_bsz, cur_bsz_tar_pc_idx, :].unsqueeze(0) #### tar_pc_idx
            
            
            
           
            
            cur_bsz_pc = cur_bsz_pc - torch.mean(cur_bsz_pc, dim=1, keepdim=True)
            cur_bsz_tar_pc = cur_bsz_tar_pc - torch.mean(cur_bsz_tar_pc, dim=1, keepdim=True)
            cur_bsz_keypts = cur_bsz_keypts - torch.mean(cur_bsz_keypts, dim=1, keepdim=True)
            
            
            # cur_bsz_pc, src_center, src_scale = utils.normalie_pc_bbox_batched(pc=cur_bsz_pc, rt_stats=True)
            # cur_bsz_keypts = (cur_bsz_keypts - src_center) / torch.clamp(src_scale, min=1e-6)
            # cur_bsz_tar_pc = utils.normalie_pc_bbox_batched(cur_bsz_tar_pc, rt_stats=False)
            
            
            
            cur_src_cvx_pc_pts.append(cur_bsz_pc.detach().cpu().numpy())
            cur_tar_cvx_pc_pts.append(cur_bsz_tar_pc.detach().cpu().numpy())
            
            cur_bsz_keypts1 = cur_bsz_keypts.unsqueeze(-1).expand(-1, -1, -1, cur_bsz_pc_nn)
            
            cur_bsz_pc1 = cur_bsz_pc.unsqueeze(-1).expand(-1, -1, -1, cur_bsz_pc_nn)
            
            
            ''' pointnet '''
            # src_feats, src_glb_feats = self.pointnet(cur_bsz_pc, return_global=False) ## bsz x 
            # ### 
            # tar_feats, tar_glb_feats = self.tar_pointnet(cur_bsz_tar_pc, return_global=False) ### tar_feats 
            ''' pointnet '''
            
            
            
            ''' pvcnn '''
            # fused_features, voxel_coords = self.pvconv((cur_bsz_pc.contiguous().transpose(-1, -2).contiguous(), cur_bsz_pc.contiguous().transpose(-1, -2).contiguous())) ### contiguous() transpose() view()  ### obj
            # ## fused_features: b x dim x n_pts 
            
            fused_features, _ = self.pn_enc(cur_bsz_pc)
            
            
            
            
            src_feats = fused_features
            
            
            src_glb_feats = src_feats.max(dim=-1)[0].unsqueeze(-1) ## bsz x dim x 1
            
            
            src_feats = torch.cat([src_feats, src_glb_feats.expand(-1, -1, src_feats.size(-1))], dim=1)
            src_feats = self.src_ppfeats_out_net(src_feats) ### for global features 

            src_glb_feats = self.src_glb_feat_out_net(src_glb_feats).squeeze(-1)


            # tar_fused_features, voxel_coords = self.tar_pvconv((cur_bsz_tar_pc.contiguous().transpose(-1, -2).contiguous(), cur_bsz_tar_pc.contiguous().transpose(-1, -2).contiguous())) ### contiguous() transpose() view()  ### obj
            # ## fused_features: b x dim x n_pts 
            
            tar_fused_features, _ = self.tar_pn_enc(cur_bsz_tar_pc)
            tar_feats = tar_fused_features
            
            
            tar_glb_feats = tar_feats.max(dim=-1)[0].unsqueeze(-1) ## bsz x dim x 1

            tar_glb_feats = self.tar_glb_feat_out_net(tar_glb_feats).squeeze(-1)      
            
            # print(f"src_feats: {src_feats.size()}, tar_glb_feats: {tar_glb_feats.size()}, src_glb_feats: {src_glb_feats.size()}")
            
            
            src_feats = F.relu(self.bn11(self.conv11(src_feats)))
            src_feats = F.relu(self.bn12(self.conv12(src_feats)))
            src_feats = F.relu(self.bn13(self.conv13(src_feats)))
        ### 
            
            ### 
            src_feats = src_feats.unsqueeze(1).expand(-1, cur_bsz_keypts_nn, -1, -1) 
            
            
            # cur_bsz_keypts_feats = torch.cat(
            #   [src_feats, cur_bsz_cur_cvx_w_pc2, cur_bsz_keypts1], dim=2
            # ).view(1 * cur_bsz_keypts_nn, -1, cur_bsz_pc_nn)
            
            
            cur_bsz_src_pc_feats = torch.cat( ### 
              [src_feats, cur_bsz_keypts1], dim=2
            ).view(1 * cur_bsz_keypts_nn, -1, cur_bsz_pc_nn)
            
            # print(f"src_feats: {src_feats.size()}, cur_bsz_cur_cvx_w_pc: {cur_bsz_cur_cvx_w_pc.size()}, cur_bsz_keypts1: {cur_bsz_keypts1.size()}")
            
            
            cur_bsz_src_pc_feats = F.relu(self.bn21(self.conv21(cur_bsz_src_pc_feats)))
            cur_bsz_src_pc_feats = self.bn22(self.conv22(cur_bsz_src_pc_feats))
            
            ### cur_bsz_keypts_feats: bsz x nn_feats
            cur_bsz_src_pc_feats = torch.max(cur_bsz_src_pc_feats, 2, keepdim=True)[0] 
            key_fea = cur_bsz_src_pc_feats.view(1 * cur_bsz_keypts_nn, -1, 1) ### 
            net = torch.cat([key_fea, cur_bsz_keypts.view(1 * cur_bsz_keypts_nn, 3, 1)], 1) 
            
            
            net = F.relu(self.bn31(self.conv31(net))) #### net
            net = F.relu(self.bn32(self.conv32(net)))
            basis = self.conv33(net).view(1, cur_bsz_keypts_nn * 3, self.num_basis).transpose(1, 2)
            basis = basis / basis.norm(p=2, dim=-1, keepdim=True)  ### basis norm 
            
            key_fea_range = key_fea.view(
                1, cur_bsz_keypts_nn, -1, 1).expand(-1, -1, -1, self.num_basis).transpose(1, 3)
            key_pts_range = cur_bsz_keypts.view(
                1, cur_bsz_keypts_nn, 3, 1).expand(-1, -1, -1, self.num_basis).transpose(1, 3)
            basis_range = basis.view(1, self.num_basis, cur_bsz_keypts_nn, 3).transpose(2, 3)
            
            
            
            ### get range
            coef_range = torch.cat([key_fea_range, key_pts_range, basis_range], 2).view(
                1 * self.num_basis, -1, cur_bsz_keypts_nn)
            coef_range = F.relu(self.bn71(self.conv71(coef_range)))
            coef_range = F.relu(self.bn72(self.conv72(coef_range)))
            coef_range = self.conv73(coef_range)
            coef_range = torch.max(coef_range, 2, keepdim=True)[0]
            # coef_range = coef_range.view(B, self.num_basis, 2) * 0.02 ### 
            # coef_range = coef_range.view(B, self.num_basis, 2) * 0.1 
            # coef_range = coef_range.view(1, self.num_basis, 2) * 0.2
            # coef_range = coef_range.view(1, self.num_basis, 2) * 0.1
            # coef_range = coef_range.view(1, self.num_basis, 2) * 0.05
            # coef_range = coef_range.view(B, self.num_basis, 2) * 0.2 # coef_multiplier
            coef_range = coef_range.view(1, self.num_basis, 2) * self.coef_multiplier
            coef_range[:, :, 0] = coef_range[:, :, 0] * -1

            src_tar = torch.cat([src_glb_feats, tar_glb_feats], 1).unsqueeze(
                1).expand(-1, cur_bsz_keypts_nn, -1).reshape(1 * cur_bsz_keypts_nn, -1, 1)     
            
            
            
            ### to the source shape ###
            ### keypoint features, srouce target fused features, and keypoint coordiantes ###
            ### key_feature; source_target
            key_fea = torch.cat([key_fea, src_tar, cur_bsz_keypts.view(1 * cur_bsz_keypts_nn, 3, 1)], 1)
            # print(f"key_fea: {key_fea.size()}")
            key_fea = F.relu(self.bn41(self.conv41(key_fea))) ### key_fea
            key_fea = F.relu(self.bn42(self.conv42(key_fea))) ### key_fea
            key_fea = F.relu(self.bn43(self.conv43(key_fea))) ### 

            key_fea = key_fea.view(1, cur_bsz_keypts_nn, 128).transpose(
                1, 2).unsqueeze(1).expand(-1, self.num_basis, -1, -1)
            cur_bsz_keypts2 = cur_bsz_keypts.view(1, cur_bsz_keypts_nn, 3).transpose(  ### key_pts2;
                1, 2).unsqueeze(1).expand(-1, self.num_basis, -1, -1)
            basis1 = basis.view(1, self.num_basis, cur_bsz_keypts_nn, 3).transpose(2, 3)

            net = torch.cat([key_fea, basis1, cur_bsz_keypts2], 2).view(
                1 * self.num_basis, -1, cur_bsz_keypts_nn)

            net = F.relu(self.bn51(self.conv51(net)))
            net = F.relu(self.bn52(self.conv52(net)))
            net = self.bn53(self.conv53(net))

            net = torch.max(net, 2, keepdim=True)[0]
            net = net.view(1 * self.num_basis, -1, 1)

            net = torch.cat([net, coef_range.view(1 * self.num_basis, 2, 1)], 1)
            net = F.relu(self.bn61(self.conv61(net)))
            net = F.relu(self.bn62(self.conv62(net)))
            coef = self.sigmoid(self.conv63(net)).view(1, self.num_basis)

            coef = (coef * coef_range[:, :, 0] + (1 - coef)
                    * coef_range[:, :, 1]).view(1, 1, self.num_basis)
            
            cur_cvx_tot_basis.append(basis) ### 1 x n_basis x (n_keypts x 3)
            cur_cvx_tot_coef_ranges.append(coef_range)
            
            cur_bsz_cur_cvx_key_pts_offset = torch.bmm(coef, basis).view(1, cur_bsz_keypts_nn, 3)
            ### def_key_pts ###
            
            
            cur_bsz_cur_cvx_def_key_pts = cur_bsz_keypts + cur_bsz_cur_cvx_key_pts_offset
            
            
            # dists_cur_bsz_keypts_pc = torch.sum((cur_bsz_keypts.unsqueeze(-2) - cur_bsz_pc.unsqueeze(1)) ** 2, dim=-1) 
            # dists_cur_bsz_keypts_pc, dists_cur_bsz_keypts_pc_minn_idx = torch.min(dists_cur_bsz_keypts_pc, dim=-1) ## 1 x nn_keypts
            # cur_bsz_cur_cvx_def_key_pts = batched_index_select(cur_bsz_cur_cvx_def_key_pts, indices=dists_cur_bsz_keypts_pc_minn_idx, dim=1) ## 1 x nn_keypts x 3 --> 
            # cur_bsz_cur_cvx_key_pts_offset = batched_index_select(cur_bsz_cur_cvx_key_pts_offset, indices=dists_cur_bsz_keypts_pc_minn_idx, dim=1)
            
            # print(f"dists_cur_bsz_keypts_pc: {dists_cur_bsz_keypts_pc.size()}, dists_cur_bsz_keypts_pc_minn_idx: {dists_cur_bsz_keypts_pc_minn_idx.size()}, cur_bsz_cur_cvx_def_key_pts: {cur_bsz_cur_cvx_def_key_pts.size()}")
            cur_bsz_cur_cvx_def_pc = torch.bmm(cur_bsz_cur_cvx_w_pc, cur_bsz_cur_cvx_def_key_pts)
            
            cd_loss = chamfer_distance(cur_bsz_cur_cvx_def_pc, cur_bsz_tar_pc)
            tot_cd_loss += cd_loss
            
            cvx_nn_to_loss[cur_bsz_pc_nn] = cd_loss
            
            # tot_keypts_basis_offset = torch.zeros_like(key_pts) ## bsz x nn_keypts x 3
            # 
            
            # if cur_bsz_pc_nn > maxx_src_pc_nn:
            #     maxx_src_pc_nn = cur_bsz_pc_nn
            #     tot_cd_loss = cd_loss
            
            
            # tot_keypts_basis_offset: bsz x nn_keypts x 3
            tot_keypts_basis_offset[i_bsz, cur_bsz_keypts_idx, :] += cur_bsz_cur_cvx_key_pts_offset[0]  # * src_scale[0]
            tot_keypts_add_nn[i_bsz, cur_bsz_keypts_idx] += 1
            
            # (cur_bsz_keypts - src_center) / torch.clamp(src_scale, min=1e-6)
            
            cur_cvx_def_pc_pts.append(cur_bsz_cur_cvx_def_pc.detach().cpu().numpy())
            cur_cvx_def_keypts_pts.append(cur_bsz_cur_cvx_def_key_pts.detach().cpu().numpy())
            
            
            
            
            for i_s in range(len(tot_rnd_tot_keypts_basis_offset)):
            
                ratio = torch.rand((1, self.num_basis)).cuda()
                sample_coef = (ratio * coef_range[:, :, 0] + (1 - ratio)
                            * coef_range[:, :, 1]).view(1, 1, self.num_basis)
                sample_def_key_pts_offset =  \
                    torch.bmm(sample_coef, basis).view(1, cur_bsz_keypts_nn, 3)
                # sample_def_key_pts_offset = batched_index_select(sample_def_key_pts_offset, indices=dists_cur_bsz_keypts_pc_minn_idx, dim=1)
                tot_rnd_tot_keypts_basis_offset[i_s][i_bsz, cur_bsz_keypts_idx, :] += sample_def_key_pts_offset[0] #  * src_scale[0]
                tot_rnd_tot_keypts_add_nn[i_s][i_bsz, cur_bsz_keypts_idx] += 1

        
          src_cvx_pc_pts.append(cur_src_cvx_pc_pts)
          tar_cvx_pc_pts.append(cur_tar_cvx_pc_pts)
          
          src_cvx_def_pc_pts.append(cur_cvx_def_pc_pts)
          src_cvx_def_keypts_pts.append(cur_cvx_def_keypts_pts)
          
          tot_basis.append(cur_cvx_tot_basis)
          tot_coef_ranges.append(cur_cvx_tot_coef_ranges)
            

        ### basis or not ###
        tot_rnd_def_key_pts = []
        tot_rnd_def_pc = []
        
        if len(cvx_nn_to_loss) > 0:
            sorted_cvx_nn_to_loss = sorted(cvx_nn_to_loss.items(), key=lambda ii: ii[0], reverse=True)
            
            sorted_cvx_nn_to_loss = [ii[1] for ii in sorted_cvx_nn_to_loss[:7]]
            
            cd_loss = sum(sorted_cvx_nn_to_loss) / float(len(sorted_cvx_nn_to_loss))
        else:
            cd_loss  = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            
        
        for i_s in range(len(tot_rnd_tot_keypts_basis_offset)): ### basis offset 
            rnd_def_key_pts = key_pts + tot_rnd_tot_keypts_basis_offset[i_s] / torch.clamp(tot_rnd_tot_keypts_add_nn[i_s].unsqueeze(-1), min=1.)
            tot_rnd_def_key_pts.append(rnd_def_key_pts)
            rnd_def_pc = torch.bmm(w_pc, rnd_def_key_pts)
            tot_rnd_def_pc.append(rnd_def_pc)
        
        def_key_pts = key_pts + tot_keypts_basis_offset / torch.clamp(tot_keypts_add_nn.unsqueeze(-1), min=1.)
        def_pc = torch.bmm(w_pc, def_key_pts)
        # cd_loss = chamfer_distance(def_pc, tar_pc)
        
        # cd_loss = tot_cd_loss / nn_cvx
        
        
        
        if glb_net is not None:
            # glb_def_key_pts, glb_def_pc, glb_cd_loss, glb_basis, glb_coef, glb_sample_def_key_pts, glb_sym_loss,  glb_coef_range, tot_sampled_def_keypts = glb_net(
            #     def_pc, tar_pc, def_key_pts, w_pc
            # )
            
            # cd_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            
            # glb_def_key_pts, glb_def_pc, glb_cd_loss, glb_basis, glb_coef, glb_sample_def_key_pts, glb_sym_loss,  glb_coef_range, tot_sampled_def_keypts = glb_net(
            #     src_pc, tar_pc, key_pts, w_pc ### w_pc and others...
            # )
            
            glb_def_key_pts, glb_def_pc, glb_cd_loss, glb_basis, glb_coef, glb_sample_def_key_pts, glb_sym_loss,  glb_coef_range, tot_sampled_def_keypts = glb_net(
                def_pc, tar_pc, def_key_pts, w_pc ### w_pc and others... ### def_key_pts 
            )
        
            rt_dict = {
                "cd_loss": cd_loss, 
                "def_key_pts": def_key_pts, 
                "def_pc": def_pc,
                "tot_rnd_def_key_pts": tot_rnd_def_key_pts,
                "tot_rnd_def_pc": tot_rnd_def_pc,
                "src_cvx_pc_pts": src_cvx_pc_pts,
                "tar_cvx_pc_pts": tar_cvx_pc_pts,
                "src_cvx_def_pc_pts": src_cvx_def_pc_pts,
                "src_cvx_def_keypts_pts": src_cvx_def_keypts_pts, 
                "tot_basis": tot_basis, 
                "tot_coef_ranges": tot_coef_ranges,
                "glb_def_key_pts": glb_def_key_pts,
                "glb_def_pc": glb_def_pc,
                "glb_cd_loss": glb_cd_loss, 
                "glb_basis": glb_basis, 
                "glb_coef": glb_coef, 
                "glb_coef_range": glb_coef_range,
                #   "tot_sampled_def_keypts": tot_sampled_def_keypts
            }
        else:
            rt_dict = {
                "cd_loss": cd_loss, 
                "def_key_pts": def_key_pts, 
                "def_pc": def_pc,
                "tot_rnd_def_key_pts": tot_rnd_def_key_pts, ### tot_rnd_def_key_pts
                "tot_rnd_def_pc": tot_rnd_def_pc, ### tot_rnd_def_pc
                "src_cvx_pc_pts": src_cvx_pc_pts,
                "tar_cvx_pc_pts": tar_cvx_pc_pts,
                "src_cvx_def_pc_pts": src_cvx_def_pc_pts,
                "src_cvx_def_keypts_pts": src_cvx_def_keypts_pts, 
                "tot_basis": tot_basis, 
                "tot_coef_ranges": tot_coef_ranges,
            }
            
        
        return rt_dict


    
    def forward4(self, src_pc, tar_pc, key_pts, dst_key_pts, w_pc,  src_verts, src_convex_pts, tar_verts, dst_convex_pts, glb_net=None):
        # src_pc = 
        ### src_pc, tar_pc;
        ### src_convex_pts: 
        ### src_convex_pts ---- only use convexes with the same idx in those two shapes ###
        ### src_convex_pts: bsz x n_cvxes x n_pts_per_convex  ### forward3
        ### src_keypts: bsz x n_keypts x 3
        #### B, N, _ = src
        #### add src_faces, tar_faces for src_meshes and tar_meshes #### 
        #### network target driven ####
        B, N, _ = src_pc.shape
        nn_cvx = src_convex_pts.size(1) ### bsz x nn_cvx 
        
        src_convex_pts = src_convex_pts * 2.
        dst_convex_pts = dst_convex_pts * 2.
        
        # w_pc: bsz x n_pc x n_keypts
        # print(f"w_pc: {w_pc.size()}, w_pc_sum: {torch.sum(w_pc, dim=-1)[0, :2]}")
        # print(f"maxx_src: {torch.max(src_pc)}, minn_src: {torch.min(src_pc)}, maxx src_convex_pts: {torch.max(src_convex_pts)}, minn src_convex_pts: {torch.min(src_convex_pts)}, ")
        
        ###  src ### 
        avg_src_pc_cvx = self.get_minn_distances_base_pc_cvx_pts(src_pc, src_convex_pts) ### 
        ### bsz x nn_pts x nn_cvx --> []
        src_pc_cvx_indicator = self.get_base_pc_cvx_indicator(src_pc, src_convex_pts, avg_src_pc_cvx) ### 
        src_keypts_cvx_indicator = self.get_base_pc_cvx_indicator(key_pts, src_convex_pts, avg_src_pc_cvx) ### 
        
        avg_dst_pc_cvx = self.get_minn_distances_base_pc_cvx_pts(tar_pc, dst_convex_pts)
        tar_pc_cvx_indicator = self.get_base_pc_cvx_indicator(tar_pc, dst_convex_pts, avg_dst_pc_cvx) ### base-pc-cvx
        
        # tot_keypts_basis_offset = torch.zeros(()) ### 
        
        tot_basis = []
        tot_coef_ranges = []
        
        
        
        tot_keypts_basis_offset = torch.zeros_like(key_pts) ## bsz x nn_keypts x 3
        
        tot_keypts_add_nn = torch.zeros_like(key_pts)[..., 0] ## bsz x nn_keypts x 3
        
        tot_rnd_tot_keypts_basis_offset = [ torch.zeros_like(key_pts) for _ in range(10)]
        tot_rnd_tot_keypts_add_nn = [ torch.zeros_like(key_pts)[..., 0]  for _ in range(10)]
        
        tot_cd_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean()
        
        # maxx_src_pc_nn = 0
        
        minn_keypts_nn = 255
        
        # minn_keypts_nn = 128
        
        minn_keypts_nn = 64
        minn_keypts_nn = 22
        
        minn_keypts_nn = 7
        minn_keypts_nn = 3
        
        sum_w_weight_thres = 3e-1
        
        src_cvx_pc_pts = []
        tar_cvx_pc_pts = []
        src_cvx_def_pc_pts = []
        src_cvx_def_keypts_pts = []
        
        cvx_nn_to_loss = {}
        
        
        tot_def_key_pts = []
        
        tot_def_pc = []
        tot_rnd_def_key_pts = []
        tot_rnd_def_pc  = []
        
        tot_basis = []
        
        tot_coefs = []
        
        
        ### add losses --> add losses on meshes other than on pointclouds ###
        
        for i_bsz in range(B):
            
            ### 1 x (tot_nn_basis x nn_cvx) x (nn_keypts x 3) --> total number of keypoints 
            cur_bsz_basis = []
            cur_bsz_ranges = []
            
            cur_bsz_tot_keypts_feats = torch.zeros((1, key_pts.size(1), 64), dtype=torch.float32).cuda()
            
            for i_cvx in range(nn_cvx):
                ### get the keypts offset ##
                cur_src_pc_cvx_indicator = src_pc_cvx_indicator[:, :, i_cvx] ### bsz x nn_pts 
                cur_src_keypts_cvx_indicator = src_keypts_cvx_indicator[:, :, i_cvx] ## bsz x nn_keypts 
                cur_tar_pc_cvx_indicator = tar_pc_cvx_indicator[:, :, i_cvx]
                
                cur_pc_nn = torch.sum(cur_src_pc_cvx_indicator, dim=-1).long() ## bsz x nn_pc
                
                
                #### 
                pc_idx_argsort = torch.argsort(cur_src_pc_cvx_indicator, dim=-1, descending=True) ### bsz x 
                cur_keypts_nn = torch.sum(cur_src_keypts_cvx_indicator, dim=-1).long()
                keypts_idx_argsort = torch.argsort(cur_src_keypts_cvx_indicator, dim=-1, descending=True)
                
                cur_tar_pc_nn = torch.sum(cur_tar_pc_cvx_indicator, dim=-1).long() ## bsz
                tar_pc_idx_argsort = torch.argsort(cur_tar_pc_cvx_indicator, dim=-1, descending=True) ### bsz x 
                cur_src_cvx_pc_pts = []
                cur_tar_cvx_pc_pts = []
                
                cur_cvx_def_pc_pts = []
                cur_cvx_def_keypts_pts = []
                
                cur_cvx_tot_basis = []
                cur_cvx_tot_coef_ranges = []
                
                
                
                cur_bsz_pc_nn = int(cur_pc_nn[i_bsz].item())
                cur_bsz_keypts_nn = int(cur_keypts_nn[i_bsz].item()) ### keypts_nn
                cur_bsz_tar_pc_nn = int(cur_tar_pc_nn[i_bsz].item()) ### 
                
                
                
                # print(f"cur_bsz_pc_nn: {cur_bsz_pc_nn}, cur_bsz_keypts_nn: {cur_bsz_keypts_nn}, cur_bsz_tar_pc_nn: {cur_bsz_tar_pc_nn}")
                if cur_bsz_pc_nn <= 100 or cur_bsz_keypts_nn <= minn_keypts_nn or cur_bsz_tar_pc_nn <= 100:
                    continue
                cur_bsz_pc_idx = pc_idx_argsort[i_bsz, :cur_bsz_pc_nn]
                cur_bsz_keypts_idx = keypts_idx_argsort[i_bsz, :cur_bsz_keypts_nn] 
                cur_bsz_tar_pc_idx = tar_pc_idx_argsort[i_bsz, : cur_bsz_tar_pc_nn]
                
                
                cur_keypts_w_pc = w_pc[:, :, cur_bsz_keypts_idx]
                # cur_keypts_w_pc_sum = torch.abs(torch.sum((cur_keypts_w_pc), dim=-1))
                cur_keypts_w_pc_sum = (torch.sum((cur_keypts_w_pc), dim=-1))
                cur_src_pc_cvx_indicator[cur_keypts_w_pc_sum < sum_w_weight_thres] = 0.
                cur_pc_nn = torch.sum(cur_src_pc_cvx_indicator, dim=-1).long() ## bsz x nn_pc
                cur_bsz_pc_nn = int(cur_pc_nn[i_bsz].item())
                if cur_bsz_pc_nn <= 100 :
                    continue
                pc_idx_argsort = torch.argsort(cur_src_pc_cvx_indicator, dim=-1, descending=True) ### bsz x 
                cur_bsz_pc_idx = pc_idx_argsort[i_bsz, :cur_bsz_pc_nn]
                
                
                ### w_pc: 
                cur_bsz_cur_cvx_w_pc = w_pc[i_bsz, cur_bsz_pc_idx, :] ## nn_cvx_pc x nn_keypts ##
                
                cur_bsz_cur_cvx_w_pc = cur_bsz_cur_cvx_w_pc[:, cur_bsz_keypts_idx] ## nn_cvx_pc x nn_cvx_keypts ##
                cur_bsz_cur_cvx_w_pc1 = cur_bsz_cur_cvx_w_pc.unsqueeze(0).transpose(2, 1).unsqueeze(2) ### 1 x nn_pc x
                cur_bsz_cur_cvx_w_pc = cur_bsz_cur_cvx_w_pc.unsqueeze(0) ## 1 x nn_cvx_pc x nn_cvx_keypts
                
                
                
                    ### cur_bsz_cur_cvx_w_pc --> 
                cur_bsz_cur_cvx_w_pc = cur_bsz_cur_cvx_w_pc / torch.sum((cur_bsz_cur_cvx_w_pc), dim=-1, keepdim=True)
                # torch.clamp(torch.sum((cur_bsz_cur_cvx_w_pc), dim=-1, keepdim=True), min=sum_w_weight_thres) ### 1 x nn_cvx_pc x 1
                # cur_bsz_cur_cvx_w_pc[cur_bsz_cur_cvx_w_pc > 1.] = 1.
                cur_bsz_cur_cvx_w_pc2 = cur_bsz_cur_cvx_w_pc.transpose(2, 1).unsqueeze(2) 
                
                
                # cur_src_cvx_pc_pt
                
                
                cur_bsz_pc = src_pc[i_bsz, cur_bsz_pc_idx, :].unsqueeze(0)
                cur_bsz_keypts = key_pts[i_bsz, cur_bsz_keypts_idx, :].unsqueeze(0)
                cur_bsz_tar_pc = tar_pc[i_bsz, cur_bsz_tar_pc_idx, :].unsqueeze(0) #### tar_pc_idx
                
                
                
            
                
                # cur_bsz_pc = cur_bsz_pc - torch.mean(cur_bsz_pc, dim=1, keepdim=True)
                # cur_bsz_tar_pc = cur_bsz_tar_pc - torch.mean(cur_bsz_tar_pc, dim=1, keepdim=True)
                # cur_bsz_keypts = cur_bsz_keypts - torch.mean(cur_bsz_keypts, dim=1, keepdim=True)
                
                
                cur_bsz_pc, src_center, src_scale = utils.normalie_pc_bbox_batched(pc=cur_bsz_pc, rt_stats=True)
                cur_bsz_keypts = (cur_bsz_keypts - src_center) / torch.clamp(src_scale, min=1e-6)
                cur_bsz_tar_pc = utils.normalie_pc_bbox_batched(cur_bsz_tar_pc, rt_stats=False)
            
            
                cur_src_cvx_pc_pts.append(cur_bsz_pc.detach().cpu().numpy())
                cur_tar_cvx_pc_pts.append(cur_bsz_tar_pc.detach().cpu().numpy())
                
                cur_bsz_keypts1 = cur_bsz_keypts.unsqueeze(-1).expand(-1, -1, -1, cur_bsz_pc_nn)
                
                cur_bsz_pc1 = cur_bsz_pc.unsqueeze(-1).expand(-1, -1, -1, cur_bsz_pc_nn)
                
                
                ''' pointnet '''
                # src_feats, src_glb_feats = self.pointnet(cur_bsz_pc, return_global=False) ## bsz x 
                # ### 
                # tar_feats, tar_glb_feats = self.tar_pointnet(cur_bsz_tar_pc, return_global=False) ### tar_feats 
                ''' pointnet '''
                
                
                
                ''' pvcnn '''
                # fused_features, voxel_coords = self.pvconv((cur_bsz_pc.contiguous().transpose(-1, -2).contiguous(), cur_bsz_pc.contiguous().transpose(-1, -2).contiguous())) ### contiguous() transpose() view()  ### obj
                # ## fused_features: b x dim x n_pts 
                
                fused_features, _ = self.pn_enc(cur_bsz_pc)
                
                
                src_feats = fused_features
            
            
                src_glb_feats = src_feats.max(dim=-1)[0].unsqueeze(-1) ## bsz x dim x 1
                
                
                ### 
                src_feats = torch.cat([src_feats, src_glb_feats.expand(-1, -1, src_feats.size(-1))], dim=1)
                src_feats = self.src_ppfeats_out_net(src_feats) ### for global features 

                src_glb_feats = self.src_glb_feat_out_net(src_glb_feats).squeeze(-1) #### src_glb_feats


                # tar_fused_features, voxel_coords = self.tar_pvconv((cur_bsz_tar_pc.contiguous().transpose(-1, -2).contiguous(), cur_bsz_tar_pc.contiguous().transpose(-1, -2).contiguous())) ### contiguous() transpose() view()  ### obj
                # ## fused_features: b x dim x n_pts 
                
                tar_fused_features, _ = self.tar_pn_enc(cur_bsz_tar_pc)
                tar_feats = tar_fused_features
                
                
                tar_glb_feats = tar_feats.max(dim=-1)[0].unsqueeze(-1) ## bsz x dim x 1

                tar_glb_feats = self.tar_glb_feat_out_net(tar_glb_feats).squeeze(-1)      
                
                # print(f"src_feats: {src_feats.size()}, tar_glb_feats: {tar_glb_feats.size()}, src_glb_feats: {src_glb_feats.size()}")
                
                
                src_feats = F.relu(self.bn11(self.conv11(src_feats)))
                src_feats = F.relu(self.bn12(self.conv12(src_feats)))
                src_feats = F.relu(self.bn13(self.conv13(src_feats)))

                
                ### 
                src_feats = src_feats.unsqueeze(1).expand(-1, cur_bsz_keypts_nn, -1, -1) 
                
                
                
                cur_bsz_src_pc_feats = torch.cat( ### 
                    [src_feats, cur_bsz_cur_cvx_w_pc1, cur_bsz_keypts1], dim=2
                ).view(1 * cur_bsz_keypts_nn, -1, cur_bsz_pc_nn)
                
                # print(f"src_feats: {src_feats.size()}, cur_bsz_cur_cvx_w_pc: {cur_bsz_cur_cvx_w_pc.size()}, cur_bsz_keypts1: {cur_bsz_keypts1.size()}")
                
                
                cur_bsz_src_pc_feats = F.relu(self.bn21(self.conv21(cur_bsz_src_pc_feats)))
                cur_bsz_src_pc_feats = self.bn22(self.conv22(cur_bsz_src_pc_feats))
                
                ### cur_bsz_keypts_feats: nn_keypts x nn_feats
                cur_bsz_src_pc_feats = torch.max(cur_bsz_src_pc_feats, 2, keepdim=True)[0] 
                
                
                key_fea = cur_bsz_src_pc_feats.view(1 * cur_bsz_keypts_nn, -1, 1) ###  ### cur_bsz_tot_keypts_feats 
                
                ####### cur_bsz_nn_keypts + pc_feats #### 
                cur_bsz_tot_keypts_feats[0, cur_bsz_keypts_idx, :] = cur_bsz_tot_keypts_feats[0, cur_bsz_keypts_idx, :] + key_fea.squeeze(-1)
                
                
                net = torch.cat([key_fea, cur_bsz_keypts.view(1 * cur_bsz_keypts_nn, 3, 1)], 1) 
                
                
                net = F.relu(self.bn31(self.conv31(net))) #### net
                net = F.relu(self.bn32(self.conv32(net)))
                basis = self.conv33(net).view(1, cur_bsz_keypts_nn * 3, self.num_basis).transpose(1, 2)
                basis = basis / torch.clamp(basis.norm(p=2, dim=-1, keepdim=True), min=1e-6)  ### basis norm  ### 1 x (cur_bsz_keypts_nn x 3) x nn_basis --> 1 x nn_basiss x (cur_bsz_keypts_nn x 3) x
                
                
                glb_basis = torch.zeros((1, self.num_basis, key_pts.size(1), 3), dtype=torch.float32).cuda() ### 
                cur_basis_exp = basis.contiguous().view(1, self.num_basis, cur_bsz_keypts_nn, 3)
                # print(f"glb_basis: {glb_basis.size()}, cur_basis_exp: {cur_basis_exp.size()}")
                glb_basis[:, :, cur_bsz_keypts_idx, :] = glb_basis[:, :, cur_bsz_keypts_idx, :] + cur_basis_exp ### glb_basis
                
                cur_bsz_basis.append(glb_basis)
                
                
                
                key_fea_range = key_fea.view(
                    1, cur_bsz_keypts_nn, -1, 1).expand(-1, -1, -1, self.num_basis).transpose(1, 3)
                key_pts_range = cur_bsz_keypts.view(
                    1, cur_bsz_keypts_nn, 3, 1).expand(-1, -1, -1, self.num_basis).transpose(1, 3)
                basis_range = basis.view(1, self.num_basis, cur_bsz_keypts_nn, 3).transpose(2, 3) ### cur_Bsz_
                
                
                
                ### get range
                coef_range = torch.cat([key_fea_range, key_pts_range, basis_range], 2).view(
                    1 * self.num_basis, -1, cur_bsz_keypts_nn)
                coef_range = F.relu(self.bn71(self.conv71(coef_range)))
                coef_range = F.relu(self.bn72(self.conv72(coef_range)))
                coef_range = self.conv73(coef_range)
                coef_range = torch.max(coef_range, 2, keepdim=True)[0]
                
                
                coef_range = coef_range.view(1, self.num_basis, 2) * self.coef_multiplier ### coef-multiplier
                coef_range[:, :, 0] = coef_range[:, :, 0] * -1 ### coef_rnages: 1 x nn_basis x 2
                
                coef_range[:, :, 0] = -1. * self.coef_multiplier
                coef_range[:, :, 1] = 1. * self.coef_multiplier
                
                cur_bsz_ranges.append(coef_range) ### 1 x nn_basis x 2 ---> 
            
            if len(cur_bsz_basis) <= 0:
                continue
            
            cur_bsz_basis = torch.cat(cur_bsz_basis, dim=1) ### 1 x (tot_nn_basis) x nn_keypts x 3
            cur_bsz_ranges = torch.cat(cur_bsz_ranges, dim=1) ### 1 x tot_nn_basis x 3 --> ? 
            
            tot_basis.append(cur_bsz_basis.view(1, cur_bsz_basis.size(1), -1).contiguous())
            
            
            # feat_pred_sc_pc = utils.normalie_pc_bbox_batched
            glb_src_fused_features, _ = self.glb_pn_enc(src_pc[i_bsz].unsqueeze(0))
            glb_tar_fused_features, _ = self.glb_tar_pn_enc(tar_pc[i_bsz].unsqueeze(0)) ### 
            
            # glb_src_fused_features, _ = self.glb_pn_enc((src_pc[i_bsz].unsqueeze(0).transpose(-1, -2), src_pc[i_bsz].unsqueeze(0).transpose(-1, -2)))
            # glb_tar_fused_features, _ = self.glb_pn_enc((tar_pc[i_bsz].unsqueeze(0).transpose(-1, -2), tar_pc[i_bsz].unsqueeze(0).transpose(-1, -2))) ### 
            
            glb_glb_src_fused_features = glb_src_fused_features.max(dim=-1)[0].unsqueeze(-1)
            glb_glb_tar_fused_features = glb_tar_fused_features.max(dim=-1)[0].unsqueeze(-1)
            
            glb_glb_src_fused_features = self.glb_src_glb_feat_out_net(glb_glb_src_fused_features).squeeze(-1)
            glb_glb_tar_fused_features = self.glb_tar_glb_feat_out_net(glb_glb_tar_fused_features).squeeze(-1) ### bsz 
            
            # print(f"glb_glb_src_fused_features: {glb_glb_src_fused_features.size()}, glb_glb_tar_fused_features: {glb_glb_tar_fused_features.size()}")
            ### src_tar features ###
            
            src_tar = torch.cat([glb_glb_src_fused_features, glb_glb_tar_fused_features], 1).unsqueeze( ### expanded features 
                1).expand(-1, key_pts.size(1), -1).reshape(1 * key_pts.size(1), -1, 1)     
            
            
            # src_tar = torch.cat([glb_glb_tar_fused_features, glb_glb_tar_fused_features], 1).unsqueeze( ### expanded features 
            #     1).expand(-1, key_pts.size(1), -1).reshape(1 * key_pts.size(1), -1, 1)     
            
                
                
            ### to the source shape ###
            ### keypoint features, srouce target fused features, and keypoint coordiantes ###
            ### key_feature; source_target
            key_fea = torch.cat([cur_bsz_tot_keypts_feats.squeeze(0).unsqueeze(-1), src_tar, key_pts.view(1 * key_pts.size(1), 3, 1)], 1)
            # print(f"key_fea: {key_fea.size()}")
            key_fea = F.relu(self.bn41(self.conv41(key_fea))) ### key_fea
            key_fea = F.relu(self.bn42(self.conv42(key_fea))) ### key_fea
            key_fea = F.relu(self.bn43(self.conv43(key_fea))) ### key_fea

            key_fea = key_fea.view(1, key_pts.size(1), 128).transpose(
                1, 2).unsqueeze(1).expand(-1, cur_bsz_basis.size(1), -1, -1)
            cur_bsz_keypts2 = key_pts[i_bsz].view(1,  key_pts.size(1), 3).transpose(  ### key_pts2;
                1, 2).unsqueeze(1).expand(-1, cur_bsz_basis.size(1), -1, -1)
            basis1 = cur_bsz_basis.view(1, -1,  key_pts.size(1), 3).transpose(2, 3) ### nn_key_pts ### nn_key_pts
            

            net = torch.cat([key_fea, basis1, cur_bsz_keypts2], 2).view( ### keypoint features, basis, coefs, --- global feautre -> fo rcoefs 
                1 * cur_bsz_basis.size(1), -1, key_pts.size(1))

            net = F.relu(self.bn51(self.conv51(net)))
            net = F.relu(self.bn52(self.conv52(net)))
            net = self.bn53(self.conv53(net))
            ### net: cur_nn_basis x dim x nn_keypts 
            if self.pred_type == "offset":
                # net = torch.cat([net, cur_bsz_ranges.view(1 * cur_bsz_basis.size(1), 2, 1).expand(-1, -1, key_pts.size(1))], 1)
                # net = torch.cat([net, cur_bsz_ranges.view(1 * cur_bsz_basis.size(1), 2, 1).expand(-1, -1, key_pts.size(1))], 1)
                net = F.relu(self.bn61(self.conv61(net)))
                net = F.relu(self.bn62(self.conv62(net)))
                offset = self.conv63(net)
                coef = torch.ones_like(offset)[:, 0, 0].unsqueeze(0).unsqueeze(0)
                offset = offset[0, :, :].unsqueeze(0).transpose(1, 2).contiguous()
                cur_bsz_key_pts_offset = offset
                
                
            else:
                net = torch.max(net, 2, keepdim=True)[0]
                net = net.view(1 * cur_bsz_basis.size(1), -1, 1)

                net = torch.cat([net, cur_bsz_ranges.view(1 * cur_bsz_basis.size(1), 2, 1)], 1)
                net = F.relu(self.bn61(self.conv61(net)))
                net = F.relu(self.bn62(self.conv62(net)))
                
                # net = F.relu((self.conv61(net)))
                # net = F.relu((self.conv62(net)))
                coef = self.sigmoid(self.conv63(net)).view(1, cur_bsz_basis.size(1))
                # print(coef[0])
                

                coef = (coef * cur_bsz_ranges[:, :, 0] + (1 - coef)
                        * cur_bsz_ranges[:, :, 1]).view(1, 1, cur_bsz_basis.size(1))
                cur_bsz_key_pts_offset = torch.bmm(coef, cur_bsz_basis.view(1, cur_bsz_basis.size(1), -1).contiguous()).view(1, key_pts.size(1), 3)
                
            
            
            tot_coefs.append(coef)
                
            cur_cvx_tot_basis.append(cur_bsz_basis) ### 1 x n_basis x (n_keypts x 3)
            cur_cvx_tot_coef_ranges.append(cur_bsz_ranges)
            
            
                ### def_key_pts ###
                
                
            cur_bsz_def_key_pts = key_pts[i_bsz].unsqueeze(0) + cur_bsz_key_pts_offset
            
            tot_def_key_pts.append(cur_bsz_def_key_pts)
            
            from common_utils.geo_operations import mean_value_coordinates
            w_pc_cur_bsz = mean_value_coordinates(src_pc[i_bsz].unsqueeze(0).transpose(1, 2), key_pts[i_bsz].unsqueeze(0).transpose(1, 2))
            w_pc_cur_bsz = w_pc_cur_bsz.transpose(1, 2)
            print(torch.sum(w_pc_cur_bsz, dim=-1)[:, :10])
            
            # cur_bsz_def_pc = torch.bmm(w_pc[i_bsz].unsqueeze(0), cur_bsz_def_key_pts)
            
            cur_bsz_def_pc = torch.bmm(w_pc_cur_bsz, cur_bsz_def_key_pts)
            cur_bsz_cd_loss = chamfer_distance(cur_bsz_def_pc, tar_pc[i_bsz].unsqueeze(0)) ### cur_bsz_def_pc 
            tot_cd_loss += cur_bsz_cd_loss
            
            tot_def_pc.append(cur_bsz_def_pc)
            
            
            ### ws --> strange --> ws 
            # rnd_coef = torch.rand_like(coef)
            # rnd_coef = (rnd_coef * cur_bsz_ranges[:, :, 0] + (1 - rnd_coef)
            #         * cur_bsz_ranges[:, :, 1]).view(1, 1, cur_bsz_basis.size(1))
            # cur_bsz_rnd_key_pts_offset = torch.bmm(rnd_coef, cur_bsz_basis.view(1, cur_bsz_basis.size(1), -1).contiguous()).view(1, key_pts.size(1), 3)
            
            # cur_bsz_rnd_def_key_pts =  key_pts[i_bsz].unsqueeze(0) + cur_bsz_rnd_key_pts_offset
            # tot_rnd_def_key_pts.append(cur_bsz_rnd_def_key_pts) ### def_key_pts 
            
            # cur_bsz_rnd_def_pc = torch.bmm(w_pc[i_bsz].unsqueeze(0), cur_bsz_rnd_def_key_pts)
            # tot_rnd_def_pc.append(cur_bsz_rnd_def_pc)
            
            for i_rnd in range(cur_bsz_basis.size(1)):
                rnd_coef = torch.zeros_like(coef) ### coefs: 1 x nn_tot_basis
                rnd_coef[:, :, i_rnd] = 1.
                cur_bsz_rnd_key_pts_offset = torch.bmm(rnd_coef, cur_bsz_basis.view(1, cur_bsz_basis.size(1), -1).contiguous()).view(1, key_pts.size(1), 3)
                
                cur_bsz_rnd_def_key_pts =  key_pts[i_bsz].unsqueeze(0) + cur_bsz_rnd_key_pts_offset
                tot_rnd_def_key_pts.append(cur_bsz_rnd_def_key_pts) ## rnd_def
                
                cur_bsz_rnd_def_pc = torch.bmm(w_pc[i_bsz].unsqueeze(0), cur_bsz_rnd_def_key_pts)
                tot_rnd_def_pc.append(cur_bsz_rnd_def_pc)
            
        np.save("tot_basis.npy", torch.cat(tot_basis, axis=0).detach().cpu().numpy())
            
        cd_loss = tot_cd_loss / float(B)
        if len(tot_def_key_pts) > 0:
            def_key_pts = torch.cat(tot_def_key_pts, dim=0)
            def_pc = torch.cat(tot_def_pc, dim=0)
            # tot_rnd_def_key_pts = [torch.cat(tot_rnd_def_key_pts, dim=0)]
            # tot_rnd_def_pc = [torch.cat(tot_rnd_def_pc, dim=0)]
            
            # tot_rnd_def_key_pts = [torch.cat(tot_rnd_def_key_pts, dim=0)]
            tot_rnd_def_key_pts = tot_rnd_def_key_pts
            # tot_rnd_def_pc = [torch.cat(tot_rnd_def_pc, dim=0)] 
            tot_rnd_def_pc = tot_rnd_def_pc
            tot_coefs = [tot_coefs]
            tot_basis  = [tot_basis]
        else:
            def_key_pts = key_pts.clone()
            def_pc = src_pc.clone()
            tot_rnd_def_key_pts = [key_pts.clone()]
            tot_rnd_def_pc = [src_pc.clone()]
            tot_coefs = [[]]
        
        # if glb_net is not None:
        #     # glb_def_key_pts, glb_def_pc, glb_cd_loss, glb_basis, glb_coef, glb_sample_def_key_pts, glb_sym_loss,  glb_coef_range, tot_sampled_def_keypts = glb_net(
        #     #     def_pc, tar_pc, def_key_pts, w_pc
        #     # )
            
        #     cd_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            
        #     glb_def_key_pts, glb_def_pc, glb_cd_loss, glb_basis, glb_coef, glb_sample_def_key_pts, glb_sym_loss,  glb_coef_range, tot_sampled_def_keypts = glb_net(
        #         src_pc, tar_pc, key_pts, w_pc ### w_pc and others...
        #     )
        
        #     rt_dict = {
        #         "cd_loss": cd_loss, 
        #         "def_key_pts": def_key_pts, 
        #         "def_pc": def_pc,
        #         "tot_rnd_def_key_pts": tot_rnd_def_key_pts,
        #         "tot_rnd_def_pc": tot_rnd_def_pc,
        #         "src_cvx_pc_pts": src_cvx_pc_pts,
        #         "tar_cvx_pc_pts": tar_cvx_pc_pts,
        #         "src_cvx_def_pc_pts": src_cvx_def_pc_pts,
        #         "src_cvx_def_keypts_pts": src_cvx_def_keypts_pts, 
        #         "tot_basis": tot_basis, 
        #         "tot_coef_ranges": tot_coef_ranges,
        #         "glb_def_key_pts": glb_def_key_pts,
        #         "glb_def_pc": glb_def_pc,
        #         "glb_cd_loss": glb_cd_loss, 
        #         "glb_basis": glb_basis, 
        #         "glb_coef": glb_coef, 
        #         "glb_coef_range": glb_coef_range,
        #         #   "tot_sampled_def_keypts": tot_sampled_def_keypts
        #     }
        # else:
        rt_dict = {
            "cd_loss": cd_loss, 
            "def_key_pts": def_key_pts, 
            "def_pc": def_pc,
            "tot_rnd_def_key_pts": tot_rnd_def_key_pts,
            "tot_rnd_def_pc": tot_rnd_def_pc,
            "src_cvx_pc_pts": src_cvx_pc_pts,
            "tar_cvx_pc_pts": tar_cvx_pc_pts,
            "src_cvx_def_pc_pts": src_cvx_def_pc_pts,
            "src_cvx_def_keypts_pts": src_cvx_def_keypts_pts, 
            "tot_basis": tot_basis, 
            "tot_coefs": tot_coefs,
            "tot_coef_ranges": tot_coef_ranges,
        }
            
        
        return rt_dict
   
   