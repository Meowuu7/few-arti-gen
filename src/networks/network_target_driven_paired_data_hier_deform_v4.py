from multiprocessing.sharedctypes import Value
from platform import java_ver
import torch
import torch.nn.functional as F
import torch.nn as nn
from pointnet_utils import pointnet_encoder
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

class deform_model(nn.Module):
    def __init__(self, num_basis, opt=None):
        super(deform_model, self).__init__()
        # print("for network", opt)
        self.opt = opt
        self.use_prob = self.opt.use_prob
        self.tar_basis = self.opt.tar_basis
        self.coef_multiplier = self.opt.coef_multiplier
        self.n_layers = self.opt.n_layers
        
        self.num_basis = num_basis
        self.pred_type = self.opt.pred_type
        self.neighbouring_k = opt.neighbouring_k
        
        self.n_samples = opt.n_samples
        self.symmetry_axis = opt.symmetry_axis ### symmetry_axis ###
        
        print(f"Using symmetry_axis: {self.symmetry_axis}")
        
        print(f"prediction type: {self.pred_type}")
        self.use_pointnet2 = self.opt.use_pointnet2
        # print(f"whether to use pointnet2: {self.use_pointnet2}")
        self.use_graphconv = self.opt.use_graphconv
        # print(f"whether to use graphconv: {self.use_graphconv}")
        
        #### pp_tar_out_feat ####
        self.use_pp_tar_out_feat = self.opt.use_pp_tar_out_feat
        
        print(f"whether to use pp_tar_out_feat: {self.use_pp_tar_out_feat}")
        
        if not self.use_graphconv:
            if self.use_pointnet2:
                # self.pointnet = PointnetPP(in_feat_dim=3, use_light_weight=False, args=opt)
                # self.tar_pointnet = PointnetPP(in_feat_dim=3, use_light_weight=False, args=opt)
                self.pp_out_dim = 128
                
                self.pointnet = edge_propagation.KNNFeatureBlock(k=self.neighbouring_k, encode_feat_dim=self.pp_out_dim)
                self.tar_pointnet = edge_propagation.KNNFeatureBlock(k=self.neighbouring_k, encode_feat_dim=self.pp_out_dim)
                self.pp_out_dim = 128
                # self.pointnet = 
                # self.pp_out_dim = 128 + 1024
            else: ### pp_out_dim...
                self.pointnet = pointnet_encoder()
                self.tar_pointnet = pointnet_encoder()
                self.pp_out_dim = 2883
        else:
            self.graph_conv_net = edge_propagation.GraphConvNet(n_layers=self.n_layers)
            self.tar_graph_conv_net = edge_propagation.GraphConvNet(n_layers=self.n_layers)
            self.pp_out_dim = 128 ### 128

        # src point feature 2883 * N
        self.conv11 = torch.nn.Conv1d(self.pp_out_dim, 128, 1)
        self.conv12 = torch.nn.Conv1d(128, 64, 1)
        self.conv13 = torch.nn.Conv1d(64, 64, 1)
        self.bn11 = nn.BatchNorm1d(128)
        self.bn12 = nn.BatchNorm1d(64)
        self.bn13 = nn.BatchNorm1d(64)
        
        self.tar_conv11 = torch.nn.Conv1d(self.pp_out_dim, 128, 1)
        self.tar_conv12 = torch.nn.Conv1d(128, 128, 1)
        self.tar_conv13 = torch.nn.Conv1d(128, 128, 1)
        self.tar_bn11 = nn.BatchNorm1d(128)
        self.tar_bn12 = nn.BatchNorm1d(128)
        self.tar_bn13 = nn.BatchNorm1d(128)

        # key point feature K (64 + 3 + 1) * N
        self.conv21 = torch.nn.Conv1d(68, 64, 1)
        self.conv22 = torch.nn.Conv1d(64, 64, 1)
        self.bn21 = nn.BatchNorm1d(64)
        self.bn22 = nn.BatchNorm1d(64)
        
        self.tar_conv21 = torch.nn.Conv1d(128, 64, 1)
        self.tar_conv22 = torch.nn.Conv1d(64, 64, 1)
        self.tar_bn21 = nn.BatchNorm1d(64)
        self.tar_bn22 = nn.BatchNorm1d(64)

        # basis feature K 64
        self.conv31 = torch.nn.Conv1d(64 + 3, 256, 1)
        self.conv32 = torch.nn.Conv1d(256, 512, 1)
        self.conv33 = torch.nn.Conv1d(512, self.num_basis * 3, 1)
        self.bn31 = nn.BatchNorm1d(256)
        self.bn32 = nn.BatchNorm1d(512)

        # key point feature with target K (2048 + 64 + 3)
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



    ### forward(self, ) ## attention is all you need? ### coarse     
    def forward(self, src_pc, tar_pc, key_pts, dst_key_pts, w_pc,  src_verts, src_edges, src_dofs, tar_verts, tar_edges, tar_dofs):
        #### B, N, _ 
        #### network target driven ####
        B, N, _ = src_pc.shape
        
        ### downsample pc ###
        n_samples = 512
        # n_samples = 2048
        n_samples = 1024
        n_samples = self.n_samples
        bsz, N = src_pc.size(0), src_pc.size(1)
        src_fps_idx = farthest_point_sampling(pos=src_pc[:, :, :3], n_sampling=n_samples) ### smapling for features xxx
        ### src_pc: bsz x N x 3
        src_pc_downsampled = src_pc.contiguous().view(bsz * N, 3)[src_fps_idx].contiguous().view(bsz, n_samples, 3).contiguous()
        
        # w_pc: bsz x N x K ### w_pc: bsz x N x K
        w_pc_expand = w_pc.contiguous().view(bsz * N, -1)[src_fps_idx].contiguous().view(bsz, n_samples, -1).contiguous()
        
        tar_fps_idx = farthest_point_sampling(pos=tar_pc[:, :, :3], n_sampling=n_samples) ### smapling for features xxx
        ### src_pc: bsz x N x 3
        tar_pc_downsampled = tar_pc.contiguous().view(bsz * N, 3)[tar_fps_idx].contiguous().view(bsz, n_samples, 3).contiguous()
        
        
        
        N = n_samples
        # bz x n_samples x pos_dim
        
        
        # if not self.use_pp_tar_out_feat:
        #     #### source out; source global ####
        #     src_out, src_global = self.pointnet(src_pc, False)
        #     #### target global #### #### tar_pc #####
        #     # tar_out, tar_global = self.pointnet(tar_pc, False) ### target pc ##3 flow predition and composition? 
        #     tar_out, tar_global = self.tar_pointnet(tar_pc, False)
            
        #     src_pp_topk_idxes = None
        #     tar_pp_topk_idxes = None
        # else:
        
        ### use vae prob for local feature pool training and testing ###
        src_out, src_pp_topk_idxes = self.pointnet(src_pc_downsampled)
        tar_out, tar_pp_topk_idxes = self.tar_pointnet(tar_pc_downsampled)
        # print(f"src_global: {src_global.size()}, tar_global: {tar_global.size()}")

        
        src_out = F.relu(self.bn11(self.conv11(src_out)))
        src_out = F.relu(self.bn12(self.conv12(src_out)))
        src_out = F.relu(self.bn13(self.conv13(src_out))) #### src_pp_features
        ### network and keypoints 

        ### keypts, key_pts, 
        _, K, _ = key_pts.shape
        key_pts1 = key_pts.unsqueeze(-1).expand(-1, -1, -1, N)  # B K 3 N
        #### w_pc --> w_pc1 --> transpose weights for further computation
        
        #### using no downssample ###
        # w_pc1 = w_pc.transpose(2, 1).unsqueeze(2)  # B K 1 N
        #### for downsample ####  ### w_pc_expand --> for pc weights ###
        w_pc1 = w_pc_expand.transpose(2, 1).unsqueeze(2) # w_pc_expand #
        src_out = src_out.unsqueeze(1).expand(-1, K, -1, -1)  # B K 64 N
        ### what should be cat here --> src_out; w_pc1; key_pts1; 
        ### for each keypoint, 1) keypoints-to-surface_points weights; 2) key_points values; 3) expanded source points 
        
        # print(f"for net cat: {src_out.size()}, w_pc1: {w_pc1.size()}, key_pts1: {key_pts1.size()}")
        # w_pc1, key_pts1; keypoints, source out, weights from keypoints to pcs 
        net = torch.cat([src_out, w_pc1, key_pts1], 2).view(B * K, 68, N) #### 

        net = F.relu(self.bn21(self.conv21(net)))
        net = self.bn22(self.conv22(net))
        
        
        tar_out = F.relu(self.tar_bn11(self.tar_conv11(tar_out)))
        tar_out = F.relu(self.tar_bn12(self.tar_conv12(tar_out)))
        tar_out = F.relu(self.tar_bn13(self.tar_conv13(tar_out)))
        if not self.use_pp_tar_out_feat:
            tar_out = torch.max(tar_out, dim=2, keepdim=True)[0] ###
        tar_out_dim = tar_out.size(1)
        tar_out_mu, tar_out_log_sigma = tar_out[:, :tar_out_dim // 2], tar_out[:, tar_out_dim // 2: ]
        tar_out = tar_out_mu
        tar_out = tar_out.unsqueeze(1).expand(-1, K, -1, -1) 
        loss_prior = torch.zeros((1,), dtype=torch.float32).cuda().mean()
        
        
        ##### pool net for glboal features ######
        ### net and netout ### ### net ###
        net = torch.max(net, 2, keepdim=True)[0] ### net with tar_out
        ### 
        key_fea = net.view(B * K, 64, 1) ### reshape to keypoints' features ### key_features 

        #### keypoints, keypoint basis, keypoints basis; num_basis x 3 ---> we need a local backbone; 
        net = torch.cat([key_fea, key_pts.view(B * K, 3, 1)], 1) #### key_points #### keypoints #### keypoints 
        net = F.relu(self.bn31(self.conv31(net)))
        net = F.relu(self.bn32(self.conv32(net)))
        
        #### net --- basis ####
        basis = self.conv33(net).view(B, K * 3, self.num_basis).transpose(1, 2)
        
        
        basis = basis / torch.clamp(basis.norm(p=2, dim=-1, keepdim=True), min=1e-6)


        key_fea_range = key_fea.view(
            B, K, 64, 1).expand(-1, -1, -1, self.num_basis).transpose(1, 3) #### B, self.num_basis, K, 64
        key_pts_range = key_pts.view(
            B, K, 3, 1).expand(-1, -1, -1, self.num_basis).transpose(1, 3) 
        basis_range = basis.view(B, self.num_basis, K, 3).transpose(2, 3)


        #### coefficient range ####
        coef_range = torch.cat([key_fea_range, key_pts_range, basis_range], 2).view(
            B * self.num_basis, 70, K)
        coef_range = F.relu(self.bn71(self.conv71(coef_range))) ### coef_range
        coef_range = F.relu(self.bn72(self.conv72(coef_range))) ### coef_range and ...
        coef_range = self.conv73(coef_range)
        coef_range = torch.max(coef_range, 2, keepdim=True)[0]
        # coef_range = coef_range.view(B, self.num_basis, 2) * 0.02 ### 
        coef_range = coef_range.view(B, self.num_basis, 2) * self.coef_multiplier #  0.1 
        # coef_range = coef_range.view(B, self.num_basis, 2) * 0.5
        coef_range[:, :, 0] = coef_range[:, :, 0] * -1
        #### coefficient range #### 
        
        
        # src_tar = torch.cat([src_global, tar_global], 1).unsqueeze(
        #     1).expand(-1, K, -1).reshape(B * K, 2048, 1)
        
        # deformation space logics

        ### to the source shape ###
        ### keypoint features, srouce target fused features, and keypoint coordiantes ###
        ### key_feature; source_target ### src_tar and key_feature...
        # key_fea = torch.cat([key_fea, src_tar, key_pts.view(B * K, 3, 1)], 1)
        # key_fea = F.relu(self.bn41(self.conv41(key_fea))) ### key_fea
        # key_fea = F.relu(self.bn42(self.conv42(key_fea))) ### key_fea
        # key_fea = F.relu(self.bn43(self.conv43(key_fea))) ### 
        
        if self.use_pp_tar_out_feat:
            # print(f"key_fea: {key_fea.size()}, tar_out: {tar_out.size()}")
            # key_fea: B K feat_dim
            key_fea_expanded = key_fea.contiguous().repeat(1, 1, N).contiguous()
            # tar_out: B fea_dim N
            # print(f"tar_out: {tar_out.size()}")
            # tar_out_expanded = tar_out.contiguous().unsqueeze(1).contiguous().repeat(1, K, 1, 1).contiguous().view(B * K, tar_out.size(1), N)
            tar_out_expanded = tar_out.contiguous().view(B * K, tar_out.size(1), N)
            # key_fea_expanded: B K (fea_dim + tar_out_fea_dim) N
            key_fea_expanded = torch.cat([key_fea_expanded, tar_out_expanded], dim=1)
            key_fea_expanded = F.relu(self.bn41(self.conv41(key_fea_expanded))) 
            key_fea_expanded = F.relu(self.bn42(self.conv42(key_fea_expanded))) ### key_fea
            key_fea_expanded = F.relu(self.bn43(self.conv43(key_fea_expanded))) ### 
            key_fea = torch.max(key_fea_expanded, dim=-1)[0]
        else:
            key_fea = torch.cat([key_fea, src_tar, key_pts.view(B * K, 3, 1)], 1)
            key_fea = F.relu(self.bn41(self.conv41(key_fea))) ### key_fea
            key_fea = F.relu(self.bn42(self.conv42(key_fea))) ### key_fea
            key_fea = F.relu(self.bn43(self.conv43(key_fea))) ### 

        key_fea = key_fea.view(B, K, 128).transpose(
            1, 2).unsqueeze(1).expand(-1, self.num_basis, -1, -1)
        key_pts2 = key_pts.view(B, K, 3).transpose(  ### key_pts2;
            1, 2).unsqueeze(1).expand(-1, self.num_basis, -1, -1)
        basis1 = basis.view(B, self.num_basis, K, 3).transpose(2, 3) ### basis and keypoints ###


        #### coefs prediction ####
        net = torch.cat([key_fea, basis1, key_pts2], 2).view( #### basis1, key_pts2
            B * self.num_basis, 3 + 128 + 3, K)

        net = F.relu(self.bn51(self.conv51(net)))
        net = F.relu(self.bn52(self.conv52(net)))
        net = self.bn53(self.conv53(net))

        net = torch.max(net, 2, keepdim=True)[0]
        net = net.view(B * self.num_basis, 128, 1)

        net = torch.cat([net, coef_range.view(B * self.num_basis, 2, 1)], 1)
        net = F.relu(self.bn61(self.conv61(net)))
        net = F.relu(self.bn62(self.conv62(net)))
        ### or just the generation of coef ---> basis of deformation combinations ### 
        
        ### basis: basis; sigmoid net
        coef = self.sigmoid(self.conv63(net)).view(B, self.num_basis) ### how to combine such basis... ### how to combine such basis...
        #### coefs prediction ####
        
        ### coef, coef_range ###
        ##### coefs.. ####
        coef = (coef * coef_range[:, :, 0] + (1 - coef)
                * coef_range[:, :, 1]).view(B, 1, self.num_basis)
        
        
        ### def_key_pts ####
        if self.pred_type == "basis":
            def_key_pts = key_pts + torch.bmm(coef, basis).view(B, K, 3)
        elif self.pred_type == "offset":
            def_key_pts = key_pts + basis[:, 0].view(B, K, 3) * coef[:, 0].unsqueeze(-1)
        else:
            raise ValueError(f"Unrecognized pred_type: {self.pred_type}.")
        
        ### 
        def_pc = torch.bmm(w_pc, def_key_pts)

        # print(f"def_pc: {def_pc.size()}, tar_pc: {tar_pc.size()}")
        cd_loss = chamfer_distance(def_pc, tar_pc)

        # ratio = torch.rand((B, self.num_basis)).cuda()
        ratio = torch.rand_like(coef)



        tot_sampled_def_key_pts = []
        for i_s in range(10):
            ratio = torch.rand((B, self.num_basis)).cuda()
            # print(f"coef_range: {coef_range.size()}, ratio: {ratio.size()}")
            sample_coef = (ratio * coef_range[:, :, 0] + (1 - ratio)
                        * coef_range[:, :, 1]).view(B, 1, self.num_basis)
            sample_def_key_pts = key_pts + \
                torch.bmm(sample_coef, basis).view(B, K, 3)
            tot_sampled_def_key_pts.append(sample_def_key_pts)
        
        
        sample_def_pc = torch.bmm(w_pc, sample_def_key_pts)


        sym_mult_tsr = torch.tensor([1, 1, 1]).cuda()
        sym_mult_tsr[self.symmetry_axis] = sym_mult_tsr[self.symmetry_axis] * -1.0 ### symmetry_axis for shapenet should be x-axis
        sample_def_pc_sym = sample_def_pc * sym_mult_tsr
        # \
        #     torch.tensor([-1, 1, 1]).cuda()  # for shapenet shapes
        # sample_def_pc_sym = sample_def_pc * \
        #     torch.tensor([1, -1, 1]).cuda()  # for shapenet shapes
        sym_loss = chamfer_distance(sample_def_pc, sample_def_pc_sym) #### sym_loss for the deformed shape
        return def_key_pts, def_pc, cd_loss, basis, coef, tot_sampled_def_key_pts, sym_loss, coef_range, loss_prior, src_pp_topk_idxes, tar_pp_topk_idxes







class model(nn.Module):
    def __init__(self, num_basis, opt=None):
        super(model, self).__init__()
        print("for network", opt)
        
        self.cat_deform_net = deform_model(num_basis=num_basis, opt=opt)
        self.opt = opt
        self.use_prob = self.opt.use_prob
        self.tar_basis = self.opt.tar_basis
        self.coef_multiplier = self.opt.coef_multiplier
        self.n_layers = self.opt.n_layers
        
        self.num_basis = num_basis
        # self.pred_offset = self.opt.pred_offset ### pred_offset ###
        self.pred_type = self.opt.pred_type
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
        
        self.tar_conv11 = torch.nn.Conv1d(self.pp_out_dim, 128, 1)
        self.tar_conv12 = torch.nn.Conv1d(128, 128, 1)
        self.tar_conv13 = torch.nn.Conv1d(128, 128, 1)
        self.tar_bn11 = nn.BatchNorm1d(128)
        self.tar_bn12 = nn.BatchNorm1d(128)
        self.tar_bn13 = nn.BatchNorm1d(128)
        
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
        self.dmt_syn_net = DMTETSynthesisNetwork( ### 
        #   w_dim=256, 
          w_dim=128,
          img_resolution=256, img_channels=3, device='cuda', geometry_type="conv3d", use_tri_plane=True, # tet_res=45 # num_ws=1
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


    #### src_pc: source shape and xxx ####
    #### source handle points... 
    #### for each basis: main keypoint; # select other keypoints nearby # select other keypoints nearby #
    def select_handles_via_radius(self, keypoints, radius=0.3, nearest_k=4): ### 
      ### keypoints: bsz x n_keypoints x 3 ###
      ### dists_keypts_keypts: bsz x n_keypts x n_keypts ###
      dists_keypts_keypts = torch.sum((keypoints.unsqueeze(2) - keypoints.unsqueeze(1)) ** 2, dim=-1) #### (n_key, n_key) ### pair-wise distance between different keypoints ####
      dists_larger_than_radius = (dists_keypts_keypts <= radius ** 2) ### dists_larger_than_radius: bsz x n_keypts x n_keypts ###
      ### keypoints ###
      ### a local handle --> select a local handle ###
      ### argsort --- 
      ### rnk_dists_keypts_keypts: bsz x n_keypts x n_keypts ### --> the orderj of each contextual point from each center point ####
      ### argsort ---> argsort argsort argsort
      rnk_dists_keypts_keypts = torch.argsort(dists_keypts_keypts, dim=-1, descending=False) #### ascending direction --> from nearest points to furtherest points ####
      rnk_dists_keypts_keypts = (rnk_dists_keypts_keypts <= nearest_k) ### rnk_
      ### bool values for neighouring keypts indicators ### --> from one keypoint to other keypoint ###
      neighbouring_keypts_indicators = ((dists_larger_than_radius.float() + rnk_dists_keypts_keypts.float()) > 1.5 ).float()
      return neighbouring_keypts_indicators
    
    #### local handles for local...
    def select_local_handles(self, bases, neighbouring_keypts_indicators):
      ### select_local_handles ###
      ### bases: bsz x n_bases x (n_keypts x 3) ###
      n_bases = bases.size(1)
      n_bsz, n_keypts = neighbouring_keypts_indicators.size(0), neighbouring_keypts_indicators.size(1)
      bases_exp = bases.contiguous().view(n_bsz, n_bases, n_keypts, 3).contiguous() #### bsz x n_bases x n_keypts x 3 ####
      bases_exp_norm = torch.norm(bases_exp, p=2, dim=-1) #### bsz x n_bases x n_keypts ###
      #### bases_largest_keypts_idxes: bsz x n_bases #### bsz x n_bases x n_keypts ###
      bases_largest_keypts_norm, bases_largest_keypts_idxes = torch.max(bases_exp_norm, dim=-1) ### 
      #### selected_valid_keypts_indicators: bsz x n_bases x n_keypts ---> indicator of whether those keypts are valid ####
      selected_valid_keypts_indicators = batched_index_select(values=neighbouring_keypts_indicators, indices=bases_largest_keypts_idxes, dim=1)
      valid_bases = bases_exp * selected_valid_keypts_indicators.unsqueeze(-1) #### bsz x n_bases x n_keypts x 3 #### ---> bsz x n_bases x (n_keypts x 3) #### 
      valid_bases = valid_bases.contiguous().view(n_bsz, n_bases, n_keypts * 3).contiguous()
      return valid_bases
    
    
    ### then select taregt keypts features via distances ###
    def select_target_local_features(self, keypoints_src, keypoints_tar, keypoints_tar_feats, neighbouring_tar_keypts_idxes=None, neighbouring_tar_k=2):
      #### keypoints_src: bsz x n_keypts x 3
      #### keypoints_tar: bsz x n_keypts x dim
      if neighbouring_tar_keypts_idxes is None:
        dists_keypts_src_tar = torch.sum((keypoints_src.unsqueeze(2) - keypoints_tar.unsqueeze(1)) ** 2, dim=-1) ######## bsz x n_keypts x n_keypts ######
        ### from dists to narest keypts ####
        # neighbouring_tar_rnk = torch.argsort(dists_keypts_src_tar, dim=-1, descending=False)
        ### neighbouring_tar_keypts_idxes: bsz x n_keypts x neighbouring_tar_k
        _, neighbouring_tar_keypts_idxes = torch.topk(dists_keypts_src_tar, k=neighbouring_tar_k, dim=-1, largest=False)
      ### neighbouring_tar_keypts_feature:  bsz x n_keypts x neighbouring_tar_k x dim ###
      neighbouring_tar_keypts_feature = batched_index_select(values=keypoints_tar_feats, indices=neighbouring_tar_keypts_idxes, dim=1)
      return neighbouring_tar_keypts_feature, neighbouring_tar_keypts_idxes
      
      
    def select_keypts_features(self, keypts, pc, pc_feats):
      # keypts: bsz x n_keypts x 3 
      # pc: bsz x n_pc x 3
      # pc_feats: bsz x n_pc x feat_dim
    #   print(f"keypts: {keypts.size()}, pc: {pc.size()}, pc_feats: {pc_feats.size()}")
      dists_keypts_pc = torch.sum((keypts.contiguous().unsqueeze(-2) - pc.unsqueeze(1)) ** 2, dim=-1) ### bsz x n_keypts x n_pc ###
      nearest_dists, nearest_pc_idxes = torch.min(dists_keypts_pc, dim=-1) ### bsz x n_keypts
      ### 
      ### keypts features via nearest neigbours ### ### pathc 
      keypts_feats = batched_index_select(values=pc_feats, indices=nearest_pc_idxes, dim=1) ### bsz x n_keypts x dim
      return keypts_feats
    
    
    
    
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
           

    def forward(self, src_pc, tar_pc, key_pts, dst_key_pts, w_pc,  src_verts, src_convex_pts, src_faces, src_edges, src_dofs, tar_verts, dst_convex_pts, tar_faces, tar_edges, tar_dofs, deform_net=None):
        ### src_convex_pts ---- only use convexes with the same idx in those two shapes ###
        ### src_convex_pts: bsz x n_cvxes x n_pts_per_convex 
        ### src_keypts: bsz x n_keypts x 3
        #### B, N, _
        #### add src_faces, tar_faces for src_meshes and tar_meshes #### 
        #### network target driven ####
        B, N, _ = src_pc.shape
        
        ### convex pts and ### convex pts and
        
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
        
        if self.opt.hier_stage == 1:
            with torch.no_grad(): ## space deformation net
                deform_net = deform_net # if deform_net is not None else self.cat_deform_net ## external deformation net ##
                if deform_net is not None:
                    def_key_pts, def_pc, cat_cd_loss, cat_basis, cat_coef, cat_tot_sampled_def_key_pts, cat_sym_loss, coef_range, loss_prior, src_pp_topk_idxes, tar_pp_topk_idxes = \
                    deform_net(src_pc, tar_pc, key_pts, dst_key_pts, w_pc,  src_verts, src_edges, src_dofs, tar_verts, tar_edges, tar_dofs)
                else:
                    def_key_pts = key_pts
                    def_pc = src_pc
        else:
            deform_net = deform_net if deform_net is not None else self.cat_deform_net
            
            def_key_pts, def_pc, cat_cd_loss, cat_basis, cat_coef, cat_tot_sampled_def_key_pts, cat_sym_loss, coef_range, loss_prior, src_pp_topk_idxes, tar_pp_topk_idxes = \
            deform_net(src_pc, tar_pc, key_pts, dst_key_pts, w_pc,  src_verts, src_edges, src_dofs, tar_verts, tar_edges, tar_dofs)

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
        
        
        if self.use_trans_range and not self.use_cond_vae and not self.use_cond_vae_cvx_feats:
            keypts_coefs = self.coef_pred_conv_nets(keypts_tot_cvx_feats.view(bsz * keypts_tot_cvx_feats.size(1), -1).unsqueeze(-1))
            keypts_coefs = keypts_coefs.squeeze(-1).view(bsz, -1, 3).contiguous()
            ### 
            keypts_rng = keypts_coefs[:, :, :2]
            keypts_coefs = keypts_coefs[:, :, 2:]
            
            ##### keypts_coefs and def_key_pts ####
            keypts_coefs = torch.sigmoid(keypts_coefs)
            keypts_rng[:, :, 0] = keypts_rng[:, :, 0] * -1.0
            keypts_rng = keypts_rng * self.coef_multiplier
            keypts_basis = keypts_basis / torch.clamp(torch.norm(keypts_basis, dim=-1, p=2, keepdim=True), min=1e-6)
            keypts_delta = (keypts_rng[:, :, 0:1] + (keypts_rng[:, :, 1: ] - keypts_rng[:, :, 0: 1]) * keypts_coefs) * keypts_basis
            def_key_pts = key_pts + keypts_delta
            ##### xxx #####
            
            rnd_keypts_coefs = torch.rand_like(keypts_coefs)
            rnd_keypts_delta = (keypts_rng[:, :, 0:1] + (keypts_rng[:, :, 1: ] - keypts_rng[:, :, 0: 1]) * rnd_keypts_coefs) * keypts_basis
            rnd_def_key_pts = key_pts + rnd_keypts_delta
        elif self.use_delta_prob:
            ### keypts_tot_cvx_feats
            ### normal_dists_statistics_pred_conv_nets
            keypts_dists_statistics = self.normal_dists_statistics_pred_conv_nets(keypts_tot_cvx_feats.view(bsz * keypts_tot_cvx_feats.size(1), -1).unsqueeze(-1))
            keypts_dists_statistics = keypts_dists_statistics.squeeze(-1).view(bsz, -1, 6).contiguous()
            keypts_mus, keypts_log_sigmas = keypts_dists_statistics[:, :, :3], keypts_dists_statistics[:, :, 3:]
            keypts_sigmas = torch.exp(keypts_log_sigmas) #### keypts_sigmas
            
            keypts_mus = keypts_mus.contiguous().transpose(-1, -2).contiguous()
            keypts_log_sigmas = keypts_log_sigmas.contiguous().transpose(-1, -2).contiguous()
            
            ''' TODO: check keypts_mus ---- its scales... '''
            keypts_z = utils.reparameterize_gaussian(mean=keypts_mus, logvar=keypts_log_sigmas)  # (B, F) ## global 
            ### zs probability ---> zs; standard normal logprob...
            keypts_log_pz = utils.standard_normal_logprob(keypts_z).sum(dim=1)  ### log_prob... ### ### standard gaussian! 
            ##### maximize the entropy of q_x(z); gaussian_entropy
            entropy = utils.gaussian_entropy(logvar=keypts_log_sigmas)
            # loss_prior = (-log_pz - entropy).mean()
            loss_prior = (-keypts_log_pz - entropy).mean() ### optimize loss_prir!
            ### minimize the negative log probability of zs ---- maximize the log probability of zs ### 
            
            loss_prior = utils.kld_loss(keypts_mus, keypts_log_sigmas).mean()
            # net = z
            
            delta_pred_in = torch.cat([rel_src_key_cvx_center.contiguous().transpose(-1, -2).contiguous(), keypts_z], dim=1)
            keypts_z = self.delta_pts_recon_net(delta_pred_in)
            keypts_z = keypts_z.contiguous().transpose(-1, -2).contiguous()
            
            
            keypts_delta = keypts_z * self.coef_multiplier
            def_key_pts = key_pts + keypts_delta
            
            
            rnd_keypts_z = torch.randn_like(keypts_z) ### randn_like keypts z
            delta_pred_in = torch.cat([rel_src_key_cvx_center, rnd_keypts_z], dim=-1)
            rnd_keypts_z = self.delta_pts_recon_net(delta_pred_in.contiguous().transpose(-1, -2).contiguous())
            rnd_keypts_z = rnd_keypts_z.contiguous().transpose(-1, -2).contiguous()
            rnd_keypts_delta = rnd_keypts_z * self.coef_multiplier
            rnd_def_key_pts = key_pts + rnd_keypts_delta
            
        elif self.use_cond_vae: ### per-point reconstruction loss...
            ### keypts_tot_cvx_feats
            ### normal_dists_statistics_pred
            keypts_tot_cvx_feats_in = torch.cat([keypts_tot_cvx_feats, src_keypts_offset], dim=-1) ### bsz x n_pts x (dim + dim + 6)
            
            keypts_dists_statistics = self.normal_dists_statistics_pred_conv_nets(keypts_tot_cvx_feats_in.view(bsz * keypts_tot_cvx_feats_in.size(1), -1).unsqueeze(-1))
            keypts_dists_statistics = keypts_dists_statistics.squeeze(-1).view(bsz, -1, 6).contiguous()
            keypts_mus, keypts_log_sigmas = keypts_dists_statistics[:, :, :3], keypts_dists_statistics[:, :, 3:]
            keypts_sigmas = torch.exp(keypts_log_sigmas) #### keypts_sigmas
            
            keypts_mus = keypts_mus.contiguous().transpose(-1, -2).contiguous()
            keypts_log_sigmas = keypts_log_sigmas.contiguous().transpose(-1, -2).contiguous()
            
            
            
            ''' TODO: check keypts_mus ---- its scales... '''
            keypts_z = utils.reparameterize_gaussian(mean=keypts_mus, logvar=keypts_log_sigmas)  # (B, F) ## global 
            ### zs probability ---> zs; standard normal logprob...
            keypts_log_pz = utils.standard_normal_logprob(keypts_z).sum(dim=1)  ### log_prob... ### ### standard gaussian! 
            ##### maximize the entropy of q_x(z); gaussian_entropy
            entropy = utils.gaussian_entropy(logvar=keypts_log_sigmas)
            # loss_prior = (-log_pz - entropy).mean()
            loss_prior = (-keypts_log_pz - entropy).mean() ### optimize loss_prir!
            ### minimize the negative log probability of zs ---- maximize the log probability of zs ### 
            
            loss_prior = utils.kld_loss(keypts_mus, keypts_log_sigmas).mean()
            # net = z
            delta_pred_in = torch.cat(
                [keypts_tot_cvx_feats.contiguous().transpose(-1, -2).contiguous(), keypts_z], dim=1
            )
            
            # delta_pred_in = torch.cat([rel_src_key_cvx_center.contiguous().transpose(-1, -2).contiguous(), keypts_z], dim=1)
            keypts_z = self.delta_pts_recon_net(delta_pred_in)
            keypts_z = keypts_z.contiguous().transpose(-1, -2).contiguous()
            
            
            ### pc_z: bsz x n_pc x 6
            if self.use_trans_range:
                keypts_delta_dir = keypts_z[:, :, :3]
                keypts_delta_rng = keypts_z[:, :, 3: 5]
                keypts_delta_coefs = keypts_z[:, :, 5:]
                # print(f"keypts_z: {keypts_z.size()}, keypts_delta_dir: {keypts_delta_dir.size()}, keypts_delta_rng: {keypts_delta_rng.size()}, keypts_delta_coefs: {keypts_delta_coefs.size()}, ")
                keypts_delta_dir = keypts_delta_dir / torch.clamp(torch.norm(keypts_delta_dir, dim=-1, p=2, keepdim=True), min=1e-6)
                # keypts_delta_rng[:, :, 0] = keypts_delta_rng[:, :, 0] * -1.0
                keypts_delta_rng = torch.cat(
                    [keypts_delta_rng[:, :, 0: 1] * -1.0, keypts_delta_rng[:, :, 1: ]], dim=-1
                )
                keypts_delta_coefs = torch.sigmoid(keypts_delta_coefs)
                keypts_z = (keypts_delta_rng[:, :, 0:1] + (keypts_delta_rng[:, :, 1: ] - keypts_delta_rng[:, :, 0: 1]) * keypts_delta_coefs) * keypts_delta_dir
            
            
            keypts_delta = keypts_z # * self.coef_multiplier
            
            
            recon_loss = torch.sum((keypts_delta - src_keypts_offset) ** 2, dim=-1).mean()
            
            loss_prior += recon_loss / self.opt.kl_weight
            
            entropy = recon_loss #### get recon_loss ####
            
            def_key_pts = key_pts + keypts_delta
            
            
            rnd_keypts_z = torch.randn_like(keypts_z) ### randn_like keypts z
            # delta_pred_in = torch.cat([rel_src_key_cvx_center, rnd_keypts_z], dim=-1)
            delta_pred_in = torch.cat(
                [keypts_tot_cvx_feats, rnd_keypts_z], dim=-1
            )
            
            
            rnd_keypts_z = self.delta_pts_recon_net(delta_pred_in.contiguous().transpose(-1, -2).contiguous())
            rnd_keypts_z = rnd_keypts_z.contiguous().transpose(-1, -2).contiguous()
            
            ### pc_z: bsz x n_pc x 6
            if self.use_trans_range:
                rnd_keypts_delta_dir = rnd_keypts_z[:, :, :3]
                rnd_keypts_delta_rng = rnd_keypts_z[:, :, 3: 5]
                rnd_keypts_delta_coefs = rnd_keypts_z[:, :, 5:]
                rnd_keypts_delta_dir = rnd_keypts_delta_dir / torch.clamp(torch.norm(rnd_keypts_delta_dir, dim=-1, p=2, keepdim=True), min=1e-6)
                # rnd_keypts_delta_rng[:, :, 0] = rnd_keypts_delta_rng[:, :, 0] * -1.0
                rnd_keypts_delta_rng = torch.cat(
                    [rnd_keypts_delta_rng[:, :, 0: 1] * -1.0, rnd_keypts_delta_rng[:, :, 1: ]], dim=-1
                )
                rnd_keypts_delta_coefs = torch.sigmoid(rnd_keypts_delta_coefs)
                rnd_keypts_z = (rnd_keypts_delta_rng[:, :, 0:1] + (rnd_keypts_delta_rng[:, :, 1: ] - rnd_keypts_delta_rng[:, :, 0: 1]) * rnd_keypts_delta_coefs) * rnd_keypts_delta_dir
            
            rnd_keypts_delta = rnd_keypts_z # * self.coef_multiplier
            rnd_def_key_pts = key_pts + rnd_keypts_delta
                       
        else:
            keypts_basis = keypts_basis * self.coef_multiplier
            def_key_pts = key_pts + keypts_basis
            
            rnd_def_key_pts = def_key_pts
        ''' Keypts to cvx features '''
        
        
        ''' pc to cvx features '''
        pc_tot_cvx_feats = torch.cat(
          [pc_cvx_feats, pc_dst_cvx_feats], dim=-1 ### pc-dst-cvx-features
        )
        src_pc_cvx_center = batched_index_select(values=src_cvx_center, indices=pc_cvx_idx, dim=1) ### bsz x n_keypts x 3
        rel_src_pc_cvx_center = sampled_src_pc - src_pc_cvx_center ### bsz x n_keypts x 3

        pc_tot_cvx_feats = torch.cat(
          [pc_tot_cvx_feats, rel_src_pc_cvx_center], dim=-1 ### bsz x n_keypts x (dim + dim + 3)
        )
        ### bsz x (in_feat_dim) x n_keyptss
        pc_basis = self.basis_pred_conv_nets(pc_tot_cvx_feats.view(bsz * pc_tot_cvx_feats.size(1), -1).unsqueeze(-1))
        pc_basis = pc_basis.squeeze(-1).view(bsz, -1, 3).contiguous() ### pc_basis
        ''' pc to cvx features '''
        
        flow_feats_in = src_pc_offset ### flow_feats_in ### ### sampled_s 
          ## sampled_src_pc: bsz x n_sampled x 3; pc_cvx: bsz x n_cvx x n_pts x 3
          
        if self.cond_tar_pc:
            dist_sampled_pc_cvx_pts = torch.sum((sampled_tar_pc.unsqueeze(2).unsqueeze(2) - dst_convex_pts.unsqueeze(1)) ** 2, dim=-1) 
            dist_sampled_pc_cvx, _ = torch.min(dist_sampled_pc_cvx_pts, dim=-1) ### bsz x n_sampled x n_cvx ### 
            ### src_pc: bsz x n_pts x 3
            dist_src_pc_pc = torch.sum((tar_pc.unsqueeze(2) - tar_pc.unsqueeze(1)) ** 2, dim=-1) ### bsz x n_pts x n_pts ###
            dist_src_pc_pc = dist_src_pc_pc + torch.eye((dist_src_pc_pc.size(1))).unsqueeze(0).cuda() * 999999.0
            dist_src_pc_pc, _ = torch.min(dist_src_pc_pc, dim=-1) ### bsz x n_pts ### 
            dist_src_pc_pc = float(dist_src_pc_pc.mean())
            
            thres = dist_src_pc_pc * 10
            # thres = 0.001
            pc_cvx_indicator = (dist_sampled_pc_cvx <= thres).float()
            ### flow_out: bsz x dim x n_pts ### ### context... ### context... ###
            flow_out, _ = self.flow_encoding_net(sampled_tar_pc, feats=None) ### bsz x n_pts x 3 ---> bsz x dim x n_pts ---> per-point features...
        else:
            # dist_sampled_pc_cvx_pts = torch.sum((sampled_src_pc.unsqueeze(2).unsqueeze(2) - src_convex_pts.unsqueeze(1)) ** 2, dim=-1) 
            dist_sampled_pc_cvx_pts = torch.sum((sampled_src_pc.unsqueeze(2).unsqueeze(2) - src_convex_pts.unsqueeze(1)) ** 2, dim=-1) 
            dist_sampled_pc_cvx, _ = torch.min(dist_sampled_pc_cvx_pts, dim=-1) ### bsz x n_sampled x n_cvx ### 
            ### src_pc: bsz x n_pts x 3
            dist_src_pc_pc = torch.sum((src_pc.unsqueeze(2) - src_pc.unsqueeze(1)) ** 2, dim=-1) ### bsz x n_pts x n_pts ###
            dist_src_pc_pc = dist_src_pc_pc + torch.eye((dist_src_pc_pc.size(1))).unsqueeze(0).cuda() * 999999.0
            dist_src_pc_pc, _ = torch.min(dist_src_pc_pc, dim=-1) ### bsz x n_pts ### 
            dist_src_pc_pc = float(dist_src_pc_pc.mean())
            

            thres = dist_src_pc_pc * 10
            # thres = 0.001
            pc_cvx_indicator = (dist_sampled_pc_cvx <= thres).float()

            ### flow_out: bsz x dim x n_pts ### ### context... ### context... ### ### flow feats
            flow_out, _ = self.flow_encoding_net(sampled_src_pc, feats=flow_feats_in) ### bsz x n_pts x 3 ---> bsz x dim x n_pts ---> per-point features...
          

        
        cvx_indicator = (pc_cvx_indicator.sum(1) > 0.5).float() ## bsz x n_cvx --> whether has pts in 

        
        ### dmt net for 
        ''' DMT net for pre-convex implicit filed prediction and mesh extraction ''' 
        n_cvx = src_convex_pts.size(1)
        ### flow_out_expanded ###
        flow_out_expanded = flow_out.unsqueeze(-1).repeat(1, 1, 1, n_cvx) ### bsz x dim x n_pts x n_cvx
        #### pc_cvx_indicators ---> flow_out
        flow_out_expanded = flow_out_expanded - (1. - pc_cvx_indicator).unsqueeze(1) *  999999.0
        cvx_flow_out = torch.max(flow_out_expanded, dim=2)[0] ### bsz x dim x n_cvx ### cvx_flow_out
        cvx_flow_out = self.flow_out_conv_net(cvx_flow_out) ### bsz x dim x n_cvx ### 
        
        if self.cond_tar_pc:
            #### from flow-out to its variational encodings ####
            cvx_flow_out = torch.cat( ### cvx_flow_out ###
                [cvx_flow_out, tar_cvx_out.contiguous().transpose(-1, -2).contiguous()], dim=1
            ) ### with src_cvx_out ### flow-net
        else:
            #### from flow-out to its variational encodings ####
            cvx_flow_out = torch.cat( ### cvx_flow_out ###
                [cvx_flow_out, src_cvx_out.contiguous().transpose(-1, -2).contiguous()], dim=1
            ) ### with src_cvx_out ### flow-net
        #### 
        
        
        
        cvx_flow_stats = self.normal_dists_statistics_pred_conv_nets_flow_net(cvx_flow_out) ### bsz x 
        
        cvx_flow_mus, cvx_flow_log_sigmas = cvx_flow_stats[:, : cvx_flow_stats.size(1) // 2, :], cvx_flow_stats[:, cvx_flow_stats.size(1) // 2: , :] 
        cvx_flow_z = utils.reparameterize_gaussian(mean=cvx_flow_mus, logvar=cvx_flow_log_sigmas) ### cvx_flow_z: bsz x dim x n_cvx
        
        
        log_pz = utils.standard_normal_logprob(cvx_flow_z).sum(dim=1)  ### log_prob... ### ### standard gaussian! 

        entropy = utils.gaussian_entropy(logvar=cvx_flow_log_sigmas)

        loss_prior = utils.kld_loss(cvx_flow_mus, cvx_flow_log_sigmas).mean() 
        
        if self.cond_tar_pc:
            rnd_cvx_flow_z = torch.randn_like(cvx_flow_z)
            if self.recon_cond == "cvx":
                cvx_flow_z_for_recon = torch.cat(
                    [cvx_flow_z, tar_cvx_out.contiguous().transpose(-1, -2).contiguous()], dim=1
                )
                rnd_cvx_flow_z_for_recon = torch.cat( 
                    [rnd_cvx_flow_z, tar_cvx_out.contiguous().transpose(-1, -2).contiguous()], dim=1
                )
            elif self.recon_cond == "bbox":
                ### tar_cvx_pts: bsz x n_cvx x n_pts x 3
                maxx_tar_cvx_pts, _ = torch.max(dst_convex_pts, dim=2)
                minn_tar_cvx_pts, _ = torch.min(dst_convex_pts, dim=2) ### bsz x n_cvx x 3 
                extends_tar_cvx_pts = maxx_tar_cvx_pts - minn_tar_cvx_pts
                cvx_flow_z_for_recon = torch.cat(
                    [cvx_flow_z, extends_tar_cvx_pts.contiguous().transpose(-1, -2).contiguous()], dim=1
                )
                rnd_cvx_flow_z_for_recon = torch.cat( ### rnd_cvx_flow_z_for_recon ###
                    [rnd_cvx_flow_z, extends_tar_cvx_pts.contiguous().transpose(-1, -2).contiguous()], dim=1
                )
            else:
                cvx_flow_z_for_recon = cvx_flow_z
                rnd_cvx_flow_z_for_recon = rnd_cvx_flow_z
        else:
            cvx_flow_z_for_recon = torch.cat(
                [cvx_flow_z, src_cvx_out.contiguous().transpose(-1, -2).contiguous()], dim=1
            )
            
            rnd_cvx_flow_z = torch.randn_like(cvx_flow_z)
            rnd_cvx_flow_z_for_recon = torch.cat( ### rnd_cvx_flow_z_for_recon ###
                [rnd_cvx_flow_z, src_cvx_out.contiguous().transpose(-1, -2).contiguous()], dim=1
            )
            
            
        n_cvx = cvx_flow_out.size(-1)
        



        
        
        tot_sdf_reg_loss = 0. #### 
        ### bsz x (dim + dim) x n_cvx ### 
        tot_cvx_vs = [[] for i_bsz in range(bsz)]
        tot_cvx_fs = [[] for i_bsz in range(bsz)]
        
        
        tot_recon_pts = []
        
        rnd_tot_cvx_vs = [[] for i_bsz in range(bsz)]
        rnd_tot_cvx_fs = [[] for i_bsz in range(bsz)]

        
        for i_cvx in range(n_cvx):
        #   cur_cvx_flow_out = cvx_flow_out[:, :, i_cvx]
          cur_cvx_flow_out = cvx_flow_z_for_recon[:, :, i_cvx]
          cur_rnd_cvx_flow_out = rnd_cvx_flow_z_for_recon[:, :, i_cvx]
          
          #### cvx_flow_out ####
          cur_cvx_flow_out = cur_cvx_flow_out.contiguous().unsqueeze(1).repeat(1, self.num_ws + 4 + 1, 1)
          
          v_list, f_list, sdf, deformation, v_deformed, sdf_reg_loss = self.dmt_syn_net.get_geometry_prediction(cur_cvx_flow_out, sdf_feature=None)
          
          
          cur_rnd_cvx_flow_out = cur_rnd_cvx_flow_out.contiguous().unsqueeze(1).repeat(1, self.num_ws + 4 + 1, 1)
          
          rnd_v_list, rnd_f_list, _, _, _, rnd_sdf_reg_loss = self.dmt_syn_net.get_geometry_prediction(cur_rnd_cvx_flow_out, sdf_feature=None)
          
          tot_cur_cvx_pts = []
          
          for cur_bsz, (cur_v, cur_f) in enumerate(zip(v_list, f_list)):
              
            cur_rnd_v = rnd_v_list[cur_bsz]
            
            
            cur_bsz_cvx_pts = dst_convex_pts[cur_bsz, i_cvx] ### n_pts x 3
            cur_bsz_cvx_dst_scale = utils.get_vertices_scale_torch_no_batch(cur_bsz_cvx_pts)
            
            
            
            cur_bsz_recon_vert_scale = utils.get_vertices_scale_torch_no_batch(cur_v) ### b
            
            # cur_v = cur_v / torch.clamp(cur_bsz_recon_vert_scale, min=1e-6)
            
            cur_v = utils.normalie_pc_bbox_batched(cur_v.unsqueeze(0)).squeeze(0)
            
            cur_v = cur_v * cur_bsz_cvx_dst_scale
            
            cur_v = cur_v + src_cvx_center[cur_bsz, i_cvx].unsqueeze(0)
            
            cur_rnd_v = utils.normalie_pc_bbox_batched(cur_rnd_v.unsqueeze(0)).squeeze(0)
            
            cur_rnd_v = cur_rnd_v * cur_bsz_cvx_dst_scale
            
            cur_rnd_v = cur_rnd_v + src_cvx_center[cur_bsz, i_cvx].unsqueeze(0)
            
            
            # cur_bsz_rnd_recon_vert_scale = utils.get_vertices_scale_torch_no_batch(cur_rnd_v)
            
            # cur_rnd_v = cur_rnd_v / torch.clamp(cur_bsz_rnd_recon_vert_scale, min=1e-6)
            
            # cur_rnd_v = cur_rnd_v * cur_bsz_cvx_dst_scale
            
            # cur_rnd_v = cur_rnd_v + src_cvx_center[cur_bsz, i_cvx].unsqueeze(0)
            
            # maxx_cur_v, _ = torch.max(cur_v, dim=0)
            # minn_cur_v, _ = torch.min(cur_v, dim=0)
            # # print(f"bsz: {cur_bsz}, cvx: {i_cvx}, cur_v: {cur_v.size()}, cur_f: {cur_f.size()}, maxx_v: {maxx_cur_v}, minn_v: {minn_cur_v}")
            
            cur_bsz_cur_cvx_indi = float(cvx_indicator[cur_bsz, i_cvx].item())
            if cur_bsz_cur_cvx_indi > 0.5:
                tot_cvx_vs[cur_bsz].append(cur_v)
                tot_cvx_fs[cur_bsz].append(cur_f)
                
                rnd_tot_cvx_vs[cur_bsz].append(cur_rnd_v)
                rnd_tot_cvx_fs[cur_bsz].append(rnd_f_list[cur_bsz])
            
            cur_cvx_sampled_pts = cur_v
            # cur_cvx_sampled_pts = model_utils.sample_pts_from_mesh(cur_v, cur_f, npoints=256) ### bsz x n_cvx_pts x 3
            cur_cvx_sampled_pts_fps = farthest_point_sampling(cur_cvx_sampled_pts.unsqueeze(0), n_sampling=512)
            cur_cvx_sampled_pts = cur_cvx_sampled_pts[cur_cvx_sampled_pts_fps]
            
            tot_cur_cvx_pts.append(cur_cvx_sampled_pts.unsqueeze(0))
          tot_cur_cvx_pts = torch.cat(tot_cur_cvx_pts, dim=0) ## bsz x 
          tot_recon_pts.append(tot_cur_cvx_pts.unsqueeze(1))
          tot_sdf_reg_loss += sdf_reg_loss
        tot_sdf_reg_loss /= float(n_cvx)
        
        
        
        tot_recon_meshes = []
        tot_rnd_recon_meshes = []
        for i_bsz in range(len(tot_cvx_vs)):
          cur_bsz_cvx_vs, cur_bsz_cvx_fs = tot_cvx_vs[i_bsz], tot_cvx_fs[i_bsz]
          cur_bsz_vs, cur_bsz_fs = model_utils.merge_meshes_torch(cur_bsz_cvx_vs, cur_bsz_cvx_fs)
          tot_recon_meshes.append((cur_bsz_vs, cur_bsz_fs))
          
          
          cur_bsz_rnd_cvx_vs, cur_bsz_rnd_cvx_fs = rnd_tot_cvx_vs[i_bsz], rnd_tot_cvx_fs[i_bsz]
          if len(cur_bsz_rnd_cvx_vs) > 0:
              cur_bsz_rnd_vs, cur_bsz_rnd_fs = model_utils.merge_meshes_torch(cur_bsz_rnd_cvx_vs, cur_bsz_rnd_cvx_fs)
          else:
              cur_bsz_rnd_vs = cur_bsz_vs
              cur_bsz_rnd_fs = cur_bsz_fs
          tot_rnd_recon_meshes.append((cur_bsz_rnd_vs, cur_bsz_rnd_fs))
        
        cvx_recon_pts = torch.cat(tot_recon_pts, dim=1) ### bszx n_cvx_pts x 3
        
        # cvx_recon_pts: bsz x n_cvx x n_pts x 3
        if torch.any(torch.isnan(cvx_recon_pts)):
            print(f"nan with cvx_recon_pts!!!!")
        
        
        # if self.cond_tar_pc:
        #     cvx_recon_pts = cvx_recon_pts + src_cvx_center.unsqueeze(-2) ### bsz x n_cvx xxx
        # else:
        #     cvx_recon_pts = cvx_recon_pts + src_cvx_center.unsqueeze(-2) ### bsz x n_cvx xxx
        
        ### pts loss for each cvx; overall cd loss ###
        ### src_pc_tar_pc: bsz x n_pts x 3  ## src
        
        if self.cond_tar_pc: ### sampled_tar_pc...
            dist_src_pc_tar_pc_recon_cvx_tot = torch.sum((sampled_tar_pc.unsqueeze(2).unsqueeze(2) - cvx_recon_pts.unsqueeze(1)) ** 2, dim=-1) ### bsz x n_pts x n_cvx x n_recon_pts
        else:
            dist_src_pc_tar_pc_recon_cvx_tot = torch.sum((src_pc_tar_pc.unsqueeze(2).unsqueeze(2) - cvx_recon_pts.unsqueeze(1)) ** 2, dim=-1) ### bsz x n_pts x n_cvx x n_recon_pts
        
        
        dist_src_pc_tar_pc_recon_cvx, _ = torch.min(dist_src_pc_tar_pc_recon_cvx_tot, dim=-1) ### bsz x n_pts x n_cvx
        
        dist_src_pc_tar_pc_recon_cvx = dist_src_pc_tar_pc_recon_cvx + (1. - cvx_indicator).unsqueeze(1) * 99999.0 ### remove no cvxes
        
        dist_src_pc_tar_pc_recon_cvx, _ = torch.min(dist_src_pc_tar_pc_recon_cvx, dim=-1) ### bsz x n_pts ### to the nearest cvx only? 
        
        
        dist_pc_recon_cvx_src_pc_tar_pc, _ = torch.min(dist_src_pc_tar_pc_recon_cvx_tot, dim=1) ## sz x n_cvx x n_reconpts ### 
        dist_pc_recon_cvx_src_pc_tar_pc = torch.mean(dist_pc_recon_cvx_src_pc_tar_pc, dim=-1) * cvx_indicator
        
        dist_pc_recon_cvx_src_pc_tar_pc = torch.mean(dist_pc_recon_cvx_src_pc_tar_pc, dim=-1).mean() ## a loss vector 
        dist_src_pc_tar_pc_recon_cvx  = torch.mean(dist_src_pc_tar_pc_recon_cvx, dim=-1).mean() ### a loss vecotr
        
        if dist_pc_recon_cvx_src_pc_tar_pc.item() > 20.0:
            dist_pc_recon_cvx_src_pc_tar_pc = torch.zeros((1,), dtype=torch.float32).cuda().mean()
        
        if dist_src_pc_tar_pc_recon_cvx.item() > 20.0:
            dist_src_pc_tar_pc_recon_cvx = torch.zeros((1,), dtype=torch.float32).cuda().mean()
        
        ### cvx_recon_pts : bsz x n_cvx x n_pts x 3
        ### cvx_recon_pts...
        # pc_loss_prior = dist_pc_recon_cvx_src_pc_tar_pc + dist_src_pc_tar_pc_recon_cvx
        pc_entropy = dist_pc_recon_cvx_src_pc_tar_pc
        pc_log_pz = dist_src_pc_tar_pc_recon_cvx
        
        cd_loss = dist_pc_recon_cvx_src_pc_tar_pc + dist_src_pc_tar_pc_recon_cvx ### reconstruction loss
            
        # print(f"cd_loss: {cd_loss}, tot_sdf_reg_loss: {tot_sdf_reg_loss}")

        # print(f"current net: {net.size()}")
        # loss_prior = torch.zeros((1,), dtype=torch.float32).cuda().mean()
        loss_log_pz = torch.zeros((1,), dtype=torch.float32).cuda().mean()
        loss_entropy = torch.zeros((1,), dtype=torch.float32).cuda().mean()


        rt_dict = {
          "cd_loss": cd_loss, 
          "tot_sdf_reg_loss": tot_sdf_reg_loss, 
          "tot_recon_meshes": tot_recon_meshes, 
          "cvx_recon_pts": cvx_recon_pts, 
          "dst_convex_pts": dst_convex_pts, 
          "loss_prior": loss_prior, 
          "loss_log_pz": loss_log_pz, 
          "loss_entropy": loss_entropy, 
          "rnd_tot_recon_meshes": tot_rnd_recon_meshes
        }
        


        return rt_dict

