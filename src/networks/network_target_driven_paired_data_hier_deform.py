from multiprocessing.sharedctypes import Value
import torch
import torch.nn.functional as F
import torch.nn as nn
from pointnet_utils import pointnet_encoder
# from losses import chamfer_distance
from losses import chamfer_distance_raw as chamfer_distance
import utils
from pointnet2 import PointnetPP
import edge_propagation

from common_utils.data_utils_torch import farthest_point_sampling, batched_index_select
from common_utils.data_utils_torch import compute_normals_o3d, get_vals_via_nearest_neighbours
from scipy.optimize import linear_sum_assignment

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
        
        print(f"whether to use pp_tar_out_feat: {self.use_pp_tar_out_feat}")
        
        self.neighbouring_tar_k = 8
        self.neighbouring_tar_k = 1
        
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
                
                # self.pointnet=
                # self.pp_out_dim = 128 + 1024
            else: ### pp_out_dim...
                self.pointnet = pointnet_encoder()
                self.tar_pointnet = pointnet_encoder()
                self.pp_out_dim = 2883
        else:
            self.graph_conv_net = edge_propagation.GraphConvNet(n_layers=self.n_layers)
            self.tar_graph_conv_net = edge_propagation.GraphConvNet(n_layers=self.n_layers)
            self.pp_out_dim = 128 ### 128

        # # src point feature 2883 * N ### feature sapce... generative model..
        # self.conv11 = torch.nn.Conv1d(self.pp_out_dim, 128, 1)
        # self.conv12 = torch.nn.Conv1d(128, 64, 1)
        # self.conv13 = torch.nn.Conv1d(64, 64, 1)
        # self.bn11 = nn.BatchNorm1d(128)
        # self.bn12 = nn.BatchNorm1d(64)
        # self.bn13 = nn.BatchNorm1d(64)
        
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
            
            
      
      ### bsp net and how to 

    def forward_bak(self, src_pc, tar_pc, key_pts, dst_key_pts, w_pc,  src_verts, src_edges, src_dofs, tar_verts, tar_edges, tar_dofs, deform_net=None):
        #### B, N, _ 
        #### network target driven ####
        B, N, _ = src_pc.shape
        
        if self.opt.hier_stage == 1:
            with torch.no_grad():
                # print("here!")
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


        cat_def_key_pts =def_key_pts.clone()
        cat_def_pc = def_pc.clone()
          
        src_pc = def_pc
        key_pts = def_key_pts #### def_kty_ptss ###
        
        ### downsample pc ###
        # n_samples = 512
        # n_samples = 1024
        n_samples = self.n_samples
        bsz, N = src_pc.size(0), src_pc.size(1)
        src_fps_idx = farthest_point_sampling(pos=src_pc[:, :, :3], n_sampling=n_samples) ### smapling for features xxx ### sampel pts for features ###
        ### src_pc: bsz x N x 3
        src_pc_downsampled = src_pc.contiguous().view(bsz * N, 3)[src_fps_idx].contiguous().view(bsz, n_samples, 3).contiguous()
        
        # w_pc: bsz x N x K ### w_pc: bsz x N x K
        w_pc_expand = w_pc.contiguous().view(bsz * N, -1)[src_fps_idx].contiguous().view(bsz, n_samples, -1).contiguous()
        
        ### tar_fps_idx ###
        tar_fps_idx = farthest_point_sampling(pos=tar_pc[:, :, :3], n_sampling=n_samples) ### smapling for features xxx
        ### src_pc: bsz x N x 3
        tar_pc_downsampled = tar_pc.contiguous().view(bsz * N, 3)[tar_fps_idx].contiguous().view(bsz, n_samples, 3).contiguous()
        
        
        
        N = n_samples
        # bz x n_samples x pos_dim
        
        if not self.use_graphconv:
            if not self.use_pp_tar_out_feat:
                #### source out; source global ####
                src_out, src_global = self.pointnet(src_pc, False)
                #### target global #### #### tar_pc #####
                # tar_out, tar_global = self.pointnet(tar_pc, False) ### target pc ##3 flow predition and composition? 
                tar_out, tar_global = self.tar_pointnet(tar_pc, False)
                
                src_pp_topk_idxes = None
                tar_pp_topk_idxes = None
            else:
                ##### src_pp_topk_idxes ##### original-pointnet #####
                # src_out, src_global, src_pp_topk_idxes = self.pointnet(None, src_pc, False)
                # tar_out, tar_global, tar_pp_topk_idxes = self.tar_pointnet(None, tar_pc, False)
                
                # src_out, src_pp_topk_idxes = self.pointnet(src_pc)
                # tar_out, tar_pp_topk_idxes = self.tar_pointnet(tar_pc)

                src_out, src_pp_topk_idxes = self.pointnet(src_pc_downsampled)
                tar_out, tar_pp_topk_idxes = self.tar_pointnet(tar_pc_downsampled)
                # print(f"src_global: {src_global.size()}, tar_global: {tar_global.size()}")
        else: ### src_mesh_features ###
            src_mesh_features = self.graph_conv_net(src_verts, src_edges, src_dofs)
            src_pc_features = edge_propagation.get_keypoints_features(src_pc, src_verts, src_mesh_features)
            src_out = src_pc_features
            src_global = torch.max(src_out, dim=-1)[0]
            
            tar_mesh_features = self.tar_graph_conv_net(tar_verts, tar_edges, tar_dofs)
            tar_pc_features = edge_propagation.get_keypoints_features(tar_pc, tar_verts, tar_mesh_features)
            tar_out = tar_pc_features
            tar_global = torch.max(tar_out, dim=-1)[0]
            
            src_pp_topk_idxes = None
            tar_pp_topk_idxes = None
        
        
        src_out = F.relu(self.bn11(self.conv11(src_out)))
        src_out = F.relu(self.bn12(self.conv12(src_out)))
        src_out = F.relu(self.bn13(self.conv13(src_out))) #### src_pp_features #### src_out ####
        ### network and keypoints 

        _, K, _ = key_pts.shape
        key_pts1 = key_pts.unsqueeze(-1).expand(-1, -1, -1, N)  # B K 3 N
        #### w_pc --> w_pc1 --> transpose weights for further computation
        
        #### using no downssample ###
        # w_pc1 = w_pc.transpose(2, 1).unsqueeze(2)  # B K 1 N
        #### for downsample ####  ### w_pc_expand --> for pc weights ###
        w_pc1 = w_pc_expand.transpose(2, 1).unsqueeze(2) # w_pc_expand #
        
        
        ''' with all source pts features '''
        # src_out = src_out.unsqueeze(1).expand(-1, K, -1, -1)  # B K 64 N
        # ### what should be cat here --> src_out; w_pc1; key_pts1; 
        # ### for each keypoint, 1) keypoints-to-surface_points weights; 2) key_points values; 3) expanded source points 
        
        # # w_pc1, key_pts1; keypoints, source out, weights from keypoints to pcs 
        # net = torch.cat([src_out, w_pc1, key_pts1], 2).view(B * K, 68, N) #### 

        # net = F.relu(self.bn21(self.conv21(net)))
        # net = self.bn22(self.conv22(net)) ### with only keypts features ###
        ''' with all source pts features '''
        
        
        ''' only keypts features '''
        net = self.select_keypts_features(key_pts, src_pc_downsampled, src_out.contiguous().transpose(-1, -2).contiguous()).transpose(-1, -2).view(B * K, -1).unsqueeze(-1) #### net for keypts ####
        ''' only keypts features '''
        
        
        if self.tar_basis > 0: # use target for deformation basis prediction ###
            ##### get tar out #####
            ### first set of convoltuions, conv11, conv12, conv13
            # tar_out = torch.max(tar_out, dim=2, keepdim=True)[0] ###
            tar_out = F.relu(self.tar_bn11(self.tar_conv11(tar_out)))
            tar_out = F.relu(self.tar_bn12(self.tar_conv12(tar_out)))
            tar_out = F.relu(self.tar_bn13(self.tar_conv13(tar_out))) ## tar_out: bsz x dim x N
            
            if not self.use_pp_tar_out_feat:
                tar_out = torch.max(tar_out, dim=2, keepdim=True)[0] ###
            ##### get tar_out feautres #####
            
            if self.use_prob:
                #### tar_out: B x dim x N
                tar_out_dim = tar_out.size(1)
                ### 
                tar_out_mu, tar_out_log_sigma = tar_out[:, :tar_out_dim // 2], tar_out[:, tar_out_dim // 2: ]
                z = utils.reparameterize_gaussian(mean=tar_out_mu, logvar=tar_out_log_sigma)  # (B, F) ## global 
                log_pz = utils.standard_normal_logprob(z).sum(dim=1) 
                ##### maximize the entropy of q_x(z); gaussian_entropy #####
                entropy = utils.gaussian_entropy(logvar=tar_out_log_sigma) ### 
                loss_prior = (-log_pz - entropy).mean()
                # kl_weight = 0.001
                # loss_prior = kl_weigh
                tar_out = z
            else:
                tar_out_dim = tar_out.size(1)
                tar_out_mu, tar_out_log_sigma = tar_out[:, :tar_out_dim // 2], tar_out[:, tar_out_dim // 2: ]
                tar_out = tar_out_mu
                loss_prior = torch.zeros((1,), dtype=torch.float32).cuda().mean()
        
            # tar_out_sample
            ### each point ###
            # net = torch.max(net, 2, keepdim=True)[0] ### feature for each keypoint 
            # ### net: bsz x K x dim
            # net = net.contiguous().view(B, K, net.size(-2)) ### bsz x K x dim
            # tar_out = tar_out.unsqueeze(1).expand(-1, K, -1, -1)  ### tar_out: B x K x 64 x N
            # net = net.contiguous().unsqueeze(-1).repeat(1, 1, 1, N).contiguous() ### bsz x K x dim x N
            # # print(f"net: {net.size()}, tar_out: {tar_out.size()}")
            # net = torch.cat([net, tar_out], dim=2).view(B * K, -1, N) ### bsz x K x (dim + 64) x N
            # net = F.relu(self.tar_bn21(self.tar_conv21(net)))
            # net = self.tar_bn22(self.tar_conv22(net)) ### bsz x K x 64 x N
            ### each point ###
            
            # net = torch.max(net, 2, keepdim=True)[0]
            ### global feature ###
            ### tar_out: B x dim x N ###
            
            net = torch.max(net, 2, keepdim=True)[0] ### feature for each keypoint 
            net = net.contiguous().view(B, K, net.size(-2)) ### bsz x K x dim
            ### tar_out for deformation direction ### ### target_out ### 
            ### target out; key_pts and tar_key_pts ...###
            
            # --> tar_out : bsz x n_pts x feat_dim 
            tar_out_trans = tar_out.contiguous().transpose(-1, -2).contiguous()
            ### deformation factorization ### ### tar_out: bsz x n_pts x feat_dim ###
            tar_keypts_out = self.select_keypts_features(dst_key_pts, tar_pc_downsampled, tar_out_trans) #### bsz x ## tar keypts out...
            ### selected_tar_keypts_out: bsz x K x neighbouring_k x feat_dim ###
            ### key_pts: bsz x n_keypts x 3;
            selected_tar_keypts_out, neighbouring_tar_keypts_idxes = self.select_target_local_features(key_pts, dst_key_pts, tar_keypts_out, neighbouring_tar_keypts_idxes=None, neighbouring_tar_k=16)
            ### selected_tar_keypts_out: bsz x K x feat_dim x neighbouring_k ###
            selected_tar_keypts_out = selected_tar_keypts_out.contiguous().transpose(2, 3).contiguous() 
            
            ### selected_tar_keypts_out

            tar_out = tar_out.unsqueeze(1).expand(-1, K, -1, -1)  ### tar_out: B x K x 64 x N; tar_out; tar_out.
            
            if not self.use_pp_tar_out_feat:
                #### reparamterize this term to make it like VAE?
                net = net.contiguous().unsqueeze(-1).repeat(1, 1, 1, 1).contiguous() ### bsz x K x dim x N
                net = torch.cat([net, tar_out], dim=2).view(B * K, -1, 1) ### bsz x K x (dim + 64) x N
            else:
                # net = net.contiguous().unsqueeze(-1).repeat(1, 1, 1, N).contiguous() ### bsz x K x dim x N
                # net = torch.cat([net, tar_out], dim=2).view(B * K, -1, N) ### bsz x K x (dim + 64) x N
                ##### only for nearesst neighbooring_k features #####
                net = net.contiguous().unsqueeze(-1).repeat(1, 1, 1, selected_tar_keypts_out.size(-1)).contiguous() ### bsz x K x dim x N
                net = torch.cat([net, selected_tar_keypts_out], dim=2).view(B * K, -1, selected_tar_keypts_out.size(-1)) ### bsz x K x (dim + 64) x N
                ##### only for nearesst neighbooring_k features #####

            # net = torch.cat([net, tar_out], dim=2).view(B * K, -1, 1) ### bsz x K x (dim + 64) x N
            net = F.relu(self.tar_bn21(self.tar_conv21(net))) ### net and net...
            net = self.tar_bn22(self.tar_conv22(net)) ### bsz x K x 64 x N
            ### global feature ###
        else:
            tar_out = F.relu(self.tar_bn11(self.tar_conv11(tar_out)))
            tar_out = F.relu(self.tar_bn12(self.tar_conv12(tar_out)))
            tar_out = F.relu(self.tar_bn13(self.tar_conv13(tar_out)))
            if not self.use_pp_tar_out_feat:
                tar_out = torch.max(tar_out, dim=2, keepdim=True)[0] ###
            tar_out_dim = tar_out.size(1)
            tar_out_mu, tar_out_log_sigma = tar_out[:, :tar_out_dim // 2], tar_out[:, tar_out_dim // 2: ]
            tar_out = tar_out_mu
            
            # --> tar_out : bsz x n_pts x feat_dim 
            tar_out_trans = tar_out.contiguous().transpose(-1, -2).contiguous()
            ### deformation factorization ### ### tar_out: bsz x n_pts x feat_dim ###
            tar_keypts_out = self.select_keypts_features(dst_key_pts, tar_pc_downsampled, tar_out_trans) #### bsz x ## tar keypts out...
            ### selected_tar_keypts_out: bsz x K x neighbouring_k x feat_dim ###
            ### key_pts: bsz x n_keypts x 3;
            selected_tar_keypts_out, neighbouring_tar_keypts_idxes = self.select_target_local_features(key_pts, dst_key_pts, tar_keypts_out, neighbouring_tar_keypts_idxes=None, neighbouring_tar_k=16)
            ### selected_tar_keypts_out: bsz x K x feat_dim x neighbouring_k ###
            selected_tar_keypts_out = selected_tar_keypts_out.contiguous().transpose(2, 3).contiguous() 
                
            
            tar_out = tar_out.unsqueeze(1).expand(-1, K, -1, -1) 
            loss_prior = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            
        
        
        ### key feat
        ##### pool net for glboal features ######
        ### net and netout ###
        net = torch.max(net, 2, keepdim=True)[0] ### net with tar_out
        ### 
        key_fea = net.view(B * K, 64, 1) ### reshape to keypoints' features ### key_features 

        #### keypoints, keypoint basis, keypoints basis; num_basis x 3 ####
        if not self.wo_keypts_abs:
            net = torch.cat([key_fea, key_pts.view(B * K, 3, 1)], 1) #### key_points #### keypoints
        else:
            net = key_fea
        net = F.relu(self.bn31(self.conv31(net)))
        net = F.relu(self.bn32(self.conv32(net)))
        
        basis = self.conv33(net).view(B, K * 3, self.num_basis).transpose(1, 2)
        
        ### how to place each keypts 
        ###### 
        radius = 90.0
        neighbouring_keypts_indicators = self.select_handles_via_radius(key_pts, radius=radius, nearest_k=8)
        ### neighbouring_keypts_indicators: bsz x n_keypts x n_keypts
        # print(neighbouring_keypts_indicators[0, 0,])
        basis = self.select_local_handles(basis, neighbouring_keypts_indicators) #### basis: bsz x num_basisx (K * 3) ### local features...
        
        # print(torch.norm(basis, p=2, dim=-1))
        
        
        if self.pred_type == "basis":
            basis = basis / torch.clamp(basis.norm(p=2, dim=-1, keepdim=True), min=1e-6)
        elif self.pred_type == "offset":
            basis = basis.contiguous().view(B, self.num_basis, K, 3).contiguous()
            basis = basis / torch.clamp(basis.norm(p=2, dim=-1, keepdim=True), min=1e-5)
            basis = basis.contiguous().view(B, self.num_basis, K * 3).contiguous() ##### basis with unit length-vectors
        else:
            raise ValueError(f"Unrecognized pred_type: {self.pred_type}.")
          
        

        key_fea_range = key_fea.view(
            B, K, 64, 1).expand(-1, -1, -1, self.num_basis).transpose(1, 3) #### B, self.num_basis, K, 64
        key_pts_range = key_pts.view(
            B, K, 3, 1).expand(-1, -1, -1, self.num_basis).transpose(1, 3) 
        basis_range = basis.view(B, self.num_basis, K, 3).transpose(2, 3)


        if self.pred_type == "basis":
            ### get range ### coef range.. ###
            coef_range = torch.cat([key_fea_range, key_pts_range, basis_range], 2).view(
                B * self.num_basis, 70, K)
            coef_range = F.relu(self.bn71(self.conv71(coef_range))) ### coef_range
            coef_range = F.relu(self.bn72(self.conv72(coef_range))) ### coef_range and ...
            coef_range = self.conv73(coef_range)
            coef_range = torch.max(coef_range, 2, keepdim=True)[0]
            # coef_range = coef_range.view(B, self.num_basis, 2) * 0.02 ### 
            coef_range = coef_range.view(B, self.num_basis, 2) * self.coef_multiplier #  0.1 
            # coef_range = coef_range.view(B, self.num_basis, 2) * 0.5
            coef_range[:, :, 0] = coef_range[:, :, 0] * -1 #### coef_range...
        elif self.pred_type == "offset":
            coef_range = torch.cat([key_fea_range, key_pts_range, basis_range], 2).view(
                B * self.num_basis, 70, K)
            coef_range = F.relu(self.bn71(self.conv71(coef_range))) ### coef_range
            coef_range = F.relu(self.bn72(self.conv72(coef_range))) ### coef_range and ...
            coef_range = self.conv73(coef_range)
            coef_range = coef_range.view(B, self.num_basis, 2, K) * self.coef_multiplier #  0.1 
            coef_range = coef_range.contiguous().transpose(-1, -2).contiguous()
            coef_range[:, :, :, 0] = coef_range[:, :, :, 0] * -1 #### coef_range: B, num_basis, K, 2
        else:
            raise ValueError(f"Unrecognized pred_type: {self.pred_type}.")

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
            
            # key_fea_expanded = key_fea.contiguous().repeat(1, 1, N).contiguous()
            
            key_fea_expanded = key_fea.contiguous().repeat(1, 1, selected_tar_keypts_out.size(-1)).contiguous()
            # tar_out: B fea_dim N
            # print(f"tar_out: {tar_out.size()}")
            # tar_out_expanded = tar_out.contiguous().unsqueeze(1).contiguous().repeat(1, K, 1, 1).contiguous().view(B * K, tar_out.size(1), N)
            
            # tar_out_expanded = tar_out.contiguous().view(B * K, tar_out.size(1), N)
            
            selected_tar_keypts_out = selected_tar_keypts_out.contiguous().view(B * K, selected_tar_keypts_out.size(-2), selected_tar_keypts_out.size(-1)) #### selected_tar_keypts_out...
            tar_out_expanded = selected_tar_keypts_out ###### seelcted_tar_keypts_out #######
            
            
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
        
        
        
        if self.pred_type == "basis":
            coef = (coef * coef_range[:, :, 0] + (1 - coef)
                    * coef_range[:, :, 1]).view(B, 1, self.num_basis)
        elif self.pred_type == "offset":
            coef = (coef * coef_range[:, :, :, 0] + (1 - coef)
                    * coef_range[:, :, :, 1])
        else:
            raise ValueError(f"Unrecognized pred_type: {self.pred_type}.")

        ### basis: --> B x n_basis x K x 3; coef: B x 1 x n_basis ---> just a linear conbination of basis handles
        ### basis
        
        
        if self.pred_type == "basis":
            def_key_pts = key_pts + torch.bmm(coef, basis).view(B, K, 3)
        elif self.pred_type == "offset":
            def_key_pts = key_pts + basis[:, 0].view(B, K, 3) * coef[:, 0].unsqueeze(-1)
        else:
            raise ValueError(f"Unrecognized pred_type: {self.pred_type}.")
        
        def_pc = torch.bmm(w_pc, def_key_pts)

        # print(f"def_pc: {def_pc.size()}, tar_pc: {tar_pc.size()}")
        cd_loss = chamfer_distance(def_pc, tar_pc)

        # ratio = torch.rand((B, self.num_basis)).cuda()
        ratio = torch.rand_like(coef)


        if self.pred_type == "basis":
            tot_sampled_def_key_pts = []
            for i_s in range(10):
                ratio = torch.rand((B, self.num_basis)).cuda()
                # print(f"coef_range: {coef_range.size()}, ratio: {ratio.size()}")
                sample_coef = (ratio * coef_range[:, :, 0] + (1 - ratio)
                            * coef_range[:, :, 1]).view(B, 1, self.num_basis)
                sample_def_key_pts = key_pts + \
                    torch.bmm(sample_coef, basis).view(B, K, 3)
                tot_sampled_def_key_pts.append(sample_def_key_pts)
        elif self.pred_type == "offset":
            
            sample_coef = (ratio * coef_range[..., 0] + (1 - ratio)
                        * coef_range[..., 1])
            sample_def_key_pts = key_pts + basis[:, 0].view(B, K, 3) * sample_coef[:, 0].unsqueeze(-1)
        else:
            raise ValueError(f"Unrecognized pred_type: {self.pred_type}.")

        sample_def_pc = torch.bmm(w_pc, sample_def_key_pts)

        # sample_def_pc_sym = sample_def_pc * \
        #     torch.tensor([-1, 1, 1]).cuda()  # for shapenet shapes
        sym_mult_tsr = torch.tensor([1, 1, 1]).cuda()
        sym_mult_tsr[self.symmetry_axis] = sym_mult_tsr[self.symmetry_axis] * -1.0
        sample_def_pc_sym = sample_def_pc * sym_mult_tsr
        # sample_def_pc_sym = sample_def_pc * \
        #     torch.tensor([1, -1, 1]).cuda()  # for shapenet shapes
        sym_loss = chamfer_distance(sample_def_pc, sample_def_pc_sym) #### sym_loss for the deformed shape
        
        
        # def_key_pts, def_pc, cat_cd_loss, cat_basis, cat_coef, tot_sampled_def_key_pts, cat_sym_loss, 
        
        
        # cd_loss = cd_loss + cat_cd_loss
        
        if deform_net is None:
            cat_cd_loss = cd_loss.clone()
            cat_basis = basis.clone()
            cat_coef = coef.clone()
            cat_tot_sampled_def_key_pts = tot_sampled_def_key_pts
            cat_sym_loss = sym_loss.clone()
        
        
        return def_key_pts, def_pc, cd_loss, basis, coef, tot_sampled_def_key_pts, sym_loss, coef_range, loss_prior, src_pp_topk_idxes, tar_pp_topk_idxes, \
          cat_def_key_pts, cat_def_pc, cat_cd_loss, cat_basis, cat_coef, cat_tot_sampled_def_key_pts, cat_sym_loss

    def forward(self, src_pc, tar_pc, key_pts, dst_key_pts, w_pc,  src_verts, src_faces, src_edges, src_dofs, tar_verts, tar_faces, tar_edges, tar_dofs, deform_net=None):
        #### B, N, _ 
        #### add src_faces, tar_faces for src_meshes and tar_meshes #### 
        #### network target driven ####
        B, N, _ = src_pc.shape
        
        if self.opt.hier_stage == 1:
            with torch.no_grad():
                # print("here!")
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


        cat_def_key_pts =def_key_pts.clone()
        cat_def_pc = def_pc.clone()
          
        src_pc = def_pc
        key_pts = def_key_pts
        
        ### downsample pc ###
        # n_samples = 512
        # n_samples = 1024
        n_samples = self.n_samples
        bsz, N = src_pc.size(0), src_pc.size(1)
        
        
        
        
        src_fps_idx = farthest_point_sampling(pos=src_pc[:, :, :3], n_sampling=n_samples) ### smapling for features xxx ### sampel pts for features ###
        ### src_pc: bsz x N x 3; src 
        src_pc_downsampled = src_pc.contiguous().view(bsz * N, 3)[src_fps_idx].contiguous().view(bsz, n_samples, 3).contiguous()
        
        # w_pc: bsz x N x K ### w_pc: bsz x N x K
        w_pc_expand = w_pc.contiguous().view(bsz * N, -1)[src_fps_idx].contiguous().view(bsz, n_samples, -1).contiguous()
        
        ### tar_fps_idx ###
        tar_fps_idx = farthest_point_sampling(pos=tar_pc[:, :, :3], n_sampling=n_samples) ### smapling for features xxx
        ### src_pc: bsz x N x 3
        tar_pc_downsampled = tar_pc.contiguous().view(bsz * N, 3)[tar_fps_idx].contiguous().view(bsz, n_samples, 3).contiguous()
        
        
        
        N = n_samples
        # bz x n_samples x pos_dim
        
        #### pointnet and tar_pointnet ####
        src_out, src_pp_topk_idxes = self.pointnet(src_pc_downsampled)
        tar_out, tar_pp_topk_idxes = self.tar_pointnet(tar_pc_downsampled)
        
        src_out = F.relu(self.bn11(self.conv11(src_out)))
        src_out = F.relu(self.bn12(self.conv12(src_out)))
        # src_out = F.relu(self.bn13(self.conv13(src_out))) #### src_pp_features #### src_out ####
        src_out = self.bn13(self.conv13(src_out)) #### src_pp_features #### src_out ####
        
        
        #### for prob calculation ####
        prob_src_out, prob_src_pp_topk_idxes = self.prob_pointnet(src_pc_downsampled)
        prob_tar_out, prob_tar_pp_topk_idxes = self.tar_prob_pointnet(tar_pc_downsampled)
        
        prob_src_out = self.prob_src_out_conv_net(prob_src_out)
        # prob_tar_out = self.prob_tar_out_conv_net(prob_tar_out)
        #### for prob calculation ####
        
        
        ### network and keypoints 

        _, K, _ = key_pts.shape
        key_pts1 = key_pts.unsqueeze(-1).expand(-1, -1, -1, N)  # B K 3 N
        #### w_pc --> w_pc1 --> transpose weights for further computation
        
        #### using no downssample ###
        # w_pc1 = w_pc.transpose(2, 1).unsqueeze(2)  # B K 1 N
        #### for downsample ####  ### w_pc_expand --> for pc weights ###
        w_pc1 = w_pc_expand.transpose(2, 1).unsqueeze(2) # w_pc_expand #
        
        
        ''' with all source pts features '''
        # src_out = src_out.unsqueeze(1).expand(-1, K, -1, -1)  # B K 64 N
        # ### what should be cat here --> src_out; w_pc1; key_pts1; 
        # ### for each keypoint, 1) keypoints-to-surface_points weights; 2) key_points values; 3) expanded source points 
        
        # # w_pc1, key_pts1; keypoints, source out, weights from keypoints to pcs 
        # net = torch.cat([src_out, w_pc1, key_pts1], 2).view(B * K, 68, N) #### 

        # net = F.relu(self.bn21(self.conv21(net)))
        # net = self.bn22(self.conv22(net)) ### with only keypts features ###
        ''' with all source pts features '''
        
        
        ''' only keypts features ''' ### bsz x n_keypts x dim ###
        net = self.select_keypts_features(key_pts, src_pc_downsampled, src_out.contiguous().transpose(-1, -2).contiguous()).view(B * K, -1).unsqueeze(-1) 
        # .transpose(-1, -2).view(B * K, -1).unsqueeze(-1) #### bsz x n_keypts x dim ####
        ''' only keypts features '''
        
        prob_net = self.select_keypts_features(key_pts, src_pc_downsampled, prob_src_out.contiguous().transpose(-1, -2).contiguous()).view(B * K, -1).unsqueeze(-1) 
        
        # if self.use_prob:
        if self.use_prob_src:
            # net_out_dim = net.size(1)
            net_out_dim = prob_net.size(1)
            # net_out_mu, net_out_log_sigma = net[:, : net_out_dim // 2], net[:, net_out_dim // 2: ] ## target f
            net_out_mu, net_out_log_sigma = prob_net[:, : net_out_dim // 2], prob_net[:, net_out_dim // 2: ]
            z = utils.reparameterize_gaussian(mean=net_out_mu, logvar=net_out_log_sigma)  # (B, F) ## global 
            ### zs probability ---> zs; standard normal logprob...
            log_pz = utils.standard_normal_logprob(z).sum(dim=1)  ### log_prob... ###
            ##### maximize the entropy of q_x(z); gaussian_entropy
            entropy = utils.gaussian_entropy(logvar=net_out_log_sigma) ### 
            loss_prior = (-log_pz - entropy).mean()
            ### minimize the negative log probability of zs ---- maximize the log probability of zs ### 
            # net = z
            prob_net = z
            
            loss_log_pz = (-log_pz).mean()
            loss_entropy = (-entropy).mean()
        else:
            # net_out_dim = net.size(1)
            net_out_dim = prob_net.size(1)
            # net_out_mu, net_out_log_sigma = net[:, : net_out_dim // 2], net[:, net_out_dim // 2: ]
            prob_net_out_mu, prob_net_out_log_sigma = prob_net[:, : net_out_dim // 2], prob_net[:, net_out_dim // 2: ]
            # net = net_out_mu
            prob_net = prob_net_out_mu
            
            # net_out_dim = net.size(1)
            net_out_dim = net.size(1)
            # net_out_mu, net_out_log_sigma = net[:, : net_out_dim // 2], net[:, net_out_dim // 2: ]
            net_out_mu, net_out_log_sigma = net[:, : net_out_dim // 2], net[:, net_out_dim // 2: ]
            # net = net_out_mu
            net = net_out_mu
            loss_prior = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            loss_log_pz = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            loss_entropy = torch.zeros((1,), dtype=torch.float32).cuda().mean()
        
        
        if self.tar_basis > 0: # use target for deformation basis prediction ###
            
            prob_tar_out = self.prob_tar_out_conv_net(prob_tar_out)
            
            ##### get tar out ##### ### tar_basis 
            ### first set of convoltuions, conv11, conv12, conv13
            # tar_out = torch.max(tar_out, dim=2, keepdim=True)[0] ###
            tar_out = F.relu(self.tar_bn11(self.tar_conv11(tar_out)))
            tar_out = F.relu(self.tar_bn12(self.tar_conv12(tar_out)))
            # tar_out = F.relu(self.tar_bn13(self.tar_conv13(tar_out))) ## tar_out: bsz x dim x N
            tar_out = self.tar_bn13(self.tar_conv13(tar_out)) ## tar_out: bsz x dim x N
            
            
            
            if not self.use_pp_tar_out_feat:
                tar_out = torch.max(tar_out, dim=2, keepdim=True)[0] ###
            ##### get tar_out feautres #####
            
            ######## use_probs ###########
            # if self.use_prob:
            #     #### tar_out: B x dim x N
            #     tar_out_dim = tar_out.size(1)
            #     ### 
            #     tar_out_mu, tar_out_log_sigma = tar_out[:, :tar_out_dim // 2], tar_out[:, tar_out_dim // 2: ]
            #     z = utils.reparameterize_gaussian(mean=tar_out_mu, logvar=tar_out_log_sigma)  # (B, F) ## global 
            #     #### normal probability of z ####
            #     log_pz_tar = utils.standard_normal_logprob(z).sum(dim=1) 
            #     ##### maximize the entropy of q_x(z); gaussian_entropy #####
            #     entropy_tar = utils.gaussian_entropy(logvar=tar_out_log_sigma) ### 
            #     ##### log probability of pz and entropy #####
            #     loss_prior_tar = (-log_pz_tar - entropy_tar).mean()
            #     # kl_weight = 0.001
            #     # loss_prior = kl_weigh
            #     tar_out = z
                
            #     loss_log_pz_tar = (-log_pz_tar).mean()
            #     loss_entropy_tar = (-entropy_tar).mean()
            # else:
                
            tar_out_dim = prob_tar_out.size(1)
            tar_out_mu, tar_out_log_sigma = prob_tar_out[:, :tar_out_dim // 2], prob_tar_out[:, tar_out_dim // 2: ]
            prob_tar_out = tar_out_mu
            loss_prior_tar = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            loss_log_pz_tar = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            loss_entropy_tar = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            ######## use_probs ##############
        
        
            # tar_out_sample
            ### each point ###
            # net = torch.max(net, 2, keepdim=True)[0] ### feature for each keypoint 
            # ### net: bsz x K x dim
            # net = net.contiguous().view(B, K, net.size(-2)) ### bsz x K x dim
            # tar_out = tar_out.unsqueeze(1).expand(-1, K, -1, -1)  ### tar_out: B x K x 64 x N
            # net = net.contiguous().unsqueeze(-1).repeat(1, 1, 1, N).contiguous() ### bsz x K x dim x N
            # # print(f"net: {net.size()}, tar_out: {tar_out.size()}")
            # net = torch.cat([net, tar_out], dim=2).view(B * K, -1, N) ### bsz x K x (dim + 64) x N
            # net = F.relu(self.tar_bn21(self.tar_conv21(net)))
            # net = self.tar_bn22(self.tar_conv22(net)) ### bsz x K x 64 x N
            ### each point ###
            
            # net = torch.max(net, 2, keepdim=True)[0]
            ### global feature ###
            ### tar_out: B x dim x N ###
            
            ''' When using the same conv network for features '''
            # net = torch.max(net, 2, keepdim=True)[0] ### feature for each keypoint 
            # net = net.contiguous().view(B, K, net.size(-2)) ### bsz x K x dim
            # ### tar_out for deformation direction ### ### target_out ### 
            # ### target out; key_pts and tar_key_pts ...###
            
            # # tar_out : bsz x n_pts x feat_dim; 
            # tar_out_trans = tar_out.contiguous().transpose(-1, -2).contiguous()
            # ### deformation factorization ### ### tar_out: bsz x n_pts x feat_dim ###
            # tar_keypts_out = self.select_keypts_features(dst_key_pts, tar_pc_downsampled, tar_out_trans)
            # ### selected_tar_keypts_out: bsz x K x neighbouring_k x feat_dim ###
            # ### key_pts: bsz x n_keypts x 3; #### select target keypts out ###
            ''' When using the same conv network for features '''
            
            net = torch.max(net, 2, keepdim=True)[0] ### feature for each keypoint 
            net = net.contiguous().view(B, K, net.size(-2)) ### bsz x K x dim
            
            prob_net = torch.max(prob_net, 2, keepdim=True)[0] ### feature for each keypoint 
            prob_net = prob_net.contiguous().view(B, K, prob_net.size(-2)) ### bsz x K x dim
            ### tar_out for deformation direction ### ### target_out ### 
            ### target out; key_pts and tar_key_pts ...###
            
            # tar_out : bsz x n_pts x feat_dim; 
            tar_out_trans = prob_tar_out.contiguous().transpose(-1, -2).contiguous()
            ### deformation factorization ### ### tar_out: bsz x n_pts x feat_dim ###
            tar_keypts_out = self.select_keypts_features(dst_key_pts, tar_pc_downsampled, tar_out_trans)
            ### selected_tar_keypts_out: bsz x K x neighbouring_k x feat_dim ###
            ### key_pts: bsz x n_keypts x 3; #### select target keypts out ###
            
            
            
            # if self.use_prob:
            #     #### tar_out: B x dim x N
            #     tar_out_dim = tar_keypts_out.size(-1)
            #     ### 
            #     tar_out_mu, tar_out_log_sigma = tar_out[:, :tar_out_dim // 2], tar_out[:, tar_out_dim // 2: ]
            #     z = utils.reparameterize_gaussian(mean=tar_out_mu, logvar=tar_out_log_sigma)  # (B, F) ## global 
            #     log_pz = utils.standard_normal_logprob(z).sum(dim=1) 
            #     ##### maximize the entropy of q_x(z); gaussian_entropy #####
            #     entropy = utils.gaussian_entropy(logvar=tar_out_log_sigma) ### 
            #     loss_prior = (-log_pz - entropy).mean()
            #     # kl_weight = 0.001
            #     # loss_prior = kl_weigh
            #     tar_out = z
            # else:
            #     tar_out_dim = tar_out.size(1)
            #     tar_out_mu, tar_out_log_sigma = tar_out[:, :tar_out_dim // 2], tar_out[:, tar_out_dim // 2: ]
            #     tar_out = tar_out_mu
            #     loss_prior = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            
            ''' select via nearest neighbours directly ''' 
            ##### for target keypts features #####
            selected_tar_keypts_out, neighbouring_tar_keypts_idxes = self.select_target_local_features(key_pts, dst_key_pts, tar_keypts_out, neighbouring_tar_keypts_idxes=None, neighbouring_tar_k=self.neighbouring_tar_k)
            ### selected_tar_keypts_out: bsz x K x feat_dim x neighbouring_k ###
            selected_tar_keypts_out = selected_tar_keypts_out.contiguous().transpose(2, 3).contiguous() 
            ''' select via nearest neighbours directly ''' 
            
            ''' select via euclidean and normal distances '''
            # selected_tar_keypts_indices = self.get_correspondances(key_pts, dst_key_pts, src_verts, src_faces, tar_verts, tar_faces)
            # ### selected_tar_keypts_indices: bsz x n_keypts ###
            # # tar_keypts_out: bsz x n_keypts x dim; bsz x n_keypts x 1 x dim
            # selected_tar_keypts_out = batched_index_select(values=tar_keypts_out, indices=selected_tar_keypts_indices.unsqueeze(-1), dim=1) ### bsz x n_keypts x 1 x dim ###
            # selected_tar_keypts_out = selected_tar_keypts_out.contiguous().transpose(-1, -2).contiguous()
            ''' select via euclidean and normal distances '''
            
            
            tar_out = tar_out.unsqueeze(1).expand(-1, K, -1, -1)
            
            prob_tar_out = prob_tar_out.unsqueeze(1).expand(-1, K, -1, -1)
            
            # B x K x 64 x N #
            # net = torch.cat([net, tar_out], dim=2).view(B * K, -1, N) ### bsz x K x (dim + 64) x N
            
            
            ##### only for nearesst neighbooring_k features #####
            net = net.contiguous().unsqueeze(-1).repeat(1, 1, 1, selected_tar_keypts_out.size(-1)).contiguous() ### bsz x K x dim x N
            
            prob_net = prob_net.contiguous().unsqueeze(-1).repeat(1, 1, 1, selected_tar_keypts_out.size(-1)).contiguous() ### bsz x K x dim x N
            ##### only for nearesst neighbooring_k features #####
            
        
            if self.use_prob:
                net_for_trans = torch.cat([prob_net, selected_tar_keypts_out], dim=2).view(B * K, -1, selected_tar_keypts_out.size(-1)) ### bsz x K x (dim + 64) x N
                
                prob_trans_tar_feats = self.prob_conv(net_for_trans.contiguous().view(B * K, -1, net_for_trans.size(-1)))
                prob_trans_nn_feats = prob_trans_tar_feats.size(1)
                prob_trans_mu, prob_trans_log_sigma = prob_trans_tar_feats[:, : prob_trans_nn_feats // 2, ], prob_trans_tar_feats[:, prob_trans_nn_feats // 2: ]
                z = utils.reparameterize_gaussian(mean=prob_trans_mu, logvar=prob_trans_log_sigma)  # (B, F) ## global 
                #### normal probability of z ####
                log_pz_tar = utils.standard_normal_logprob(z).sum(dim=1) 
                ##### maximize the entropy of q_x(z); gaussian_entropy #####
                entropy_tar = utils.gaussian_entropy(logvar=prob_trans_log_sigma) ### 
                ##### log probability of pz and entropy #####
                loss_prior_tar = (-log_pz_tar - entropy_tar).mean()
                # kl_weight = 0.001
                # loss_prior = kl_weigh
                # selected_tar_keypts_out = z #### reparameterized selected tar_out_features j#####
                
                if self.opt.with_recon:
                    ### recon_tar_features: bsz x dim x n_tar_neis ###
                    recon_tar_features = self.prob_tar_recon_net(z)
                    recon_loss = torch.sum((selected_tar_keypts_out - recon_tar_features) ** 2, dim=-2).mean() 
                    loss_prior_tar += recon_loss
                    selected_tar_keypts_out = recon_tar_features
                else:
                    selected_tar_keypts_out = z
                
                rnd_selected_tar_keypts_out = torch.randn_like(selected_tar_keypts_out)
                
                if self.opt.with_recon:
                    rnd_selected_tar_keypts_out = self.prob_tar_recon_net(rnd_selected_tar_keypts_out)
                    
            
                selected_tar_keypts_out = selected_tar_keypts_out.contiguous().view(B, K, -1, selected_tar_keypts_out.size(-1)).contiguous()
                
                rnd_selected_tar_keypts_out = rnd_selected_tar_keypts_out.contiguous().view(B, K, -1, rnd_selected_tar_keypts_out.size(-1)).contiguous()
                
                ##### only for nearesst neighbooring_k features #####
                ### log_pz_tar ###
                loss_log_pz_tar = (-log_pz_tar).mean()
                loss_entropy_tar = (-entropy_tar).mean()
                if self.opt.with_recon:
                    loss_log_pz_tar = recon_loss.mean()
            else:
                rnd_selected_tar_keypts_out = selected_tar_keypts_out
                loss_prior_tar = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                loss_log_pz_tar = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                loss_entropy_tar = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                
            rnd_net = torch.cat([net, rnd_selected_tar_keypts_out], dim=2).view(B * K, -1, rnd_selected_tar_keypts_out.size(-1)) ### bsz x K x (dim + 64) x N

            rnd_net = F.relu(self.tar_bn21(self.tar_conv21(rnd_net)))
            rnd_net = self.tar_bn22(self.tar_conv22(rnd_net)) ### bsz x K x 64 x N

            net = torch.cat([net, selected_tar_keypts_out], dim=2).view(B * K, -1, selected_tar_keypts_out.size(-1)) ### bsz x K x (dim + 64) x N

            net = F.relu(self.tar_bn21(self.tar_conv21(net)))
            net = self.tar_bn22(self.tar_conv22(net)) ### bsz x K x 64 x N
            
        else:
            tar_out = F.relu(self.tar_bn11(self.tar_conv11(tar_out)))
            tar_out = F.relu(self.tar_bn12(self.tar_conv12(tar_out)))
            tar_out = F.relu(self.tar_bn13(self.tar_conv13(tar_out)))
            if not self.use_pp_tar_out_feat:
                tar_out = torch.max(tar_out, dim=2, keepdim=True)[0] ###
            tar_out_dim = tar_out.size(1)
            tar_out_mu, tar_out_log_sigma = tar_out[:, :tar_out_dim // 2], tar_out[:, tar_out_dim // 2: ]
            tar_out = tar_out_mu
            
            # --> tar_out : bsz x n_pts x feat_dim 
            tar_out_trans = tar_out.contiguous().transpose(-1, -2).contiguous()
            ### deformation factorization ### ### tar_out: bsz x n_pts x feat_dim ###
            tar_keypts_out = self.select_keypts_features(dst_key_pts, tar_pc_downsampled, tar_out_trans) #### bsz x ## tar keypts out...
            ### selected_tar_keypts_out: bsz x K x neighbouring_k x feat_dim ###
            ### key_pts: bsz x n_keypts x 3;
            
            ''' select via nearest neighbours directly ''' 
            selected_tar_keypts_out, neighbouring_tar_keypts_idxes = self.select_target_local_features(key_pts, dst_key_pts, tar_keypts_out, neighbouring_tar_keypts_idxes=None, neighbouring_tar_k=self.neighbouring_tar_k)
            ### selected_tar_keypts_out: bsz x K x feat_dim x neighbouring_k ###
            selected_tar_keypts_out = selected_tar_keypts_out.contiguous().transpose(2, 3).contiguous() 
            ''' select via nearest neighbours directly '''
            
            #### euclidean and normal distances ###
            ''' select via euclidean and normal distances '''
            # selected_tar_keypts_indices = self.get_correspondances(key_pts, dst_key_pts, src_verts, src_faces, tar_verts, tar_faces)
            # ### selected_tar_keypts_indices: bsz x n_keypts ###
            # # tar_keypts_out: bsz x n_keypts x dim; bsz x n_keypts x 1 x dim
            # selected_tar_keypts_out = batched_index_select(values=tar_keypts_out, indices=selected_tar_keypts_indices.unsqueeze(-1), dim=1) ### bsz x n_keypts x 1 x dim ###
            # selected_tar_keypts_out = selected_tar_keypts_out.contiguous().transpose(-1, -2).contiguous()
            ''' select via euclidean and normal distances '''
                
            
            tar_out = tar_out.unsqueeze(1).expand(-1, K, -1, -1) 
            loss_prior_tar = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            loss_log_pz_tar = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            loss_entropy_tar = torch.zeros((1,), dtype=torch.float32).cuda().mean()
        
        if not self.use_prob_src:
            loss_prior = loss_prior_tar
            loss_log_pz = loss_log_pz_tar
            loss_entropy = loss_entropy_tar #### loss_entropy and tar... ###

        ### key feat
        ##### pool net for glboal features ######
        ### net and netout ###
        net = torch.max(net, 2, keepdim=True)[0] ### net with tar_out
        ### 
        key_fea = net.view(B * K, 64, 1) ### reshape to keypoints' features ### key_features 

        #### keypoints, keypoint basis, keypoints basis; num_basis x 3 ####
        if not self.wo_keypts_abs:
            net = torch.cat([key_fea, key_pts.view(B * K, 3, 1)], 1) #### key_points #### keypoints
        else:
            net = key_fea
        net = F.relu(self.bn31(self.conv31(net)))
        net = F.relu(self.bn32(self.conv32(net)))
        
        
        
        rnd_net = torch.max(rnd_net, 2, keepdim=True)[0] ### net with tar_out
        rnd_key_fea = rnd_net.view(B * K, 64, 1) ### reshape to keypoints' features ### key_features 
        #### keypoints, keypoint basis, keypoints basis; num_basis x 3 ####
        if not self.wo_keypts_abs:
            rnd_net = torch.cat([rnd_key_fea, key_pts.view(B * K, 3, 1)], 1) #### key_points #### keypoints
        else:
            rnd_net = rnd_key_fea
        rnd_net = F.relu(self.bn31(self.conv31(rnd_net)))
        rnd_net = F.relu(self.bn32(self.conv32(rnd_net)))
        
        
        
        
        
        ##### basis as offsets #####
        basis = self.conv33(net).view(B, K * 3, self.num_basis)[..., 0] ### keypts delta transformations ###
        basis = basis.contiguous().view(B, K, 3).contiguous()
        ##### basis as offsets #####
        basis = basis * self.coef_multiplier ### a smaller offset value ###
        
        
        
        ##### basis as offsets #####
        rnd_basis = self.conv33(rnd_net).view(B, K * 3, self.num_basis)[..., 0] ### keypts delta transformations ###
        rnd_basis = rnd_basis.contiguous().view(B, K, 3).contiguous()
        ##### basis as offsets #####
        rnd_basis = rnd_basis * self.coef_multiplier ### a smaller offset value ###
        
        
        
        
        coef_range = torch.ones((B, self.num_basis, 2), dtype=torch.float32).cuda() * self.coef_multiplier
        coef_range[:, :, 0] = coef_range[:, :, 0] * -1.0
        
        

        def_key_pts = key_pts + basis ### B x K x 3
        def_pc = torch.bmm(w_pc, def_key_pts)
        
        
        
        
        rnd_def_key_pts = key_pts + rnd_basis ### B x K x 3
        rnd_def_pc = torch.bmm(w_pc, rnd_def_key_pts)

        #### def_pc and tar_pc ####
        cd_loss = chamfer_distance(def_pc, tar_pc)

        #### def_key_pts ####
        sample_def_key_pts = def_key_pts
        sample_def_pc = torch.bmm(w_pc, sample_def_key_pts)
        
        # sample_def_pc_sym = sample_def_pc * \
        #     torch.tensor([-1, 1, 1]).cuda()  # for shapenet shapes
        #### 
        sym_mult_tsr = torch.tensor([1, 1, 1]).cuda()
        sym_mult_tsr[self.symmetry_axis] = sym_mult_tsr[self.symmetry_axis] * -1.0
        sample_def_pc_sym = sample_def_pc * sym_mult_tsr
        # \
        #     torch.tensor([-1, 1, 1]).cuda()  # for shapenet shapes
        # sample_def_pc_sym = sample_def_pc * \
        #     torch.tensor([1, -1, 1]).cuda()  # for shapenet shapes
        sym_loss = chamfer_distance(sample_def_pc, sample_def_pc_sym) #### sym_loss for the deformed shape
        

        rnd_def_pc_sym = rnd_def_pc * sym_mult_tsr
        rnd_sym_loss = chamfer_distance(rnd_def_pc, rnd_def_pc_sym) #### sym_loss for the deformed shape
        
        
        sym_loss = (sym_loss + rnd_sym_loss) / 2.
        
        
        
        ### basis: bsz x n_keypts x 3 ###
        basis = basis.contiguous().unsqueeze(1).repeat(1, self.num_basis, 1, 1).contiguous() #### 
        basis = basis.view(B, self.num_basis, -1).contiguous()
        tot_sampled_def_key_pts = [def_key_pts, rnd_def_key_pts]
        coef = torch.ones((B, self.num_basis), dtype=torch.float32).cuda()
        
        if deform_net is None:
            cat_cd_loss = cd_loss.clone()
            cat_basis = basis.clone()
            cat_coef = coef.clone()
            cat_tot_sampled_def_key_pts = tot_sampled_def_key_pts
            cat_sym_loss = sym_loss.clone()
        
        
        return def_key_pts, def_pc, cd_loss, basis, coef, tot_sampled_def_key_pts, sym_loss, coef_range, loss_prior, loss_log_pz, loss_entropy, src_pp_topk_idxes, tar_pp_topk_idxes, \
          cat_def_key_pts, cat_def_pc, cat_cd_loss, cat_basis, cat_coef, cat_tot_sampled_def_key_pts, cat_sym_loss


    def forward_2(self, src_pc, tar_pc, key_pts, dst_key_pts, w_pc,  src_verts, src_faces, src_edges, src_dofs, tar_verts, tar_faces, tar_edges, tar_dofs, deform_net=None):
        #### B, N, _ 
        #### add src_faces, tar_faces for src_meshes and tar_meshes #### 
        #### network target driven ####
        B, N, _ = src_pc.shape
        
        if self.opt.hier_stage == 1:
            with torch.no_grad():
                # print("here!")
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

        cat_def_key_pts =def_key_pts.clone()
        cat_def_pc = def_pc.clone()
          
        src_pc = def_pc
        key_pts = def_key_pts
        
        ### downsample pc ###
        # n_samples = 512
        # n_samples = 1024
        n_samples = self.n_samples
        n_samples = 2048
        
        N = src_verts[0].size(0)
        bsz = src_pc.size(0)
        
        
        
        
        ### mesh vertices 
        src_verts = src_verts[0].unsqueeze(0)
        tar_verts = tar_verts[0].unsqueeze(0)
        
        # print(f"src_verts: {src_verts.size()}, tar_verts: {tar_verts.size()}")
        
        src_fps_idx = farthest_point_sampling(pos=src_verts[:, :, :3], n_sampling=n_samples) ### smapling for features xxx ### sampel pts for features ###
        ### src_pc: bsz x N x 3; src 
        src_pc_downsampled = src_verts.contiguous().view(bsz * N, 3)[src_fps_idx].contiguous().view(bsz, n_samples, 3).contiguous()
        
        ### tar_fps_idx ###
        tar_fps_idx = farthest_point_sampling(pos=tar_verts[:, :, :3], n_sampling=n_samples) ### smapling for features xxx
        ### src_pc: bsz x N x 3
        tar_pc_downsampled = tar_verts.contiguous().view(bsz * tar_verts.size(1), 3)[tar_fps_idx].contiguous().view(bsz, n_samples, 3).contiguous()
        
        # #### pointnet and tar_pointnet ####
        # src_out, src_pp_topk_idxes = self.pointnet(src_verts)
        # tar_out, tar_pp_topk_idxes = self.tar_pointnet(tar_verts)
        
        #### pointnet and tar_pointnet ####
        src_out, src_pp_topk_idxes = self.pointnet(src_pc_downsampled)
        tar_out, tar_pp_topk_idxes = self.tar_pointnet(tar_pc_downsampled)
        
        src_out = self.select_keypts_features(src_verts, src_pc_downsampled, src_out.contiguous().transpose(-1, -2).contiguous()).contiguous().transpose(-1, -2).contiguous()
        tar_out = self.select_keypts_features(tar_verts, tar_pc_downsampled, tar_out.contiguous().transpose(-1, -2).contiguous()).contiguous().transpose(-1, -2).contiguous()
        
        # src_out = self.select_keypts_features(key_pts, src_pc_downsampled, src_out.contiguous().transpose(-1, -2).contiguous()).contiguous().transpose(-1, -2).contiguous()
        # tar_out = self.select_keypts_features(dst_key_pts, tar_pc_downsampled, tar_out.contiguous().transpose(-1, -2).contiguous()).contiguous().transpose(-1, -2).contiguous()
        
        # print(f"src_out: {src_out.size()}, tar_out: {tar_out.size()}")
        
        # src_fps_idx = farthest_point_sampling(pos=src_pc[:, :, :3], n_sampling=n_samples) ### smapling for features xxx ### sampel pts for features ###
        # ### src_pc: bsz x N x 3; src 
        # src_pc_downsampled = src_pc.contiguous().view(bsz * N, 3)[src_fps_idx].contiguous().view(bsz, n_samples, 3).contiguous()
        
        # # w_pc: bsz x N x K ### w_pc: bsz x N x K
        # w_pc_expand = w_pc.contiguous().view(bsz * N, -1)[src_fps_idx].contiguous().view(bsz, n_samples, -1).contiguous()
        
        # ### tar_fps_idx ###
        # tar_fps_idx = farthest_point_sampling(pos=tar_pc[:, :, :3], n_sampling=n_samples) ### smapling for features xxx
        # ### src_pc: bsz x N x 3
        # tar_pc_downsampled = tar_pc.contiguous().view(bsz * N, 3)[tar_fps_idx].contiguous().view(bsz, n_samples, 3).contiguous()
        
        
        
        # N = n_samples
        # bz x n_samples x pos_dim
        
        #### pointnet and tar_pointnet ####
        # src_out, src_pp_topk_idxes = self.pointnet(src_pc_downsampled)
        # tar_out, tar_pp_topk_idxes = self.tar_pointnet(tar_pc_downsampled)
        
        src_out = F.relu(self.bn11(self.conv11(src_out)))
        src_out = F.relu(self.bn12(self.conv12(src_out)))
        # src_out = F.relu(self.bn13(self.conv13(src_out))) #### src_pp_features #### src_out ####
        src_out = self.bn13(self.conv13(src_out)) ### for each vertex, bn...
        
        # print(f"src_out: {src_out.size()}")
        #### for prob calculation ####
        prob_src_out, prob_src_pp_topk_idxes = self.prob_pointnet(src_verts)
        prob_tar_out, prob_tar_pp_topk_idxes = self.tar_prob_pointnet(tar_verts)
        
        prob_src_out = self.prob_src_out_conv_net(prob_src_out)
        # prob_tar_out = self.prob_tar_out_conv_net(prob_tar_out) ### tar_out
        #### for prob calculation ####
        
        # _, K, _ = key_pts.shape
        K = N
        # K = key_pts.size(1)
        # key_pts1 = key_pts.unsqueeze(-1).expand(-1, -1, -1, N)  # B K 3 N
        # #### w_pc --> w_pc1 --> transpose weights for further computation
        
        # #### using no downssample ###
        # # w_pc1 = w_pc.transpose(2, 1).unsqueeze(2)  # B K 1 N
        # #### for downsample ####  ### w_pc_expand --> for pc weights ###
        # w_pc1 = w_pc_expand.transpose(2, 1).unsqueeze(2) # w_pc_expand #
        
        
        ''' with all source pts features '''
        # src_out = src_out.unsqueeze(1).expand(-1, K, -1, -1)  # B K 64 N
        # ### what should be cat here --> src_out; w_pc1; key_pts1; 
        # ### for each keypoint, 1) keypoints-to-surface_points weights; 2) key_points values; 3) expanded source points 
        
        # # w_pc1, key_pts1; keypoints, source out, weights from keypoints to pcs 
        # net = torch.cat([src_out, w_pc1, key_pts1], 2).view(B * K, 68, N) #### 

        # net = F.relu(self.bn21(self.conv21(net)))
        # net = self.bn22(self.conv22(net)) ### with only keypts features ###
        ''' with all source pts features '''
        
        
        ''' only keypts features ''' ### bsz x n_keypts x dim ###
        # net = self.select_keypts_features(key_pts, src_pc_downsampled, src_out.contiguous().transpose(-1, -2).contiguous()).view(B * K, -1).unsqueeze(-1) 
        net = src_out.transpose(-1, -2).view(B * K, -1).unsqueeze(-1)
        # .transpose(-1, -2).view(B * K, -1).unsqueeze(-1) #### bsz x n_keypts x dim ####
        ''' only keypts features '''
        
        # prob_net = self.select_keypts_features(key_pts, src_pc_downsampled, prob_src_out.contiguous().transpose(-1, -2).contiguous()).view(B * K, -1).unsqueeze(-1) 
        prob_net = prob_src_out.transpose(-1, -2).view(B * K, -1).unsqueeze(-1)
        

        # net_out_dim = net.size(1)
        net_out_dim = prob_net.size(1)
        # net_out_mu, net_out_log_sigma = net[:, : net_out_dim // 2], net[:, net_out_dim // 2: ]
        prob_net_out_mu, prob_net_out_log_sigma = prob_net[:, : net_out_dim // 2], prob_net[:, net_out_dim // 2: ]
        # net = net_out_mu
        prob_net = prob_net_out_mu
        
        # net_out_dim = net.size(1)
        net_out_dim = net.size(1)
        # net_out_mu, net_out_log_sigma = net[:, : net_out_dim // 2], net[:, net_out_dim // 2: ]
        net_out_mu, net_out_log_sigma = net[:, : net_out_dim // 2], net[:, net_out_dim // 2: ]
        # net = net_out_mu
        net = net_out_mu
        # print(f"current net: {net.size()}")
        loss_prior = torch.zeros((1,), dtype=torch.float32).cuda().mean()
        loss_log_pz = torch.zeros((1,), dtype=torch.float32).cuda().mean()
        loss_entropy = torch.zeros((1,), dtype=torch.float32).cuda().mean()

        if self.tar_basis > 0: # use target for deformation basis prediction ###
            prob_tar_out = self.prob_tar_out_conv_net(prob_tar_out) ### prob tar out... ###
            
            ##### get tar out ##### ### tar_basis 
            ### first set of convoltuions, conv11, conv12, conv13
            # tar_out = torch.max(tar_out, dim=2, keepdim=True)[0] ###
            tar_out = F.relu(self.tar_bn11(self.tar_conv11(tar_out)))
            tar_out = F.relu(self.tar_bn12(self.tar_conv12(tar_out)))
            # tar_out = F.relu(self.tar_bn13(self.tar_conv13(tar_out))) ## tar_out: bsz x dim x N
            tar_out = self.tar_bn13(self.tar_conv13(tar_out)) ## tar_out: bsz x dim x N

            if not self.use_pp_tar_out_feat:
                tar_out = torch.max(tar_out, dim=2, keepdim=True)[0] ###
            ##### get tar_out feautres #####

            tar_out_dim = prob_tar_out.size(1)
            tar_out_mu, tar_out_log_sigma = prob_tar_out[:, :tar_out_dim // 2], prob_tar_out[:, tar_out_dim // 2: ]
            prob_tar_out = tar_out_mu
            loss_prior_tar = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            loss_log_pz_tar = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            loss_entropy_tar = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            ######## use_probs ##############

            # print(f"nettt size: {net.size()}")
            
            net = torch.max(net, 2, keepdim=True)[0] ### feature for each keypoint 
            net = net.contiguous().view(B, K, net.size(-2)) ### bsz x K x dim
            
            prob_net = torch.max(prob_net, 2, keepdim=True)[0] ### feature for each keypoint 
            prob_net = prob_net.contiguous().view(B, K, prob_net.size(-2)) ### bsz x K x dim
            ### tar_out for deformation direction ### ### target_out ### 
            ### target out; key_pts and tar_key_pts ...###
            
            # tar_out : bsz x n_pts x feat_dim; 
            tar_out_trans = prob_tar_out.contiguous().transpose(-1, -2).contiguous()
            ### deformation factorization ### ### tar_out: bsz x n_pts x feat_dim ###
            tar_keypts_out = tar_out_trans
            ### selected_tar_keypts_out: bsz x K x neighbouring_k x feat_dim ###
            ### key_pts: bsz x n_keypts x 3; #### select target keypts out ###
            
            
            
            ''' select via nearest neighbours directly '''
            # print(f"")
            selected_tar_keypts_out, _ = self.select_target_local_features(src_verts, tar_verts, tar_keypts_out, neighbouring_tar_keypts_idxes=None, neighbouring_tar_k=self.neighbouring_tar_k)
            ### selected_tar_keypts_out: bsz x K x feat_dim x neighbouring_k ###
            selected_tar_keypts_out = selected_tar_keypts_out.contiguous().transpose(2, 3).contiguous()  ### transpose(2, 3) --> 
            ''' select via nearest neighbours directly ''' 
            
            # tar_out = tar_out.unsqueeze(1).expand(-1, K, -1, -1)
            
            # prob_tar_out = prob_tar_out.unsqueeze(1).expand(-1, K, -1, -1)
            
            # B x K x 64 x N #
            # net = torch.cat([net, tar_out], dim=2).view(B * K, -1, N) ### bsz x K x (dim + 64) x N
            
            
            ##### only for nearesst neighbooring_k features #####
            net = net.contiguous().unsqueeze(-1).repeat(1, 1, 1, selected_tar_keypts_out.size(-1)).contiguous() ### bsz x K x dim x N
            
            prob_net = prob_net.contiguous().unsqueeze(-1).repeat(1, 1, 1, selected_tar_keypts_out.size(-1)).contiguous() ### bsz x K x dim x N
            ##### only for nearesst neighbooring_k features #####
            
        
            if self.use_prob:
                net_for_trans = torch.cat([prob_net, selected_tar_keypts_out], dim=2).view(B * K, -1, selected_tar_keypts_out.size(-1)) ### bsz x K x (dim + 64) x N
                
                prob_trans_tar_feats = self.prob_conv(net_for_trans.contiguous().view(B * K, -1, net_for_trans.size(-1)))
                prob_trans_nn_feats = prob_trans_tar_feats.size(1)
                prob_trans_mu, prob_trans_log_sigma = prob_trans_tar_feats[:, : prob_trans_nn_feats // 2, ], prob_trans_tar_feats[:, prob_trans_nn_feats // 2: ]
                z = utils.reparameterize_gaussian(mean=prob_trans_mu, logvar=prob_trans_log_sigma)  # (B, F) ## global 
                #### normal probability of z ####
                log_pz_tar = utils.standard_normal_logprob(z).sum(dim=1) 
                ##### maximize the entropy of q_x(z); gaussian_entropy #####
                entropy_tar = utils.gaussian_entropy(logvar=prob_trans_log_sigma) ### 
                ##### log probability of pz and entropy #####
                loss_prior_tar = (-log_pz_tar - entropy_tar).mean()
                # kl_weight = 0.001
                # loss_prior = kl_weigh
                # selected_tar_keypts_out = z #### reparameterized selected tar_out_features j#####
                
                if self.opt.with_recon:
                    ### recon_tar_features: bsz x dim x n_tar_neis ###
                    recon_tar_features = self.prob_tar_recon_net(z)
                    recon_loss = torch.sum((selected_tar_keypts_out - recon_tar_features) ** 2, dim=-2).mean() 
                    loss_prior_tar += recon_loss
                    selected_tar_keypts_out = recon_tar_features
                else:
                    selected_tar_keypts_out = z
                
                rnd_selected_tar_keypts_out = torch.randn_like(selected_tar_keypts_out) ### tar_keypts_out ###
                
                if self.opt.with_recon:
                    rnd_selected_tar_keypts_out = self.prob_tar_recon_net(rnd_selected_tar_keypts_out)
                selected_tar_keypts_out = selected_tar_keypts_out.contiguous().view(B, K, -1, selected_tar_keypts_out.size(-1)).contiguous()
                
                rnd_selected_tar_keypts_out = rnd_selected_tar_keypts_out.contiguous().view(B, K, -1, rnd_selected_tar_keypts_out.size(-1)).contiguous()
                
                ##### only for nearesst neighbooring_k features #####
                ### log_pz_tar ###
                loss_log_pz_tar = (-log_pz_tar).mean()
                loss_entropy_tar = (-entropy_tar).mean()
                if self.opt.with_recon:
                    loss_log_pz_tar = recon_loss.mean()
            else:
                rnd_selected_tar_keypts_out = selected_tar_keypts_out
                loss_prior_tar = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                loss_log_pz_tar = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                loss_entropy_tar = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                
            rnd_net = torch.cat([net, rnd_selected_tar_keypts_out], dim=2).view(B * K, -1, rnd_selected_tar_keypts_out.size(-1)) ### bsz x K x (dim + 64) x N

            rnd_net = F.relu(self.tar_bn21(self.tar_conv21(rnd_net)))
            rnd_net = self.tar_bn22(self.tar_conv22(rnd_net)) ### bsz x K x 64 x N

            net = torch.cat([net, selected_tar_keypts_out], dim=2).view(B * K, -1, selected_tar_keypts_out.size(-1)) ### bsz x K x (dim + 64) x N

            net = F.relu(self.tar_bn21(self.tar_conv21(net)))
            net = self.tar_bn22(self.tar_conv22(net)) ### bsz x K x 64 x N
            
        else:
            tar_out = F.relu(self.tar_bn11(self.tar_conv11(tar_out)))
            tar_out = F.relu(self.tar_bn12(self.tar_conv12(tar_out)))
            tar_out = F.relu(self.tar_bn13(self.tar_conv13(tar_out)))
            if not self.use_pp_tar_out_feat:
                tar_out = torch.max(tar_out, dim=2, keepdim=True)[0] ###
            tar_out_dim = tar_out.size(1)
            tar_out_mu, tar_out_log_sigma = tar_out[:, :tar_out_dim // 2], tar_out[:, tar_out_dim // 2: ]
            tar_out = tar_out_mu
            
            # --> tar_out : bsz x n_pts x feat_dim 
            tar_out_trans = tar_out.contiguous().transpose(-1, -2).contiguous()
            ### deformation factorization ### ### tar_out: bsz x n_pts x feat_dim ###
            tar_keypts_out = self.select_keypts_features(dst_key_pts, tar_pc_downsampled, tar_out_trans) #### bsz x ## tar keypts out...
            ### selected_tar_keypts_out: bsz x K x neighbouring_k x feat_dim ###
            ### key_pts: bsz x n_keypts x 3;
            
            ''' select via nearest neighbours directly ''' 
            selected_tar_keypts_out, neighbouring_tar_keypts_idxes = self.select_target_local_features(key_pts, dst_key_pts, tar_keypts_out, neighbouring_tar_keypts_idxes=None, neighbouring_tar_k=self.neighbouring_tar_k)
            ### selected_tar_keypts_out: bsz x K x feat_dim x neighbouring_k ###
            selected_tar_keypts_out = selected_tar_keypts_out.contiguous().transpose(2, 3).contiguous() 
            ''' select via nearest neighbours directly '''
            
            
            tar_out = tar_out.unsqueeze(1).expand(-1, K, -1, -1) 
            loss_prior_tar = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            loss_log_pz_tar = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            loss_entropy_tar = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            
            rnd_net = net
        
        if not self.use_prob_src:
            loss_prior = loss_prior_tar
            loss_log_pz = loss_log_pz_tar
            loss_entropy = loss_entropy_tar #### loss_entropy and tar... ###

        ##### pool net for glboal features ######
        ### net and netout ###
        net = torch.max(net, 2, keepdim=True)[0] ### net with tar_out
        ### 
        key_fea = net.view(B * K, 64, 1) ### reshape to keypoints' features ### key_features 

        #### keypoints, keypoint basis, keypoints basis; num_basis x 3 ####
        if not self.wo_keypts_abs:
            net = torch.cat([key_fea, src_verts.view(B * K, 3, 1)], 1) #### key_points #### keypoints
        else:
            net = key_fea
        net = F.relu(self.bn31(self.conv31(net)))
        net = F.relu(self.bn32(self.conv32(net)))
        
        
        
        rnd_net = torch.max(rnd_net, 2, keepdim=True)[0] ### net with tar_out
        rnd_key_fea = rnd_net.view(B * K, 64, 1) ### reshape to keypoints' features ### key_features 
        #### keypoints, keypoint basis, keypoints basis; num_basis x 3 ####
        if not self.wo_keypts_abs:
            rnd_net = torch.cat([rnd_key_fea, key_pts.view(B * K, 3, 1)], 1) #### key_points #### keypoints
        else:
            rnd_net = rnd_key_fea
        rnd_net = F.relu(self.bn31(self.conv31(rnd_net)))
        rnd_net = F.relu(self.bn32(self.conv32(rnd_net)))
        
        
        ##### basis as offsets #####
        basis = self.conv33(net).view(B, K * 3, self.num_basis)[..., 0] ### keypts delta transformations ###
        basis = basis.contiguous().view(B, K, 3).contiguous()
        ##### basis as offsets #####
        basis = basis * self.coef_multiplier ### a smaller offset ###
        
        # print(f"basis: {basis.size()}")
        
        
        
        ##### basis as offsets #####
        rnd_basis = self.conv33(rnd_net).view(B, K * 3, self.num_basis)[..., 0] ### keypts delta transformations ###
        rnd_basis = rnd_basis.contiguous().view(B, K, 3).contiguous()
        ##### basis as offsets #####
        rnd_basis = rnd_basis * self.coef_multiplier ### a smaller offset value ###
        
        
        coef_range = torch.ones((B, self.num_basis, 2), dtype=torch.float32).cuda() * self.coef_multiplier
        coef_range[:, :, 0] = coef_range[:, :, 0] * -1.0
        
        

        # def_key_pts = key_pts + basis ### B x K x 3
        # def_pc = torch.bmm(w_pc, def_key_pts)
        def_key_pts = src_verts + basis
        def_pc = def_key_pts
        
        
        
        rnd_def_key_pts = src_verts + rnd_basis
        rnd_def_pc = rnd_def_key_pts
        # rnd_def_key_pts = key_pts + rnd_basis ### B x K x 3
        # rnd_def_pc = torch.bmm(w_pc, rnd_def_key_pts)

        #### def_pc and tar_pc ####
        # cd_loss = chamfer_distance(def_pc, tar_pc)
        tar_verts = tar_verts[0].unsqueeze(0)
        cd_loss = chamfer_distance(def_pc, tar_verts)

        #### def_key_pts ####
        sample_def_key_pts = def_key_pts
        sample_def_pc = sample_def_key_pts
        

        sym_mult_tsr = torch.tensor([1, 1, 1]).cuda()
        sym_mult_tsr[self.symmetry_axis] = sym_mult_tsr[self.symmetry_axis] * -1.0
        sample_def_pc_sym = sample_def_pc * sym_mult_tsr

        sym_loss = chamfer_distance(sample_def_pc, sample_def_pc_sym) #### sym_loss for the deformed shape

        rnd_def_pc_sym = rnd_def_pc * sym_mult_tsr
        rnd_sym_loss = chamfer_distance(rnd_def_pc, rnd_def_pc_sym) #### sym_loss for the deformed shape

        sym_loss = (sym_loss + rnd_sym_loss) / 2.
        
        
        
        ### basis: bsz x n_keypts x 3 ###
        basis = basis.contiguous().unsqueeze(1).repeat(1, self.num_basis, 1, 1).contiguous() #### 
        basis = basis.view(B, self.num_basis, -1).contiguous()
        tot_sampled_def_key_pts = [def_key_pts, rnd_def_key_pts]
        coef = torch.ones((B, self.num_basis), dtype=torch.float32).cuda()
        
        if deform_net is None:
            cat_cd_loss = cd_loss.clone()
            cat_basis = basis.clone()
            cat_coef = coef.clone()
            cat_tot_sampled_def_key_pts = tot_sampled_def_key_pts
            cat_sym_loss = sym_loss.clone()
        
        
        return def_key_pts, def_pc, cd_loss, basis, coef, tot_sampled_def_key_pts, sym_loss, coef_range, loss_prior, loss_log_pz, loss_entropy, src_pp_topk_idxes, tar_pp_topk_idxes, \
          cat_def_key_pts, cat_def_pc, cat_cd_loss, cat_basis, cat_coef, cat_tot_sampled_def_key_pts, cat_sym_loss


    def sample(self, src_pc, tar_pc, key_pts, dst_key_pts, w_pc,  src_verts, src_faces, src_edges, src_dofs, tar_verts, tar_faces, tar_edges, tar_dofs, deform_net=None):
        #### B, N, _ 
        #### add src_faces, tar_faces for src_meshes and tar_meshes #### 
        #### network target driven ####
        B, N, _ = src_pc.shape
        
        if self.opt.hier_stage == 1:
            with torch.no_grad():
                # print("here!")
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


        cat_def_key_pts =def_key_pts.clone()
        cat_def_pc = def_pc.clone()
          
        src_pc = def_pc
        key_pts = def_key_pts
        
        ### downsample pc ###
        # n_samples = 512
        # n_samples = 1024
        n_samples = self.n_samples
        bsz, N = src_pc.size(0), src_pc.size(1)
        src_fps_idx = farthest_point_sampling(pos=src_pc[:, :, :3], n_sampling=n_samples) ### smapling for features xxx ### sampel pts for features ###
        ### src_pc: bsz x N x 3; src 
        src_pc_downsampled = src_pc.contiguous().view(bsz * N, 3)[src_fps_idx].contiguous().view(bsz, n_samples, 3).contiguous()
        
        # w_pc: bsz x N x K ### w_pc: bsz x N x K
        w_pc_expand = w_pc.contiguous().view(bsz * N, -1)[src_fps_idx].contiguous().view(bsz, n_samples, -1).contiguous()
        
        ### tar_fps_idx ###
        tar_fps_idx = farthest_point_sampling(pos=tar_pc[:, :, :3], n_sampling=n_samples) ### smapling for features xxx
        ### src_pc: bsz x N x 3
        tar_pc_downsampled = tar_pc.contiguous().view(bsz * N, 3)[tar_fps_idx].contiguous().view(bsz, n_samples, 3).contiguous()
        
        
        
        N = n_samples
        # bz x n_samples x pos_dim
        
        #### pointnet and tar_pointnet ####
        src_out, src_pp_topk_idxes = self.pointnet(src_pc_downsampled)
        tar_out, tar_pp_topk_idxes = self.tar_pointnet(tar_pc_downsampled)
        
        src_out = F.relu(self.bn11(self.conv11(src_out)))
        src_out = F.relu(self.bn12(self.conv12(src_out)))
        # src_out = F.relu(self.bn13(self.conv13(src_out))) #### src_pp_features #### src_out ####
        src_out = self.bn13(self.conv13(src_out)) #### src_pp_features #### src_out ####
        
        
        #### for prob calculation ####
        prob_src_out, prob_src_pp_topk_idxes = self.prob_pointnet(src_pc_downsampled)
        prob_tar_out, prob_tar_pp_topk_idxes = self.tar_prob_pointnet(tar_pc_downsampled)
        
        prob_src_out = self.prob_src_out_conv_net(prob_src_out)
        # prob_tar_out = self.prob_tar_out_conv_net(prob_tar_out)
        #### for prob calculation ####
        
        
        ### network and keypoints 

        _, K, _ = key_pts.shape
        key_pts1 = key_pts.unsqueeze(-1).expand(-1, -1, -1, N)  # B K 3 N
        #### w_pc --> w_pc1 --> transpose weights for further computation
        
        #### using no downssample ###
        # w_pc1 = w_pc.transpose(2, 1).unsqueeze(2)  # B K 1 N
        #### for downsample ####  ### w_pc_expand --> for pc weights ###
        w_pc1 = w_pc_expand.transpose(2, 1).unsqueeze(2) # w_pc_expand #
        
        
        ''' with all source pts features '''
        # src_out = src_out.unsqueeze(1).expand(-1, K, -1, -1)  # B K 64 N
        # ### what should be cat here --> src_out; w_pc1; key_pts1; 
        # ### for each keypoint, 1) keypoints-to-surface_points weights; 2) key_points values; 3) expanded source points 
        
        # # w_pc1, key_pts1; keypoints, source out, weights from keypoints to pcs 
        # net = torch.cat([src_out, w_pc1, key_pts1], 2).view(B * K, 68, N) #### 

        # net = F.relu(self.bn21(self.conv21(net)))
        # net = self.bn22(self.conv22(net)) ### with only keypts features ###
        ''' with all source pts features '''
        
        
        ''' only keypts features ''' ### bsz x n_keypts x dim ###
        net = self.select_keypts_features(key_pts, src_pc_downsampled, src_out.contiguous().transpose(-1, -2).contiguous()).view(B * K, -1).unsqueeze(-1) 
        # .transpose(-1, -2).view(B * K, -1).unsqueeze(-1) #### bsz x n_keypts x dim ####
        ''' only keypts features '''
        
        prob_net = self.select_keypts_features(key_pts, src_pc_downsampled, prob_src_out.contiguous().transpose(-1, -2).contiguous()).view(B * K, -1).unsqueeze(-1) 
        
        # if self.use_prob:
        if self.use_prob_src:
            # net_out_dim = net.size(1)
            net_out_dim = prob_net.size(1)
            # net_out_mu, net_out_log_sigma = net[:, : net_out_dim // 2], net[:, net_out_dim // 2: ]
            net_out_mu, net_out_log_sigma = prob_net[:, : net_out_dim // 2], prob_net[:, net_out_dim // 2: ]
            z = utils.reparameterize_gaussian(mean=net_out_mu, logvar=net_out_log_sigma)  # (B, F) ## global 
            ### zs probability ---> zs; standard normal logprob...
            log_pz = utils.standard_normal_logprob(z).sum(dim=1)  ### log_prob... ###
            ##### maximize the entropy of q_x(z); gaussian_entropy
            entropy = utils.gaussian_entropy(logvar=net_out_log_sigma) ### 
            loss_prior = (-log_pz - entropy).mean()
            ### minimize the negative log probability of zs ---- maximize the log probability of zs ### 
            # net = z
            prob_net = z
            
            loss_log_pz = (-log_pz).mean()
            loss_entropy = (-entropy).mean()
        else:
            # net_out_dim = net.size(1)
            net_out_dim = prob_net.size(1)
            # net_out_mu, net_out_log_sigma = net[:, : net_out_dim // 2], net[:, net_out_dim // 2: ]
            prob_net_out_mu, prob_net_out_log_sigma = prob_net[:, : net_out_dim // 2], prob_net[:, net_out_dim // 2: ]
            # net = net_out_mu
            prob_net = prob_net_out_mu
            
            # net_out_dim = net.size(1)
            net_out_dim = net.size(1)
            # net_out_mu, net_out_log_sigma = net[:, : net_out_dim // 2], net[:, net_out_dim // 2: ]
            net_out_mu, net_out_log_sigma = net[:, : net_out_dim // 2], net[:, net_out_dim // 2: ]
            # net = net_out_mu
            net = net_out_mu
            
            loss_prior = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            loss_log_pz = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            loss_entropy = torch.zeros((1,), dtype=torch.float32).cuda().mean()
        
        
        if self.tar_basis > 0: # use target for deformation basis prediction ###
            
            prob_tar_out = self.prob_tar_out_conv_net(prob_tar_out)
            
            ##### get tar out ##### ### tar_basis 
            ### first set of convoltuions, conv11, conv12, conv13
            # tar_out = torch.max(tar_out, dim=2, keepdim=True)[0] ###
            tar_out = F.relu(self.tar_bn11(self.tar_conv11(tar_out)))
            tar_out = F.relu(self.tar_bn12(self.tar_conv12(tar_out)))
            # tar_out = F.relu(self.tar_bn13(self.tar_conv13(tar_out))) ## tar_out: bsz x dim x N
            tar_out = self.tar_bn13(self.tar_conv13(tar_out)) ## tar_out: bsz x dim x N
            
            
            
            if not self.use_pp_tar_out_feat:
                tar_out = torch.max(tar_out, dim=2, keepdim=True)[0] ###
            ##### get tar_out feautres #####
            
            ######## use_probs ###########
            # if self.use_prob:
            #     #### tar_out: B x dim x N
            #     tar_out_dim = tar_out.size(1)
            #     ### 
            #     tar_out_mu, tar_out_log_sigma = tar_out[:, :tar_out_dim // 2], tar_out[:, tar_out_dim // 2: ]
            #     z = utils.reparameterize_gaussian(mean=tar_out_mu, logvar=tar_out_log_sigma)  # (B, F) ## global 
            #     #### normal probability of z ####
            #     log_pz_tar = utils.standard_normal_logprob(z).sum(dim=1) 
            #     ##### maximize the entropy of q_x(z); gaussian_entropy #####
            #     entropy_tar = utils.gaussian_entropy(logvar=tar_out_log_sigma) ### 
            #     ##### log probability of pz and entropy #####
            #     loss_prior_tar = (-log_pz_tar - entropy_tar).mean()
            #     # kl_weight = 0.001
            #     # loss_prior = kl_weigh
            #     tar_out = z
                
            #     loss_log_pz_tar = (-log_pz_tar).mean()
            #     loss_entropy_tar = (-entropy_tar).mean()
            # else:
                
            tar_out_dim = prob_tar_out.size(1)
            tar_out_mu, tar_out_log_sigma = prob_tar_out[:, :tar_out_dim // 2], prob_tar_out[:, tar_out_dim // 2: ]
            prob_tar_out = tar_out_mu
            loss_prior_tar = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            loss_log_pz_tar = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            loss_entropy_tar = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            ######## use_probs ##############
        
        
            # tar_out_sample
            ### each point ###
            # net = torch.max(net, 2, keepdim=True)[0] ### feature for each keypoint 
            # ### net: bsz x K x dim
            # net = net.contiguous().view(B, K, net.size(-2)) ### bsz x K x dim
            # tar_out = tar_out.unsqueeze(1).expand(-1, K, -1, -1)  ### tar_out: B x K x 64 x N
            # net = net.contiguous().unsqueeze(-1).repeat(1, 1, 1, N).contiguous() ### bsz x K x dim x N
            # # print(f"net: {net.size()}, tar_out: {tar_out.size()}")
            # net = torch.cat([net, tar_out], dim=2).view(B * K, -1, N) ### bsz x K x (dim + 64) x N
            # net = F.relu(self.tar_bn21(self.tar_conv21(net)))
            # net = self.tar_bn22(self.tar_conv22(net)) ### bsz x K x 64 x N
            ### each point ###
            
            # net = torch.max(net, 2, keepdim=True)[0]
            ### global feature ###
            ### tar_out: B x dim x N ###
            
            ''' When using the same conv network for features '''
            # net = torch.max(net, 2, keepdim=True)[0] ### feature for each keypoint 
            # net = net.contiguous().view(B, K, net.size(-2)) ### bsz x K x dim
            # ### tar_out for deformation direction ### ### target_out ### 
            # ### target out; key_pts and tar_key_pts ...###
            
            # # tar_out : bsz x n_pts x feat_dim; 
            # tar_out_trans = tar_out.contiguous().transpose(-1, -2).contiguous()
            # ### deformation factorization ### ### tar_out: bsz x n_pts x feat_dim ###
            # tar_keypts_out = self.select_keypts_features(dst_key_pts, tar_pc_downsampled, tar_out_trans)
            # ### selected_tar_keypts_out: bsz x K x neighbouring_k x feat_dim ###
            # ### key_pts: bsz x n_keypts x 3; #### select target keypts out ###
            ''' When using the same conv network for features '''
            
            net = torch.max(net, 2, keepdim=True)[0] ### feature for each keypoint 
            net = net.contiguous().view(B, K, net.size(-2)) ### bsz x K x dim
            
            prob_net = torch.max(prob_net, 2, keepdim=True)[0] ### feature for each keypoint 
            prob_net = prob_net.contiguous().view(B, K, prob_net.size(-2)) ### bsz x K x dim
            ### tar_out for deformation direction ### ### target_out ### 
            ### target out; key_pts and tar_key_pts ...###
            
            # tar_out : bsz x n_pts x feat_dim; 
            tar_out_trans = prob_tar_out.contiguous().transpose(-1, -2).contiguous()
            ### deformation factorization ### ### tar_out: bsz x n_pts x feat_dim ###
            tar_keypts_out = self.select_keypts_features(dst_key_pts, tar_pc_downsampled, tar_out_trans)
            ### selected_tar_keypts_out: bsz x K x neighbouring_k x feat_dim ###
            ### key_pts: bsz x n_keypts x 3; #### select target keypts out ###
            
            
            
            # if self.use_prob:
            #     #### tar_out: B x dim x N
            #     tar_out_dim = tar_keypts_out.size(-1)
            #     ### 
            #     tar_out_mu, tar_out_log_sigma = tar_out[:, :tar_out_dim // 2], tar_out[:, tar_out_dim // 2: ]
            #     z = utils.reparameterize_gaussian(mean=tar_out_mu, logvar=tar_out_log_sigma)  # (B, F) ## global 
            #     log_pz = utils.standard_normal_logprob(z).sum(dim=1) 
            #     ##### maximize the entropy of q_x(z); gaussian_entropy #####
            #     entropy = utils.gaussian_entropy(logvar=tar_out_log_sigma) ### 
            #     loss_prior = (-log_pz - entropy).mean()
            #     # kl_weight = 0.001
            #     # loss_prior = kl_weigh
            #     tar_out = z
            # else:
            #     tar_out_dim = tar_out.size(1)
            #     tar_out_mu, tar_out_log_sigma = tar_out[:, :tar_out_dim // 2], tar_out[:, tar_out_dim // 2: ]
            #     tar_out = tar_out_mu
            #     loss_prior = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            
            ''' select via nearest neighbours directly ''' 
            ##### for target keypts features #####
            selected_tar_keypts_out, neighbouring_tar_keypts_idxes = self.select_target_local_features(key_pts, dst_key_pts, tar_keypts_out, neighbouring_tar_keypts_idxes=None, neighbouring_tar_k=self.neighbouring_tar_k)
            ### selected_tar_keypts_out: bsz x K x feat_dim x neighbouring_k ###
            selected_tar_keypts_out = selected_tar_keypts_out.contiguous().transpose(2, 3).contiguous() 
            ''' select via nearest neighbours directly ''' 
            
            ''' select via euclidean and normal distances '''
            # selected_tar_keypts_indices = self.get_correspondances(key_pts, dst_key_pts, src_verts, src_faces, tar_verts, tar_faces)
            # ### selected_tar_keypts_indices: bsz x n_keypts ###
            # # tar_keypts_out: bsz x n_keypts x dim; bsz x n_keypts x 1 x dim
            # selected_tar_keypts_out = batched_index_select(values=tar_keypts_out, indices=selected_tar_keypts_indices.unsqueeze(-1), dim=1) ### bsz x n_keypts x 1 x dim ###
            # selected_tar_keypts_out = selected_tar_keypts_out.contiguous().transpose(-1, -2).contiguous()
            ''' select via euclidean and normal distances '''
            
            
            tar_out = tar_out.unsqueeze(1).expand(-1, K, -1, -1)
            
            prob_tar_out = prob_tar_out.unsqueeze(1).expand(-1, K, -1, -1)
            
            # B x K x 64 x N #
            # net = torch.cat([net, tar_out], dim=2).view(B * K, -1, N) ### bsz x K x (dim + 64) x N
            
            
            ##### only for nearesst neighbooring_k features #####
            net = net.contiguous().unsqueeze(-1).repeat(1, 1, 1, selected_tar_keypts_out.size(-1)).contiguous() ### bsz x K x dim x N
            
            prob_net = prob_net.contiguous().unsqueeze(-1).repeat(1, 1, 1, selected_tar_keypts_out.size(-1)).contiguous() ### bsz x K x dim x N
            ##### only for nearesst neighbooring_k features #####
            
        
            if self.use_prob:
                net_for_trans = torch.cat([prob_net, selected_tar_keypts_out], dim=2).view(B * K, -1, selected_tar_keypts_out.size(-1)) ### bsz x K x (dim + 64) x N
                
                prob_trans_tar_feats = self.prob_conv(net_for_trans.contiguous().view(B * K, -1, net_for_trans.size(-1)))
                prob_trans_nn_feats = prob_trans_tar_feats.size(1)
                prob_trans_mu, prob_trans_log_sigma = prob_trans_tar_feats[:, : prob_trans_nn_feats // 2, ], prob_trans_tar_feats[:, prob_trans_nn_feats // 2: ]
                z = utils.reparameterize_gaussian(mean=prob_trans_mu, logvar=prob_trans_log_sigma)  # (B, F) ## global 
                #### normal probability of z ####
                log_pz_tar = utils.standard_normal_logprob(z).sum(dim=1) 
                ##### maximize the entropy of q_x(z); gaussian_entropy #####
                entropy_tar = utils.gaussian_entropy(logvar=prob_trans_log_sigma) ### 
                ##### log probability of pz and entropy #####
                loss_prior_tar = (-log_pz_tar - entropy_tar).mean()
                # kl_weight = 0.001
                # loss_prior = kl_weigh
                # selected_tar_keypts_out = z #### reparameterized selected tar_out_features j#####
                
                z = torch.randn_like(z)
                
                if self.opt.with_recon:
                    ### recon_tar_features: bsz x dim x n_tar_neis ###
                    recon_tar_features = self.prob_tar_recon_net(z)
                    recon_loss = torch.sum((selected_tar_keypts_out - recon_tar_features) ** 2, dim=-2).mean() 
                    loss_prior_tar += recon_loss
                    selected_tar_keypts_out = recon_tar_features
                else:
                    selected_tar_keypts_out = z
                
                
                
            
                selected_tar_keypts_out = selected_tar_keypts_out.contiguous().view(B, K, -1, selected_tar_keypts_out.size(-1)).contiguous()
                ##### only for nearesst neighbooring_k features #####
                ### log_pz_tar ###
                # loss_log_pz_tar = (-log_pz_tar).mean()
                # loss_entropy_tar = (-entropy_tar).mean()
            else:
                loss_prior_tar = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                # loss_log_pz_tar = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                # loss_entropy_tar = torch.zeros((1,), dtype=torch.float32).cuda().mean()

            net = torch.cat([net, selected_tar_keypts_out], dim=2).view(B * K, -1, selected_tar_keypts_out.size(-1)) ### bsz x K x (dim + 64) x N
            
            ###### net and convnets ######
            net = F.relu(self.tar_bn21(self.tar_conv21(net)))
            net = self.tar_bn22(self.tar_conv22(net)) ### bsz x K x 64 x N
            ### global feature ###
        else:
            tar_out = F.relu(self.tar_bn11(self.tar_conv11(tar_out)))
            tar_out = F.relu(self.tar_bn12(self.tar_conv12(tar_out)))
            tar_out = F.relu(self.tar_bn13(self.tar_conv13(tar_out)))
            if not self.use_pp_tar_out_feat:
                tar_out = torch.max(tar_out, dim=2, keepdim=True)[0] ###
            tar_out_dim = tar_out.size(1)
            tar_out_mu, tar_out_log_sigma = tar_out[:, :tar_out_dim // 2], tar_out[:, tar_out_dim // 2: ]
            tar_out = tar_out_mu
            
            # --> tar_out : bsz x n_pts x feat_dim 
            tar_out_trans = tar_out.contiguous().transpose(-1, -2).contiguous()
            ### deformation factorization ### ### tar_out: bsz x n_pts x feat_dim ###
            tar_keypts_out = self.select_keypts_features(dst_key_pts, tar_pc_downsampled, tar_out_trans) #### bsz x ## tar keypts out...
            ### selected_tar_keypts_out: bsz x K x neighbouring_k x feat_dim ###
            ### key_pts: bsz x n_keypts x 3;
            
            ''' select via nearest neighbours directly ''' 
            selected_tar_keypts_out, neighbouring_tar_keypts_idxes = self.select_target_local_features(key_pts, dst_key_pts, tar_keypts_out, neighbouring_tar_keypts_idxes=None, neighbouring_tar_k=self.neighbouring_tar_k)
            ### selected_tar_keypts_out: bsz x K x feat_dim x neighbouring_k ###
            selected_tar_keypts_out = selected_tar_keypts_out.contiguous().transpose(2, 3).contiguous() 
            ''' select via nearest neighbours directly '''
            
            #### euclidean and normal distances ###
            ''' select via euclidean and normal distances '''
            # selected_tar_keypts_indices = self.get_correspondances(key_pts, dst_key_pts, src_verts, src_faces, tar_verts, tar_faces)
            # ### selected_tar_keypts_indices: bsz x n_keypts ###
            # # tar_keypts_out: bsz x n_keypts x dim; bsz x n_keypts x 1 x dim
            # selected_tar_keypts_out = batched_index_select(values=tar_keypts_out, indices=selected_tar_keypts_indices.unsqueeze(-1), dim=1) ### bsz x n_keypts x 1 x dim ###
            # selected_tar_keypts_out = selected_tar_keypts_out.contiguous().transpose(-1, -2).contiguous()
            ''' select via euclidean and normal distances '''
                
            
            tar_out = tar_out.unsqueeze(1).expand(-1, K, -1, -1) 
            loss_prior_tar = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            loss_log_pz_tar = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            loss_entropy_tar = torch.zeros((1,), dtype=torch.float32).cuda().mean()
        
        if not self.use_prob_src:
            loss_prior = loss_prior_tar
            # loss_log_pz = loss_log_pz_tar
            # loss_entropy = loss_entropy_tar #### loss_entropy and tar... ###

        
        ### key feat
        ##### pool net for glboal features ######
        ### net and netout ###
        net = torch.max(net, 2, keepdim=True)[0] ### net with tar_out
        ### 
        key_fea = net.view(B * K, 64, 1) ### reshape to keypoints' features ### key_features 

        #### keypoints, keypoint basis, keypoints basis; num_basis x 3 ####
        if not self.wo_keypts_abs:
            net = torch.cat([key_fea, key_pts.view(B * K, 3, 1)], 1) #### key_points #### keypoints
        else:
            net = key_fea
        net = F.relu(self.bn31(self.conv31(net)))
        net = F.relu(self.bn32(self.conv32(net)))
        
        ##### basis as offsets #####
        basis = self.conv33(net).view(B, K * 3, self.num_basis)[..., 0] ### keypts delta transformations ###
        basis = basis.contiguous().view(B, K, 3).contiguous()
        ##### basis as offsets #####
        
        basis = basis * self.coef_multiplier ### a smaller offset value ###
        
        coef_range = torch.ones((B, self.num_basis, 2), dtype=torch.float32).cuda() * self.coef_multiplier
        coef_range[:, :, 0] = coef_range[:, :, 0] * -1.0
        
        

        def_key_pts = key_pts + basis ### B x K x 3
        def_pc = torch.bmm(w_pc, def_key_pts)

        #### def_pc and tar_pc ####
        cd_loss = chamfer_distance(def_pc, tar_pc)

        #### def_key_pts ####
        sample_def_key_pts = def_key_pts
        sample_def_pc = torch.bmm(w_pc, sample_def_key_pts)
        
        # sample_def_pc_sym = sample_def_pc * \
        #     torch.tensor([-1, 1, 1]).cuda()  # for shapenet shapes
        #### 
        sym_mult_tsr = torch.tensor([1, 1, 1]).cuda()
        sym_mult_tsr[self.symmetry_axis] = sym_mult_tsr[self.symmetry_axis] * -1.0
        sample_def_pc_sym = sample_def_pc * sym_mult_tsr
        # \
        #     torch.tensor([-1, 1, 1]).cuda()  # for shapenet shapes
        # sample_def_pc_sym = sample_def_pc * \
        #     torch.tensor([1, -1, 1]).cuda()  # for shapenet shapes
        sym_loss = chamfer_distance(sample_def_pc, sample_def_pc_sym) #### sym_loss for the deformed shape
        
        
        # cd_loss = cd_loss + cat_cd_loss
        
        ### basis: bsz x n_keypts x 3 ###
        # for i_s in range(10):
        basis = basis.contiguous().unsqueeze(1).repeat(1, self.num_basis, 1, 1).contiguous() #### 
        basis = basis.view(B, self.num_basis, -1).contiguous()
        tot_sampled_def_key_pts = [def_key_pts]
        coef = torch.ones((B, self.num_basis), dtype=torch.float32).cuda()
        
        if deform_net is None:
            cat_cd_loss = cd_loss.clone()
            cat_basis = basis.clone()
            cat_coef = coef.clone()
            cat_tot_sampled_def_key_pts = tot_sampled_def_key_pts
            cat_sym_loss = sym_loss.clone()
        
        
        return def_key_pts, def_pc, cd_loss, basis, coef, tot_sampled_def_key_pts, sym_loss, coef_range, loss_prior, src_pp_topk_idxes, tar_pp_topk_idxes, \
          cat_def_key_pts, cat_def_pc, cat_cd_loss, cat_basis, cat_coef, cat_tot_sampled_def_key_pts, cat_sym_loss

    def sample_2(self, src_pc, tar_pc, key_pts, dst_key_pts, w_pc,  src_verts, src_faces, src_edges, src_dofs, tar_verts, tar_faces, tar_edges, tar_dofs, deform_net=None):
        #### B, N, _ 
        #### add src_faces, tar_faces for src_meshes and tar_meshes #### 
        #### network target driven ####
        B, N, _ = src_pc.shape
        
        if self.opt.hier_stage == 1:
            with torch.no_grad():
                # print("here!")
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


        cat_def_key_pts =def_key_pts.clone()
        cat_def_pc = def_pc.clone()
          
        src_pc = def_pc
        key_pts = def_key_pts
        
        ### downsample pc ###
        # n_samples = 512
        # n_samples = 1024
        n_samples = self.n_samples
        n_samples = 2048
        
        N = src_verts[0].size(0)
        bsz = src_pc.size(0)
        
        ### mesh verticesc
        src_verts = src_verts[0].unsqueeze(0)
        tar_verts = tar_verts[0].unsqueeze(0)
        
        src_fps_idx = farthest_point_sampling(pos=src_verts[:, :, :3], n_sampling=n_samples) ### smapling for features xxx ### sampel pts for features ###
        ### src_pc: bsz x N x 3; src 
        src_pc_downsampled = src_verts.contiguous().view(bsz * N, 3)[src_fps_idx].contiguous().view(bsz, n_samples, 3).contiguous()
        
        ### tar_fps_idx ###
        tar_fps_idx = farthest_point_sampling(pos=tar_verts[:, :, :3], n_sampling=n_samples) ### smapling for features xxx
        ### src_pc: bsz x N x 3
        # tar_pc_downsampled = tar_verts.contiguous().view(bsz * N, 3)[tar_fps_idx].contiguous().view(bsz, n_samples, 3).contiguous()
        tar_pc_downsampled = tar_verts.contiguous().view(bsz * tar_verts.size(1), 3)[tar_fps_idx].contiguous().view(bsz, n_samples, 3).contiguous()
        
        # #### pointnet and tar_pointnet ####
        # src_out, src_pp_topk_idxes = self.pointnet(src_verts)
        # tar_out, tar_pp_topk_idxes = self.tar_pointnet(tar_verts)
        
        #### pointnet and tar_pointnet ####
        src_out, src_pp_topk_idxes = self.pointnet(src_pc_downsampled)
        tar_out, tar_pp_topk_idxes = self.tar_pointnet(tar_pc_downsampled)
        
        src_out = self.select_keypts_features(src_verts, src_pc_downsampled, src_out.contiguous().transpose(-1, -2).contiguous()).contiguous().transpose(-1, -2).contiguous()
        tar_out = self.select_keypts_features(tar_verts, tar_pc_downsampled, tar_out.contiguous().transpose(-1, -2).contiguous()).contiguous().transpose(-1, -2).contiguous()
        
        
        
        # N = n_samples
        # bz x n_samples x pos_dim
        
        #### pointnet and tar_pointnet ####
        # src_out, src_pp_topk_idxes = self.pointnet(src_pc_downsampled)
        # tar_out, tar_pp_topk_idxes = self.tar_pointnet(tar_pc_downsampled)
        
        src_out = F.relu(self.bn11(self.conv11(src_out)))
        src_out = F.relu(self.bn12(self.conv12(src_out)))
        # src_out = F.relu(self.bn13(self.conv13(src_out))) #### src_pp_features #### src_out ####
        src_out = self.bn13(self.conv13(src_out)) ### for each vertex, bn...
        
        
        #### for prob calculation ####
        prob_src_out, prob_src_pp_topk_idxes = self.prob_pointnet(src_verts)
        prob_tar_out, prob_tar_pp_topk_idxes = self.tar_prob_pointnet(tar_verts)
        
        prob_src_out = self.prob_src_out_conv_net(prob_src_out)
        # prob_tar_out = self.prob_tar_out_conv_net(prob_tar_out)
        #### for prob calculation ####
        
        
        ### network and keypoints 

        # _, K, _ = key_pts.shape
        K = N
        # key_pts1 = key_pts.unsqueeze(-1).expand(-1, -1, -1, N)  # B K 3 N
        # #### w_pc --> w_pc1 --> transpose weights for further computation
        
        # #### using no downssample ###
        # # w_pc1 = w_pc.transpose(2, 1).unsqueeze(2)  # B K 1 N
        # #### for downsample ####  ### w_pc_expand --> for pc weights ###
        # w_pc1 = w_pc_expand.transpose(2, 1).unsqueeze(2) # w_pc_expand #
        
        
        ''' with all source pts features '''
        # src_out = src_out.unsqueeze(1).expand(-1, K, -1, -1)  # B K 64 N
        # ### what should be cat here --> src_out; w_pc1; key_pts1; 
        # ### for each keypoint, 1) keypoints-to-surface_points weights; 2) key_points values; 3) expanded source points 
        
        # # w_pc1, key_pts1; keypoints, source out, weights from keypoints to pcs 
        # net = torch.cat([src_out, w_pc1, key_pts1], 2).view(B * K, 68, N) #### 

        # net = F.relu(self.bn21(self.conv21(net)))
        # net = self.bn22(self.conv22(net)) ### with only keypts features ###
        ''' with all source pts features '''
        
        
        ''' only keypts features ''' ### bsz x n_keypts x dim ###
        # net = self.select_keypts_features(key_pts, src_pc_downsampled, src_out.contiguous().transpose(-1, -2).contiguous()).view(B * K, -1).unsqueeze(-1) 
        net = src_out.view(B * K, -1).unsqueeze(-1)
        # .transpose(-1, -2).view(B * K, -1).unsqueeze(-1) #### bsz x n_keypts x dim ####
        ''' only keypts features '''
        
        # prob_net = self.select_keypts_features(key_pts, src_pc_downsampled, prob_src_out.contiguous().transpose(-1, -2).contiguous()).view(B * K, -1).unsqueeze(-1) 
        prob_net = prob_src_out.view(B * K, -1).unsqueeze(-1)
        

        # net_out_dim = net.size(1)
        net_out_dim = prob_net.size(1)
        # net_out_mu, net_out_log_sigma = net[:, : net_out_dim // 2], net[:, net_out_dim // 2: ]
        prob_net_out_mu, prob_net_out_log_sigma = prob_net[:, : net_out_dim // 2], prob_net[:, net_out_dim // 2: ]
        # net = net_out_mu
        prob_net = prob_net_out_mu
        
        # net_out_dim = net.size(1)
        net_out_dim = net.size(1)
        # net_out_mu, net_out_log_sigma = net[:, : net_out_dim // 2], net[:, net_out_dim // 2: ]
        net_out_mu, net_out_log_sigma = net[:, : net_out_dim // 2], net[:, net_out_dim // 2: ]
        # net = net_out_mu
        net = net_out_mu
        # print(f"current net: {net.size()}")
        loss_prior = torch.zeros((1,), dtype=torch.float32).cuda().mean()
        loss_log_pz = torch.zeros((1,), dtype=torch.float32).cuda().mean()
        loss_entropy = torch.zeros((1,), dtype=torch.float32).cuda().mean()

        if self.tar_basis > 0: # use target for deformation basis prediction ###
            prob_tar_out = self.prob_tar_out_conv_net(prob_tar_out) ### prob tar out... ###
            
            ##### get tar out ##### ### tar_basis 
            ### first set of convoltuions, conv11, conv12, conv13
            # tar_out = torch.max(tar_out, dim=2, keepdim=True)[0] ###
            tar_out = F.relu(self.tar_bn11(self.tar_conv11(tar_out)))
            tar_out = F.relu(self.tar_bn12(self.tar_conv12(tar_out)))
            # tar_out = F.relu(self.tar_bn13(self.tar_conv13(tar_out))) ## tar_out: bsz x dim x N
            tar_out = self.tar_bn13(self.tar_conv13(tar_out)) ## tar_out: bsz x dim x N

            if not self.use_pp_tar_out_feat:
                tar_out = torch.max(tar_out, dim=2, keepdim=True)[0] ###
            ##### get tar_out feautres #####

            tar_out_dim = prob_tar_out.size(1)
            tar_out_mu, tar_out_log_sigma = prob_tar_out[:, :tar_out_dim // 2], prob_tar_out[:, tar_out_dim // 2: ]
            prob_tar_out = tar_out_mu
            loss_prior_tar = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            loss_log_pz_tar = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            loss_entropy_tar = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            ######## use_probs ##############

            # print(f"nettt size: {net.size()}")
            
            net = torch.max(net, 2, keepdim=True)[0] ### feature for each keypoint 
            net = net.contiguous().view(B, K, net.size(-2)) ### bsz x K x dim
            
            prob_net = torch.max(prob_net, 2, keepdim=True)[0] ### feature for each keypoint 
            prob_net = prob_net.contiguous().view(B, K, prob_net.size(-2)) ### bsz x K x dim
            ### tar_out for deformation direction ### ### target_out ### 
            ### target out; key_pts and tar_key_pts ...###
            
            # tar_out : bsz x n_pts x feat_dim; 
            tar_out_trans = prob_tar_out.contiguous().transpose(-1, -2).contiguous()
            ### deformation factorization ### ### tar_out: bsz x n_pts x feat_dim ###
            tar_keypts_out = tar_out_trans
            ### selected_tar_keypts_out: bsz x K x neighbouring_k x feat_dim ###
            ### key_pts: bsz x n_keypts x 3; #### select target keypts out ###
            
            
            
            # if self.use_prob:
            #     #### tar_out: B x dim x N
            #     tar_out_dim = tar_keypts_out.size(-1)
            #     ### 
            #     tar_out_mu, tar_out_log_sigma = tar_out[:, :tar_out_dim // 2], tar_out[:, tar_out_dim // 2: ]
            #     z = utils.reparameterize_gaussian(mean=tar_out_mu, logvar=tar_out_log_sigma)  # (B, F) ## global 
            #     log_pz = utils.standard_normal_logprob(z).sum(dim=1) 
            #     ##### maximize the entropy of q_x(z); gaussian_entropy #####
            #     entropy = utils.gaussian_entropy(logvar=tar_out_log_sigma) ### 
            #     loss_prior = (-log_pz - entropy).mean()
            #     # kl_weight = 0.001
            #     # loss_prior = kl_weigh
            #     tar_out = z
            # else:
            #     tar_out_dim = tar_out.size(1)
            #     tar_out_mu, tar_out_log_sigma = tar_out[:, :tar_out_dim // 2], tar_out[:, tar_out_dim // 2: ]
            #     tar_out = tar_out_mu
            #     loss_prior = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            
            ''' select via nearest neighbours directly '''
            # print(f"")
            selected_tar_keypts_out, _ = self.select_target_local_features(src_verts, tar_verts, tar_keypts_out, neighbouring_tar_keypts_idxes=None, neighbouring_tar_k=self.neighbouring_tar_k)
            ### selected_tar_keypts_out: bsz x K x feat_dim x neighbouring_k ###
            selected_tar_keypts_out = selected_tar_keypts_out.contiguous().transpose(2, 3).contiguous()  ### transpose(2, 3) --> 
            ''' select via nearest neighbours directly ''' 
            
            # tar_out = tar_out.unsqueeze(1).expand(-1, K, -1, -1)
            
            # prob_tar_out = prob_tar_out.unsqueeze(1).expand(-1, K, -1, -1)
            
            # B x K x 64 x N #
            # net = torch.cat([net, tar_out], dim=2).view(B * K, -1, N) ### bsz x K x (dim + 64) x N
            
            
            ##### only for nearesst neighbooring_k features #####
            net = net.contiguous().unsqueeze(-1).repeat(1, 1, 1, selected_tar_keypts_out.size(-1)).contiguous() ### bsz x K x dim x N
            
            prob_net = prob_net.contiguous().unsqueeze(-1).repeat(1, 1, 1, selected_tar_keypts_out.size(-1)).contiguous() ### bsz x K x dim x N
            ##### only for nearesst neighbooring_k features #####
            
        
            if self.use_prob:
                net_for_trans = torch.cat([prob_net, selected_tar_keypts_out], dim=2).view(B * K, -1, selected_tar_keypts_out.size(-1)) ### bsz x K x (dim + 64) x N
                
                prob_trans_tar_feats = self.prob_conv(net_for_trans.contiguous().view(B * K, -1, net_for_trans.size(-1)))
                prob_trans_nn_feats = prob_trans_tar_feats.size(1)
                prob_trans_mu, prob_trans_log_sigma = prob_trans_tar_feats[:, : prob_trans_nn_feats // 2, ], prob_trans_tar_feats[:, prob_trans_nn_feats // 2: ]
                z = utils.reparameterize_gaussian(mean=prob_trans_mu, logvar=prob_trans_log_sigma)  # (B, F) ## global 
                #### normal probability of z ####
                log_pz_tar = utils.standard_normal_logprob(z).sum(dim=1) 
                ##### maximize the entropy of q_x(z); gaussian_entropy #####
                entropy_tar = utils.gaussian_entropy(logvar=prob_trans_log_sigma) ### 
                ##### log probability of pz and entropy #####
                loss_prior_tar = (-log_pz_tar - entropy_tar).mean()
                # kl_weight = 0.001
                # loss_prior = kl_weigh
                # selected_tar_keypts_out = z #### reparameterized selected tar_out_features j#####
                
                if self.opt.with_recon:
                    ### recon_tar_features: bsz x dim x n_tar_neis ###
                    recon_tar_features = self.prob_tar_recon_net(z)
                    recon_loss = torch.sum((selected_tar_keypts_out - recon_tar_features) ** 2, dim=-2).mean() 
                    loss_prior_tar += recon_loss
                    selected_tar_keypts_out = recon_tar_features
                else:
                    selected_tar_keypts_out = z
                selected_tar_keypts_out = torch.randn_like(selected_tar_keypts_out)
                rnd_selected_tar_keypts_out = torch.randn_like(selected_tar_keypts_out) ### tar_keypts_out ###
                
                if self.opt.with_recon:
                    rnd_selected_tar_keypts_out = self.prob_tar_recon_net(rnd_selected_tar_keypts_out)
                selected_tar_keypts_out = selected_tar_keypts_out.contiguous().view(B, K, -1, selected_tar_keypts_out.size(-1)).contiguous()
                
                rnd_selected_tar_keypts_out = rnd_selected_tar_keypts_out.contiguous().view(B, K, -1, rnd_selected_tar_keypts_out.size(-1)).contiguous()
                
                ##### only for nearesst neighbooring_k features #####
                ### log_pz_tar ###
                loss_log_pz_tar = (-log_pz_tar).mean()
                loss_entropy_tar = (-entropy_tar).mean()
                if self.opt.with_recon:
                    loss_log_pz_tar = recon_loss.mean()
            else:
                rnd_selected_tar_keypts_out = selected_tar_keypts_out
                loss_prior_tar = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                loss_log_pz_tar = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                loss_entropy_tar = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                
            rnd_net = torch.cat([net, rnd_selected_tar_keypts_out], dim=2).view(B * K, -1, rnd_selected_tar_keypts_out.size(-1)) ### bsz x K x (dim + 64) x N

            rnd_net = F.relu(self.tar_bn21(self.tar_conv21(rnd_net)))
            rnd_net = self.tar_bn22(self.tar_conv22(rnd_net)) ### bsz x K x 64 x N

            net = torch.cat([net, selected_tar_keypts_out], dim=2).view(B * K, -1, selected_tar_keypts_out.size(-1)) ### bsz x K x (dim + 64) x N

            net = F.relu(self.tar_bn21(self.tar_conv21(net)))
            net = self.tar_bn22(self.tar_conv22(net)) ### bsz x K x 64 x N
            
        else:
            tar_out = F.relu(self.tar_bn11(self.tar_conv11(tar_out)))
            tar_out = F.relu(self.tar_bn12(self.tar_conv12(tar_out)))
            tar_out = F.relu(self.tar_bn13(self.tar_conv13(tar_out)))
            if not self.use_pp_tar_out_feat:
                tar_out = torch.max(tar_out, dim=2, keepdim=True)[0] ###
            tar_out_dim = tar_out.size(1)
            tar_out_mu, tar_out_log_sigma = tar_out[:, :tar_out_dim // 2], tar_out[:, tar_out_dim // 2: ]
            tar_out = tar_out_mu
            
            # --> tar_out : bsz x n_pts x feat_dim 
            tar_out_trans = tar_out.contiguous().transpose(-1, -2).contiguous()
            ### deformation factorization ### ### tar_out: bsz x n_pts x feat_dim ###
            tar_keypts_out = self.select_keypts_features(dst_key_pts, tar_pc_downsampled, tar_out_trans) #### bsz x ## tar keypts out...
            ### selected_tar_keypts_out: bsz x K x neighbouring_k x feat_dim ###
            ### key_pts: bsz x n_keypts x 3;
            
            ''' select via nearest neighbours directly ''' 
            selected_tar_keypts_out, neighbouring_tar_keypts_idxes = self.select_target_local_features(key_pts, dst_key_pts, tar_keypts_out, neighbouring_tar_keypts_idxes=None, neighbouring_tar_k=self.neighbouring_tar_k)
            ### selected_tar_keypts_out: bsz x K x feat_dim x neighbouring_k ###
            selected_tar_keypts_out = selected_tar_keypts_out.contiguous().transpose(2, 3).contiguous() 
            ''' select via nearest neighbours directly '''
            
            rnd_net = net
            
            tar_out = tar_out.unsqueeze(1).expand(-1, K, -1, -1) 
            loss_prior_tar = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            loss_log_pz_tar = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            loss_entropy_tar = torch.zeros((1,), dtype=torch.float32).cuda().mean()
        
        if not self.use_prob_src:
            loss_prior = loss_prior_tar
            loss_log_pz = loss_log_pz_tar
            loss_entropy = loss_entropy_tar #### loss_entropy and tar... ###

        ##### pool net for glboal features ######
        ### net and netout ###
        net = torch.max(net, 2, keepdim=True)[0] ### net with tar_out
        ### 
        key_fea = net.view(B * K, 64, 1) ### reshape to keypoints' features ### key_features 

        #### keypoints, keypoint basis, keypoints basis; num_basis x 3 ####
        if not self.wo_keypts_abs:
            net = torch.cat([key_fea, src_verts.view(B * K, 3, 1)], 1) #### key_points #### keypoints
        else:
            net = key_fea
        net = F.relu(self.bn31(self.conv31(net)))
        net = F.relu(self.bn32(self.conv32(net)))
        
        
        
        rnd_net = torch.max(rnd_net, 2, keepdim=True)[0] ### net with tar_out
        rnd_key_fea = rnd_net.view(B * K, 64, 1) ### reshape to keypoints' features ### key_features 
        #### keypoints, keypoint basis, keypoints basis; num_basis x 3 ####
        if not self.wo_keypts_abs:
            rnd_net = torch.cat([rnd_key_fea, key_pts.view(B * K, 3, 1)], 1) #### key_points #### keypoints
        else:
            rnd_net = rnd_key_fea
        rnd_net = F.relu(self.bn31(self.conv31(rnd_net)))
        rnd_net = F.relu(self.bn32(self.conv32(rnd_net)))
        
        
        ##### basis as offsets #####
        basis = self.conv33(net).view(B, K * 3, self.num_basis)[..., 0] ### keypts delta transformations ###
        basis = basis.contiguous().view(B, K, 3).contiguous()
        ##### basis as offsets #####
        basis = basis * self.coef_multiplier ### a smaller offset value ###
        
        
        
        ##### basis as offsets #####
        rnd_basis = self.conv33(rnd_net).view(B, K * 3, self.num_basis)[..., 0] ### keypts delta transformations ###
        rnd_basis = rnd_basis.contiguous().view(B, K, 3).contiguous()
        ##### basis as offsets #####
        rnd_basis = rnd_basis * self.coef_multiplier ### a smaller offset value ###
        
        
        coef_range = torch.ones((B, self.num_basis, 2), dtype=torch.float32).cuda() * self.coef_multiplier
        coef_range[:, :, 0] = coef_range[:, :, 0] * -1.0
        
        

        # def_key_pts = key_pts + basis ### B x K x 3
        # def_pc = torch.bmm(w_pc, def_key_pts)
        def_key_pts = src_verts + basis
        def_pc = def_key_pts
        
        
        
        rnd_def_key_pts = src_verts + rnd_basis
        rnd_def_pc = rnd_def_key_pts
        # rnd_def_key_pts = key_pts + rnd_basis ### B x K x 3
        # rnd_def_pc = torch.bmm(w_pc, rnd_def_key_pts)

        #### def_pc and tar_pc ####
        # cd_loss = chamfer_distance(def_pc, tar_pc)
        tar_verts = tar_verts[0].unsqueeze(0)
        cd_loss = chamfer_distance(def_pc, tar_verts)

        #### def_key_pts ####
        sample_def_key_pts = def_key_pts
        sample_def_pc = sample_def_key_pts
        

        sym_mult_tsr = torch.tensor([1, 1, 1]).cuda()
        sym_mult_tsr[self.symmetry_axis] = sym_mult_tsr[self.symmetry_axis] * -1.0
        sample_def_pc_sym = sample_def_pc * sym_mult_tsr

        sym_loss = chamfer_distance(sample_def_pc, sample_def_pc_sym) #### sym_loss for the deformed shape

        rnd_def_pc_sym = rnd_def_pc * sym_mult_tsr
        rnd_sym_loss = chamfer_distance(rnd_def_pc, rnd_def_pc_sym) #### sym_loss for the deformed shape

        sym_loss = (sym_loss + rnd_sym_loss) / 2.
        
        
        
        ### basis: bsz x n_keypts x 3 ###
        basis = basis.contiguous().unsqueeze(1).repeat(1, self.num_basis, 1, 1).contiguous() #### 
        basis = basis.view(B, self.num_basis, -1).contiguous()
        tot_sampled_def_key_pts = [def_key_pts, rnd_def_key_pts]
        coef = torch.ones((B, self.num_basis), dtype=torch.float32).cuda()
        
        if deform_net is None:
            cat_cd_loss = cd_loss.clone()
            cat_basis = basis.clone()
            cat_coef = coef.clone()
            cat_tot_sampled_def_key_pts = tot_sampled_def_key_pts
            cat_sym_loss = sym_loss.clone()
        
        
        return def_key_pts, def_pc, cd_loss, basis, coef, tot_sampled_def_key_pts, sym_loss, coef_range, loss_prior, src_pp_topk_idxes, tar_pp_topk_idxes, \
          cat_def_key_pts, cat_def_pc, cat_cd_loss, cat_basis, cat_coef, cat_tot_sampled_def_key_pts, cat_sym_loss


    def sample_bak(self, src_pc, tar_pc, key_pts, dst_key_pts, w_pc,  src_verts, src_edges, src_dofs, tar_verts, tar_edges, tar_dofs, deform_net=None):
        #### B, N, _ 
        #### network target driven ####
        B, N, _ = src_pc.shape
        
        if self.opt.hier_stage == 1:
            with torch.no_grad():
                # print("here!")
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
        
          
        cat_def_key_pts =def_key_pts.clone()
        cat_def_pc = def_pc.clone()
          
        src_pc = def_pc
        key_pts = def_key_pts #### def_kty_ptss ###
        
        ### downsample pc ###
        n_samples = 512
        n_samples = 1024
        n_samples = self.n_samples
        bsz, N = src_pc.size(0), src_pc.size(1)
        src_fps_idx = farthest_point_sampling(pos=src_pc[:, :, :3], n_sampling=n_samples) ### smapling for features xxx
        ### src_pc: bsz x N x 3
        src_pc_downsampled = src_pc.contiguous().view(bsz * N, 3)[src_fps_idx].contiguous().view(bsz, n_samples, 3).contiguous()
        
        # w_pc: bsz x N x K ### w_pc: bsz x N x K
        w_pc_expand = w_pc.contiguous().view(bsz * N, -1)[src_fps_idx].contiguous().view(bsz, n_samples, -1).contiguous()
        
        ### tar_fps_idx ###
        tar_fps_idx = farthest_point_sampling(pos=tar_pc[:, :, :3], n_sampling=n_samples) ### smapling for features xxx
        ### src_pc: bsz x N x 3
        tar_pc_downsampled = tar_pc.contiguous().view(bsz * N, 3)[tar_fps_idx].contiguous().view(bsz, n_samples, 3).contiguous()
        
        
        
        N = n_samples
        # bz x n_samples x pos_dim
        
        if not self.use_graphconv:
            if not self.use_pp_tar_out_feat:
                #### source out; source global ####
                src_out, src_global = self.pointnet(src_pc, False)
                #### target global #### #### tar_pc #####
                # tar_out, tar_global = self.pointnet(tar_pc, False) ### target pc ##3 flow predition and composition? 
                tar_out, tar_global = self.tar_pointnet(tar_pc, False)
                
                src_pp_topk_idxes = None
                tar_pp_topk_idxes = None
            else:
                ##### src_pp_topk_idxes ##### original-pointnet #####
                # src_out, src_global, src_pp_topk_idxes = self.pointnet(None, src_pc, False)
                # tar_out, tar_global, tar_pp_topk_idxes = self.tar_pointnet(None, tar_pc, False)
                
                # src_out, src_pp_topk_idxes = self.pointnet(src_pc)
                # tar_out, tar_pp_topk_idxes = self.tar_pointnet(tar_pc)

                src_out, src_pp_topk_idxes = self.pointnet(src_pc_downsampled)
                tar_out, tar_pp_topk_idxes = self.tar_pointnet(tar_pc_downsampled)
                # print(f"src_global: {src_global.size()}, tar_global: {tar_global.size()}")
        else: ### src_mesh_features ###
            src_mesh_features = self.graph_conv_net(src_verts, src_edges, src_dofs)
            src_pc_features = edge_propagation.get_keypoints_features(src_pc, src_verts, src_mesh_features)
            src_out = src_pc_features
            src_global = torch.max(src_out, dim=-1)[0]
            
            tar_mesh_features = self.tar_graph_conv_net(tar_verts, tar_edges, tar_dofs)
            tar_pc_features = edge_propagation.get_keypoints_features(tar_pc, tar_verts, tar_mesh_features)
            tar_out = tar_pc_features
            tar_global = torch.max(tar_out, dim=-1)[0]
            
            src_pp_topk_idxes = None
            tar_pp_topk_idxes = None
        
        
        src_out = F.relu(self.bn11(self.conv11(src_out)))
        src_out = F.relu(self.bn12(self.conv12(src_out)))
        src_out = F.relu(self.bn13(self.conv13(src_out))) #### src_pp_features #### src_out ####
        ### network and keypoints 

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
        
        
        # w_pc1, key_pts1; keypoints, source out, weights from keypoints to pcs 
        net = torch.cat([src_out, w_pc1, key_pts1], 2).view(B * K, 68, N) #### 

        net = F.relu(self.bn21(self.conv21(net)))
        net = self.bn22(self.conv22(net))
        
        if self.tar_basis > 0: # use target for deformation basis prediction ###
            ##### get tar out #####
            ### first set of convoltuions, conv11, conv12, conv13
            # tar_out = torch.max(tar_out, dim=2, keepdim=True)[0] ###
            tar_out = F.relu(self.tar_bn11(self.tar_conv11(tar_out)))
            tar_out = F.relu(self.tar_bn12(self.tar_conv12(tar_out)))
            tar_out = F.relu(self.tar_bn13(self.tar_conv13(tar_out))) ## tar_out: bsz x dim x N
            
            if not self.use_pp_tar_out_feat:
                tar_out = torch.max(tar_out, dim=2, keepdim=True)[0] ###
            ##### get tar_out feautres #####
            
            if self.use_prob:
                #### tar_out: B x dim x N
                tar_out_dim = tar_out.size(1)
                ### 
                tar_out_mu, tar_out_log_sigma = tar_out[:, :tar_out_dim // 2], tar_out[:, tar_out_dim // 2: ]
                z = utils.reparameterize_gaussian(mean=tar_out_mu, logvar=tar_out_log_sigma)  # (B, F) ## global 
                log_pz = utils.standard_normal_logprob(z).sum(dim=1) 
                ##### maximize the entropy of q_x(z); gaussian_entropy #####
                entropy = utils.gaussian_entropy(logvar=tar_out_log_sigma) ### 
                loss_prior = (-log_pz - entropy).mean()
                # kl_weight = 0.001
                # loss_prior = kl_weigh
                tar_out = z
                tar_out = torch.randn_like(tar_out)
            else:
                tar_out_dim = tar_out.size(1)
                tar_out_mu, tar_out_log_sigma = tar_out[:, :tar_out_dim // 2], tar_out[:, tar_out_dim // 2: ]
                tar_out = tar_out_mu
                loss_prior = torch.zeros((1,), dtype=torch.float32).cuda().mean()
        
            # tar_out_sample
            ### each point ###
            # net = torch.max(net, 2, keepdim=True)[0] ### feature for each keypoint 
            # ### net: bsz x K x dim
            # net = net.contiguous().view(B, K, net.size(-2)) ### bsz x K x dim
            # tar_out = tar_out.unsqueeze(1).expand(-1, K, -1, -1)  ### tar_out: B x K x 64 x N
            # net = net.contiguous().unsqueeze(-1).repeat(1, 1, 1, N).contiguous() ### bsz x K x dim x N
            # # print(f"net: {net.size()}, tar_out: {tar_out.size()}")
            # net = torch.cat([net, tar_out], dim=2).view(B * K, -1, N) ### bsz x K x (dim + 64) x N
            # net = F.relu(self.tar_bn21(self.tar_conv21(net)))
            # net = self.tar_bn22(self.tar_conv22(net)) ### bsz x K x 64 x N
            ### each point ###
            
            # net = torch.max(net, 2, keepdim=True)[0]
            ### global feature ###
            ### tar_out: B x dim x N ###
            
            net = torch.max(net, 2, keepdim=True)[0] ### feature for each keypoint 
            net = net.contiguous().view(B, K, net.size(-2)) ### bsz x K x dim
            ### tar_out for deformation direction ### ### target_out ### 
            ### target out; key_pts and tar_key_pts ...###
            
            # --> tar_out : bsz x n_pts x feat_dim 
            tar_out_trans = tar_out.contiguous().transpose(-1, -2).contiguous()
            ### deformation factorization ### ### tar_out: bsz x n_pts x feat_dim ###
            tar_keypts_out = self.select_keypts_features(dst_key_pts, tar_pc_downsampled, tar_out_trans) #### bsz x ## tar keypts out...
            ### selected_tar_keypts_out: bsz x K x neighbouring_k x feat_dim ###
            ### key_pts: bsz x n_keypts x 3;
            selected_tar_keypts_out, neighbouring_tar_keypts_idxes = self.select_target_local_features(key_pts, dst_key_pts, tar_keypts_out, neighbouring_tar_keypts_idxes=None, neighbouring_tar_k=16)
            ### selected_tar_keypts_out: bsz x K x feat_dim x neighbouring_k ###
            selected_tar_keypts_out = selected_tar_keypts_out.contiguous().transpose(2, 3).contiguous() 
            
            ### selected_tar_keypts_out

            tar_out = tar_out.unsqueeze(1).expand(-1, K, -1, -1)  ### tar_out: B x K x 64 x N; tar_out; tar_out.
            
            if not self.use_pp_tar_out_feat:
                #### reparamterize this term to make it like VAE?
                net = net.contiguous().unsqueeze(-1).repeat(1, 1, 1, 1).contiguous() ### bsz x K x dim x N
                net = torch.cat([net, tar_out], dim=2).view(B * K, -1, 1) ### bsz x K x (dim + 64) x N
            else:
                # net = net.contiguous().unsqueeze(-1).repeat(1, 1, 1, N).contiguous() ### bsz x K x dim x N
                # net = torch.cat([net, tar_out], dim=2).view(B * K, -1, N) ### bsz x K x (dim + 64) x N
                ##### only for nearesst neighbooring_k features #####
                net = net.contiguous().unsqueeze(-1).repeat(1, 1, 1, selected_tar_keypts_out.size(-1)).contiguous() ### bsz x K x dim x N
                net = torch.cat([net, selected_tar_keypts_out], dim=2).view(B * K, -1, selected_tar_keypts_out.size(-1)) ### bsz x K x (dim + 64) x N
                ##### only for nearesst neighbooring_k features #####

            # net = torch.cat([net, tar_out], dim=2).view(B * K, -1, 1) ### bsz x K x (dim + 64) x N
            net = F.relu(self.tar_bn21(self.tar_conv21(net))) ### net and net...
            net = self.tar_bn22(self.tar_conv22(net)) ### bsz x K x 64 x N
            ### global feature ###
        else:
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
        ### net and netout ###
        net = torch.max(net, 2, keepdim=True)[0] ### net with tar_out
        ### 
        key_fea = net.view(B * K, 64, 1) ### reshape to keypoints' features ### key_features 

        #### keypoints, keypoint basis, keypoints basis; num_basis x 3 ---> we need a local backbone; 
        net = torch.cat([key_fea, key_pts.view(B * K, 3, 1)], 1) #### key_points #### keypoints
        net = F.relu(self.bn31(self.conv31(net)))
        net = F.relu(self.bn32(self.conv32(net)))
        
        basis = self.conv33(net).view(B, K * 3, self.num_basis).transpose(1, 2)
        
        ###### 
        neighbouring_keypts_indicators = self.select_handles_via_radius(key_pts, radius=0.5, nearest_k=8)
        ### neighbouring_keypts_indicators: bsz x n_keypts x n_keypts
        # print(neighbouring_keypts_indicators[0, 0,])
        basis = self.select_local_handles(basis, neighbouring_keypts_indicators) #### basis: bsz x num_basisx (K * 3) ### local features...
        
        
        if self.pred_type == "basis":
            basis = basis / torch.clamp(basis.norm(p=2, dim=-1, keepdim=True), min=1e-6)
        elif self.pred_type == "offset":
            basis = basis.contiguous().view(B, self.num_basis, K, 3).contiguous()
            basis = basis / torch.clamp(basis.norm(p=2, dim=-1, keepdim=True), min=1e-5)
            basis = basis.contiguous().view(B, self.num_basis, K * 3).contiguous() ##### basis with unit length-vectors
        else:
            raise ValueError(f"Unrecognized pred_type: {self.pred_type}.")
          
        

        key_fea_range = key_fea.view(
            B, K, 64, 1).expand(-1, -1, -1, self.num_basis).transpose(1, 3) #### B, self.num_basis, K, 64
        key_pts_range = key_pts.view(
            B, K, 3, 1).expand(-1, -1, -1, self.num_basis).transpose(1, 3) 
        basis_range = basis.view(B, self.num_basis, K, 3).transpose(2, 3)


        if self.pred_type == "basis":
            ### get range ### coef range.. ###
            coef_range = torch.cat([key_fea_range, key_pts_range, basis_range], 2).view(
                B * self.num_basis, 70, K)
            coef_range = F.relu(self.bn71(self.conv71(coef_range))) ### coef_range
            coef_range = F.relu(self.bn72(self.conv72(coef_range))) ### coef_range and ...
            coef_range = self.conv73(coef_range)
            coef_range = torch.max(coef_range, 2, keepdim=True)[0]
            # coef_range = coef_range.view(B, self.num_basis, 2) * 0.02 ### 
            coef_range = coef_range.view(B, self.num_basis, 2) * self.coef_multiplier #  0.1 
            # coef_range = coef_range.view(B, self.num_basis, 2) * 0.5
            coef_range[:, :, 0] = coef_range[:, :, 0] * -1 #### coef_range...
        elif self.pred_type == "offset":
            coef_range = torch.cat([key_fea_range, key_pts_range, basis_range], 2).view(
                B * self.num_basis, 70, K)
            coef_range = F.relu(self.bn71(self.conv71(coef_range))) ### coef_range
            coef_range = F.relu(self.bn72(self.conv72(coef_range))) ### coef_range and ...
            coef_range = self.conv73(coef_range)
            coef_range = coef_range.view(B, self.num_basis, 2, K) * self.coef_multiplier #  0.1 
            coef_range = coef_range.contiguous().transpose(-1, -2).contiguous()
            coef_range[:, :, :, 0] = coef_range[:, :, :, 0] * -1 #### coef_range: B, num_basis, K, 2
        else:
            raise ValueError(f"Unrecognized pred_type: {self.pred_type}.")

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
            
            # key_fea_expanded = key_fea.contiguous().repeat(1, 1, N).contiguous()
            
            key_fea_expanded = key_fea.contiguous().repeat(1, 1, selected_tar_keypts_out.size(-1)).contiguous()
            # tar_out: B fea_dim N
            # print(f"tar_out: {tar_out.size()}")
            # tar_out_expanded = tar_out.contiguous().unsqueeze(1).contiguous().repeat(1, K, 1, 1).contiguous().view(B * K, tar_out.size(1), N)
            
            # tar_out_expanded = tar_out.contiguous().view(B * K, tar_out.size(1), N)
            
            selected_tar_keypts_out = selected_tar_keypts_out.contiguous().view(B * K, selected_tar_keypts_out.size(-2), selected_tar_keypts_out.size(-1)) #### selected_tar_keypts_out...
            tar_out_expanded = selected_tar_keypts_out ###### seelcted_tar_keypts_out #######
            
            
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
        
        
        
        if self.pred_type == "basis":
            coef = (coef * coef_range[:, :, 0] + (1 - coef)
                    * coef_range[:, :, 1]).view(B, 1, self.num_basis)
        elif self.pred_type == "offset":
            coef = (coef * coef_range[:, :, :, 0] + (1 - coef)
                    * coef_range[:, :, :, 1])
        else:
            raise ValueError(f"Unrecognized pred_type: {self.pred_type}.")

        ### basis: --> B x n_basis x K x 3; coef: B x 1 x n_basis ---> just a linear conbination of basis handles
        ### basis
        
        
        if self.pred_type == "basis":
            def_key_pts = key_pts + torch.bmm(coef, basis).view(B, K, 3)
        elif self.pred_type == "offset":
            def_key_pts = key_pts + basis[:, 0].view(B, K, 3) * coef[:, 0].unsqueeze(-1)
        else:
            raise ValueError(f"Unrecognized pred_type: {self.pred_type}.")
        
        def_pc = torch.bmm(w_pc, def_key_pts)

        # print(f"def_pc: {def_pc.size()}, tar_pc: {tar_pc.size()}")
        cd_loss = chamfer_distance(def_pc, tar_pc)

        # ratio = torch.rand((B, self.num_basis)).cuda()
        ratio = torch.rand_like(coef)


        if self.pred_type == "basis":
            tot_sampled_def_key_pts = []
            for i_s in range(10):
                ratio = torch.rand((B, self.num_basis)).cuda()
                # print(f"coef_range: {coef_range.size()}, ratio: {ratio.size()}")
                sample_coef = (ratio * coef_range[:, :, 0] + (1 - ratio)
                            * coef_range[:, :, 1]).view(B, 1, self.num_basis)
                sample_def_key_pts = key_pts + \
                    torch.bmm(sample_coef, basis).view(B, K, 3)
                tot_sampled_def_key_pts.append(sample_def_key_pts)
        elif self.pred_type == "offset":
            
            sample_coef = (ratio * coef_range[..., 0] + (1 - ratio)
                        * coef_range[..., 1])
            sample_def_key_pts = key_pts + basis[:, 0].view(B, K, 3) * sample_coef[:, 0].unsqueeze(-1)
        else:
            raise ValueError(f"Unrecognized pred_type: {self.pred_type}.")


        sample_def_pc = torch.bmm(w_pc, sample_def_key_pts)
        
        # sample_def_pc_sym = sample_def_pc * \
        #     torch.tensor([-1, 1, 1]).cuda()  # for shapenet shapes
        #### 
        sym_mult_tsr = torch.tensor([1, 1, 1]).cuda()
        sym_mult_tsr[self.symmetry_axis] = sym_mult_tsr[self.symmetry_axis] * -1.0
        sample_def_pc_sym = sample_def_pc * sym_mult_tsr
        # \
        #     torch.tensor([-1, 1, 1]).cuda()  # for shapenet shapes
        # sample_def_pc_sym = sample_def_pc * \
        #     torch.tensor([1, -1, 1]).cuda()  # for shapenet shapes
        sym_loss = chamfer_distance(sample_def_pc, sample_def_pc_sym) #### sym_loss for the deformed shape
        
        
        # def_key_pts, def_pc, cat_cd_loss, cat_basis, cat_coef, tot_sampled_def_key_pts, cat_sym_loss, 
        
        
        # cd_loss = cd_loss + cat_cd_loss
        
        if deform_net is None:
            cat_cd_loss = cd_loss.clone()
            cat_basis = basis.clone()
            cat_coef = coef.clone()
            cat_tot_sampled_def_key_pts = tot_sampled_def_key_pts
            cat_sym_loss = sym_loss.clone()
        
        
        return def_key_pts, def_pc, cd_loss, basis, coef, tot_sampled_def_key_pts, sym_loss, coef_range, loss_prior, src_pp_topk_idxes, tar_pp_topk_idxes, \
          cat_def_key_pts, cat_def_pc, cat_cd_loss, cat_basis, cat_coef, cat_tot_sampled_def_key_pts, cat_sym_loss