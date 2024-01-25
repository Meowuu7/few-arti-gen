from multiprocessing.sharedctypes import Value
import torch
import torch.nn.functional as F
import torch.nn as nn
from src.networks.pointnet_utils import pointnet_encoder
# from losses import chamfer_distance
from src.common_utils.losses import chamfer_distance_raw as chamfer_distance
import src.common_utils.utils as utils
from src.networks.pointnet2 import PointnetPP
import src.networks.edge_propagation as edge_propagation
from src.common_utils.data_utils_torch import farthest_point_sampling, batched_index_select


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
        self.opt = opt
        self.use_prob = self.opt.use_prob
        self.tar_basis = self.opt.tar_basis
        self.coef_multiplier = self.opt.coef_multiplier
        self.n_layers = self.opt.n_layers
        
        self.num_basis = num_basis
        # self.pred_offset = self.opt.pred_offset
        self.pred_type = self.opt.pred_type
        self.neighbouring_k = opt.neighbouring_k
        
        self.n_samples = opt.n_samples
        self.symmetry_axis = opt.symmetry_axis
        
        
        print(f"Using symmetry_axis: {self.symmetry_axis}")
        
        print(f"prediction type: {self.pred_type}")
        self.use_pointnet2 = self.opt.use_pointnet2
        print(f"whether to use pointnet2: {self.use_pointnet2}")
        self.use_graphconv = self.opt.use_graphconv
        print(f"whether to use graphconv: {self.use_graphconv}")
        
        #### pp_tar_out_feat ####
        self.use_pp_tar_out_feat = self.opt.use_pp_tar_out_feat
        
        print(f"whether to use pp_tar_out_feat: {self.use_pp_tar_out_feat}")
        
        if not self.use_graphconv:
            if self.use_pointnet2:
                self.pointnet = PointnetPP(in_feat_dim=3, use_light_weight=False, args=opt)
                self.tar_pointnet = PointnetPP(in_feat_dim=3, use_light_weight=False, args=opt)
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


    def forward(self, src_pc, tar_pc, key_pts, w_pc,  src_verts, src_edges, src_dofs, tar_verts, tar_edges, tar_dofs):
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
                #### 
                # src_out, src_global = self.pointnet(None, src_pc, False)
                # tar_out, tar_global = self.tar_pointnet(None, tar_pc, False)

                ##### src_pp_topk_idxes ##### original-pointnet #####
                # src_out, src_global, src_pp_topk_idxes = self.pointnet(None, src_pc, False)
                # tar_out, tar_global, tar_pp_topk_idxes = self.tar_pointnet(None, tar_pc, False)
                
                # src_out, src_pp_topk_idxes = self.pointnet(src_pc)
                # tar_out, tar_pp_topk_idxes = self.tar_pointnet(tar_pc)
                
                
                src_out, src_pp_topk_idxes = self.pointnet(src_pc_downsampled)
                tar_out, tar_pp_topk_idxes = self.tar_pointnet(tar_pc_downsampled)
                # print(f"src_global: {src_global.size()}, tar_global: {tar_global.size()}")
        else:
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
        src_out = F.relu(self.bn13(self.conv13(src_out)))
        ### network and keypoints 

        _, K, _ = key_pts.shape
        key_pts1 = key_pts.unsqueeze(-1).expand(-1, -1, -1, N)  # B K 3 N
        #### w_pc --> w_pc1 --> transpose weights for further computation
        
        #### using no downssample ###
        # w_pc1 = w_pc.transpose(2, 1).unsqueeze(2)  # B K 1 N
        #### for downsample #### 
        w_pc1 = w_pc_expand.transpose(2, 1).unsqueeze(2)
        src_out = src_out.unsqueeze(1).expand(-1, K, -1, -1)  # B K 64 N
        ### what should be cat here --> src_out; w_pc1; key_pts1; 
        ### for each keypoint, 1) keypoints-to-surface_points weights; 2) key_points values; 3) expanded source points 
        
        # print(f"for net cat: {src_out.size()}, w_pc1: {w_pc1.size()}, key_pts1: {key_pts1.size()}")
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
            ### tar_out for deformation direction ### ### target_out
            tar_out = tar_out.unsqueeze(1).expand(-1, K, -1, -1)  ### tar_out: B x K x 64 x N; tar_out; tar_out.
            
            if not self.use_pp_tar_out_feat:
                #### reparamterize this term to make it like VAE?
                net = net.contiguous().unsqueeze(-1).repeat(1, 1, 1, 1).contiguous() ### bsz x K x dim x N
                net = torch.cat([net, tar_out], dim=2).view(B * K, -1, 1) ### bsz x K x (dim + 64) x N
            else:
                net = net.contiguous().unsqueeze(-1).repeat(1, 1, 1, N).contiguous() ### bsz x K x dim x N
                net = torch.cat([net, tar_out], dim=2).view(B * K, -1, N) ### bsz x K x (dim + 64) x N
            # net = torch.cat([net, tar_out], dim=2).view(B * K, -1, 1) ### bsz x K x (dim + 64) x N
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
            tar_out = tar_out.unsqueeze(1).expand(-1, K, -1, -1) 
            loss_prior = torch.zeros((1,), dtype=torch.float32).cuda().mean()
        
        
        ##### pool net for glboal features ######
        ### net and netout ###
        net = torch.max(net, 2, keepdim=True)[0] ### net with tar_out
        ### 
        key_fea = net.view(B * K, 64, 1) ### reshape to keypoints' features ### key_features 

        #### keypoints, keypoint basis, keypoints basis; num_basis x 3 ---> we need a local backbone; 
        net = torch.cat([key_fea, key_pts.view(B * K, 3, 1)], 1) #### key_points #### keypoints
        net = F.relu(self.bn31(self.conv31(net))) #### net
        net = F.relu(self.bn32(self.conv32(net)))
        basis = self.conv33(net).view(B, K * 3, self.num_basis).transpose(1, 2)
        
        if self.pred_type == "basis":
            basis = basis / basis.norm(p=2, dim=-1, keepdim=True)
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

        if self.pred_type == "basis":
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
        elif self.pred_type == "offset":
            net = torch.cat([key_fea, basis1, key_pts2], 2).view( #### basis1, key_pts2
                B * self.num_basis, 3 + 128 + 3, K)
            net = F.relu(self.bn51(self.conv51(net)))
            net = F.relu(self.bn52(self.conv52(net)))
            net = self.bn53(self.conv53(net))
            net = net.view(B * self.num_basis, 128, K)
            net = torch.cat([net, coef_range.contiguous().transpose(-1, -2).contiguous().view(B * self.num_basis, 2, K)], 1)
            net = F.relu(self.bn61(self.conv61(net)))
            net = F.relu(self.bn62(self.conv62(net)))
            ### or just the generation of coef ---> basis of deformation combinations ### 
            ### basis: basis; sigmoid net
            coef = self.sigmoid(self.conv63(net)).view(B, self.num_basis, K) ### how to combine such basis... ### how to combine such basis...
        else:
            raise ValueError(f"Unrecognized pred_type: {self.pred_type}.")


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
        return def_key_pts, def_pc, cd_loss, basis, coef, tot_sampled_def_key_pts, sym_loss, coef_range, loss_prior, src_pp_topk_idxes, tar_pp_topk_idxes


    def sample(self, src_pc, tar_pc, key_pts, dst_key_pts, w_pc, src_verts, src_edges, src_dofs, tar_verts, tar_edges, tar_dofs):
        #### B, N, _
        B, N, _ = src_pc.shape
        
        ###### pointnet ######
        # #### source out; source global ####
        # src_out, src_global = self.pointnet(src_pc, False)
        # #### target global ####
        # tar_out, tar_global = self.tar_pointnet(tar_pc, False) ### target pc ##3 flow predition and composition? 
        ###### pointnet ######
        
        
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
                # src_out, src_global = self.pointnet(None, src_pc, False)
                # tar_out, tar_global = self.tar_pointnet(None, tar_pc, False)
                ###### pointnet and tar_pointnet for per-point features ######
                src_out, src_global, src_pp_topk_idxes = self.pointnet(None, src_pc, False)
                tar_out, tar_global, tar_pp_topk_idxes = self.tar_pointnet(None, tar_pc, False)
                # print(f"src_global: {src_global.size()}, tar_global: {tar_global.size()}")
        else:
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
        src_out = F.relu(self.bn13(self.conv13(src_out)))
        ### network and keypoints 

        _, K, _ = key_pts.shape
        key_pts1 = key_pts.unsqueeze(-1).expand(-1, -1, -1, N)  # B K 3 N
        #### w_pc --> w_pc1 --> transpose weights for further computation
        w_pc1 = w_pc.transpose(2, 1).unsqueeze(2)  # B K 1 N
        src_out = src_out.unsqueeze(1).expand(-1, K, -1, -1)  # B K 64 N # B K 64 N 
        ### what should be cat here --> src_out; w_pc1; key_pts1; 
        ### for each keypoint, 1) keypoints-to-surface_points weights; 2) key_points values; 3) expanded source points 
        net = torch.cat([src_out, w_pc1, key_pts1], 2).view(B * K, 68, N) #### 

        net = F.relu(self.bn21(self.conv21(net)))
        net = self.bn22(self.conv22(net))
        
        if self.tar_basis > 0:
            ####  ####
            # tar_out = torch.max(tar_out, dim=2, keepdim=True)[0] ###
            tar_out = F.relu(self.tar_bn11(self.tar_conv11(tar_out)))
            tar_out = F.relu(self.tar_bn12(self.tar_conv12(tar_out)))
            tar_out = F.relu(self.tar_bn13(self.tar_conv13(tar_out)))
            tar_out = torch.max(tar_out, dim=2, keepdim=True)[0] ###
            
            #### z -- reparameterize_gaussian ####
            if self.use_prob:
                #### for training and prior learning ####
                # #### tar_out: B x dim x N
                tar_out_dim = tar_out.size(1)
                tar_out_mu, tar_out_log_sigma = tar_out[:, :tar_out_dim // 2], tar_out[:, tar_out_dim // 2: ]
                z = utils.reparameterize_gaussian(mean=tar_out_mu, logvar=tar_out_log_sigma)  # (B, F) ## global 
                log_pz = utils.standard_normal_logprob(z).sum(dim=1) 
                entropy = utils.gaussian_entropy(logvar=tar_out_log_sigma)  ## entropy 
                loss_prior = (-log_pz - entropy).mean() ### entropy; gaussian prior for the global feature ###
                # kl_weight = 0.001
                # loss_prior = kl_weigh
                tar_out = z
                #### for training and prior learning ####
                
                #### for random sampling ####
                tar_out = torch.randn_like(tar_out)
                #### for random sampling ####
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
            tar_out = tar_out.unsqueeze(1).expand(-1, K, -1, -1)  ### tar_out: B x K x 64 x N
            #### reparamterize this term to make it like VAE? 
            
            net = net.contiguous().unsqueeze(-1).repeat(1, 1, 1, 1).contiguous() ### bsz x K x dim x N
            net = torch.cat([net, tar_out], dim=2).view(B * K, -1, 1) ### bsz x K x (dim + 64) x N
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
            tar_out = tar_out.unsqueeze(1).expand(-1, K, -1, -1) 
            loss_prior = torch.zeros((1,), dtype=torch.float32).cuda().mean()
        
        net = torch.max(net, 2, keepdim=True)[0]
        
        ### 
        key_fea = net.view(B * K, 64, 1) ### reshape to keypoints' features

        net = torch.cat([key_fea, key_pts.view(B * K, 3, 1)], 1) #### key_points 
        net = F.relu(self.bn31(self.conv31(net))) #### net
        net = F.relu(self.bn32(self.conv32(net)))
        basis = self.conv33(net).view(B, K * 3, self.num_basis).transpose(1, 2)
        # basis = basis / basis.norm(p=2, dim=-1, keepdim=True) 
        
        ### basis: B x n_basis x (K x 3) ###
        
        if self.pred_type == "basis":
            basis = basis / basis.norm(p=2, dim=-1, keepdim=True)
        elif self.pred_type == "offset":
            basis = basis.contiguous().view(B, self.num_basis, K, 3).contiguous()
            basis = basis / torch.clamp(basis.norm(p=2, dim=-1, keepdim=True), min=1e-5)
            basis = basis.contiguous().view(B, self.num_basis, K * 3).contiguous() ##### basis with unit length-vectors
        else:
            raise ValueError(f"Unrecognized pred_type: {self.pred_type}.")

        key_fea_range = key_fea.view(
            B, K, 64, 1).expand(-1, -1, -1, self.num_basis).transpose(1, 3)
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

        # ### get range
        # coef_range = torch.cat([key_fea_range, key_pts_range, basis_range], 2).view(
        #     B * self.num_basis, 70, K)
        # coef_range = F.relu(self.bn71(self.conv71(coef_range)))
        # coef_range = F.relu(self.bn72(self.conv72(coef_range)))
        # coef_range = self.conv73(coef_range)
        # coef_range = torch.max(coef_range, 2, keepdim=True)[0]
        # # coef_range = coef_range.view(B, self.num_basis, 2) * 0.02 ### 
        # coef_range = coef_range.view(B, self.num_basis, 2) * self.coef_multiplier #  0.1 
        # # coef_range = coef_range.view(B, self.num_basis, 2) * 0.5
        # coef_range[:, :, 0] = coef_range[:, :, 0] * -1

        # src_tar = torch.cat([src_global, tar_global], 1).unsqueeze(
        #     1).expand(-1, K, -1).reshape(B * K, 2048, 1)        
        

        # ### to the source shape ###
        # ### keypoint features, srouce target fused features, and keypoint coordiantes ###
        # ### key_feature; source_target
        # key_fea = torch.cat([key_fea, src_tar, key_pts.view(B * K, 3, 1)], 1)
        # key_fea = F.relu(self.bn41(self.conv41(key_fea))) ### key_fea
        # key_fea = F.relu(self.bn42(self.conv42(key_fea))) ### key_fea
        # key_fea = F.relu(self.bn43(self.conv43(key_fea))) ### 
        
        
        if self.use_pp_tar_out_feat:
            # key_fea: B K feat_dim
            key_fea_expanded = key_fea.contiguous().repeat(1, 1, N).contiguous()
            # tar_out: B fea_dim N
            # tar_out_expanded = tar_out.contiguous().unsqueeze(1).contiguous().repeat(1, K, 1, 1).contiguous().view(B * K, tar_out.size(1), N)
            tar_out_expanded = tar_out.contiguous().view(B * K, tar_out.size(1), N)
            # key_fea_expanded: B K (fea_dim + tar_out_fea_dim) N
            # key_fea_expanded = torch.cat()
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
        basis1 = basis.view(B, self.num_basis, K, 3).transpose(2, 3)
        
        
        if self.pred_type == "basis":
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
        elif self.pred_type == "offset":
            net = torch.cat([key_fea, basis1, key_pts2], 2).view( #### basis1, key_pts2
                B * self.num_basis, 3 + 128 + 3, K)
            net = F.relu(self.bn51(self.conv51(net)))
            net = F.relu(self.bn52(self.conv52(net)))
            net = self.bn53(self.conv53(net))
            net = net.view(B * self.num_basis, 128, K)
            net = torch.cat([net, coef_range.contiguous().transpose(-1, -2).contiguous().view(B * self.num_basis, 2, K)], 1)
            net = F.relu(self.bn61(self.conv61(net)))
            net = F.relu(self.bn62(self.conv62(net)))
            ### or just the generation of coef ---> basis of deformation combinations ### 
            ### basis: basis; sigmoid net
            coef = self.sigmoid(self.conv63(net)).view(B, self.num_basis, K) ### how to combine such basis... ### how to combine such basis...
        else:
            raise ValueError(f"Unrecognized pred_type: {self.pred_type}.")

        # net = torch.cat([key_fea, basis1, key_pts2], 2).view(
        #     B * self.num_basis, 3 + 128 + 3, K)

        ### target driven prob... ###
        # net = F.relu(self.bn51(self.conv51(net)))
        # net = F.relu(self.bn52(self.conv52(net)))
        # net = self.bn53(self.conv53(net))

        # net = torch.max(net, 2, keepdim=True)[0]
        # net = net.view(B * self.num_basis, 128, 1)

        # net = torch.cat([net, coef_range.view(B * self.num_basis, 2, 1)], 1)
        # net = F.relu(self.bn61(self.conv61(net)))
        # net = F.relu(self.bn62(self.conv62(net)))
        # coef = self.sigmoid(self.conv63(net)).view(B, self.num_basis)


        if self.pred_type == "basis":
            coef = (coef * coef_range[:, :, 0] + (1 - coef)
                    * coef_range[:, :, 1]).view(B, 1, self.num_basis)
        elif self.pred_type == "offset":
            coef = (coef * coef_range[:, :, :, 0] + (1 - coef)
                    * coef_range[:, :, :, 1])
        else:
            raise ValueError(f"Unrecognized pred_type: {self.pred_type}.")
        # coef = (coef * coef_range[:, :, 0] + (1 - coef)
        #         * coef_range[:, :, 1]).view(B, 1, self.num_basis)
        
        if self.pred_type == "basis":
            def_key_pts = key_pts + torch.bmm(coef, basis).view(B, K, 3)
        elif self.pred_type == "offset":
            def_key_pts = key_pts + basis[:, 0].view(B, K, 3) * coef[:, 0].unsqueeze(-1)
        else:
            raise ValueError(f"Unrecognized pred_type: {self.pred_type}.")

        # def_key_pts = key_pts + torch.bmm(coef, basis).view(B, K, 3) ### coef basis
        def_pc = torch.bmm(w_pc, def_key_pts)

        cd_loss = chamfer_distance(def_pc, tar_pc)

        ratio = torch.rand((B, self.num_basis)).cuda()
        
        
        # sample_coef = (ratio * coef_range[:, :, 0] + (1 - ratio)
        #                * coef_range[:, :, 1]).view(B, 1, self.num_basis)
        # sample_def_key_pts = key_pts + \
        #     torch.bmm(sample_coef, basis).view(B, K, 3)
            
        if self.pred_type == "basis":
            
            ratio = torch.rand((B, self.num_basis)).cuda()
            # print(f"coef_range: {coef_range.size()}, ratio: {ratio.size()}")
            sample_coef = (ratio * coef_range[:, :, 0] + (1 - ratio)
                        * coef_range[:, :, 1]).view(B, 1, self.num_basis)
            sample_def_key_pts = key_pts + \
                torch.bmm(sample_coef, basis).view(B, K, 3)
        elif self.pred_type == "offset":
            sample_coef = (ratio * coef_range[..., 0] + (1 - ratio)
                        * coef_range[..., 1])
            sample_def_key_pts = key_pts + basis[:, 0].view(B, K, 3) * sample_coef[:, 0].unsqueeze(-1)
        else:
            raise ValueError(f"Unrecognized pred_type: {self.pred_type}.")
        
        
        sample_def_pc = torch.bmm(w_pc, sample_def_key_pts)
        sample_def_pc_sym = sample_def_pc * \
            torch.tensor([1, -1, 1]).cuda()  # for shapenet shapes
        sym_loss = chamfer_distance(sample_def_pc, sample_def_pc_sym) #### sym_loss for the deformed shape
        return def_key_pts, def_pc, cd_loss, basis, coef, sample_def_key_pts, sym_loss, coef_range, loss_prior, src_pp_topk_idxes, tar_pp_topk_idxes
