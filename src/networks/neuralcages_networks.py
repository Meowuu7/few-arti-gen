from __future__ import print_function
import warnings
from pprint import pprint
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data

from common_utils.data_utils_torch import farthest_point_sampling

from pointnet_utils import pointnet_encoder, PointFlowEncoder
# from pytorch_points.misc import logger
# from pytorch_points.network.layers import Conv1d, Linear
# from pytorch_points.network.pointnet2_modules import PointnetSAModuleMSG
# from pytorch_points.network.model_loss import (ChamferLoss, MeshLaplacianLoss,
#                                                PointEdgeLengthLoss,
#                                                PointLaplacianLoss, PointStretchLoss,
#                                                nndistance)
# from pytorch_points.network.geo_operations import (mean_value_coordinates,
#                                                    mean_value_coordinates_3D,
#                                                    normalize_point_batch_to_sphere)
# from pytorch_points.network.operations import faiss_knn

# from common import deform_with_MVC

class STN(nn.Module):
    def __init__(self, num_points = 2500, dim=3):
        super(STN, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        #self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, dim*dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #x = self.mp1(x)
        #print(x.size())
        x,_ = torch.max(x, 2)
        #print(x.size())
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = torch.eye(3, dtype=x.dtype, device=x.device).view(1,9).expand(batchsize,1)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class PointNet2feat(nn.Module):
    """
    pointcloud (B,3,N)
    return (B,bottleneck_size)
    """
    def __init__(self, dim=3, num_points=2048, num_levels=3, bottleneck_size=512, normalization=None):
        super().__init__()
        assert(dim==3)
        self.SA_modules = nn.ModuleList()
        self.postSA_mlp = nn.ModuleList()
        NPOINTS = []
        RADIUS = []
        MLPS = []
        start_radius = 0.2
        start_mlp = 24
        self.l_output = []
        for i in range(num_levels):
            NPOINTS += [num_points//4]
            num_points = num_points//4
            RADIUS += [[start_radius, ]]
            start_radius *= 2
            final_mlp = min(256, start_mlp*4)
            MLPS += [[[start_mlp, start_mlp*2, final_mlp], ]]
            start_mlp *= 2
            self.l_output.append(start_mlp)

        bottleneck_size_per_SA = bottleneck_size // len(MLPS)
        self.bottleneck_size = bottleneck_size_per_SA*len(MLPS)

        in_channels = 0
        for k in range(len(MLPS)):
            mlps = [[in_channels]+mlp for mlp in MLPS[k]]
            in_channels = 0
            for idx in range(len(MLPS[k])):
                in_channels += MLPS[k][idx][-1]
            self.SA_modules.append(
                PointnetSAModuleMSG(npoint=NPOINTS[k], radii=RADIUS[k], nsamples=[32,], mlps=mlps, normalization=normalization)
                )
            self.postSA_mlp.append(Conv1d(in_channels, bottleneck_size_per_SA, 1, normalization=normalization, activation="tanh"))

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, return_all=False):
        pointcloud = pointcloud.transpose(1,2).contiguous()
        li_xyz, li_features = self._break_up_pc(pointcloud)

        # B,C,N
        # l_xyz, l_features = [xyz], [li_features]
        l_xyz, l_features = [], []
        for i in range(len(self.SA_modules)):
            # Pointnetmodule + MLP + maxpool
            li_xyz, li_features = self.SA_modules[i](li_xyz, li_features)
            li_features_post = self.postSA_mlp[i](li_features)
            l_xyz.append(li_xyz)
            l_features.append(li_features_post)

        # max pool (B,4*#SA,1) all SAmodules
        # exclude the first None features
        global_code = torch.cat([torch.max(l_feat, dim=-1)[0] for l_feat in l_features], dim=1)

        l_features.append(global_code)
        l_xyz.append(None)
        if return_all:
            return l_features, l_xyz
        else:
            return global_code

class PointNetfeat3DCoded(nn.Module):
    def __init__(self, npoint=2500, nlatent=1024):
        """Encoder"""

        super(PointNetfeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, nlatent, 1)
        self.lin1 = nn.Linear(nlatent, nlatent)
        self.lin2 = nn.Linear(nlatent, nlatent)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(nlatent)
        self.bn4 = torch.nn.BatchNorm1d(nlatent)
        self.bn5 = torch.nn.BatchNorm1d(nlatent)

        self.npoint = npoint
        self.nlatent = nlatent

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        point_feat = self.bn3(self.conv3(x))
        x, _ = torch.max(point_feat, 2)
        x = x.view(-1, self.nlatent)
        x = F.relu(self.bn4(self.lin1(x).unsqueeze(-1)))
        x = F.relu(self.bn5(self.lin2(x.squeeze(2)).unsqueeze(-1)))
        return x.squeeze(2), point_feat


class UnetCageGen(nn.Module):
    """ ## unet-cage-gen ### location from encoder ###
    Receive sampled feature and location from encoder, for each point in the template,
    find k NN points in l_xyz, concatenate their features to get the code
    Params:
        template   Tensor (B,D,N)
        l_xyz      List(Tensor) of shape (B,N_l,D)
        l_features List(Tensor) of shape (B,C_l,N_l)
    Return:
        xyz        Tensor (B,D,N)
        feat_all   Tensor (B,sum_l(C_l), N) if return_aux
    """
    def __init__(self, bottleneck_size, dim=3, knn_k=3,
                 normalization=None, concat_prim=True, n_fold=2, feat_NN=False, **kwargs):
        super().__init__()
        self.decoder = MultiFoldPointGen(bottleneck_size, dim, n_fold=n_fold, normalization=normalization, concat_prim=concat_prim)
        self.feat_NN = feat_NN
        self.knn_k = knn_k


    def interpolate_features(self, query, points, feats, q_normals=None, p_normals=None):
        """
        compute knn point distance and interpolation weight
        :param
            query           (B,M,D)
            points          (B,N,D)
            normals         (B,N,D)
            feats           (B,C,N)
        :return
            distance    Bx1xNxK
            weight      Bx1xNxK
        """
        B, M, D = query.shape
        feats_t = feats.transpose(1,2).contiguous()

        # compute weights based on exponential
        grouped_points, grouped_idx, grouped_dist = faiss_knn(self.knn_k, query, points, NCHW=False)

        # dynamic variance mean_P(min_K distance)
        h = torch.mean(torch.min(grouped_dist, dim=2, keepdim=True)[0], dim=1, keepdim=True) + 1e-8
        # (B,M,K) TODO try linear correlation like in KPconv

        weight = torch.exp(-grouped_dist / (h / 2)).detach()
        sumW = torch.sum(weight, dim=2, keepdim=True)
        weight = weight / sumW

        import pdb; pdb.set_trace()
        # (B, M, K, C)
        grouped_feats_ref = torch.gather(feats_t.unsqueeze(1).expand(-1, M, -1, -1), 2, grouped_idx.unsqueeze(-1).expand(-1,-1,-1, feats_t.shape[-1]))
        # (B, C, M, K)
        grouped_feats = torch.gather(feats.unsqueeze(2).expand(-1, -1, M, -1), 3, grouped_idx.unsqueeze(1).expand(-1, feats.shape[1], -1, -1))
        print(torch.all(torch.eq(grouped_feats, grouped_feats_ref.permute([0, 3, 1, 2]))))
        # (B,C,M,K)
        weighted_feats = torch.sum(grouped_feats*weight.unsqueeze(1), dim=-1)

        return weighted_feats


    def forward(self, template, l_xyz, l_features, return_aux=False):
        B, D, N = template.shape
        template_t = template.transpose(1,2).contiguous()
        interpolated = []
        for i, xyz_feat in enumerate(zip(l_xyz, l_features)):
            # (B,N,3) and (B,C,N)
            xyz, feat = xyz_feat
            # expand global features
            if xyz is None:
                feat = feat.unsqueeze(-1).expand(-1, -1, N)
                interpolated += [feat]
                continue
            # merge neighbors with point distance+normal similarity
            feat = self.interpolate_features(template_t, xyz, feat, q_normals=None, p_normals=None)
            interpolated += [feat]

        # (B,sum(feat_l.shape[1]), M)
        feat_all = torch.cat(interpolated, dim=1)

        xyz = self.decoder(feat_all, template)
        if return_aux:
            return xyz, feat_all
        return xyz

class UnetDeformGen(UnetCageGen):
    """
    Params:
        template            Tensor (B,D,N)
        template_features   Tensor (B, sum_l(C_l), N_l) from UnetCageGen
        l_xyz               List(Tensor) of shape (B,N_l,D)
        l_features          List(Tensor) of shape (B,C_l,N_l)
    Return:
        xyz        Tensor (B,D,N)
        feat_all   Tensor (B,sum_l(C_l), N) if return_aux
    """
    def interpolate_features(self, query_feats, feats, points):
        """
        find the kNN in feature space, interpolate these feature with exponential weights
        :param
            query_feats (B,C,M)
            feats       (B,C,N)
            points      (B,N,D)
        :return
            weighted_feats (B,C,M)
            weighted_xyz   (B,M,dim)
        """
        B, C, M = query_feats.shape
        query_feats_t = query_feats.transpose(1,2).contiguous()
        feats_t = feats.transpose(1,2).contiguous()
        # compute weights based on exponential
        grouped_feats_t, grouped_idx, grouped_dist = faiss_knn(self.knn_k, query_feats_t, feats_t, NCHW=False)
        grouped_feats = grouped_feats_t.permute((0, 3, 1, 2))

        # dynamic variance mean_P(min_K distance)
        h = torch.mean(torch.min(grouped_dist, dim=2, keepdim=True)[0], dim=1, keepdim=True) + 1e-8
        # (B,M,K) TODO try linear correlation like in KPconv
        weight = torch.exp(-grouped_dist / (h / 2)).detach()

        sumW = torch.sum(weight, dim=2, keepdim=True)
        weight = weight / sumW
        weighted_feats = torch.sum(grouped_feats*weight.unsqueeze(1), dim=-1)

        grouped_xyz = torch.gather(points.unsqueeze(1).expand(-1, M, -1, -1), 2, grouped_idx.unsqueeze(-1).expand(-1, -1, -1, points.shape[-1]))
        weighted_xyz = torch.sum(grouped_xyz*weight.unsqueeze(-1), dim=2)

        return weighted_feats, weighted_xyz

    def forward(self, template, template_features, l_xyz, l_features, return_aux=False):
        B, D, N = template.shape
        template_t = template.transpose(1,2).contiguous()
        interpolated = []
        for i, feat_xyz_feat in enumerate(zip(template_features, l_xyz, l_features)):
            # (B,N,3) and (B,C,N)
            query_feat, xyz, feat = feat_xyz_feat
            # expand global features
            if xyz is None:
                feat = feat.unsqueeze(-1).expand(-1, -1, N)
                interpolated += [feat]
                continue
            # merge neighbors with feature distance
            feat, matched_xyz = self.interpolate_features(query_feat, feat, xyz, q_normals=None, p_normals=None)
            interpolated += [feat]

        # (B,sum(feat_l.shape[1])+dim, M)
        feat_all = torch.cat(interpolated+[matched_xyz.transpose(1,2)], dim=1)

        xyz = self.decoder(feat_all, template)
        if return_aux:
            return xyz, feat_all
        return xyz

# encoding input points
class PointNetfeat(nn.Module):
    def __init__(self, dim=3, num_points=2500, global_feat=True, trans=False, bottleneck_size=512, activation="relu", normalization=None):
        super().__init__()
        self.conv1 = Conv1d(dim, 64, 1, activation=activation, normalization=normalization)
        # self.stn_embedding = STN(num_points = num_points, K=64)
        self.conv2 = Conv1d(64, 128, 1, activation=activation, normalization=normalization)
        self.conv3 = Conv1d(128, bottleneck_size, 1, activation=None, normalization=normalization)
        #self.mp1 = torch.nn.MaxPool1d(num_points)

        self.trans = trans
        #self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x):
        batchsize = x.size()[0]
        if self.trans:
            trans = self.stn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans)
            x = x.transpose(2,1)
        x = self.conv1(x)
        pointfeat = x
        x = self.conv2(x)
        x = self.conv3(x)
        x,_ = torch.max(x, dim=2)
        if self.trans:
            if self.global_feat:
                return x, trans
            else:
                x = x.view(batchsize, -1, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans
        else:
            return x

class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size, out_dim, prim_dim, normalization=None, concat_prim=False):
        """
        param:
            cat_prim: keep concatenate atlas coordinate to the features
        """
        super(PointGenCon, self).__init__()
        self.concat_prim = concat_prim
        if concat_prim:
            self.layers = nn.ModuleList([
                # Conv1d(bottleneck_size+prim_dim, bottleneck_size//2, 1, activation="lrelu", normalization=normalization),
                # Conv1d(bottleneck_size//2+prim_dim, bottleneck_size//4, 1, activation="lrelu", normalization=normalization),
                # Conv1d(bottleneck_size//4+prim_dim, out_dim, 1, activation=None, normalization=None),
                
                nn.Conv1d(bottleneck_size+prim_dim, bottleneck_size//2, 1), nn.ReLU(),
                # Conv1d(bottleneck_size//2+prim_dim, bottleneck_size//4, 1, activation="lrelu", normalization=normalization),
                nn.Conv1d(bottleneck_size//2+prim_dim, bottleneck_size//4, 1), nn.ReLU(),
                # Conv1d(bottleneck_size//4+prim_dim, out_dim, 1, activation=None, normalization=None),
                nn.Conv1d(bottleneck_size//4+prim_dim, out_dim, 1)
            ])
        else:
            self.layers = nn.ModuleList([
                # Conv1d(bottleneck_size+prim_dim, bottleneck_size//2, 1, activation="lrelu", normalization=normalization),
                # Conv1d(bottleneck_size//2, bottleneck_size//4, 1, activation="lrelu", normalization=normalization),
                # Conv1d(bottleneck_size//4, out_dim, 1, activation=None, normalization=None),
                ### layers and ###
                nn.Conv1d(bottleneck_size+prim_dim, bottleneck_size//2, 1), nn.ReLU(),
                # Conv1d(bottleneck_size//2+prim_dim, bottleneck_size//4, 1, activation="lrelu", normalization=normalization),
                nn.Conv1d(bottleneck_size//2, bottleneck_size//4, 1), nn.ReLU(),
                # Conv1d(bottleneck_size//4+prim_dim, out_dim, 1, activation=None, normalization=None),
                nn.Conv1d(bottleneck_size//4, out_dim, 1)
            ])

    def forward(self, x, primative):
        if x.ndimension() != primative.ndimension():
            x = x.unsqueeze(-1).expand(-1, -1, primative.shape[-1])

        for i, layer in enumerate(self.layers):
            if self.concat_prim or i==0:
                x = torch.cat([x, primative], dim=1)
            if (i+1) == len(self.layers):
                xyz = layer(x)
            else:
                x = layer(x)
        return xyz, x

class MultiFoldPointGen(nn.Module):
    """
    :params:
        code (B,C,1) or (B,C)
        primative (B,dim,P)

    :return:
        primative (B,dim,P)
        [point_feat (B,C,P)] decoder's last feature layer before getting the primiative coordinates
    """
    def __init__(self, bottleneck_size, out_dim=3, prim_dim=3,
                n_fold=3, normalization=None, concat_prim=True, residual=True, return_aux=True):
        super().__init__()
        folds = []
        self.prim_dim = prim_dim
        for i in range(n_fold):
            cur_out_dim = min(bottleneck_size, 64 * (n_fold-i)) if (i+1) < n_fold else 3
            folds += [PointGenCon(bottleneck_size, cur_out_dim, prim_dim, normalization=normalization, concat_prim=concat_prim)]
        self.folds = nn.ModuleList(folds)
        self.return_aux = return_aux
        self.residual = residual
        if self.residual:
            assert(prim_dim==out_dim)

    def forward(self, code, primative): # bsz x dim
        for i, fold in enumerate(self.folds):
            if code.ndimension() != primative.ndimension():
                code_exp = code.unsqueeze(-1).expand(-1, -1, primative.shape[-1])
            else: # primitive -1 --> 
                code_exp = code.expand(-1, -1, primative.shape[-1])
            assert(primative.shape[1] == self.prim_dim)
            xyz, point_feat = fold(code_exp, primative)

        if self.residual:
            xyz = primative+xyz
        if self.return_aux:
            return xyz, point_feat
        return xyz


class MLPDeformer(nn.Module):
    def __init__(self, dim, bottleneck_size, npoint, residual=True, normalization=None):
        super().__init__()
        self.npoint = npoint
        self.dim = dim
        self.residual = residual
        self.layers = nn.Sequential(
                Linear(bottleneck_size, 512, activation="lrelu", normalization=normalization),
                Linear(512, 256, activation="lrelu", normalization=normalization),
                Linear(256, npoint*dim)
            )
    def forward(self, code, template):
        B, C, N = template.shape
        assert(self.npoint == N)
        assert(self.dim == C)
        if code.ndim > 2:
            code = code.view(B, -1)
        x = self.layers(code)
        x = x.reshape(B,C,N)
        if self.residual:
            x += template
        return x



from pytorch_points.network.geo_operations import mean_value_coordinates_3D
def deform_with_MVC(cage, cage_deformed, cage_face, query, weights=None, verbose=False):
    """
    cage (B,C,3)
    cage_deformed (B,C,3)
    cage_face (B,F,3) int64
    query (B,Q,3)
    """
    if weights is None:
        weights, weights_unnormed = mean_value_coordinates_3D(query, cage, cage_face, verbose=True)
    else:
        weights_unnormed = weights
    # print(f"weights: {weights.size()}, maxx_weights: {torch.max(weights)}, minn_weigths: {torch.min(weights)}")
    
    # weights = 
#     weights = weights.detach() ## B x N x C x 1   B x 1 x C x 3
    deformed = torch.sum(weights.unsqueeze(-1) * cage_deformed.unsqueeze(1), dim=2)

    if verbose:
        return deformed, weights, weights_unnormed
    return deformed


from scipy.optimize import linear_sum_assignment

def biparti_matching_batched_torch(pc1_th, pc2_th):
  
  mathced_pc1s = []
  for i_bsz in range(pc1_th.size(0)):
    cur_bsz_pc1 = pc1_th[i_bsz].detach().cpu().numpy()
    cur_bsz_pc2 = pc2_th[i_bsz].detach().cpu().numpy()
    
    dists = np.sum((np.reshape(cur_bsz_pc1, (cur_bsz_pc1.shape[0], 1, 3)) - np.reshape(cur_bsz_pc2, (1, cur_bsz_pc2.shape[0], 3)) ) ** 2, axis=-1)
    row_ind, col_ind = linear_sum_assignment(dists) ### 
    
    
    reversed_row_ind = [None for _ in range(cur_bsz_pc1.shape[0])]
    row_ind_list = row_ind.tolist()
    for i in range(len(row_ind_list)):
      reversed_row_ind[row_ind_list[i]] = i
    reversed_row_ind = np.array(reversed_row_ind, dtype=np.int32)
    row_ind = row_ind[reversed_row_ind]
    col_ind = col_ind[reversed_row_ind]
    
    
    matched_pc1 = cur_bsz_pc2[col_ind]
    src_p1 = cur_bsz_pc1[row_ind]
    col_ind_th = torch.from_numpy(col_ind).long().cuda()
    matched_pc1_th = pc2_th[i_bsz][col_ind_th]
    # mathced_pc1s.append(torch.from_numpy(matched_pc1).float().cuda().unsqueeze(0))
    mathced_pc1s.append(matched_pc1_th.unsqueeze(0))
  
  mathced_pc1s = torch.cat(mathced_pc1s, dim=0)
  return mathced_pc1s



class NetworkFull(nn.Module):
    def __init__(self, opt, dim, bottleneck_size,
                 template_vertices, template_faces,
                 **kargs
                 ):

        super().__init__()
        self.cages_mult_coef = 0.95
        self.opt = opt
        self.dim = dim
        z_dim = 512
        self.z_dim = z_dim
        self.coef_multiplier = self.opt.coef_multiplier
        self.set_up_template(template_vertices, template_faces)
        
        self.use_gt_cages = self.opt.use_gt_cages


        self.pred_basis = True if  self.opt.pred_type == "basis" else False
        ###### source and target encoder ########
        # we don't share encoder because if we want to update deformer only, we don't want to change cage
        # if opt.pointnet2:
        # self.encoder = PointNet2feat(dim=dim, num_points=opt.num_point, bottleneck_size=bottleneck_size)
        
        self.encoder = PointFlowEncoder(zdim=self.z_dim, input_dim=3, use_deterministic_encoder=True)
        
        self.src_glb_feat_out_net = nn.Sequential(
             torch.nn.Conv1d(self.z_dim, self.z_dim, 1), nn.ReLU(), 
             torch.nn.Conv1d(z_dim, z_dim, 1), nn.ReLU(), 
             torch.nn.Conv1d(z_dim, z_dim, 1)
        )
        # else:
        #     self.encoder = nn.Sequential(
        #         PointNetfeat(dim=dim, num_points=opt.num_point,bottleneck_size=bottleneck_size),
        #         Linear(bottleneck_size, bottleneck_size, activation="lrelu", normalization=opt.normalization)
        #         )

        ###### cage prediction and cage deformation ########
        # if opt.full_net:
        #     if not opt.atlas:
        #         self.nc_decoder = MLPDeformer(dim=dim, bottleneck_size=bottleneck_size, npoint=self.template_vertices.shape[-1],
        #                                     residual=opt.c_residual, normalization=opt.normalization)
        #     else: ## concat_prim --> 
        self.nc_decoder = MultiFoldPointGen(bottleneck_size, dim, dim, n_fold=opt.n_fold, normalization=opt.normalization,
                                            concat_prim=opt.concat_prim, return_aux=False, residual=True)
        
        ### range predictor network ### range_predictor --> range_predictor --> range_predictor
        ### normalizaion ###
        self.range_predictor = MultiFoldPointGen(bottleneck_size, dim, dim, n_fold=opt.n_fold, normalization=opt.normalization, concat_prim=opt.concat_prim, return_aux=True, residual=True)
        n_basis = self.opt.num_basis
        self.n_basis = n_basis
        self.range_predictor_out_dim = 64 * (opt.n_fold - 1) #  bottleneck_size
        
        ragne_predict_in_dim = self.range_predictor_out_dim + 3
        self.range_predict_net = nn.Sequential(
            nn.Conv1d(ragne_predict_in_dim, self.range_predictor_out_dim // 2, 1), nn.LeakyReLU(),
            nn.Conv1d(self.range_predictor_out_dim // 2, self.range_predictor_out_dim // 2, 1), nn.LeakyReLU(),
            nn.Conv1d(self.range_predictor_out_dim // 2, 2, 1) 
        )
        
        basis_predict_net_in_dim = self.range_predictor_out_dim + 3
        self.basis_predict_net = nn.Sequential(
            nn.Conv1d(basis_predict_net_in_dim, self.range_predictor_out_dim // 2, 1), # nn.BatchNorm1d(self.range_predictor_out_dim // 2),
            nn.LeakyReLU(),
            nn.Conv1d(self.range_predictor_out_dim // 2, self.range_predictor_out_dim // 2, 1), # nn.BatchNorm1d(self.range_predictor_out_dim // 2),  #
            nn.LeakyReLU(),
            nn.Conv1d(self.range_predictor_out_dim // 2, 3 * self.n_basis, 1) 
        )
        


        self.D_use_C_global_code = False #  opt.c_global
        self.merger = nn.Sequential(
                nn.Conv1d(bottleneck_size*2, bottleneck_size*2, 1),
                nn.LeakyReLU()
            )
        # if not opt.atlas:
        #     self.nd_decoder = MLPDeformer(dim=dim, bottleneck_size=bottleneck_size*2, npoint=self.template_vertices.shape[-1],
        #                                 residual=opt.d_residual, normalization=opt.normalization)
        # else:
        # self.nd_decoder = torch.nn.ModuleList(
        #     [
        #         MultiFoldPointGen(bottleneck_size*2, 64, dim, n_fold=opt.n_fold, normalization=opt.normalization,
        #                        concat_prim=opt.concat_prim, return_aux=False, residual=False),
        #         MultiFoldPointGen(bottleneck_size, dim, 64, n_fold=opt.n_fold, normalization=opt.normalization,
        #                        concat_prim=opt.concat_prim, return_aux=False, residual=False),
        #     ] 
        # multifold point gen for cage offset prediction? #
        self.nd_decoder = MultiFoldPointGen(bottleneck_size*2, dim, dim, n_fold=opt.n_fold, normalization=opt.normalization, concat_prim=opt.concat_prim, return_aux=True, residual=True)
        self.nd_decoder_out_dim = 64 * (opt.n_fold - 1) * 2 #  bottleneck_size
        
        ### bsz x (out_dim + rng_dim (2) + bais_dim (3) ) x nn_pts x nn_basis ###
        # coef_pred_in_dim = self.nd_decoder_out_dim + 2 + 3
        coef_pred_in_dim = self.nd_decoder_out_dim + 2 + 3 + 3
        #### 
        self.coef_predictor = nn.Sequential( ### coef_predictor; coef_predictor_basis? 
            nn.Conv2d(coef_pred_in_dim, self.nd_decoder_out_dim // 2, 1), nn.LeakyReLU(),
            nn.Conv2d(self.nd_decoder_out_dim // 2, self.nd_decoder_out_dim // 2, 1), nn.LeakyReLU(),
            nn.Conv2d(self.nd_decoder_out_dim // 2, self.nd_decoder_out_dim // 2, 1), nn.Sigmoid()
        )
        self.coef_predictor_basis = nn.Sequential(
            nn.Conv2d(self.nd_decoder_out_dim // 2, self.nd_decoder_out_dim // 2, 1), nn.LeakyReLU(),
            nn.Conv2d(self.nd_decoder_out_dim // 2, self.nd_decoder_out_dim // 2, 1), nn.LeakyReLU(),
            nn.Conv2d(self.nd_decoder_out_dim // 2, 1, 1), nn.Sigmoid()
        )


    def set_up_template(self, template_vertices, template_faces):
        # save template as buffer
        assert(template_vertices.ndim==3 and template_vertices.shape[1]==self.dim) # (1,3,V)
        if self.dim == 3:
            assert(template_faces.ndim==3 and template_faces.shape[2]==3) # (1,F,3)

        print(f"Setting up template with vertices: {template_vertices.shape}, faces: {template_faces.shape}")
        # self.register_buffer("template_faces", template_faces)
        # self.register_buffer("template_vertices", template_vertices)
        # self.template_vertices = nn.Parameter(self.template_vertices, requires_grad=(self.opt.optimize_template))
        # self.template_vertices = nn.Parameter(self.template_vertices, requires_grad=False)
        self.template_vertices = nn.Parameter(template_vertices, requires_grad=False)
        self.template_faces = nn.Parameter(template_faces, requires_grad=False)
        # if self.template_vertices.requires_grad:
        #     logger.info("Enabled vertex optimization")

    def get_cages(self, input_pc):
         ### gt_def_new_cage ---> matching... ###
        ''' biparti mathcing for cages ''' ### for cages;
        B = input_pc.size(0)
        
        if self.template_vertices.size(-1) > 100:
            dn_nn_pts = 1024
            # dn_nn_pts = self.template_vertices.size(-1)
            pc_fps_idx = farthest_point_sampling(input_pc, n_sampling=dn_nn_pts)
            
            register_pc = input_pc[0, pc_fps_idx].unsqueeze(0)
        else:
            register_pc = input_pc
        
        ori_cage = self.template_vertices.view(1,self.dim,-1).expand(B,-1,-1) # ori cages for the cages # 
          
        # print(f"maxx_ori_cage: {torch.max(ori_cage, dim=-1)}, minn_ori_cage: {torch.min(ori_cage, dim=-1)}, maxx_input_pc: {torch.max(input_pc, dim=1)}, minn_input_pc: {torch.min(input_pc, dim=1)}, ")
        ######## heuristic deformation ##########
        def_cage = biparti_matching_batched_torch(ori_cage.contiguous().transpose(1, 2).contiguous(), register_pc)
        flow_cage_def_cage = def_cage - ori_cage.contiguous().transpose(1, 2).contiguous()
        cur_cages_mult_coef = 0.85 if self.template_vertices.size(-1) > 100 else self.cages_mult_coef
        cage = ori_cage.contiguous().transpose(1, 2).contiguous() + cur_cages_mult_coef * flow_cage_def_cage  
        cage = cage # input_pc.contiguous().transpose(1, 2).contiguous()
        # print(f"retruning tempalte with cage: {cage.size()}")
        ''' biparti mathcing for cages ''' 
        return cage

    def forward(self, source_shape, target_shape, source_cvx_shp=None, target_cvx_shp=None, alpha=1.0, rnd_smaple_nn=0):
        """
        source_shape (B,3,N)
        target_shape (B,3,M)
        init_cage    (B,3,P)
        return:
            deformed (B,3,N)
            cage     (B,3,P)
            new_cage (B,3,P)
            weights  (B,N,P)
        """
        cages_mult_coef = self.cages_mult_coef #  0.95
        EPS = 1e-6
        B, _, N = source_shape.shape
        _, M, N = target_shape.shape
        _, _, P = self.template_vertices.shape
        
        source_cvx_shp = source_shape if source_cvx_shp is None else source_cvx_shp
        target_cvx_shp = target_shape if target_cvx_shp is None else target_cvx_shp
        
        ############ Encoder #############
        # input_shapes = torch.cat([source_shape, target_shape], dim=0)
        # shape_code, _ = self.encoder(input_shapes.contiguous().transpose(1, 2).contiguous())
        
        # shape_code = torch.max(shape_code, dim=-1)[0]
        # shape_code = self.src_glb_feat_out_net(shape_code.unsqueeze(-1))
        
        # # shape_code.unsqueeze_(-1)
        # s_code, t_code = torch.split(shape_code, B, dim=0)
        ############ Encoder #############
        
        # s_code, _ = self.encoder(source_shape.contiguous().transpose(1, 2).contiguous())
        # s_code = torch.max(s_code, dim=-1)[0]
        # s_code = self.src_glb_feat_out_net(s_code.unsqueeze(-1))
        
        _, s_code = self.encoder(source_shape.contiguous().transpose(1, 2).contiguous())
        # s_code = torch.max(s_code, dim=-1)[0]
        # s_code = self.src_glb_feat_out_net(s_code.unsqueeze(-1))
        s_code = s_code.unsqueeze(-1)
        
        
        # t_code, _ = self.encoder(target_shape.contiguous().transpose(1, 2).contiguous())
        # t_code = torch.max(t_code, dim=-1)[0]
        # t_code = self.src_glb_feat_out_net(t_code.unsqueeze(-1))
        
        _, t_code = self.encoder(target_shape.contiguous().transpose(1, 2).contiguous())
        # t_code = torch.max(t_code, dim=-1)[0]
        # t_code = self.src_glb_feat_out_net(t_code.unsqueeze(-1))
        t_code = t_code.unsqueeze(-1)
        
        
        ### cages, range predict network ###
        # ############ Cage ################  # ori_cages # ori_cages #
        ori_cage = self.template_vertices.view(1,self.dim,-1).expand(B,-1,-1)
        
        # if self.opt.full_net:
        # cage = self.nc_decoder(s_code, ori_cage)
        # cage.register_hook(save_grad("d_cage"))
        
        ######## heuristic deformation #########
        # def_cage = biparti_matching_batched_torch(ori_cage.contiguous().transpose(1, 2).contiguous(), source_cvx_shp.contiguous().transpose(1, 2).contiguous())
        # flow_cage_def_cage = def_cage - ori_cage.contiguous().transpose(1, 2).contiguous()
        # cage = ori_cage.contiguous().transpose(1, 2).contiguous() + self.cages_mult_coef * flow_cage_def_cage 
        
        cage = self.get_cages(source_cvx_shp.contiguous().transpose(1, 2).contiguous())
        cage = cage.contiguous().transpose(1, 2).contiguous()
        
        if self.pred_basis:
            ####### range & offset-basis prediction #######
            ## 
            cage_basis, cage_def_range_feats = self.range_predictor(s_code, cage) ## cage_def_range_feats: 
            # cage_basis = cage_basis - cage
            # cage_def_range_feats: bsz x dim x n_pts
            nk = cage_def_range_feats.size(-1)
            cage_def_range_feats_in_feats = torch.cat(
                [cage_def_range_feats, cage], dim=1 ### bsz x (dim + 1) x n_pts ###
            )
            # cage_def_range_feats_in_feats = cage_def_range_feats_in_feats.permute(0, 2, 1).view(-1, cage_def_range_feats_in_feats.size(1)).unsqueeze(-1).contiguous() # preidct network  --> is that same for different cages? 
            cage_def_basis = self.basis_predict_net(cage_def_range_feats_in_feats) ### basis prediction network ###
            # cage_def_basis = cage_def_basis.contiguous().view(cage_basis.size(0), nk, 3, self.n_basis).contiguous().transpose(-1, -2).contiguous()
            ## bsz x nn_pts x nb x 3 
            # cage_def_basis = cage_def_basis.contiguous().transpose(-1, -2).contiguous()
            cage_def_basis = cage_def_basis.contiguous().transpose(1, 2).contiguous().view(cage_def_basis.size(0), cage_def_basis.size(2), self.n_basis, 3).contiguous()
            # 
            if self.opt.bais_per_vert:
                # nn_pts x nn_basis x 3 --> nn_basis x nn_pts x 3 --> nn_basis x (nn_pts x 3) --> 
                cage_def_basis = cage_def_basis / torch.clamp(torch.norm(cage_def_basis, dim=-1, p=2, keepdim=True), min=EPS) ### ### bsz x nn_pts x nn_basis x 3
                cage_def_range_feats_exp = cage_def_range_feats.unsqueeze(-1).expand(-1, -1, -1, self.n_basis) ### bsz x dim x nn_pts x nn_basis
                cage_def_basis_trans = cage_def_basis.contiguous().permute(0, 3, 1, 2).contiguous() 
                cage_def_range_in_feats = torch.cat(
                    [cage_def_range_feats_exp, cage_def_basis_trans], dim=1 ### bsz x (dim + 3) x nn_pts x nn_basiss
                )
                
                cage_def_range_in_feats = cage_def_range_in_feats.contiguous().view(cage_def_range_in_feats.size(0), cage_def_range_in_feats.size(1), -1).contiguous() ### bsz x ()
                
                
                ### predict basis ###
                ### predict ranges of basis ###
                # cage_def_range_feats #
                ### 
                cage_def_range = self.range_predict_net(cage_def_range_in_feats) ### bsz x (2 x n_basis) x nn_pts ###
                #### bsz x nn_pts x nn_basis x 2 ####
                cage_def_range = cage_def_range.contiguous().view(cage_def_range.size(0), cage_def_range.size(1), -1, self.n_basis) ### bsz x (2) x nn_pts x nn_basis ###
                cage_def_range = cage_def_range.contiguous().permute(0, 2, 3, 1).contiguous()
                cage_def_range = cage_def_range * self.coef_multiplier
                ### cage_def_basis: bsz x nn_pts x nn_basis x 3 ### --> unit vectors ###
                cage_def_range[..., 0] = cage_def_range[..., 0] * -1.0  ### bsz x nn_pts x nn_basis x 2 ### 
            else:
                nk = cage_def_basis.size(1)
                B = cage_def_basis.size(0)
                cage_def_basis = cage_def_basis.contiguous().transpose(1, 2).contiguous().view(cage_def_basis.size(0), self.n_basis, -1) ### B x nb x (nk x 3)
                # cage_def_basis = cage_def_basis.contiguous().view(cage_def_basis.size(0), self.n_basis, -1)
                cage_def_basis = cage_def_basis / torch.clamp(torch.norm(cage_def_basis, dim=-1, p=2, keepdim=True), min=EPS)
                cage_def_basis_trans = cage_def_basis.contiguous().view(B, self.n_basis, nk, 3).contiguous()
                cage_def_basis_trans = cage_def_basis_trans.contiguous().permute(0, 3, 2, 1) ### B x 3 x nk x nb ### 
                cage_def_range_feats_exp = cage_def_range_feats.unsqueeze(-1).expand(-1, -1, -1, self.n_basis) ### bsz x dim x nn_pts x nn_basis
                cage_def_range_in_feats = torch.cat(
                    [cage_def_range_feats_exp, cage_def_basis_trans], dim=1 ### bsz x (dim + 3) x nn_pts x nn_basiss
                )
                cage_def_range_in_feats = cage_def_range_in_feats.contiguous().view(cage_def_range_in_feats.size(0), cage_def_range_in_feats.size(1), -1).contiguous() ### bsz x ()
                cage_def_range = self.range_predict_net(cage_def_range_in_feats) ### bsz x (2 x n_basis) x nn_pts ###
                #### bsz x nn_pts x nn_basis x 2 ####
                cage_def_range = cage_def_range.contiguous().view(cage_def_range.size(0), cage_def_range.size(1), -1, self.n_basis) ### bsz x (2) x nn_pts x nn_basis ###
                cage_def_range = cage_def_range.contiguous().permute(0, 2, 3, 1).contiguous() ### bsz x nk x nb x 2
                cage_def_range = torch.max(cage_def_range, dim=1)[0] ## bsz x nb x 2
                cage_def_range = cage_def_range * self.coef_multiplier
                cage_def_range[..., 0] = cage_def_range[..., 0] * -1.0 ### bsz x 
                
            
        
        
        ### gt_def_new_cage ---> matching... ###
        ''' biparti mathcing for cages ''' ###
        # gt_def_new_cage = biparti_matching_batched_torch(ori_cage.contiguous().transpose(1, 2).contiguous(), target_cvx_shp.contiguous().transpose(1, 2).contiguous())
        # # def_new_cage = def_new_cage.contiguous().transpose(1, 2).contiguous()
        # ### flow_cage_def_new_cage: bsz x n_pts x 3 ###; flow_cage_def_new_cage
        # flow_cage_def_new_cage = gt_def_new_cage - ori_cage.contiguous().transpose(1, 2).contiguous()
        
        # flow_cage_def_new_cage_cage = gt_def_new_cage - cage.contiguous().transpose(1, 2).contiguous()
        
        gt_new_cage = self.get_cages(target_cvx_shp.contiguous().transpose(1, 2).contiguous())
        # gt_new_cage = ori_cage.contiguous().transpose(1, 2).contiguous() + cages_mult_coef * flow_cage_def_new_cage  ## 
        gt_new_cage = gt_new_cage # .contiguous().transpose(1, 2).contiguous()
        if self.use_gt_cages:
            new_cage = gt_new_cage.contiguous().transpose(1, 2).contiguous()
        ''' biparti mathcing for cages ''' 
    

        ########### Deform ##########
        # first fold use global feature ### fold? ---> what does `fold` used for? ###
        target_code = torch.cat([s_code, t_code], dim=1)
        target_code = self.merger(target_code)
        
        ### TODO: increase n_folds ###
        if not self.use_gt_cages:
            ############ Predict offset --> new_cages ##############
            new_cage, new_cage_feats = self.nd_decoder(target_code, cage)
            ############ Predict offset --> new_cages ##############
        
        
        # rnd_sample_nn = 10
        
        
        if self.pred_basis:
            
            if self.opt.optimize_coef:
                ### flow_cage_def_new_cage: bsz x n_pts x 3  --> 
                #### cage_coef_basis: bsz x n_b x (n_pts x 3) 
                n_bsz = cage_def_basis.size(0)
                flow_cage_def_new_cage_cage = gt_new_cage - cage.contiguous().transpose(1, 2).contiguous()
                mag_flow = torch.sum((flow_cage_def_new_cage_cage) ** 2, dim=-1).mean()
                
                flow_cage_def_new_cage_flat = flow_cage_def_new_cage_cage.contiguous().view(flow_cage_def_new_cage_cage.size(0), -1).contiguous() ### 
                
                # flow_cage_def_new_cage_flat = flow_cage_def_new_cage_flat * 0.90
                
                # b = flow_cage_def_new_cage_flat.contiguous().transpose(0, 1).contiguous() ### (n_pts x 3) x n_b
                # A = cage_coef_basis.
                tot_optim_coefs = []
                
                for i_bsz in range(n_bsz):
                    A = cage_def_basis[i_bsz].contiguous().transpose(0, 1).contiguous() ### (n_k x 3) x n_b 
                    b = flow_cage_def_new_cage_flat[i_bsz].unsqueeze(-1)
                    A_np = A.detach().cpu().numpy()
                    b_np = b.detach().cpu().numpy()
                    
                    A_T_A_np = np.matmul(np.transpose(A_np, (1, 0)), A_np) ### 
                    A_T_b_np = np.matmul(np.transpose(A_np, (1, 0)), b_np)
                    c_np = np.linalg.solve(A_T_A_np, A_T_b_np) ### (n_basis) x 1
                    c = torch.from_numpy(c_np).float().cuda().squeeze(-1)
                    
                    ### cshould be 
                    c_norm_1 = torch.norm(c, p=1).item()
                    # if c_norm_1 > self.coef_multiplier  * self.n_basis:
                        # c = c / c_norm_1 * self.coef_multiplier  * self.n_basis
                    # c_norm_reg_coef_multiplier = 0.5
                    
                    # c_norm_reg_coef_multiplier = 10
                    # if c_norm_1 > c_norm_reg_coef_multiplier :
                    #     c = c / c_norm_1 * c_norm_reg_coef_multiplier  
                        
                        
                    
                    
                    # maxx_nn_activated = self.n_basis // 2
                    # topk_c_val, topk_c_idx = torch.topk(torch.abs(c), k=maxx_nn_activated, largest=True)
                    # # print(topk_c_val)
                    # c[torch.abs(c) < float(topk_c_val[-1].item())] = 0.
                    # print(f"c: {c}")
                    tot_optim_coefs.append(c.unsqueeze(0))
                    
                tot_optim_coefs = torch.cat(tot_optim_coefs, dim=0) ## n_bsz x n_basis ##
                # new_cage_coefs = (tot_optim_coefs - cage_def_range[..., 0]) # n_bsz x n_basis ## new cage 
                new_cage_coefs = tot_optim_coefs
                
                if not self.use_gt_cages:
                    new_cage_def  = new_cage_coefs.unsqueeze(-1) * cage_def_basis ### bsz x n_b x (n_k x 3) 
                    
                    new_cage_def = torch.sum(new_cage_def, dim=-2) ### bsz x nn_pts x 3
                    new_cage_def = new_cage_def.contiguous().view(new_cage_def.size(0), nk, 3).contiguous()
                    new_cage = cage + new_cage_def.contiguous().transpose(1, 2).contiguous()
                    
                else:
                    wk_new_cage_def  = new_cage_coefs.unsqueeze(-1) * cage_def_basis ### bsz x n_b x (n_k x 3) 
                    
                    wk_new_cage_def = torch.sum(wk_new_cage_def, dim=-2) ### bsz x nn_pts x 3
                    wk_new_cage_def = wk_new_cage_def.contiguous().view(wk_new_cage_def.size(0), nk, 3).contiguous()
                    wk_new_cage = cage + wk_new_cage_def.contiguous().transpose(1, 2).contiguous()
                
                    # bsz x 3 x nk
                    delta_pred_new_cage_real_new_cage = torch.sum((gt_new_cage - wk_new_cage.contiguous().transpose(1, 2).contiguous()) ** 2, dim=-1).mean()
                
                
                
                    if rnd_smaple_nn == 6:
                        print(f'cur jelta_flow: {delta_pred_new_cage_real_new_cage.item()}, mag_flow: {mag_flow.item()}')
                
    
                
                # rnd_new_cage_coefs = torch.rand_like(new_cage_coefs).squeeze(-1)
                rnd_new_cage_coefs = torch.randn_like(new_cage_coefs) # .squeeze(-1)
                # rnd_new_cage_coefs = rnd_new_cage_coefs * cage_def_range[..., 0] + (1. - rnd_new_cage_coefs) * cage_def_range[..., 1]
                # rnd_new_cage_coefs = rnd_new_cage_coefs * (-self.coef_multiplier) + (1. - rnd_new_cage_coefs) * self.coef_multiplier
                rnd_new_cage_coefs = rnd_new_cage_coefs * (self.coef_multiplier) + new_cage_coefs
                rnd_new_cage_def  = rnd_new_cage_coefs.unsqueeze(-1) * cage_def_basis
                rnd_new_cage_def = torch.sum(rnd_new_cage_def, dim=-2)
                rnd_new_cage_def = rnd_new_cage_def.contiguous().view(rnd_new_cage_def.size(0), nk, 3).contiguous()
                rnd_new_cage = cage + rnd_new_cage_def.contiguous().transpose(1, 2).contiguous()
                
                tot_rnd_sample_new_cages = []
                for i_s in range(rnd_smaple_nn):
                    cur_rnd_new_cage_coefs = torch.randn_like(new_cage_coefs) # .squeeze(-1)
                    cur_rnd_new_cage_coefs = cur_rnd_new_cage_coefs * (self.coef_multiplier) + new_cage_coefs
                    # cur_rnd_new_cage_coefs = cur_rnd_new_cage_coefs * (-self.coef_multiplier) + (1. - cur_rnd_new_cage_coefs) * self.coef_multiplier
                    cur_rnd_new_cage_def  = cur_rnd_new_cage_coefs.unsqueeze(-1) * cage_def_basis ### get coefs basis
                    cur_rnd_new_cage_def = torch.sum(cur_rnd_new_cage_def, dim=-2)
                    cur_rnd_new_cage_def = cur_rnd_new_cage_def.contiguous().view(cur_rnd_new_cage_def.size(0), nk, 3).contiguous()
                    cur_rnd_new_cage = cage + cur_rnd_new_cage_def.contiguous().transpose(1, 2).contiguous()
                    tot_rnd_sample_new_cages.append(cur_rnd_new_cage) ### cur_rnd_new_cage ---> rnd_new_cages 
            
            elif self.opt.bais_per_vert: ## new_cage_feats: 
                # new_cage_feats = 
                new_cage_feats_exp = new_cage_feats.contiguous().unsqueeze(-1).expand(-1, -1, -1, self.n_basis).contiguous() ### 
                cage_def_range_trans = cage_def_range.contiguous().permute(0, 3, 1, 2).contiguous()
                cage_def_basis_trans = cage_def_basis.contiguous().permute(0, 3, 1, 2).contiguous()  ## cage_def_basis
                
                ######## bsz x n_k x n_b x 3 ########### --> ########## bsz x 3 x n_k x n_b ###########
                flow_cage_def_new_cage_exp = flow_cage_def_new_cage.contiguous().unsqueeze(2).expand(-1, -1, self.n_basis, -1).contiguous() ### contiguous() for others xxx ###
                flow_cage_def_new_cage_exp = flow_cage_def_new_cage_exp.contiguous().permute(0, 3, 1, 2).contiguous() ### bsz x 3 x n_k x n_b
                
                
                # coef_pred_in_feats = torch.cat(
                #     [new_cage_feats_exp, cage_def_range_trans, cage_def_basis_trans], dim=1 ### xx in_dm
                # )
                
                coef_pred_in_feats = torch.cat(
                    [new_cage_feats_exp, cage_def_range_trans, cage_def_basis_trans, flow_cage_def_new_cage_exp], dim=1 ### xx in_dm
                )
                
                
                new_cage_coefs = self.coef_predictor(coef_pred_in_feats) # bsz x 1 x nn_keypts x n_basis
                new_cage_coefs = new_cage_coefs.contiguous().permute(0, 2, 3, 1).contiguous().squeeze(-1)
                new_cage_coefs = new_cage_coefs * cage_def_range[..., 0] + (1. - new_cage_coefs) * cage_def_range[..., 1]
                new_cage_def  = new_cage_coefs.unsqueeze(-1) * cage_def_basis
                new_cage_def = torch.sum(new_cage_def, dim=-2) ### bsz x nn_pts x 3
                new_cage = cage + new_cage_def.contiguous().transpose(1, 2).contiguous()
                
                ############ Predict coefs ##############
                # new_cage_coefs = self.coef_predictor(new_cage_feats) # bsz x 1 x nn_keypts
                # new_cage_coefs = new_cage_coefs.contiguous().transpose(1, 2).contiguous().squeeze(-1)
                # new_cage_coefs = new_cage_coefs * cage_def_range[..., 0] + (1. - new_cage_coefs) * cage_def_range[..., 1]
                # new_cage_def  = new_cage_coefs.unsqueeze(-1) * cage_basis
                # new_cage = cage + new_cage_def.contiguous().transpose(1, 2).contiguous()
                ############ Predict coefs ##############
                
                
                rnd_new_cage_coefs = torch.rand_like(new_cage_coefs).squeeze(-1)
                rnd_new_cage_coefs = rnd_new_cage_coefs * cage_def_range[..., 0] + (1. - rnd_new_cage_coefs) * cage_def_range[..., 1]
                rnd_new_cage_def  = rnd_new_cage_coefs.unsqueeze(-1) * cage_def_basis
                rnd_new_cage_def = torch.sum(rnd_new_cage_def, dim=-2)
                rnd_new_cage = cage + rnd_new_cage_def.contiguous().transpose(1, 2).contiguous()
            else:
                nk = cage_def_basis_trans.size(2)
                new_cage_feats_exp = new_cage_feats.contiguous().unsqueeze(-1).expand(-1, -1, -1, self.n_basis).contiguous() ###
                cage_def_range_trans = cage_def_range.contiguous().permute(0, 2, 1).contiguous().unsqueeze(2).expand(-1, -1, nk, -1) ### bsz x 2 x nk x nb --> the range of trans --> 
                # cage_def_basis_trans = cage_def_basis.contiguous().permute(0, 3, 1, 2).contiguous() ##### cage_def_bais_trans --> ### 
                ### flow_cage_def_new_cage: bsz x n_pts x 3
                
                ######## bsz x n_k x n_b x 3 ########### --> ########## bsz x 3 x n_k x n_b ###########
                flow_cage_def_new_cage_exp = flow_cage_def_new_cage.contiguous().unsqueeze(2).expand(-1, -1, self.n_basis, -1).contiguous() ### contiguous() for others xxx ###
                flow_cage_def_new_cage_exp = flow_cage_def_new_cage_exp.contiguous().permute(0, 3, 1, 2).contiguous() ### bsz x 3 x n_k x n_b
                
                
                # coef_pred_in_feats = torch.cat(
                #     [new_cage_feats_exp, cage_def_range_trans, cage_def_basis_trans], dim=1 ### xx in_dm ## flow_cage_def_new_cage_exp
                # )
                
                coef_pred_in_feats = torch.cat(
                    [new_cage_feats_exp, cage_def_range_trans, cage_def_basis_trans, flow_cage_def_new_cage_exp], dim=1 ### xx in_dm ## flow_cage_def_new_cage_exp
                )
                
                # coef_pred_in_feats = torch.cat(
                #     []
                # )
                
                new_cage_coefs = self.coef_predictor(coef_pred_in_feats) # bsz x dimx x nn_keypts x n_basis
                new_cage_coefs = torch.max(new_cage_coefs, dim=2)[0] ## bsz x dimx x n_basis
                new_cage_coefs = self.coef_predictor_basis(new_cage_coefs.unsqueeze(-1))[0] ## bsz x 1 x n_basi
                # new_cage_coefs = new_cage_coefs.squeeze(-1)
                
                ####### 
                new_cage_coefs = new_cage_coefs.contiguous().permute(0, 2, 1).contiguous().squeeze(-1)
                new_cage_coefs = new_cage_coefs * cage_def_range[..., 0] + (1. - new_cage_coefs) * cage_def_range[..., 1]
                new_cage_def  = new_cage_coefs.unsqueeze(-1) * cage_def_basis ### bsz x n_b x (n_k x 3) 
                
                new_cage_def = torch.sum(new_cage_def, dim=-2) ### bsz x nn_pts x 3
                new_cage_def = new_cage_def.contiguous().view(new_cage_def.size(0), nk, 3).contiguous()
                new_cage = cage + new_cage_def.contiguous().transpose(1, 2).contiguous()
                
                ############ Predict coefs ##############
                # new_cage_coefs = self.coef_predictor(new_cage_feats) # bsz x 1 x nn_keypts
                # new_cage_coefs = new_cage_coefs.contiguous().transpose(1, 2).contiguous().squeeze(-1)
                # new_cage_coefs = new_cage_coefs * cage_def_range[..., 0] + (1. - new_cage_coefs) * cage_def_range[..., 1]
                # new_cage_def  = new_cage_coefs.unsqueeze(-1) * cage_basis
                # new_cage = cage + new_cage_def.contiguous().transpose(1, 2).contiguous()
                ############ Predict coefs ##############
                
                
                rnd_new_cage_coefs = torch.rand_like(new_cage_coefs).squeeze(-1)
                rnd_new_cage_coefs = rnd_new_cage_coefs * cage_def_range[..., 0] + (1. - rnd_new_cage_coefs) * cage_def_range[..., 1]
                rnd_new_cage_def  = rnd_new_cage_coefs.unsqueeze(-1) * cage_def_basis
                rnd_new_cage_def = torch.sum(rnd_new_cage_def, dim=-2)
                rnd_new_cage_def = rnd_new_cage_def.contiguous().view(rnd_new_cage_def.size(0), nk, 3).contiguous()
                rnd_new_cage = cage + rnd_new_cage_def.contiguous().transpose(1, 2).contiguous()
                
        else:
            rnd_new_cage = new_cage.clone()
            
            

        
        ''' biparti mathcing for cages ''' 
        # new_cage = biparti_matching_batched_torch(ori_cage.contiguous().transpose(1, 2).contiguous(), target_shape.contiguous().transpose(1, 2).contiguous())
        # new_cage = new_cage.contiguous().transpose(1, 2).contiguous()
        ''' biparti mathcing for cages ''' 
        

        ########### MVC ##############
        if self.dim == 3:
            cage = cage.transpose(1,2).contiguous()
            new_cage = new_cage.transpose(1,2).contiguous()
            # new_cage = new_cage.detach()
            deformed_shapes, weights, weights_unnormed = deform_with_MVC(cage,
                                                                         new_cage,
                                                                         self.template_faces.expand(B,-1,-1),
                                                                         source_shape.transpose(1,2).contiguous(),
                                                                         verbose=True)
            
            
            rnd_new_cage = rnd_new_cage.transpose(1,2).contiguous()
            rnd_deformed_shapes, rnd_weights, rnd_weights_unnormed = deform_with_MVC(cage,
                                                                         rnd_new_cage,
                                                                         self.template_faces.expand(B,-1,-1),
                                                                         source_shape.transpose(1,2).contiguous(),
                                                                         verbose=True)
            
            tot_rnd_deformed_shapes = []
            for i_s in range(rnd_smaple_nn):
                cur_rnd_new_cage = tot_rnd_sample_new_cages[i_s]
                cur_rnd_new_cage = cur_rnd_new_cage.transpose(1,2).contiguous()
                tot_rnd_sample_new_cages[i_s] = cur_rnd_new_cage
                cur_rnd_deformed_shapes, cur_rnd_weights, cur_rnd_weights_unnormed = deform_with_MVC(cage,
                                                                         cur_rnd_new_cage,
                                                                         self.template_faces.expand(B,-1,-1),
                                                                         source_shape.transpose(1,2).contiguous(),
                                                                         verbose=True)
                tot_rnd_deformed_shapes.append(cur_rnd_deformed_shapes)
            
            
        elif self.dim == 2:
            weights, weights_unnormed = mean_value_coordinates(source_shape, cage, verbose=True)
            deformed_shapes = torch.sum(weights.unsqueeze(1)*new_cage.unsqueeze(-1), dim=2).transpose(1,2).contiguous()
            cage = cage.transpose(1,2)
            new_cage = new_cage.transpose(1,2)

        return {
            "cage": cage,
            "new_cage": new_cage,
            "rnd_new_cage": rnd_new_cage,
            "tot_rnd_sample_new_cages": tot_rnd_sample_new_cages,
            "tot_rnd_deformed_shapes": tot_rnd_deformed_shapes,
            "gt_new_cage": gt_new_cage,
            "deformed": deformed_shapes,
            "rnd_deformed": rnd_deformed_shapes,
            "cage_face": self.template_faces,
            "weight": weights,
            "weight_unnormed": weights_unnormed,
            "cage_def_basis": cage_def_basis if self.opt.pred_type == "basis" else [],
            "new_cage_coefs": new_cage_coefs if self.opt.pred_type == "basis" else [],
            
        }



if __name__ == "__main__":
    npoint = 5000
    # net = NetworkFull(dim=3, num_points=npoint, nc_bottleneck=512, nd_bottleneck=512,
    #                 C_residual=True, D_residual=True, D_use_enc_code=True, MLP=False,
    #                 concat_prim=True, multi_fold=True, normalization=None).cuda()
    # net = NetworkSharedEnc(dim=3, num_points=npoint, bottleneck_size=512,
    #                        C_residual=True, D_resiual=True, D_use_enc_code=True, template=None, normalization=None,
    #                        concat_prim=True, multi_fold=True)
    net = UNetwork(dim=3, num_points=npoint, bottleneck_size=512,
                   C_residual=True, D_residual=True, D_use_enc_code=True, template=None, normalization=None,
                   concat_prim=False, multi_fold=False)
    print(net)
    # points = torch.rand((4, 3, npoint)).cuda()
    # cage_V = torch.rand((4, 3, 120)).cuda()
