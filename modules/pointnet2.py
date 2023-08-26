import torch
import torch.nn as nn
# from .point_convolution_universal import TransitionDown, TransitionUp
from utils.model_utils import construct_conv_modules, set_bn_not_training, set_grad_to_none, apply_module_with_conv2d_bn
from utils.data_utils_torch import farthest_point_sampling, batched_index_select


class PointnetPP(nn.Module):
    def __init__(self, in_feat_dim: int, use_light_weight=False, args=None):
        super(PointnetPP, self).__init__()

        # if args is not None:
        #     self.skip_global = args.skip_global
        # else:
        self.skip_global = False

        # self.n_samples = [512, 128, 1] if "motion" not in args.task else [256, 128, 1]
        # self.n_samples = [512, 128, 1]
        # self.n_samples = [1024, 512, 1]
        # self.n_samples = [1024, 512, 1]

        # mlps = [[64,64,128], [128,128,256], [256,512,1024]]
        # mlps_in = [[in_feat_dim,64,64], [128+3,128,128], [256+3,256,512]]

        # up_mlps = [[256, 256], [256, 128], [128, 128, 128]]
        # # up_mlps_in = [1024+256, 256+128, 128+3+3]
        # up_mlps_in = [1024 + 256, 256 + 128, 128 + in_feat_dim]

        self.n_samples = [128, 1]
        mlps = [[64,64,128], [128,128,128 if use_light_weight else 512]]
        mlps_in = [[in_feat_dim,64,64], [128+3,128,128]]

        # up_mlps = [[256, 256], [256, 128]]
        # # up_mlps_in = [1024+256, 256+128, 128+3+3]
        # up_mlps_in = [1024 + 256, 256 + 128, 128 + in_feat_dim]

        self.in_feat_dim = in_feat_dim
        # self.radius = [0.2, 0.4, None]

        self.radius = [0.4, None]

        # if args is not None:
        #     n_layers = args.pnpp_n_layers
        #     self.n_samples = self.n_samples[:n_layers]
        #     mlps, mlps_in = mlps[:n_layers], mlps_in[:n_layers]
        #     self.radius = self.radius[:n_layers]

        #     up_mlps = up_mlps[-n_layers:]
        #     up_mlps_in = up_mlps_in[-n_layers:]

        self.mlp_layers = nn.ModuleList()

        for i, (dims_in, dims_out) in enumerate(zip(mlps_in, mlps)):
            if self.skip_global and i == len(mlps_in) - 1:
                break
            conv_layers = construct_conv_modules(
                mlp_dims=dims_out, n_in=dims_in[0],
                last_act=True,
                # last_act=False,
                bn=True
            )
            self.mlp_layers.append(conv_layers)

        # self.up_mlp_layers = nn.ModuleList()

        # for i, (dim_in, dims_out) in enumerate(zip(up_mlps_in, up_mlps)):
        #     if self.skip_global and i == 0:
        #         continue
        #     conv_layers = construct_conv_modules(
        #         mlp_dims=dims_out, n_in=dim_in,
        #         # last_act=False,
        #         last_act=True,
        #         bn=True
        #     )
        #     self.up_mlp_layers.append(conv_layers)

    def set_bn_no_training(self):
        for sub_module in self.mlp_layers:
            set_bn_not_training(sub_module)
        for sub_module in self.up_mlp_layers:
            set_bn_not_training(sub_module)

    def set_grad_to_none(self):
        for sub_module in self.mlp_layers:
            set_grad_to_none(sub_module)
        for sub_module in self.up_mlp_layers:
            set_grad_to_none(sub_module)

    def sample_and_group(self, feat, pos, n_samples, use_pos=True, k=64):
        bz, N = pos.size(0), pos.size(1)
        fps_idx = farthest_point_sampling(pos=pos[:, :, :3], n_sampling=n_samples)
        # bz x n_samples x pos_dim
        # sampled_pos = batched_index_select(values=pos, indices=fps_idx, dim=1)
        sampled_pos = pos.contiguous().view(bz * N, -1)[fps_idx, :].contiguous().view(bz, n_samples, -1)
        ppdist = torch.sum((sampled_pos.unsqueeze(2) - pos.unsqueeze(1)) ** 2, dim=-1)
        ppdist = torch.sqrt(ppdist)
        topk_dist, topk_idx = torch.topk(ppdist, k=k, dim=2, largest=False)

        # if n_samples == 1:
        #

        grouped_pos = batched_index_select(values=pos, indices=topk_idx, dim=1)
        grouped_pos = grouped_pos - sampled_pos.unsqueeze(2)
        if feat is not None:
            grouped_feat = batched_index_select(values=feat, indices=topk_idx, dim=1)
            if use_pos:
                grouped_feat = torch.cat([grouped_pos, grouped_feat], dim=-1)
        else:
            grouped_feat = grouped_pos
        return grouped_feat, topk_dist, sampled_pos

    def max_pooling_with_r(self, grouped_feat, ppdist, r=None):
        if r is None:
            res, _ = torch.max(grouped_feat, dim=2)
        else:
            # bz x N x k
            indicators = (ppdist <= r).float()
            indicators_expand = indicators.unsqueeze(-1).repeat(1, 1, 1, grouped_feat.size(-1))
            indicators[indicators < 0.5] = -1e8
            grouped_feat[indicators_expand < 0.5] = -1e8
            res, _ = torch.max(grouped_feat, dim=2)
        return res

    def interpolate_features(self, feat, p1, p2, ):
        dist = p2[:, :, None, :] - p1[:, None, :, :]
        dist = torch.norm(dist, dim=-1, p=2, keepdim=False)
        topkk = min(3, dist.size(-1))
        dist, idx = dist.topk(topkk, dim=-1, largest=False)

        # bz x N2 x 3
        # print(dist.size(), idx.size())
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        # weight.size() = bz x N2 x 3; idx.size() = bz x N2 x 3
        three_nearest_features = batched_index_select(feat, idx, dim=1)  # 1 is the idx dimension
        interpolated_feats = torch.sum(three_nearest_features * weight[:, :, :, None], dim=2, keepdim=False)
        return interpolated_feats

    def forward(self, x: torch.FloatTensor, pos: torch.FloatTensor, return_global=False):
        # bz, N = x.size(0), x.size(1)
        # # only for feature map; no issue w.r.t. the communication between different points' features
        # ###### For relative feature test... ######

        # x = x[:, :, 3:] # x = 
        bz = pos.size(0)

        cache = []
        cache.append((None if x is None else x.clone(), pos.clone()))

        # sample_and_group_all is different from sample_and_group!!!!
        if self.skip_global:
            n_samples = self.n_samples[:-1]
        else:
            n_samples = self.n_samples
        for i, n_samples in enumerate(n_samples):
            if n_samples == 1:
                grouped_feat = x.unsqueeze(1)
                grouped_feat = torch.cat(
                    [pos.unsqueeze(1), grouped_feat], dim=-1
                )
                #
                grouped_feat = apply_module_with_conv2d_bn(
                    grouped_feat, self.mlp_layers[i]
                ).squeeze(1)
                x, _ = torch.max(grouped_feat, dim=1, keepdim=True)
                sampled_pos = torch.zeros((bz, 1, 3), dtype=torch.float, device=pos.device)
                pos = sampled_pos
            else:
                grouped_feat, topk_dist, pos = self.sample_and_group(x, pos, n_samples, use_pos=True, k=64)
                grouped_feat = apply_module_with_conv2d_bn(
                    grouped_feat, self.mlp_layers[i]
                )
                cur_radius = self.radius[i]
                x = self.max_pooling_with_r(grouped_feat, topk_dist, r=cur_radius)
            cache.append((x.clone(), pos.clone()))

        # if self.skip_global:
        #     up_mlp_layers = self.up_mlp_layers # [1:]
        # else:
        #     up_mlp_layers = self.up_mlp_layers

        global_x = x
        # for i, up_conv_layers in enumerate(up_mlp_layers):
        #     prev_x, prev_pos = cache[-i-2][0], cache[-i-2][1]
        #     # print(prev_pos.size(), x.size(), pos.size())
        #     # interpolate x via pos & prev_pos
        #     interpolated_feats = self.interpolate_features(x, pos, prev_pos)
        #     # if prev_x is not None:
        #     #     prev_x = torch.cat([prev_pos, prev_x], dim=-1)
        #     # else:
        #     #     prev_x = prev_pos
        #     if prev_x is None:
        #         prev_x = prev_pos
        #     elif i == len(self.up_mlp_layers) - 1:
        #         prev_x = torch.cat([prev_x, prev_pos], dim=-1)
        #     # if without previous x, we only have the interpolated feature
        #     cur_up_feats = torch.cat([interpolated_feats, prev_x], dim=-1)
        #     x = apply_module_with_conv2d_bn(
        #         cur_up_feats.unsqueeze(2), up_conv_layers
        #     ).squeeze(2)
        #     pos = prev_pos
        return global_x, pos
        # if return_global:
        #     return x, global_x, pos
        # else:
        #     return x, pos
