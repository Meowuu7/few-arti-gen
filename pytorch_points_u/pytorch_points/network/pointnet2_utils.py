import torch
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn as nn
from typing import Tuple

from .._ext import sampling
from .operations import grouping_operation, ball_query


class ThreeNN(Function):

    @staticmethod
    def forward(ctx, unknown: torch.Tensor, known: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the three nearest neighbors of unknown in known
        :param ctx:
        :param unknown: (B, N, 3)
        :param known: (B, M, 3)
        :return:
            dist: (B, N, 3) l2 distance to the three nearest neighbors
            idx: (B, N, 3) index of 3 nearest neighbors
        """
        assert unknown.is_contiguous()
        assert known.is_contiguous()

        B, N, _ = unknown.size()
        m = known.size(1)
        dist2 = torch.cuda.FloatTensor(B, N, 3)
        idx = torch.cuda.IntTensor(B, N, 3)

        sampling.three_nn_wrapper(B, N, m, unknown, known, dist2, idx)
        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Performs weight linear interpolation on 3 features
        :param ctx:
        :param features: (B, C, M) Features descriptors to be interpolated from
        :param idx: (B, n, 3) three nearest neighbors of the target features in features
        :param weight: (B, n, 3) weights
        :return:
            output: (B, C, N) tensor of the interpolated features
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()
        assert weight.is_contiguous()

        B, c, m = features.size()
        n = idx.size(1)
        ctx.three_interpolate_for_backward = (idx, weight, m)
        output = torch.cuda.FloatTensor(B, c, n)

        sampling.three_interpolate_wrapper(B, c, m, n, features, idx, weight, output)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param ctx:
        :param grad_out: (B, C, N) tensor with gradients of outputs
        :return:
            grad_features: (B, C, M) tensor with gradients of features
            None:
            None:
        """
        idx, weight, m = ctx.three_interpolate_for_backward
        B, c, n = grad_out.size()

        grad_features = torch.cuda.FloatTensor(B, c, m).zero_()
        grad_out_data = grad_out.data.contiguous()

        sampling.three_interpolate_grad_wrapper(B, c, n, m, grad_out_data, idx, weight, grad_features.data)
        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply


class QueryAndGroup(nn.Module):
    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        """
        :param radius: float, radius of ball
        :param nsample: int, maximum number of features to gather in the ball
        :param use_xyz:
        """
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None) -> Tuple[torch.Tensor]:
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features


class GroupAll(nn.Module):
    def __init__(self, use_xyz: bool = True):
        super().__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None):
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: ignored
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, C + 3, 1, N)
        """
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features
