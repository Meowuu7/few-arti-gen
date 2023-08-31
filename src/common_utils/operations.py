"""
code courtesy of
https://github.com/erikwijmans/Pointnet2_PyTorch
"""

import torch
import numpy as np
from scipy import sparse

def channel_shuffle(x, groups=2):
    '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
    N, C, H, W = x.size()
    g = groups
    return x.view(N, g, C/g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)


def jitter_perturbation_point_cloud(batch_data, sigma=0.005, clip=0.02, is_2D=False, NCHW=True):
    if NCHW:
        batch_data = batch_data.transpose(1, 2)

    batch_size = batch_data.shape[0]
    chn = 2 if is_2D else 3
    jittered_data = sigma * torch.randn_like(batch_data)
    for b in range(batch_size):
        jittered_data[b].clamp_(-clip[b].item(), clip[b].item())
    jittered_data[:, :, chn:] = 0
    jittered_data += batch_data
    if NCHW:
        jittered_data = jittered_data.transpose(1, 2)
    return jittered_data



class GatherFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, idx):
        r"""
        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor
        idx : torch.Tensor
            (B, npoint) tensor of the features to gather
        Returns
        -------
        torch.Tensor
            (B, C, npoint) tensor
        """
        features = features.contiguous()
        idx = idx.contiguous()
        idx = idx.to(dtype=torch.int32)

        B, npoint = idx.size()
        _, C, N = features.size()

        output = torch.empty(
            B, C, npoint, dtype=features.dtype, device=features.device)
        sampling.gather_forward(
            B, C, N, npoint, features, idx, output
        )

        ctx.save_for_backward(idx)
        ctx.C = C
        ctx.N = N
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, = ctx.saved_tensors
        B, npoint = idx.size()

        grad_features = torch.zeros(
            B, ctx.C, ctx.N, dtype=grad_out.dtype, device=grad_out.device)
        sampling.gather_backward(
            B, ctx.C, ctx.N, npoint, grad_out.contiguous(), idx, grad_features
        )

        return grad_features, None


gather_points = GatherFunction.apply  # type: ignore


class BallQuery(torch.autograd.Function):
    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz):
        r"""
        Parameters
        ----------
        radius : float
            radius of the balls
        nsample : int
            maximum number of features in the balls
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the ball query
        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        return sampling.ball_query(new_xyz, xyz, radius, nsample)

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply  # type: ignore


class GroupingOperation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, idx):
        r"""
        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor of features to group
        idx : torch.Tensor
            (B, npoint, nsample) tensor containing the indicies of features to group with
        Returns
        -------
        torch.Tensor
            (B, C, npoint, nsample) tensor
        """
        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()

        ctx.for_backwards = (idx, N)

        return sampling.group_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        r"""
        Parameters
        ----------
        grad_out : torch.Tensor
            (B, C, npoint, nsample) tensor of the gradients of the output from forward
        Returns
        -------
        torch.Tensor
            (B, C, N) gradient of the features
        None
        """
        idx, N = ctx.for_backwards

        grad_features = sampling.group_points_grad(grad_out.contiguous(), idx, N)

        return grad_features, None


grouping_operation = GroupingOperation.apply  # type: ignore


class QueryAndGroup(torch.nn.Module):
    r"""
    Groups with a ball query of radius
    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, nsample, use_xyz=True):
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz, new_xyz, features=None):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)
        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """
        # (B, npoint, k)
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        # (B, 3, N)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features

class BatchSVDFunction(torch.autograd.Function):
    """
    batched svd implemented by https://github.com/KinglittleQ/torch-batch-svd
    """
    @staticmethod
    def forward(ctx, x):
        ctx.device = x.device
        if not torch.cuda.is_available():
            assert(RuntimeError), "BatchSVDFunction only runs on gpu"
        x = x.cuda()
        U, S, V = linalg.batch_svd_forward(x, True, 1e-7, 100)
        k = S.size(1)
        U = U[:, :, :k]
        V = V[:, :, :k]
        ctx.save_for_backward(x, U, S, V)
        U = U.to(ctx.device)
        S = S.to(ctx.device)
        V = V.to(ctx.device)
        return U, S, V

    @staticmethod
    def backward(ctx, grad_u, grad_s, grad_v):
        x, U, S, V = ctx.saved_variables

        grad_out = linalg.batch_svd_backward(
            [grad_u, grad_s, grad_v],
            x, True, True, U, S, V
        )

        return grad_out.to(device=ctx.device)


def batch_svd(x):
    """
    input:
        x --- shape of [B, M, N], k = min(M,N)
    return:
        U, S, V = batch_svd(x) where x = USV^T
        U [M, k]
        V [N, k]
        S [B, k] in decending order
    """
    assert(x.dim() == 3)
    return BatchSVDFunction.apply(x)

def normalize(tensor, dim=-1):
    """normalize tensor in specified dimension"""
    return torch.nn.functional.normalize(tensor, p=2, dim=dim, eps=1e-12, out=None)

def sqrNorm(tensor, dim=-1, keepdim=False):
    """squared L2 norm"""
    return torch.sum(tensor*tensor, dim=dim, keepdim=keepdim)


def dot_product(tensor1, tensor2, dim=-1, keepdim=False):
    return torch.sum(tensor1*tensor2, dim=dim, keepdim=keepdim)

def cross_product_2D(tensor1, tensor2, dim=1):
    # assert(tensor1.shape[dim] == tensor2.shape[dim] and tensor1.shape[dim] == 2)
    output = torch.narrow(tensor1, dim, 0, 1) * torch.narrow(tensor2, dim, 1, 1) - torch.narrow(tensor1, dim, 1, 1) * torch.narrow(tensor2, dim, 0, 1)
    return output.squeeze(dim)

class ScatterAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, idx, dim, out_size, fill=0.0):
        out = torch.full(out_size, fill, device=src.device, dtype=src.dtype)
        ctx.save_for_backward(idx)
        out.scatter_add_(dim, idx, src)
        ctx.mark_non_differentiable(idx)
        ctx.dim = dim
        return out

    @staticmethod
    def backward(ctx, ograd):
        idx, = ctx.saved_tensors
        grad = torch.gather(ograd, ctx.dim, idx)
        return grad, None, None, None, None

_scatter_add = ScatterAdd.apply

def scatter_add(src, idx, dim, out_size=None, fill=0.0):
    if out_size is None:
        out_size = list(src.size())
        dim_size = idx.max().item()+1
        out_size[dim] = dim_size
    return _scatter_add(src, idx, dim, out_size, fill)



