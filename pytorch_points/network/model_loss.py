import torch
import pytorch3d.ops as ops
import numpy as np
from .._ext import losses
from . import geo_operations as geo_op


class UniformLaplacianSmoothnessLoss(torch.nn.Module):
    """
    Encourages minimal mean curvature shapes.
    """
    def __init__(self, num_point, faces, metric):
        super().__init__()
        self.laplacian = geo_op.UniformLaplacian()
        self.metric = metric
        self.faces = faces

    def __call__(self, vert, vert_ref=None):
        lap = self.laplacian(vert, self.faces)
        curve = torch.norm(lap, p=2, dim=-1)
        if vert_ref is not None:
            lap_ref = self.laplacian(vert, self.faces)
            curve_gt = torch.norm(lap_ref, p=2, dim=-1)
            loss = self.metric(curve, curve_gt)
        else:
            loss = curve
        return loss

class MeshLaplacianLoss(torch.nn.Module):
    """
    compare laplacian of two meshes with the same connectivity assuming known correspondence
    metric: an instance of a module e.g. L1Loss
    use_cot: cot laplacian is used instead of uniformlaplacian
    consistent_topology: assume face matrix is the same during the entire use
    precompute_L: assume vert1 is always the same
    """
    def __init__(self, metric, use_cot=False, use_norm=False, consistent_topology=False,
                 precompute_L=False):
        super().__init__()
        if use_cot:
            self.laplacian = geo_op.CotLaplacian()
        else:
            self.laplacian = geo_op.UniformLaplacian()

        self.use_norm = use_norm
        self.consistent_topology = consistent_topology
        self.metric = metric
        self.precompute_L = precompute_L
        self.L = None

    def forward(self, vert1, vert2=None, face=None):
        if not self.consistent_topology:
            self.laplacian.L = None

        if self.L is None or (not self.precompute_L):
            lap1 = self.laplacian(vert1, face)
            if self.use_norm:
                lap1 = torch.norm(lap1, dim=-1, p=2)
            if self.precompute_L:
                self.L = lap1
        else:
            lap1 = self.L

        if vert2 is not None:
            lap2 = self.laplacian(vert2, face)
            if self.use_norm:
                lap2 = torch.norm(lap2, dim=-1, p=2)
            return self.metric(lap1,lap2)
        else:
            assert(~self.precompute_L), "precompute_L must be False"
            return lap1.mean()

class PointLaplacianLoss(torch.nn.Module):
    """
    compare uniform laplacian of two point clouds assuming known or given correspondence
    metric: an instance of a module e.g. L1Loss
    """
    def __init__(self, nn_size, metric, use_norm=False):
        super().__init__()
        self.metric = metric
        self.nn_size = nn_size
        self.use_norm = use_norm

    def forward(self, point1, point2, idx12=None, *args, **kwargs):
        """
        point1: (B,N,D) ref points (where connectivity is computed)
        point2: (B,M,D) pred points, uses connectivity of point1
        idx12:  (B,N)   correspondence from 1 to 2
        """
        B = point1.shape[0]
        lap1, knn_idx = geo_op.pointUniformLaplacian(point1, nn_size=self.nn_size)
        if idx12 is not None:
            point2 = torch.gather(point2, 1, idx12.unsqueeze(-1).expand(-1,-1,3))
            lap2, _ = geo_op.pointUniformLaplacian(point2, nn_size=self.nn_size)
        else:
            assert(point2.shape[1] == point1.shape[1])
            lap2, _ = geo_op.pointUniformLaplacian(point2, knn_idx=knn_idx)
        if self.use_norm:
            lap1 = torch.norm(lap1, dim=-1, p=2)
            lap2 = torch.norm(lap2, dim=-1, p=2)
        return self.metric(lap1, lap2)


class PointEdgeLengthLoss(torch.nn.Module):
    """
    Penalize edge length change
    metric: an instance of a module e.g. L1Loss
    """
    def __init__(self, nn_size, metric):
        super().__init__()
        self.metric = metric
        self.nn_size = nn_size

    def forward(self, points_ref, points):
        """
        point1: (B,N,D) ref points (where connectivity is computed)
        point2: (B,N,D) pred points, uses connectivity of point1
        """
        # find neighborhood, (B,N,K,3), (B,N,K)
        _, knn_idx, group_points = ops.knn_points(points_ref, points_ref, K=self.nn_size+1, return_nn=True)
        knn_idx = knn_idx[:, :, 1:]
        group_points= group_points[:,:,1:,:]
        dist_ref = torch.norm(group_points - points_ref.unsqueeze(2), dim=-1, p=2)
        # dist_ref = torch.sqrt(dist_ref)
        # B,N,K,D
        group_points = torch.gather(points.unsqueeze(1).expand(-1, knn_idx.shape[1], -1, -1), 2, knn_idx.unsqueeze(-1).expand(-1, -1, -1, points.shape[-1]))
        dist = torch.norm(group_points - points.unsqueeze(2), dim=-1, p=2)
        # print(group_points, group_points2)
        return self.metric(dist_ref, dist)


class PointStretchLoss(torch.nn.Module):
    """
    penalize stretch only max(d/d_ref-1, 0)
    """
    def __init__(self, nn_size, reduction="mean"):
        super().__init__()
        self.nn_size = nn_size
        self.reduction = reduction

    def forward(self, points_ref, points):
        """
        point1: (B,N,D) ref points (where connectivity is computed)
        point2: (B,N,D) pred points, uses connectivity of point1
        """
        # find neighborhood, (B,N,K,3), (B,N,K), (B,N,K)
        _, knn_idx, group_points_ref = ops.knn_points(points_ref, points_ref, K=self.nn_size+1, return_nn=True)
        knn_idx = knn_idx[:, :, 1:]
        group_points_ref = group_points_ref[:,:,1:,:]
        dist_ref = torch.norm(group_points_ref - points_ref.unsqueeze(2), dim=-1, p=2)
        group_points = torch.gather(points.unsqueeze(1).expand(-1, knn_idx.shape[1], -1, -1), 2, knn_idx.unsqueeze(-1).expand(-1, -1, -1, points.shape[-1]))
        dist = torch.norm(group_points - points.unsqueeze(2), dim=-1, p=2)
        stretch = torch.max(dist/(dist_ref+1e-10)-1, torch.zeros_like(dist))
        if self.reduction == "mean":
            return torch.mean(stretch)
        elif self.reduction == "sum":
            return torch.mean(torch.sum(stretch, dim=-1))
        elif self.reduction == "none":
            return stretch
        elif self.reduction == "max":
            return torch.mean(torch.max(stretch, dim=-1)[0])
        else:
            raise NotImplementedError


class MeshEdgeLengthLoss(torch.nn.Module):
    """
    Penalize large edge deformation for meshes of the same topology (assuming correspondance)
    faces (B,F,L)
    """
    def __init__(self, metric, consistent_topology=False):
        super().__init__()
        self.metric = metric
        self.E = None
        self.consistent_topology = consistent_topology

    @staticmethod
    def getEV(faces, n_vertices):
        """return a list of B (E, 2) int64 tensor"""
        B, F, _ = faces.shape
        EV = []
        for b in range(B):
            EV.append(geo_op.edge_vertex_indices(faces[b]))
        return EV

    def forward(self, vert1, vert2, face=None):
        """
        vert1: (B, N, 3)
        vert2: (B, N, 3)
        faces:  (B, F, L)
        """
        assert(vert1.shape == vert2.shape)
        B, P, _ = vert1.shape
        F = face.shape[1]
        if (not self.consistent_topology) or (self.E is None):
            assert(face is not None), "Face is required"
            self.E = self.getEV(face, P)

        # (B, E, 2, 3)
        loss = []
        for b in range(B):
            edge_length1 = geo_op.get_edge_lengths(vert1[b], self.E[b])
            edge_length2 = geo_op.get_edge_lengths(vert2[b], self.E[b])
            loss.append(self.metric(edge_length1, edge_length2))

        loss = torch.stack(loss, dim=0)
        loss = torch.mean(loss)

        return loss


class MeshStretchLoss(torch.nn.Module):
    """
    Penalize increase of edge length max(len2/len1-1, 0)
    assuming the same triangulation
    ======
    Input:
        vert1 reference vertices (B,N,3)
        vert2 vertices (B,N,3)
        faces face vertex indices (same between vert1 and vert2)
    """
    def __init__(self, reduction="mean", consistent_topology=False):
        self.E = None
        self.reduction = reduction
        self.consistent_topology = consistent_topology
        super().__init__()


    @staticmethod
    def getEV(faces, n_vertices):
        """return a list of B (E, 2) int64 tensor"""
        B, F, _ = faces.shape
        EV = []
        for b in range(B):
            EV.append(geo_op.edge_vertex_indices(faces[b]))
        return EV

    def forward(self, vert1, vert2, face=None):
        assert(vert1.shape == vert2.shape)
        B, P, _ = vert1.shape
        F = face.shape[1]
        if (not self.consistent_topology) or self.E is None:
            assert(face is not None), "Face is required"
            self.E = self.getEV(face, P)

        # (B, E, 2, 3)
        loss = []
        for b in range(B):
            edge_length1 = geo_op.get_edge_lengths(vert1[b], self.E[b])
            edge_length2 = geo_op.get_edge_lengths(vert2[b], self.E[b])

            stretch = torch.max(edge_length2/edge_length1-1, torch.zeros_like(edge_length1))
            if self.reduction in ("mean", "none"):
                loss.append(stretch.mean())
            elif self.reduction == "max":
                loss.append(stretch.max())
            elif self.reduction == "sum":
                loss.append(stretch.sum())
            else:
                raise NotImplementedError

        loss = torch.stack(loss, dim=0)
        if self.reduction != "none":
            loss = loss.mean()

        return loss


class SimpleMeshRepulsionLoss(torch.nn.Module):
    """
    Penalize very short mesh edges 1/d
    """
    def __init__(self, threshold, edges=None, reduction="mean", consistent_topology=False):
        super().__init__()
        self.threshold2 = threshold*threshold
        self.edges = edges
        self.reduction = reduction

    def forward(self, verts, edges=None):
        """
        verts: (B, N, 3)
        edges:  (B, E, 2)
        """
        B, P, _ = verts.shape
        if edges is None:
            edges = self.edges
        assert(edges is not None)
        # (B, E, 2, 3)
        loss = []
        for b in range(B):
            edge_length1 = geo_op.get_edge_lengths(verts[b], edges)
            tmp = 1/(edge_length1+1e-6)
            tmp = torch.where(edge_length1 < self.threshold2, tmp, torch.zeros_like(tmp))
            if self.reduction in ("mean", "none"):
                tmp = tmp.mean()
            elif self.reduction == "max":
                tmp = tmp.max()
            elif self.reduction == "sum":
                tmp = tmp.sum()
            else:
                raise NotImplementedError
            loss.append(tmp)

        loss = torch.stack(loss, dim=0)
        if self.reduction != "none":
            loss = loss.mean()

        return loss

class SmapeLoss(torch.nn.Module):
    """
    relative L1 norm
    http://drz.disneyresearch.com/~jnovak/publications/KPAL/KPAL.pdf eq(2)
    """
    def __init__(self, epsilon=1e-8):
        super(SmapeLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, x, y):
        """
        x pred  (N,3)
        y label (N,3)
        """
        return torch.mean(torch.abs(x-y)/(torch.abs(x)+torch.abs(y)+self.epsilon))

class NormalLoss(torch.nn.Module):
    """
    compare the PCA normals of two point clouds assuming known or given correspondence
    ===
    params:
        pred : (B,N,3)
        gt   : (B,N,3)
        idx12: (B,N)
    """
    def __init__(self, nn_size=10, reduction="mean"):
        super().__init__()
        self.nn_size = nn_size
        self.reduction = reduction
        self.cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)

    def forward(self, gt, pred, idx12=None):
        gt_normals, idx = geo_op.batch_normals(gt, nn_size=self.nn_size, NCHW=False)
        if idx12 is not None:
            pred = torch.gather(pred, 1, idx12.unsqueeze(-1).expand(-1,-1,3))
            pred_normals, _ = geo_op.batch_normals(pred, nn_size=self.nn_size, NCHW=False)
        else:
            pred_normals, _ = geo_op.batch_normals(pred, nn_size=self.nn_size, NCHW=False, idx=idx)

        # compare the normal with the closest point
        loss = 1-self.cos(pred_normals, gt_normals)
        if self.reduction == "mean":
            return loss.mean(loss)
        elif self.reduction == "max":
            return (torch.max(loss, dim=-1)[0]).mean()
        elif self.reduction == "sum":
            return torch.sum(loss, dim=-1).mean()
        elif self.reduction == "none":
            return loss



class SimplePointRepulsionLoss(torch.nn.Module):
    """
    Penalize point-to-point distance which is smaller than a threshold
    params:
        points:  (B,N,C)
        nn_size: neighborhood size
    """
    def __init__(self, nn_size, radius, reduction="mean"):
        super().__init__()
        self.nn_size = nn_size
        self.reduction = reduction
        self.radius2 = radius*radius

    def forward(self, points, knn_idx=None):
        batchSize, PN, _ = points.shape
        if knn_idx is None:
            distance2, knn_idx, knn_points=ops.knn_points(points, points, K=self.nn_size+1, return_nn=True)
            knn_points = knn_points[:, :, 1:, :].contiguous().detach()
            knn_idx = knn_idx[:, :, 1:].contiguous()
        else:
            knn_points = torch.gather(points.unsqueeze(1).expand(-1, PN, -1, -1), 2, knn_idx.unsqueeze(-1).expand(-1, -1, -1, points.shape[-1]))

        knn_v = knn_points - points.unsqueeze(dim=2)
        distance2 = torch.sum(knn_v * knn_v, dim=-1)
        loss = 1/torch.sqrt(distance2+1e-4)
        loss = torch.where(distance2 < self.radius2, loss, torch.zeros_like(loss))
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "max":
            return torch.mean(torch.max(loss, dim=-1)[0])
        elif self.reduction == "sum":
            return loss.mean(torch.sum(loss, dim=-1))
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError
        return loss


class NmDistanceFunction(torch.autograd.Function):
    """3D point set to 3D point set distance"""
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        assert(xyz1.is_contiguous())
        assert(xyz2.is_contiguous())
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        assert(xyz1.dtype==xyz2.dtype)
        dist1 = torch.zeros(batchsize, n, dtype=xyz1.dtype)
        dist2 = torch.zeros(batchsize, m, dtype=xyz1.dtype)

        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)

        dist1 = dist1.cuda()
        dist2 = dist2.cuda()
        idx1 = idx1.cuda()
        idx2 = idx2.cuda()
        losses.nmdistance_forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        ctx.mark_non_differentiable(idx1, idx2)
        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, graddist1, graddist2, gradNone1, gradNone2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        gradxyz1 = torch.zeros_like(xyz1)
        gradxyz2 = torch.zeros_like(xyz2)

        gradxyz1 = gradxyz1.cuda()
        gradxyz2 = gradxyz2.cuda()
        losses.nmdistance_backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        return gradxyz1, gradxyz2


nndistance = NmDistanceFunction.apply  # type: ignore


class LabeledNmdistanceFunction(torch.autograd.Function):
    """ CD within the same category, ignore points that have no matching category """
    @staticmethod
    def forward(ctx, xyz1, xyz2, label1, label2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        assert(xyz1.dtype==xyz2.dtype)
        label1 = label1.to(dtype=xyz1.dtype)
        label2 = label2.to(dtype=xyz1.dtype)
        dist1 = torch.zeros(batchsize, n, dtype=xyz1.dtype)
        dist2 = torch.zeros(batchsize, m, dtype=xyz1.dtype)

        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)

        dist1 = dist1.cuda()
        dist2 = dist2.cuda()
        idx1 = idx1.cuda()
        idx2 = idx2.cuda()
        losses.labeled_nmdistance_forward(xyz1, xyz2, label1, label2,  dist1, dist2, idx1, idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        ctx.mark_non_differentiable(idx1, idx2)
        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, graddist1, graddist2, gradNone1, gradNone2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        gradxyz1 = torch.zeros_like(xyz1)
        gradxyz2 = torch.zeros_like(xyz2)

        gradxyz1 = gradxyz1.cuda()
        gradxyz2 = gradxyz2.cuda()
        losses.nmdistance_backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        return gradxyz1, gradxyz2, None, None

labeled_nndistance = LabeledNmdistanceFunction.apply
