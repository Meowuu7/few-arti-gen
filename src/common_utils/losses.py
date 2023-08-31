from typing import Union
import torch
from torch_batch_svd import svd

# from pytorch3d.ops.knn import knn_gather, knn_points
# from pytorch3d.structures.pointclouds import Pointclouds

from laplacian import mesh_laplacian

# import pytorch3d

def triangle_area(v, f):
    A = torch.gather(v, 0, f[:, 0].unsqueeze(-1).expand(-1, 3))
    B = torch.gather(v, 0, f[:, 1].unsqueeze(-1).expand(-1, 3))
    C = torch.gather(v, 0, f[:, 2].unsqueeze(-1).expand(-1, 3))
    normal = torch.cross((B - A), (C - A), dim = -1)
    return normal.norm(p = 2, dim = -1)

def normal_loss(src_meshes, def_meshes):
    N = len(src_meshes)
    src_v = src_meshes.verts_packed()  
    def_v = def_meshes.verts_packed() 
    src_f = src_meshes.faces_packed()
    
    num_faces_per_mesh = src_meshes.num_faces_per_mesh()  
    faces_packed_idx = src_meshes.faces_packed_to_mesh_idx() 
    w = num_faces_per_mesh.gather(0, faces_packed_idx)  
    w = 1.0 / w.float()
    
    cos = torch.nn.CosineSimilarity(dim = -1, eps = 1e-18)
    src_A = torch.gather(src_v, 0, src_f[:, 0].unsqueeze(-1).expand(-1, 3))
    src_B = torch.gather(src_v, 0, src_f[:, 1].unsqueeze(-1).expand(-1, 3))
    src_C = torch.gather(src_v, 0, src_f[:, 2].unsqueeze(-1).expand(-1, 3))
    def_A = torch.gather(def_v, 0, src_f[:, 0].unsqueeze(-1).expand(-1, 3))
    def_B = torch.gather(def_v, 0, src_f[:, 1].unsqueeze(-1).expand(-1, 3))
    def_C = torch.gather(def_v, 0, src_f[:, 2].unsqueeze(-1).expand(-1, 3))
    src_normal = torch.cross((src_B - src_A), (src_C - src_A))
    def_normal = torch.cross((def_B - def_A), (def_C - def_A))
    with torch.no_grad():
        weight = triangle_area(src_v, src_f) + triangle_area(def_v, src_f)
        weight = (weight / weight.mean() + 1) / 2
    loss = (1 - cos(src_normal, def_normal)) * weight
    loss = loss * w
    return loss.sum() / N


def normal_loss_raw(src_meshes, def_meshes):
    # N = len(src_meshes)
    src_v, src_f = src_meshes
    def_v, def_f = def_meshes
    
    # src_v = src_meshes # .verts_packed()  
    # def_v = def_meshes # .verts_packed() 
    # src_f = src_meshes.faces_packed()
    
    # num_faces_per_mesh = src_meshes.num_faces_per_mesh()  
    # faces_packed_idx = src_meshes.faces_packed_to_mesh_idx() 
    # w = num_faces_per_mesh.gather(0, faces_packed_idx)  
    # w = 1.0 / w.float()
    
    num_faces_per_mesh = def_f.size(0)
    w = 1.0 / float(num_faces_per_mesh)
    
    cos = torch.nn.CosineSimilarity(dim = -1, eps = 1e-18)
    src_A = torch.gather(src_v, 0, src_f[:, 0].unsqueeze(-1).expand(-1, 3))
    src_B = torch.gather(src_v, 0, src_f[:, 1].unsqueeze(-1).expand(-1, 3))
    src_C = torch.gather(src_v, 0, src_f[:, 2].unsqueeze(-1).expand(-1, 3))
    def_A = torch.gather(def_v, 0, src_f[:, 0].unsqueeze(-1).expand(-1, 3))
    def_B = torch.gather(def_v, 0, src_f[:, 1].unsqueeze(-1).expand(-1, 3))
    def_C = torch.gather(def_v, 0, src_f[:, 2].unsqueeze(-1).expand(-1, 3))
    src_normal = torch.cross((src_B - src_A), (src_C - src_A))
    def_normal = torch.cross((def_B - def_A), (def_C - def_A))
    with torch.no_grad():
        weight = triangle_area(src_v, src_f) + triangle_area(def_v, src_f)
        weight = (weight / weight.mean() + 1) / 2
    loss = (1 - cos(src_normal, def_normal)) * weight
    loss = loss * w
    return loss.sum() # / N
    
    

def laplacian_loss(src_meshes, def_meshes):
    return mesh_laplacian(src_meshes, def_meshes) 

def _validate_chamfer_reduction_inputs(
    batch_reduction: Union[str, None], point_reduction: str
):
    """Check the requested reductions are valid.

    Args:
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].
    """
    if batch_reduction is not None and batch_reduction not in ["mean", "sum"]:
        raise ValueError('batch_reduction must be one of ["mean", "sum"] or None')
    if point_reduction not in ["mean", "sum"]:
        raise ValueError('point_reduction must be one of ["mean", "sum"]')

# def _handle_pointcloud_input(
#     points: Union[torch.Tensor, Pointclouds],
#     lengths: Union[torch.Tensor, None],
#     normals: Union[torch.Tensor, None],
# ):
#     """
#     If points is an instance of Pointclouds, retrieve the padded points tensor
#     along with the number of points per batch and the padded normals.
#     Otherwise, return the input points (and normals) with the number of points per cloud
#     set to the size of the second dimension of `points`.
#     """
#     if isinstance(points, Pointclouds):
#         X = points.points_padded()
#         lengths = points.num_points_per_cloud()
#         normals = points.normals_padded()  # either a tensor or None
#     elif torch.is_tensor(points):
#         # pyre-fixme[16]: `Tensor` has no attribute `ndim`.
#         if points.ndim != 3:
#             raise ValueError("Expected points to be of shape (N, P, D)")
#         X = points
#         if lengths is not None and (
#             lengths.ndim != 1 or lengths.shape[0] != X.shape[0]
#         ):
#             raise ValueError("Expected lengths to be of shape (N,)")
#         if lengths is None:
#             lengths = torch.full(
#                 (X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device
#             )
#         if normals is not None and normals.ndim != 3:
#             raise ValueError("Expected normals to be of shape (N, P, 3")
#     else:
#         raise ValueError(
#             "The input pointclouds should be either "
#             + "Pointclouds objects or torch.Tensor of shape "
#             + "(minibatch, num_points, 3)."
#         )
#     return X, lengths, normals




def chamfer_distance_raw(x, y):
    dist_x_y = torch.sum((x.unsqueeze(2) - y.unsqueeze(1)) ** 2, dim=-1) ### N x P1 x P2
    dist_x_to_y, dist_y_to_x = torch.min(dist_x_y, dim=-1)[0], torch.min(dist_x_y, dim=-2)[0] #
    dist_x_to_y = dist_x_to_y.mean(dim=-1)
    dist_y_to_x = dist_y_to_x.mean(dim=-1)
    dist = dist_x_to_y.mean() + dist_y_to_x.mean()
    return dist


def cage_shift_loss(cage, pc):
    cage_shift = torch.mean(cage, dim=1) - torch.mean(pc, dim=1)
    ori_cage_shift_loss = torch.mean(torch.nn.functional.softshrink(torch.sum(cage_shift**2, dim=-1), lambd=0.1))
    return ori_cage_shift_loss


def get_extends(pc):
    pc_maxx, _ = torch.max(pc, dim=1)
    pc_minn, _ = torch.min(pc, dim=1)
    pc_extents = pc_maxx - pc_minn #### bsz x 3 
    return pc_extents


def cage_extends_loss(cage, pc):
    cage_extents = get_extends(cage)
    pc_extents = get_extends(pc)
    diff = torch.sum((cage_extents - pc_extents) ** 2, dim=-1).mean()
    return diff

def cage_verts_offset_loss(gt_def_cages, def_cages):
    ## bsz x nn_pts x 3 
    mse_loss = torch.sum((gt_def_cages - def_cages) ** 2, dim=-1).mean().mean()
    return mse_loss


def chamfer_distance(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    x_normals=None,
    y_normals=None,
    weights=None,
    batch_reduction: Union[str, None] = None,
    point_reduction: str = "mean",
):
    """
    Chamfer distance between two pointclouds x and y.

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].

    Returns:
        2-element tuple containing

        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None.
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    y_mask = (
        torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_norm_x = x.new_zeros(())
    cham_norm_y = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if is_y_heterogeneous:
        cham_y[y_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)
        cham_y *= weights.view(N, 1)

    if return_normals:
        # Gather the normals using the indices and keep only value for k=0
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]
        y_normals_near = knn_gather(x_normals, y_nn.idx, x_lengths)[..., 0, :]

        cham_norm_x = 1 - torch.abs(
            F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
        )
        cham_norm_y = 1 - torch.abs(
            F.cosine_similarity(y_normals, y_normals_near, dim=2, eps=1e-6)
        )

        if is_x_heterogeneous:
            # pyre-fixme[16]: `int` has no attribute `__setitem__`.
            cham_norm_x[x_mask] = 0.0
        if is_y_heterogeneous:
            cham_norm_y[y_mask] = 0.0

        if weights is not None:
            cham_norm_x *= weights.view(N, 1)
            cham_norm_y *= weights.view(N, 1)

    # Apply point reduction
    cham_x = cham_x.sum(1)  # (N,)
    cham_y = cham_y.sum(1)  # (N,)
    if return_normals:
        cham_norm_x = cham_norm_x.sum(1)  # (N,)
        cham_norm_y = cham_norm_y.sum(1)  # (N,)
    if point_reduction == "mean":
        cham_x /= x_lengths
        cham_y /= y_lengths
        if return_normals:
            cham_norm_x /= x_lengths
            cham_norm_y /= y_lengths

    if batch_reduction is not None:
        # batch_reduction == "sum"
        cham_x = cham_x.sum()
        cham_y = cham_y.sum()
        if return_normals:
            cham_norm_x = cham_norm_x.sum()
            cham_norm_y = cham_norm_y.sum()
        if batch_reduction == "mean":
            div = weights.sum() if weights is not None else N
            cham_x /= div
            cham_y /= div
            if return_normals:
                cham_norm_x /= div
                cham_norm_y /= div

    cham_dist = cham_x + cham_y
    cham_normals = cham_norm_x + cham_norm_y if return_normals else None

    return cham_dist#, cham_normals


def basis_reg_losses(tot_basis, tot_coefs, ):
    try:
        nb = tot_basis[0][0].size(1) ### bsz x nb x (nk x 3)
        # print("here for dis!")
        tot_ortho_loss = []
        tot_svd_loss = []
        tot_sp_loss = []
        for i_bsz in range(len(tot_basis)):
            cur_bsz_basis = tot_basis[i_bsz]
            cur_bsz_coefs = tot_coefs[i_bsz]
            cur_bsz_ortho_loss = []
            cur_bsz_svd_loss = []
            cur_bsz_sp_loss = []
            for i_cvx, cur_cvx_basis in enumerate(cur_bsz_basis):
                dot = torch.bmm(cur_cvx_basis.abs(), cur_cvx_basis.transpose(1, 2).abs())
                dot[:, range(nb), range(nb)] = 0 ### dot of the basis
                cur_cvx_ortho_loss = dot.norm(p=2, dim=(1, 2)).mean()
                cur_bsz_ortho_loss.append(cur_cvx_ortho_loss)
                
                cur_cvx_basis = cur_cvx_basis.reshape(1 * nb, -1, 3)
                nn_cvx_keypts = cur_cvx_basis.size(-2)
                tmp = torch.bmm(cur_cvx_basis.transpose(1, 2), cur_cvx_basis)
                _, s, _ = svd(tmp)
                cur_cvx_svd_loss = s[:, 2].mean()
                cur_bsz_svd_loss.append(cur_cvx_svd_loss)
                
                cur_cvx_coefs = cur_bsz_coefs[i_cvx]
                cat_sp_loss = cur_cvx_basis.view(1, -1, nn_cvx_keypts * 3).norm(p=1, dim=2).mean() \
            + cur_cvx_coefs.view(1, -1).norm(p=1, dim=-1).mean()
                cur_bsz_sp_loss.append(cat_sp_loss)
                
            cur_bsz_ortho_loss = sum(cur_bsz_ortho_loss) / float(len(cur_bsz_ortho_loss))
            cur_bsz_svd_loss = sum(cur_bsz_svd_loss) / float(len(cur_bsz_svd_loss))
            cur_bsz_sp_loss = sum(cur_bsz_sp_loss) / float(len(cur_bsz_sp_loss))
            tot_ortho_loss.append(cur_bsz_ortho_loss)
            tot_svd_loss.append(cur_bsz_svd_loss)
            tot_sp_loss.append(cur_bsz_sp_loss)
        
        tot_ortho_loss = sum(tot_ortho_loss) / float(len(tot_ortho_loss))
        tot_svd_loss = sum(tot_svd_loss) / float(len(tot_svd_loss))
        tot_sp_loss = sum(tot_sp_loss) / float(len(tot_sp_loss))
    except:
        tot_ortho_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean()
        tot_svd_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean()
        tot_sp_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean()
    return tot_ortho_loss, tot_svd_loss, tot_sp_loss