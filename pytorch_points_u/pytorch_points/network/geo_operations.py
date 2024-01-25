import torch
import pytorch3d.ops as ops
from .._ext import sampling
from ..utils.pytorch_utils import check_values, save_grad, saved_variables
from .operations import batch_svd, normalize, dot_product, scatter_add, cross_product_2D, gather_points
import numpy as np
from scipy import sparse

PI = 3.1415927

class FurthestPointSampling(torch.autograd.Function):

    @staticmethod
    def forward(ctx, xyz, npoint, seedIdx):
        r"""
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of features in the sampled set
        Returns
        -------
        torch.LongTensor
            (B, npoint) tensor containing the indices

        """
        B, N, _ = xyz.size()

        idx = torch.empty([B, npoint], dtype=torch.int32, device=xyz.device)
        temp = torch.full([B, N], 1e10, dtype=torch.float32, device=xyz.device)
        sampling.furthest_sampling(
            npoint, seedIdx, xyz, temp, idx
        )
        ctx.mark_non_differentiable(idx)
        return idx


__furthest_point_sample = FurthestPointSampling.apply  # type: ignore


def furthest_point_sample(xyz, npoint, NCHW=True, seedIdx=0):
    """
    :param
        xyz (B, 3, N) or (B, N, 3)
        npoint a constant
    :return
        torch.LongTensor
            (B, npoint) tensor containing the indices
        torch.FloatTensor
            (B, npoint, 3) or (B, 3, npoint) point sets"""
    assert(xyz.dim() == 3), "input for furthest sampling must be a 3D-tensor, but xyz.size() is {}".format(xyz.size())
    # need transpose
    if NCHW:
        xyz = xyz.transpose(2, 1).contiguous()

    assert(xyz.size(2) == 3), "furthest sampling is implemented for 3D points"
    idx = __furthest_point_sample(xyz, npoint, seedIdx)
    sampled_pc = gather_points(xyz.transpose(2, 1).contiguous(), idx)
    if not NCHW:
        sampled_pc = sampled_pc.transpose(2, 1).contiguous()
    return idx, sampled_pc


def normalize_point_batch_to_sphere(pc: torch.Tensor, NCHW=True):
    """
    normalize a batch of point clouds
    :param
        pc      [B, N, 3] or [B, 3, N]
        NCHW    if True, treat the second dimension as channel dimension
    :return
        pc      normalized point clouds, same shape as input
        centroid [B, 1, 3] or [B, 3, 1] center of point clouds
        furthest_distance [B, 1, 1] scale of point clouds
    """
    point_axis = 2 if NCHW else 1
    dim_axis = 1 if NCHW else 2
    centroid = torch.mean(pc, dim=point_axis, keepdim=True)
    pc = pc - centroid
    furthest_distance, _ = torch.max(
        torch.sqrt(torch.sum(pc ** 2, dim=dim_axis, keepdim=True)), dim=point_axis, keepdim=True)
    pc = pc / furthest_distance
    return pc, centroid, furthest_distance


def batch_normals(points, base=None, nn_size=20, NCHW=True, idx=None):
    """
    compute normals vectors for batched points [B, C, M]
    If base is given, compute the normals of points using the neighborhood in base
    The direction of normal could flip.

    Args:
        points:  (B,C,M)
        base:    (B,C,N)
        idx      (B,M,nn_size)
    Returns:
        normals: (B,C,M)
    """
    if base is None:
        base = points

    if NCHW:
        points = points.transpose(2, 1).contiguous()
        base = base.transpose(2, 1).contiguous()

    assert(nn_size < base.shape[1])
    batch_size, M, C = points.shape
    # B,M,k,C
    if idx is None:
        _, idx, grouped_points = ops.knn_points(points, base, K=nn_size, return_nn=True)
    else:
        grouped_points = torch.gather(base.unsqueeze(1).expand(-1,M,-1,-1), 2, idx.unsqueeze(-1).expand(-1,-1,-1,C))
    group_center = torch.mean(grouped_points, dim=2, keepdim=True)
    points = grouped_points - group_center
    allpoints = points.view(-1, nn_size, C).contiguous()
    # MB,C,k
    U, S, V = batch_svd(allpoints)
    # V is MBxCxC, last_u MBxC
    normals = V[:, :, -1]
    normals = normals.view(batch_size, M, C)
    if NCHW:
        normals = normals.transpose(1, 2)
    return normals, idx


def pointUniformLaplacian(points, knn_idx=None, nn_size=3):
    """
    Args:
        points: (B, N, 3)
        knn_idx: (B, N, K)
    Returns:
        laplacian: (B, N, 1)
    """
    batch_size, num_points, _ = points.shape
    if knn_idx is None:
        # find neighborhood, (B,N,K,3), (B,N,K)
        _, knn_idx, group_points = ops.knn_points(points, points, K=nn_size+1, return_nn=True)
        knn_idx = knn_idx[:, :, 1:]
        group_points = group_points[:, :, 1:, :]
    else:
        points_expanded = points.unsqueeze(dim=1).expand(
            (-1, num_points, -1, -1))
        # BxNxk -> BxNxNxC
        index_batch_expanded = knn_idx.unsqueeze(dim=-1).expand(
            (-1, -1, -1, points.size(-1)))
        # BxMxkxC
        group_points = torch.gather(points_expanded, 2, index_batch_expanded)

    lap = -torch.sum(group_points, dim=2)/knn_idx.shape[2] + points
    return lap, knn_idx


class UniformLaplacian(torch.nn.Module):
    """
    uniform laplacian for mesh
    vertex B,N,D
    faces  B,F,L
    """
    def __init__(self):
        super().__init__()
        self.L = None

    def computeLaplacian(self, V, F):
        batch, nv = V.shape[:2]
        V = V.reshape(-1, V.shape[-1])
        face_deg = F.shape[-1]
        offset = torch.arange(0, batch).reshape(-1, 1, 1) * nv
        faces = F + offset.to(device=F.device)
        faces = faces.reshape(-1, face_deg)
        # offset index by batch
        row = faces[:, [i for i in range(face_deg)]].reshape(-1)
        col = faces[:, [i for i in range(1, face_deg)]+[0]].reshape(-1)
        indices = torch.stack([row, col], dim=0)

        # (BN,BN)
        L = torch.sparse_coo_tensor(indices, -torch.ones_like(col, dtype=V.dtype, device=V.device), size=[nv*batch, nv*batch])
        L = L.t() + L
        # (BN)
        Lii = -torch.sparse.sum(L, dim=[1]).to_dense()
        M = torch.sparse_coo_tensor(torch.arange(nv*batch).unsqueeze(0).expand(2, -1), Lii, size=(nv*batch, nv*batch))
        L = L + M
        self.L = L
        self.Lii = Lii

    def forward(self, verts, faces=None):
        batch, nv = verts.shape[:2]
        assert(verts.shape[0] == batch)
        assert(verts.shape[1] == nv)

        if self.L is None:
            assert(faces is not None)
            self.computeLaplacian(verts, faces)

        if self.L.shape[0] != (verts.shape[0]*verts.shape[1]):
            # during initialization, used a single batch point set
            assert(self.L.shape[0] == verts.shape[1])
            x = [torch.sparse.mm(self.L, verts[b])/(self.Lii.unsqueeze(-1)+1e-12) for b in range(batch)]
            x = torch.stack(x, dim=0)
        else:
            x = torch.sparse.mm(self.L, verts.reshape(-1,3))
            x = x / (self.Lii.unsqueeze(-1)+1e-12)
            x = x.reshape([batch, nv, -1])
        return x

#############
### cotangent laplacian from 3D-coded ###
#############
def convert_as(src, trg):
    src = src.type_as(trg)
    if src.is_cuda:
        src = src.cuda(device=trg.get_device())

    src.requires_grad_(trg.requires_grad)
    return src

class CotLaplacian(torch.nn.Module):
    def __init__(self):
        """
        Faces is B x F x 3, cuda torch Variabe.
        Reuse faces.
        """
        super().__init__()
        self.L = None

    def computeLaplacian(self, V, F):
        print('Computing the Laplacian!')
        F_np = F.data.cpu().numpy()
        F = F.data
        B,N,_ = V.shape
        # Compute cotangents
        C = cotangent(V, F)
        assert(check_values(C))
        C_np = C.cpu().numpy()
        batchC = C_np.reshape(-1, 3)
        # Adjust face indices to stack:
        offset = np.arange(0, V.size(0)).reshape(-1, 1, 1) * V.size(1)
        F_np = F_np + offset
        batchF = F_np.reshape(-1, 3)

        rows = batchF[:, [1, 2, 0]].reshape(-1) #1,2,0 i.e to vertex 2-3 associate cot(23)
        cols = batchF[:, [2, 0, 1]].reshape(-1) #2,0,1 This works because triangles are oriented ! (otherwise 23 could be associated to more than 1 cot))

        # Final size is BN x BN
        BN = B*N
        L = sparse.csr_matrix((batchC.reshape(-1), (rows, cols)), shape=(BN,BN))
        L = L + L.T
        # np.sum on sparse is type 'matrix', so convert to np.array
        M = sparse.diags(np.array(np.sum(L, 1)).reshape(-1), format='csr')
        L = L - M
        # remember this
        self.L = L

    def forward(self, V, F=None):
        """
        If forward is explicitly called, V is still a Parameter or Variable
        But if called through __call__ it's a tensor.
        This assumes __call__ was used.

        Input:
           V: B x N x 3
           F: B x F x 3
        Outputs: L x B x N x 3

        Numpy also doesnt support sparse tensor, so stack along the batch
        """
        if self.L is None:
            assert(F is not None)
            self.computeLaplacian(V, F)
        Lx = _cotLx(V, self.L)
        Lx.requires_grad_(V.requires_grad)
        return Lx


class _CotLaplacianBatchLx(torch.autograd.Function):
    @staticmethod
    def forward(ctx, V, L):
        """V: (B,N,3), L: numpy sparse matrix (BN, 3)"""
        ctx.L = L
        batchV = V.reshape(-1, 3).cpu().numpy()
        Lx = L.dot(batchV)
        Lx = Lx.reshape(V.shape)
        Lx = convert_as(torch.Tensor(Lx), V)
        return Lx

    @staticmethod
    def backward(ctx, grad_out):
        """
        Just L'g = Lg
        Args:
           grad_out: B x N x 3
        Returns:
           grad_vertices: B x N x 3
        """
        L = ctx.L
        sh = grad_out.shape
        g_o = grad_out.cpu().numpy()
        # Stack
        g_o = g_o.reshape(-1, 3)
        Lg = L.dot(g_o).reshape(sh)
        return convert_as(torch.Tensor(Lg), grad_out), None

_cotLx = _CotLaplacianBatchLx.apply  # type: ignore

def cotangent(V, F):
    """
    Input:
      V: B x N x 3
      F: B x F x 3
    Outputs:
      C: B x F x 3 list of cotangents corresponding
        angles for triangles, columns correspond to edges 23,31,12
    B x F x 3 x 3
    """
    B, N, _ = V.shape
    indices_repeat = torch.stack([F, F, F], dim=2).to(device=V.device).expand(B, -1, -1, -1)

    #v1 is the list of first triangles B*F*3, v2 second and v3 third
    v1 = torch.gather(V, 1, indices_repeat[:, :, :, 0].long())
    v2 = torch.gather(V, 1, indices_repeat[:, :, :, 1].long())
    v3 = torch.gather(V, 1, indices_repeat[:, :, :, 2].long())

    l1 = torch.sqrt(((v2 - v3)**2).sum(2)) #distance of edge 2-3 for every face B*F
    l2 = torch.sqrt(((v3 - v1)**2).sum(2))
    l3 = torch.sqrt(((v1 - v2)**2).sum(2))

    # semiperimieters
    sp = (l1 + l2 + l3) * 0.5

    # Heron's formula for area #FIXME why the *2 ? Heron formula is without *2 It's the 0.5 than appears in the (0.5(cotalphaij + cotbetaij))
    inside_sqrt = sp * (sp-l1)*(sp-l2)*(sp-l3)
    inside_sqrt.masked_fill_(inside_sqrt<0, 0)
    A = 2*torch.sqrt(inside_sqrt)
    if not check_values(A):
        import pdb; pdb.set_trace()

    # Theoreme d Al Kashi : c2 = a2 + b2 - 2ab cos(angle(ab))
    cot23 = (l2**2 + l3**2 - l1**2)
    cot31 = (l1**2 + l3**2 - l2**2)
    cot12 = (l1**2 + l2**2 - l3**2)

    # 2 in batch #proof page 98 http://www.cs.toronto.edu/~jacobson/images/alec-jacobson-thesis-2013-compressed.pdf
    C = torch.stack([cot23, cot31, cot12], 2) / (torch.unsqueeze(A, 2)+1e-10) / 4
    C.masked_fill_(A.unsqueeze(2)==0, 0.0)
    return C


def mean_value_coordinates_3D(query, vertices, faces, verbose=False):
    """
    Tao Ju et.al. MVC for 3D triangle meshes
    params:
        query    (B,P,3)
        vertices (B,N,3)
        faces    (B,F,3)
    return:
        wj       (B,P,N)
    """
    B, F, _ = faces.shape
    _, P, _ = query.shape
    _, N, _ = vertices.shape
    # u_i = p_i - x (B,P,N,3)
    uj = vertices.unsqueeze(1) - query.unsqueeze(2)
    # \|u_i\| (B,P,N,1)
    dj = torch.norm(uj, dim=-1, p=2, keepdim=True)
    uj = normalize(uj, dim=-1)
    # gather triangle B,P,F,3,3
    ui = torch.gather(uj.unsqueeze(2).expand(-1,-1,F,-1,-1),
                                   3,
                                   faces.unsqueeze(1).unsqueeze(-1).expand(-1,P,-1,-1,3))
    # li = \|u_{i+1}-u_{i-1}\| (B,P,F,3)
    li = torch.norm(ui[:,:,:,[1, 2, 0],:] - ui[:, :, :,[2, 0, 1],:], dim=-1, p=2)
    eps = 2e-5
    li = torch.where(li>=2, li-(li.detach()-(2-eps)), li)
    li = torch.where(li<=-2, li-(li.detach()+(2-eps)), li)
    # asin(x) is inf at +/-1
    # θi =  2arcsin[li/2] (B,P,F,3)
    theta_i = 2*torch.asin(li/2)
    assert(check_values(theta_i))
    # B,P,F,1
    h = torch.sum(theta_i, dim=-1, keepdim=True)/2
    # wi← sin[θi]d{i−1}d{i+1}
    # (B,P,F,3) ci ← (2sin[h]sin[h−θi])/(sin[θ_{i+1}]sin[θ_{i−1}])−1
    ci = 2*torch.sin(h)*torch.sin(h-theta_i)/(torch.sin(theta_i[:,:,:,[1, 2, 0]])*torch.sin(theta_i[:,:,:,[2, 0, 1]]))-1

    # NOTE: because of floating point ci can be slightly larger than 1, causing problem with sqrt(1-ci^2)
    # NOTE: sqrt(x)' is nan for x=0, hence use eps
    eps = 1e-5
    ci = torch.where(ci>=1, ci-(ci.detach()-(1-eps)), ci)
    ci = torch.where(ci<=-1, ci-(ci.detach()+(1-eps)), ci)
    # si← sign[det[u1,u2,u3]]sqrt(1-ci^2)
    # (B,P,F)*(B,P,F,3)

    si = torch.sign(torch.det(ui)).unsqueeze(-1)*torch.sqrt(1-ci**2)  # sqrt gradient nan for 0
    assert(check_values(si))
    # (B,P,F,3)
    di = torch.gather(dj.unsqueeze(2).squeeze(-1).expand(-1,-1,F,-1), 3,
                      faces.unsqueeze(1).expand(-1,P,-1,-1))
    assert(check_values(di))
    # if si.requires_grad:
    #     vertices.register_hook(save_grad("mvc/dv"))
    #     li.register_hook(save_grad("mvc/dli"))
    #     theta_i.register_hook(save_grad("mvc/dtheta"))
    #     ci.register_hook(save_grad("mvc/dci"))
    #     si.register_hook(save_grad("mvc/dsi"))
    #     di.register_hook(save_grad("mvc/ddi"))

    # wi← (θi −c[i+1]θ[i−1] −c[i−1]θ[i+1])/(disin[θi+1]s[i−1])
    # B,P,F,3
    # CHECK is there a 2* in the denominator
    wi = (theta_i-ci[:,:,:,[1,2,0]]*theta_i[:,:,:,[2,0,1]]-ci[:,:,:,[2,0,1]]*theta_i[:,:,:,[1,2,0]])/(di*torch.sin(theta_i[:,:,:,[1,2,0]])*si[:,:,:,[2,0,1]])
    # if ∃i,|si| ≤ ε, set wi to 0. coplaner with T but outside
    # ignore coplaner outside triangle
    # alternative check
    # (B,F,3,3)
    # triangle_points = torch.gather(vertices.unsqueeze(1).expand(-1,F,-1,-1), 2, faces.unsqueeze(-1).expand(-1,-1,-1,3))
    # # (B,P,F,3), (B,1,F,3) -> (B,P,F,1)
    # determinant = dot_product(triangle_points[:,:,:,0].unsqueeze(1)-query.unsqueeze(2),
    #                           torch.cross(triangle_points[:,:,:,1]-triangle_points[:,:,:,0],
    #                                       triangle_points[:,:,:,2]-triangle_points[:,:,:,0], dim=-1).unsqueeze(1), dim=-1, keepdim=True).detach()
    # # (B,P,F,1)
    # sqrdist = determinant*determinant / (4 * sqrNorm(torch.cross(triangle_points[:,:,:,1]-triangle_points[:,:,:,0], triangle_points[:,:,:,2]-triangle_points[:,:,:,0], dim=-1), keepdim=True))

    wi = torch.where(torch.any(torch.abs(si) <= 1e-5, keepdim=True, dim=-1), torch.zeros_like(wi), wi)
    # wi = torch.where(sqrdist <= 1e-5, torch.zeros_like(wi), wi)

    # if π −h < ε, x lies on t, use 2D barycentric coordinates
    # inside triangle
    inside_triangle = (PI-h).squeeze(-1)<1e-4
    # set all F for this P to zero
    wi = torch.where(torch.any(inside_triangle, dim=-1, keepdim=True).unsqueeze(-1), torch.zeros_like(wi), wi)
    # CHECK is it di https://www.cse.wustl.edu/~taoju/research/meanvalue.pdf or li http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.516.1856&rep=rep1&type=pdf
    wi = torch.where(inside_triangle.unsqueeze(-1).expand(-1,-1,-1,wi.shape[-1]), torch.sin(theta_i)*di[:,:,:,[2,0,1]]*di[:,:,:,[1,2,0]], wi)

    # sum over all faces face -> vertex (B,P,F*3) -> (B,P,N)
    wj = scatter_add(wi.reshape(B,P,-1).contiguous(), faces.unsqueeze(1).expand(-1,P,-1,-1).reshape(B,P,-1), 2, out_size=(B,P,N))

    # close to vertex (B,P,N)
    close_to_point = dj.squeeze(-1) < 1e-8
    # set all F for this P to zero
    wj = torch.where(torch.any(close_to_point, dim=-1, keepdim=True), torch.zeros_like(wj), wj)
    wj = torch.where(close_to_point, torch.ones_like(wj), wj)

    # (B,P,1)
    sumWj = torch.sum(wj, dim=-1, keepdim=True)
    sumWj = torch.where(sumWj==0, torch.ones_like(sumWj), sumWj)

    wj_normalised = wj / sumWj
    # if wj.requires_grad:
    #     saved_variables["mvc/wi"] = wi
    #     wi.register_hook(save_grad("mvc/dwi"))
    #     wj.register_hook(save_grad("mvc/dwj"))
    if verbose:
        return wj_normalised, wi
    else:
        return wj_normalised


def mean_value_coordinates(points, polygon, verbose=False):
    """
    compute wachspress MVC of points wrt a polygon
    https://www.mn.uio.no/math/english/people/aca/michaelf/papers/barycentric.pdf
    Args:
        points: (B,D,N)
        polygon: (B,D,M)
    Returns:
        phi: (B,M,N)
    """
    D = polygon.shape[1]
    N = points.shape[-1]
    M = polygon.shape[-1]
    # (B,D,M,1) - (B,D,1,N) = (B,D,M,N)
    si = polygon.unsqueeze(3)-points.unsqueeze(2)
    # ei = normalize(s, dim=1)
    # B,M,N
    ri = torch.norm(si, p=2, dim=1)
    rip = torch.cat([ri[:,1:,:], ri[:,:1,:]], dim=1)
    sip = torch.cat([si[:,:,1:,:], si[:,:,:1,:]], dim=2)
    # sip = si[:,:,[i%M+1 for i in range(M)], :]
    # (B,M,N)
    # cos = dot_product(e, eplus, dim=1)
    # sin = cross_product_2D(e, eplus, dim=1)
    # (r_i*r_{i+1}-D_i)/A_i
    # D_i <e_i, e_{i+1}>
    # A_i det(e_i, e_{i+1})/2
    Ai = cross_product_2D(si, sip, dim=1)/2
    Aim = torch.cat([Ai[:,-1:,:], Ai[:,:-1,:]], dim=1)
    Di = dot_product(si, sip, dim=1)
    # Dim = torch.cat([Di[:,-1:,:], Di[:,:-1,:]], dim=1)
    # tanhalf = sin / (1+cos+1e-12)
    # w = torch.where(Ai!=0, (rip - Di/ri)/Ai, torch.zeros_like(Ai))+ torch.where(Aim!=0, (rim-Dim/ri)/Aim)
    tanhalf = torch.where(torch.abs(Ai) > 1e-5, (rip*ri-Di)/(Ai+1e-10), torch.zeros_like(Ai))
    tanhalf_minus  = torch.cat([tanhalf[:,-1:,:], tanhalf[:,:-1,:]], dim=1)
    w = (tanhalf_minus + tanhalf)/(ri+1e-10)

    # special case: on boundary
    # mask = ((torch.abs(sin) == 0) & (cos <= 0)| (cos == -1))
    mask = (torch.abs(Ai) <= 1e-5) & (Di < 0.0)
    mask_plus = torch.cat([mask[:,-1:,:], mask[:,:-1,:]], dim=1)
    mask_point = torch.any(mask, dim=1, keepdim=True)
    w = torch.where(mask_point, torch.zeros_like(w), w)
    pe = polygon - torch.cat([polygon[:,:,1:], polygon[:,:,:1]],dim=2)
    # (B,M,1)
    dL = torch.norm(pe, p=2, dim=1).unsqueeze(-1)
    w = torch.where(mask, 1-ri/(dL+1e-10), w)
    # w = torch.where(mask_plus, 1-ri/dL, w)
    w = torch.where(mask_plus, 1-torch.sum(w, dim=1, keepdim=True), w)
    # special case: close to polygon vertex
    # (B,N)
    mask = torch.lt(ri, 1e-8)
    # if an cage edge is very very short, can happen that this is true for both vertices
    mask_point = torch.any(mask, dim=1, keepdim=True)
    # set all weights of those points to zero
    w = torch.where(mask_point, torch.zeros_like(w), w)
    # set vertex weight of those points to 1
    w = torch.where(mask, torch.ones_like(w), w)

    # finally, normalize
    sumW = torch.sum(w, dim=1, keepdim=True)
    # sometimes sumw is 0?!
    if torch.nonzero(sumW==0).numel() > 0:
        sumW = torch.where(sumW==0, torch.ones_like(w), w)
    phi = w/sumW
    if verbose:
        return phi, w
    return phi


def compute_face_normals_and_areas(vertices: torch.Tensor, faces: torch.Tensor):
    """
    :params
        vertices   (B,N,3)
        faces      (B,F,3)
    :return
        face_normals         (B,F,3)
        face_areas   (B,F)
    """
    ndim = vertices.ndimension()
    if vertices.ndimension() == 2 and faces.ndimension() == 2:
        vertices.unsqueeze_(0)
        faces.unsqueeze_(0)

    B,N,D = vertices.shape
    F = faces.shape[1]
    # (B,F*3,3)
    face_vertices = torch.gather(vertices, 1, faces.view(B, -1, 1).expand(-1, -1, D)).view(B,F,3,D)
    face_normals = torch.cross(face_vertices[:,:,1,:] - face_vertices[:,:,0,:],
                               face_vertices[:,:,2,:] - face_vertices[:,:,1,:], dim=-1)
    face_areas = face_normals.clone()
    face_areas = torch.sqrt((face_areas ** 2).sum(dim=-1))
    face_areas /= 2
    face_normals = normalize(face_normals, dim=-1)
    if ndim == 2:
        vertices.squeeze_(0)
        faces.squeeze_(0)
        face_normals.squeeze_(0)
        face_areas.squeeze_(0)
    # assert (not np.any(face_areas.unsqueeze(-1) == 0)), 'has zero area face: %s' % mesh.filename
    return face_normals, face_areas


def edge_vertex_indices(F):
    """
    Given F matrix of a triangle mesh return unique edge vertices of a mesh Ex2 tensor
    params:
        F (F,L) tensor or numpy
    return:
        E (E,2) tensor or numpy
    """
    if isinstance(F, torch.Tensor):
        # F,L,2
        edges = torch.stack([F, F[:,[1, 2, 0]]], dim=-1)
        edges = torch.sort(edges, dim=-1)[0]
        # FxL,2
        edges = edges.reshape(-1, 2)
        # E,2
        edges = torch.unique(edges, dim=0)
    else:
        edges = np.stack([F, F[:,[1,2,0]]], axis=-1)
        edges = np.sort(edges, axis=-1)
        edges = edges.reshape([-1, 2])
        edges = np.unique(edges, axis=0)
    return edges


def get_edge_lengths(vertices, edge_points):
    """
    get edge squared length using edge_points from get_edge_points(mesh) or edge_vertex_indices(faces)
    :params
        vertices        (N,3)
        edge_points     (E,4)
    """
    N, D = vertices.shape
    E = edge_points.shape[0]
    # E,2,D (OK to do this kind of indexing on the first dimension)
    edge_vertices = vertices[edge_points[:,:2]]

    edges = (edge_vertices[:,0,:]-edge_vertices[:,1,:])
    edges_sqrlen = torch.sum(edges * edges, dim=-1)
    return edges_sqrlen


def get_normals(vertices: torch.Tensor, edge_points: torch.Tensor, side: int):
    """
    return the face normal of 4 edge points on the specified side
    :params
        vertices     (N,3)
        edge_poitns  (E,4)
        side          0-4
    :return
        normal       (E,3)
    """
    edge_a = vertices[edge_points[:, side // 2 + 2]] - vertices[edge_points[:, side // 2]]
    edge_b = vertices[edge_points[:, 1 - side // 2]] - vertices[edge_points[:, side // 2]]
    normals = torch.cross(edge_a, edge_b, dim=-1)
    normals = normalize(normals, dim=-1)
    if not check_values:
        import pdb; pdb.set_trace()
    return normals

def green_coordinates_2D(query, vertices, faces, edge_normals=None, verbose=False):
    pass

# def mean_value_coordinates_3D(query, vertices, faces, verbose=False):
def green_coordinates_3D(query, vertices, faces, face_normals=None, verbose=False):
    """
    Lipman et.al. sum_{i\in N}(phi_i*v_i)+sum_{j\in F}(psi_j*n_j)
    http://www.wisdom.weizmann.ac.il/~ylipman/GC/CageMesh_GreenCoords.cpp
    params:
        query    (B,P,D), D=3
        vertices (B,N,D), D=3
        faces    (B,F,3)
    return:
        phi_i    (B,P,N)
        psi_j    (B,P,F)
        exterior_flag (B,P)
    """
    B, F, _ = faces.shape
    _, P, D = query.shape
    _, N, D = vertices.shape
    # (B,F,D)
    n_t = face_normals
    if n_t is None:
        # compute face normal
        n_t, _ = compute_face_normals_and_areas(vertices, faces)

    vertices = vertices.detach()
    # (B,N,D) (B,F,3) -> (B,F,3,3) face points
    v_jl = torch.gather(vertices.unsqueeze(1).expand(-1,F,-1,-1), 2, faces.unsqueeze(-1).expand(-1,-1,-1,3))

    # v_jl = v_jl - x (B,P,F,3,3)
    v_jl = v_jl.view(B,1,F,3,3) - query.view(B,P,1,1,3)
    # (B,P,F,D).(B,1,F,D) -> (B,P,F,1)*(B,P,F,D) projection of v1_x on the normal
    p = dot_product(v_jl[:,:,:,0,:], n_t.unsqueeze(1).expand(-1,P,-1,-1), dim=-1, keepdim=True)*n_t.unsqueeze(1)

    # B,P,F,3,D -> B,P,F,3
    s_l = torch.sign(dot_product(torch.cross(v_jl-p.unsqueeze(-2), v_jl[:,:,:,[1,2,0],:]-p.unsqueeze(-2), dim=-1), n_t.view(B,1,F,1,D)))
    # import pdb; pdb.set_trace()
    # (B,P,F,3)
    I_l = _gcTriInt(p, v_jl, v_jl[:,:,:,[1,2,0],:], None)
    # (B,P,F)
    I = -torch.abs(torch.sum(s_l*I_l, dim=-1))
    GC_face = -I
    assert(check_values(GC_face))
    II_l = _gcTriInt(torch.zeros_like(p), v_jl[:,:,:,[1,2,0], :], v_jl, None)
    # (B,P,F,3,D)
    N_l = torch.cross(v_jl[:,:,:,[1,2,0],:], v_jl, dim=-1)
    N_l_norm = torch.norm(N_l, dim=-1, p=2)
    II_l.masked_fill_(N_l_norm<1e-7, 0)
    # normalize but ignore those with small norms
    N_l = torch.where((N_l_norm>1e-7).unsqueeze(-1), N_l/N_l_norm.unsqueeze(-1), N_l)
    # (B,P,F,D)
    omega = n_t.unsqueeze(1)*I.unsqueeze(-1)+torch.sum(N_l*II_l.unsqueeze(-1), dim=-2)
    eps = 1e-6
    # (B,P,F,3)
    phi_jl = dot_product(N_l[:,:,:,[1,2,0],:], omega.unsqueeze(-2), dim=-1)/(dot_product(N_l[:,:,:,[1,2,0],:], v_jl, dim=-1)+1e-10)
    # on the same plane don't contribute to phi
    phi_jl.masked_fill_((torch.norm(omega, p=2, dim=-1)<eps).unsqueeze(-1), 0)
    # sum per face weights to per vertex weights
    GC_vertex = scatter_add(phi_jl.reshape(B,P,-1).contiguous(), faces.unsqueeze(1).expand(-1,P,-1,-1).reshape(B,P,-1), 2, out_size=(B,P,N))
    assert(check_values(GC_vertex))

    # NOTE the point is inside the face, remember factor 2
    # insideFace = (torch.norm(omega,dim=-1)<1e-5)&torch.all(s_l>0,dim=-1)
    # phi_jl = torch.where(insideFace.unsqueeze(-1), phi_jl, torch.zeros_like(phi_jl))

    # normalize
    sumGC_V = torch.sum(GC_vertex, dim=2, keepdim=True)

    exterior_flag = sumGC_V<0.5

    GC_vertex = GC_vertex/(sumGC_V+1e-10)
    # GC_vertex.masked_fill_(sumGC_V.abs()<eps, 0.0)

    return GC_vertex, GC_face, exterior_flag


def _gcTriInt(p, v1, v2, x):
    """
    part of the gree coordinate 3D pseudo code
    params:
        p  (B,P,F,3)
        v1 (B,P,F,3,3)
        v2 (B,P,F,3,3)
        x  (B,P,F,3)
    return:
        (B,P,F,3)
    """
    eps = 1e-6
    angle_eps = 1e-3
    div_guard = 1e-12
    # (B,P,F,3,D)
    p_v1 = p.unsqueeze(-2)-v1
    v2_p = v2-p.unsqueeze(-2)
    v2_v1 = v2-v1
    # (B,P,F,3)
    p_v1_norm = torch.norm(p_v1, dim=-1, p=2)
    # (B,P,F,3)
    tempval = dot_product(v2_v1, p_v1, dim=-1)/(p_v1_norm*torch.norm(v2_v1, dim=-1, p=2)+div_guard)
    tempval.clamp_(-1.0,1.0)
    filter_mask = tempval.abs()>(1-eps)
    tempval.clamp_(-1.0+eps,1.0-eps)
    alpha = torch.acos(tempval)
    filter_mask = filter_mask | (torch.abs(alpha-np.pi)<angle_eps)|(torch.abs(alpha)<angle_eps)

    tempval = dot_product(-p_v1, v2_p, dim=-1)/(p_v1_norm*torch.norm(v2_p, dim=-1, p=2)+div_guard)
    tempval.clamp_(-1.0, 1.0)
    filter_mask = filter_mask|(torch.abs(tempval)>(1-eps))
    tempval.clamp_(-1.0+eps,1.0-eps)
    beta = torch.acos(tempval)
    assert(check_values(alpha))
    assert(check_values(beta))
    # (B,P,F,3)
    lambd = (p_v1_norm*torch.sin(alpha))**2
    # c (B,P,F,1)
    if x is not None:
        c = torch.sum((p-x)*(p-x), dim=-1,keepdim=True)
    else:
        c = torch.sum(p*p, dim=-1,keepdim=True)
    # theta in (pi-alpha, pi-alpha-beta)
    # (B,P,F,3)
    theta_1 = torch.clamp(np.pi - alpha, 0, np.pi)
    theta_2 = torch.clamp(np.pi - alpha - beta, -np.pi, np.pi)

    S_1, S_2 = torch.sin(theta_1), torch.sin(theta_2)
    C_1, C_2 = torch.cos(theta_1), torch.cos(theta_2)
    sqrt_c = torch.sqrt(c+div_guard)
    sqrt_lmbd = torch.sqrt(lambd+div_guard)
    theta_half = theta_1/2
    filter_mask = filter_mask | ((C_1-1).abs()<eps)
    sqcot_1 = torch.where((C_1-1).abs()<eps, torch.zeros_like(C_1), S_1*S_1/((1-C_1)**2+div_guard))
    # sqcot_1 = torch.where(theta_half.abs()<angle_eps, torch.zeros_like(theta_half), 1/(torch.tan(theta_half)**2+div_guard))
    theta_half = theta_2/2
    filter_mask = filter_mask | ((C_2-1).abs()<eps)
    sqcot_2 = torch.where((C_2-1).abs()<eps, torch.zeros_like(C_2), S_2*S_2/((1-C_2)**2+div_guard))
    # sqcot_2 = torch.where(theta_half.abs()<angle_eps, torch.zeros_like(theta_half), 1/(torch.tan(theta_half)**2+div_guard))
    # I=-0.5*Sign(sx)* ( 2*sqrtc*atan((sqrtc*cx) / (sqrt(a+c*sx*sx) ) )+
    #                 sqrta*log(((sqrta*(1-2*c*cx/(c*(1+cx)+a+sqrta*sqrt(a+c*sx*sx)))))*(2*sx*sx/pow((1-cx),2))))
    # assign a value to invalid entries, backward
    inLog = sqrt_lmbd*(1-2*c*C_1/( div_guard +c*(1+C_1)+lambd+sqrt_lmbd*torch.sqrt(lambd+c*S_1*S_1+div_guard) ) )*2*sqcot_1
    inLog.masked_fill_(filter_mask | (inLog<=0), 1.0)
    # inLog = torch.where(invalid_values|(lambd==0), torch.ones_like(theta_1), div_guard +sqrt_lmbd*(1-2*c*C_1/( div_guard +c*(1+C_1)+lambd+sqrt_lmbd*torch.sqrt(lambd+c*S_1*S_1)+div_guard ) )*2*cot_1)
    I_1 = -0.5*torch.sign(S_1)*(2*sqrt_c*torch.atan((sqrt_c*C_1) / (torch.sqrt(lambd+S_1*S_1*c+div_guard) ) )+sqrt_lmbd*torch.log(inLog))
    assert(check_values(I_1))
    inLog = sqrt_lmbd*(1-2*c*C_2/( div_guard +c*(1+C_2)+lambd+sqrt_lmbd*torch.sqrt(lambd+c*S_2*S_2+div_guard) ) )*2*sqcot_2
    inLog.masked_fill_(filter_mask | (inLog<=0), 1.0)
    I_2 = -0.5*torch.sign(S_2)*(2*sqrt_c*torch.atan((sqrt_c*C_2) / (torch.sqrt(lambd+S_2*S_2*c+div_guard) ) )+sqrt_lmbd*torch.log(inLog))
    assert(check_values(I_2))
    myInt = -1/(4*np.pi)*torch.abs(I_1-I_2-sqrt_c*beta)
    myInt.masked_fill_(filter_mask, 0.0)
    return myInt



def dihedral_angle(vertices: torch.Tensor, edge_points: torch.Tensor):
    """
    return the face-to-face angle of an edge specified by the 4 edge_points
    :params
        vertices     (N,3)
        edge_poitns  (E,4)
        side          0-4
    :return
        normal       (E,)
    """
    normals_a = get_normals(vertices, edge_points, 0)
    normals_b = get_normals(vertices, edge_points, 3)
    dot = dot_product(normals_a, normals_b, dim=-1).clamp(-1+1e-6, 1-1e-6)
    angles = PI - torch.acos(dot)
    return angles
