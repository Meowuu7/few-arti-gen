import torch
import time
import numpy as np

from scipy.optimize import linear_sum_assignment




def merge_meshes(vertices_list, faces_list):
  tot_vertices = []
  tot_faces = []
  nn_verts = 0
  for cur_vertices, cur_faces in zip(vertices_list, faces_list):
    tot_vertices.append(cur_vertices)
    new_cur_faces = []
    for cur_face_idx in cur_faces:
      new_cur_face_idx = [vert_idx + nn_verts for vert_idx in cur_face_idx]
      new_cur_faces.append(new_cur_face_idx)
    nn_verts += cur_vertices.shape[0]
    tot_faces += new_cur_faces # get total-faces
  tot_vertices = np.concatenate(tot_vertices, axis=0)
  return tot_vertices, tot_faces


def get_corrs(pc1, pc2):
  dists = np.sum((np.reshape(pc1, (pc1.shape[0], 1, 3)) - np.reshape(pc2, (1, pc2.shape[0], 3)) ) ** 2, dim=-1)
  row_ind, col_ind = linear_sum_assignment(dists) ### 
  matched_pc1 = pc2[col_ind]
  src_p1 = pc1[row_ind]
  return src_p1, matched_pc1
  

def hungarian_matching(pred_x, gt_x, curnmasks, include_background=True):
    """ pred_x, gt_x: B x nmask x nsmp
        curnmasks: B
        return matching_idx: B x nmask x 2 """
    batch_size = gt_x.shape[0]
    nmask = gt_x.shape[1]

    matching_score = np.matmul(gt_x,np.transpose(pred_x, axes=[0,2,1])) # B x nmask x nmask
    # matching_score = torch.matmul(gt_x, pred_x.transpose(1, 2))
    matching_score = 1-np.divide(matching_score,
                                 np.expand_dims(np.sum(pred_x, 2), 1) + np.sum(gt_x, 2, keepdims=True) - matching_score+1e-8)
    matching_idx = np.zeros((batch_size, nmask, 2)).astype('int32')
    curnmasks = curnmasks.astype('int32')
    for i, curnmask in enumerate(curnmasks):
        # print(curnmask.shape)
        curnmask = int(curnmask)
        # curnmask = min(curnmask, pred_x.shape[1])
        assert pred_x.shape[1] >= curnmask, "Should predict no less than n_max_instance segments!"
        # Truncate invalid masks in GT predictions
        row_ind, col_ind = linear_sum_assignment(matching_score[i,:curnmask,:])
        # row_ind, col_ind = linear_sum_assignment(matching_score[i,:,:])
        matching_idx[i,:curnmask,0] = row_ind[:curnmask]
        matching_idx[i,:curnmask,1] = col_ind[:curnmask]
    return torch.from_numpy(matching_idx).long()


def save_obj_file(vertices, face_list, obj_fn, add_one=False):
  with open(obj_fn, "w") as wf:
    for i_v in range(vertices.shape[0]):
      cur_v_values = vertices[i_v]
      wf.write("v")
      for i_v_v in range(cur_v_values.shape[0]):
        wf.write(f" {float(cur_v_values[i_v_v].item())}")
      wf.write("\n")
    for i_f in range(len(face_list)):
      cur_face_idxes = face_list[i_f]
      wf.write("f")
      for cur_f_idx in range(len(cur_face_idxes)):
        wf.write(f" {cur_face_idxes[cur_f_idx] if not add_one else cur_face_idxes[cur_f_idx] + 1}")
      wf.write("\n")
    wf.close()



    
def normalize_vertices_scale_torch_batched(vertices, rt_stats=False):
  vert_min, _ = torch.min(vertices, dim=1)
  vert_max, _ = torch.max(vertices, dim=1)
  extents = vert_max - vert_min
  
  scale = torch.sqrt(torch.sum(extents ** 2, dim=-1))
  # normed_vertices = 
  # if rt_stats:
  #   return 
  return vertices / scale.unsqueeze(-1).unsqueeze(-1)
  # return scale

def normalize_vertices_scale(vertices):
  """Scale the vertices so that the long diagonal of the bounding box is one."""
  vert_min = vertices.min(axis=0)
  vert_max = vertices.max(axis=0)
  extents = vert_max - vert_min
  scale = np.sqrt(np.sum(extents**2)) # normalize the diagonal line to 1.
  return vertices / scale

def get_vertices_center(vertices):
  vert_min = vertices.min(axis=0)
  vert_max = vertices.max(axis=0)
  vert_center = 0.5 * (vert_min + vert_max)
  return vert_center


def get_vertices_scale(vertices):
  vert_min = vertices.min(axis=0)
  vert_max = vertices.max(axis=0)
  extents = vert_max - vert_min
  scale = np.sqrt(np.sum(extents**2))
  return scale

def get_vertices_scale_torch_no_batch(vertices):
  verts_min = vertices.min(dim=0)[0]
  verts_max = vertices.max(dim=0)[0]
  extents = verts_max - verts_min
  scale = torch.sqrt(torch.sum(extents**2))
  return scale

def get_vertices_scale_torch_batch(vertices): ### bsz x n_pts x 3
  verts_min, _ = torch.min(vertices, dim=1) ### bsz  x 3
  verts_max, _ = torch.max(vertices, dim=1) ### bsz x 3
  verts_extents = verts_max - verts_min
  verts_scale = torch.sqrt(torch.sum(verts_extents ** 2, dim=-1)) ### bsz
  return verts_scale
  

def get_vertices_extends(vertices):
  vert_min = vertices.min(axis=0)
  vert_max = vertices.max(axis=0)
  extents = vert_max - vert_min
  return extents

def get_vertices_center_torch(vertices):
  vert_min = torch.min(vertices, dim=1)[0]
  vert_max = torch.max(vertices, dim=1)[0]
  # vert_min = vertices.min(axis=0)
  # vert_max = vertices.max(axis=0)
  vert_center = 0.5 * (vert_min + vert_max)
  return vert_center


def get_vertices_scale_torch(vertices):
  vert_min = torch.min(vertices, dim=1)[0]
  vert_max = torch.max(vertices, dim=1)[0]
  
  # vert_min = vertices.min(axis=0)
  # vert_max = vertices.max(axis=0)
  extents = vert_max - vert_min
  # scale = np.sqrt(np.sum(extents**2))
  scale = torch.sqrt(torch.sum(extents ** 2))
  return scale

def normalie_pc_bbox_batched(pc, rt_stats=False):
  pc_min = torch.min(pc, dim=1, keepdim=True)[0]
  pc_max = torch.max(pc, dim=1, keepdim=True)[0]
  pc_center = 0.5 * (pc_min + pc_max)
  
  pc = pc - pc_center
  extents = pc_max - pc_min
  scale = torch.sqrt(torch.sum(extents ** 2, dim=-1, keepdim=True))
  
  pc = pc / torch.clamp(scale, min=1e-6)
  if rt_stats:
    return pc, pc_center, scale
  else:
    return pc

def get_pc_scale_batched(pc):
  pc_min = torch.min(pc, dim=1, keepdim=True)[0]
  pc_max = torch.max(pc, dim=1, keepdim=True)[0]
  extents = pc_max - pc_min
  scale = torch.sqrt(torch.sum(extents ** 2, dim=-1, keepdim=True))
  return scale
  

def reparameterize_gaussian(mean, logvar):
    std = torch.exp(0.5 * logvar)  ### std and eps --> 
    eps = torch.randn(std.size()).to(mean)
    return mean + std * eps
  

def gaussian_entropy(logvar): ### gaussian entropy... ###
    const = 0.5 * float(logvar.size(1)) * (1. + np.log(np.pi * 2)) #### const  #### logvar.size(1) ---> use logvar's feature dimension as one term of gaussian entropy
    #### gaussian entropy... --> maximize entropy --> maximize the entropy term 
    ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const #### const float and const
    return ent


def standard_normal_logprob(z):
    # dim = z.size(-1)
    dim = z.size(1)
    log_z = -0.5 * dim * np.log(2 * np.pi)
    return log_z - z.pow( 2) / 2


def kld_loss(mu, log_var):
  kl_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1)
  return kl_loss 

def truncated_normal_(tensor, mean=0, std=1, trunc_std=2):
    """
    Taken from https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < trunc_std) & (tmp > -trunc_std)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
      try:
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
          torch.nn.init.constant_(m.bias.data, 0.0)
      except:
        pass
    elif classname.find('Linear') != -1:
      try:
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
          torch.nn.init.constant_(m.bias.data, 0.0)
      except:
        pass
        
def weights_init1(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)
        
def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum

def read_off_file_ours(off_fn, sub_one=False):
  vertices = []
  faces = []
  ii = 0
  with open(off_fn, "r") as rf:
    for line in rf:
      items = line.strip().split(" ")
      if ii <= 1:
        ii += 1
        continue
      if len(items) == 3:
        cur_verts = items
        cur_verts = [float(vv) for vv in cur_verts]
        vertices.append(cur_verts)
      elif len(items) == 4:
        cur_faces = items[1:] # faces
        cur_face_idxes = []
        for cur_f in cur_faces:
          cur_face_idxes.append(int(cur_f))
        faces.append(cur_face_idxes)
      
    rf.close()
  vertices = np.array(vertices, dtype=np.float)
  faces = np.array(faces, dtype=np.int32)
  return vertices, faces

def read_obj_file_ours(obj_fn, sub_one=False):
  vertices = []
  faces = []
  with open(obj_fn, "r") as rf:
    for line in rf:
      items = line.strip().split(" ")
      if items[0] == 'v':
        cur_verts = items[1:]
        cur_verts = [float(vv) for vv in cur_verts]
        vertices.append(cur_verts)
      elif items[0] == 'f':
        cur_faces = items[1:] # faces
        cur_face_idxes = []
        for cur_f in cur_faces:
          if len(cur_f) <= 0:
            continue
          try:
            cur_f_idx = int(cur_f.split("/")[0])
          except:
            cur_f_idx = int(cur_f.split("//")[0])
          cur_face_idxes.append(cur_f_idx if not sub_one else cur_f_idx - 1)
        faces.append(cur_face_idxes)
    rf.close()
  vertices = np.array(vertices, dtype=np.float)
  return vertices, faces

def load_txt(file_name):
    with open(file_name) as f:
        data = f.readlines()
    n = int(data[0].split(" ")[0])
    m = int(data[0].split(" ")[1])
    weights = np.zeros((n, m))
    for i in range(n):
        w = data[i + 1].split(" ")
        for j in range(m):
            weights[i][j] = float(w[j])
    return weights

def get_avg_minn_shp_pts_to_cvx_pts(shp_pts, cvx_to_pts):
    ### shp_pts: 
    ### 
    tot_cvx_pts = []
    for cvx in cvx_to_pts:
        tot_cvx_pts.append(cvx_to_pts[cvx].unsqueeze(0))
    tot_cvx_pts = torch.cat(tot_cvx_pts, dim=0)
    dist_shp_pts_cvx_pts = torch.sum((shp_pts.unsqueeze(1).unsqueeze(1) - tot_cvx_pts.unsqueeze(0)) ** 2, dim=-1)
    dist_shp_pts_cvx_pts = torch.sqrt(dist_shp_pts_cvx_pts)
    dist_shp_pts_cvx_pts = torch.min(dist_shp_pts_cvx_pts, dim=-1)[0] ### min distance to each convex ###
    dist_shp_pts_cvx_pts = torch.min(dist_shp_pts_cvx_pts, dim=-1)[0] #### n_shp_pts ### minn distance to lla cvx
    avg_dist_shp_pts_cvx_pts = torch.mean(dist_shp_pts_cvx_pts).item()
    return float(avg_dist_shp_pts_cvx_pts)
  
def get_avg_minn_shp_pts_to_cvx_pts_batch(shp_pts, cvx_to_pts):
  ### shp_pts: bsz x n_pts x 1 x 1 x 3 
  ### cvx_to_pts: bsz x n_cvx x n_cvx_pts x 3 ---> cvx_to_pts
  ### dist: bsz x n_pts x n_cvx x n_cvx_pts
  dist_shp_pts_cvx_pts = torch.sum((shp_pts.unsqueeze(2).unsqueeze(2) - cvx_to_pts.unsqueeze(1)) ** 2, dim=-1) ###
  dist_shp_pts_cvx_pts = torch.sqrt(dist_shp_pts_cvx_pts)
  dist_shp_pts_cvx_pts = torch.min(dist_shp_pts_cvx_pts, dim=-1)[0] ### min distance to each convex ###
  dist_shp_pts_cvx_pts = torch.min(dist_shp_pts_cvx_pts, dim=-1)[0] #### n_shp_pts ### minn distance to lla cvx
  avg_dist_shp_pts_cvx_pts = torch.mean(dist_shp_pts_cvx_pts, dim=-1)
  return avg_dist_shp_pts_cvx_pts

