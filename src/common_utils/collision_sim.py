
import torch
import time
import numpy as np
import utils
from common_utils import data_utils_torch as data_utils
from common_utils.part_transform import revoluteTransform
import random



def get_faces_from_verts(verts, faces, sel_verts_idxes):
  
  ### n_sel_faces x 3 x 3 --> 
  sel_faces = []
  if not isinstance(sel_verts_idxes, list):
    sel_verts_idxes = sel_verts_idxes.tolist() 
  sel_verts_idxes_dict = {sel_idx : 1 for sel_idx in sel_verts_idxes}
  print(f"len(sel_verts_idxes_dict): {len(sel_verts_idxes_dict)}")
  for i_f in range(faces.size(0)):
    cur_f = faces[i_f]
    va, vb, vc = cur_f.tolist()
    # va = va - 1
    # vb = vb - 1
    # vc = vc - 1
    if va  in sel_verts_idxes_dict or vb in sel_verts_idxes_dict or vc in sel_verts_idxes_dict:
      sel_faces.append(faces[i_f].tolist()) ### sel_faces items...
  # print(f"number of sel_faces: {len(sel_faces)}")
  sel_faces = torch.tensor(sel_faces, dtype=torch.long).cuda() ### n_sel_faces x 3 ## sel_
  print(f"verts: {verts.size()}, max_sel_faces: {torch.max(sel_faces)}, min_sel_faces: {torch.min(sel_faces)}")
  sel_faces_vals = data_utils.batched_index_select(values=verts, indices=sel_faces, dim=0) ### self_faces_vals: n_sel_faces x 3 x 3 ### sel_fces_vals...
  return sel_faces, sel_faces_vals


def sel_faces_values_from_sel_faces(verts, sel_faces):
  sel_faces_vals = data_utils.batched_index_select(values=verts, indices=sel_faces, dim=0) #
  return sel_faces_vals

def get_sub_verts_faces_from_pts(verts, faces, pts, rt_sel_faces=False):
  ### return tyep: sel_verts: n_pts x 3; sel_faces: faces selected from sel_verts ###
  ## verts: n_verts x 3; pts: n_pts x 3
  print(f"pts: {pts.size()}, verts: {verts.size()}, maxx_pts: {torch.max(pts, dim=0)}, minn_pts: {torch.min(pts, dim=0)},  maxx_verts: {torch.max(verts, dim=0)}, minn_verts: {torch.min(verts, dim=0)}")
  dis_pts_verts = torch.sum((pts.unsqueeze(1) - verts.unsqueeze(0)) ** 2, dim=-1) ### n_pts x n_verts ###
  minn_dist_pts_verts, minn_dist_pts_verts_idx = torch.min(dis_pts_verts, dim=-1) ###
  sel_verts = verts[minn_dist_pts_verts_idx] ### should be close to pts in the euclidean distance
  print(f"verts: {verts.size()}, faces: {faces.size()}, minn_dist_pts_verts_idx: {minn_dist_pts_verts_idx.size()}, sel_verts: {sel_verts.size()}")
  # print(f"verts: {verts.size()}, minn_faces: {torch.max(faces)}, faces: {torch.min(faces)}")
  sel_faces, sel_faces_vals = get_faces_from_verts(verts, faces, minn_dist_pts_verts_idx) ### sel_faces_vals: n_sel_faces x 3 x 3
  if rt_sel_faces:
    return sel_verts, sel_faces, sel_faces_vals
  else:
    return sel_verts, sel_faces_vals

##### distance of each sel_vert in mesh_2 to each face in sel_faces in mesh_1 ##### ---> 

def get_faces_normals(faces_vals):
  ### faces_vals: n_faces x 3 x 3
  vas = faces_vals[:, 0, :]
  vbs = faces_vals[:, 1, :]
  vcs = faces_vals[:, 2, :]
  vabs = vbs - vas
  vacs = vcs - vas ### n_faces x 3
  vns = torch.cross(vabs, vacs) ### n_faces x 3 ---> cross product between two vectors
  vns = vns / torch.clamp(torch.norm(vns, dim=-1, p=2, keepdim=True), min=1e-6) ### vns: n_faces x 3
  return vns

### 

def get_distance_pts_faces(pts, faces_vals, faces_vns):
  ### faces_vals: n_faces x 3 x 3  ## 
  ### faces_vns: n_faces x 3
  ### ax + by + cz = d  ### one pts and another pts --> faces_vals --> faces_ds
  faces_ds = torch.sum(faces_vals[:, 0, :] * faces_vns, dim=-1) ## n_faces x 3 xxx n_faces x 3 --> n_faces
  ### distance from one point to another point ###
  ### pts: n_pts x 3; faces_vns: n_faces x 3
  faces_pts_ds = torch.sum(pts.unsqueeze(1) * faces_vns.unsqueeze(0), dim=-1) ### n_pts x n_faces ### ### negative distances --> 
  delta_faces_pts_ds = faces_pts_ds - faces_ds.unsqueeze(0) ### n_pts x n_faces ### ### as an distance vector is pts can be projected to the faces ### pts_ds; 
  ### 1 x n_faces x 3 xxxxxx n_pts x n_faces x 1 --> n_pts x n_faces x 3
  projected_pts = pts.unsqueeze(1) - faces_vns.unsqueeze(0) * delta_faces_pts_ds.unsqueeze(-1)
  
  ### n_faces x 3 x 3
  ### vab vac ### 
  va, vb, vc = faces_vals[:, 0, :], faces_vals[:, 1, :], faces_vals[:, 2, :] ## n_faces x 3
  
  projected_pts = projected_pts - va.unsqueeze(0)
  
  vab, vac = vb - va, vc - va
  vab_norm, vac_norm = vab / torch.clamp(torch.norm(vab, dim=-1, p=2, keepdim=True), min=1e-7), vac / torch.clamp(torch.norm(vac, dim=-1, p=2, keepdim=True), min=1e-7)
  
  coeff_vab = torch.sum(vab_norm.unsqueeze(0) * projected_pts, dim=-1) / torch.clamp(torch.norm(vab, dim=-1, p=2, keepdim=False), min=1e-7)
  coeff_vac = torch.sum(vac_norm.unsqueeze(0) * projected_pts, dim=-1) / torch.clamp(torch.norm(vac, dim=-1, p=2, keepdim=False), min=1e-7)
  
  # coeff_vab = torch.sum(vab.unsqueeze(0) * projected_pts, dim=-1) ### n_pts x n_faces
  # coeff_vac = torch.sum(vac.unsqueeze(0) * projected_pts, dim=-1) ### n_pts x n_faces
  
  pts_in_faces = (((coeff_vab >= 0.).float() + (coeff_vac >= 0.).float() + (coeff_vab + coeff_vac <= 0.5).float()) > 2.5).float() ### n_pts x n_faces ### pts_in_faces --> 
  ### pts_in_faces and delta_faces_pts_ds 
  ### pts_in_faces: the projected pts in faces...
  return delta_faces_pts_ds, pts_in_faces ### delta_faces_pts_ds: n_pts x n_faces; pts_in_faces: n_pts x n_faces ###
   
  
### mesh_list, joints and joint types 

def transform_vertices_keypts(keypts, verts, faces, sel_faces, joint_state, joint_dir, joint_center, joint_type):
  ### verts: N x 3 
  if joint_type == 0: ### the a revolute joint
    pts, m = revoluteTransform(keypts.detach().cpu().numpy(), joint_center.detach().cpu().numpy(), joint_dir.detach().cpu().numpy(), joint_state)
    m = torch.from_numpy(m).float().cuda() ### 4 x 4
    verts_expanded = torch.cat([verts, torch.ones((verts.size(0), 1), dtype=torch.float32).cuda()], dim=-1)
    verts_expanded = torch.matmul(verts_expanded, m)
    verts_expanded = verts_expanded[:, :3] ### rotated verts
    sel_faces_vals = sel_faces_values_from_sel_faces(verts=verts_expanded, sel_faces=sel_faces)
    sel_faces_normals = get_faces_normals(sel_faces_vals)
  elif joint_type == 1: ### prismatic joint
    delta_dis =joint_dir * joint_state
    verts_expanded = verts + delta_dis.unsqueeze(0)
    sel_faces_vals = sel_faces_values_from_sel_faces(verts=verts_expanded, sel_faces=sel_faces)
    sel_faces_normals = get_faces_normals(sel_faces_vals)
  else:
    raise ValueError(f"Unrecognized joint_type: {joint_type}!!")
  return verts_expanded, sel_faces_vals, sel_faces_normals




def collision_loss_sim_sequence(verts1, keypts_1, verts2, sel_faces_vals2, sel_faces_vns2, joints, n_sim_steps=100, back_sim=False, use_delta=False):
  # joint_dir, joint_pvp, joint_angle = joints
  # joint_dir = joint_dir.cuda()
  # joint_pvp = joint_pvp.cuda()
  
  # print(f"verts1: {verts1.size()}, keypts_1: {keypts_1.size()}, verts2: {verts2.size()}, sel_faces_vals2: {sel_faces_vals2.size()}, sel_faces_vns2: {sel_faces_vns2.size()}")
  
  ### should save the sequence of transformed shapes ... ###
  
  # verts1, faces1 = mesh_1
  # verts2, faces2 = mesh_2
  
  ### verts2, keypts_2.detach() ###
  ### verts2, keypts_2 ###
  verts2 = verts2.detach()
  # keypts_2 = keypts_2.detach()

  sel_faces_vns2 = sel_faces_vns2.detach()
  
  
  ### pts_in_faces & delta_ds in two adjacent time stamps ###
  ### collision response for the loss term? ### 
  ### penetration depth * face_ns for all collided pts
  # keypts_sequence = [keypts_1.clone()]
  keypts_sequence = []
  delta_faces_pts_ds_sequence = []
  pts_in_faces_sequence = [] 
  ### pivot point prediction, joint axis direction prediction ### 
  # ### joint axis direction prediction ### #### get part joints...
  
  joint_dir = joints["axis"]["dir"]
  joint_pvp = joints["axis"]["center"]
  joint_a = joints["axis"]["a"]
  joint_b = joints["axis"]["b"]
  
  
  delta_joint_angle = (float(joint_b) - float(joint_a)) / float(n_sim_steps - 1) ### delta_joint_angles ###
  
  tot_collision_loss = 0.
  
  non_collided_pts = torch.ones((keypts_1.size(0), ), dtype=torch.float32, requires_grad=False) # .cuda()
  
  
  mesh_pts_sequence = []
  
  def_pcs_sequence = []
  
  
  
  
  for i in range(0, n_sim_steps): ### joint_angle ###
    if not back_sim:
      cur_joint_angle = joint_a +  i * delta_joint_angle ### delta_joint_angle ###
    else:
      cur_joint_angle = joint_b - i * delta_joint_angle  
      
    if use_delta:
      delta_joint_angle = (joint_b - joint_a) / 100.0
      
      ''' Prev. arti. state '''  
      cur_st_joint_angle = np.random.uniform(low=joint_a + delta_joint_angle, high=joint_b, size=(1,)).item() #### lower and upper limits of simulation angles
      
      ### revoluteTransform ### joint_pvp
      prev_pts, prev_m = revoluteTransform(keypts_1.detach().cpu().numpy(), joint_pvp.detach().cpu().numpy(), joint_dir.detach().cpu().numpy(), cur_st_joint_angle) ### st_joint_angle
      prev_m = torch.from_numpy(prev_m).float().cuda() ### 4 x 4
      
      kpts_expanded = torch.cat([keypts_1, torch.ones((keypts_1.size(0), 1), dtype=torch.float32).cuda()], dim=-1) #### kpts_expanded
      prev_pts = torch.matmul(kpts_expanded, prev_m)
      # pts = torch.matmul(keypts_1, m[:3]) ### n_keypts x 3 xxx 3 x 4 --> 
      prev_pts = prev_pts[:, :3] ### pts: n_keypts x 3
      
      #### distance_pts_faces, pts_in_faces, delta_faces_pts_ds ####
      prev_delta_faces_pts_ds, prev_pts_in_faces = get_distance_pts_faces(prev_pts, sel_faces_vals2, sel_faces_vns2)
      ''' Prev. arti. state ''' 
      
      ''' Current state ''' 
      cur_ed_joint_angle = cur_st_joint_angle - delta_joint_angle
      ### revoluteTransform ### joint_pvp
      pts, m = revoluteTransform(keypts_1.detach().cpu().numpy(), joint_pvp.detach().cpu().numpy(), joint_dir.detach().cpu().numpy(), cur_st_joint_angle) ### st_joint_angle
      m = torch.from_numpy(m).float().cuda() ### 4 x 4
      
      kpts_expanded = torch.cat([keypts_1, torch.ones((keypts_1.size(0), 1), dtype=torch.float32).cuda()], dim=-1) #### kpts_expanded
      pts = torch.matmul(kpts_expanded, m)
      # pts = torch.matmul(keypts_1, m[:3]) ### n_keypts x 3 xxx 3 x 4 --> 
      pts = pts[:, :3] ### pts: n_keypts x 3
      
      #### distance_pts_faces, pts_in_faces, delta_faces_pts_ds ####
      delta_faces_pts_ds, pts_in_faces = get_distance_pts_faces(pts, sel_faces_vals2, sel_faces_vns2)
      ''' Current state ''' 
      
      sgn_delta_faces_ds = torch.sign(delta_faces_pts_ds) ### sign of faces_ds
      sgn_prev_delta_faces_ds = torch.sign(prev_delta_faces_pts_ds) ### sign of prev_faces_ds
      ### different signs ###
      collision_pts_faces = (sgn_delta_faces_ds != sgn_prev_delta_faces_ds).float() ### n_pts x n_faces 
      
      cur_delta_faces_pts_ds = delta_faces_pts_ds
      
      collision_dists = collision_pts_faces * cur_delta_faces_pts_ds ### n_pts x n_faces
      ### collision_dists ### n_pts x n_faces
       ### i think the sim step 
       ### whether tow meshes collide with each other: in the target mesh, 
      ### collision_pulse, collision_dists
      # collision_pulse = (collision_dists * pts_in_faces * non_collided_pts.unsqueeze(-1)).unsqueeze(-1) * sel_faces_vns2.unsqueeze(0) ### n_pts x n_faces x 3 --> pulse
      collision_pulse = (collision_dists * pts_in_faces).unsqueeze(-1) * sel_faces_vns2.unsqueeze(0) ### n_pts x n_faces x 3 --> pulse
      
      # collided_indicator = ((pts_in_faces * non_collided_pts.unsqueeze(-1) * collision_pts_faces).sum(-1) > 0.1).float()
      
      ### loss version v2: for pts directly ###
      collision_loss = torch.sum(collision_pulse * pts.unsqueeze(1), dim=-1) ### n_pts x n_faces ### 
      ### loss version v2: for pts directly ###
      
      # non_collided_indicator = 1.0  - collided_indicator # - (collision_pulse.sum(-1).sum(-1) > 1e-6).float()
      # # print(f"collision_pulse: {collision_pulse.size()}, collided_indicator: {collided_indicator.size()}, non_collided_pts: {non_collided_pts.size()}")
      # # non_collided_pts[collided_indicator] = non_collided_pts[collided_indicator] * 0.
      # non_collided_pts = non_collided_pts * non_collided_indicator
      # # print(f"collision_loss: {collision_loss.sum().mean().item()}, collided_indicator: {collided_indicator.sum(-1).item()}, non_collided_pts: {non_collided_pts.sum(-1).item()}")
      
      # ### loss version v2: for pts directly ###
      # collision_loss = torch.sum(collision_pulse * pts.unsqueeze(1), dim=-1) ### n_pts x n_faces ### 
      # ### loss version v2: for pts directly ###
      collision_loss = torch.sum(collision_loss, dim=-1).sum() ###  ## for all faces ###
      tot_collision_loss += collision_loss
    else:
      ### revoluteTransform ### joint_pvp
      pts, m = revoluteTransform(keypts_1.detach().cpu().numpy(), joint_pvp.detach().cpu().numpy(), joint_dir.detach().cpu().numpy(), cur_joint_angle)
      m = torch.from_numpy(m).float().cuda() ### 4 x 4
      
      kpts_expanded = torch.cat([keypts_1, torch.ones((keypts_1.size(0), 1), dtype=torch.float32).cuda()
                                 ], dim=-1) #### kpts_expanded
      pts = torch.matmul(kpts_expanded, m)
      # pts = torch.matmul(keypts_1, m[:3]) ### n_keypts x 3 xxx 3 x 4 --> 
      pts = pts[:, :3] ### pts: n_keypts x 3
      
      #### distance_pts_faces, pts_in_faces, delta_faces_pts_ds ####
      delta_faces_pts_ds, pts_in_faces = get_distance_pts_faces(pts.detach().cpu(), sel_faces_vals2.detach().cpu(), sel_faces_vns2.detach().cpu())
      
      
      mesh_pts_expanded = torch.cat([verts1, torch.ones((verts1.size(0), 1), dtype=torch.float32).cuda()], dim=-1) #### kpts_expanded
      mesh_pts_expanded = torch.matmul(mesh_pts_expanded, m)
      # pts = torch.matmul(keypts_1, m[:3]) ### n_keypts x 3 xxx 3 x 4 --> 
      mesh_pts_expanded = mesh_pts_expanded[:, :3] ### pts: n_keypts x 3
      mesh_pts_sequence.append(mesh_pts_expanded.clone())
    
    
      # if len(delta_faces_pts_ds_sequence) > 0 and i == 0 or i == n_sim_steps - 1:
      if len(delta_faces_pts_ds_sequence) > 0:
        prev_delta_faces_pts_ds = delta_faces_pts_ds_sequence[-1]
        # prev_pts_in_faces = pts_in_faces_sequence[-1] ### not important
        prev_pts = keypts_sequence[-1]
        
        sgn_delta_faces_ds = torch.sign(delta_faces_pts_ds) ### sign of faces_ds
        sgn_prev_delta_faces_ds = torch.sign(prev_delta_faces_pts_ds) ### sign of prev_faces_ds
        ### different signs ###
        collision_pts_faces = (sgn_delta_faces_ds != sgn_prev_delta_faces_ds).float() ### n_pts x n_faces 

        cur_delta_faces_pts_ds = delta_faces_pts_ds

        collision_dists = collision_pts_faces * cur_delta_faces_pts_ds ### n_pts x n_faces
        ### collision_dists ### n_pts x n_faces
        ### i think the sim step 
        ### whether tow meshes collide with each other: in the target mesh, 
        ### collision_pulse, collision_dists
        collision_pulse = 1.0 * (collision_dists * pts_in_faces * non_collided_pts.unsqueeze(-1)).unsqueeze(-1) * sel_faces_vns2.unsqueeze(0).detach().cpu() ### n_pts x n_faces x 3 --> pulse
        
        collided_indicator = ((pts_in_faces * non_collided_pts.unsqueeze(-1) * collision_pts_faces).sum(-1) > 0.1).float()

        ### loss version v2: for pts directly ### ### calculate collision_loss from collision_pulse and pts ###
        collision_loss = torch.sum(collision_pulse.cuda() * pts.unsqueeze(1), dim=-1) ### n_pts x n_faces ### 
        ### loss version v2: for pts directly ###
        
        
        non_collided_indicator = 1.0  - collided_indicator # - (collision_pulse.sum(-1).sum(-1) > 1e-6).float()
        # print(f"collision_pulse: {collision_pulse.size()}, collided_indicator: {collided_indicator.size()}, non_collided_pts: {non_collided_pts.size()}")
        # non_collided_pts[collided_indicator] = non_collided_pts[collided_indicator] * 0.
        non_collided_pts = non_collided_pts * non_collided_indicator
        # print(f"collision_loss: {collision_loss.sum().mean().item()}, collided_indicator: {collided_indicator.sum(-1).item()}, non_collided_pts: {non_collided_pts.sum(-1).item()}")
        
        # ### loss version v2: for pts directly ###
        # collision_loss = torch.sum(collision_pulse * pts.unsqueeze(1), dim=-1) ### n_pts x n_faces ### 
        # ### loss version v2: for pts directly ###
        collision_loss = torch.sum(collision_loss, dim=-1).sum() ###  ## for all faces ###
        tot_collision_loss += collision_loss
        # if early_stop and collision_loss.item() > 0.0001:
        #   break


    delta_faces_pts_ds_sequence.append(delta_faces_pts_ds.clone())
    pts_in_faces_sequence.append(pts_in_faces.clone())
    keypts_sequence.append(pts.clone())
    
  # tot_collision_loss /= n_sim_steps
    ### delta_faces_
    # if 
    
  # print(f"tot_collision_loss: {tot_collision_loss}")
    
  ### can even test for one part at first
  return tot_collision_loss, keypts_sequence, mesh_pts_sequence ### collision_loss for all sim steps ###
    



def collision_loss_sim_sequence_prismatic(verts1, keypts_1, verts2, sel_faces_vals2, sel_faces_vns2, joints, n_sim_steps=100, back_sim=False, use_delta=False):
  # joint_dir, joint_pvp, joint_angle = joints
  # joint_dir = joint_dir.cuda()
  # joint_pvp = joint_pvp.cuda()
  
  ### should save the sequence of transformed shapes ... ###
  
  # verts1, faces1 = mesh_1
  # verts2, faces2 = mesh_2
  
  ### verts2, keypts_2.detach() ###
  ### verts2, keypts_2 ###
  verts2 = verts2.detach()
  # keypts_2 = keypts_2.detach()

  sel_faces_vns2 = sel_faces_vns2.detach()
  
  keypts_sequence = []
  delta_faces_pts_ds_sequence = []
  pts_in_faces_sequence = [] 
  ### pivot point prediction, joint axis direction prediction ### 
  # ### joint axis direction prediction ###
  
  joint_dir = joints["axis"]["dir"]
  # joint_pvp = joints["axis"]["center"]
  joint_a = joints["axis"]["a"]
  joint_b = joints["axis"]["b"]
  
  
  delta_joint_angle = (float(joint_b) - float(joint_a)) / float(n_sim_steps - 1) ### delta_joint_angles ###
  
  tot_collision_loss = 0.
  
  non_collided_pts = torch.ones((keypts_1.size(0), ), dtype=torch.float32, requires_grad=False) # .cuda()
  
  
  mesh_pts_sequence = []
  
  # def_pcs_sequence = []
  

  for i in range(0, n_sim_steps): ### joint_angle ###
    # cur_joint_angle = i * delta_joint_angle ### delta_joint_angle ###
    if not back_sim:
      cur_delta_dis = i * delta_joint_angle
    else:
      if i == 0:
        cur_delta_dis = 1.4
      else:
        cur_delta_dis = float(joint_b) - i * delta_joint_angle

    if use_delta:
      delta_joint_angle = (float(joint_b) - float(joint_a)) / float(100.0)
      ''' Prev. arti. state '''  
      cur_st_joint_dis = np.random.uniform(low=joint_a + delta_joint_angle, high=joint_b, size=(1,)).item() #### lower and upper limits of simulation angles
      moving_dis = joint_dir * cur_st_joint_dis ## (3,)
      moving_dis = moving_dis.unsqueeze(0)
      prev_pts = moving_dis + keypts_1
      
      #### distance_pts_faces, pts_in_faces, delta_faces_pts_ds ####
      prev_delta_faces_pts_ds, prev_pts_in_faces = get_distance_pts_faces(prev_pts, sel_faces_vals2, sel_faces_vns2)
      ''' Prev. arti. state ''' 
      
      ''' Current state ''' #### current states and dists ####
      cur_ed_joint_dis = cur_st_joint_dis - delta_joint_angle
      moving_dis = joint_dir * cur_ed_joint_dis ## (3,)
      moving_dis = moving_dis.unsqueeze(0)
      pts = moving_dis + keypts_1

      #### distance_pts_faces, pts_in_faces, delta_faces_pts_ds ####
      delta_faces_pts_ds, pts_in_faces = get_distance_pts_faces(pts, sel_faces_vals2, sel_faces_vns2)
      ''' Current state ''' 
      
      sgn_delta_faces_ds = torch.sign(delta_faces_pts_ds) ### sign of faces_ds
      sgn_prev_delta_faces_ds = torch.sign(prev_delta_faces_pts_ds) ### sign of prev_faces_ds
      ### different signs ###
      collision_pts_faces = (sgn_delta_faces_ds != sgn_prev_delta_faces_ds).float() ### n_pts x n_faces 
      
      cur_delta_faces_pts_ds = delta_faces_pts_ds
      
      collision_dists = collision_pts_faces * cur_delta_faces_pts_ds ### n_pts x n_faces
      ### collision_dists ### n_pts x n_faces
       ### i think the sim step 
       ### whether tow meshes collide with each other: in the target mesh, 
      ### collision_pulse, collision_dists
      # collision_pulse = (collision_dists * pts_in_faces * non_collided_pts.unsqueeze(-1)).unsqueeze(-1) * sel_faces_vns2.unsqueeze(0) ### n_pts x n_faces x 3 --> pulse
      collision_pulse = (collision_dists * pts_in_faces).unsqueeze(-1) * sel_faces_vns2.unsqueeze(0) ### n_pts x n_faces x 3 --> pulse
      
      # collided_indicator = ((pts_in_faces * non_collided_pts.unsqueeze(-1) * collision_pts_faces).sum(-1) > 0.1).float()
      
      ### loss version v2: for pts directly ###
      collision_loss = torch.sum(collision_pulse * pts.unsqueeze(1), dim=-1) ### n_pts x n_faces ### 
      ### loss version v2: for pts directly ###
      
      # non_collided_indicator = 1.0  - collided_indicator # - (collision_pulse.sum(-1).sum(-1) > 1e-6).float()
      # # print(f"collision_pulse: {collision_pulse.size()}, collided_indicator: {collided_indicator.size()}, non_collided_pts: {non_collided_pts.size()}")
      # # non_collided_pts[collided_indicator] = non_collided_pts[collided_indicator] * 0.
      # non_collided_pts = non_collided_pts * non_collided_indicator
      # # print(f"collision_loss: {collision_loss.sum().mean().item()}, collided_indicator: {collided_indicator.sum(-1).item()}, non_collided_pts: {non_collided_pts.sum(-1).item()}")
      
      # ### loss version v2: for pts directly ###
      # collision_loss = torch.sum(collision_pulse * pts.unsqueeze(1), dim=-1) ### n_pts x n_faces ### 
      # ### loss version v2: for pts directly ###
      collision_loss = torch.sum(collision_loss, dim=-1).sum() ###  ## for all faces ###
      tot_collision_loss += collision_loss
    else:
      moving_dis = joint_dir * cur_delta_dis ## (3,)
      moving_dis = moving_dis.unsqueeze(0)
    
      pts = moving_dis + keypts_1 ### keypts_1, moving_dis
      
      #### distance_pts_faces, pts_in_faces, delta_faces_pts_ds #### ### prev delta distsanees...
      delta_faces_pts_ds, pts_in_faces = get_distance_pts_faces(pts.detach().cpu(), sel_faces_vals2.detach().cpu(), sel_faces_vns2.detach().cpu())
      
      mesh_pts_expanded = verts1 + moving_dis
      mesh_pts_sequence.append(mesh_pts_expanded.clone())

      # if len(delta_faces_pts_ds_sequence) > 0 and i == 0 or i == n_sim_steps - 1:
      if len(delta_faces_pts_ds_sequence) > 0:
        prev_delta_faces_pts_ds = delta_faces_pts_ds_sequence[-1]
        # prev_pts_in_faces = pts_in_faces_sequence[-1] ### not important
        prev_pts = keypts_sequence[-1]
        
        sgn_delta_faces_ds = torch.sign(delta_faces_pts_ds) ### sign of faces_ds
        sgn_prev_delta_faces_ds = torch.sign(prev_delta_faces_pts_ds) ### sign of prev_faces_ds
        collision_pts_faces = (sgn_delta_faces_ds != sgn_prev_delta_faces_ds).float() ### 

        cur_delta_faces_pts_ds = delta_faces_pts_ds
 
        # collision_dists = collision_pts_faces * delta_faces_pts_ds ### n_pts x n_faces
        collision_dists = collision_pts_faces * cur_delta_faces_pts_ds ### n_pts x n_faces
        ### collision_dists ### n_pts x n_faces
        ### collision_pulse, collision_dists ### vns and collided pts.... ---> collided distances
        collision_pulse = 1.0 * (collision_dists * pts_in_faces * non_collided_pts.unsqueeze(-1)).unsqueeze(-1) * sel_faces_vns2.detach().cpu().unsqueeze(0) ### n_pts x n_faces x 3 --> pulse
        
        collided_indicator = ((pts_in_faces * non_collided_pts.unsqueeze(-1) * collision_pts_faces).sum(-1) > 0.1).float()
        
        ### loss version v2: for pts directly ###
        collision_loss = torch.sum(collision_pulse.cuda() * pts.unsqueeze(1), dim=-1) ### n_pts x n_faces ### 
        ### loss version v2: for pts directly ###
        
        non_collided_indicator = 1.0  - collided_indicator # - (collision_pulse.sum(-1).sum(-1) > 1e-6).float()
        # print(f"collision_pulse: {collision_pulse.size()}, collided_indicator: {collided_indicator.size()}, non_collided_pts: {non_collided_pts.size()}")
        # non_collided_pts[collided_indicator] = non_collided_pts[collided_indicator] * 0.
        non_collided_pts = non_collided_pts * non_collided_indicator
        # print(f"collision_loss: {collision_loss.sum().mean().item()}, collided_indicator: {collided_indicator.sum(-1).item()}, non_collided_pts: {non_collided_pts.sum(-1).item()}")
        
        # ### loss version v2: for pts directly ###
        # collision_loss = torch.sum(collision_pulse * pts.unsqueeze(1), dim=-1) ### n_pts x n_faces ### 
        # ### loss version v2: for pts directly ###
        collision_loss = torch.sum(collision_loss, dim=-1).sum() ###  ## for all faces ###
        tot_collision_loss += collision_loss


    delta_faces_pts_ds_sequence.append(delta_faces_pts_ds.clone())
    pts_in_faces_sequence.append(pts_in_faces.clone())
    keypts_sequence.append(pts.clone())
    
    
  ### can even test for one part at first
  return tot_collision_loss, keypts_sequence, mesh_pts_sequence ### collision_loss for all sim steps ###
    



def collision_loss_joint_structure_with_rest_pose_rnd_detect(meshes, keyptses, joints, n_sim_steps=100, selected_moving_part_idx=None, selected_state_idx=None, fix_other_parts=False):
  print("Input to collision sim function:")
  print([cur_keypts.size() for cur_keypts in keyptses])
  print([cur_v.size() for (cur_v, cur_f) in meshes] )
  ### meshes, keyptses, joints, n_sim_steps
  n_sim_steps = n_sim_steps + 1 #### add on one sim_steps ## not a sequence but a sequence of initial states
  
  tot_n_parts = len(meshes) ### tot_n_parts ###
  print(f"tot_n_parts: {tot_n_parts}")
  #### n_parts x (n_sim_steps ** n_parts)
  # for i_p in range(tot_n_parts)
  tot_moving_part_idxes = []
  tot_n_moving_parts = 0
  for i_p in range(tot_n_parts):
    cur_p_joint = joints[i_p]
    j_ty = cur_p_joint["type"]
    if j_ty != "none_motion":
      tot_n_moving_parts += 1
      tot_moving_part_idxes.append(i_p)
      
  selected_moving_part_idx = random.choice(tot_moving_part_idxes) if selected_moving_part_idx is None else selected_moving_part_idx
  tot_n_states = n_sim_steps ** (tot_n_moving_parts - 1)
  selected_state_idx = np.random.randint(low=0, high=tot_n_states, size=(1,)).item() if selected_state_idx is None else selected_state_idx
  
  print(f"selected_state_idx: {selected_state_idx}")
  cur_sel_part_moving_ty = joints[selected_moving_part_idx]["type"]
  

  cur_state_verts_list, cur_state_faces_vals, cur_state_faces_vns = [], [], []
  
  total_collision_loss = 0.
  tot_accum_moving_parts = 0
  
  tot_cur_other_p_states = []
  for i_other_p in range(tot_n_parts):
    # cur_p_verts, cur_p_faces = meshes[i_other_p] ### rest verts, rest faces 
    # cur_p_keypts = keyptses[i_other_p] ### rest keypts
    cur_other_p_joint_ty = joints[i_other_p]["type"]
    if i_other_p == selected_moving_part_idx:
      continue
    
    if cur_other_p_joint_ty != "none_motion" and not fix_other_parts: ## not fixed
      mod_base = n_sim_steps ** tot_accum_moving_parts ### mod base ---> mod n_sim_steps ###
      cur_other_p_state = (selected_state_idx // mod_base) % n_sim_steps
      tot_cur_other_p_states.append(cur_other_p_state)
      cur_other_p_joint_dir, cur_other_p_joint_center = joints[i_other_p]["axis"]["dir"], joints[i_other_p]["axis"]["center"]
      cur_other_p_joint_a, cur_other_p_joint_b = joints[i_other_p]["axis"]["a"], joints[i_other_p]["axis"]["b"]
      cur_other_p_axis_moving_delta = (float(cur_other_p_joint_b) - float(cur_other_p_joint_a)) / (n_sim_steps - 1)
      #### forward simulating ####
      # cur_other_p_joint_state = cur_other_p_joint_a + cur_other_p_axis_moving_delta * cur_other_p_state ### joint_state
      #### backward simulating ####
      cur_other_p_joint_state = cur_other_p_joint_b - (cur_other_p_axis_moving_delta) * (cur_other_p_state ) ### backward simlation ---- minus the selected moving state for the current state ###
      
      ### need to sel vertices and veritces values from
      cur_joint_type = 0 if cur_other_p_joint_ty == "revolute" else 1
      cur_other_p_keypts = keyptses[i_other_p]
      cur_other_p_verts, cur_other_p_faces = meshes[i_other_p][0], meshes[i_other_p][1]
      
      cur_other_p_sel_verts, cur_other_p_sel_faces, cur_other_p_sel_faces_vals = get_sub_verts_faces_from_pts(cur_other_p_verts, cur_other_p_faces, cur_other_p_keypts, rt_sel_faces=True)
      # cur_other_p_sel_faces = tot_sel_faces[i_other_p]
      ### other_p_trans_faces_vals
      cur_other_p_trans_verts, cur_other_p_trans_faces_vals, cur_other_p_trans_faces_ns = transform_vertices_keypts(cur_other_p_keypts, cur_other_p_verts, cur_other_p_faces, cur_other_p_sel_faces, cur_other_p_joint_state, cur_other_p_joint_dir, cur_other_p_joint_center, cur_joint_type)
      # cur_state_verts.append(cur_other_p_trans_verts)
      cur_state_verts_list.append(cur_other_p_trans_verts)
      cur_state_faces_vals.append(cur_other_p_trans_faces_vals)
      cur_state_faces_vns.append(cur_other_p_trans_faces_ns)
      tot_accum_moving_parts += 1
    else:
      print("HEre..,,")
      cur_other_p_keypts = keyptses[i_other_p]
      cur_other_p_verts, cur_other_p_faces = meshes[i_other_p][0], meshes[i_other_p][1]
      # cur_other_p_trans_faces_vals = tot_sel_faces_vals[i_other_p]
      # cur_other_p_trans_faces_ns = tot_sel_faces_vns[i_other_p]
      cur_other_p_sel_verts, cur_other_p_sel_faces, cur_other_p_trans_faces_vals = get_sub_verts_faces_from_pts(cur_other_p_verts, cur_other_p_faces, cur_other_p_keypts, rt_sel_faces=True)
      cur_other_p_trans_faces_ns = get_faces_normals(cur_other_p_trans_faces_vals) ### get_faces_normals
      # cur_state_verts.append(cur_other_p_verts)
      cur_state_verts_list.append(cur_other_p_verts)
      cur_state_faces_vals.append(cur_other_p_trans_faces_vals)
      cur_state_faces_vns.append(cur_other_p_trans_faces_ns)
  print(f"tot_cur_other_p_states: {tot_cur_other_p_states}")
  cur_state_verts = torch.cat(cur_state_verts_list, dim=0)
  cur_state_faces_vals = torch.cat(cur_state_faces_vals, dim=0)
  cur_state_faces_vns = torch.cat(cur_state_faces_vns, dim=0) 


  if cur_sel_part_moving_ty == "revolute":
    tot_collision_loss, keypts_sequence, mesh_pts_sequence = collision_loss_sim_sequence(meshes[selected_moving_part_idx][0], keyptses[selected_moving_part_idx], cur_state_verts, cur_state_faces_vals, cur_state_faces_vns, joints[selected_moving_part_idx], n_sim_steps=n_sim_steps, back_sim=True)
  elif cur_sel_part_moving_ty == "prismatic":
    tot_collision_loss, keypts_sequence, mesh_pts_sequence = collision_loss_sim_sequence_prismatic(meshes[selected_moving_part_idx][0], keyptses[selected_moving_part_idx], cur_state_verts, cur_state_faces_vals, cur_state_faces_vns, joints[selected_moving_part_idx], n_sim_steps=n_sim_steps, back_sim=True) ### whether to use backward simulating ###
  else:
    raise ValueError(f"Unrecognized j_ty: {selected_moving_part_idx}.")
  
  total_collision_loss = tot_collision_loss
  print(f"type: {cur_sel_part_moving_ty}, losss: {tot_collision_loss}")
  #### the last part's simulation process when other parts are put into their largest articulation states ####
  return total_collision_loss, keypts_sequence, mesh_pts_sequence,  cur_state_verts_list, selected_moving_part_idx, selected_state_idx
    
  
 