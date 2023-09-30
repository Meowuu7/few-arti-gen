from ctypes import util
import torch
import numpy as np
import os
import random
import utils
from common_utils import dataset_utils


def my_collate(batch):
    src_pc = []
    src_name = []
    key_pts = []
    dst_key_pts = []
    tar_pc = []
    tar_name = []
    w_pc = []
    src_ver = []
    src_face = []
    tar_ver = []
    tar_face = []
    real_ver = []
    real_face = []
    w_mesh = []
    src_cvx_to_pts = []
    dst_cvx_to_pts = []
    real_pc = []
    real_key_pts = []
    real_vertices = []
    real_faces = []
    real_w_pc = []
    real_w_mesh = []
    real_cvx_to_pts = []
    
    src_cvx_idx_list = []
    dst_cvx_idx_list = []
    
    # "src_cvx_idx_list": tot_src_cvx_idx_list, 
                # "dst_cvx_idx_list": tot_dst_cvx_idx_list ## bsz xzzz 
    
    for data in batch: # "src_edges": src_edges, "src_dofs": src_dofs
        src_pc.append(torch.from_numpy(data["src_pc"]).unsqueeze(0).float())
        src_name.append(data["src_name"])
        tar_pc.append(torch.from_numpy(data["tar_pc"]).unsqueeze(0).float())
        tar_name.append(data["tar_name"])
        key_pts.append(torch.from_numpy(data["key_pts"]).unsqueeze(0).float())
        w_pc.append(torch.from_numpy(data["w_pc"]).unsqueeze(0).float())
        
        src_ver.append(torch.from_numpy(data["src_ver"]).float().cuda())
        
        real_pc.append(torch.from_numpy(data["real_pc"]).unsqueeze(0).float())
        real_key_pts.append(torch.from_numpy(data["real_key_pts"]).unsqueeze(0).float())
        real_vertices.append(torch.from_numpy(data["real_vertices"]).float().cuda())
        real_faces.append(torch.from_numpy(data["real_faces"]))
        real_w_pc.append(torch.from_numpy(data["real_w_pc"]).unsqueeze(0).float())
        real_w_mesh.append(torch.from_numpy(data["real_w_mesh"]).float())
        real_cvx_to_pts.append(torch.from_numpy(data["real_cvx_to_pts"]).float().unsqueeze(0).cuda()) ### 
        
        
        
        src_face.append(torch.from_numpy(data["src_face"]))
        
        tar_ver.append(torch.from_numpy(data["tar_ver"]).float().cuda())
        
        tar_face.append(torch.from_numpy(data["tar_face"]))
        real_ver.append(torch.from_numpy(data["real_ver"]))
        real_face.append(torch.from_numpy(data["real_face"]))
        w_mesh.append(torch.from_numpy(data["w_mesh"]).float())
        
        src_cvx_idx_list.append(data["src_cvx_idx_list"])
        dst_cvx_idx_list.append(data["dst_cvx_idx_list"])
        
        dst_key_pts.append(torch.from_numpy(data["dst_key_pts"]).float().unsqueeze(0).cuda()) ### dst_key-pts
        
        src_cvx_to_pts.append(torch.from_numpy(data["src_cvx_to_pts"]).float().unsqueeze(0).cuda())
        dst_cvx_to_pts.append(torch.from_numpy(data["dst_cvx_to_pts"]).float().unsqueeze(0).cuda())
        
    src_pc = torch.cat(src_pc).cuda()
    tar_pc = torch.cat(tar_pc).cuda()
    key_pts = torch.cat(key_pts).cuda()
    w_pc = torch.cat(w_pc).cuda()
    dst_key_pts = torch.cat(dst_key_pts).cuda()
    src_cvx_to_pts = torch.cat(src_cvx_to_pts, dim=0).cuda()
    dst_cvx_to_pts = torch.cat(dst_cvx_to_pts, dim=0).cuda()
    
    real_pc = torch.cat(real_pc).cuda()
    real_key_pts = torch.cat(real_key_pts).cuda()
    real_w_pc = torch.cat(real_w_pc).cuda()
    real_cvx_to_pts = torch.cat(real_cvx_to_pts, dim=0).cuda()
    
    return {"src_pc": src_pc, "src_name": src_name, "key_pts": key_pts,
            "tar_pc": tar_pc, "tar_name": tar_name, "w_pc": w_pc,
            "src_ver": src_ver, "src_face": src_face, "w_mesh": w_mesh,
            "tar_ver": tar_ver, "tar_face": tar_face,
            "real_ver": real_ver, "real_face": real_face,
            "real_pc": real_pc, 
            "real_key_pts": real_key_pts, 
            "real_w_pc": real_w_pc, 
            "real_cvx_to_pts": real_cvx_to_pts, 
            "real_vertices": real_vertices, 
            "real_faces": real_faces, 
            "real_w_mesh": real_w_mesh, 
            "dst_key_pts": dst_key_pts, 
            "src_cvx_to_pts": src_cvx_to_pts, "dst_cvx_to_pts": dst_cvx_to_pts, 
            "src_cvx_idx_list": src_cvx_idx_list, "dst_cvx_idx_list": dst_cvx_idx_list
            }

### dataset def 
class ConvexDataset(torch.utils.data.Dataset):
    def __init__(self, phase="train", data_dir="../data/chair", split=None, opt=None, cvx_to_pts_sufix=None, n_keypoints=None, src_folder_fn=None, dst_folder_fn=None):
        super().__init__()
        # cur_part_cvx_to_pts_sufix
        self.data_dir = data_dir
        self.phase = phase
        self.opt = opt
        self.load_meta = self.opt.load_meta
        self.num_sampled_pts = 4096
        self.use_paired_data = self.opt.use_paired_data
        self.random_scaling = self.opt.random_scaling
        self.cvx_list_filter = self.opt.cvx_list_filter
        
        try:
          self.src_data_dir = self.opt.src_data_dir
          self.dst_data_dir = self.opt.dst_data_dir
          if len(self.src_data_dir) == 0:
            self.src_data_dir = self.data_dir
          if len(self.dst_data_dir) == 0:
            self.dst_data_dir = self.data_dir
        except:
          self.src_data_dir = self.data_dir
          self.dst_data_dir = self.data_dir
        
        
        self.cvx_to_pts_sufix = cvx_to_pts_sufix if cvx_to_pts_sufix is not None else self.opt.cvx_to_pts_sufix 
        
        try:
          self.src_cvx_to_pts_sufix = self.opt.src_cvx_to_pts_sufix 
          self.dst_cvx_to_pts_sufix = self.opt.dst_cvx_to_pts_sufix
        except:
          self.src_cvx_to_pts_sufix = self.cvx_to_pts_sufix
          self.dst_cvx_to_pts_sufix = self.cvx_to_pts_sufix ### cvx_to_pts_sufx ###j
        
        self.one_shp = opt.one_shp
        
        try:
          self.n_keypoints = self.opt.n_keypoints if n_keypoints is None else n_keypoints
          self.src_n_keypoints = self.opt.src_n_keypoints
          self.dst_n_keypoints = self.opt.dst_n_keypoints
        except:
          self.src_n_keypoints = self.n_keypoints
          self.dst_n_keypoints = self.n_keypoints
          
        # self.n_keypoints = self.opt.n_keypoints if n_keypoints is None else n_keypoints
        
        self.src_folder_fn = self.opt.src_folder_fn if src_folder_fn is None else src_folder_fn
        self.dst_folder_fn = self.opt.dst_folder_fn if dst_folder_fn is None else dst_folder_fn
        self.cvx_dim = self.opt.cvx_dim
        self.only_tar = self.opt.only_tar
        
        self.src_index = self.opt.src_index
        
        # src_cvx -> 
        self.only_src_cvx = self.opt.only_src_cvx
        
        self.small_tar_nn = self.opt.small_tar_nn
        self.cvx_folder_fn = self.opt.cvx_folder_fn
        if len(self.cvx_folder_fn) == 0:
          self.cvx_folder_fn = self.dst_folder_fn 
        
        
        self.split = split
        
        
        src_folder = os.path.join(self.src_data_dir, self.src_folder_fn if os.path.exists(os.path.join(self.src_data_dir, self.src_folder_fn)) else "src")
        dst_folder = os.path.join(self.dst_data_dir, self.dst_folder_fn if os.path.exists(os.path.join(self.dst_data_dir, self.dst_folder_fn)) else "dst")
        
        print(f"src_data_dir: {self.src_data_dir}, src_folder: {src_folder}")
        print(f"dst_data_dir: {self.dst_data_dir}, dst_folder: {dst_folder}")
        
        self.src_folder = src_folder ## src_folder --> 
        self.dst_folder = dst_folder
        
        
        # /data/datasets/genn/ShapeNetCoreV2_Deform/03636649_c_512/dst_def/1a9c1cbf1ca9ca24274623f5a5d0bcdc_manifold_tetra.mesh__sf.keypoints_256.weights.txt
        self.keypts_sufix = f"_manifold_tetra.mesh__sf.keypoints_{self.n_keypoints}.weights.sampled_4096.txt"
        
        ''' src_models ''' 
        if not os.path.exists(os.path.join(self.src_data_dir, "all.txt")): ## src_n_keypoints
          self.keypts_sufix = f"_manifold_tetra.mesh__sf.keypoints_{self.src_n_keypoints}.weights.sampled_4096.txt"
          print(f"src_model_dir:", os.path.join(self.src_data_dir, self.src_folder_fn))
          self.src_models = os.listdir(os.path.join(self.src_data_dir, self.src_folder_fn)) ## dst_fodler_fn 
          # print(f"src_models: {self.src_models}")
          print(f"src keypts_sufix: {self.keypts_sufix}")
          self.src_models = [fn[: -len(self.keypts_sufix)] for fn in self.src_models if fn.endswith(self.keypts_sufix)]
          self.src_models = sorted(self.src_models)
          # print(f'models: {self.models}')
          # self.dst_models = self.models
        else:
          with open(os.path.join(self.src_folder, "all.txt"), "r") as rf: ### src_folder and all.txt
            lines = rf.readlines()
            self.src_models = [line.rstrip() for line in lines]
        ''' src_models ''' 
        
        
        ''' src_models ''' 
        if not os.path.exists(os.path.join(self.dst_data_dir, "all.txt")): ## src_n_keypoints
          self.keypts_sufix = f"_manifold_tetra.mesh__sf.keypoints_{self.dst_n_keypoints}.weights.sampled_4096.txt"
          print(f"dst_model_dir:", os.path.join(self.dst_data_dir, self.dst_folder_fn))
          self.dst_models = os.listdir(os.path.join(self.dst_data_dir, self.dst_folder_fn)) ## dst_fodler_fn 
          # print(f"dst_models: {self.dst_models}")
          print(f"dst keypts_sufix: {self.keypts_sufix}")
          self.dst_models = [fn[: -len(self.keypts_sufix)] for fn in self.dst_models if fn.endswith(self.keypts_sufix)]
          self.dst_models = sorted(self.dst_models)
          # print(f'models: {self.models}')
          # self.dst_models = self.models
        else:
          with open(os.path.join(self.dst_folder, "all.txt"), "r") as rf: ### src_folder and all.txt
            lines = rf.readlines()
            self.dst_models = [line.rstrip() for line in lines]
        ''' src_models ''' 
        
        
        ### surface points, mesh points, cvx points 
        
        print(f"current dst_models: {len(self.dst_models)}, cur src_models: {len(self.src_models)}")
        
        self.n_shots = self.opt.n_shots
        if self.split == "train":
          self.src_models = self.src_models[:self.n_shots]
          # self.dst_models = self.dst_models[:self.n_shots]
        else:
          n_ref_samples = 1000 ## n_ref_sampless ### default n_ref_samples ###
          n_ref_samples = min(n_ref_samples, len(self.src_models) - self.n_shots) ### n_shots, n_ref_sampelsxxx ###
          self.src_models = self.src_models[-n_ref_samples: ]
          self.dst_models = self.dst_models[-n_ref_samples: ]
          
        
        
        print(f"Meta-infos loaded with src_models: {len(self.src_models)}, dst_models: {len(self.dst_models)}")
        
        # self.valid_models = {model_fn: 1 for model_fn in self.src_models if model_fn in self.dst_models}
        
        # self.src_models = [model_fn for model_fn in self.src_models if model_fn in self.valid_models]
        # self.dst_models = [model_fn for model_fn in self.dst_models if model_fn in self.valid_models]
        
        if opt.debug:
          self.src_models = self.src_models[:2]
          self.dst_models = self.dst_models[:2]
        
        
        
        self.src_n_models = len(self.src_models)
        self.dst_n_models = len(self.dst_models)
        
        ##### source pc, key_pts, mesh_vertices, mesh_faces, w_pc, w_mesh #####
        self.src_pc = [None for _ in range(self.src_n_models)]
        self.src_key_pts = [None for _ in range(self.src_n_models)]
        self.src_mesh_vertices = [None for _ in range(self.src_n_models)]
        self.src_mesh_faces = [None for _ in range(self.src_n_models)]
        self.src_w_pc = [None for _ in range(self.src_n_models)]
        self.src_w_mesh = [None for _ in range(self.src_n_models)]
        self.src_cvx_to_pts = [None for _ in range(self.src_n_models)]
        
        ##### source pc, key_pts, mesh_vertices, mesh_faces, w_pc, w_mesh #####
        self.dst_pc = [None for _ in range(self.dst_n_models)]
        self.dst_key_pts = [None for _ in range(self.dst_n_models)]
        self.dst_mesh_vertices = [None for _ in range(self.dst_n_models)]
        self.dst_mesh_faces = [None for _ in range(self.dst_n_models)]
        self.dst_w_pc = [None for _ in range(self.dst_n_models)]
        self.dst_w_mesh = [None for _ in range(self.dst_n_models)]
        self.dst_cvx_to_pts = [None for _ in range(self.dst_n_models)]
        
    def load_data_from_model_indicator(self, rt_folder, model_indicator):
        
        cur_n_keypoints = self.src_n_keypoints if rt_folder == self.src_folder else self.dst_n_keypoints 
        
        ### cur_cvx_to_pts_sufix --> for src_folder or  dst_folder 
        cur_cvx_to_pts_sufix = self.src_cvx_to_pts_sufix if rt_folder == self.src_folder else self.dst_cvx_to_pts_sufix
        cur_data_dir = self.src_data_dir if rt_folder == self.src_folder else self.dst_data_dir
        
      
        cur_model = model_indicator
        
        cur_keypoints_fn = cur_model + f"_manifold_tetra.mesh__sf.keypoints_{cur_n_keypoints}.obj"
        cur_sampled_pts_fn = cur_model + "_manifold_tetra.mesh__sf.sampled_4096.obj"
        cur_surface_pts_fn = cur_model + "_manifold_tetra.mesh__sf.obj"
        cur_weights_sampled_fn = cur_model + f"_manifold_tetra.mesh__sf.keypoints_{cur_n_keypoints}.weights.sampled_4096.txt"
        cur_weights_tot_fn = cur_model + f"_manifold_tetra.mesh__sf.keypoints_{cur_n_keypoints}.weights.txt"
        
        cvx_to_pts_sufix = f'_cvx_to_verts_cdim_{self.cvx_dim}.npy'
        cvx_to_pts_sufix = "_cvx_to_verts.npy"
        cvx_to_pts_sufix = self.cvx_to_pts_sufix
        cvx_to_pts_sufix = cur_cvx_to_pts_sufix ## for current cvx_to_pts_sufix ### 
        if self.cvx_dim > 0:
          cur_cvx_to_verts_fn = cur_model + cvx_to_pts_sufix
        else:
          cur_cvx_to_verts_fn = cur_model + cvx_to_pts_sufix
          
        
        
        # print(f"loading from {os.path.join(rt_folder, cur_cvx_to_verts_fn)}")
        cur_keypoints, _ =  utils.read_obj_file_ours(os.path.join(rt_folder, cur_keypoints_fn))
        cur_sampled, _ = utils.read_obj_file_ours(os.path.join(rt_folder, cur_sampled_pts_fn))
        cur_surface, cur_faces = utils.read_obj_file_ours(os.path.join(rt_folder, cur_surface_pts_fn), sub_one=True)
        cur_weights_sampled_keypoints = utils.load_txt(os.path.join(rt_folder, cur_weights_sampled_fn))
        cur_weights_mesh_keypoints = utils.load_txt(os.path.join(rt_folder, cur_weights_tot_fn))
        cur_faces = np.array(cur_faces, dtype=np.long) # .long()
        
        
        # cur_data_dir; self.cvx_folder_fn, cvx_to_verts_fn #### ---> cvx_to_verts_fns ####
        cvx_to_pts_load_fn = os.path.join(os.path.join(cur_data_dir, self.cvx_folder_fn), cur_cvx_to_verts_fn)
        
        
        if not os.path.exists(cvx_to_pts_load_fn):
          print(f"No existing cvx_to_pts_load_fn: {cvx_to_pts_load_fn}.")
          cur_scaled_sampled = cur_sampled / 2.
          cvx_to_pts = {
            0: cur_scaled_sampled
          }
        else:
          # cvx_to_pts = np.load(os.path.join(rt_folder, cur_cvx_to_verts_fn),allow_pickle=True).item()
          cvx_to_pts = np.load(cvx_to_pts_load_fn, allow_pickle=True).item()
        
        return cur_sampled, cur_keypoints, cur_surface, cur_faces, cur_weights_sampled_keypoints, cur_weights_mesh_keypoints, cvx_to_pts
    
    def get_src_dst_cvx_to_pts(self, src_cvx_to_pts, dst_cvx_to_pts, src_pc, tar_pc, cvx_list_filter=True):
      src_cvx_to_pts_list = []
      dst_cvx_to_pts_list = []
      
      
      tot_src_cvx_idx_list = []
      tot_dst_cvx_idx_list = []
      if cvx_list_filter:
        for cvx_idx in src_cvx_to_pts: ### 
          if cvx_idx in dst_cvx_to_pts:
            cur_src_cvx_pts, cur_dst_cvx_pts = src_cvx_to_pts[cvx_idx], dst_cvx_to_pts[cvx_idx]
            src_cvx_to_pts_list.append(np.reshape(cur_src_cvx_pts, (1, cur_src_cvx_pts.shape[0], 3)))
            dst_cvx_to_pts_list.append(np.reshape(cur_dst_cvx_pts, (1, cur_dst_cvx_pts.shape[0], 3)))
            tot_src_cvx_idx_list.append(cvx_idx)
            tot_dst_cvx_idx_list.append(cvx_idx)
      else:
        for cvx_idx in src_cvx_to_pts:
          cur_src_cvx_pts = src_cvx_to_pts[cvx_idx]
          src_cvx_to_pts_list.append(np.reshape(cur_src_cvx_pts, (1, cur_src_cvx_pts.shape[0], 3)))
          tot_src_cvx_idx_list.append(cvx_idx)
        for cvx_idx in dst_cvx_to_pts: ### dst..
          cur_dst_cvx_pts = dst_cvx_to_pts[cvx_idx]
          dst_cvx_to_pts_list.append(np.reshape(cur_dst_cvx_pts, (1, cur_dst_cvx_pts.shape[0], 3)))
          tot_dst_cvx_idx_list.append(cvx_idx)
      if len(src_cvx_to_pts_list) == 0: ## src_cvx_to_pts_idx ##
        src_cvx_to_pts_list = np.reshape(src_pc, (1, src_pc.shape[0], 3))
        dst_cvx_to_pts_list = np.reshape(tar_pc, (1, tar_pc.shape[0], 3))
        # tot_src_cvx_idx_list.
      else:
        src_cvx_to_pts_list = np.concatenate(src_cvx_to_pts_list, axis=0)
        dst_cvx_to_pts_list = np.concatenate(dst_cvx_to_pts_list, axis=0)
      # print(f"Src_cvx_to_pts_list: {src_cvx_to_pts_list.shape}, dst_cvx_to_pts_list: {dst_cvx_to_pts_list.shape}")
      return src_cvx_to_pts_list, dst_cvx_to_pts_list, tot_src_cvx_idx_list, tot_dst_cvx_idx_list
        
    
    def __len__(self):
        return self.src_n_models
        # return self.dst_n_models
        # return self.src_n_models * self.dst_n_models
        

    
    def apply_random_scaling(self, vertices, pc, keypoints, dst_cvx_to_pts_list):
        scale_normalizing_factors = np.random.uniform(low=0.75, high=1.25, size=(3,))
        # min + (max - min) * scale_normalizing_factors (normalizing_factors)
        scale_normalizing_factors = np.reshape(scale_normalizing_factors, [1, 3])
        vertices_center = utils.get_vertices_center(vertices)
        vertices_center = np.reshape(vertices_center, (1, 3))
        
        centered_vertices = vertices - vertices_center
        centered_pc = pc - vertices_center
        centered_keypoints = keypoints - vertices_center
        
        vertices_scale = utils.get_vertices_scale(centered_vertices)
        
        scaled_vertices = centered_vertices * scale_normalizing_factors
        scaled_pc = centered_pc * scale_normalizing_factors
        scaled_keypoints = centered_keypoints * scale_normalizing_factors
        
        scaled_vertices = utils.normalize_vertices_scale(scaled_vertices) * vertices_scale
        scaled_pc = utils.normalize_vertices_scale(scaled_pc) * vertices_scale
        scaled_keypoints = utils.normalize_vertices_scale(scaled_keypoints) * vertices_scale
        
        ori_cvx_pts_shp = dst_cvx_to_pts_list.shape
        dst_cvx_to_pts_list_exp = np.reshape(dst_cvx_to_pts_list, (dst_cvx_to_pts_list.shape[0] * dst_cvx_to_pts_list.shape[1], 3))
        dst_cvx_to_pts_list_exp = utils.normalize_vertices_scale(dst_cvx_to_pts_list_exp * scale_normalizing_factors) * vertices_scale
        dst_cvx_to_pts_list = np.reshape(dst_cvx_to_pts_list, ori_cvx_pts_shp)
        return scaled_vertices, scaled_pc, scaled_keypoints, dst_cvx_to_pts_list
      
    def dict_to_list(self, cvx_to_pts_dict, cur_pc):
      dst_cvx_to_pts_list = []
      for cvx in cvx_to_pts_dict:
        cur_cvx_pts = cvx_to_pts_dict[cvx]
        dst_cvx_to_pts_list.append(np.reshape(cur_cvx_pts, (1, cur_cvx_pts.shape[0], 3)))
      if len(dst_cvx_to_pts_list) == 0:
        dst_cvx_to_pts_list = np.reshape(cur_pc, (1, cur_pc.shape[0], 3))
      else:
        dst_cvx_to_pts_list = np.concatenate(dst_cvx_to_pts_list, axis=0)
      return dst_cvx_to_pts_list
        
        
        
    def get_pc_via_idx(self, idxx=None, return_batched=True, return_vertices=False):
      if idxx is None:
        idxx = random.choice(range(self.dst_n_models))
        
      src_pc, src_key_pts, src_vertices, src_faces, src_w_pc, src_w_mesh, src_cvx_to_pts = \
          self.get_info_via_idx(idxx)
      ### 
      src_pc = torch.from_numpy(src_pc).float().cuda()
      src_vertices = torch.from_numpy(src_vertices).float().cuda()
      if return_batched:
        src_vertices = src_vertices.unsqueeze(0)
        src_pc = src_pc.unsqueeze(0)
      if return_vertices:
        return src_vertices
      else:
        return src_pc

    def get_objs_info_via_idxess(self, dst_idx):
      if self.dst_pc[dst_idx] is None: ### is None 
          tar_pc, dst_key_pts, dst_vertices, dst_faces, dst_w_pc, dst_w_mesh, dst_cvx_to_pts = self.load_data_from_model_indicator(self.dst_folder, self.dst_models[dst_idx])
          self.dst_pc[dst_idx] = tar_pc
          self.dst_key_pts[dst_idx] = dst_key_pts
          self.dst_mesh_vertices[dst_idx] = dst_vertices
          self.dst_mesh_faces[dst_idx] = dst_faces
          self.dst_w_pc[dst_idx] = dst_w_pc
          self.dst_w_mesh[dst_idx] = dst_w_mesh
          self.dst_cvx_to_pts[dst_idx] = dst_cvx_to_pts
      idxx = dst_idx
      cur_pc, cur_key_pts, cur_vertices, cur_faces, cur_w_pc, cur_w_mesh, cur_cvx_to_pts = self.dst_pc[idxx], self.dst_key_pts[idxx], self.dst_mesh_vertices[idxx], self.dst_mesh_faces[idxx], self.dst_w_pc[idxx], self.dst_w_mesh[idxx], self.dst_cvx_to_pts[idxx]
      rt_dict = {
        "pc": (cur_pc), 
        "key_pts": (cur_key_pts),
        "vertices": cur_vertices,
        "faces": cur_faces,
        "cvx_to_pts": cur_cvx_to_pts
      }
      return rt_dict
    
    
    def __getitem__(self, idx, dst_idx=None):
        
        
        def get_info_via_idx(idxx, for_src=False):
          if for_src:
            cur_pc, cur_key_pts, cur_vertices, cur_faces, cur_w_pc, cur_w_mesh, cur_cvx_to_pts = self.src_pc[src_idx], self.src_key_pts[src_idx], self.src_mesh_vertices[src_idx], self.src_mesh_faces[src_idx], self.src_w_pc[src_idx], self.src_w_mesh[src_idx], self.src_cvx_to_pts[src_idx]
          else:
            cur_pc, cur_key_pts, cur_vertices, cur_faces, cur_w_pc, cur_w_mesh, cur_cvx_to_pts = self.dst_pc[idxx], self.dst_key_pts[idxx], self.dst_mesh_vertices[idxx], self.dst_mesh_faces[idxx], self.dst_w_pc[idxx], self.dst_w_mesh[idxx], self.dst_cvx_to_pts[idxx]
          return cur_pc, cur_key_pts, cur_vertices, cur_faces, cur_w_pc, cur_w_mesh, cur_cvx_to_pts
        
        ori_idx = idx ### 
        #### randomly choose index for src_model and dst_model ####
        idx = random.choice(range(self.src_n_models * self.dst_n_models)) ## dst_n_models
        # ori_idx = idx // self.dst_n_models
        
        ## dst_n_models --> 
        real_idx = random.choice(range(self.dst_n_models))
        
        
        # src_idx = idx // self.dst_n_models
        if self.one_shp:
          src_idx = 0
        else:
          src_idx = ori_idx ### src_idx ###
          
        if self.src_index >= 0:
          src_idx = self.src_index
        # dst_idx = idx % self.dst_n_models
        
        if dst_idx is None:
          if self.small_tar_nn:
            dst_idx = (idx % 3) + 1
          else: ### idx %
            dst_idx = idx % self.dst_n_models
            
        forbit_idxes = [17, 14, 26, 25, 15, 18, 21, 35, 14, 31, 34, 19, 29, 11]
        if src_idx in forbit_idxes:
          src_idx = 0

        if dst_idx in forbit_idxes:
          dst_idx = 0

        # src_idx = 29
        
        print(f"src_idx: {src_idx}, dst_idx: {dst_idx}, src_models: { self.src_models[src_idx]}, dst_models: {self.dst_models[dst_idx]}")
        
        if self.src_pc[src_idx] is None:
          src_pc, src_key_pts, src_vertices, src_faces, src_w_pc, src_w_mesh, src_cvx_to_pts = self.load_data_from_model_indicator(self.src_folder, self.src_models[src_idx])
          self.src_pc[src_idx] = src_pc
          self.src_key_pts[src_idx] = src_key_pts
          self.src_mesh_vertices[src_idx] = src_vertices
          self.src_mesh_faces[src_idx] = src_faces
          self.src_w_pc[src_idx] = src_w_pc
          self.src_w_mesh[src_idx] = src_w_mesh
          self.src_cvx_to_pts[src_idx] = src_cvx_to_pts ## cvx_to_pts
        while not (self.src_w_pc[src_idx].shape[1] == self.src_n_keypoints and self.src_w_pc[src_idx].shape[0] == self.num_sampled_pts):
          src_idx = random.choice(range(len(self.src_pc)))
          if self.src_pc[src_idx] is None:
            src_pc, src_key_pts, src_vertices, src_faces, src_w_pc, src_w_mesh, src_cvx_to_pts = self.load_data_from_model_indicator(self.src_folder, self.src_models[src_idx])
            self.src_pc[src_idx] = src_pc
            self.src_key_pts[src_idx] = src_key_pts
            self.src_mesh_vertices[src_idx] = src_vertices
            self.src_mesh_faces[src_idx] = src_faces
            self.src_w_pc[src_idx] = src_w_pc
            self.src_w_mesh[src_idx] = src_w_mesh
            self.src_cvx_to_pts[src_idx] = src_cvx_to_pts
        
        # print(f"src_idx: {src_idx}, dst_idx: {dst_idx}")
        if self.dst_pc[dst_idx] is None: ### is None 
          tar_pc, dst_key_pts, dst_vertices, dst_faces, dst_w_pc, dst_w_mesh, dst_cvx_to_pts = self.load_data_from_model_indicator(self.dst_folder, self.dst_models[dst_idx])
          self.dst_pc[dst_idx] = tar_pc
          self.dst_key_pts[dst_idx] = dst_key_pts
          self.dst_mesh_vertices[dst_idx] = dst_vertices
          self.dst_mesh_faces[dst_idx] = dst_faces
          self.dst_w_pc[dst_idx] = dst_w_pc
          self.dst_w_mesh[dst_idx] = dst_w_mesh
          self.dst_cvx_to_pts[dst_idx] = dst_cvx_to_pts
          
        if self.dst_pc[real_idx] is None:
          real_pc, real_key_pts, real_vertices, real_faces, real_w_pc, real_w_mesh, real_cvx_to_pts = self.load_data_from_model_indicator(self.dst_folder, self.dst_models[dst_idx])
          self.dst_pc[real_idx] = real_pc
          self.dst_key_pts[real_idx] = real_key_pts
          self.dst_mesh_vertices[real_idx] = real_vertices
          self.dst_mesh_faces[real_idx] = real_faces
          self.dst_w_pc[real_idx] = real_w_pc
          self.dst_w_mesh[real_idx] = real_w_mesh
          self.dst_cvx_to_pts[real_idx] = real_cvx_to_pts # 

        src_pc, src_key_pts, src_vertices, src_faces, src_w_pc, src_w_mesh, src_cvx_to_pts = \
          get_info_via_idx(src_idx, for_src=True) 
          # self.src_pc[src_idx], self.src_key_pts[src_idx], self.src_mesh_vertices[src_idx], self.src_mesh_faces[src_idx], self.src_w_pc[src_idx], self.src_w_mesh[src_idx], self.src_cvx_to_pts[src_idx]
        
        ### numpy arrays ###
        tar_pc, dst_key_pts, dst_vertices, dst_faces, dst_w_pc, dst_w_mesh, dst_cvx_to_pts = \
          get_info_via_idx(dst_idx) ## va idx
          # self.dst_pc[dst_idx], self.dst_key_pts[dst_idx], self.dst_mesh_vertices[dst_idx], self.dst_mesh_faces[dst_idx], self.dst_w_pc[dst_idx], self.dst_w_mesh[dst_idx], self.dst_cvx_to_pts[dst_idx]
        
        real_pc, real_key_pts, real_vertices, real_faces, real_w_pc, real_w_mesh, real_cvx_to_pts = \
          get_info_via_idx(real_idx)
          # self.dst_pc[real_idx], self.dst_key_pts[real_idx], self.dst_mesh_vertices[real_idx], self.dst_mesh_faces[real_idx], self.dst_w_pc[real_idx], self.dst_w_mesh[real_idx], self.dst_cvx_to_pts[real_idx]
        
        # src_edges, src_dofs = dataset_utils.get_edges_from_faces(src_vertices, src_faces)
        # tar_edges, tar_dofs = dataset_utils.get_edges_from_faces(dst_vertices, dst_faces)
        
        src_cvx_to_pts_list, dst_cvx_to_pts_list, tot_src_cvx_idx_list, tot_dst_cvx_idx_list = self.get_src_dst_cvx_to_pts(src_cvx_to_pts, dst_cvx_to_pts, src_pc, tar_pc, cvx_list_filter=self.cvx_list_filter)
        
        # dict_to_list(self, cvx_to_pts_dict, cur_pc)
        #  "src_cvx_idx_list": src_cvx_idx_list, "dst_cvx_idx_list": dst_cvx_idx_list
        real_cvx_to_pts = self.dict_to_list(real_cvx_to_pts, real_pc)
        
        # if self.random_scaling:
        #     dst_vertices, tar_pc, dst_key_pts, dst_cvx_to_pts_list = self.apply_random_scaling(dst_vertices, tar_pc, dst_key_pts, dst_cvx_to_pts_list)
        
        src_name = f"src_{src_idx}"
        tar_name = f"dst_{dst_idx}"

        #### dst_key_pts: n_key_pts x 3 ####
        return {"src_name": src_name, "tar_name": tar_name,
                "src_pc": src_pc, "tar_pc": tar_pc,
                "src_ver": src_vertices, "src_face": src_faces,
                "tar_ver": dst_vertices, "tar_face": dst_faces,
                "real_ver": dst_vertices, "real_face": dst_faces,
                "key_pts": src_key_pts, ## src_
                # "w_mesh": src_w_mesh, "w_pc": src_w_pc, 
                "w_mesh": src_w_mesh, "w_pc": src_w_pc, 
                "dst_key_pts": dst_key_pts, 
                "src_cvx_to_pts": src_cvx_to_pts_list,
                "dst_cvx_to_pts": dst_cvx_to_pts_list, 
                #### real examples ####
                "real_pc": real_pc, "real_key_pts": real_key_pts, 
                "real_vertices": real_vertices, "real_faces": real_faces, 
                "real_w_pc": real_vertices, "real_w_mesh": real_faces, 
                "real_cvx_to_pts": real_cvx_to_pts,  ### cvx to pts ...
                "src_cvx_idx_list": tot_src_cvx_idx_list, 
                "dst_cvx_idx_list": tot_dst_cvx_idx_list
                }
