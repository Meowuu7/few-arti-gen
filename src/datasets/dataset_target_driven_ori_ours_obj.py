from ctypes import util
import torch
import numpy as np
import os
import random
import utils
from common_utils import dataset_utils

### collect data from the dataset ###
def my_collate(batch):
    nn_parts = len(batch[0]["src_pc"])
    src_pc = [[] for _ in range(nn_parts)]
    src_name = []
    key_pts = [[] for _ in range(nn_parts)]
    dst_key_pts = [[] for _ in range(nn_parts)]
    tar_pc = [[] for _ in range(nn_parts)]
    tar_name = []
    w_pc = [[] for _ in range(nn_parts)]
    src_ver = [[] for _ in range(nn_parts)]
    src_face = [[] for _ in range(nn_parts)]
    tar_ver = [[] for _ in range(nn_parts)]
    tar_face = [[] for _ in range(nn_parts)]
    real_ver = [[] for _ in range(nn_parts)]
    real_face = [[] for _ in range(nn_parts)]
    w_mesh = [[] for _ in range(nn_parts)]
    # src_edges = []
    # src_dofs = []
    # tar_edges = []
    # tar_dofs = []
    # src_cvx_to_pts = []
    # dst_cvx_to_pts = []
    # rt_dict = {
        #   "src_name": src_name, "tar_name": tar_name,
        #   "src_pc": tot_src_pc, "tar_pc": tot_dst_pc,
        #   "src_ver": tot_src_vertices, "src_face": tot_src_faces,
        #   "tar_ver": tot_dst_vertices, "tar_face": tot_dst_faces,
        #   "real_ver": tot_dst_vertices, "real_face": tot_dst_faces,
        #   "key_pts": tot_src_key_pts,
        #   "w_mesh": tot_src_w_mesh, "w_pc": tot_src_w_pc, 
        # }
    for data in batch: # "src_edges": src_edges, "src_dofs": src_dofs
        src_name.append(data["src_name"])
        tar_name.append(data["tar_name"])
        for i_p in range(nn_parts):
          src_pc[i_p].append(torch.from_numpy(data["src_pc"][i_p]).unsqueeze(0).float())
          
          tar_pc[i_p].append(torch.from_numpy(data["tar_pc"][i_p]).unsqueeze(0).float())
          
          key_pts[i_p].append(torch.from_numpy(data["key_pts"][i_p]).unsqueeze(0).float())
          w_pc[i_p].append(torch.from_numpy(data["w_pc"][i_p]).unsqueeze(0).float())
          
          src_ver[i_p].append(torch.from_numpy(data["src_ver"][i_p]).float().cuda())
          
          src_face[i_p].append(torch.from_numpy(data["src_face"][i_p]))
          
          tar_ver[i_p].append(torch.from_numpy(data["tar_ver"][i_p]).float().cuda())
          
          tar_face[i_p].append(torch.from_numpy(data["tar_face"][i_p]))
          real_ver[i_p].append(torch.from_numpy(data["real_ver"][i_p]))
          real_face[i_p].append(torch.from_numpy(data["real_face"][i_p]))
          w_mesh[i_p].append(torch.from_numpy(data["w_mesh"][i_p]).float())
        
        # src_edges.append(torch.from_numpy(data['src_edges']).long().unsqueeze(0).cuda()) #### src_edges
        # src_dofs.append(torch.from_numpy(data['src_dofs']).float().unsqueeze(0).cuda()) #### vert_dofs 
        
        # tar_edges.append(torch.from_numpy(data['tar_edges']).long().unsqueeze(0).cuda()) #### src_edges
        # tar_dofs.append(torch.from_numpy(data['tar_dofs']).float().unsqueeze(0).cuda()) #### vert_dofs 
        
        # dst_key_pts.append(torch.from_numpy(data["dst_key_pts"]).float().unsqueeze(0).cuda()) ### dst_key-pts
        
        # src_cvx_to_pts.append(torch.from_numpy(data["src_cvx_to_pts"]).float().unsqueeze(0).cuda())
        # dst_cvx_to_pts.append(torch.from_numpy(data["dst_cvx_to_pts"]).float().unsqueeze(0).cuda())
        
    src_pc = [torch.cat(cur_src_pc).cuda() for cur_src_pc in src_pc]
    tar_pc = [torch.cat(cur_tar_pc).cuda() for cur_tar_pc in tar_pc]
    key_pts = [torch.cat(cur_key_pts).cuda() for cur_key_pts in key_pts]
    w_pc = [torch.cat(cur_w_pc).cuda() for cur_w_pc in w_pc]
    # dst_key_pts = torch.cat(dst_key_pts).cuda()
    # src_cvx_to_pts = torch.cat(src_cvx_to_pts, dim=0).cuda()
    # dst_cvx_to_pts = torch.cat(dst_cvx_to_pts, dim=0).cuda()
    
    return {"src_pc": src_pc, "src_name": src_name, "key_pts": key_pts,
            "tar_pc": tar_pc, "tar_name": tar_name, "w_pc": w_pc,
            "src_ver": src_ver, "src_face": src_face, "w_mesh": w_mesh,
            "tar_ver": tar_ver, "tar_face": tar_face,
            "real_ver": real_ver, "real_face": real_face,
            # "src_edges": src_edges, "src_dofs": src_dofs,
            # "tar_edges": tar_edges, "tar_dofs": tar_dofs, "dst_key_pts": dst_key_pts, 
            # "src_cvx_to_pts": src_cvx_to_pts, "dst_cvx_to_pts": dst_cvx_to_pts
            }


class ChairDataset(torch.utils.data.Dataset):
    def __init__(self, phase="train", data_dir="../data/chair", split=None, opt=None):
        super().__init__()
        self.data_dir = data_dir
        self.phase = phase
        self.opt = opt
        self.load_meta = self.opt.load_meta
        self.num_sampled_pts = 4096
        self.n_shots = self.opt.n_shots ### nn_shots for training ## n_shots
        # self.use_paired_data = self.opt.use_paired_data
        # self.random_scaling = self.opt.random_scaling
        
        self.n_keypoints = self.opt.n_keypoints # 
        
        self.src_folder_fn = self.opt.src_folder_fn ## src folder fn; dst folder fn ##
        self.dst_folder_fn = self.opt.dst_folder_fn
        self.cvx_dim = self.opt.cvx_dim
        self.only_tar = self.opt.only_tar
        self.split = split
        self.cvx_folder_fn = self.opt.cvx_folder_fn
        if len(self.cvx_folder_fn) == 0:
          self.cvx_folder_fn = self.dst_folder_fn
        
        # src_folder = os.path.join(self.data_dir, "src")
        # dst folder fn #
        # self.data_dir should be a list of data root directory #
        src_folders = []
        dst_folders = []
        for i_p, cur_data_dir in enumerate(self.data_dir):
          cur_src_folder = os.path.join(cur_data_dir, self.src_folder_fn if os.path.exists(os.path.join(cur_data_dir, self.src_folder_fn)) else "src")
          cur_dst_folder = os.path.join(cur_data_dir, self.dst_folder_fn if os.path.exists(os.path.join(cur_data_dir, self.dst_folder_fn)) else "dst")
          src_folders.append(cur_src_folder) # src folders #
          dst_folders.append(cur_dst_folder) # dst folders #
        
        
        #### if we can use any shape as the source shape for deformation? ####
        # if not self.use_paired_data: ### not use paired data...
        #     src_folder = os.path.join(self.data_dir, "dst")
        # else:
        #     src_folder = os.path.join(self.data_dir, "src")
        
        ### dst_folder_fn ###
        # src_folder = os.path.join(self.data_dir, "src")
        # src_folder = os.path.join(self.data_dir, self.src_folder_fn if os.path.exists(os.path.join(self.data_dir, self.src_folder_fn)) else "src")
        # # dst_folder = os.path.join(self.data_dir, "dst")
        # dst_folder = os.path.join(self.data_dir, self.dst_folder_fn if os.path.exists(os.path.join(self.data_dir, self.dst_folder_fn)) else "dst")
        
        self.src_folders = src_folders ## src_folders ##
        self.dst_folders = dst_folders ## dst_folders ##
        self.src_folder =src_folders
        self.dst_folder =dst_folders
        self.cvx_to_pts_sufix = opt.cvx_to_pts_sufix
        
        # with open(os.path.join(self.data_dir, "all.txt")) as f:
        #     lines = f.readlines()
        #     self.models = [line.rstrip() for line in lines]

        # with open(os.path.join(self.src_folder, "all.txt"), "r") as rf:
        #   lines = rf.readlines()
        #   self.src_models = [line.rstrip() for line in lines]
        
        # with open(os.path.join(self.dst_folder, "all.txt"), "r") as rf:
        #   lines = rf.readlines()
        #   self.dst_models = [line.rstrip() for line in lines]

        ### keypts sufix ###
        self.keypts_sufix = f"_manifold_tetra.mesh__sf.keypoints_{self.n_keypoints}.weights.sampled_4096.txt"
        if not os.path.exists(os.path.join(self.data_dir[0], "all.txt")):
          # dst_models #
          self.models = os.listdir(os.path.join(self.data_dir[0], self.dst_folder_fn))
          self.models = [fn[: -len(self.keypts_sufix)] for fn in self.models if fn.endswith(self.keypts_sufix)] # two points ##
          for cur_data_dir in self.data_dir[1:]:
            cur_data_models = os.listdir(
              os.path.join(cur_data_dir, self.dst_folder_fn)
            )
            cur_data_models = [fn[: -len(self.keypts_sufix)] for fn in cur_data_models if fn.endswith(self.keypts_sufix)]
            self.models = [fn for fn in self.models if fn in cur_data_models]
          # self.models = os.listdir(os.path.join(self.data_dir, self.dst_folder_fn))
          # self.models = [fn[: -len(self.keypts_sufix)] for fn in self.models if fn.endswith(self.keypts_sufix)] # two points ##
          self.dst_models = self.models ### src models ###
          self.src_models = self.models ### dst models ###
        else: # dst_models
          with open(os.path.join(self.src_folder, "all.txt"), "r") as rf:
            lines = rf.readlines()
            self.src_models = [line.rstrip() for line in lines]
          
          with open(os.path.join(self.dst_folder, "all.txt"), "r") as rf:
            lines = rf.readlines()
            self.dst_models = [line.rstrip() for line in lines]
        
        print(f"Meta-infos loaded with src_models: {len(self.src_models)}, dst_models: {len(self.dst_models)}")
        
        
        if self.split == "train":
          # train_nns = 36
          # train_nns = 5
          # self.src_models = self.src_models[:train_nns]
          # self.dst_models = self.dst_models[:train_nns]
          self.src_models = self.src_models[:self.n_shots]
          self.dst_models = self.dst_models[:self.n_shots]
        else:
          train_nns = 5
          val_nns = 360
          val_nns = 1000
          
          val_nns = min(val_nns, len(self.src_models) - self.n_shots) 
          
          # self.src_models = self.src_models[train_nns: train_nns + val_nns]
          # self.dst_models = self.dst_models[train_nns: train_nns + val_nns]
          
          self.src_models = self.src_models[-val_nns: ]
          self.dst_models = self.dst_models[-val_nns: ]

        print(f"Meta-infos loaded with src_models: {len(self.src_models)}, dst_models: {len(self.dst_models)}")
        
        # self.valid_models = {model_fn: 1 for model_fn in self.src_models if model_fn in self.dst_models}
        
        # self.src_models = [model_fn for model_fn in self.src_models if model_fn in self.valid_models]
        # self.dst_models = [model_fn for model_fn in self.dst_models if model_fn in self.valid_models]
        
        
        self.cvx_to_verts_sufix="_manifold_cvx_to_verts.npy"
        self.surface_pts_sufix="_manifold_tetra.mesh__sf.sampled_4096.obj"
        
        
        # self.dst_models = [fn for fn in self.dst_models if os.path.exists(os.path.join(self.dst_folder, fn + f"_manifold_tetra.mesh__sf.keypoints_{self.n_keypoints}.weights.sampled_4096.txt"))]
        # self.src_models = self.dst_models
        
        
        
        # self.dst_folder_path = os.path.join(self.data_dir, self.dst_folder_fn)
        # self.dst_models = os.listdir(self.dst_folder_path)
        # self.dst_models = [fn for fn in self.dst_models if fn.endswith(self.surface_pts_sufix)]
        # self.dst_models = [fn[: len(fn) - len(self.surface_pts_sufix)] for fn in self.dst_models]
        # self.src_models = self.dst_models
        
        
        
        
        # if opt.few_shot:
        #   tot_n_src_models = 10
        #   if self.split == "train":
        #     self.src_models = self.src_models[: tot_n_src_models]
        #     self.dst_models = self.dst_models[: tot_n_src_models]
        #   else:
        #     self.src_models = self.src_models[tot_n_src_models: ]
        #     self.dst_models = self.dst_models[tot_n_src_models: ]
        # else:
        #   if self.split is not None:
        #     if self.split == "train":
        #       tot_n_src_models = int(len(self.src_models) * 0.9)
        #       self.src_models = self.src_models[: tot_n_src_models]
        #       self.dst_models = self.dst_models[: tot_n_src_models]
        #     elif self.split == "test":
        #       tot_n_src_models = int(len(self.src_models) * 0.9)
        #       self.src_models = self.src_models[tot_n_src_models: ]
        #       self.dst_models = self.dst_models[tot_n_src_models: ]
        #     else:
        #       raise ValueError(f"Unrecognized split: {self.split}")
          
        
        ### src n models ###
        self.src_n_models = len(self.src_models)
        self.dst_n_models = len(self.dst_models)
        ### keypoint 256: gt_step_0_subd_0_manifold_tetra.mesh__sf.keypoints_256 
        ### sampled 4096: gt_step_0_subd_0_manifold_tetra.mesh__sf.sampled_4096
        ### surface points: gt_step_0_subd_0_manifold_tetra.mesh__sf
        ### weights sampled 4096 -> 256: gt_step_0_subd_0_manifold_tetra.mesh__sf.keypoints_256.weights.sampled_4096
        ### weights tot pts -> 256: gt_step_0_subd_0_manifold_tetra.mesh__sf.keypoints_256.weights

        ##### source pc, key_pts, mesh_vertices, mesh_faces, w_pc, w_mesh #####
        nn_parts = len(self.data_dir)
        self.nn_parts = nn_parts
        
        self.src_pc = [[None for _ in range(self.src_n_models)] for _ in range(nn_parts)]
        self.src_key_pts = [[None for _ in range(self.src_n_models)] for _ in range(nn_parts)]
        self.src_mesh_vertices = [[None for _ in range(self.src_n_models)] for _ in range(nn_parts)]
        self.src_mesh_faces = [[None for _ in range(self.src_n_models)] for _ in range(nn_parts)]
        self.src_w_pc = [[None for _ in range(self.src_n_models)] for _ in range(nn_parts)]
        self.src_w_mesh = [[None for _ in range(self.src_n_models)] for _ in range(nn_parts)]
        self.src_cvx_to_pts = [[None for _ in range(self.src_n_models)] for _ in range(nn_parts)]
        
        ##### source pc, key_pts, mesh_vertices, mesh_faces, w_pc, w_mesh #####
        self.dst_pc = [[None for _ in range(self.dst_n_models)] for _ in range(nn_parts)]
        self.dst_key_pts = [[None for _ in range(self.dst_n_models)] for _ in range(nn_parts)]
        self.dst_mesh_vertices = [[None for _ in range(self.dst_n_models)] for _ in range(nn_parts)]
        self.dst_mesh_faces = [[None for _ in range(self.dst_n_models)] for _ in range(nn_parts)]
        self.dst_w_pc = [[None for _ in range(self.dst_n_models)] for _ in range(nn_parts)]
        self.dst_w_mesh = [[None for _ in range(self.dst_n_models)] for _ in range(nn_parts)]
        self.dst_cvx_to_pts = [[None for _ in range(self.dst_n_models)] for _ in range(nn_parts)]
        
    def load_data_from_model_indicator(self, rt_folder, model_indicator):
        cur_model = model_indicator
        
        cur_keypoints_fn = cur_model + f"_manifold_tetra.mesh__sf.keypoints_{self.n_keypoints}.obj"
        cur_sampled_pts_fn = cur_model + "_manifold_tetra.mesh__sf.sampled_4096.obj"
        cur_surface_pts_fn = cur_model + "_manifold_tetra.mesh__sf.obj"
        cur_weights_sampled_fn = cur_model + f"_manifold_tetra.mesh__sf.keypoints_{self.n_keypoints}.weights.sampled_4096.txt"
        cur_weights_tot_fn = cur_model + f"_manifold_tetra.mesh__sf.keypoints_{self.n_keypoints}.weights.txt"
        
        if self.cvx_dim > 0:
          cur_cvx_to_verts_fn = cur_model + f'_cvx_to_verts_cdim_{self.cvx_dim}.npy'
        else:
          cur_cvx_to_verts_fn = cur_model + "_cvx_to_verts.npy"
          
        cur_cvx_to_verts_fn = cur_model + self.cvx_to_pts_sufix
        
        # print(f"loading from {os.path.join(rt_folder, cur_cvx_to_verts_fn)}")
        cur_keypoints, _ =  utils.read_obj_file_ours(os.path.join(rt_folder, cur_keypoints_fn))
        cur_sampled, _ = utils.read_obj_file_ours(os.path.join(rt_folder, cur_sampled_pts_fn))
        cur_surface, cur_faces = utils.read_obj_file_ours(os.path.join(rt_folder, cur_surface_pts_fn), sub_one=True)
        cur_weights_sampled_keypoints = utils.load_txt(os.path.join(rt_folder, cur_weights_sampled_fn))
        cur_weights_mesh_keypoints = utils.load_txt(os.path.join(rt_folder, cur_weights_tot_fn))
        cur_faces = np.array(cur_faces, dtype=np.long) # .long()
        
        
        # cur_cvx_to_verts_fn = cur_model + "_cvx_to_verts.npy"
        cvx_to_pts_load_fn = os.path.join(rt_folder, cur_cvx_to_verts_fn)
        # print(f"cur_cvx_to_verts_fn: {cvx_to_pts_load_fn}")
        # if not os.path.exists(cvx_to_pts_load_fn):
        #   cvx_to_pts_load_fn = os.path.join(self.dst_folder, cur_cvx_to_verts_fn)
        # if not os.path.exists(cvx_to_pts_load_fn):
        #   if self.cvx_dim > 0:
        #     cvx_to_pts_load_fn = os.path.join(self.dst_folder, cur_model + f"_manifold_cvx_to_verts_cdim_{self.cvx_dim}.npy")
        #   else:
        #     cvx_to_pts_load_fn = os.path.join(self.dst_folder, cur_model + f"_manifold_cvx_to_verts.npy")
        
        # if not os.path.exists(cvx_to_pts_load_fn):
        #   cvx_to_pts_load_fn = os.path.join(os.path.join(self.data_dir, self.cvx_folder_fn), cur_cvx_to_verts_fn)
          
        if not os.path.exists(cvx_to_pts_load_fn):
          # print(f"No existing cvx_to_pts_load_fn: {cvx_to_pts_load_fn}.")
          cur_scaled_sampled = cur_sampled / 2.
          cvx_to_pts = {
            0: cur_scaled_sampled
          }
        else:
          # cvx_to_pts = np.load(os.path.join(rt_folder, cur_cvx_to_verts_fn),allow_pickle=True).item()
          cvx_to_pts = np.load(cvx_to_pts_load_fn, allow_pickle=True).item()
        
        # # cvx_to_pts = np.load(os.path.join(rt_folder, cur_cvx_to_verts_fn),allow_pickle=True).item()
        # cvx_to_pts = np.load(cvx_to_pts_load_fn, allow_pickle=True).item()
        
        return cur_sampled, cur_keypoints, cur_surface, cur_faces, cur_weights_sampled_keypoints, cur_weights_mesh_keypoints, cvx_to_pts
    
    def get_src_dst_cvx_to_pts(self, src_cvx_to_pts, dst_cvx_to_pts, src_pc, tar_pc):
      src_cvx_to_pts_list = []
      dst_cvx_to_pts_list = []
      
      if self.only_tar:
        for cvx_idx in dst_cvx_to_pts:
          cur_dst_cvx_pts = dst_cvx_to_pts[cvx_idx]
          src_cvx_to_pts_list.append(np.reshape(cur_dst_cvx_pts, (1, cur_dst_cvx_pts.shape[0], 3)))
          dst_cvx_to_pts_list.append(np.reshape(cur_dst_cvx_pts, (1, cur_dst_cvx_pts.shape[0], 3)))
      else:
        for cvx_idx in src_cvx_to_pts:
          if cvx_idx in dst_cvx_to_pts:
            cur_src_cvx_pts, cur_dst_cvx_pts = src_cvx_to_pts[cvx_idx], dst_cvx_to_pts[cvx_idx]
            src_cvx_to_pts_list.append(np.reshape(cur_src_cvx_pts, (1, cur_src_cvx_pts.shape[0], 3)))
            dst_cvx_to_pts_list.append(np.reshape(cur_dst_cvx_pts, (1, cur_dst_cvx_pts.shape[0], 3)))
      if len(src_cvx_to_pts_list) == 0:
        src_cvx_to_pts_list = np.reshape(src_pc, (1, src_pc.shape[0], 3))
        dst_cvx_to_pts_list = np.reshape(tar_pc, (1, tar_pc.shape[0], 3))
      else:
        src_cvx_to_pts_list = np.concatenate(src_cvx_to_pts_list, axis=0)
        dst_cvx_to_pts_list = np.concatenate(dst_cvx_to_pts_list, axis=0)
      return src_cvx_to_pts_list, dst_cvx_to_pts_list
        
    
    def __len__(self):
        return self.dst_n_models
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
        
    
    def __getitem__(self, idx):
      
        # if self.use_paired_data:
        #   # idx = idx * self.dst_n_models + idx ### paried data
        #   src_idx = idx
        #   dst_idx = idx
        #   # print(f"src_idx: {src_idx}, dst_idx: {dst_idx}, src_model_fn: {self.src_models[src_idx]}, dst_model_fn: {self.dst_models[dst_idx]}")
        # else:
        self.src_folder = self.dst_folder


        idx = random.choice(range(self.src_n_models * self.dst_n_models))
      
      
        src_idx = idx // self.dst_n_models
        dst_idx = idx % self.dst_n_models
        ## seperate them into different parts ##
        # src_idx for src #
        if self.src_pc[0][src_idx] is None:
          for i_p in range(self.nn_parts): # load source model from model indicator #
            src_pc, src_key_pts, src_vertices, src_faces, src_w_pc, src_w_mesh, src_cvx_to_pts = self.load_data_from_model_indicator(self.src_folder[i_p], self.src_models[src_idx])
            self.src_pc[i_p][src_idx] = src_pc # i_p for src_pc #
            self.src_key_pts[i_p][src_idx] = src_key_pts
            self.src_mesh_vertices[i_p][src_idx] = src_vertices
            self.src_mesh_faces[i_p][src_idx] = src_faces
            self.src_w_pc[i_p][src_idx] = src_w_pc
            self.src_w_mesh[i_p][src_idx] = src_w_mesh
            self.src_cvx_to_pts[i_p][src_idx] = src_cvx_to_pts
        while not (self.src_w_pc[0][src_idx].shape[1] == self.n_keypoints and self.src_w_pc[0][src_idx].shape[0] == self.num_sampled_pts and self.src_w_pc[1][src_idx].shape[1] == self.n_keypoints and self.src_w_pc[1][src_idx].shape[0] == self.num_sampled_pts):
          src_idx = random.choice(range(len(self.src_pc[0])))
          for i_p in range(self.nn_parts):
            src_pc, src_key_pts, src_vertices, src_faces, src_w_pc, src_w_mesh, src_cvx_to_pts = self.load_data_from_model_indicator(self.src_folder[i_p], self.src_models[src_idx])
            self.src_pc[i_p][src_idx] = src_pc # i_p for src_pc #
            self.src_key_pts[i_p][src_idx] = src_key_pts
            self.src_mesh_vertices[i_p][src_idx] = src_vertices
            self.src_mesh_faces[i_p][src_idx] = src_faces
            self.src_w_pc[i_p][src_idx] = src_w_pc
            self.src_w_mesh[i_p][src_idx] = src_w_mesh
            self.src_cvx_to_pts[i_p][src_idx] = src_cvx_to_pts
          # if self.src_pc[0][src_idx] is None:
          #   src_pc, src_key_pts, src_vertices, src_faces, src_w_pc, src_w_mesh, src_cvx_to_pts = self.load_data_from_model_indicator(self.src_folder, self.src_models[src_idx])
          #   self.src_pc[src_idx] = src_pc
          #   self.src_key_pts[src_idx] = src_key_pts
          #   self.src_mesh_vertices[src_idx] = src_vertices
          #   self.src_mesh_faces[src_idx] = src_faces
          #   self.src_w_pc[src_idx] = src_w_pc
          #   self.src_w_mesh[src_idx] = src_w_mesh
          #   self.src_cvx_to_pts[src_idx] = src_cvx_to_pts
        
        # print(f"src_idx: {src_idx}, dst_idx: {dst_idx}")
        if self.dst_pc[0][dst_idx] is None: # dst_pc
          for i_p in range(self.nn_parts): # load data from model indicator
            tar_pc, dst_key_pts, dst_vertices, dst_faces, dst_w_pc, dst_w_mesh, dst_cvx_to_pts = self.load_data_from_model_indicator(self.dst_folder[i_p], self.dst_models[dst_idx])
            self.dst_pc[i_p][dst_idx] = tar_pc
            self.dst_key_pts[i_p][dst_idx] = dst_key_pts
            self.dst_mesh_vertices[i_p][dst_idx] = dst_vertices
            self.dst_mesh_faces[i_p][dst_idx] = dst_faces
            self.dst_w_pc[i_p][dst_idx] = dst_w_pc
            self.dst_w_mesh[i_p][dst_idx] = dst_w_mesh
            self.dst_cvx_to_pts[i_p][dst_idx] = dst_cvx_to_pts

        tot_src_pc = []
        tot_src_key_pts = []
        tot_src_vertices = []
        tot_src_faces = []
        tot_src_w_pc = []
        tot_src_w_mesh = []
        tot_src_cvx_to_pts = []
        
        tot_dst_pc = []
        tot_dst_key_pts = []
        tot_dst_vertices = []
        tot_dst_faces = []
        tot_dst_w_pc = []
        tot_dst_w_mesh = []
        tot_dst_cvx_to_pts = []
        
        for i_p in range(self.nn_parts):
        
          src_pc, src_key_pts, src_vertices, src_faces, src_w_pc, src_w_mesh, src_cvx_to_pts = self.src_pc[i_p][src_idx], self.src_key_pts[i_p][src_idx], self.src_mesh_vertices[i_p][src_idx], self.src_mesh_faces[i_p][src_idx], self.src_w_pc[i_p][src_idx], self.src_w_mesh[i_p][src_idx], self.src_cvx_to_pts[i_p][src_idx]
          
          tar_pc, dst_key_pts, dst_vertices, dst_faces, dst_w_pc, dst_w_mesh, dst_cvx_to_pts = self.dst_pc[i_p][dst_idx], self.dst_key_pts[i_p][dst_idx], self.dst_mesh_vertices[i_p][dst_idx], self.dst_mesh_faces[i_p][dst_idx], self.dst_w_pc[i_p][dst_idx], self.dst_w_mesh[i_p][dst_idx], self.dst_cvx_to_pts[i_p][dst_idx]

          tot_src_pc.append(src_pc)
          tot_src_key_pts.append(src_key_pts)
          tot_src_vertices.append(src_vertices)
          tot_src_faces.append(src_faces)
          tot_src_w_pc.append(src_w_pc)
          tot_src_w_mesh.append(src_w_mesh)
          tot_src_cvx_to_pts.append(src_cvx_to_pts)
          
          tot_dst_pc.append(tar_pc)
          tot_dst_key_pts.append(dst_key_pts)
          tot_dst_vertices.append(dst_vertices)
          tot_dst_faces.append(dst_faces)
          tot_dst_w_pc.append(dst_w_pc)
          tot_dst_w_mesh.append(dst_w_mesh)
          tot_dst_cvx_to_pts.append(dst_cvx_to_pts)
        
        # src_edges, src_dofs = dataset_utils.get_edges_from_faces(src_vertices, src_faces)
        # tar_edges, tar_dofs = dataset_utils.get_edges_from_faces(dst_vertices, dst_faces)
        
        src_name = f"src_{src_idx}"
        tar_name = f"dst_{dst_idx}"
        
        rt_dict = {
          "src_name": src_name, "tar_name": tar_name,
          "src_pc": tot_src_pc, "tar_pc": tot_dst_pc,
          "src_ver": tot_src_vertices, "src_face": tot_src_faces,
          "tar_ver": tot_dst_vertices, "tar_face": tot_dst_faces,
          "real_ver": tot_dst_vertices, "real_face": tot_dst_faces,
          "key_pts": tot_src_key_pts,
          "w_mesh": tot_src_w_mesh, "w_pc": tot_src_w_pc, 
        }

        #### dst_key_pts: n_key_pts x 3 ####
        return rt_dict





class ChairDataset_HumanBody(torch.utils.data.Dataset):
    def __init__(self, phase="train", data_dir="../data/chair", split=None, opt=None):
        super().__init__()
        self.data_dir = data_dir
        self.phase = phase
        self.opt = opt
        self.load_meta = self.opt.load_meta
        self.num_sampled_pts = 4096
        self.n_shots = self.opt.n_shots ### nn_shots for training ## n_shots
        # self.use_paired_data = self.opt.use_paired_data
        # self.random_scaling = self.opt.random_scaling
        
        self.n_keypoints = self.opt.n_keypoints # 
        
        self.src_folder_fn = self.opt.src_folder_fn ## src folder fn; dst folder fn ##
        self.dst_folder_fn = self.opt.dst_folder_fn
        self.cvx_dim = self.opt.cvx_dim
        self.only_tar = self.opt.only_tar
        self.split = split
        self.cvx_folder_fn = self.opt.cvx_folder_fn
        if len(self.cvx_folder_fn) == 0:
          self.cvx_folder_fn = self.dst_folder_fn
        
        # src_folder = os.path.join(self.data_dir, "src")
        # dst folder fn #
        # self.data_dir should be a list of data root directory #
        src_folders = []
        dst_folders = []
        for i_p, cur_data_dir in enumerate(self.data_dir):
          # cur_src_folder = os.path.join(cur_data_dir, self.src_folder_fn if os.path.exists(os.path.join(cur_data_dir, self.src_folder_fn)) else "src")
          # cur_dst_folder = os.path.join(cur_data_dir, self.dst_folder_fn if os.path.exists(os.path.join(cur_data_dir, self.dst_folder_fn)) else "dst")
          ### get cur folder; cur data dir ###
          cur_src_folder = cur_data_dir
          cur_dst_folder = cur_data_dir
          src_folders.append(cur_src_folder) # src folders #
          dst_folders.append(cur_dst_folder) # dst folders #
          # src_folders.append()
        
        
        #### if we can use any shape as the source shape for deformation? ####
        # if not self.use_paired_data: ### not use paired data...
        #     src_folder = os.path.join(self.data_dir, "dst")
        # else:
        #     src_folder = os.path.join(self.data_dir, "src")
        
        ### dst_folder_fn ###
        # src_folder = os.path.join(self.data_dir, "src")
        # src_folder = os.path.join(self.data_dir, self.src_folder_fn if os.path.exists(os.path.join(self.data_dir, self.src_folder_fn)) else "src")
        # # dst_folder = os.path.join(self.data_dir, "dst")
        # dst_folder = os.path.join(self.data_dir, self.dst_folder_fn if os.path.exists(os.path.join(self.data_dir, self.dst_folder_fn)) else "dst")
        
        self.src_folders = src_folders ## src_folders ##
        self.dst_folders = dst_folders ## dst_folders ##
        self.src_folder =src_folders
        self.dst_folder =dst_folders
        self.cvx_to_pts_sufix = opt.cvx_to_pts_sufix
        
        # with open(os.path.join(self.data_dir, "all.txt")) as f:
        #     lines = f.readlines()
        #     self.models = [line.rstrip() for line in lines]

        # with open(os.path.join(self.src_folder, "all.txt"), "r") as rf:
        #   lines = rf.readlines()
        #   self.src_models = [line.rstrip() for line in lines]
        
        # with open(os.path.join(self.dst_folder, "all.txt"), "r") as rf:
        #   lines = rf.readlines()
        #   self.dst_models = [line.rstrip() for line in lines]

        ### keypts sufix ### ## keypts_sufix for the sufix ##
        self.keypts_sufix = f"_manifold_tetra.mesh__sf.keypoints_{self.n_keypoints}.weights.sampled_4096.txt"
        if not os.path.exists(os.path.join(self.data_dir[0], "all.txt")):
          # dst_models #
          # self.models = os.listdir(os.path.join(self.data_dir[0], self.dst_folder_fn))
          self.models = os.listdir(self.data_dir[0])
          self.models = [fn[: -len(self.keypts_sufix)] for fn in self.models if fn.endswith(self.keypts_sufix)] # two points ##
          for cur_data_dir in self.data_dir[1:]:
            cur_data_models = os.listdir(
              os.path.join(cur_data_dir, self.dst_folder_fn)
            )
            cur_data_models = [fn[: -len(self.keypts_sufix)] for fn in cur_data_models if fn.endswith(self.keypts_sufix)]
            self.models = [fn for fn in self.models if fn in cur_data_models]
          # self.models = os.listdir(os.path.join(self.data_dir, self.dst_folder_fn))
          # self.models = [fn[: -len(self.keypts_sufix)] for fn in self.models if fn.endswith(self.keypts_sufix)] # two points ##
          self.dst_models = self.models ### src models ###
          self.src_models = self.models ### dst models ###
        else: # dst_models
          with open(os.path.join(self.src_folder, "all.txt"), "r") as rf:
            lines = rf.readlines()
            self.src_models = [line.rstrip() for line in lines]
          
          with open(os.path.join(self.dst_folder, "all.txt"), "r") as rf:
            lines = rf.readlines()
            self.dst_models = [line.rstrip() for line in lines]
        
        ### src models; dst_models ###
        print(f"Meta-infos loaded with src_models: {len(self.src_models)}, dst_models: {len(self.dst_models)}")
        
        
        # if self.split == "train":
        #   # train_nns = 36
        #   # train_nns = 5
        #   # self.src_models = self.src_models[:train_nns]
        #   # self.dst_models = self.dst_models[:train_nns]
        #   self.src_models = self.src_models[:self.n_shots]
        #   self.dst_models = self.dst_models[:self.n_shots]
        # else:
        #   train_nns = 5
        #   val_nns = 360
        #   val_nns = 1000
          
        #   val_nns = min(val_nns, len(self.src_models) - self.n_shots) 
          
        #   # self.src_models = self.src_models[train_nns: train_nns + val_nns]
        #   # self.dst_models = self.dst_models[train_nns: train_nns + val_nns]
          
        #   self.src_models = self.src_models[-val_nns: ] # trai
        #   self.dst_models = self.dst_models[-val_nns: ]

        ### src_models and dst_models ##
        self.src_models = self.src_models
        self.dst_models = self.dst_models

        print(f"Meta-infos loaded with src_models: {len(self.src_models)}, dst_models: {len(self.dst_models)}")
        
        # self.valid_models = {model_fn: 1 for model_fn in self.src_models if model_fn in self.dst_models}
        
        # self.src_models = [model_fn for model_fn in self.src_models if model_fn in self.valid_models]
        # self.dst_models = [model_fn for model_fn in self.dst_models if model_fn in self.valid_models]
        
        
        self.cvx_to_verts_sufix="_manifold_cvx_to_verts.npy"
        self.surface_pts_sufix="_manifold_tetra.mesh__sf.sampled_4096.obj"
        
        
        # self.dst_models = [fn for fn in self.dst_models if os.path.exists(os.path.join(self.dst_folder, fn + f"_manifold_tetra.mesh__sf.keypoints_{self.n_keypoints}.weights.sampled_4096.txt"))]
        # self.src_models = self.dst_models
        
        
        
        # self.dst_folder_path = os.path.join(self.data_dir, self.dst_folder_fn)
        # self.dst_models = os.listdir(self.dst_folder_path)
        # self.dst_models = [fn for fn in self.dst_models if fn.endswith(self.surface_pts_sufix)]
        # self.dst_models = [fn[: len(fn) - len(self.surface_pts_sufix)] for fn in self.dst_models]
        # self.src_models = self.dst_models
        
        
        
        
        # if opt.few_shot:
        #   tot_n_src_models = 10
        #   if self.split == "train":
        #     self.src_models = self.src_models[: tot_n_src_models]
        #     self.dst_models = self.dst_models[: tot_n_src_models]
        #   else:
        #     self.src_models = self.src_models[tot_n_src_models: ]
        #     self.dst_models = self.dst_models[tot_n_src_models: ]
        # else:
        #   if self.split is not None:
        #     if self.split == "train":
        #       tot_n_src_models = int(len(self.src_models) * 0.9)
        #       self.src_models = self.src_models[: tot_n_src_models]
        #       self.dst_models = self.dst_models[: tot_n_src_models]
        #     elif self.split == "test":
        #       tot_n_src_models = int(len(self.src_models) * 0.9)
        #       self.src_models = self.src_models[tot_n_src_models: ]
        #       self.dst_models = self.dst_models[tot_n_src_models: ]
        #     else:
        #       raise ValueError(f"Unrecognized split: {self.split}")
          
        
        ### src n models ###
        self.src_n_models = len(self.src_models)
        self.dst_n_models = len(self.dst_models)
        ### keypoint 256: gt_step_0_subd_0_manifold_tetra.mesh__sf.keypoints_256 
        ### sampled 4096: gt_step_0_subd_0_manifold_tetra.mesh__sf.sampled_4096
        ### surface points: gt_step_0_subd_0_manifold_tetra.mesh__sf
        ### weights sampled 4096 -> 256: gt_step_0_subd_0_manifold_tetra.mesh__sf.keypoints_256.weights.sampled_4096
        ### weights tot pts -> 256: gt_step_0_subd_0_manifold_tetra.mesh__sf.keypoints_256.weights

        ##### source pc, key_pts, mesh_vertices, mesh_faces, w_pc, w_mesh #####
        nn_parts = len(self.data_dir)
        self.nn_parts = nn_parts
        
        self.src_pc = [[None for _ in range(self.src_n_models)] for _ in range(nn_parts)]
        self.src_key_pts = [[None for _ in range(self.src_n_models)] for _ in range(nn_parts)]
        self.src_mesh_vertices = [[None for _ in range(self.src_n_models)] for _ in range(nn_parts)]
        self.src_mesh_faces = [[None for _ in range(self.src_n_models)] for _ in range(nn_parts)]
        self.src_w_pc = [[None for _ in range(self.src_n_models)] for _ in range(nn_parts)]
        self.src_w_mesh = [[None for _ in range(self.src_n_models)] for _ in range(nn_parts)]
        self.src_cvx_to_pts = [[None for _ in range(self.src_n_models)] for _ in range(nn_parts)]
        
        ##### source pc, key_pts, mesh_vertices, mesh_faces, w_pc, w_mesh #####
        self.dst_pc = [[None for _ in range(self.dst_n_models)] for _ in range(nn_parts)]
        self.dst_key_pts = [[None for _ in range(self.dst_n_models)] for _ in range(nn_parts)]
        self.dst_mesh_vertices = [[None for _ in range(self.dst_n_models)] for _ in range(nn_parts)]
        self.dst_mesh_faces = [[None for _ in range(self.dst_n_models)] for _ in range(nn_parts)]
        self.dst_w_pc = [[None for _ in range(self.dst_n_models)] for _ in range(nn_parts)]
        self.dst_w_mesh = [[None for _ in range(self.dst_n_models)] for _ in range(nn_parts)]
        self.dst_cvx_to_pts = [[None for _ in range(self.dst_n_models)] for _ in range(nn_parts)]
        
    def load_data_from_model_indicator(self, rt_folder, model_indicator):
        cur_model = model_indicator
        
        cur_keypoints_fn = cur_model + f"_manifold_tetra.mesh__sf.keypoints_{self.n_keypoints}.obj"
        cur_sampled_pts_fn = cur_model + "_manifold_tetra.mesh__sf.sampled_4096.obj"
        cur_surface_pts_fn = cur_model + "_manifold_tetra.mesh__sf.obj"
        cur_weights_sampled_fn = cur_model + f"_manifold_tetra.mesh__sf.keypoints_{self.n_keypoints}.weights.sampled_4096.txt"
        cur_weights_tot_fn = cur_model + f"_manifold_tetra.mesh__sf.keypoints_{self.n_keypoints}.weights.txt"
        
        # if self.cvx_dim > 0:
        #   cur_cvx_to_verts_fn = cur_model + f'_cvx_to_verts_cdim_{self.cvx_dim}.npy'
        # else:
        #   cur_cvx_to_verts_fn = cur_model + "_cvx_to_verts.npy"
          
        # cur_cvx_to_verts_fn = cur_model + self.cvx_to_pts_sufix
        
        # print(f"loading from {os.path.join(rt_folder, cur_cvx_to_verts_fn)}")
        cur_keypoints, _ =  utils.read_obj_file_ours(os.path.join(rt_folder, cur_keypoints_fn))
        cur_sampled, _ = utils.read_obj_file_ours(os.path.join(rt_folder, cur_sampled_pts_fn))
        cur_surface, cur_faces = utils.read_obj_file_ours(os.path.join(rt_folder, cur_surface_pts_fn), sub_one=True)
        cur_weights_sampled_keypoints = utils.load_txt(os.path.join(rt_folder, cur_weights_sampled_fn))
        cur_weights_mesh_keypoints = utils.load_txt(os.path.join(rt_folder, cur_weights_tot_fn))
        cur_faces = np.array(cur_faces, dtype=np.long) # .long()
        
        
        # cur_cvx_to_verts_fn = cur_model + "_cvx_to_verts.npy"
        # cvx_to_pts_load_fn = os.path.join(rt_folder, cur_cvx_to_verts_fn)
        # print(f"cur_cvx_to_verts_fn: {cvx_to_pts_load_fn}")
        # if not os.path.exists(cvx_to_pts_load_fn):
        #   cvx_to_pts_load_fn = os.path.join(self.dst_folder, cur_cvx_to_verts_fn)
        # if not os.path.exists(cvx_to_pts_load_fn):
        #   if self.cvx_dim > 0:
        #     cvx_to_pts_load_fn = os.path.join(self.dst_folder, cur_model + f"_manifold_cvx_to_verts_cdim_{self.cvx_dim}.npy")
        #   else:
        #     cvx_to_pts_load_fn = os.path.join(self.dst_folder, cur_model + f"_manifold_cvx_to_verts.npy")
        
        # if not os.path.exists(cvx_to_pts_load_fn):
        #   cvx_to_pts_load_fn = os.path.join(os.path.join(self.data_dir, self.cvx_folder_fn), cur_cvx_to_verts_fn)
          
        # if not os.path.exists(cvx_to_pts_load_fn):
        #   # print(f"No existing cvx_to_pts_load_fn: {cvx_to_pts_load_fn}.")
        #   cur_scaled_sampled = cur_sampled / 2.
        #   cvx_to_pts = {
        #     0: cur_scaled_sampled
        #   }
        # else:
        #   # cvx_to_pts = np.load(os.path.join(rt_folder, cur_cvx_to_verts_fn),allow_pickle=True).item()
        #   cvx_to_pts = np.load(cvx_to_pts_load_fn, allow_pickle=True).item()
        
        # # cvx_to_pts = np.load(os.path.join(rt_folder, cur_cvx_to_verts_fn),allow_pickle=True).item()
        # cvx_to_pts = np.load(cvx_to_pts_load_fn, allow_pickle=True).item()
        
        return cur_sampled, cur_keypoints, cur_surface, cur_faces, cur_weights_sampled_keypoints, cur_weights_mesh_keypoints, None
    
    def get_src_dst_cvx_to_pts(self, src_cvx_to_pts, dst_cvx_to_pts, src_pc, tar_pc):
      src_cvx_to_pts_list = []
      dst_cvx_to_pts_list = []
      
      if self.only_tar:
        for cvx_idx in dst_cvx_to_pts:
          cur_dst_cvx_pts = dst_cvx_to_pts[cvx_idx]
          src_cvx_to_pts_list.append(np.reshape(cur_dst_cvx_pts, (1, cur_dst_cvx_pts.shape[0], 3)))
          dst_cvx_to_pts_list.append(np.reshape(cur_dst_cvx_pts, (1, cur_dst_cvx_pts.shape[0], 3)))
      else:
        for cvx_idx in src_cvx_to_pts:
          if cvx_idx in dst_cvx_to_pts:
            cur_src_cvx_pts, cur_dst_cvx_pts = src_cvx_to_pts[cvx_idx], dst_cvx_to_pts[cvx_idx]
            src_cvx_to_pts_list.append(np.reshape(cur_src_cvx_pts, (1, cur_src_cvx_pts.shape[0], 3)))
            dst_cvx_to_pts_list.append(np.reshape(cur_dst_cvx_pts, (1, cur_dst_cvx_pts.shape[0], 3)))
      if len(src_cvx_to_pts_list) == 0:
        src_cvx_to_pts_list = np.reshape(src_pc, (1, src_pc.shape[0], 3))
        dst_cvx_to_pts_list = np.reshape(tar_pc, (1, tar_pc.shape[0], 3))
      else:
        src_cvx_to_pts_list = np.concatenate(src_cvx_to_pts_list, axis=0)
        dst_cvx_to_pts_list = np.concatenate(dst_cvx_to_pts_list, axis=0)
      return src_cvx_to_pts_list, dst_cvx_to_pts_list
        
    
    def __len__(self):
        return self.dst_n_models
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
        
    
    def __getitem__(self, idx):
      
        # if self.use_paired_data:
        #   # idx = idx * self.dst_n_models + idx ### paried data
        #   src_idx = idx
        #   dst_idx = idx
        #   # print(f"src_idx: {src_idx}, dst_idx: {dst_idx}, src_model_fn: {self.src_models[src_idx]}, dst_model_fn: {self.dst_models[dst_idx]}")
        # else:
        self.src_folder = self.dst_folder


        idx = random.choice(range(self.src_n_models * self.dst_n_models))
      
      
        src_idx = idx // self.dst_n_models
        dst_idx = idx % self.dst_n_models
        ## seperate them into different parts ##
        # src_idx for src #
        if self.src_pc[0][src_idx] is None:
          for i_p in range(self.nn_parts): # load source model from model indicator #
            src_pc, src_key_pts, src_vertices, src_faces, src_w_pc, src_w_mesh, src_cvx_to_pts = self.load_data_from_model_indicator(self.src_folder[i_p], self.src_models[src_idx])
            self.src_pc[i_p][src_idx] = src_pc # i_p for src_pc #
            self.src_key_pts[i_p][src_idx] = src_key_pts
            self.src_mesh_vertices[i_p][src_idx] = src_vertices
            self.src_mesh_faces[i_p][src_idx] = src_faces
            self.src_w_pc[i_p][src_idx] = src_w_pc
            self.src_w_mesh[i_p][src_idx] = src_w_mesh
            self.src_cvx_to_pts[i_p][src_idx] = src_cvx_to_pts
        while not (self.src_w_pc[0][src_idx].shape[1] == self.n_keypoints and self.src_w_pc[0][src_idx].shape[0] == self.num_sampled_pts):
          src_idx = random.choice(range(len(self.src_pc[0])))
          for i_p in range(self.nn_parts):
            src_pc, src_key_pts, src_vertices, src_faces, src_w_pc, src_w_mesh, src_cvx_to_pts = self.load_data_from_model_indicator(self.src_folder[i_p], self.src_models[src_idx])
            self.src_pc[i_p][src_idx] = src_pc # i_p for src_pc #
            self.src_key_pts[i_p][src_idx] = src_key_pts
            self.src_mesh_vertices[i_p][src_idx] = src_vertices
            self.src_mesh_faces[i_p][src_idx] = src_faces
            self.src_w_pc[i_p][src_idx] = src_w_pc
            self.src_w_mesh[i_p][src_idx] = src_w_mesh
            self.src_cvx_to_pts[i_p][src_idx] = src_cvx_to_pts
          # if self.src_pc[0][src_idx] is None:
          #   src_pc, src_key_pts, src_vertices, src_faces, src_w_pc, src_w_mesh, src_cvx_to_pts = self.load_data_from_model_indicator(self.src_folder, self.src_models[src_idx])
          #   self.src_pc[src_idx] = src_pc
          #   self.src_key_pts[src_idx] = src_key_pts
          #   self.src_mesh_vertices[src_idx] = src_vertices
          #   self.src_mesh_faces[src_idx] = src_faces
          #   self.src_w_pc[src_idx] = src_w_pc
          #   self.src_w_mesh[src_idx] = src_w_mesh
          #   self.src_cvx_to_pts[src_idx] = src_cvx_to_pts
        
        # print(f"src_idx: {src_idx}, dst_idx: {dst_idx}")
        if self.dst_pc[0][dst_idx] is None: # dst_pc
          for i_p in range(self.nn_parts): # load data from model indicator
            tar_pc, dst_key_pts, dst_vertices, dst_faces, dst_w_pc, dst_w_mesh, dst_cvx_to_pts = self.load_data_from_model_indicator(self.dst_folder[i_p], self.dst_models[dst_idx])
            self.dst_pc[i_p][dst_idx] = tar_pc
            self.dst_key_pts[i_p][dst_idx] = dst_key_pts
            self.dst_mesh_vertices[i_p][dst_idx] = dst_vertices
            self.dst_mesh_faces[i_p][dst_idx] = dst_faces
            self.dst_w_pc[i_p][dst_idx] = dst_w_pc
            self.dst_w_mesh[i_p][dst_idx] = dst_w_mesh
            self.dst_cvx_to_pts[i_p][dst_idx] = dst_cvx_to_pts

        tot_src_pc = []
        tot_src_key_pts = []
        tot_src_vertices = []
        tot_src_faces = []
        tot_src_w_pc = []
        tot_src_w_mesh = []
        tot_src_cvx_to_pts = []
        
        tot_dst_pc = []
        tot_dst_key_pts = []
        tot_dst_vertices = []
        tot_dst_faces = []
        tot_dst_w_pc = []
        tot_dst_w_mesh = []
        tot_dst_cvx_to_pts = []
        
        for i_p in range(self.nn_parts):
        
          src_pc, src_key_pts, src_vertices, src_faces, src_w_pc, src_w_mesh, src_cvx_to_pts = self.src_pc[i_p][src_idx], self.src_key_pts[i_p][src_idx], self.src_mesh_vertices[i_p][src_idx], self.src_mesh_faces[i_p][src_idx], self.src_w_pc[i_p][src_idx], self.src_w_mesh[i_p][src_idx], self.src_cvx_to_pts[i_p][src_idx]
          
          tar_pc, dst_key_pts, dst_vertices, dst_faces, dst_w_pc, dst_w_mesh, dst_cvx_to_pts = self.dst_pc[i_p][dst_idx], self.dst_key_pts[i_p][dst_idx], self.dst_mesh_vertices[i_p][dst_idx], self.dst_mesh_faces[i_p][dst_idx], self.dst_w_pc[i_p][dst_idx], self.dst_w_mesh[i_p][dst_idx], self.dst_cvx_to_pts[i_p][dst_idx]

          tot_src_pc.append(src_pc)
          tot_src_key_pts.append(src_key_pts)
          tot_src_vertices.append(src_vertices)
          tot_src_faces.append(src_faces)
          tot_src_w_pc.append(src_w_pc)
          tot_src_w_mesh.append(src_w_mesh)
          # tot_src_cvx_to_pts.append(src_cvx_to_pts)
          
          tot_dst_pc.append(tar_pc)
          tot_dst_key_pts.append(dst_key_pts)
          tot_dst_vertices.append(dst_vertices)
          tot_dst_faces.append(dst_faces)
          tot_dst_w_pc.append(dst_w_pc)
          tot_dst_w_mesh.append(dst_w_mesh)
          # tot_dst_cvx_to_pts.append(dst_cvx_to_pts)
        
        # src_edges, src_dofs = dataset_utils.get_edges_from_faces(src_vertices, src_faces)
        # tar_edges, tar_dofs = dataset_utils.get_edges_from_faces(dst_vertices, dst_faces)
        
        src_name = f"src_{src_idx}"
        tar_name = f"dst_{dst_idx}"
        
        rt_dict = {
          "src_name": src_name, "tar_name": tar_name,
          "src_pc": tot_src_pc, "tar_pc": tot_dst_pc,
          "src_ver": tot_src_vertices, "src_face": tot_src_faces,
          "tar_ver": tot_dst_vertices, "tar_face": tot_dst_faces,
          "real_ver": tot_dst_vertices, "real_face": tot_dst_faces,
          "key_pts": tot_src_key_pts,
          "w_mesh": tot_src_w_mesh, "w_pc": tot_src_w_pc, 
        }

        #### dst_key_pts: n_key_pts x 3 ####
        return rt_dict
