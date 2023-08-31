from ctypes import util
import torch
import numpy as np
import os
import random
import utils
from common_utils import dataset_utils

### collect data from the dataset ###
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
    src_edges = []
    src_dofs = []
    tar_edges = []
    tar_dofs = []
    src_cvx_to_pts = []
    dst_cvx_to_pts = []
    for data in batch: # "src_edges": src_edges, "src_dofs": src_dofs
        src_pc.append(torch.from_numpy(data["src_pc"]).unsqueeze(0).float())
        src_name.append(data["src_name"])
        tar_pc.append(torch.from_numpy(data["tar_pc"]).unsqueeze(0).float())
        tar_name.append(data["tar_name"])
        key_pts.append(torch.from_numpy(data["key_pts"]).unsqueeze(0).float())
        w_pc.append(torch.from_numpy(data["w_pc"]).unsqueeze(0).float())
        
        src_ver.append(torch.from_numpy(data["src_ver"]).float().cuda())
        
        
        
        src_face.append(torch.from_numpy(data["src_face"]))
        
        tar_ver.append(torch.from_numpy(data["tar_ver"]).float().cuda())
        
        tar_face.append(torch.from_numpy(data["tar_face"]))
        real_ver.append(torch.from_numpy(data["real_ver"]))
        real_face.append(torch.from_numpy(data["real_face"]))
        w_mesh.append(torch.from_numpy(data["w_mesh"]).float())
        
        src_edges.append(torch.from_numpy(data['src_edges']).long().unsqueeze(0).cuda()) #### src_edges
        src_dofs.append(torch.from_numpy(data['src_dofs']).float().unsqueeze(0).cuda()) #### vert_dofs 
        
        tar_edges.append(torch.from_numpy(data['tar_edges']).long().unsqueeze(0).cuda()) #### src_edges
        tar_dofs.append(torch.from_numpy(data['tar_dofs']).float().unsqueeze(0).cuda()) #### vert_dofs 
        
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
    
    return {"src_pc": src_pc, "src_name": src_name, "key_pts": key_pts,
            "tar_pc": tar_pc, "tar_name": tar_name, "w_pc": w_pc,
            "src_ver": src_ver, "src_face": src_face, "w_mesh": w_mesh,
            "tar_ver": tar_ver, "tar_face": tar_face,
            "real_ver": real_ver, "real_face": real_face,
            "src_edges": src_edges, "src_dofs": src_dofs,
            "tar_edges": tar_edges, "tar_dofs": tar_dofs, "dst_key_pts": dst_key_pts, 
            "src_cvx_to_pts": src_cvx_to_pts, "dst_cvx_to_pts": dst_cvx_to_pts
            }


class ChairDataset(torch.utils.data.Dataset):
    def __init__(self, phase="train", data_dir="../data/chair", split=None, opt=None):
        super().__init__()
        self.data_dir = data_dir
        self.phase = phase
        self.opt = opt
        self.load_meta = self.opt.load_meta
        self.num_sampled_pts = 4096
        self.use_paired_data = self.opt.use_paired_data
        self.random_scaling = self.opt.random_scaling
        
        self.n_keypoints = self.opt.n_keypoints
        
        self.src_folder_fn = self.opt.src_folder_fn
        self.dst_folder_fn = self.opt.dst_folder_fn
        self.cvx_dim = self.opt.cvx_dim
        self.only_tar = self.opt.only_tar
        
        # src_folder = os.path.join(self.data_dir, "src")
        
        #### if we can use any shape as the source shape for deformation? ####
        # if not self.use_paired_data: ### not use paired data...
        #     src_folder = os.path.join(self.data_dir, "dst")
        # else:
        #     src_folder = os.path.join(self.data_dir, "src")
        
        # src_folder = os.path.join(self.data_dir, "src")
        src_folder = os.path.join(self.data_dir, self.src_folder_fn if os.path.exists(os.path.join(self.data_dir, self.src_folder_fn)) else "src")
        # dst_folder = os.path.join(self.data_dir, "dst")
        dst_folder = os.path.join(self.data_dir, self.dst_folder_fn if os.path.exists(os.path.join(self.data_dir, self.dst_folder_fn)) else "dst")
        
        self.src_folder = src_folder
        self.dst_folder = dst_folder
        
        
        # if opt.debug:
          
        
        # with open(os.path.join(self.data_dir, "all.txt")) as f:
        #     lines = f.readlines()
        #     self.models = [line.rstrip() for line in lines]
            
        with open(os.path.join(self.src_folder, "all.txt"), "r") as rf:
          lines = rf.readlines()
          self.src_models = [line.rstrip() for line in lines]
        
        with open(os.path.join(self.dst_folder, "all.txt"), "r") as rf:
          lines = rf.readlines()
          self.dst_models = [line.rstrip() for line in lines]
        
        print(f"Meta-infos loaded with src_models: {len(self.src_models)}, dst_models: {len(self.dst_models)}")
        
        self.valid_models = {model_fn: 1 for model_fn in self.src_models if model_fn in self.dst_models}
        
        self.src_models = [model_fn for model_fn in self.src_models if model_fn in self.valid_models]
        self.dst_models = [model_fn for model_fn in self.dst_models if model_fn in self.valid_models]
        
        if opt.debug:
          self.src_models = self.src_models[:2]
          self.dst_models = self.dst_models[:2]
        
        
        
        self.split = split
        if self.split is not None:
          if self.split == "train":
            tot_n_src_models = int(len(self.src_models) * 0.9)
            self.src_models = self.src_models[: tot_n_src_models]
            self.dst_models = self.dst_models[: tot_n_src_models]
          elif self.split == "test":
            tot_n_src_models = int(len(self.src_models) * 0.9)
            self.src_models = self.src_models[tot_n_src_models: ]
            self.dst_models = self.dst_models[tot_n_src_models: ]
          else:
            raise ValueError(f"Unrecognized split: {self.split}")
          
        
        
        self.src_n_models = len(self.src_models)
        self.dst_n_models = len(self.dst_models)
        ### keypoint 256: gt_step_0_subd_0_manifold_tetra.mesh__sf.keypoints_256 
        ### sampled 4096: gt_step_0_subd_0_manifold_tetra.mesh__sf.sampled_4096
        ### surface points: gt_step_0_subd_0_manifold_tetra.mesh__sf
        ### weights sampled 4096 -> 256: gt_step_0_subd_0_manifold_tetra.mesh__sf.keypoints_256.weights.sampled_4096
        ### weights tot pts -> 256: gt_step_0_subd_0_manifold_tetra.mesh__sf.keypoints_256.weights

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
        if not os.path.exists(cvx_to_pts_load_fn):
          cvx_to_pts_load_fn = os.path.join(self.dst_folder, cur_cvx_to_verts_fn)
        if not os.path.exists(cvx_to_pts_load_fn):
          if self.cvx_dim > 0:
            cvx_to_pts_load_fn = os.path.join(self.dst_folder, cur_model + f"_manifold_cvx_to_verts_cdim_{self.cvx_dim}.npy")
          else:
            cvx_to_pts_load_fn = os.path.join(self.dst_folder, cur_model + f"_manifold_cvx_to_verts.npy")
        
        # cvx_to_pts = np.load(os.path.join(rt_folder, cur_cvx_to_verts_fn),allow_pickle=True).item()
        cvx_to_pts = np.load(cvx_to_pts_load_fn, allow_pickle=True).item()
        
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
      
        if self.use_paired_data:
          # idx = idx * self.dst_n_models + idx ### paried data
          src_idx = idx
          dst_idx = idx
          # print(f"src_idx: {src_idx}, dst_idx: {dst_idx}, src_model_fn: {self.src_models[src_idx]}, dst_model_fn: {self.dst_models[dst_idx]}")
        else:
          
          idx = random.choice(range(self.src_n_models * self.dst_n_models))
        
        
          src_idx = idx // self.dst_n_models
          dst_idx = idx % self.dst_n_models
        
        if self.src_pc[src_idx] is None:
          src_pc, src_key_pts, src_vertices, src_faces, src_w_pc, src_w_mesh, src_cvx_to_pts = self.load_data_from_model_indicator(self.src_folder, self.src_models[src_idx])
          self.src_pc[src_idx] = src_pc
          self.src_key_pts[src_idx] = src_key_pts
          self.src_mesh_vertices[src_idx] = src_vertices
          self.src_mesh_faces[src_idx] = src_faces
          self.src_w_pc[src_idx] = src_w_pc
          self.src_w_mesh[src_idx] = src_w_mesh
          self.src_cvx_to_pts[src_idx] = src_cvx_to_pts
        while not (self.src_w_pc[src_idx].shape[1] == self.n_keypoints and self.src_w_pc[src_idx].shape[0] == self.num_sampled_pts):
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
        if self.dst_pc[dst_idx] is None:
          tar_pc, dst_key_pts, dst_vertices, dst_faces, dst_w_pc, dst_w_mesh, dst_cvx_to_pts = self.load_data_from_model_indicator(self.dst_folder, self.dst_models[dst_idx])
          self.dst_pc[dst_idx] = tar_pc
          self.dst_key_pts[dst_idx] = dst_key_pts
          self.dst_mesh_vertices[dst_idx] = dst_vertices
          self.dst_mesh_faces[dst_idx] = dst_faces
          self.dst_w_pc[dst_idx] = dst_w_pc
          self.dst_w_mesh[dst_idx] = dst_w_mesh
          self.dst_cvx_to_pts[dst_idx] = dst_cvx_to_pts

        src_pc, src_key_pts, src_vertices, src_faces, src_w_pc, src_w_mesh, src_cvx_to_pts = self.src_pc[src_idx], self.src_key_pts[src_idx], self.src_mesh_vertices[src_idx], self.src_mesh_faces[src_idx], self.src_w_pc[src_idx], self.src_w_mesh[src_idx], self.src_cvx_to_pts[src_idx]
        
        tar_pc, dst_key_pts, dst_vertices, dst_faces, dst_w_pc, dst_w_mesh, dst_cvx_to_pts = self.dst_pc[dst_idx], self.dst_key_pts[dst_idx], self.dst_mesh_vertices[dst_idx], self.dst_mesh_faces[dst_idx], self.dst_w_pc[dst_idx], self.dst_w_mesh[dst_idx], self.dst_cvx_to_pts[dst_idx]
        
        src_edges, src_dofs = dataset_utils.get_edges_from_faces(src_vertices, src_faces)
        tar_edges, tar_dofs = dataset_utils.get_edges_from_faces(dst_vertices, dst_faces)
        
        
        src_cvx_to_pts_list, dst_cvx_to_pts_list = self.get_src_dst_cvx_to_pts(src_cvx_to_pts, dst_cvx_to_pts, src_pc, tar_pc)
        
        if self.random_scaling:
            dst_vertices, tar_pc, dst_key_pts, dst_cvx_to_pts_list = self.apply_random_scaling(dst_vertices, tar_pc, dst_key_pts, dst_cvx_to_pts_list)
        
        src_name = f"src_{src_idx}"
        tar_name = f"dst_{dst_idx}"

        #### dst_key_pts: n_key_pts x 3 ####
        return {"src_name": src_name, "tar_name": tar_name,
                "src_pc": tar_pc, "tar_pc": tar_pc,
                "src_ver": dst_vertices, "src_face": src_faces,
                "tar_ver": dst_vertices, "tar_face": dst_faces,
                "real_ver": dst_vertices, "real_face": dst_faces,
                "key_pts": dst_key_pts,
                # "w_mesh": src_w_mesh, "w_pc": src_w_pc, 
                "w_mesh": dst_w_mesh, "w_pc": dst_w_pc, ## use targets...
                "src_edges": src_edges, "src_dofs": src_dofs,
                "tar_edges": tar_edges, "tar_dofs": tar_dofs, "dst_key_pts": dst_key_pts, 
                "src_cvx_to_pts": dst_cvx_to_pts_list, "dst_cvx_to_pts": dst_cvx_to_pts_list
                }
