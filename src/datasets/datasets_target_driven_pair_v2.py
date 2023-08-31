from ctypes import util
import torch
import numpy as np
import os
import random
import utils
from common_utils import dataset_utils

### collect data from the dataset ###

n_parts = 2
def my_collate(batch):
    # src_pc = []
    # src_name = []
    # key_pts = []
    # tar_pc = []
    # tar_name = []
    # w_pc = []
    # src_ver = []
    # src_face = []
    # tar_ver = []
    # tar_face = []
    # real_ver = []
    # real_face = []
    # w_mesh = []
    # src_edges = []
    # src_dofs = []
    # tar_edges = []
    # tar_dofs = []
    
    src_pc = [[] for _ in range(n_parts)]
    src_name = [[] for _ in range(n_parts)]
    tar_pc = [[ ] for _ in range(n_parts)]
    tar_name = [[ ] for _ in range(n_parts)]
    key_pts = [[ ] for _ in range(n_parts)]
    w_pc = [[ ] for _ in range(n_parts)]
    src_ver = [[ ] for _ in range(n_parts)]
    src_face = [[ ] for _ in range(n_parts)]
    tar_ver = [[ ] for _ in range(n_parts)]
    tar_face = [[ ] for _ in range(n_parts)]
    real_ver = [[ ] for _ in range(n_parts)]
    real_face = [[ ] for _ in range(n_parts)]
    w_mesh = [[ ] for _ in range(n_parts)]
    src_edges = [[ ] for _ in range(n_parts)]
    src_dofs = [[ ] for _ in range(n_parts)]
    tar_edges = [[ ] for _ in range(n_parts)]
    tar_dofs = [[ ] for _ in range(n_parts)]
    # tar_pc = [[ ] for _ in range(n_parts)]
    # tar_pc = [[ ] for _ in range(n_parts)]
    
    for data in batch: 
      # "src_edges": src_edges, "src_dofs": src_dofs
      ### src_pc
        for i_p in range(n_parts):
        
          src_pc[i_p].append(torch.from_numpy(data["src_pc"][i_p]).unsqueeze(0).float())
          src_name[i_p].append(data["src_name"][i_p])
          
          tar_pc[i_p].append(torch.from_numpy(data["tar_pc"][i_p]).unsqueeze(0).float())
          tar_name[i_p].append(data["tar_name"][i_p])
          
          key_pts[i_p].append(torch.from_numpy(data["key_pts"][i_p]).unsqueeze(0).float())
          w_pc[i_p].append(torch.from_numpy(data["w_pc"][i_p]).unsqueeze(0).float())
          
          src_ver[i_p].append(torch.from_numpy(data["src_ver"][i_p]).float().cuda())
          
          src_face[i_p].append(torch.from_numpy(data["src_face"][i_p]))
          
          tar_ver[i_p].append(torch.from_numpy(data["tar_ver"][i_p]).float().cuda())
          
          tar_face[i_p].append(torch.from_numpy(data["tar_face"][i_p]))
          real_ver[i_p].append(torch.from_numpy(data["real_ver"][i_p]))
          real_face[i_p].append(torch.from_numpy(data["real_face"][i_p]))
          w_mesh[i_p].append(torch.from_numpy(data["w_mesh"][i_p]).float())
          
          src_edges[i_p].append(torch.from_numpy(data['src_edges'][i_p]).long().unsqueeze(0).cuda()) #### src_edges
          src_dofs[i_p].append(torch.from_numpy(data['src_dofs'][i_p]).float().unsqueeze(0).cuda()) #### vert_dofs 
          
          tar_edges[i_p].append(torch.from_numpy(data['tar_edges'][i_p]).long().unsqueeze(0).cuda()) #### src_edges
          tar_dofs[i_p].append(torch.from_numpy(data['tar_dofs'][i_p]).float().unsqueeze(0).cuda()) #### vert_dofs 
        
    for i_p in range(n_parts):
      src_pc[i_p] = torch.cat(src_pc[i_p]).cuda()
      tar_pc[i_p] = torch.cat(tar_pc[i_p]).cuda()
      key_pts[i_p] = torch.cat(key_pts[i_p]).cuda()
      w_pc[i_p] = torch.cat(w_pc[i_p]).cuda()
    return {"src_pc": src_pc, "src_name": src_name, "key_pts": key_pts,
            "tar_pc": tar_pc, "tar_name": tar_name, "w_pc": w_pc,
            "src_ver": src_ver, "src_face": src_face, "w_mesh": w_mesh,
            "tar_ver": tar_ver, "tar_face": tar_face,
            "real_ver": real_ver, "real_face": real_face,
            "src_edges": src_edges, "src_dofs": src_dofs,
            "tar_edges": tar_edges, "tar_dofs": tar_dofs
            }


class ChairDataset(torch.utils.data.Dataset):
    def __init__(self, phase="train", data_dir="../data/chair", opt=None):
        super().__init__()
        self.data_dir = data_dir
        self.phase = phase
        self.opt = opt
        self.load_meta = self.opt.load_meta
        self.num_sampled_pts = 4096
        self.use_paired_data = self.opt.use_paired_data
        self.random_scaling = self.opt.random_scaling
        
        self.n_keypoints = self.opt.n_keypoints
        
        # src_folder = os.path.join(self.data_dir, "src")
        
        #### if we can use any shape as the source shape for deformation? ####
        # if not self.use_paired_data: ### not use paired data...
        #     src_folder = os.path.join(self.data_dir, "dst")
        # else:
        #     src_folder = os.path.join(self.data_dir, "src")
        
        self.data_dir = data_dir ### a list of data_dirs for all parts
        
        src_folder = [os.path.join(cur_data_dir, "src") for cur_data_dir in self.data_dir]
        dst_folder = [os.path.join(cur_data_dir, "dst") for cur_data_dir in self.data_dir]
        
        
        # src_folder = os.path.join(self.data_dir, "src")
        # dst_folder = os.path.join(self.data_dir, "dst")
        
        ### src folders and dst folders ###
        self.src_folder = src_folder
        self.dst_folder = dst_folder
        
        # with open(os.path.join(self.data_dir, "all.txt")) as f:
        #     lines = f.readlines()
        #     self.models = [line.rstrip() for line in lines]
        
        ### all src models ###
        self.src_models = []
        for cur_src_folder in self.src_folder:
          with open(os.path.join(cur_src_folder, "all.txt"), "r") as rf:
            lines = rf.readlines()
            cur_src_models = [line.rstrip() for line in lines]
            self.src_models.append(cur_src_models)
        
        self.dst_models = []
        for cur_dst_folder in self.dst_folder:
          with open(os.path.join(cur_dst_folder, "all.txt"), "r") as rf:
            lines = rf.readlines()
            cur_dst_models = [line.rstrip() for line in lines]
            self.dst_models.append(cur_dst_models) ### current dst_models 
            
        # with open(os.path.join(self.src_folder, "all.txt"), "r") as rf:
        #   lines = rf.readlines()
        #   self.src_models = [line.rstrip() for line in lines]
        
        # with open(os.path.join(self.dst_folder, "all.txt"), "r") as rf:
        #   lines = rf.readlines()
        #   self.dst_models = [line.rstrip() for line in lines]
        
        print(f"Meta-infos loaded with src_models: {len(self.src_models)}, dst_models: {len(self.dst_models)}")
        
        # tot_n_models = len(self.models)
        # 
        
        # self.src_n_models = len(self.src_models)
        # self.dst_n_models = len(self.dst_models)
        
        #### src_n_models ####
        self.src_n_models = [len(cur_src_models) for cur_src_models in self.src_models]
        #### dst_n_models ####
        self.dst_n_models = [len(cur_dst_models) for cur_dst_models in self.dst_models]

        print(f"src_n_models: {self.src_n_models}, dst_n_models: {self.dst_n_models}")
        ### keypoint 256: gt_step_0_subd_0_manifold_tetra.mesh__sf.keypoints_256 
        ### sampled 4096: gt_step_0_subd_0_manifold_tetra.mesh__sf.sampled_4096
        ### surface points: gt_step_0_subd_0_manifold_tetra.mesh__sf
        ### weights sampled 4096 -> 256: gt_step_0_subd_0_manifold_tetra.mesh__sf.keypoints_256.weights.sampled_4096
        ### weights tot pts -> 256: gt_step_0_subd_0_manifold_tetra.mesh__sf.keypoints_256.weights

        ###### src_pc, key_pts, mesh_vertices... ######
        # ##### source pc, key_pts, mesh_vertices, mesh_faces, w_pc, w_mesh #####
        # self.src_pc = [None for _ in range(self.src_n_models)]
        # self.src_key_pts = [None for _ in range(self.src_n_models)]
        # self.src_mesh_vertices = [None for _ in range(self.src_n_models)]
        # self.src_mesh_faces = [None for _ in range(self.src_n_models)]
        # self.src_w_pc = [None for _ in range(self.src_n_models)]
        # self.src_w_mesh = [None for _ in range(self.src_n_models)]
        
        # ##### source pc, key_pts, mesh_vertices, mesh_faces, w_pc, w_mesh #####
        # self.dst_pc = [None for _ in range(self.dst_n_models)]
        # self.dst_key_pts = [None for _ in range(self.dst_n_models)]
        # self.dst_mesh_vertices = [None for _ in range(self.dst_n_models)]
        # self.dst_mesh_faces = [None for _ in range(self.dst_n_models)]
        # self.dst_w_pc = [None for _ in range(self.dst_n_models)]
        # self.dst_w_mesh = [None for _ in range(self.dst_n_models)]
        
        self.src_pc = [[None for _ in range(cur_src_n_models)] for cur_src_n_models in self.src_n_models]
        self.src_key_pts = [[None for _ in range(cur_src_n_models)] for cur_src_n_models in self.src_n_models]
        self.src_mesh_vertices = [[None for _ in range(cur_src_n_models)] for cur_src_n_models in self.src_n_models]
        self.src_mesh_faces = [[None for _ in range(cur_src_n_models)] for cur_src_n_models in self.src_n_models]
        self.src_w_pc = [[None for _ in range(cur_src_n_models)] for cur_src_n_models in self.src_n_models]
        self.src_w_mesh = [[None for _ in range(cur_src_n_models)] for cur_src_n_models in self.src_n_models]
        
        self.dst_pc = [[None for _ in range(cur_dst_n_models)] for cur_dst_n_models in self.dst_n_models]
        self.dst_key_pts = [[None for _ in range(cur_dst_n_models)] for cur_dst_n_models in self.dst_n_models]
        self.dst_mesh_vertices = [[None for _ in range(cur_dst_n_models)] for cur_dst_n_models in self.dst_n_models]
        self.dst_mesh_faces = [[None for _ in range(cur_dst_n_models)] for cur_dst_n_models in self.dst_n_models]
        self.dst_w_pc = [[None for _ in range(cur_dst_n_models)] for cur_dst_n_models in self.dst_n_models]
        self.dst_w_mesh = [[None for _ in range(cur_dst_n_models)] for cur_dst_n_models in self.dst_n_models]
        
        
        
    def load_data_from_model_indicator(self, rt_folder, model_indicator):
        ### read data from rt_folder and the model_indicator ###
        cur_model = model_indicator
        
        cur_keypoints_fn = cur_model + f"_manifold_tetra.mesh__sf.keypoints_{self.n_keypoints}.obj"
        cur_sampled_pts_fn = cur_model + "_manifold_tetra.mesh__sf.sampled_4096.obj"
        cur_surface_pts_fn = cur_model + "_manifold_tetra.mesh__sf.obj"
        cur_weights_sampled_fn = cur_model + f"_manifold_tetra.mesh__sf.keypoints_{self.n_keypoints}.weights.sampled_4096.txt"
        cur_weights_tot_fn = cur_model + f"_manifold_tetra.mesh__sf.keypoints_{self.n_keypoints}.weights.txt"
        
        cur_keypoints, _ =  utils.read_obj_file_ours(os.path.join(rt_folder, cur_keypoints_fn))
        cur_sampled, _ = utils.read_obj_file_ours(os.path.join(rt_folder, cur_sampled_pts_fn))
        cur_surface, cur_faces = utils.read_obj_file_ours(os.path.join(rt_folder, cur_surface_pts_fn), sub_one=True)
        cur_weights_sampled_keypoints = utils.load_txt(os.path.join(rt_folder, cur_weights_sampled_fn))
        cur_weights_mesh_keypoints = utils.load_txt(os.path.join(rt_folder, cur_weights_tot_fn))
        cur_faces = np.array(cur_faces, dtype=np.long) # .long()
        
        return cur_sampled, cur_keypoints, cur_surface, cur_faces, cur_weights_sampled_keypoints, cur_weights_mesh_keypoints
        
        
    
    def __len__(self):
        return self.dst_n_models[0]
        # return self.src_n_models * self.dst_n_models

    
    def apply_random_scaling(self, vertices, pc, keypoints):
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
        return scaled_vertices, scaled_pc, scaled_keypoints
        
    
    def __getitem__(self, idx):
      
        
        
        if self.use_paired_data:
          idxes = [idx * cur_dst_n_models + idx for cur_dst_n_models in self.dst_n_models]
        else:
          idxes = [random.choice(range(cur_src_n_models * cur_dst_n_models)) for cur_src_n_models, cur_dst_n_models in zip(self.src_n_models, self.dst_n_models)]
        
        tot_rt_dict = []
        
        # cur_rt_dict = {"src_name": src_name, "tar_name": tar_name,
        #         "src_pc": src_pc, "tar_pc": tar_pc,
        #         "src_ver": src_vertices, "src_face": src_faces,
        #         "tar_ver": dst_vertices, "tar_face": dst_faces,
        #         "real_ver": dst_vertices, "real_face": dst_faces,
        #         "key_pts": src_key_pts, "w_mesh": src_w_mesh, "w_pc": src_w_pc,
        #         "src_edges": src_edges, "src_dofs": src_dofs,
        #         "tar_edges": tar_edges, "tar_dofs": tar_dofs,
        #         }
        
        tot_src_name, tot_tar_name, tot_src_pc, tot_tar_pc, tot_src_ver, tot_src_face, tot_tar_ver, tot_tar_face, tot_real_ver, tot_real_face, tot_key_pts, tot_w_mesh, tot_w_pc, tot_src_edges, tot_src_dofs, tot_tar_edges, tot_tar_dofs = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
      
        # idx = random.choice(range(self.src_n_models * self.dst_n_models))
        
        for i_p, (idxx, cur_src_n_models, cur_dst_n_models) in enumerate(zip(idxes, self.src_n_models, self.dst_n_models)):
          src_idx = idxx // cur_dst_n_models
          dst_idx = idxx % cur_dst_n_models
          
          # print(f"i_p: {i_p}, idx: {idxx}, cur_src_n_models: {cur_src_n_models}, cur_dst_n_models: {cur_dst_n_models}, src_idx:{src_idx}, dst_idx: {dst_idx}")
          
          if self.src_pc[i_p][src_idx] is None:
            src_pc, src_key_pts, src_vertices, src_faces, src_w_pc, src_w_mesh = self.load_data_from_model_indicator(self.src_folder[i_p], self.src_models[i_p][src_idx])
            self.src_pc[i_p][src_idx] = src_pc
            self.src_key_pts[i_p][src_idx] = src_key_pts
            self.src_mesh_vertices[i_p][src_idx] = src_vertices
            self.src_mesh_faces[i_p][src_idx] = src_faces
            self.src_w_pc[i_p][src_idx] = src_w_pc
            self.src_w_mesh[i_p][src_idx] = src_w_mesh
          while not (self.src_w_pc[i_p][src_idx].shape[1] == self.n_keypoints and self.src_w_pc[i_p][src_idx].shape[0] == self.num_sampled_pts):
            src_idx = random.choice(range(len(self.src_pc[i_p]))) #### src_pc ---> 
            if self.src_pc[i_p][src_idx] is None:
              src_pc, src_key_pts, src_vertices, src_faces, src_w_pc, src_w_mesh = self.load_data_from_model_indicator(self.src_folder[i_p], self.src_models[i_p][src_idx])
              self.src_pc[i_p][src_idx] = src_pc
              self.src_key_pts[i_p][src_idx] = src_key_pts
              self.src_mesh_vertices[i_p][src_idx] = src_vertices
              self.src_mesh_faces[i_p][src_idx] = src_faces
              self.src_w_pc[i_p][src_idx] = src_w_pc
              self.src_w_mesh[i_p][src_idx] = src_w_mesh
              
          if self.dst_pc[i_p][dst_idx] is None:
            tar_pc, dst_key_pts, dst_vertices, dst_faces, dst_w_pc, dst_w_mesh = self.load_data_from_model_indicator(self.dst_folder[i_p], self.dst_models[i_p][dst_idx])
            self.dst_pc[i_p][dst_idx] = tar_pc
            self.dst_key_pts[i_p][dst_idx] = dst_key_pts
            self.dst_mesh_vertices[i_p][dst_idx] = dst_vertices
            self.dst_mesh_faces[i_p][dst_idx] = dst_faces
            self.dst_w_pc[i_p][dst_idx] = dst_w_pc
            self.dst_w_mesh[i_p][dst_idx] = dst_w_mesh

          src_pc, src_key_pts, src_vertices, src_faces, src_w_pc, src_w_mesh = self.src_pc[i_p][src_idx], self.src_key_pts[i_p][src_idx], self.src_mesh_vertices[i_p][src_idx], self.src_mesh_faces[i_p][src_idx], self.src_w_pc[i_p][src_idx], self.src_w_mesh[i_p][src_idx]
          
          tar_pc, dst_key_pts, dst_vertices, dst_faces, dst_w_pc, dst_w_mesh = self.dst_pc[i_p][dst_idx], self.dst_key_pts[i_p][dst_idx], self.dst_mesh_vertices[i_p][dst_idx], self.dst_mesh_faces[i_p][dst_idx], self.dst_w_pc[i_p][dst_idx], self.dst_w_mesh[i_p][dst_idx]
          
          src_edges, src_dofs = dataset_utils.get_edges_from_faces(src_vertices, src_faces)
          tar_edges, tar_dofs = dataset_utils.get_edges_from_faces(dst_vertices, dst_faces)
          
          if self.random_scaling:
              dst_vertices, tar_pc, dst_key_pts = self.apply_random_scaling(dst_vertices, tar_pc, dst_key_pts)
        
          src_name = f"src_{src_idx}"
          tar_name = f"dst_{dst_idx}"
          
          tot_src_name.append(src_name)
          tot_tar_name.append(tar_name)
          tot_src_pc.append(src_pc)
          tot_tar_pc.append(tar_pc)
          tot_src_ver.append(src_vertices)
          tot_src_face.append(src_faces)
          tot_tar_ver.append(dst_vertices)
          tot_tar_face.append(dst_faces)
          tot_real_ver.append(dst_vertices)
          tot_real_face.append(dst_faces)
          tot_key_pts.append(src_key_pts)
          tot_w_mesh.append(src_w_mesh)
          tot_w_pc.append(src_w_pc)
          tot_src_edges.append(src_edges)
          tot_src_dofs.append(src_dofs)
          tot_tar_edges.append(tar_edges)
          tot_tar_dofs.append(tar_dofs)
          
          
        cur_rt_dict = {"src_name": tot_src_name, "tar_name": tot_tar_name,
              "src_pc": tot_src_pc, "tar_pc": tot_tar_pc,
              "src_ver": tot_src_ver, "src_face": tot_src_face,
              "tar_ver": tot_tar_ver, "tar_face": tot_tar_face,
              "real_ver": tot_tar_ver, "real_face": tot_tar_face,
              "key_pts": tot_key_pts, "w_mesh": tot_w_mesh, "w_pc": tot_w_pc,
              "src_edges": tot_src_edges, "src_dofs": tot_src_dofs,
              "tar_edges": tot_tar_edges, "tar_dofs": tot_tar_dofs,
              }
        
        ### rt_dict ###
        # tot_rt_dict.append(
        #   cur_rt_dict
        # )
        ### rt_dict ###
        return cur_rt_dict
