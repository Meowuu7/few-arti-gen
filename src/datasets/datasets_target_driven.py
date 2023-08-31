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
        
    src_pc = torch.cat(src_pc).cuda()
    tar_pc = torch.cat(tar_pc).cuda()
    key_pts = torch.cat(key_pts).cuda()
    w_pc = torch.cat(w_pc).cuda()
    dst_key_pts = torch.cat(dst_key_pts).cuda()
    return {"src_pc": src_pc, "src_name": src_name, "key_pts": key_pts,
            "tar_pc": tar_pc, "tar_name": tar_name, "w_pc": w_pc,
            "src_ver": src_ver, "src_face": src_face, "w_mesh": w_mesh,
            "tar_ver": tar_ver, "tar_face": tar_face,
            "real_ver": real_ver, "real_face": real_face,
            "src_edges": src_edges, "src_dofs": src_dofs,
            "tar_edges": tar_edges, "tar_dofs": tar_dofs, "dst_key_pts": dst_key_pts
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
        if not self.use_paired_data:
            src_folder = os.path.join(self.data_dir, "dst")
        else:
            src_folder = os.path.join(self.data_dir, "src")
        dst_folder = os.path.join(self.data_dir, "dst")
        
        self.src_folder = src_folder
        self.dst_folder = dst_folder

        with open(os.path.join(self.data_dir, "all.txt")) as f:
            lines = f.readlines()
            self.models = [line.rstrip() for line in lines]
        
        tot_n_models = len(self.models)
        ### keypoint 256: gt_step_0_subd_0_manifold_tetra.mesh__sf.keypoints_256 
        ### sampled 4096: gt_step_0_subd_0_manifold_tetra.mesh__sf.sampled_4096
        ### surface points: gt_step_0_subd_0_manifold_tetra.mesh__sf
        ### weights sampled 4096 -> 256: gt_step_0_subd_0_manifold_tetra.mesh__sf.keypoints_256.weights.sampled_4096
        ### weights tot pts -> 256: gt_step_0_subd_0_manifold_tetra.mesh__sf.keypoints_256.weights

        self.pc = []
        self.key_pts = []
        self.mesh_vertices = []
        self.mesh_faces = []
        self.w_pc = []
        self.w_mesh = []
        
        if not self.load_meta:
            for i_model, cur_model in enumerate(self.models):
                cur_keypoints_fn = cur_model + f"_manifold_tetra.mesh__sf.keypoints_{self.n_keypoints}.obj"
                cur_sampled_pts_fn = cur_model + "_manifold_tetra.mesh__sf.sampled_4096.obj"
                cur_surface_pts_fn = cur_model + "_manifold_tetra.mesh__sf.obj"
                cur_weights_sampled_fn = cur_model + f"_manifold_tetra.mesh__sf.keypoints_{self.n_keypoints}.weights.sampled_4096.txt"
                cur_weights_tot_fn = cur_model + f"_manifold_tetra.mesh__sf.keypoints_{self.n_keypoints}.weights.txt"
                cur_src_keypoints, _ =  utils.read_obj_file_ours(os.path.join(src_folder, cur_keypoints_fn))
                cur_dst_keypoints, _ = utils.read_obj_file_ours(os.path.join(dst_folder, cur_keypoints_fn))
                cur_src_sampled, _ = utils.read_obj_file_ours(os.path.join(src_folder, cur_sampled_pts_fn))
                cur_dst_sampled, _ = utils.read_obj_file_ours(os.path.join(dst_folder, cur_sampled_pts_fn))
                cur_src_surface, cur_src_faces = utils.read_obj_file_ours(os.path.join(src_folder, cur_surface_pts_fn), sub_one=True)
                cur_dst_surface, cur_dst_faces = utils.read_obj_file_ours(os.path.join(dst_folder, cur_surface_pts_fn), sub_one=True)
                cur_src_weights_sampled_keypoints = utils.load_txt(os.path.join(src_folder, cur_weights_sampled_fn))
                cur_dst_weights_sampled_keypoints = utils.load_txt(os.path.join(dst_folder, cur_weights_sampled_fn))
                cur_src_weights_mesh_keypoints = utils.load_txt(os.path.join(src_folder, cur_weights_tot_fn))
                cur_dst_weights_mesh_keypoints = utils.load_txt(os.path.join(dst_folder, cur_weights_tot_fn))
                cur_src_faces = np.array(cur_src_faces, dtype=np.long) # .long()
                cur_dst_faces = np.array(cur_dst_faces, dtype=np.long) # .long()
                self.pc.append({"src": cur_src_sampled, "dst": cur_dst_sampled})
                self.key_pts.append({"src": cur_src_keypoints, "dst": cur_dst_keypoints})
                self.mesh_vertices.append({"src": cur_src_surface, "dst": cur_dst_surface})
                self.mesh_faces.append({"src": cur_src_faces, "dst": cur_dst_faces})
                self.w_pc.append({"src": cur_src_weights_sampled_keypoints, "dst": cur_dst_weights_sampled_keypoints})
                self.w_mesh.append({"src": cur_src_weights_mesh_keypoints, "dst": cur_dst_weights_mesh_keypoints})
            # self.pc = 
        else:
            self.pc = [None for _ in range(tot_n_models)]
            self.key_pts = [None for _ in range(tot_n_models)]
            self.mesh_vertices = [None for _ in range(tot_n_models)]
            self.mesh_faces = [None for _ in range(tot_n_models)]
            self.w_pc = [None for _ in range(tot_n_models)]
            self.w_mesh = [None for _ in range(tot_n_models)]

        # self.k = 20

        # with open(os.path.join(self.data_dir, "%s.txt" % self.phase)) as f:
        #     lines = f.readlines()
        #     self.ids = [int(line.rstrip()) for line in lines]

        # self.pc = np.load(os.path.join(
        #     self.data_dir, "pc_4096.npy"), allow_pickle=True)
        # self.key_pts = np.load(os.path.join(
        #     self.data_dir, "key_point_50.npy"), allow_pickle=True)
        # self.mesh_vertices = np.load(os.path.join(
        #     self.data_dir, "mesh_vertices.npy"), allow_pickle=True)
        # self.mesh_faces = np.load(os.path.join(
        #     self.data_dir, "mesh_faces.npy"), allow_pickle=True)
        # # biharmonic weights
        # self.w_pc = np.load(os.path.join(
        #     self.data_dir, "w_pc_4096.npy"), allow_pickle=True)
        # w_mesh_0 = np.load(os.path.join(
        #     self.data_dir, "w_mesh_0.npy"), allow_pickle=True)
        # w_mesh_1 = np.load(os.path.join(
        #     self.data_dir, "w_mesh_1.npy"), allow_pickle=True)
        # w_mesh_2 = np.load(os.path.join(
        #     self.data_dir, "w_mesh_2.npy"), allow_pickle=True)
        # w_mesh_3 = np.load(os.path.join(
        #     self.data_dir, "w_mesh_3.npy"), allow_pickle=True)
        # self.w_mesh = list(w_mesh_0) + list(w_mesh_1) + \
        #     list(w_mesh_2) + list(w_mesh_3)
    def load_data_from_model_indicator(self, model_indicator, model_idx):
        cur_model = model_indicator
        src_folder = self.src_folder
        dst_folder = self.dst_folder
        
        cur_keypoints_fn = cur_model + f"_manifold_tetra.mesh__sf.keypoints_{self.n_keypoints}.obj"
        cur_sampled_pts_fn = cur_model + "_manifold_tetra.mesh__sf.sampled_4096.obj"
        cur_surface_pts_fn = cur_model + "_manifold_tetra.mesh__sf.obj"
        cur_weights_sampled_fn = cur_model + f"_manifold_tetra.mesh__sf.keypoints_{self.n_keypoints}.weights.sampled_4096.txt"
        cur_weights_tot_fn = cur_model + f"_manifold_tetra.mesh__sf.keypoints_{self.n_keypoints}.weights.txt"
        cur_src_keypoints, _ =  utils.read_obj_file_ours(os.path.join(src_folder, cur_keypoints_fn))
        cur_dst_keypoints, _ = utils.read_obj_file_ours(os.path.join(dst_folder, cur_keypoints_fn))
        cur_src_sampled, _ = utils.read_obj_file_ours(os.path.join(src_folder, cur_sampled_pts_fn))
        cur_dst_sampled, _ = utils.read_obj_file_ours(os.path.join(dst_folder, cur_sampled_pts_fn))
        cur_src_surface, cur_src_faces = utils.read_obj_file_ours(os.path.join(src_folder, cur_surface_pts_fn), sub_one=True)
        cur_dst_surface, cur_dst_faces = utils.read_obj_file_ours(os.path.join(dst_folder, cur_surface_pts_fn), sub_one=True)
        cur_src_weights_sampled_keypoints = utils.load_txt(os.path.join(src_folder, cur_weights_sampled_fn))
        cur_dst_weights_sampled_keypoints = cur_src_weights_sampled_keypoints #  utils.load_txt(os.path.join(dst_folder, cur_weights_sampled_fn))
        cur_src_weights_mesh_keypoints = utils.load_txt(os.path.join(src_folder, cur_weights_tot_fn))
        cur_dst_weights_mesh_keypoints = cur_src_weights_mesh_keypoints #  utils.load_txt(os.path.join(dst_folder, cur_weights_tot_fn))
        cur_src_faces = np.array(cur_src_faces, dtype=np.long) # .long()
        cur_dst_faces = np.array(cur_dst_faces, dtype=np.long) # .long() ### dst_faces
        pc_dict = {"src": cur_src_sampled, "dst": cur_dst_sampled}
        key_pts_dict = {"src": cur_src_keypoints, "dst": cur_dst_keypoints}
        mesh_vertices_dict = {"src": cur_src_surface, "dst": cur_dst_surface}
        mesh_faces_dict = {"src": cur_src_faces, "dst": cur_dst_faces}
        w_pc_dict = {"src": cur_src_weights_sampled_keypoints, "dst": cur_dst_weights_sampled_keypoints}
        w_mesh_dict = {"src": cur_src_weights_mesh_keypoints, "dst": cur_dst_weights_mesh_keypoints}
        self.pc[model_idx] = pc_dict
        self.key_pts[model_idx] = key_pts_dict
        self.mesh_vertices[model_idx] = mesh_vertices_dict
        self.mesh_faces[model_idx] = mesh_faces_dict
        self.w_pc[model_idx] = w_pc_dict
        self.w_mesh[model_idx] = w_mesh_dict
        
    
    def __len__(self):
        return len(self.pc) # len(self.ids) * self.k

    
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
        if self.load_meta and self.pc[idx] is None:
            self.load_data_from_model_indicator(self.models[idx], idx) ### load data at the origianl idx...
            
        while not (self.w_pc[idx]['src'].shape[1] == self.n_keypoints and self.w_pc[idx]['src'].shape[0] ==  self.num_sampled_pts):
            # print(f"[DATASET] idx: {idx}, model_name: {self.models[idx]}, w_pc.shape: {self.w_pc[idx]['src'].shape}") 
            idx = random.choice(range(len(self.pc)))
            #### get data for w_pc ####
            if self.load_meta and self.pc[idx] is None: ### laod data at the sampled idx-th shape ###
                self.load_data_from_model_indicator(self.models[idx], idx)
        # print(f"[DATAEST] tests w_pc: {self.w_pc[idx]['src'].shape}")
        # src_id = self.ids[idx // self.k]
        tar_id = random.choice(range(len(self.pc)))
        
        #### load_data_from_model_indicator --> tar_id...
        if self.load_meta and self.pc[tar_id] is None:
            self.load_data_from_model_indicator(self.models[tar_id], tar_id)
        # positive sample for discriminator network
        # real_id = random.choice(self.ids) 
        # src_name = self.models[src_id]
        # tar_name = self.models[tar_id]
        src_name = self.models[idx] + "_src"
        tar_name = self.models[idx] + "_tar"
        pc_pair = self.pc[idx]
        
        
        src_pc, tar_pc = pc_pair["src"], pc_pair["dst"]
        
        tar_pc = self.pc[tar_id]["dst"]
        # src_pc = self.pc[src_id]
        # tar_pc = self.pc[tar_id]
        # key_pts = self.key_pts[src_id]
        # key_pts_pair = self.key_pts[idx]
        src_key_pts, dst_key_pts = self.key_pts[idx]['src'], self.key_pts[tar_id]['dst']
        src_w_mesh, dst_w_mesh = self.w_mesh[idx]['src'], self.w_mesh[tar_id]['dst']
        src_w_pc, dst_w_pc = self.w_pc[idx]['src'], self.w_pc[tar_id]['dst']
        # src_vertices, dst_vertices = self.mesh_vertices[idx]['src'], self.mesh_vertices[idx]['dst']
        src_vertices, dst_vertices = self.mesh_vertices[idx]['src'], self.mesh_vertices[tar_id]['dst'] ### paired training data
        src_faces, dst_faces = self.mesh_faces[idx]['src'], self.mesh_faces[tar_id]['dst']
        
        real_vertices = self.mesh_vertices[tar_id]['dst']
        real_faces = self.mesh_vertices[tar_id]['dst']
        
        src_edges, src_dofs = dataset_utils.get_edges_from_faces(src_vertices, src_faces)
        tar_edges, tar_dofs = dataset_utils.get_edges_from_faces(dst_vertices, dst_faces)
        
        if self.random_scaling:
            dst_vertices, tar_pc, dst_key_pts = self.apply_random_scaling(dst_vertices, tar_pc, dst_key_pts)
        
        
        ### for dst vertices transforamtion ###
        
        # w_mesh = self.w_mesh[src_id]
        # w_pc = self.w_pc[src_id]
        # src_ver = self.mesh_vertices[src_id]
        # src_face = self.mesh_faces[src_id]
        # tar_ver = self.mesh_vertices[tar_id]
        # tar_face = self.mesh_faces[tar_id]
        # real_ver = self.mesh_vertices[real_id]
        # real_face = self.mesh_faces[real_id]
        # print(f"[DTASET] test for w_pc: {src_w_pc.shape}")
        return {"src_name": src_name, "tar_name": tar_name,
                "src_pc": src_pc, "tar_pc": tar_pc,
                "src_ver": src_vertices, "src_face": src_faces,
                "tar_ver": dst_vertices, "tar_face": dst_faces,
                # "real_ver": dst_vertices, "real_face": dst_faces,
                "real_ver": real_vertices, "real_face": real_faces,
                "key_pts": src_key_pts, "w_mesh": src_w_mesh, "w_pc": src_w_pc,
                "src_edges": src_edges, "src_dofs": src_dofs,
                "tar_edges": tar_edges, "tar_dofs": tar_dofs, "dst_key_pts": dst_key_pts
                }
