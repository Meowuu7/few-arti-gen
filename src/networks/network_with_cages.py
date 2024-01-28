from multiprocessing.sharedctypes import Value
from platform import java_ver
import torch
import torch.nn.functional as F
import torch.nn as nn

from src.networks.pointnet_utils import pointnet_encoder, PointFlowEncoder
# from losses import chamfer_distance
from src.common_utils.losses import chamfer_distance_raw as chamfer_distance
import src.common_utils.losses as losses
import src.common_utils.utils as utils
# from pointnet2 import PointnetPP
# import edge_propagation
from src.common_utils.data_utils_torch import farthest_point_sampling, batched_index_select
# from src.common_utils.data_utils_torch import compute_normals_o3d, get_vals_via_nearest_neighbours
# from scipy.optimize import linear_sum_assignment
import numpy as np
import src.common_utils.model_utils as model_utils



class ImageRender:
    def __init__(self):
        # Camera defination, we follow the defination from Blender (check the render_shapenet_data/rener_shapenet.py for more details)
        fovy = np.arctan(32 / 2 / 35) * 2 ## 
        fovyangle = fovy / np.pi * 180.0
        dmtet_camera = PerspectiveCamera(fovy=fovyangle, device='cuda')
        # Renderer we used. ### image render
        dmtet_renderer = NeuralRender('cuda', camera_model=dmtet_camera)
        self.render_type = 'neural_render'
        self.render = dmtet_renderer
        # self.data_camera_mode = 'carla'
        self.data_camera_mode = 'shapenet_chair'
        self.n_views = 2
        self.device = 'cuda'

    ### render meshes... ###
    def render_mesh(self, mesh_v_nx3, mesh_f_fx3, resolution=256, hierarchical_mask=False):
        # Generate random camera
        with torch.no_grad():
            # if camera is None:
            campos, cam_mv, rotation_angle, elevation_angle, sample_r = self.generate_random_camera(
                1, n_views=self.n_views)
            # gen_camera = (campos, cam_mv, sample_r, rotation_angle, elevation_angle)
            # run_n_view = self.n_views

        # Render the mesh into 2D image (get 3d position of each image plane)
        # antilias_mask, hard_mask, return_value = self.render_mesh(mesh_v, mesh_f, cam_mv)
        
        return_value = dict()
        if self.render_type == 'neural_render':
            tex_pos, mask, hard_mask, rast, v_pos_clip, mask_pyramid, depth = self.render.render_mesh(
                mesh_v_nx3.unsqueeze(dim=0), ### n x 3
                mesh_f_fx3.int(), ### 
                cam_mv,
                mesh_v_nx3.unsqueeze(dim=0),
                resolution=resolution,
                device=self.device,
                hierarchical_mask=hierarchical_mask
            )

            return_value['tex_pos'] = tex_pos
            return_value['mask'] = mask
            return_value['hard_mask'] = hard_mask
            return_value['rast'] = rast
            return_value['v_pos_clip'] = v_pos_clip
            return_value['mask_pyramid'] = mask_pyramid
            return_value['depth'] = depth
        else:
            raise NotImplementedError

        return return_value
    
    def generate_random_camera(self, batch_size, n_views=2):
        '''
        Sample a random camera from the camera distribution during training
        :param batch_size: batch size for the generator
        :param n_views: number of views for each shape within a batch
        :return:
        '''
        # carla
        sample_r = None
        world2cam_matrix, forward_vector, camera_origin, rotation_angle, elevation_angle = sample_camera(
            self.data_camera_mode, batch_size * n_views, 'cuda')
        mv_batch = world2cam_matrix
        campos = camera_origin
        return campos.reshape(batch_size, n_views, 3), mv_batch.reshape(batch_size, n_views, 4, 4), \
               rotation_angle, elevation_angle, sample_r




### 
class Discriminator(nn.Module): ### 
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.leakyrelu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.leakyrelu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(32, 32, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.leakyrelu3 = nn.LeakyReLU(0.2, inplace=True)

        self.fc = nn.Linear(8192, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.leakyrelu1(self.bn1(self.conv1(x)))
        x = self.leakyrelu2(self.bn2(self.conv2(x)))
        x = self.leakyrelu3(self.bn3(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.sigmoid(self.fc(x))
        return x


import pymesh

def read_trimesh(path, normal=False, clean=True):
    mesh = pymesh.load_mesh(path)
    if clean:
        mesh, info = pymesh.remove_isolated_vertices(mesh)
        print("Removed {} isolated vertices".format(info["num_vertex_removed"]))
        mesh, info = pymesh.remove_duplicated_vertices(mesh)
        print("Merged {} duplicated vertices".format(info["num_vertex_merged"]))
        mesh, info = pymesh.remove_degenerated_triangles(mesh)
        mesh = pymesh.form_mesh(mesh.vertices, mesh.faces)

    vertices = mesh.vertices
    if normal:
        mesh.add_attribute("vertex_normal")
        vertex_normals = mesh.get_attribute("vertex_normal").reshape(-1, 3)
        vertices = np.concatenate([vertices, vertex_normals], axis=-1) # vertices #
    return vertices, mesh.faces


def loadInitCage(templates):
    init_cage_Fs = []
    for i, template in enumerate(templates):
        if template.endswith(".off") or template.endswith(".OFF"):
            init_cage_V, init_cage_F = utils.read_off_file_ours(template, sub_one=False)
        else:
            init_cage_V, init_cage_F = utils.read_obj_file_ours(template, sub_one=True)
            init_cage_F = np.array(init_cage_F, dtype=np.int64)
        # init_cage_V, init_cage_F = read_trimesh(template)
        init_cage_V = torch.from_numpy(init_cage_V[:,:3].astype(np.float32)).unsqueeze(0).cuda() # *2.0
        init_cage_F = torch.from_numpy(init_cage_F[:,:3].astype(np.int64)).unsqueeze(0).cuda()
        init_cage_Fs.append(init_cage_F)
    return init_cage_V, init_cage_Fs


from pytorch_points.utils.geometry_utils import read_trimesh, write_trimesh, build_gemm, Mesh, get_edge_points, generatePolygon

# from pytorch_points.network.geo_operations import mean_value_coordinates_3D, edge_vertex_indices
from src.networks.neuralcages_networks import NetworkFull, deform_with_MVC

from pytorch_points.utils.pytorch_utils import weights_init

class Cages(nn.Module):
  def __init__(self, opt, cage_template=None):
    super().__init__()
    # cage (1,N,3)
    cur_template = cage_template if cage_template is not None else opt.template
    init_cage_V, init_cage_Fs = loadInitCage([cur_template]) ### 
    self.init_cage_V = init_cage_V # load init cages 
    cage_V_t = init_cage_V.transpose(1,2).detach().cuda()
    cage_edge_points_list = []
    cage_edges_list = []
    for F in init_cage_Fs:
        mesh = Mesh(vertices=init_cage_V[0], faces=F[0])
        build_gemm(mesh, F[0])
        cage_edge_points = torch.from_numpy(get_edge_points(mesh)).cuda()
        cage_edge_points_list.append(cage_edge_points)
        # cage_edges_list = [edge_vertex_indices(F[0])]
        
    self.bottleneck_size = 512
    self.cage_network = NetworkFull(opt, 3, self.bottleneck_size,
                 template_vertices=cage_V_t, template_faces=init_cage_Fs[-1]
                 )
    self.cage_network.apply(weights_init)

  def setup_template(self, template_fn):
        init_cage_V, init_cage_Fs = loadInitCage([template_fn]) ### 
        self.init_cage_V = init_cage_V
        cage_V_t = init_cage_V.transpose(1,2).detach().cuda()
        cage_edge_points_list = []
        cage_edges_list = []
        for F in init_cage_Fs:
            mesh = Mesh(vertices=init_cage_V[0], faces=F[0])
            build_gemm(mesh, F[0])
            cage_edge_points = torch.from_numpy(get_edge_points(mesh)).cuda()
            cage_edge_points_list.append(cage_edge_points)
            # cage_edges_list = [edge_vertex_indices(F[0])]
        self.cage_network.set_up_template(cage_V_t, init_cage_Fs[-1])
      
    
    # rnd_sample_nn #
  def forward(self, pc, target_pc, pc_cvx=None, target_pc_cvx=None, rnd_sample_nn=0):
    # print(torch.max(self.init_cage_V, dim=1)[0], torch.max(pc, dim=1)[0], torch.max(target_pc, dim=1)[0])
    pc = pc.contiguous().transpose(1, 2).contiguous()
    target_pc = target_pc.contiguous().transpose(1, 2).contiguous()
    pc_cvx = pc_cvx.contiguous().transpose(1, 2).contiguous() if pc_cvx is not None else pc
    target_pc_cvx = target_pc_cvx.contiguous().transpose(1, 2).contiguous() if target_pc_cvx is not None else target_pc
    outputs = self.cage_network(pc, target_pc, source_cvx_shp=pc_cvx, target_cvx_shp=target_pc_cvx, alpha=1.0, rnd_smaple_nn=rnd_sample_nn) ### pc, target_pc...
    return outputs
    
    

class model(nn.Module):
    def __init__(self, num_basis, training_dataset=None, opt=None):
        super(model, self).__init__()
        print("for network", opt)
        
        self.opt = opt
        
        ## pointnet; tar_pointnet ##
        self.pointnet = pointnet_encoder()
        self.tar_pointnet = pointnet_encoder()
        
        self.feat_dim = 2883
        self.feat_dim = 64
        self.feat_dim = 512
        
        self.dis_factor = opt.dis_factor
        self.training_dataset = training_dataset
        
        
        # self.pvconv = PVConv(3, self.feat_dim, 5,  64, with_se=False, normalize=True, eps=0)
        # self.tar_pvconv = PVConv(3, self.feat_dim, 5,  64, with_se=False, normalize=True, eps=0)
        
        # (self, zdim, input_dim=3, use_deterministic_encoder=False):
        self.pn_enc = PointFlowEncoder(zdim=self.feat_dim, input_dim=3, use_deterministic_encoder=True)
        self.tar_pn_enc = PointFlowEncoder(zdim=self.feat_dim, input_dim=3, use_deterministic_encoder=True)
        
        
        self.glb_pn_enc = PointFlowEncoder(zdim=self.feat_dim, input_dim=3, use_deterministic_encoder=True)
        self.glb_tar_pn_enc = PointFlowEncoder(zdim=self.feat_dim, input_dim=3, use_deterministic_encoder=True)
        
        # self.glb_pn_enc = PVConv(3, self.feat_dim, 5,  64, with_se=False, normalize=True, eps=0)
        # self.glb_tar_pn_enc = PVConv(3, self.feat_dim, 5,  64, with_se=False, normalize=True, eps=0)
        
        
        self.src_ppfeats_out_net = nn.Sequential(
             torch.nn.Conv1d(self.feat_dim * 2, self.feat_dim, 1), nn.ReLU(), 
             torch.nn.Conv1d(self.feat_dim, self.feat_dim, 1), nn.ReLU(), 
             torch.nn.Conv1d(self.feat_dim, self.feat_dim, 1)
        )
        
        
        self.src_glb_feat_out_net = nn.Sequential(
             torch.nn.Conv1d(self.feat_dim, self.feat_dim, 1), nn.ReLU(), 
             torch.nn.Conv1d(self.feat_dim, self.feat_dim, 1), nn.ReLU(), 
             torch.nn.Conv1d(self.feat_dim, self.feat_dim, 1)
        )
        
        self.tar_glb_feat_out_net = nn.Sequential(
             torch.nn.Conv1d(self.feat_dim, self.feat_dim, 1), nn.ReLU(), 
             torch.nn.Conv1d(self.feat_dim, self.feat_dim, 1), nn.ReLU(), 
             torch.nn.Conv1d(self.feat_dim, self.feat_dim, 1)
        )
        
        self.glb_src_glb_feat_out_net = nn.Sequential(
             torch.nn.Conv1d(self.feat_dim, self.feat_dim, 1), nn.ReLU(), 
             torch.nn.Conv1d(self.feat_dim, self.feat_dim, 1), nn.ReLU(), 
             torch.nn.Conv1d(self.feat_dim, self.feat_dim, 1)
        )
        
        self.glb_tar_glb_feat_out_net = nn.Sequential(
             torch.nn.Conv1d(self.feat_dim, self.feat_dim, 1), nn.ReLU(), 
             torch.nn.Conv1d(self.feat_dim, self.feat_dim, 1), nn.ReLU(), 
             torch.nn.Conv1d(self.feat_dim, self.feat_dim, 1)
        )
        
        
        self.num_basis = num_basis
        
        self.coef_multiplier = opt.coef_multiplier
        
        self.pred_type = opt.pred_type
        
        

        # src point feature 2883 * N
        # self.conv11 = torch.nn.Conv1d(2883, 128, 1)
        self.conv11 = torch.nn.Conv1d(self.feat_dim, 128, 1)
        self.conv12 = torch.nn.Conv1d(128, 64, 1)
        self.conv13 = torch.nn.Conv1d(64, 64, 1)
        self.bn11 = nn.BatchNorm1d(128)
        self.bn12 = nn.BatchNorm1d(64)
        self.bn13 = nn.BatchNorm1d(64)

        # key point feature K (64 + 3 + 1) * N
        self.conv21_in_feats = 68 if opt.def_version == "v4" else 67
        self.conv21 = torch.nn.Conv1d(self.conv21_in_feats, 64, 1)
        # self.conv21 = torch.nn.Conv1d(67, 64, 1)
        self.conv22 = torch.nn.Conv1d(64, 64, 1)
        self.bn21 = nn.BatchNorm1d(64)
        self.bn22 = nn.BatchNorm1d(64)

        # basis feature K 64
        
        self.conv31 = torch.nn.Conv1d(64 + 3, 256, 1)
        self.conv32 = torch.nn.Conv1d(256, 512, 1)
        self.conv33 = torch.nn.Conv1d(512, self.num_basis * 3, 1)
        self.bn31 = nn.BatchNorm1d(256)
        self.bn32 = nn.BatchNorm1d(512)

        # key point feature with target K (2048 + 64 + 3)
        # self.conv41 = torch.nn.Conv1d(2048 + 64 + 3, 256, 1)
        
        self.conv41 = torch.nn.Conv1d(self.feat_dim + self.feat_dim + 64 + 3, 256, 1)
        self.conv42 = torch.nn.Conv1d(256, 128, 1)
        self.conv43 = torch.nn.Conv1d(128, 128, 1)
        self.bn41 = nn.BatchNorm1d(256)
        self.bn42 = nn.BatchNorm1d(128)
        self.bn43 = nn.BatchNorm1d(128)

        # coef feature 15 (128 + 3 + 3) K
        self.conv51 = torch.nn.Conv1d(128 + 3 + 3, 256, 1)
        self.conv52 = torch.nn.Conv1d(256, 128, 1)
        self.conv53 = torch.nn.Conv1d(128, 128, 1)
        self.bn51 = nn.BatchNorm1d(256)
        self.bn52 = nn.BatchNorm1d(128)
        self.bn53 = nn.BatchNorm1d(128)


        coef_pred_out_dim = 3 if self.pred_type == "offset" else 1
        coef_pred_in_dim = 128 if self.pred_type == "offset" else 128 + 2
        # coef feature 15 128
        self.conv61 = torch.nn.Conv1d(coef_pred_in_dim, 64, 1)
        self.conv62 = torch.nn.Conv1d(64, 32, 1)
        self.conv63 = torch.nn.Conv1d(32, coef_pred_out_dim, 1)
        self.bn61 = nn.BatchNorm1d(64)
        self.bn62 = nn.BatchNorm1d(32)

        self.conv71 = torch.nn.Conv1d(64 + 3 + 3, 32, 1)
        self.conv72 = torch.nn.Conv1d(32, 16, 1)
        self.conv73 = torch.nn.Conv1d(16, 2, 1)
        self.bn71 = nn.BatchNorm1d(32)
        self.bn72 = nn.BatchNorm1d(16)

        self.sigmoid = nn.Sigmoid()
        
        self.cages = Cages(self.opt)
        self.glb_cages = Cages(self.opt, cage_template=self.opt.glb_template)
        # self.glb_deform_model = glb_deform_model(num_basis=self.num_basis, opt=opt)
        



    ## local chamfer distances, global chamfer distances? 
    def get_minn_distances_base_pc_cvx_pts(self, base_pc, cvx_pts): 
      ### base_pc: bsz x nn_pts x 3; cvx_pts: bsz x n_cvx x nn_cvx_pts x 3 ###
      dists_base_pc_cvx_pts = torch.sum((base_pc.unsqueeze(-2).unsqueeze(-2) - cvx_pts.unsqueeze(1)) ** 2, dim=-1) 
      dists_base_pc_cvx_pts = torch.sqrt(dists_base_pc_cvx_pts) ### bsz x nn_pts x n_cvx x nn_cvx_pts 
      dists_base_pc_cvx_pts, _ = torch.min(dists_base_pc_cvx_pts, dim=-1) ### bsz x nn_pts x nn_cvx  ### minn_dist_to_cvx
      dists_base_pc_cvx_pts, _ = torch.min(dists_base_pc_cvx_pts, dim=-1) ### bsz x nn_pts --> dists_base_pc_cvx_pts ### bsz x nn_pts 
      avg_dists_base_pc_cvx_pts = torch.mean(dists_base_pc_cvx_pts, dim=-1) ### bsz
      return avg_dists_base_pc_cvx_pts #### average distance between pc and cvx pts ####

    
    ### base pc cv indicators ###
    def get_base_pc_cvx_indicator(self, base_pc, cvx_pts, minn_dist_base_to_cvx_pts): ### minn_dist_base_to_cvx_pts -> a value
      dists_base_pc_cvx_pts = torch.sum((base_pc.unsqueeze(-2).unsqueeze(-2) - cvx_pts.unsqueeze(1)) ** 2, dim=-1) 
      dists_base_pc_cvx_pts = torch.sqrt(dists_base_pc_cvx_pts) ### bsz x nn_pts x n_cvx x nn_cvx_pts 
      dists_base_pc_cvx_pts, _ = torch.min(dists_base_pc_cvx_pts, dim=-1) ### bsz x nn_pts x nn_cvx  ### minn_dist_tocvx
    #   coef = 2.0
      coef = 2.5
      coef = self.dis_factor
      base_pc_cvx_indicators = (dists_base_pc_cvx_pts <= minn_dist_base_to_cvx_pts.unsqueeze(-1).unsqueeze(-1) * coef).float() ### indicator: bsz x nn_pts x nn_cvx --> nn_pts, nn_cvx 
      return base_pc_cvx_indicators ### base
    
    

    def forward7(
        self, src_pc, tar_pc, key_pts, dst_key_pts, w_pc,  src_verts, src_convex_pts, tar_verts, dst_convex_pts, glb_net=None
        ):
        # src_pc = 
        ### src_pc, tar_pc;
        ### src_convex_pts:
        ### src_convex_pts: 
        # ---- only use convexes with the same idx in those two shapes ###
        ### src_convex_pts: bsz x n_cvxes x n_pts_per_convex  ### forward3
        ### src_keypts: bsz x n_keypts x 3
        #### B, N, _ = src
        #### add src_faces, tar_faces for src_meshes and tar_meshes #### 
        #### network target driven #### ## target driven ##
        B, N, _ = src_pc.shape
        nn_cvx = src_convex_pts.size(1) ### bsz x nn_cvx 
        
        src_convex_pts = src_convex_pts * 2.
        dst_convex_pts = dst_convex_pts * 2.
        
        # w_pc: bsz x n_pc x n_keypts
        # print(f"w_pc: {w_pc.size()}, w_pc_sum: {torch.sum(w_pc, dim=-1)[0, :2]}")
        # print(f"maxx_src: {torch.max(src_pc)}, minn_src: {torch.min(src_pc)}, maxx src_convex_pts: {torch.max(src_convex_pts)}, minn src_convex_pts: {torch.min(src_convex_pts)}, ")
        
        ###  src ### # get_minn_distances_base_pc_cvx_pts ##
        avg_src_pc_cvx = self.get_minn_distances_base_pc_cvx_pts(src_pc, src_convex_pts) ### 
        ### bsz x nn_pts x nn_cvx ###
        src_pc_cvx_indicator = self.get_base_pc_cvx_indicator(src_pc, src_convex_pts, avg_src_pc_cvx) ### 
        # src_keypts_cvx_indicator = self.get_base_pc_cvx_indicator(key_pts, src_convex_pts, avg_src_pc_cvx) ### 
        
        ### dst pc cvx ###
        avg_dst_pc_cvx = self.get_minn_distances_base_pc_cvx_pts(tar_pc, dst_convex_pts)
        tar_pc_cvx_indicator = self.get_base_pc_cvx_indicator(tar_pc, dst_convex_pts, avg_dst_pc_cvx) ### base-pc-cvx
        
        
        tot_cd_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean()
        
        tot_cvx_pts_idxes = []
        
        
        EPS = 1e-6
        
        tot_def_pcs = []
        tot_rnd_def_pcs = []
        tot_rnd_tot_def_pcs = [[] for _ in range(self.opt.rnd_sample_nn)]
        tot_merged_def_pcs = []
        tot_pure_merged_def_pcs = []
        tot_merged_rnd_def_pcs = []
        tot_tot_sampled_merged_rnd_def_pcs = [[] for _ in range(self.opt.rnd_sample_nn)]
        tot_tot_sampled_glb_rnd_def_pcs = [[] for _ in range(self.opt.rnd_sample_nn)]
        tot_cage_extents_loss = []
        tot_cd_loss = []
        tot_cage_offset_loss = []
        
        tot_cages = []
        tot_new_cages = []
        tot_cages_faces = []
        tot_rnd_new_cages = []
        # tot_new_cages_faces = []
        
        tot_cage_of_merged_cages = []
        tot_cage_of_def_cages = []
        
        tot_cage_basis = []
        tot_cage_coefs = []
        # cur_bsz_cat_offset_loss
        tot_cat_offset_loss = []
        
        tot_cat_cur_bsz_cage_coefs_argsort = []
        tot_avg_coefs = []
        tot_sorted_basis = []
        
        tot_bsz_cages_ori = []
        
        tot_valid_cvx_idxes = []
        
        
        tot_bsz_valid_cvx_idxes = []
        
        # nn_minn_pts = 800
        nn_minn_pts = 400
        # nn_minn_pts = 1000
        # nn_minn_pts = 100
        
        ### add losses --> add losses on meshes other than on pointclouds ###
        tot_bsz_assembled_def_cvx_pts = []
        
        for i_bsz in range(B):
            cur_bsz_valid_cvx_idxes = []
            ### 
            ### 1 x (tot_nn_basis x nn_cvx) x (nn_keypts x 3) --> total number of keypoints 
            # cur_bsz_basis = []
            # cur_bsz_ranges = []
            cur_bsz_cd_loss = []
            # tar_cage_extents_loss = losses.cage_extends_loss(outputs["new_cage"], tar_pc)
            #     tar_cage_offset_loss = losses.cage_verts_offset_loss(outputs["gt_new_cage"], outputs["new_cage"])
            cur_bsz_cage_extents_loss = []
            cur_bsz_cage_offset_loss = []
            
            cur_bsz_cvx_pts_idxes = []
            
            
            cur_bsz_tot_keypts_feats = torch.zeros((1, key_pts.size(1), 64), dtype=torch.float32).cuda()
            cur_bsz_def_pcs_offset = torch.zeros_like(src_pc[i_bsz: i_bsz + 1]) ### 1 x nn_pts x 3
            cur_bsz_tot_rnd_def_pcs_offset = [torch.zeros_like(src_pc[i_bsz: i_bsz + 1])  for _ in range(self.opt.rnd_sample_nn)]
            cur_bsz_tot_rnd_def_pcs_nn = [torch.zeros_like(cur_bsz_def_pcs_offset)[..., 0]   for _ in range(self.opt.rnd_sample_nn)]
            
            cur_bsz_rnd_def_pcs_offset = torch.zeros_like(src_pc[i_bsz: i_bsz + 1]) ### 1 x nn_pts x 3
            
            cur_bsz_def_pcs_nn = torch.zeros_like(cur_bsz_def_pcs_offset)[..., 0] ### 1 x nn_pts
            cur_bsz_cages = []
            cur_bsz_cages_ori = []
            cur_bsz_new_cages = []
            cur_bsz_rnd_new_cages = []
            
            cur_bsz_cages_faces = []
            cur_bsz_new_cages_faces = []
            
            cur_bsz_tot_rnd_new_cages = [[] for _ in range(self.opt.rnd_sample_nn)]
            
            cur_bsz_sorted_basis = []
            cur_bsz_cage_basis = []
            cur_bsz_cage_coefs = []
            cur_bsz_cage_weights_tot = []
            cur_bsz_src_scale_center = []
            cur_bsz_assembled_def_cvx_pts = []
            
            for i_cvx in range(nn_cvx): # i_cvx #
                
                import random
                ### get the keypts offset ##
                cur_src_pc_cvx_indicator = src_pc_cvx_indicator[:, :, i_cvx] ### bsz x nn_pts 
                # cur_src_keypts_cvx_indicator = src_keypts_cvx_indicator[:, :, i_cvx] ## bsz x nn_keypts 
                if self.opt.cvx_list_filter:
                    # ji
                    tar_i_cvx = i_cvx
                else:
                    # if i_cvx >= tar_pc_cvx_indicator.size(-1):
                    tar_i_cvx = random.choice(range(tar_pc_cvx_indicator.size(-1)))
                # tar_i_cvx = i_cvx
                print(f"i_cvx: {i_cvx}, tar_i_cvx: {tar_i_cvx}")
                ### 
                cur_tar_pc_cvx_indicator = tar_pc_cvx_indicator[:, :, tar_i_cvx]

                cur_pc_nn = torch.sum(cur_src_pc_cvx_indicator, dim=-1).long() ## bsz x nn_pc


                pc_idx_argsort = torch.argsort(cur_src_pc_cvx_indicator, dim=-1, descending=True) ### bsz x 
                # cur_keypts_nn = torch.sum(cur_src_keypts_cvx_indicator, dim=-1).long()
                # keypts_idx_argsort = torch.argsort(cur_src_keypts_cvx_indicator, dim=-1, descending=True)
                
                cur_tar_pc_nn = torch.sum(cur_tar_pc_cvx_indicator, dim=-1).long() ## bsz
                tar_pc_idx_argsort = torch.argsort(cur_tar_pc_cvx_indicator, dim=-1, descending=True) ### bsz x 

                
                
                cur_bsz_pc_nn = int(cur_pc_nn[i_bsz].item())
                # cur_bsz_keypts_nn = int(cur_keypts_nn[i_bsz].item()) ### keypts_nn
                cur_bsz_tar_pc_nn = int(cur_tar_pc_nn[i_bsz].item()) ### 
                
                
                
                # print(f"cur_bsz_pc_nn: {cur_bsz_pc_nn}, cur_bsz_keypts_nn: {cur_bsz_keypts_nn}, cur_bsz_tar_pc_nn: {cur_bsz_tar_pc_nn}")
                if cur_bsz_pc_nn <= nn_minn_pts or cur_bsz_tar_pc_nn <= nn_minn_pts:
                    continue
                # if cur_bsz_pc_nn <= nn_minn_pts:
                #     continue
                cur_bsz_valid_cvx_idxes.append(i_cvx)
                
                cur_bsz_pc_idx = pc_idx_argsort[i_bsz, :cur_bsz_pc_nn]
                # cur_bsz_keypts_idx = keypts_idx_argsort[i_bsz, :cur_bsz_keypts_nn] 
                cur_bsz_tar_pc_idx = tar_pc_idx_argsort[i_bsz, : cur_bsz_tar_pc_nn]
                
                cur_bsz_cvx_pts_idxes.append(cur_bsz_pc_idx)
                
                
                
                cur_bsz_src_cvx_pts = src_convex_pts[i_bsz, i_cvx, :].unsqueeze(0)
                cur_bsz_tar_cvx_pts = dst_convex_pts[i_bsz, tar_i_cvx, :].unsqueeze(0)
                
                cur_bsz_pc = src_pc[i_bsz, cur_bsz_pc_idx, :].unsqueeze(0)
                cur_bsz_tar_pc = tar_pc[i_bsz, cur_bsz_tar_pc_idx, :].unsqueeze(0) #### tar_pc_idx
                
                
                cur_bsz_pc, src_center, src_scale = utils.normalie_pc_bbox_batched(pc=cur_bsz_pc, rt_stats=True) ## mult by sclae --> real offsets
                # cur_bsz_keypts = (cur_bsz_keypts - src_center) / torch.clamp(src_scale, min=1e-6)
                cur_bsz_tar_pc, tar_center, tar_scale = utils.normalie_pc_bbox_batched(cur_bsz_tar_pc, rt_stats=True) 
                
                cur_bsz_src_scale_center.append((src_scale, src_center))
                
                
                cur_bsz_src_cvx_pts = (cur_bsz_src_cvx_pts - src_center) / torch.clamp(src_scale, min=EPS)
                cur_bsz_tar_cvx_pts = (cur_bsz_tar_cvx_pts - tar_center) / torch.clamp(tar_scale, min=EPS) ### tar_cvx_pts
                
                cur_bsz_src_cvx_pts = cur_bsz_pc
                cur_bsz_tar_cvx_pts = cur_bsz_tar_pc 
                ## cur_bsz_tar_pc --> for tar_pcs --> 
                
                
                outputs = self.cages(cur_bsz_pc, cur_bsz_tar_pc, cur_bsz_src_cvx_pts, cur_bsz_tar_cvx_pts, rnd_sample_nn=self.opt.rnd_sample_nn)
                
                ## deformed basis ## ## def_pc ##
                def_pc = outputs["deformed"]
                cd_loss = chamfer_distance(def_pc, cur_bsz_tar_pc) ### normalized!  
                ## cage extends loss ##
                tar_cage_extents_loss = losses.cage_extends_loss(outputs["new_cage"], cur_bsz_tar_pc) ## cages ##
                tar_cage_offset_loss = losses.cage_verts_offset_loss(outputs["gt_new_cage"], outputs["new_cage"]) ## cage offset loss
                
                tot_rnd_deformed_shps = outputs["tot_rnd_deformed_shapes"] ### rnd_deformed_shps ###
                
                cur_nnormed_def_pc = def_pc * src_scale + src_center
                cur_bsz_assembled_def_cvx_pts.append(cur_nnormed_def_pc)
                
                
                # "cage_def_basis": cage_def_basis, # cage_def_basis -> for def basis
                # "new_cage_coefs": new_cage_coefs,
                if self.opt.pred_type == "basis":
                    cur_bsz_cage_basis.append(outputs["cage_def_basis"]) ## cage_def_basis: bsz x n_b x (nk x 3 ) ## new_cage_coefs
                    cur_bsz_cage_coefs.append(outputs["new_cage_coefs"]) ## new_cage_coefs: bsz x n_b
                
                cur_bsz_cages_ori.append(outputs['cage'].clone())
                
                ''' Use unnormalized weights '''
                # cur_bsz_cage_unnormed_weights = outputs["weight_unnormed"]
                cur_bsz_cage_unnormed_weights = outputs["weight"]
                # print(f"cur_bsz_cage_unnormed_weights: {cur_bsz_cage_unnormed_weights.size()}")
                cur_bsz_cage_weights = torch.zeros((src_pc[i_bsz].size(0), cur_bsz_cage_unnormed_weights.size(-1)), dtype=torch.float32).cuda()
                cur_bsz_cage_weights[cur_bsz_pc_idx] = cur_bsz_cage_unnormed_weights[0]
                cur_bsz_cage_weights_tot.append(cur_bsz_cage_weights) ### nn_pts x nn_cur_cage_pts
                
                
                
                
                cur_cvx_cages = outputs['cage'] * src_scale + src_center
                cur_cvx_new_cages = outputs['new_cage'] * src_scale + src_center
                # cur_cvx_new_cages = outputs['new_cage'] * tar_scale + tar_center
                cur_cvx_rnd_new_cages =  outputs['rnd_new_cage'] * src_scale + src_center
                # cur_cvx_rnd_new_cages =  outputs['rnd_new_cage'] * tar_scale + tar_center
                cur_bsz_cages.append(cur_cvx_cages)
                cur_bsz_new_cages.append(cur_cvx_new_cages)
                cur_bsz_rnd_new_cages.append(cur_cvx_rnd_new_cages)
                
                cur_bsz_cages_faces.append(outputs["cage_face"])
                
                cur_cvx_tot_rnd_new_cages = outputs["tot_rnd_sample_new_cages"]
                for i_rnd in range(len(cur_cvx_tot_rnd_new_cages)):
                    cur_bsz_tot_rnd_new_cages[i_rnd].append(cur_cvx_tot_rnd_new_cages[i_rnd] * src_scale + src_center) 
                    # cur_bsz_tot_rnd_new_cages[i_rnd].append(cur_cvx_tot_rnd_new_cages[i_rnd] * src_scale + src_center) 
                ### yours v.s. DeepMetahandels ###
                
                
                cur_bsz_cd_loss.append(cd_loss)
                cur_bsz_cage_extents_loss.append(tar_cage_extents_loss)
                cur_bsz_cage_offset_loss.append(tar_cage_offset_loss)
                
                ''' offsets '''
                def_pc_offset = def_pc - cur_bsz_pc ### 1 x nn_cvx_pts x 3
                def_pc_offset = def_pc_offset * src_scale
                
                rnd_def_pc = outputs["rnd_deformed"]
                rnd_def_pc_offset = rnd_def_pc - cur_bsz_pc
                rnd_def_pc_offset = rnd_def_pc_offset * src_scale
                
                cur_bsz_def_pcs_offset[:, cur_bsz_pc_idx, :] += def_pc_offset
                cur_bsz_def_pcs_nn[:, cur_bsz_pc_idx] += 1
                
                cur_bsz_rnd_def_pcs_offset[:, cur_bsz_pc_idx, :] += rnd_def_pc_offset
                
                for i_s in range(len(tot_rnd_deformed_shps)):
                    cur_rnd_def_shp = tot_rnd_deformed_shps[i_s] - cur_bsz_pc
                    cur_rnd_def_shp_offset = cur_rnd_def_shp * src_scale
                    cur_bsz_tot_rnd_def_pcs_offset[i_s][:, cur_bsz_pc_idx, :] += cur_rnd_def_shp_offset
                    cur_bsz_tot_rnd_def_pcs_nn[i_s][:, cur_bsz_pc_idx] += 1
            
            
            tot_bsz_valid_cvx_idxes.append(cur_bsz_valid_cvx_idxes)
            
            tot_bsz_assembled_def_cvx_pts.append(cur_bsz_assembled_def_cvx_pts)
            
            ### cur_bsz_cages_ori ---> cages_ori
            tot_bsz_cages_ori.append(cur_bsz_cages_ori)
            
            ''' Instance level synchronization ''' 
            cur_bsz_tot_rnd_new_cages = [[] for _ in range(self.opt.rnd_sample_nn)] # cags 
            # cur_bsz_cage_coefs: bsz x n_b for each convex
            cur_bsz_cage_coefs_argsort = [torch.argsort(cur_coef, dim=-1) for cur_coef in cur_bsz_cage_coefs] ### rnd_new_cages
            n_basis = self.num_basis
            n_bsz = src_pc.size(0)
            # n_basis = cur_bsz_cage_coefs[0].size(-1) # number of basis #
            # n_bsz = cur_bsz_cage_coefs[0].size(0) # number of batch #
            n_cvx = len(cur_bsz_cage_coefs)
            


            sorted_coefs = []
            for i_cvx in range(n_cvx):
                cur_coef = cur_bsz_cage_coefs[i_cvx][:, cur_bsz_cage_coefs_argsort[i_cvx]]
                sorted_coefs.append(cur_coef.unsqueeze(1))
            if len(sorted_coefs) > 0:
                sorted_coefs  = torch.cat(sorted_coefs, dim=1) ## 1 x bsz x xxxxx
                avg_coefs = torch.mean(sorted_coefs, dim=1) # bsz x n_basis
                tot_avg_coefs.append(avg_coefs)

            cat_cur_bsz_cage_coefs_argsort = [cur_sort.unsqueeze(1) for cur_sort in cur_bsz_cage_coefs_argsort]
            if len(cat_cur_bsz_cage_coefs_argsort) > 0:
                cat_cur_bsz_cage_coefs_argsort = torch.cat(cat_cur_bsz_cage_coefs_argsort, dim=1)
            else:
                cat_cur_bsz_cage_coefs_argsort = torch.arange(self.num_basis, dtype=torch.long).unsqueeze(0).unsqueeze(0).repeat(1, n_cvx, 1).contiguous()
            tot_cat_cur_bsz_cage_coefs_argsort.append(cat_cur_bsz_cage_coefs_argsort)
            for i_s in range(self.opt.rnd_sample_nn):
                if len(cur_bsz_cage_coefs) > 0:
                    cur_sampled_rnd_s = torch.randn_like(cur_bsz_cage_coefs[0]) # bsz x n_b
                cur_s_cvx_cage = []
                for i_cvx in range(n_cvx):
                    cur_cvx_src_scale, cur_cvx_src_center = cur_bsz_src_scale_center[i_cvx]
                    cur_cvx_cage_argsort = cur_bsz_cage_coefs_argsort[i_cvx]
                    cur_cvx_cage_coefs = cur_bsz_cage_coefs[i_cvx] # bsz x n_b
                    cur_cvx_s_cage_coefs = cur_cvx_cage_coefs.clone()
                    # cur_cvx_s_cage_coefs[:, cur_cvx_cage_argsort] += cur_sampled_rnd_s * self.opt.coef_multiplier
                    cur_cvx_s_cage_coefs[:, cur_cvx_cage_argsort] = avg_coefs + cur_sampled_rnd_s * self.opt.coef_multiplier
                    # bsz x n_b x (nn_key x 3) ## cage basis ##
                    cur_cvx_s_new_flow =  cur_bsz_cage_basis[i_cvx] * cur_cvx_s_cage_coefs.unsqueeze(-1) 
                    
                    if i_s == 0: ## bsz
                        cur_bsz_cur_cvx_sorted_basis = cur_bsz_cage_basis[i_cvx][:, cur_cvx_cage_argsort]
                        cur_bsz_sorted_basis.append(cur_bsz_cur_cvx_sorted_basis)
                    
                    cur_cvx_s_new_flow = cur_cvx_s_new_flow.contiguous().view(n_bsz, n_basis, -1, 3).sum(dim=1)
                    
                    cur_cvx_s_new_cage = cur_bsz_cages_ori[i_cvx] +  cur_cvx_s_new_flow
                    cur_cvx_s_new_cage = cur_cvx_s_new_cage * cur_cvx_src_scale + cur_cvx_src_center
                    cur_s_cvx_cage.append(cur_cvx_s_new_cage)
                cur_bsz_tot_rnd_new_cages[i_s] = cur_s_cvx_cage
            # print(f"Here with cur_bsz_tot_rnd_new_cages: {len(cur_bsz_tot_rnd_new_cages)}")
                
            cur_bsz_sorted_basis = [cur_cvx_sorted_basis.unsqueeze(1) for cur_cvx_sorted_basis in cur_bsz_sorted_basis]
            cur_bsz_sorted_basis = torch.cat(cur_bsz_sorted_basis, dim=1)
            tot_sorted_basis.append(cur_bsz_sorted_basis)
            
            
            
            ### bsz def pcs offset --> 
            cur_bsz_def_pcs_offset = cur_bsz_def_pcs_offset / torch.clamp(cur_bsz_def_pcs_nn.unsqueeze(-1), min=EPS)
            cur_bsz_def_pc = src_pc[i_bsz: i_bsz + 1] + cur_bsz_def_pcs_offset ### unnormalized scale!
            
            cur_bsz_rnd_def_pcs_offset = cur_bsz_rnd_def_pcs_offset / torch.clamp(cur_bsz_def_pcs_nn.unsqueeze(-1), min=EPS)
            cur_bsz_rnd_def_pc = src_pc[i_bsz: i_bsz + 1] + cur_bsz_rnd_def_pcs_offset ### unnormalized scale!
            
            for i_s in range(len(cur_bsz_tot_rnd_def_pcs_offset)):
                cur_bsz_cur_rnd_def_pcs_offset = cur_bsz_tot_rnd_def_pcs_offset[i_s] / torch.clamp(cur_bsz_tot_rnd_def_pcs_nn[i_s].unsqueeze(-1), min=EPS)
                cur_bsz_cur_rnd_def_pc = src_pc[i_bsz: i_bsz + 1] + cur_bsz_cur_rnd_def_pcs_offset
                tot_rnd_tot_def_pcs[i_s].append(cur_bsz_cur_rnd_def_pc)
            
            tot_def_pcs.append(cur_bsz_def_pc)
            tot_rnd_def_pcs.append(cur_bsz_rnd_def_pc)
            
            if len(cur_bsz_cd_loss) > 0:
                cur_bsz_cd_loss = sum(cur_bsz_cd_loss) / float(len(cur_bsz_cd_loss))
                cur_bsz_cage_extents_loss = sum(cur_bsz_cage_extents_loss) / float(len(cur_bsz_cage_extents_loss))
                cur_bsz_cage_offset_loss = sum(cur_bsz_cage_offset_loss) /float(len(cur_bsz_cage_offset_loss)) 
            else:
                cur_bsz_cd_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                cur_bsz_cage_extents_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                cur_bsz_cage_offset_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                
            tot_cd_loss.append(cur_bsz_cd_loss)
            tot_cage_extents_loss.append(cur_bsz_cage_extents_loss)
            tot_cage_offset_loss.append(cur_bsz_cage_offset_loss)
            
        # try:
            ''' cages of the current batch ''' 
            ### cur_bsz_cages: 1 x tot_nn_pts x 3 --> 
            cat_ori_cages = torch.cat(cur_bsz_cages, dim=1) ### get cages for the merged cage 
            # cat_def_cages = self.glb_cages.cage_network.get_cages(cat_ori_cages) ## 1 x nn_cage pts x .. ## 
            cat_def_cages = self.glb_cages.cage_network.get_cages(src_pc[i_bsz: i_bsz + 1]) ## 1 x nn_cage pts x .. ## 
            def get_global_warping_by_cages(cur_towards_cages):
                ''' Global warping for cages ''' 
                ### warping for cages ### ## 1 x cage_nn_pts x 3
                # 
                # cat_cur_bsz_cages = torch.cat(cur_bsz_new_cages, dim=1) ### 1 x tot_cage_nn_pts x 3 ###
                if isinstance(cur_towards_cages, list):
                    cat_cur_bsz_cages = torch.cat(cur_towards_cages, dim=1) ### 1 x tot_cage_nn_pts x 3 ###
                else:
                    cat_cur_bsz_cages = cur_towards_cages
                cat_cur_bsz_cages = cat_cur_bsz_cages ### tot_cage_nn_pts x 3 ### ###  
                # self.cages(cur_bsz_pc, cur_bsz_tar_pc, cur_bsz_src_cvx_pts, cur_bsz_tar_cvx_pts)
                cat_cur_bsz_tar_pc = tar_pc[i_bsz: i_bsz + 1] ### 1 x nn_tar_pc x 3
                cur_cat_cages_cvx_pts = cat_cur_bsz_cages
                cur_cat_tar_pc_cvx_pts = cat_cur_bsz_tar_pc
                ### warp original predicted cages to cages of the target pc ### ### cat_cur_bsz_cages ### --> to target...
                glb_cages_outputs = self.glb_cages(cat_cur_bsz_cages, cat_cur_bsz_tar_pc, cur_cat_cages_cvx_pts, cur_cat_tar_pc_cvx_pts, rnd_sample_nn=self.opt.rnd_sample_nn)
                def_cur_bsz_cages = glb_cages_outputs["deformed"] ## 1 x tot_cage_nn_pts x 3 --> defpcs 
                ### cur_bsz_cat_cages: 
                cur_bsz_cat_cages = glb_cages_outputs["cage"] ## rpedicted glb cage  global new cages j ##cat_cages 
                ### use new cages for deformation for pure target-driven deformation ###
                cur_bsz_cat_new_cages = glb_cages_outputs["new_cage"] ### new cage ### cur_bsz_cat_new_cagesi
                cur_bsz_cat_get_cages = glb_cages_outputs["gt_new_cage"] ## 
                
                cur_tot_rnd_new_cages = glb_cages_outputs["tot_rnd_sample_new_cages"]
                

                ## a los term ##
                cur_bsz_cat_offset_loss = losses.cage_verts_offset_loss(cur_bsz_cat_get_cages, cur_bsz_cat_new_cages)
                ###### def_cur_bsz_cages: 1 x tot_cage_nn_pts x 3 ######
                # cur_bsz_new_cages = torch.split(def_cur_bsz_cages, cur_bsz_new_cages[0].size(1),  dim=1) ### as the new cage


                
                
                
                if self.training_dataset is not None:
                    #### cur_bsz_real_pc: bsz x nn_pc x 3 ####
                    cur_bsz_real_pc = self.training_dataset.get_pc_via_idx(idxx=None, return_batched=True, return_vertices=False)
                    cur_bsz_real_cages = self.glb_cages.cage_network.get_cages(cur_bsz_real_pc) #### bsz x nn_cages_pts x 3 --> nn_cages_pts
                    cur_bsz_cat_new_cages = cur_bsz_real_cages
                
                # cat_def_cages = utils.normalie_pc_bbox_batched(cat_def_cages)
                # cur_bsz_cat_cages = utils.normalie_pc_bbox_batched(cur_bsz_cat_cages)
                # cur_bsz_cat_new_cages = utils.normalie_pc_bbox_batched(cur_bsz_cat_new_cages)
                # cur_bsz_src_pc_for_def = utils.normalie_pc_bbox_batched(src_pc[i_bsz: i_bsz + 1])
                
                ### network with cages ###
                cur_bsz_src_pc_for_def = src_pc[i_bsz: i_bsz + 1] ## cur_bsz_src_pc_for_def --> cur_bsz_src_pc_for_def and others 
                
                
                
                ### Checking ###
                # print(f"Checking... maxx_cat_def_cages: {torch.max(cat_def_cages, dim=1)[0]}, minn_cat_def_cages: {torch.min(cat_def_cages, dim=1)[0]}, maxx_cur_bsz_cat_cages: {torch.max(cur_bsz_cat_cages, dim=1)[0]}, minn_cur_bsz_cat_cages: {torch.min(cur_bsz_cat_cages, dim=1)[0]}, maxx_src_pc: {torch.max(src_pc[i_bsz: i_bsz + 1], dim=1)[0]}, minn_src_pc: {torch.min(src_pc[i_bsz: i_bsz + 1], dim=1)[0]}")
                ## and category-level skinning 
                if "eyeglasses" in self.opt.data_dir and "none_motion" in self.opt.data_dir:
                    extents_mult = torch.tensor([1.0, 0.7, 1.0], dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(0)
                # elif "SAPIEN" in self.opt.data_dir:
                #     extents_mult = torch.tensor([0.8, 1.0, 0.8], dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(0)
                else:
                    extents_mult = torch.tensor([0.8, 0.8, 0.8], dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(0)
                # print(f"merged_cages: {merged_cages.size()}, merged_new_cages: {merged_new_cages.size()}, merged_faces: {merged_faces.size()}")
                ## merged_def
                ### source pc -> independent new_pc -> target pc 
                ### to the current cage ###
                merged_def_pc, weights, weights_unnormed = deform_with_MVC(cat_def_cages, ### new cage the the merged original cages ###
                                                                            cur_bsz_cat_cages, ## new cage of the merged cage
                                                                            # self.cages.cage_network.template_faces.expand(-1,-1,-1),
                                                                            self.glb_cages.cage_network.template_faces.expand(-1,-1,-1),
                                                                            # src_pc[i_bsz: i_bsz + 1] * extents_mult, # .transpose(1,2).contiguous(),
                                                                            cur_bsz_src_pc_for_def * extents_mult,
                                                                            # weights=cur_bsz_cage_weights_tot,
                                                                            verbose=True,)
                # tot_merged_def_pcs.append(merged_def_pc) ##### deform_with_MVC ###################
                ################### deform_with_MVC ###################
                new_cages_merged_def_pc, weights, weights_unnormed = deform_with_MVC(cat_def_cages, ### defjcats
                                                                            cur_bsz_cat_new_cages, ## use this cage for deformation ##
                                                                            # self.cages.cage_network.template_faces.expand(-1,-1,-1),
                                                                            self.glb_cages.cage_network.template_faces.expand(-1,-1,-1),
                                                                            # src_pc[i_bsz: i_bsz + 1] * extents_mult, # .transpose(1,2).contiguous(),
                                                                            cur_bsz_src_pc_for_def * extents_mult,
                                                                            # weights=cur_bsz_cage_weights_tot,
                                                                            verbose=True,)
                
                
                merged_def_pc /= extents_mult
                new_cages_merged_def_pc /= extents_mult
                
                merged_glb_def_pc, _, _ = deform_with_MVC(cur_bsz_cat_cages, cur_bsz_cat_new_cages, ## use this cage for deformation ##
                                                                            # self.cages.cage_network.template_faces.expand(-1,-1,-1),
                                                                            self.glb_cages.cage_network.template_faces.expand(-1,-1,-1),
                                                                            # src_pc[i_bsz: i_bsz + 1] * extents_mult, # .transpose(1,2).contiguous(),
                                                                            merged_def_pc * extents_mult,
                                                                            # weights=cur_bsz_cage_weights_tot,
                                                                            verbose=True,)
                merged_glb_def_pc /= extents_mult
                
                
                # tot_merged_rnd_def_pcs.append(new_cages_merged_def_pc)
                # return cur_bsz_cat_offset_loss, merged_def_pc, new_cages_merged_def_pc, cat_def_cages, cur_bsz_cat_cages, cur_tot_rnd_new_cages
                return cur_bsz_cat_offset_loss, merged_def_pc, merged_glb_def_pc, new_cages_merged_def_pc, cat_def_cages, cur_bsz_cat_cages, cur_tot_rnd_new_cages

            # cur_bsz_cat_offset_loss, merged_def_pc, new_cages_merged_def_pc, cur_bsz_cage_of_merged_cages, cur_bsz_cage_of_def_cages, cur_bsz_glb_tot_rnd_new_cages = get_global_warping_by_cages(cur_bsz_new_cages)
            
            cur_bsz_cat_offset_loss, pure_merged_def_pc, merged_def_pc, new_cages_merged_def_pc, cur_bsz_cage_of_merged_cages, cur_bsz_cage_of_def_cages, cur_bsz_glb_tot_rnd_new_cages = get_global_warping_by_cages(cur_bsz_def_pc)
            tot_merged_def_pcs.append(merged_def_pc) ##### deform_with_MVC
            # tot_pure
            tot_pure_merged_def_pcs.append(pure_merged_def_pc)
            tot_merged_rnd_def_pcs.append(new_cages_merged_def_pc)
            # print(f"cur_bsz_cage_of_merged_cages: {cur_bsz_cage_of_merged_cages.size()}")
            tot_cage_of_merged_cages.append(cur_bsz_cage_of_merged_cages)
            tot_cage_of_def_cages.append(cur_bsz_cage_of_def_cages) ## bsz x nn_cage_pts x 3 ##
            
                
            ## cur_bsz_glb_tot_rnd_new_cages --> for each sampled cage --> deform from source to target 
            for i_s in range(len(cur_bsz_tot_rnd_new_cages)):
                # cur_sampled_rnd_new_cages = cur_bsz_tot_rnd_new_cages[i_s]
                cur_sampled_rnd_new_cages = tot_rnd_tot_def_pcs[i_s][-1]
                cur_sampled_rnd_cat_offset_loss, _, cur_sampled_merged_def_pc, _, _, _, _ = get_global_warping_by_cages(cur_sampled_rnd_new_cages)
                tot_tot_sampled_merged_rnd_def_pcs[i_s].append(cur_sampled_merged_def_pc)
                
            cur_bsz_glb_rnd_def_pcs = []
            for i_s in range(len(cur_bsz_glb_tot_rnd_new_cages)):
                cur_bsz_src_pc_for_def = src_pc[i_bsz: i_bsz + 1] ## cur_bsz_src_pc_for_def --> cur_bsz_src_pc_for_def and others 
                ## and category-level skinning 
                # print(f"sampling {i_s}")
                if "eyeglasses" in self.opt.data_dir and "none_motion" in self.opt.data_dir:
                    extents_mult = torch.tensor([1.0, 0.7, 1.0], dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(0)
                # elif "SAPIEN" in self.opt.data_dir:
                #     extents_mult = torch.tensor([0.8, 1.0, 0.8], dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(0)
                else:
                    extents_mult = torch.tensor([0.8, 0.8, 0.8], dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(0)
                # print(f"merged_cages: {merged_cages.size()}, merged_new_cages: {merged_new_cages.size()}, merged_faces: {merged_faces.size()}")
                ## merged_def
                ### to the current cage ###
                
                cur_glb_rnd_cage = cur_bsz_glb_tot_rnd_new_cages[i_s]
                new_glb_rnd_def_pc, _, _ = deform_with_MVC(
                    # cat_def_cages, ### defjcats
                    cat_def_cages,
                                                        #    cur_bsz_cage_of_merged_cages.clone(),
                                                        cur_glb_rnd_cage, ## use this cage for deformation ##
                                                        self.cages.cage_network.template_faces.expand(-1,-1,-1),
                                                        # src_pc[i_bsz: i_bsz + 1] * extents_mult, # .transpose(1,2).contiguous(),
                                                        cur_bsz_src_pc_for_def * extents_mult,
                                                        # weights=cur_bsz_cage_weights_tot,
                                                        verbose=True,)
                
                new_glb_rnd_def_pc /= extents_mult
                tot_tot_sampled_glb_rnd_def_pcs[i_s].append(new_glb_rnd_def_pc)
                
                
                
                
            # except:
            #     tot_merged_def_pcs.append(cur_bsz_def_pc)
            #     tot_merged_rnd_def_pcs.append(cur_bsz_def_pc)
            #     cur_bsz_cat_offset_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            #     # print(f"others: {self.cages.cage_network.template_vertices.size()}")
            #     tot_cage_of_merged_cages.append(self.glb_cages.cage_network.template_vertices.expand(-1,-1,-1).transpose(-1, -2))
            #     tot_cage_of_def_cages.append(self.glb_cages.cage_network.template_vertices.expand(-1,-1,-1))
                
                
            #     for i_s in range(len(cur_bsz_tot_rnd_new_cages)):
            #         tot_tot_sampled_merged_rnd_def_pcs[i_s].append(cur_bsz_def_pc)
                
                
                
            ### global deformation...? no..
            tot_cat_offset_loss.append(cur_bsz_cat_offset_loss) ### cat offset loss
            
            
            
            ### cur_bsz_cages: 
            tot_cages.append(cur_bsz_cages)
            tot_new_cages.append(cur_bsz_new_cages)
            tot_rnd_new_cages.append(cur_bsz_rnd_new_cages)
            
            tot_cages_faces.append(cur_bsz_cages_faces)
            
            if self.opt.pred_type == "basis":
                tot_cage_basis.append(cur_bsz_cage_basis)
                # tot_cage_coefs.append(cur_bsz_def_pc)
                # tot_cage_basis.append(cur_bsz_def_pc)
                tot_cage_coefs.append(cur_bsz_cage_coefs)


        if len(tot_cd_loss) > 0:
            tot_cd_loss = sum(tot_cd_loss) / float(len(tot_cd_loss))
            tot_cage_extents_loss = sum(tot_cage_extents_loss) / float(len(tot_cage_extents_loss))
            tot_cage_offset_loss = sum(tot_cage_offset_loss) /float(len(tot_cage_offset_loss)) 
            tot_cat_offset_loss = sum(tot_cat_offset_loss) / float(len(tot_cat_offset_loss))
        else:
            tot_cd_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            tot_cage_extents_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            tot_cage_offset_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            tot_cat_offset_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean() ### cat offset loss ###
        tot_def_pcs = torch.cat(tot_def_pcs, dim=0) ### bsz x nn_pc x 3 --> 
        tot_rnd_def_pcs = torch.cat(tot_rnd_def_pcs, dim=0)
        tot_merged_def_pcs = torch.cat(tot_merged_def_pcs, dim=0)
        tot_pure_merged_def_pcs = torch.cat(tot_pure_merged_def_pcs, dim=0)
        tot_merged_rnd_def_pcs = torch.cat(tot_merged_rnd_def_pcs, dim=0)
        tot_cage_of_merged_cages = torch.cat(tot_cage_of_merged_cages, dim=0)
        tot_cage_of_def_cages = torch.cat(tot_cage_of_def_cages, dim=0)
        tot_cat_cur_bsz_cage_coefs_argsort = torch.cat(tot_cat_cur_bsz_cage_coefs_argsort, dim=0)
        
        tot_cvx_pts_idxes.append(cur_bsz_cvx_pts_idxes)
        
        for i_s in range(len(tot_rnd_tot_def_pcs)):
            tot_rnd_tot_def_pcs[i_s] = torch.cat(tot_rnd_tot_def_pcs[i_s], dim=0)
        
        
        for i_s in range(len(tot_tot_sampled_merged_rnd_def_pcs)):
            tot_tot_sampled_merged_rnd_def_pcs[i_s] = torch.cat(tot_tot_sampled_merged_rnd_def_pcs[i_s], dim=0)
            try:
                # print(f"Concating: {i_s}")
                tot_tot_sampled_glb_rnd_def_pcs[i_s] = torch.cat(tot_tot_sampled_glb_rnd_def_pcs[i_s], dim=0)
            except:
                tot_tot_sampled_glb_rnd_def_pcs[i_s] = tot_tot_sampled_merged_rnd_def_pcs[i_s]
        
        # tot_avg_coefs = []
        # tot_sorted_basis = []
        
        rt_dict = { 
            "cage": tot_cages,  ### cages, new cages, 
            "new_cage": tot_new_cages,
            "rnd_new_cage": tot_rnd_new_cages, 
            "cage_face": tot_cages_faces,
            "cd_loss": tot_cd_loss,
            "extents_loss": tot_cage_extents_loss,
            "offset_loss": tot_cage_offset_loss,
            "cat_offset_loss": tot_cat_offset_loss,
            "deformed": tot_def_pcs,
            "rnd_deformed": tot_rnd_def_pcs,
            "tot_rnd_tot_def_pcs": tot_rnd_tot_def_pcs,
            "merged_deformed": tot_merged_def_pcs,
            "pure_merged_deformed": tot_pure_merged_def_pcs,
            "merged_rnd_deformed": tot_merged_rnd_def_pcs,
            "tot_cage_basis": tot_cage_basis if self.opt.pred_type == "basis" else [],
            "tot_cage_coefs": tot_cage_coefs if self.opt.pred_type == "basis" else [],
            # "tot_tot_sampled_merged_rnd_def_pcs": tot_tot_sampled_glb_rnd_def_pcs, #  tot_tot_sampled_merged_rnd_def_pcs,
            "tot_tot_sampled_merged_rnd_def_pcs":  tot_tot_sampled_merged_rnd_def_pcs,
            "tot_cage_of_merged_cages": tot_cage_of_merged_cages,
            "tot_cage_of_def_cages": tot_cage_of_def_cages,
            "coefs_argsort": tot_cat_cur_bsz_cage_coefs_argsort,
            "tot_avg_coefs": tot_avg_coefs, ### avg-coefs, sorted-bais ###
            "tot_sorted_basis": tot_sorted_basis,
            "tot_bsz_cages_ori": tot_bsz_cages_ori,
            "tot_cvx_pts_idxes": tot_cvx_pts_idxes, ### total cvx pts idxes --> list of list of cvx pts idxes ###
            "tot_bsz_assembled_def_cvx_pts": tot_bsz_assembled_def_cvx_pts,  
            "tot_bsz_valid_cvx_idxes": tot_bsz_valid_cvx_idxes ## cvx idx list of all bszs ###  --> ...
        }
        return rt_dict


    
    def sample6(self, src_pc, tar_pc, key_pts, dst_key_pts, w_pc,  src_verts, src_convex_pts, tar_verts, dst_convex_pts, glb_net=None):
        # src_pc = 
        ### src_pc, tar_pc;
        ### src_convex_pts: 
        ### src_convex_pts ---- only use convexes with the same idx in those two shapes ###
        ### src_convex_pts: bsz x n_cvxes x n_pts_per_convex  ### forward3
        ### src_keypts: bsz x n_keypts x 3
        #### B, N, _ = src
        #### add src_faces, tar_faces for src_meshes and tar_meshes #### 
        #### network target driven ####
        B, N, _ = src_pc.shape
        nn_cvx = src_convex_pts.size(1) ### bsz x nn_cvx 
        
        src_convex_pts = src_convex_pts * 2.
        dst_convex_pts = dst_convex_pts * 2.
        
        # w_pc: bsz x n_pc x n_keypts
        # print(f"w_pc: {w_pc.size()}, w_pc_sum: {torch.sum(w_pc, dim=-1)[0, :2]}")
        # print(f"maxx_src: {torch.max(src_pc)}, minn_src: {torch.min(src_pc)}, maxx src_convex_pts: {torch.max(src_convex_pts)}, minn src_convex_pts: {torch.min(src_convex_pts)}, ")
        
        ###  src ###
        avg_src_pc_cvx = self.get_minn_distances_base_pc_cvx_pts(src_pc, src_convex_pts) ### 
        ### bsz x nn_pts x nn_cvx --> []
        src_pc_cvx_indicator = self.get_base_pc_cvx_indicator(src_pc, src_convex_pts, avg_src_pc_cvx) ### 
        # src_keypts_cvx_indicator = self.get_base_pc_cvx_indicator(key_pts, src_convex_pts, avg_src_pc_cvx) ### 
        
        avg_dst_pc_cvx = self.get_minn_distances_base_pc_cvx_pts(tar_pc, dst_convex_pts)
        tar_pc_cvx_indicator = self.get_base_pc_cvx_indicator(tar_pc, dst_convex_pts, avg_dst_pc_cvx) ### base-pc-cvx
        
        
        tot_cd_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean()
        
        
        EPS = 1e-6
        
        tot_def_pcs = []
        tot_rnd_def_pcs = []
        tot_merged_def_pcs = []
        tot_merged_rnd_def_pcs = []
        tot_cage_extents_loss = []
        tot_cd_loss = []
        tot_cage_offset_loss = []
        
        tot_cages = []
        tot_new_cages = []
        tot_cages_faces = []
        tot_rnd_new_cages = []
        # tot_new_cages_faces = []
        
        tot_cage_basis = []
        tot_cage_coefs = []
        
        nn_minn_pts = 800
        # nn_minn_pts = 1000
        # nn_minn_pts = 100
        
        ### add losses --> add losses on meshes other than on pointclouds ###
        
        for i_bsz in range(B):
            
            ### 1 x (tot_nn_basis x nn_cvx) x (nn_keypts x 3) --> total number of keypoints 
            cur_bsz_basis = []
            cur_bsz_ranges = []
            cur_bsz_cd_loss = []
            # tar_cage_extents_loss = losses.cage_extends_loss(outputs["new_cage"], tar_pc)
            #     tar_cage_offset_loss = losses.cage_verts_offset_loss(outputs["gt_new_cage"], outputs["new_cage"])
            # cur_bsz_
            cur_bsz_cage_extents_loss = []
            cur_bsz_cage_offset_loss = []
            
            
            cur_bsz_tot_keypts_feats = torch.zeros((1, key_pts.size(1), 64), dtype=torch.float32).cuda()
            cur_bsz_def_pcs_offset = torch.zeros_like(src_pc[i_bsz: i_bsz + 1]) ### 1 x nn_pts x 3
            
            cur_bsz_rnd_def_pcs_offset = torch.zeros_like(src_pc[i_bsz: i_bsz + 1]) ### 1 x nn_pts x 3
            
            cur_bsz_def_pcs_nn = torch.zeros_like(cur_bsz_def_pcs_offset)[..., 0] ### 1 x nn_pts
            cur_bsz_cages = []
            cur_bsz_new_cages = []
            cur_bsz_rnd_new_cages = []
            cur_bsz_cages_faces = []
            cur_bsz_new_cages_faces = []
            
            cur_bsz_cage_basis = []
            cur_bsz_cage_coefs = []
            cur_bsz_cage_weights_tot = []
            
            for i_cvx in range(nn_cvx):
                
                
                ### get the keypts offset ##
                cur_src_pc_cvx_indicator = src_pc_cvx_indicator[:, :, i_cvx] ### bsz x nn_pts 
                # cur_src_keypts_cvx_indicator = src_keypts_cvx_indicator[:, :, i_cvx] ## bsz x nn_keypts 
                cur_tar_pc_cvx_indicator = tar_pc_cvx_indicator[:, :, i_cvx]
                
                cur_pc_nn = torch.sum(cur_src_pc_cvx_indicator, dim=-1).long() ## bsz x nn_pc
                
                
                #### 
                pc_idx_argsort = torch.argsort(cur_src_pc_cvx_indicator, dim=-1, descending=True) ### bsz x 
                # cur_keypts_nn = torch.sum(cur_src_keypts_cvx_indicator, dim=-1).long()
                # keypts_idx_argsort = torch.argsort(cur_src_keypts_cvx_indicator, dim=-1, descending=True)
                
                cur_tar_pc_nn = torch.sum(cur_tar_pc_cvx_indicator, dim=-1).long() ## bsz
                tar_pc_idx_argsort = torch.argsort(cur_tar_pc_cvx_indicator, dim=-1, descending=True) ### bsz x 

                
                
                cur_bsz_pc_nn = int(cur_pc_nn[i_bsz].item())
                # cur_bsz_keypts_nn = int(cur_keypts_nn[i_bsz].item()) ### keypts_nn
                cur_bsz_tar_pc_nn = int(cur_tar_pc_nn[i_bsz].item()) ### 
                
                
                
                # print(f"cur_bsz_pc_nn: {cur_bsz_pc_nn}, cur_bsz_keypts_nn: {cur_bsz_keypts_nn}, cur_bsz_tar_pc_nn: {cur_bsz_tar_pc_nn}")
                if cur_bsz_pc_nn <= nn_minn_pts or cur_bsz_tar_pc_nn <= nn_minn_pts:
                    continue
                cur_bsz_pc_idx = pc_idx_argsort[i_bsz, :cur_bsz_pc_nn]
                # cur_bsz_keypts_idx = keypts_idx_argsort[i_bsz, :cur_bsz_keypts_nn] 
                cur_bsz_tar_pc_idx = tar_pc_idx_argsort[i_bsz, : cur_bsz_tar_pc_nn]
                
                
                
                cur_bsz_src_cvx_pts = src_convex_pts[i_bsz, i_cvx, :].unsqueeze(0)
                cur_bsz_tar_cvx_pts = dst_convex_pts[i_bsz, i_cvx, :].unsqueeze(0)
                
                cur_bsz_pc = src_pc[i_bsz, cur_bsz_pc_idx, :].unsqueeze(0)
                cur_bsz_tar_pc = tar_pc[i_bsz, cur_bsz_tar_pc_idx, :].unsqueeze(0) #### tar_pc_idx
                
                
                cur_bsz_pc, src_center, src_scale = utils.normalie_pc_bbox_batched(pc=cur_bsz_pc, rt_stats=True) ## mult by sclae --> real offsets
                # cur_bsz_keypts = (cur_bsz_keypts - src_center) / torch.clamp(src_scale, min=1e-6)
                cur_bsz_tar_pc, tar_center, tar_scale = utils.normalie_pc_bbox_batched(cur_bsz_tar_pc, rt_stats=True) 
                
                
                # cur_bsz_src_cvx_pts = (cur_bsz_src_cvx_pts - src_center) / torch.clamp(src_scale, min=EPS)
                # cur_bsz_tar_cvx_pts = (cur_bsz_tar_cvx_pts - tar_center) / torch.clamp(tar_scale, min=EPS) ### tar_cvx_pts
                
                cur_bsz_src_cvx_pts = cur_bsz_pc
                cur_bsz_tar_cvx_pts = cur_bsz_tar_pc
                
                
                outputs = self.cages(cur_bsz_pc, cur_bsz_tar_pc, cur_bsz_src_cvx_pts, cur_bsz_tar_cvx_pts)
                
                def_pc = outputs["deformed"]
                cd_loss = chamfer_distance(def_pc, cur_bsz_tar_pc) ### normalized! 
                tar_cage_extents_loss = losses.cage_extends_loss(outputs["new_cage"], cur_bsz_tar_pc)
                tar_cage_offset_loss = losses.cage_verts_offset_loss(outputs["gt_new_cage"], outputs["new_cage"])
                
                # "cage_def_basis": cage_def_basis,
            # "new_cage_coefs": new_cage_coefs,
                if self.opt.pred_type == "basis":
                    cur_bsz_cage_basis.append(outputs["cage_def_basis"]) ## cage_def_basis: bsz x n_b x (nk x 3 ) ## new_cage_coefs
                    cur_bsz_cage_coefs.append(outputs["new_cage_coefs"]) ## new_cage_coefs: bsz x n_b
                
                ''' Use unnormalized weights '''
                # cur_bsz_cage_unnormed_weights = outputs["weight_unnormed"]
                cur_bsz_cage_unnormed_weights = outputs["weight"]
                # print(f"cur_bsz_cage_unnormed_weights: {cur_bsz_cage_unnormed_weights.size()}")
                cur_bsz_cage_weights = torch.zeros((src_pc[i_bsz].size(0), cur_bsz_cage_unnormed_weights.size(-1)), dtype=torch.float32).cuda()
                cur_bsz_cage_weights[cur_bsz_pc_idx] = cur_bsz_cage_unnormed_weights[0]
                cur_bsz_cage_weights_tot.append(cur_bsz_cage_weights) ### nn_pts x nn_cur_cage_pts
                
                
                
                
                cur_cvx_cages = outputs['cage'] * src_scale + src_center
                cur_cvx_new_cages = outputs['new_cage'] * src_scale + src_center
                # cur_cvx_new_cages = outputs['new_cage'] * tar_scale + tar_center
                cur_cvx_rnd_new_cages =  outputs['rnd_new_cage'] * src_scale + src_center
                # cur_cvx_rnd_new_cages =  outputs['rnd_new_cage'] * tar_scale + tar_center
                cur_bsz_cages.append(cur_cvx_cages)
                cur_bsz_new_cages.append(cur_cvx_new_cages)
                cur_bsz_rnd_new_cages.append(cur_cvx_rnd_new_cages)
                
                cur_bsz_cages_faces.append(outputs["cage_face"])
                
                
                
                cur_bsz_cd_loss.append(cd_loss)
                cur_bsz_cage_extents_loss.append(tar_cage_extents_loss)
                cur_bsz_cage_offset_loss.append(tar_cage_offset_loss)
                
                ''' offsets '''
                def_pc_offset = def_pc - cur_bsz_pc ### 1 x nn_cvx_pts x 3
                def_pc_offset = def_pc_offset * src_scale
                
                rnd_def_pc = outputs["rnd_deformed"]
                rnd_def_pc_offset = rnd_def_pc - cur_bsz_pc
                rnd_def_pc_offset = rnd_def_pc_offset * src_scale
                
                cur_bsz_def_pcs_offset[:, cur_bsz_pc_idx, :] += def_pc_offset
                cur_bsz_def_pcs_nn[:, cur_bsz_pc_idx] += 1
                
                cur_bsz_rnd_def_pcs_offset[:, cur_bsz_pc_idx, :] += rnd_def_pc_offset
            
            
            cur_bsz_def_pcs_offset = cur_bsz_def_pcs_offset / torch.clamp(cur_bsz_def_pcs_nn.unsqueeze(-1), min=EPS)
            cur_bsz_def_pc = src_pc[i_bsz: i_bsz + 1] + cur_bsz_def_pcs_offset ### unnormalized scale!
            
            cur_bsz_rnd_def_pcs_offset = cur_bsz_rnd_def_pcs_offset / torch.clamp(cur_bsz_def_pcs_nn.unsqueeze(-1), min=EPS)
            cur_bsz_rnd_def_pc = src_pc[i_bsz: i_bsz + 1] + cur_bsz_rnd_def_pcs_offset ### unnormalized scale!
            
            tot_def_pcs.append(cur_bsz_def_pc)
            tot_rnd_def_pcs.append(cur_bsz_rnd_def_pc)
            
            if len(cur_bsz_cd_loss) > 0:
                cur_bsz_cd_loss = sum(cur_bsz_cd_loss) / float(len(cur_bsz_cd_loss))
                cur_bsz_cage_extents_loss = sum(cur_bsz_cage_extents_loss) / float(len(cur_bsz_cage_extents_loss))
                cur_bsz_cage_offset_loss = sum(cur_bsz_cage_offset_loss) /float(len(cur_bsz_cage_offset_loss)) 
            else:
                cur_bsz_cd_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                cur_bsz_cage_extents_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                cur_bsz_cage_offset_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                
            tot_cd_loss.append(cur_bsz_cd_loss)
            tot_cage_extents_loss.append(cur_bsz_cage_extents_loss)
            tot_cage_offset_loss.append(cur_bsz_cage_offset_loss)
            
            tot_cages.append(cur_bsz_cages)
            tot_new_cages.append(cur_bsz_new_cages)
            tot_rnd_new_cages.append(cur_bsz_rnd_new_cages)
            
            tot_cages_faces.append(cur_bsz_cages_faces)
            
            if self.opt.pred_type == "basis":
                tot_cage_basis.append(cur_bsz_cage_basis)
                tot_cage_coefs.append(cur_bsz_cage_coefs)
            
            tot_cages_no_batched = [cur_cage[0] for cur_cage in cur_bsz_cages]
            tot_new_cages_no_batched = [cur_cage[0] for cur_cage in cur_bsz_new_cages]
            tot_cages_faces_no_batched = [cur_cage[0] for cur_cage in cur_bsz_cages_faces]
            tot_rnd_new_cages_no_batched = [cur_cage[0] for cur_cage in cur_bsz_rnd_new_cages]
            
            ### 
            
            try:
                cur_bsz_cage_weights_tot = torch.cat(cur_bsz_cage_weights_tot, dim=0) ### nn_pc x (nn_tot_cage_pts) 
                cur_bsz_cage_weights_tot_sum = torch.sum(cur_bsz_cage_weights_tot, dim=-1)
                cur_bsz_cage_weights_tot_sum = torch.where(cur_bsz_cage_weights_tot_sum == 0., torch.ones_like(cur_bsz_cage_weights_tot_sum), cur_bsz_cage_weights_tot_sum)
                cur_bsz_cage_weights_tot = cur_bsz_cage_weights_tot / cur_bsz_cage_weights_tot_sum.unsqueeze(-1)
                cur_bsz_cage_weights_tot = cur_bsz_cage_weights_tot.unsqueeze(0)
            
                
                
                merged_cages, merged_faces = model_utils.merge_meshes_torch(tot_cages_no_batched, tot_cages_faces_no_batched)
                # merged_cages, merged_faces = model_utils.merge_meshes_torch(tot_cages_no_batched, tot_cages_faces_no_batched)
                merged_new_cages, merged_new_faces = model_utils.merge_meshes_torch(tot_new_cages_no_batched, tot_cages_faces_no_batched)
                merged_rnd_new_cages, merged_rnd_new_faces = model_utils.merge_meshes_torch(tot_rnd_new_cages_no_batched, tot_cages_faces_no_batched)
                # print(f"maxx_merged_cages: {torch.max(merged_cages, dim=0)[0]}, minn_merged_cages: {torch.min(merged_cages, dim=0)[0]}, maxx_merged_new_cages: {torch.max(merged_new_cages, dim=0)[0]}, minn_merged_new_cages: {torch.min(merged_new_cages, dim=0)[0]}, ")
                # print(f"maxx_src_pc: {torch.max(src_pc[i_bsz: i_bsz + 1], dim=0)[0]}, minn_src_pc: {torch.min(src_pc[i_bsz: i_bsz + 1], dim=0)[0]}, src_pc[i_bsz: i_bsz + 1]: ")
                
                merged_cages_np = merged_cages.detach().cpu().numpy()
                merged_faces_np = merged_faces.detach().cpu().numpy()
                merged_rnd_new_cages_np = merged_rnd_new_cages.detach().cpu().numpy()
                # sv_merged_fn = "test_cages.obj"
                # utils.save_obj_file(merged_new_cages.detach().cpu().numpy(), merged_faces_np.tolist(), sv_merged_fn, add_one=True)
                
                merged_cages = merged_cages.unsqueeze(0)
                merged_new_cages = merged_new_cages.unsqueeze(0)
                merged_rnd_new_cages = merged_rnd_new_cages.unsqueeze(0)
                merged_faces = merged_faces.unsqueeze(0)
                
                
                # print(f"merged_cages: {merged_cages.size()}, merged_new_cages: {merged_new_cages.size()}, merged_faces: {merged_faces.size()}")
                merged_def_pc, weights, weights_unnormed = deform_with_MVC(merged_cages,
                                                                            merged_new_cages,
                                                                            # merged_cages,
                                                                            #  self.cages.cage_network.template_faces.expand(B,-1,-1),
                                                                            merged_faces,
                                                                            src_pc[i_bsz: i_bsz + 1], # .transpose(1,2).contiguous(),
                                                                            weights=cur_bsz_cage_weights_tot,
                                                                            verbose=True,)
                tot_merged_def_pcs.append(merged_def_pc)
                
                merged_rnd_def_pc, weights, weights_unnormed = deform_with_MVC(merged_cages,
                                                                            merged_rnd_new_cages,
                                                                            # merged_cages,
                                                                            #  self.cages.cage_network.template_faces.expand(B,-1,-1),
                                                                            merged_faces,
                                                                            src_pc[i_bsz: i_bsz + 1], # .transpose(1,2).contiguous(),
                                                                            weights=cur_bsz_cage_weights_tot,
                                                                            verbose=True)
                tot_merged_rnd_def_pcs.append(merged_rnd_def_pc)
            except:
                tot_merged_def_pcs.append(cur_bsz_def_pc)
                tot_merged_rnd_def_pcs.append(cur_bsz_def_pc)
                
        if len(tot_cd_loss) > 0:
            tot_cd_loss = sum(tot_cd_loss) / float(len(tot_cd_loss))
            tot_cage_extents_loss = sum(tot_cage_extents_loss) / float(len(tot_cage_extents_loss))
            tot_cage_offset_loss = sum(tot_cage_offset_loss) /float(len(tot_cage_offset_loss)) 
        else:
            tot_cd_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            tot_cage_extents_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean()
            tot_cage_offset_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean()
        tot_def_pcs = torch.cat(tot_def_pcs, dim=0) ### bsz x nn_pc x 3 --> 
        tot_rnd_def_pcs = torch.cat(tot_rnd_def_pcs, dim=0)
        tot_merged_def_pcs = torch.cat(tot_merged_def_pcs, dim=0)
        tot_merged_rnd_def_pcs = torch.cat(tot_merged_rnd_def_pcs, dim=0)
        
        rt_dict = {
            "cage": tot_cages, 
            "new_cage": tot_new_cages,
            "rnd_new_cage": tot_rnd_new_cages,
            "cage_face": tot_cages_faces,
            "cd_loss": tot_cd_loss,
            "extents_loss": tot_cage_extents_loss,
            "offset_loss": tot_cage_offset_loss,
            "deformed": tot_def_pcs,
            "rnd_deformed": tot_rnd_def_pcs,
            "merged_deformed": tot_merged_def_pcs, 
            "merged_rnd_deformed": tot_merged_rnd_def_pcs,
            "tot_cage_basis": tot_cage_basis if self.opt.pred_type == "basis" else [],
            "tot_cage_coefs": tot_cage_coefs if self.opt.pred_type == "basis" else []
        }
        return rt_dict
         
    ### render meshes. .. ###
    def render_mesh(self, mesh_v_nx3, mesh_f_fx3, camera_mv_bx4x4, resolution=256, hierarchical_mask=False):
        
        
        # Generate random camera
        with torch.no_grad():
            # if camera is None:
            campos, cam_mv, rotation_angle, elevation_angle, sample_r = self.generate_random_camera(
                1, n_views=self.n_views)
            # gen_camera = (campos, cam_mv, sample_r, rotation_angle, elevation_angle)
            # run_n_view = self.n_views

        # Render the mesh into 2D image (get 3d position of each image plane)
        # antilias_mask, hard_mask, return_value = self.render_mesh(mesh_v, mesh_f, cam_mv)
        
        return_value = dict()
        if self.render_type == 'neural_render':
            tex_pos, mask, hard_mask, rast, v_pos_clip, mask_pyramid, depth = self.renderer.render_mesh(
                mesh_v_nx3.unsqueeze(dim=0), ### n x 3
                mesh_f_fx3.int(), ### 
                cam_mv,
                mesh_v_nx3.unsqueeze(dim=0),
                resolution=resolution,
                device=self.device,
                hierarchical_mask=hierarchical_mask
            )

            return_value['tex_pos'] = tex_pos
            return_value['mask'] = mask
            return_value['hard_mask'] = hard_mask
            return_value['rast'] = rast
            return_value['v_pos_clip'] = v_pos_clip
            return_value['mask_pyramid'] = mask_pyramid
            return_value['depth'] = depth
        else:
            raise NotImplementedError

        return return_value
    
    def generate_random_camera(self, batch_size, n_views=2):
        '''
        Sample a random camera from the camera distribution during training
        :param batch_size: batch size for the generator
        :param n_views: number of views for each shape within a batch
        :return:
        '''
        # carla
        sample_r = None
        world2cam_matrix, forward_vector, camera_origin, rotation_angle, elevation_angle = sample_camera(
            self.data_camera_mode, batch_size * n_views, 'cuda')
        mv_batch = world2cam_matrix
        campos = camera_origin
        return campos.reshape(batch_size, n_views, 3), mv_batch.reshape(batch_size, n_views, 4, 4), \
               rotation_angle, elevation_angle, sample_r
