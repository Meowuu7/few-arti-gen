import imp
from multiprocessing.sharedctypes import Value
# from platform import java_ver
# from quad_mesh_simplify import simplify_mesh
import data_utils
import numpy as np
import os
from options.preprocess_options import opt
import utils.binvox_rw as binvox_rw

def test_mesh_simplification(obj_fn):
  cur_vertices, cur_faces = data_utils.read_obj_file_ours(obj_fn=obj_fn, minus_one=True)
  cur_faces = np.array(cur_faces, dtype=np.uint32)
  
  print(f"cur_facses; {cur_faces.shape}")
  print(f"cur_verticess: {cur_vertices.shape}")
  new_positions, new_face = simplify_mesh(cur_vertices, cur_faces, cur_vertices.shape[0] // 5, threshold=0.5)
  new_faces = [new_face[i] for i in range(new_face.shape[0])]
  tmp_sv_fn = "tmp.obj"
  data_utils.save_obj_file(new_positions, new_faces, tmp_sv_fn, add_one=True)

def simplify_category(fr_root_folder, sv_root_folder):
  tot_obj_fns = os.listdir(fr_root_folder)
  tot_sv_obj_fns = os.listdir(sv_root_folder)
  tot_obj_fns = [fn for fn in tot_obj_fns if os.path.isdir(os.path.join(fr_root_folder, fn))]
  tot_sv_obj_fns = [fn for fn in tot_sv_obj_fns if os.path.isdir(os.path.join(sv_root_folder, fn)) ]
  tot_obj_fns = [fn for fn in tot_obj_fns if fn not in tot_sv_obj_fns]
  for cur_obj_fn in tot_obj_fns:
    print(f"Processing obj: {cur_obj_fn}...")
    cur_obj_folder = os.path.join(fr_root_folder, cur_obj_fn)
    cur_obj_sv_folder = os.path.join(sv_root_folder, cur_obj_fn)
    if os.path.exists(cur_obj_sv_folder):
      continue
    os.makedirs(cur_obj_sv_folder, exist_ok=True)
    cur_obj_parts = os.listdir(cur_obj_folder)
    cur_obj_parts = [fn for fn in cur_obj_parts if fn.endswith(".obj")]
    for cur_part in cur_obj_parts:
      # pritn()
      tot_cur_part_fn = os.path.join(cur_obj_folder, cur_part)
      tot_cur_part_sv_fn = os.path.join(cur_obj_sv_folder, cur_part)
      cur_vertices, cur_faces = data_utils.read_obj_file_ours(obj_fn=tot_cur_part_fn, minus_one=True)
      cur_faces = np.array(cur_faces, dtype=np.uint32)
      print(f"current part obj fn: {tot_cur_part_fn}, cur_vertices: {cur_vertices.shape}, cur_faces: {cur_faces.shape}")
      new_positions, new_face = simplify_mesh(cur_vertices, cur_faces, min(cur_vertices.shape[0] // 5, maxx_verts), threshold=0.5)
      new_faces = [new_face[i] for i in range(new_face.shape[0])]
      # tmp_sv_fn = "tmp.obj"
      data_utils.save_obj_file(new_positions, new_faces, tot_cur_part_sv_fn, add_one=True)

# maxx_verts = 250
# motion_cat = "eyeglasses"
# fr_root_folder = "/nas/datasets/gen/datasets/MotionDataset_processed"
# sv_root_folder = f"/nas/datasets/gen/datasets/MotionDataset_processed_sim_{maxx_verts}"
# os.makedirs(sv_root_folder, exist_ok=True)
# fr_root_folder = os.path.join(fr_root_folder, motion_cat)
# sv_root_folder = os.path.join(sv_root_folder, motion_cat)
# os.makedirs(sv_root_folder, exist_ok=True)

# simplify_category(fr_root_folder, sv_root_folder)


# obj_fn = "/nas/datasets/gen/datasets/MotionDataset_processed/eyeglasses/0006/none_motion.obj"
# obj_fn = "/nas/datasets/gen/datasets/MotionDataset_processed/water_bottle/0022/none_motion.obj"
# test_mesh_simplification(obj_fn=obj_fn)




# def modify_file_names(folder):
#   obj_fns = os.listdir(folder)
#   obj_fns = [fn for fn in obj_fns if fn.endswith(".obj")]
#   for obj_fn in obj_fns:
#     cur_obj_fn = os.path.join(folder, obj_fn)
#     modified_fn = "{:03d}".format(int(obj_fn.split(".")[0]))
#     print(modified_fn)
#     cur_modified_fn = os.path.join(folder, modified_fn + ".obj")
#     os.system(f"mv {cur_obj_fn} {cur_modified_fn}")

# root_folder = "/home/xueyi/gen/neuralSubdiv/data_meshes/screw_1"
# sub_folders = os.listdir(root_folder)
# sub_folders = [fn for fn in sub_folders if os.path.isdir(os.path.join(root_folder, fn))]
# for fn in sub_folders:
#   cur_sub_folder = os.path.join(root_folder, fn)
#   modify_file_names(cur_sub_folder)



def trimesh_voxilization(fr_cat_folder, sv_cat_folder):
    import trimesh
    from trimesh.voxel import creation
    pitch = 0.01

    insts = os.listdir(fr_cat_folder)
    insts = [fn for fn in insts if os.path.isdir(os.path.join(fr_cat_folder, fn))]
    for inst in insts:
        sv_inst_fn = os.path.join(sv_cat_folder, inst)
        os.makedirs(sv_inst_fn, exist_ok=True)
        fr_inst_fn = os.path.join(fr_cat_folder, inst)
        objs = os.listdir(fr_inst_fn)
        objs = [fn for fn in objs if fn.endswith(".obj")]
        for obj_fn in objs:
            cur_obj_fn = os.path.join(fr_inst_fn, obj_fn)
            cur_obj_sv_fn = os.path.join(sv_inst_fn, obj_fn)
            cur_verts, cur_faces = data_utils.read_obj_file_ours(cur_obj_fn, minus_one=True)
            if cur_verts.shape[0] == 0:
              continue
            cur_verts = data_utils.center_vertices(cur_verts)
            cur_verts = data_utils.normalize_vertices_scale(cur_verts)
            cur_mesh = trimesh.Trimesh(vertices=cur_verts, faces=cur_faces)
            # voxels = creation.voxelize(cur_mesh, pitch=0.005, method='subdivide')
            voxels = creation.voxelize(cur_mesh, pitch=pitch, method='subdivide')
            tot_pts = []
            n_x, n_y, n_z = voxels.shape
            for i_x in range(n_x):
                for i_y in range(n_y):
                    for i_z in range(n_z):
                        cur_voxel_val = voxels.matrix[i_x, i_y, i_z].item()
                        if cur_voxel_val > 0.1:
                            # print(i_x, i_y, i_z, cur_voxel_val)
                            tot_pts.append([i_x, i_y, i_z])

            tot_pts = np.array(tot_pts, dtype=np.float32)
            tot_pts = data_utils.center_vertices(tot_pts)
            tot_pts = data_utils.normalize_vertices_scale(tot_pts)
            
            dist_cur_verts_tot_res_verts = np.sum((np.reshape(cur_verts, (cur_verts.shape[0], 1, 3)) - np.reshape(tot_pts, (1, tot_pts.shape[0], 3)) ) ** 2, axis=-1) # cur_verts.shape[0] x tot_res_verts_pts.shape[0]
            minn_cur_verts_tot_res_verts_idx = np.argmin(dist_cur_verts_tot_res_verts, axis=-1).tolist() # cur_verts.ahepe[0]
            # cur_vert_idx_to_tot_vert_idx: 
            cur_vert_idx_to_tot_vert_idx = {ii : kk  for ii, kk in enumerate(minn_cur_verts_tot_res_verts_idx)}
            tot_res_faces = []
            for cur_face in cur_faces:
                cur_res_face = [cur_vert_idx_to_tot_vert_idx[f_idx] for f_idx in cur_face]
                tot_res_faces.append(cur_res_face)
            cur_faces = tot_res_faces
            data_utils.save_obj_file(tot_pts, cur_faces, cur_obj_sv_fn, add_one=True)
        
            

# fr_rt_folder = "/nas/datasets/gen/datasets/MotionDataset_processed"
# sv_rt_folder = "/nas/datasets/gen/datasets/MotionDataset_processed_voxelized_0.01"
# os.makedirs(sv_rt_folder, exist_ok=True)
# tot_motion_cats = os.listdir(fr_rt_folder)

# # tot_motion_cats = [fn for fn in tot_motion_cats if os.path.isdir(os.path.join(fr_rt_folder, fn)) and fn != "globe" and fn != "screwdriver"]
# sv_tot_motion_cats = os.listdir(sv_rt_folder)

# sv_tot_motion_cats = [fn for fn in sv_tot_motion_cats if os.path.isdir(os.path.join(fr_rt_folder, fn))]
# tot_motion_cats = [fn for fn in tot_motion_cats if fn not in sv_tot_motion_cats]
# for cur_motion_cat in tot_motion_cats:
#   print(f"Processing motion category: {cur_motion_cat}")
#   cur_motion_cat_sv = os.path.join(sv_rt_folder, cur_motion_cat)
#   os.makedirs(cur_motion_cat_sv, exist_ok=True)
#   cur_motion_cat_fr = os.path.join(fr_rt_folder, cur_motion_cat)
#   trimesh_voxilization(cur_motion_cat_fr, cur_motion_cat_sv)

def augment_vertices_scale(vertices):
    scale_normalizing_factors = np.random.uniform(low=0.75, high=1.25, size=(3,))
    # min + (max - min) * scale_normalizing_factors (normalizing_factors)
    scale_normalizing_factors = np.reshape(scale_normalizing_factors, [1, 3])
    minn_verts = np.min(vertices, axis=0, keepdims=True)
    vertices = minn_verts + (vertices - minn_verts) * scale_normalizing_factors # scale 
    # vertices = vertices * scale_normalizing_factors
    return vertices

def apply_random_scaling(vertices, with_scale=True):
    # dequan_verts = data_utils.dequantize_verts(vertices, n_bits=self.quantization_bits)
    # normalized_verts = dequan_verts + 0.5
    
    # ar
    if with_scale:
      normalized_verts = augment_vertices_scale(vertices)
    else:
      normalized_verts = vertices
    normalized_verts = data_utils.center_vertices(normalized_verts)
    normalized_verts = data_utils.normalize_vertices_scale(normalized_verts)
    # scaled_verts = data_utils.quantize_verts(normalized_verts, self.quantization_bits)

    return normalized_verts


def get_binvox_data(root="/share/xueyi/proj_data/Motion_Aligned_2", sv_root="", shape_type="oven", shape_idxes=None):
    os.makedirs(sv_root, exist_ok=True)
    root = os.path.join(root, shape_type)
    
    ''' get shape idxes '''
    shape_idxes = os.listdir(root) 
    shape_idxes = sorted(shape_idxes)
    # shape_idxes = [tmpp for tmpp in shape_idxes if tmpp[0] != "."]
    shape_idxes = [fn for fn in shape_idxes if os.path.isdir(os.path.join(root, fn))]
    ''' get shape idxes '''
    
    ''' get shape idxes '''
    shape_idxes = os.listdir(root) # /data/datasets/genn/Shape2Motion_Deform_key_256/eyeglasses/none_motion/dst/inst_0_manifold.obj
    shape_idxes = [fn for fn in shape_idxes if fn.endswith("_manifold.obj")]
    
    

    sv_root = os.path.join(sv_root, shape_type)
    os.makedirs(sv_root, exist_ok=True) # make motion category folder ### 
    
    ## sv_root, voxelized pts, 
    
    
    for shp_idx in shape_idxes:
      cur_shape_folder = os.path.join(root, shp_idx)
      cur_shape_sv_folder = os.path.join(sv_root, shp_idx)
      os.makedirs(cur_shape_sv_folder, exist_ok=True) # make shape folder

      cur_shape_part_folder = cur_shape_folder
      cur_shape_part_sv_folder = cur_shape_sv_folder
      
      cur_shape_part_folder = os.path.join(cur_shape_part_folder, "models")
      
      # cur_parts = os.listdir(cur_shape_folder)
      # cur_parts = [fn for fn in cur_parts if os.path.isdir(os.path.join(cur_shape_folder, fn))]
      # for cur_part in cur_parts: # current parts...
      # cur_shape_part_folder = os.path.join(cur_shape_folder, cur_part)
      # cur_shape_part_sv_folder = os.path.join(cur_shape_sv_folder, cur_part)
      # os.makedirs(cur_shape_part_sv_folder, exist_ok=True) # make part folder
      
      obj_files = os.listdir(cur_shape_part_folder)
      obj_files = [fn for fn in obj_files if fn.endswith(".obj")]
      
      cuda_voxelizer_path = "/home/zhangji/equiapp/code/cuda_voxelizer-0.4.8/build/cuda_voxelizer"
      
      for obj_fn in obj_files:

        cur_obj_fn = os.path.join(cur_shape_part_folder, obj_fn)

        if n_scales == 1: # no scale
          cur_obj_sv_fn = os.path.join(cur_shape_part_sv_folder, obj_fn)

          os.system(f"cp {cur_obj_fn} {cur_obj_sv_fn}")
          # os.system(f"/home/xueyi/gen/cuda_voxelizer/build/cuda_voxelizer -f {cur_obj_sv_fn} -s {vox_size}")
          os.system(f"{cuda_voxelizer_path} -f {cur_obj_sv_fn} -s {vox_size}")
        else:
          cur_verts, cur_faces = data_utils.read_obj_file_ours(cur_obj_fn)
          if cur_verts.shape[0] == 0 or len(cur_faces) == 0:
            continue
          for i_s in range(n_scales + 1):
            cur_scaled_sample_obj_fn = obj_fn.split(".")[0] + f"_s_{i_s}.obj"
            cur_scaled_sample_fn = os.path.join(cur_shape_part_sv_folder, cur_scaled_sample_obj_fn)
            cur_scaled_verts = apply_random_scaling(cur_verts, with_scale=True if i_s >= 1 else False)
            data_utils.save_obj_file(cur_scaled_verts, cur_faces, obj_fn=cur_scaled_sample_fn)
            # os.system(f"/home/xueyi/gen/cuda_voxelizer/build/cuda_voxelizer -f {cur_scaled_sample_fn} -s {vox_size}")
            os.system(f"{cuda_voxelizer_path} -f {cur_scaled_sample_fn} -s {vox_size}")

    # shape_idxes = 


### binvox root folders for motion datasets
#### root dir for binvox data
#### root 
#### sv_root; shape_type;
def get_binvox_data_for_deform(root="/share/xueyi/proj_data/Motion_Aligned_2", sv_root="", shape_type="oven", shape_idxes=None): ### for deform...
    os.makedirs(sv_root, exist_ok=True)
    root = os.path.join(root, shape_type) ### fr data root; ### shape_type --> none_motion
    ### dst and src folders ###
    deform_src_folder = os.path.join(root, "src")
    deform_dst_folder = os.path.join(root, "dst")
    src_obj_list = os.listdir(deform_src_folder)
    dst_obj_list = os.listdir(deform_dst_folder)
    src_obj_list = [fn for fn in src_obj_list if fn.endswith("_manifold.obj")]
    dst_obj_list = [fn for fn in dst_obj_list if fn.endswith("_manifold.obj")]
    
    # obj_nms = []
    # shape_idxes = []
    
    shape_idxes_obj_nms_pair = []
    for obj_nm in src_obj_list:
      cur_obj_fn = os.path.join(deform_src_folder, obj_nm)
      # shape_idxes.append(cur_obj_fn)
      # obj_nms.append(obj_nm)
      shape_idxes_obj_nms_pair.append((cur_obj_fn, obj_nm))
    for obj_nm in dst_obj_list:
      cur_obj_fn = os.path.join(deform_dst_folder, obj_nm)
      # shape_idxes.append(cur_obj_fn)
      # obj_nms.append(obj_nm)
      shape_idxes_obj_nms_pair.append((cur_obj_fn, obj_nm))
      
    # print(len(shape_idxes), len(obj_nms))
    
    # shape_idxes = sorted(shape_idxes)
    shape_idxes_obj_nms_pair = sorted(shape_idxes_obj_nms_pair, key=lambda ii: ii[0])
    # shape_idxes = [ii[0] for ii in shape_idxes_obj_nms_pair]
    
    # shape_idxes = [tmpp for tmpp in shape_idxes if tmpp[0] != "."]
    # shape_idxes = [fn for fn in shape_idxes if os.path.isdir(os.path.join(root, fn))]

    ### sv_root for the current type ###
    sv_root = os.path.join(sv_root, shape_type)
    os.makedirs(sv_root, exist_ok=True) # make motion category folder

    ### obj_idxes ###
    # for i_obj, (shp_idx, obj_nm) in enumerate(zip(shape_idxes, obj_nms)):
    for i_obj, (shp_idx, obj_nm) in enumerate(shape_idxes_obj_nms_pair):
      print(i_obj)
      cur_shape_part_sv_folder = sv_root
      
      obj_files = [shp_idx]
      
      print(obj_files)
      cuda_voxelizer_path = "/home/zhangji/equiapp/code/cuda_voxelizer-0.4.8/build/cuda_voxelizer"
      
      for obj_fn in obj_files:

        # cur_obj_fn = os.path.join(cur_shape_part_folder, obj_fn)
        cur_obj_fn = obj_fn
        
        obj_pure_nm = obj_nm

        if "/src" in cur_obj_fn:
          cur_obj_sv_fn = os.path.join(cur_shape_part_sv_folder, "src_" + obj_nm)
          obj_pure_nm = "src_" + obj_nm
        elif "/dst" in cur_obj_fn:
          cur_obj_sv_fn = os.path.join(cur_shape_part_sv_folder, "dst_" + obj_nm)
          obj_pure_nm = "dst_" + obj_nm
        else:
          raise ValueError(f"Error with cur_obj_fn: {cur_obj_fn}.")
        
        if os.path.exists(cur_obj_sv_fn):
          continue
        

        print(f"cur_obj_fn: {cur_obj_fn}, cur_obj_sv_fn: {cur_obj_sv_fn}")
        os.system(f"cp {cur_obj_fn} {cur_obj_sv_fn}")
        # os.system(f"/home/xueyi/gen/cuda_voxelizer/build/cuda_voxelizer -f {cur_obj_sv_fn} -s {vox_size}")
        os.system(f"{cuda_voxelizer_path} -f {cur_obj_sv_fn} -s {vox_size}")
        
        


#### sv_root; shape_type;
def get_binvox_data_for_deform_v2(root="/share/xueyi/proj_data/Motion_Aligned_2", sv_root="", src_dst_indicator="dst", shape_type="oven", shape_idxes=None): ### for deform...
    os.makedirs(sv_root, exist_ok=True)
    root = os.path.join(root, shape_type)  ### shape type for xxx ###
    ### dst and src folders ###
    
    # deform_src_folder = os.path.join(root, "src")
    # deform_dst_folder = os.path.join(root, "dst")
    # src_obj_list = os.listdir(deform_src_folder)
    # dst_obj_list = os.listdir(deform_dst_folder)
    # src_obj_list = [fn for fn in src_obj_list if fn.endswith("_manifold.obj")]
    # dst_obj_list = [fn for fn in dst_obj_list if fn.endswith("_manifold.obj")]
    
    deform_root_folder = os.path.join(root, src_dst_indicator)
    deform_obj_list = os.listdir(deform_root_folder)
    deform_obj_list = [fn for fn in deform_obj_list if fn.endswith("_manifold.obj")]
    
     ### sv_root for the current type ###
    sv_root = os.path.join(sv_root, shape_type)
    os.makedirs(sv_root, exist_ok=True)
    sv_root = os.path.join(sv_root, src_dst_indicator)
    os.makedirs(sv_root, exist_ok=True)
    
    # obj_nms = []
    # shape_idxes = []
    
    deform_obj_list = sorted(deform_obj_list) ### sorted_obj_list ###
    
    tot_obj_fns = []
    tot_obj_sv_fns = []
    
    
    for obj_nm in deform_obj_list:
      cur_obj_fn = os.path.join(deform_root_folder, obj_nm)
      tot_obj_fns.append(cur_obj_fn)
      
      cur_obj_sv_fn = os.path.join(sv_root, obj_nm)
      tot_obj_sv_fns.append(cur_obj_sv_fn)


    ### obj_idxes ###
    # for i_obj, (shp_idx, obj_nm) in enumerate(zip(shape_idxes, obj_nms)):
    for i_obj, obj_fn in enumerate(tot_obj_fns):
      print(i_obj)
      cur_shape_part_sv_folder = sv_root
      cur_obj_sv_fn = tot_obj_sv_fns[i_obj]
      
      obj_files = [obj_fn]
      obj_sv_fns = [cur_obj_sv_fn]
      
      print(obj_files)
      cuda_voxelizer_path = "/home/zhangji/equiapp/code/cuda_voxelizer-0.4.8/build/cuda_voxelizer"
      
      for obj_fn, obj_sv_fn in zip(obj_files, obj_sv_fns):

        # cur_obj_fn = os.path.join(cur_shape_part_folder, obj_fn)
        cur_obj_fn = obj_fn
        
        # obj_pure_nm = obj_nm

        # if "/src" in cur_obj_fn:
        #   cur_obj_sv_fn = os.path.join(cur_shape_part_sv_folder, "src_" + obj_nm)
        #   # obj_pure_nm = "src_" + obj_nm
        # elif "/dst" in cur_obj_fn:
        #   cur_obj_sv_fn = os.path.join(cur_shape_part_sv_folder, "dst_" + obj_nm)
        #   obj_pure_nm = "dst_" + obj_nm
        # else:
        #   raise ValueError(f"Error with cur_obj_fn: {cur_obj_fn}.")
        
        cur_obj_sv_fn = obj_sv_fn
        
        if os.path.exists(cur_obj_sv_fn):
          continue
        

        print(f"cur_obj_fn: {cur_obj_fn}, cur_obj_sv_fn: {cur_obj_sv_fn}")
        os.system(f"cp {cur_obj_fn} {cur_obj_sv_fn}")
        # os.system(f"/home/xueyi/gen/cuda_voxelizer/build/cuda_voxelizer -f {cur_obj_sv_fn} -s {vox_size}")
        os.system(f"{cuda_voxelizer_path} -f {cur_obj_sv_fn} -s {vox_size}")
        
     


def get_binvox_data_part_net(root="/share/xueyi/proj_data/Motion_Aligned_2", sv_root="", shape_type="oven", shape_idxes=None):
    root = os.path.join(root, shape_type)
    shape_idxes = os.listdir(root) 
    shape_idxes = sorted(shape_idxes)
    # shape_idxes = [tmpp for tmpp in shape_idxes if tmpp[0] != "."]
    shape_idxes = [fn for fn in shape_idxes if os.path.isdir(os.path.join(root, fn))]

    sv_root = os.path.join(sv_root, shape_type)
    os.makedirs(sv_root, exist_ok=True) # make motion category folder
    
    for shp_idx in shape_idxes:
      cur_shape_folder = os.path.join(root, shp_idx)
      cur_shape_sv_folder = os.path.join(sv_root, shp_idx)
      os.makedirs(cur_shape_sv_folder, exist_ok=True) # make shape folder

      cur_shape_folder = os.path.join(cur_shape_folder, "objs")

      cur_shape_part_folder = cur_shape_folder
      cur_shape_part_sv_folder = cur_shape_sv_folder
      
      # cur_parts = os.listdir(cur_shape_folder)
      # cur_parts = [fn for fn in cur_parts if os.path.isdir(os.path.join(cur_shape_folder, fn))]
      # for cur_part in cur_parts: # current parts...
      # cur_shape_part_folder = os.path.join(cur_shape_folder, cur_part)
      # cur_shape_part_sv_folder = os.path.join(cur_shape_sv_folder, cur_part)
      # os.makedirs(cur_shape_part_sv_folder, exist_ok=True) # make part folder
      
      obj_files = os.listdir(cur_shape_part_folder)
      obj_files = [fn for fn in obj_files if fn.endswith(".obj")]
      
      for obj_fn in obj_files:

        cur_obj_fn = os.path.join(cur_shape_part_folder, obj_fn)

        if n_scales == 1: # no scale
          cur_obj_sv_fn = os.path.join(cur_shape_part_sv_folder, obj_fn)

          os.system(f"cp {cur_obj_fn} {cur_obj_sv_fn}")
          os.system(f"/home/xueyi/gen/cuda_voxelizer/build/cuda_voxelizer -f {cur_obj_sv_fn} -s {vox_size}")
        else:
          cur_verts, cur_faces = data_utils.read_obj_file_ours(cur_obj_fn)
          if cur_verts.shape[0] == 0 or len(cur_faces) == 0:
            continue
          for i_s in range(n_scales + 1):
            cur_scaled_sample_obj_fn = obj_fn.split(".")[0] + f"_s_{i_s}.obj"
            cur_scaled_sample_fn = os.path.join(cur_shape_part_sv_folder, cur_scaled_sample_obj_fn)
            cur_scaled_verts = apply_random_scaling(cur_verts, with_scale=True if i_s >= 1 else False)
            data_utils.save_obj_file(cur_scaled_verts, cur_faces, obj_fn=cur_scaled_sample_fn)
            os.system(f"/home/xueyi/gen/cuda_voxelizer/build/cuda_voxelizer -f {cur_scaled_sample_fn} -s {vox_size}")




def get_binvox_data_obj(root="/share/xueyi/proj_data/Motion_Aligned_2", sv_root="", shape_type="oven", shape_idxes=None):
    root = os.path.join(root, shape_type)
    shape_idxes = os.listdir(root) 
    shape_idxes = sorted(shape_idxes)
    # shape_idxes = [tmpp for tmpp in shape_idxes if tmpp[0] != "."]
    shape_idxes = [fn for fn in shape_idxes if os.path.isdir(os.path.join(root, fn))]

    sv_root = os.path.join(sv_root, shape_type)
    os.makedirs(sv_root, exist_ok=True) # make motion category folder # sv_root; shape_type
    
    
    
    for shp_idx in shape_idxes:
      obj_fn_to_verts_faces = {}
      print(f"Processing shape instance {shp_idx}...")
      
      cur_shape_folder = os.path.join(root, shp_idx)
      cur_shape_sv_folder = os.path.join(sv_root, shp_idx)
      os.makedirs(cur_shape_sv_folder, exist_ok=True) # make shape folder; make shape folder

      obj_files = os.listdir(cur_shape_folder)
      obj_files = [fn for fn in obj_files if fn.endswith(".obj")]

      print(f"cur_shape_folder: {cur_shape_folder}, Shape idx: {shp_idx}")

      tot_verts, tot_faces = [], []
      obj_nm_to_vert_idxes = {}
      tot_n_verts = 0

      for obj_fn in obj_files:
        cur_obj_fn = os.path.join(cur_shape_folder, obj_fn)
        cur_verts, cur_faces = data_utils.read_obj_file_ours(cur_obj_fn) # obj_file_ours
        if cur_verts.shape[0] == 0 or len(cur_faces) == 0: # cur_faces
          continue
        cur_obj_verts_idxes = np.arange(tot_n_verts, tot_n_verts + cur_verts.shape[0], step=1)
        tot_n_verts += cur_verts.shape[0]
        # obj_nm_to_vert_idxes[o]
        # cur_obj_nm = cur_obj_fn.split(".")[0]
        cur_obj_nm = obj_fn.split(".")[0]
        # #### current obj verts idxes ####
        obj_nm_to_vert_idxes[cur_obj_nm] = cur_obj_verts_idxes
        obj_fn_to_verts_faces[cur_obj_nm] =(cur_verts, cur_faces)

        tot_verts.append(cur_verts)
        tot_faces.append(cur_faces)
        # obj_nm_to_vert_idxes[]
        ##### cur_verts, cur_faces ##### ### merge meshes
      cur_verts, cur_faces = data_utils.merge_meshes(tot_verts, tot_faces)
      if cur_verts.shape[0] == 0 or len(cur_faces) == 0:
        continue
      obj_nm_to_verts_idxes_sv_fn = "obj_nm_to_verts_idxes.npy"
      ###### vertices indexes
      obj_nm_to_verts_idxes_sv_fn = os.path.join(cur_shape_sv_folder, obj_nm_to_verts_idxes_sv_fn)
      ##### Save objnm_to_vert_idxes #####
      np.save(obj_nm_to_verts_idxes_sv_fn, obj_nm_to_vert_idxes)

      # print(f"obj_nm_to_vert_idxes: {obj_nm_to_vert_idxes.keys()}")
      if n_scales == 1:
        cur_obj_sv_fn = "summary.obj"
        cur_obj_sv_fn = os.path.join(cur_shape_sv_folder, cur_obj_sv_fn)
        data_utils.save_obj_file(cur_verts, cur_faces, obj_fn=cur_obj_sv_fn)
        os.system(f"/home/xueyi/gen/cuda_voxelizer/build/cuda_voxelizer -f {cur_obj_sv_fn} -s {vox_size}")

        for part_obj_fn in obj_fn_to_verts_faces:
          cur_part_sv_fn = f"{part_obj_fn}.obj"
          cur_part_sv_fn = os.path.join(cur_shape_sv_folder, cur_part_sv_fn)
          cur_part_verts, cur_part_faces = obj_fn_to_verts_faces[cur_part_sv_fn]
          data_utils.save_obj_file(cur_part_verts, cur_part_faces, obj_fn=cur_part_sv_fn)
          os.system(f"/home/xueyi/gen/cuda_voxelizer/build/cuda_voxelizer -f {cur_part_sv_fn} -s {vox_size}") # voxelize part meshes
      else:
        for i_s in range(n_scales + 1):
          cur_scaled_sample_obj_fn = f"summary_s_{i_s}.obj"
          cur_scaled_sample_obj_fn = os.path.join(cur_shape_sv_folder, cur_scaled_sample_obj_fn)
          cur_scaled_verts = apply_random_scaling(cur_verts, with_scale=True if i_s >= 1 else False)
          data_utils.save_obj_file(cur_scaled_verts, cur_faces, obj_fn=cur_scaled_sample_obj_fn)
          os.system(f"/home/xueyi/gen/cuda_voxelizer/build/cuda_voxelizer -f {cur_scaled_sample_obj_fn} -s {vox_size}")

          for part_obj_fn in obj_fn_to_verts_faces:
            cur_part_verts, cur_part_faces = obj_fn_to_verts_faces[part_obj_fn]
            cur_part_obj_verts_idxes =  obj_nm_to_vert_idxes[part_obj_fn]
            cur_scaled_part_obj_fn = f"{part_obj_fn}_s_{i_s}.obj" # scaled part obj sv file name
            cur_scaled_part_obj_fn = os.path.join(cur_shape_sv_folder, cur_scaled_part_obj_fn) # real file name to save
            print(f"Svaing saled part to {cur_scaled_part_obj_fn}")
            cur_scaled_part_verts = cur_scaled_verts[cur_part_obj_verts_idxes]  # scaled part verts
            data_utils.save_obj_file(cur_scaled_part_verts, cur_part_faces, obj_fn=cur_scaled_part_obj_fn) # current scaled part obj fn
            os.system(f"/home/xueyi/gen/cuda_voxelizer/build/cuda_voxelizer -f {cur_scaled_part_obj_fn} -s {vox_size}")
            


def get_binvox_data_obj_part_net(root="/share/xueyi/proj_data/Motion_Aligned_2", sv_root="", shape_type="oven", shape_idxes=None):
    root = os.path.join(root, shape_type)

    if shape_idxes is None:
      shape_idxes = os.listdir(root) 
      shape_idxes = sorted(shape_idxes)
      # shape_idxes = [tmpp for tmpp in shape_idxes if tmpp[0] != "."]
      shape_idxes = [fn for fn in shape_idxes if os.path.isdir(os.path.join(root, fn))]

    sv_root = os.path.join(sv_root, shape_type)
    os.makedirs(sv_root, exist_ok=True) # make motion category folder # sv_root; shape_type
    
    
    
    for shp_idx in shape_idxes:
      obj_fn_to_verts_faces = {}
      print(f"Processing shape instance {shp_idx}...")
      
      cur_shape_folder = os.path.join(root, shp_idx)
      cur_shape_sv_folder = os.path.join(sv_root, shp_idx)
      if os.path.exists(cur_shape_sv_folder):
        continue
      os.makedirs(cur_shape_sv_folder, exist_ok=True) # make shape folder; make shape folder
    
      cur_shape_folder = os.path.join(cur_shape_folder, "objs")

      obj_files = os.listdir(cur_shape_folder)
      obj_files = [fn for fn in obj_files if fn.endswith(".obj")]

      print(f"cur_shape_folder: {cur_shape_folder}, Shape idx: {shp_idx}")

      tot_verts, tot_faces = [], []
      obj_nm_to_vert_idxes = {}
      tot_n_verts = 0

      for obj_fn in obj_files:
        cur_obj_fn = os.path.join(cur_shape_folder, obj_fn)
        cur_verts, cur_faces = data_utils.read_obj_file_ours(cur_obj_fn) # obj_file_ours
        if cur_verts.shape[0] == 0 or len(cur_faces) == 0: # cur_faces
          continue
        cur_obj_verts_idxes = np.arange(tot_n_verts, tot_n_verts + cur_verts.shape[0], step=1)
        tot_n_verts += cur_verts.shape[0]
        # obj_nm_to_vert_idxes[o]
        # cur_obj_nm = cur_obj_fn.split(".")[0]
        cur_obj_nm = obj_fn.split(".")[0]
        # #### current obj verts idxes ####
        obj_nm_to_vert_idxes[cur_obj_nm] = cur_obj_verts_idxes
        obj_fn_to_verts_faces[cur_obj_nm] =(cur_verts, cur_faces)

        tot_verts.append(cur_verts)
        tot_faces.append(cur_faces)
        # obj_nm_to_vert_idxes[]
        ##### cur_verts, cur_faces ##### ### merge meshes
      cur_verts, cur_faces = data_utils.merge_meshes(tot_verts, tot_faces)
      if cur_verts.shape[0] == 0 or len(cur_faces) == 0:
        continue
      obj_nm_to_verts_idxes_sv_fn = "obj_nm_to_verts_idxes.npy"
      ###### vertices indexes
      obj_nm_to_verts_idxes_sv_fn = os.path.join(cur_shape_sv_folder, obj_nm_to_verts_idxes_sv_fn)
      ##### Save objnm_to_vert_idxes #####
      np.save(obj_nm_to_verts_idxes_sv_fn, obj_nm_to_vert_idxes)

      # print(f"obj_nm_to_vert_idxes: {obj_nm_to_vert_idxes.keys()}")
      if n_scales == 1:
        cur_obj_sv_fn = "summary.obj"
        cur_obj_sv_fn = os.path.join(cur_shape_sv_folder, cur_obj_sv_fn)
        data_utils.save_obj_file(cur_verts, cur_faces, obj_fn=cur_obj_sv_fn)
        os.system(f"/home/xueyi/gen/cuda_voxelizer/build/cuda_voxelizer -f {cur_obj_sv_fn} -s {vox_size}")

        # for part_obj_fn in obj_fn_to_verts_faces:
        #   cur_part_sv_fn = f"{part_obj_fn}.obj"
        #   cur_part_sv_fn = os.path.join(cur_shape_sv_folder, cur_part_sv_fn)
        #   cur_part_verts, cur_part_faces = obj_fn_to_verts_faces[cur_part_sv_fn]
        #   data_utils.save_obj_file(cur_part_verts, cur_part_faces, obj_fn=cur_part_sv_fn)
        #   os.system(f"/home/xueyi/gen/cuda_voxelizer/build/cuda_voxelizer -f {cur_part_sv_fn} -s {vox_size}") # voxelize part meshes
      else:
        for i_s in range(n_scales + 1):
          cur_scaled_sample_obj_fn = f"summary_s_{i_s}.obj"
          cur_scaled_sample_obj_fn = os.path.join(cur_shape_sv_folder, cur_scaled_sample_obj_fn)
          cur_scaled_verts = apply_random_scaling(cur_verts, with_scale=True if i_s >= 1 else False)
          data_utils.save_obj_file(cur_scaled_verts, cur_faces, obj_fn=cur_scaled_sample_obj_fn)
          os.system(f"/home/xueyi/gen/cuda_voxelizer/build/cuda_voxelizer -f {cur_scaled_sample_obj_fn} -s {vox_size}")

          # for part_obj_fn in obj_fn_to_verts_faces:
          #   cur_part_verts, cur_part_faces = obj_fn_to_verts_faces[part_obj_fn]
          #   cur_part_obj_verts_idxes =  obj_nm_to_vert_idxes[part_obj_fn]
          #   cur_scaled_part_obj_fn = f"{part_obj_fn}_s_{i_s}.obj" # scaled part obj sv file name
          #   cur_scaled_part_obj_fn = os.path.join(cur_shape_sv_folder, cur_scaled_part_obj_fn) # real file name to save
          #   print(f"Svaing saled part to {cur_scaled_part_obj_fn}")
          #   cur_scaled_part_verts = cur_scaled_verts[cur_part_obj_verts_idxes]  # scaled part verts
          #   data_utils.save_obj_file(cur_scaled_part_verts, cur_part_faces, obj_fn=cur_scaled_part_obj_fn) # current scaled part obj fn
          #   os.system(f"/home/xueyi/gen/cuda_voxelizer/build/cuda_voxelizer -f {cur_scaled_part_obj_fn} -s {vox_size}")
            

def get_binvox_data_shape_net(root="/share/xueyi/proj_data/Motion_Aligned_2", sv_root="", shape_type="oven", shape_idxes=None):
    root = os.path.join(root, shape_type)
    # if shape_idxes is None:
    shape_idxes = os.listdir(root) #### list data and get shape_idxes
    shape_idxes = sorted(shape_idxes)
    # shape_idxes = [tmpp for tmpp in shape_idxes if tmpp[0] != "."]
    shape_idxes = [fn for fn in shape_idxes if os.path.isdir(os.path.join(root, fn))]

    sv_root = os.path.join(sv_root, shape_type) ### 
    os.makedirs(sv_root, exist_ok=True) # make motion category folder

    models_folder = "models"
    
    for shp_idx in shape_idxes:
      cur_shape_folder = os.path.join(root, shp_idx)
      cur_shape_sv_folder = os.path.join(sv_root, shp_idx)
      os.makedirs(cur_shape_sv_folder, exist_ok=True) # make shape folder
      
      cur_shape_folder = os.path.join(cur_shape_folder, models_folder)
      cur_shape_sv_folder = os.path.join(cur_shape_sv_folder, models_folder)

      os.makedirs(cur_shape_sv_folder, exist_ok=True)

      # cur_shape_folder = os.path.join(cur_shape_folder, "objs")

      cur_shape_part_folder = cur_shape_folder
      cur_shape_part_sv_folder = cur_shape_sv_folder
      
      # cur_parts = os.listdir(cur_shape_folder)
      # cur_parts = [fn for fn in cur_parts if os.path.isdir(os.path.join(cur_shape_folder, fn))]
      # for cur_part in cur_parts: # current parts...
      # cur_shape_part_folder = os.path.join(cur_shape_folder, cur_part)
      # cur_shape_part_sv_folder = os.path.join(cur_shape_sv_folder, cur_part)
      # os.makedirs(cur_shape_part_sv_folder, exist_ok=True) # make part folder
      
      obj_files = os.listdir(cur_shape_part_folder)
      obj_files = [fn for fn in obj_files if fn.endswith(".obj")]
      
      for obj_fn in obj_files:

        cur_obj_fn = os.path.join(cur_shape_part_folder, obj_fn)

        if n_scales == 1: # no scale
          cur_obj_sv_fn = os.path.join(cur_shape_part_sv_folder, obj_fn)

          os.system(f"cp {cur_obj_fn} {cur_obj_sv_fn}")
          os.system(f"/home/xueyi/gen/cuda_voxelizer/build/cuda_voxelizer -f {cur_obj_sv_fn} -s {vox_size}")
        else:
          cur_verts, cur_faces = data_utils.read_obj_file_ours(cur_obj_fn)
          if cur_verts.shape[0] == 0 or len(cur_faces) == 0:
            continue
          for i_s in range(n_scales + 1):
            cur_scaled_sample_obj_fn = obj_fn.split(".")[0] + f"_s_{i_s}.obj"
            cur_scaled_sample_fn = os.path.join(cur_shape_part_sv_folder, cur_scaled_sample_obj_fn)
            cur_scaled_verts = apply_random_scaling(cur_verts, with_scale=True if i_s >= 1 else False)
            data_utils.save_obj_file(cur_scaled_verts, cur_faces, obj_fn=cur_scaled_sample_fn)
            os.system(f"/home/xueyi/gen/cuda_voxelizer/build/cuda_voxelizer -f {cur_scaled_sample_fn} -s {vox_size}")
  


# import 
def get_statistics_info(dataset_root_folder, motion_cats=None, sv_total_meta_info=True):
  
  if motion_cats is None:
    sv_total_meta_info = True
    motion_cats = os.listdir(dataset_root_folder)
    motion_cats = [fn for fn in motion_cats if os.path.isdir(os.path.join(dataset_root_folder, fn))]
  else:
    sv_total_meta_info = False
    motion_cats = motion_cats
  print()
  motion_cat_to_infos = {}
  for cat in motion_cats:
    print(f"process category: {cat}")
    cur_cat_folder = os.path.join(dataset_root_folder, cat)
    sv_meta_info_fn = os.path.join(cur_cat_folder, "meta_infos.npy")
    if os.path.exists(sv_meta_info_fn):
      continue
    
    cur_cat_insts = os.listdir(cur_cat_folder)
    cur_cat_insts = [fn for fn in cur_cat_insts if os.path.isdir(os.path.join(cur_cat_folder, fn))]
    inst_idx_to_fn_to_pts_nn = {}
    for cur_inst in cur_cat_insts: # inst_fn
      print(f"process instance: {cur_inst}")
      cur_cat_cur_inst_folder = os.path.join(cur_cat_folder, cur_inst)
      cur_cat_cur_insts = os.listdir(cur_cat_cur_inst_folder)
      cur_cat_cur_insts = [fn for fn in cur_cat_cur_insts if fn.endswith(".binvox")]
      fn_to_pts_nn = {}
      for cur_cat_cur_inst_fn in cur_cat_cur_insts:
        cur_inst_fn = os.path.join(cur_cat_cur_inst_folder, cur_cat_cur_inst_fn)
        # import binvox_rw
        with open(cur_inst_fn, 'rb') as f:
            model = binvox_rw.read_as_3d_array(f)
        pts = []
        dim_x, dim_y, dim_z = model.dims

        for i_x in range(dim_x):
            for i_y in range(dim_y):
                for i_z in range(dim_z):
                    cur_model_data = int(model.data[i_x, i_y, i_z].item())
                    if cur_model_data > 0:
                        # print(cur_model_data)
                        pts.append([i_x, i_y, i_z])
        cur_inst_pts_nn = len(pts)
        fn_to_pts_nn[cur_inst_fn] = cur_inst_pts_nn
        # cur_inst_split_fns = cur_cat_cur_inst_fn.split(".")[0].split("_") # split(".") --> [...] + [obj_64] + [binvox]
        # cur_inst_scale_idx = int(cur_inst_split_fns[-1])
        # cur_inst_part_fn = "_".join(cur_inst_scale_idx[:-2]) # [...] + [s] + [scale_idx]
        # cur_inst_part_fn
        # cur_inst_part_fn
      inst_idx_to_fn_to_pts_nn[cur_inst] = fn_to_pts_nn
    
    sv_meta_info_fn = os.path.join(cur_cat_folder, "meta_infos.npy")
    np.save(sv_meta_info_fn, inst_idx_to_fn_to_pts_nn)
    
    motion_cat_to_infos[cat] = inst_idx_to_fn_to_pts_nn
  print(motion_cat_to_infos['eyeglasses'])

  if sv_total_meta_info: # sv_total_meta_info --> 
    sv_meta_info_fn = os.path.join(dataset_root_folder, "meta_infos.npy")
    np.save(sv_meta_info_fn, motion_cat_to_infos)


# import 
def get_statistics_info_part_net(dataset_root_folder, motion_cats=None, sv_total_meta_info=True):
  
  if motion_cats is None:
    sv_total_meta_info = True
    motion_cats = os.listdir(dataset_root_folder)
    motion_cats = [fn for fn in motion_cats if os.path.isdir(os.path.join(dataset_root_folder, fn))]
  else:
    sv_total_meta_info = False
    motion_cats = motion_cats
  print()
  motion_cat_to_infos = {}
  for cat in motion_cats:
    print(f"process category: {cat}")
    cur_cat_folder = os.path.join(dataset_root_folder, cat)
    sv_meta_info_fn = os.path.join(cur_cat_folder, "meta_infos.npy")
    if os.path.exists(sv_meta_info_fn):
      continue
    
    #### cur_cat_insts --> cur_cat_folder
    cur_cat_insts = os.listdir(cur_cat_folder)
    cur_cat_insts = [fn for fn in cur_cat_insts if os.path.isdir(os.path.join(cur_cat_folder, fn))]
    inst_idx_to_fn_to_pts_nn = {}
    for cur_inst in cur_cat_insts: # inst_fn
      print(f"process instance: {cur_inst}")
      cur_cat_cur_inst_folder = os.path.join(cur_cat_folder, cur_inst)
      cur_cat_cur_insts = os.listdir(cur_cat_cur_inst_folder)
      cur_cat_cur_insts = [fn for fn in cur_cat_cur_insts if fn.endswith(".binvox")]
      fn_to_pts_nn = {}
      for cur_cat_cur_inst_fn in cur_cat_cur_insts:
        cur_inst_fn = os.path.join(cur_cat_cur_inst_folder, cur_cat_cur_inst_fn)
        # import binvox_rw
        with open(cur_inst_fn, 'rb') as f:
            model = binvox_rw.read_as_3d_array(f)
        pts = []
        dim_x, dim_y, dim_z = model.dims

        for i_x in range(dim_x):
            for i_y in range(dim_y):
                for i_z in range(dim_z):
                    cur_model_data = int(model.data[i_x, i_y, i_z].item())
                    if cur_model_data > 0:
                        # print(cur_model_data)
                        pts.append([i_x, i_y, i_z])
        cur_inst_pts_nn = len(pts)
        fn_to_pts_nn[cur_inst_fn] = cur_inst_pts_nn
        # cur_inst_split_fns = cur_cat_cur_inst_fn.split(".")[0].split("_") # split(".") --> [...] + [obj_64] + [binvox]
        # cur_inst_scale_idx = int(cur_inst_split_fns[-1])
        # cur_inst_part_fn = "_".join(cur_inst_scale_idx[:-2]) # [...] + [s] + [scale_idx]
        # cur_inst_part_fn
        # cur_inst_part_fn
      inst_idx_to_fn_to_pts_nn[cur_inst] = fn_to_pts_nn
    
    sv_meta_info_fn = os.path.join(cur_cat_folder, "meta_infos.npy")
    np.save(sv_meta_info_fn, inst_idx_to_fn_to_pts_nn)
    
    motion_cat_to_infos[cat] = inst_idx_to_fn_to_pts_nn
  print(motion_cat_to_infos['eyeglasses'])

  if sv_total_meta_info: # sv_total_meta_info --> 
    sv_meta_info_fn = os.path.join(dataset_root_folder, "meta_infos.npy")
    np.save(sv_meta_info_fn, motion_cat_to_infos)

import multiprocessing
from multiprocessing import Pool

# import 
def get_statistics_info_shape_net(dataset_root_folder, motion_cats=None, sv_total_meta_info=True):
  
  #### solid voxels currently #####
  if motion_cats is None:
    sv_total_meta_info = True
    motion_cats = os.listdir(dataset_root_folder)
    motion_cats = [fn for fn in motion_cats if os.path.isdir(os.path.join(dataset_root_folder, fn))]
  else:
    sv_total_meta_info = False
    motion_cats = motion_cats
  print()

  meta_info_fn = "meta_infos.npy"
  surfix_binvox = ".surface.binvox"
  surfix_binvox = ".solid.binvox"
  surfix_binvox = ".binvox"
  motion_cat_to_infos = {}
  for cat in motion_cats:
    print(f"process category: {cat}")
    cur_cat_folder = os.path.join(dataset_root_folder, cat)
    sv_meta_info_fn = os.path.join(cur_cat_folder, meta_info_fn)
    if os.path.exists(sv_meta_info_fn):
      continue
    
    #### cur_cat_insts --> cur_cat_folder
    cur_cat_insts = os.listdir(cur_cat_folder)
    cur_cat_insts = [fn for fn in cur_cat_insts if os.path.isdir(os.path.join(cur_cat_folder, fn))]
    inst_idx_to_fn_to_pts_nn = {}
    for cur_inst in cur_cat_insts: # inst_fn
      print(f"process instance: {cur_inst}")
      cur_cat_cur_inst_folder = os.path.join(cur_cat_folder, cur_inst, "models")
      cur_cat_cur_insts = os.listdir(cur_cat_cur_inst_folder)
      # cur_cat_cur_insts = [fn for fn in cur_cat_cur_insts if fn.endswith(".solid.binvox")]
      cur_cat_cur_insts = [fn for fn in cur_cat_cur_insts if fn.endswith(surfix_binvox)]
      fn_to_pts_nn = {}
      for cur_cat_cur_inst_fn in cur_cat_cur_insts:
        ##### cur_inst_fn #####
        cur_inst_fn = os.path.join(cur_cat_cur_inst_folder, cur_cat_cur_inst_fn)
        # import binvox_rw
        with open(cur_inst_fn, 'rb') as f:
            model = binvox_rw.read_as_3d_array(f)
        pts = []
        dim_x, dim_y, dim_z = model.dims

        for i_x in range(dim_x):
            for i_y in range(dim_y):
                for i_z in range(dim_z):
                    cur_model_data = int(model.data[i_x, i_y, i_z].item())
                    if cur_model_data > 0:
                        # print(cur_model_data)
                        pts.append([i_x, i_y, i_z])
        cur_inst_pts_nn = len(pts)
        fn_to_pts_nn[cur_inst_fn] = cur_inst_pts_nn
        # cur_inst_split_fns = cur_cat_cur_inst_fn.split(".")[0].split("_") # split(".") --> [...] + [obj_64] + [binvox]
        # cur_inst_scale_idx = int(cur_inst_split_fns[-1])
        # cur_inst_part_fn = "_".join(cur_inst_scale_idx[:-2]) # [...] + [s] + [scale_idx]
        # cur_inst_part_fn
        # cur_inst_part_fn
      inst_idx_to_fn_to_pts_nn[cur_inst] = fn_to_pts_nn
    
    sv_meta_info_fn = os.path.join(cur_cat_folder, meta_info_fn)
    np.save(sv_meta_info_fn, inst_idx_to_fn_to_pts_nn)
    
    motion_cat_to_infos[cat] = inst_idx_to_fn_to_pts_nn
  print(motion_cat_to_infos['eyeglasses'])

  if sv_total_meta_info: # sv_total_meta_info --> 
    sv_meta_info_fn = os.path.join(dataset_root_folder, meta_info_fn)
    np.save(sv_meta_info_fn, motion_cat_to_infos)



def get_statistics_info_shape_net_all(dataset_root_folder, motion_cats=None, sv_total_meta_info=True, nprocs=20):
  if motion_cats is None:
    # sv_total_meta_info = True
    motion_cats = os.listdir(dataset_root_folder)
    motion_cats = [fn for fn in motion_cats if os.path.isdir(os.path.join(dataset_root_folder, fn))]
  else:
    # sv_total_meta_info = False
    motion_cats = motion_cats
  # nprocs = 
  with Pool(processes=len(motion_cats)) as pool:  
    for cur_cat in motion_cats:
      pool.apply_async(get_statistics_info_shape_net, (dataset_root_folder, [cur_cat], False))
    pool.close()
    pool.join()
    print("None!")



def get_binvox_data_obj_part_net_all(root, sv_root=None, nprocs=20):
  # if motion_cats is None:
  #   # sv_total_meta_info = True
  #   motion_cats = os.listdir(root)
  #   motion_cats = [fn for fn in motion_cats if os.path.isdir(os.path.join(dataset_root_folder, fn))]
  # else:
  #   # sv_total_meta_info = False
  #   motion_cats = motion_cats

  # root="/share/xueyi/proj_data/Motion_Aligned_2", sv_root="", shape_type="oven", shape_idxes=None

  motion_cats = os.listdir(root)
  motion_cats = [fn for fn in motion_cats if os.path.isdir(os.path.join(root, fn))]
  first_motion_cat = motion_cats[0]
  cur_motion_cat_folder = os.path.join(root, first_motion_cat)
  shape_idxes = os.listdir(cur_motion_cat_folder)
  shape_idxes = [fn for fn in shape_idxes if os.path.isdir(os.path.join(cur_motion_cat_folder, fn))]
  # nprocs = 
  with Pool(processes=nprocs) as pool:  
    nn_cats_per_proc = len(shape_idxes) // nprocs
    if nn_cats_per_proc * nprocs < len(shape_idxes):
      nn_cats_per_proc += 1

    for i_proc in range(nprocs):
      cur_proc_motion_cats = shape_idxes[i_proc * nn_cats_per_proc: (i_proc + 1) * nn_cats_per_proc]
      print(f"Starting {i_proc}-th process with number of shapes to process: {len(cur_proc_motion_cats)}.")
      pool.apply_async(get_binvox_data_obj_part_net, (root, sv_root, first_motion_cat, cur_proc_motion_cats))

    pool.close()
    pool.join()
    print("None!")
  

def get_binvox_data_obj_shape_net_all(root, sv_root=None, nprocs=20):
  # if motion_cats is None:
  #   # sv_total_meta_info = True
  #   motion_cats = os.listdir(root)
  #   motion_cats = [fn for fn in motion_cats if os.path.isdir(os.path.join(dataset_root_folder, fn))]
  # else:
  #   # sv_total_meta_info = False
  #   motion_cats = motion_cats

  # root="/share/xueyi/proj_data/Motion_Aligned_2", sv_root="", shape_type="oven", shape_idxes=None

  motion_cats = os.listdir(root)
  motion_cats = [fn for fn in motion_cats if os.path.isdir(os.path.join(root, fn))]
  # first_motion_cat = motion_cats[0]
  # cur_motion_cat_folder = os.path.join(root, first_motion_cat)
  # shape_idxes = os.listdir(cur_motion_cat_folder)
  # shape_idxes = [fn for fn in shape_idxes if os.path.isdir(os.path.join(cur_motion_cat_folder, fn))]
  # nprocs = 
  nprocs = len(motion_cats)
  with Pool(processes=nprocs) as pool:  
    # nn_cats_per_proc = len(shape_idxes) // nprocs
    # if nn_cats_per_proc * nprocs < len(shape_idxes):
    #   nn_cats_per_proc += 1

    for i_proc in range(nprocs):
      cur_proc_motion_cat = motion_cats[i_proc]
      # cur_proc_motion_cats = shape_idxes[i_proc * nn_cats_per_proc: (i_proc + 1) * nn_cats_per_proc]
      print(f"Starting {i_proc}-th process with category: {cur_proc_motion_cat}.")
      # cur_category_root = os.path.join(root, cur_proc_motion_cat)
      pool.apply_async(get_binvox_data_shape_net, (root, sv_root, cur_proc_motion_cat))

    pool.close()
    pool.join()
    print("None!")



def get_binvox_data_from_deform_v2(root="/share/xueyi/proj_data/Motion_Aligned_2", sv_root="", shape_type="oven", shape_idxes=None):
    os.makedirs(sv_root, exist_ok=True)
    root = os.path.join(root, shape_type) # roto 
    
    # root = "/data/datasets/genn/SAPIEN_Deform/StorageFurniture_Style3/link_0/dst"
    
    
    ''' get shape idxes '''
    # shape_idxes = os.listdir(root) 
    # shape_idxes = sorted(shape_idxes)
    # # shape_idxes = [tmpp for tmpp in shape_idxes if tmpp[0] != "."]
    # shape_idxes = [fn for fn in shape_idxes if os.path.isdir(os.path.join(root, fn))]
    ''' get shape idxes '''
    
    ''' get shape idxes '''
    shape_idxes = os.listdir(root) # /data/datasets/genn/Shape2Motion_Deform_key_256/eyeglasses/none_motion/dst/inst_0_manifold.obj
    shape_idxes = [fn for fn in shape_idxes if fn.endswith("_manifold.obj")]
    # 
    # sv_root = "/data/datasets/genn/SAPIEN_Deform_voxelized_64/StorageFurniture_Style3/link_0" ## jave root
    sv_root = os.path.join(sv_root, shape_type)
    os.makedirs(sv_root, exist_ok=True) # make motion category folder ### 
    
    ## sv_root, voxelized pts ##
    for shp_idx in shape_idxes: ## shape_idxes ##
      cur_obj_fn = os.path.join(root, shp_idx) ### fr obj fn
      cur_obj_sv_fn = os.path.join(sv_root, shp_idx) ### sv obj fn
       ### single one...
      cuda_voxelizer_path = "/home/zhangji/equiapp/code/cuda_voxelizer-0.4.8/build/cuda_voxelizer"
      
      os.system(f"cp {cur_obj_fn} {cur_obj_sv_fn}")
      
      os.system(f"{cuda_voxelizer_path} -f {cur_obj_sv_fn} -s {vox_size}")
      


if __name__ == '__main__': # back propagation

  n_scales = opt.dataset.n_scales
  vox_size = opt.dataset.vox_size
  exclude_existing = opt.dataset.exclude_existing
  nprocs = opt.dataset.nprocs
  # save_folder = "/share/xueyi/proj_data/Motion_Samples"
  # shape_type = "eyeglasses"
  # root = "/nas/datasets/gen/datasets/MotionDataset_processed"

  # root = "/nas/datasets/gen/datasets/PartNet"
  # # root = "/nas/datasets/gen/datasets/PartNet_voxelized_64"
  # root = "/nas/datasets/gen/datasets/ShapeNetCore.v2"

  # sv_root = "/nas/datasets/gen/datasets/MotionDataset_processed_voxelized_64"
  # sv_root = "/nas/datasets/gen/datasets/MotionDataset_processed_voxelized_64"

  # # sv_root = f"/nas/datasets/gen/datasets/MotionDataset_processed_voxelized_{vox_size}_v9_obj_part_ns_{n_scales}" if n_scales > 1 else f"/nas/datasets/gen/datasets/MotionDataset_processed_voxelized_{vox_size}"

  # # sv_root = f"/nas/datasets/gen/datasets/PartNet_voxelized_{vox_size}_ns_{n_scales}" if n_scales > 1 else f"/nas/datasets/gen/datasets/PartNet_voxelized_{vox_size}"
  # # sv_root = f"/nas/datasets/gen/datasets/ShapeNetCore.v2_voxelized_{vox_size}_ns_{n_scales}" if n_scales > 1 else f"/nas/datasets/gen/datasets/ShapeNetCore.v2_voxelized_{vox_size}"

  # # os.makedirs(sv_root, exist_ok=True)

  # get_binvox_data_obj_shape_net_all(root, sv_root)
  
  
  
  
  # data_root = "/share/xueyi/datasets/gen/datasets/SAPIEN_Merged"
  # sv_root = "/share/xueyi/datasets/gen/datasets/SAPIEN_Merged_voxelized_64"
  
  # shape_type = "drawer"
  
  
  # # data_root = "/localdata_ssd/tmpp/datasets/SAPIEN"
  # data_root = "/data/datasets/genn/SAPIEN_processed"
  
  # sv_root = "/data/datasets/genn/SAPIEN_voxelized_64"
  # os.makedirs(sv_root, exist_ok=True)
  # # shape_type = "StorageFurniture_Style2"
  # shape_type = "StorageFurniture_Style3"
  
  # get_binvox_data(root=data_root, sv_root=sv_root, shape_type=shape_type, shape_idxes=None)


  #### binvox data for deform ####
  # # data_root = "/share/xueyi/datasets/gen/datasets/Shape2Motion_Deform/eyeglasses"
  # data_root = "/data/datasets/genn/Shape2Motion_Deform_key_256/eyeglasses"
  # sv_root = "/data/datasets/genn/Shape2Motion_Deform_key_256_voxelized_64"
  # os.makedirs(sv_root, exist_ok=True) ### sv root
  # shape_type = "eyeglasses"
  # sv_root = os.path.join(sv_root, shape_type)
  # os.makedirs(sv_root, exist_ok=True)
  
  # get_binvox_data_for_deform(root=data_root, sv_root=sv_root, shape_type="none_motion",  shape_idxes=None)
  #### binvox data for deform ####
  
  
  # data_root = "/share/xueyi/datasets/gen/datasets/Shape2Motion_Deform/eyeglasses"
  # data_root = "/data/datasets/genn/SAPIEN_Deform/StorageFurniture_Style3"
  # sv_root = "/data/datasets/genn/SAPIEN_Deform_voxelized_64"
  
  ''' get binvox data for deform v1 ''' 
  # data_root = "/data/datasets/genn/ShapeNetCoreV2_Deform"
  # sv_root = "/data/datasets/genn/ShapeNetCoreV2_Deform_voxelized_64"
  # os.makedirs(sv_root, exist_ok=True)
  # # shape_type = "eyeglasses"
  # # shape_type = "StorageFurniture_Style3" ### storage furniture 3
  
  # # shape_type = "02691156"
  # # sv_root = os.path.join(sv_root, shape_type)
  # os.makedirs(sv_root, exist_ok=True)
  
  # # shape_type = "link_2"
  # # shape_type = "02691156"
  # shape_type = "04379243"
  
  # get_binvox_data_for_deform(root=data_root, sv_root=sv_root, shape_type=shape_type,  shape_idxes=None)
  ''' get binvox data for deform v1 ''' 
  
  ''' get binvox data for deform v2 ''' 
  # data_root = "/data/datasets/genn/ShapeNetCoreV2_Deform"
  # # sv_root = "/data/datasets/genn/ShapeNetCoreV2_Deform_voxelized_64"
  # sv_root = "/data/datasets/genn/ShapeNetCoreV2_Deform_voxelized_64_v2"
  # os.makedirs(sv_root, exist_ok=True)
  
  # shape_type = "eyeglasses"
  # shape_type = "StorageFurniture_Style3" ### storage furniture 3
  
  # data_root = "/data/datasets/genn/SAPIEN_Deform"
  # cat_nm = "StorageFurniture_Style3"
  # data_root = os.path.join(data_root, cat_nm)
  
  # sv_root = "/data/datasets/genn/SAPIEN_Deform_voxelized_64_v2"
  # os.makedirs(sv_root, exist_ok=True)
  # sv_root = os.path.join(sv_root, cat_nm)
  # os.makedirs(sv_root, exist_ok=True)
  
  
  # /share/xueyi/datasets/gen/datasets/ShapeNetCore.v2/03001627/1ab8a3b55c14a7b27eaeab1f0c9120b7/models/model_normalized.obj
  data_root = "/share/xueyi/datasets/gen/datasets/ShapeNetCore.v2"
  sv_root = "/share/xueyi/datasets/gen/datasets/ShapeNetCore.v2_voxelized_64"
  # shape_type = "link_2"
  # shape_type = "02691156"
  # shape_type = "04379243"
  shape_type = "03001627"
  # shape_type = "link_1"


  # os.makedirs(sv_root, exist_ok=True)
  # # tot_src_dst_indicators = ["src", "dst"]
  # tot_src_dst_indicators = ["dst"]
  # for src_dst_indicator in tot_src_dst_indicators:
  #   print(f"data_root: {data_root}, sv_root: {sv_root}, src_dst_indicator: {src_dst_indicator}, shape_type: {shape_type}")
  #   # get_binvox_data_for_deform(root=data_root, sv_root=sv_root, shape_type=shape_type,  shape_idxes=None)
  #   # get_binvox_data_for_deform_v2(root=data_root, sv_root=sv_root, src_dst_indicator=src_dst_indicator, shape_type=shape_type, shape_idxes=None)
  #   get_binvox_data(root=data_root, sv_root=sv_root, shape_type=shape_type, shape_idxes=None)
  ''' get binvox data for deform v2 ''' 
  
  
  
# def get_binvox_data(root="/share/xueyi/proj_data/Motion_Aligned_2", sv_root="", shape_type="oven", shape_idxes=None):
#     os.makedirs(sv_root, exist_ok=True)
#     root = os.path.join(root, shape_type) ### data root; data root for saving;
#     shape_idxes = os.listdir(root) 
#     shape_idxes = sorted(shape_idxes)
#     # shape_idxes = [tmpp for tmpp in shape_idxes if tmpp[0] != "."]
#     shape_idxes = [fn for fn in shape_idxes if os.path.isdir(os.path.join(root, fn))]

#     sv_root = os.path.join(sv_root, shape_type)
#     os.makedirs(sv_root, exist_ok=True) # make motion category folder ### 
    
#     ## sv_root, voxelized pts, 
    
    
#     for shp_idx in shape_idxes:
#       cur_shape_folder = os.path.join(root, shp_idx)
#       cur_shape_sv_folder = os.path.join(sv_root, shp_idx)
#       os.makedirs(cur_shape_sv_folder, exist_ok=True) # make shape folder

#       cur_shape_part_folder = cur_shape_folder
#       cur_shape_part_sv_folder = cur_shape_sv_folder
      
#       cur_shape_part_folder = os.path.join(cur_shape_part_folder, "models")
      
  
  ## /data/datasets/genn/SAPIEN_Deform/Eyeglasses/link_2/dst
  # data_root = "/share/xueyi/datasets/gen/datasets/Shape2Motion_Deform/eyeglasses/dof_rootd_Aa001_r"
  # sv_root = "/share/xueyi/datasets/gen/datasets/Shape2Motion_Deform_voxelized_64"
  
  # data_root = "/share/xueyi/datasets/gen/datasets/Shape2Motion_Deform/eyeglasses/dof_rootd_Aa001_r"
  # sv_root = "/share/xueyi/datasets/gen/datasets/Shape2Motion_Deform_voxelized_64"
  
  data_root = "/data/datasets/genn/SAPIEN_Deform/Eyeglasses/link_2"
  sv_root = "/data/datasets/genn/SAPIEN_Deform_voxelized_64_v2/"
  
  os.makedirs(sv_root, exist_ok=True)
  motion_cat = "Eyeglasses"
  sv_root = os.path.join(sv_root, motion_cat)
  os.makedirs(sv_root, exist_ok=True)
  motion_cat = "link_2"
  sv_root = os.path.join(sv_root, motion_cat)
  os.makedirs(sv_root, exist_ok=True)
  # sv_root = "/share/xueyi/datasets/gen/datasets/Shape2Motion_Deform_voxelized_64/eyeglasses/dof_rootd_Aa001_r"
  
  print(f"Start processing data...")
  get_binvox_data_from_deform_v2(root=data_root, sv_root=sv_root, shape_type="dst", shape_idxes=None)


  # # motion_cats = os.listdir(root) # root
  # motion_cats = ["data_v0"]
  # motion_cats = [fn for fn in motion_cats if os.path.isdir(os.path.join(root, fn))]
  # motion_sv_cats = os.listdir(sv_root)

  # # motion_cats = ["screwdriver", "eyeglasses",  "pen", "scissors", "stapler"]
  # motion_sv_cats = [fn for fn in motion_sv_cats if os.path.isdir(os.path.join(sv_root, fn))] # is dir --->  
  # if exclude_existing:
  #   motion_cats = [fn for fn in motion_cats if fn not in motion_sv_cats]


  # shape_idxes = None
  # for motion_cat in motion_cats:
  #   print(f"processing motion cat: {motion_cat}")
  #   # cur_motion_
  #   # get_binvox_data(root, sv_root=sv_root, shape_type=motion_cat, shape_idxes=shape_idxes) # binvox data for...
  #   # get_binvox_data_obj(root, sv_root=sv_root, shape_type=motion_cat, shape_idxes=shape_idxes)
  #   # get_binvox_data_obj_part_net(root, sv_root=sv_root, shape_type=motion_cat, shape_idxes=shape_idxes)
  #   get_binvox_data_obj_part_net_all(root, sv_root=sv_root, nprocs=nprocs)


  # # # dataset_root_folder = "/nas/datasets/gen/datasets/MotionDataset_processed_voxelized_32_v9_ns_50" 
  # # # dataset_root_folder = "/nas/datasets/gen/datasets/MotionDataset_processed_voxelized_64"
  # # # dataset_root_folder = "/nas/datasets/gen/datasets/MotionDataset_processed_voxelized_64_ns_50"
  # # dataset_root_folder = "/nas/datasets/gen/datasets/PartNet_voxelized_64"
  # dataset_root_folder = "/nas/datasets/gen/datasets/ShapeNetCore.v2_voxelized_64"
  # # # motion_cats = ["closestool"]
  # # # motion_cats = ['bucket', 'cabinet', 'cannon', 'carton', 'eyeglasses', 'water_bottle', 'wine_bottle', 'windmill']
  # # # motion_cats = ['data_v0']
  # # # get_statistics_info(dataset_root_folder, motion_cats=motion_cats, sv_total_meta_info=False)
  # # # get_statistics_info_part_net(dataset_root_folder, motion_cats=motion_cats, sv_total_meta_info=False)

  # # # get_statistics_info_shape_net(dataset_root_folder, motion_cats=None, sv_total_meta_info=True)
  # get_statistics_info_shape_net_all(dataset_root_folder, motion_cats=None, sv_total_meta_info=False)

  

# CUDA_VISIBLE_DEVICES=2  python notebooks/process_data_sr.py --n-scales=50 --vox-size=64