import enum
from platform import java_ver
from options.options import opt
import os
import utils.data_utils_torch as data_utils
import numpy as np
import utils.binvox_rw as binvox_rw



def balance_class_idxes(category_name_to_mesh_dict_idxes):
  nn_max_mesh_dicts = 0
  nn_max_cat_nm = ""
  for cat_nm in category_name_to_mesh_dict_idxes:
    cur_cat_nn_mesh_dicts = len(category_name_to_mesh_dict_idxes[cat_nm])
    if cur_cat_nn_mesh_dicts > nn_max_mesh_dicts:
      nn_max_mesh_dicts = cur_cat_nn_mesh_dicts
      nn_max_cat_nm = cat_nm
  print(f"Balancing classes with dominate class {nn_max_cat_nm} ({nn_max_mesh_dicts} samples).")
  balanced_category_name_to_mesh_dict_idxes = {} 
  tot_mesh_dicts = []
  for cat_nm in category_name_to_mesh_dict_idxes:
    cur_cat_mesh_dicts = category_name_to_mesh_dict_idxes[cat_nm]
    cur_cat_nn_mesh_dicts = len(cur_cat_mesh_dicts)
    if cur_cat_nn_mesh_dicts < nn_max_mesh_dicts:
      cur_nn_added = nn_max_mesh_dicts - cur_cat_nn_mesh_dicts
      print(f"cur_cat_nn_mesh_dicts: {cur_cat_nn_mesh_dicts}, cur_nn_added: {cur_nn_added}")
      cur_added_instance_idxes = np.random.choice(cur_cat_nn_mesh_dicts, size=cur_nn_added, replace=True).tolist() ### sample with replacenment
      cur_added_mesh_instances = [cur_cat_mesh_dicts[ii] for ii in cur_added_instance_idxes]
      cur_cat_mesh_dicts = cur_cat_mesh_dicts + cur_added_mesh_instances
    tot_mesh_dicts = tot_mesh_dicts + cur_cat_mesh_dicts
    balanced_category_name_to_mesh_dict_idxes[cat_nm] = cur_cat_mesh_dicts
  return balanced_category_name_to_mesh_dict_idxes, tot_mesh_dicts
    

# get masked mesh dicts list
def get_mesh_dict_list(dataset_root_folder, samples_list, valid_indices=None):
  # max permit vertices
  nn_max_permite_vertices = opt.dataset.max_permit_vertices
  nn_max_permite_faces = opt.dataset.max_permit_faces

  # n_mask_samples = opt.model.n_mask_samples

  part_tree = {'idx': 0}

  summary_obj_fn = "summary.obj"
  quantization_bits = opt.model.quantization_bits
  recenter_mesh = opt.model.recenter_mesh

  mesh_dicts = []

  # permite_fns_nm = os.path.join(dataset_root_folder, f"nn_max_verts_{nn_max_permite_vertices}_max_faces_{nn_max_permite_faces}.npy")
  # if os.path.exists(permite_fns_nm):
  #   print(f"Loading data fns from pre-saved file: {permite_fns_nm}")
  #   samples_list = np.load(permite_fns_nm, allow_pickle=True).tolist()

  for i_s, sample_nm in enumerate(samples_list):

    if valid_indices is not None:
      valid = False
      for valid_indi in valid_indices:
        if valid_indi in sample_nm:
          valid = True
          break
      if not valid:
        continue
    
    cur_sample_folder = os.path.join(dataset_root_folder, sample_nm)
    cur_sample_summary_obj = os.path.join(cur_sample_folder, summary_obj_fn)
    cur_sample_vertices, cur_sample_faces = data_utils.read_obj(cur_sample_summary_obj)

    nn_face_indices = sum([len(sub_face) for sub_face in cur_sample_faces])

    # if not (cur_sample_vertices.shape[0] <= nn_max_permite_vertices and nn_face_indices <= nn_max_permite_faces):
    #   continue
    
    ### centralize vertices ###
    ins_vert_min = cur_sample_vertices.min(axis=0) # vert_min
    ins_vert_max = cur_sample_vertices.max(axis=0) # vert_max
    ins_vert_center = 0.5 * (ins_vert_min + ins_vert_max)
    cur_sample_vertices = cur_sample_vertices - ins_vert_center
    ins_vert_min = cur_sample_vertices.min(axis=0) # vert_min
    ins_vert_max = cur_sample_vertices.max(axis=0) # vert_max
    ins_extents = ins_vert_max - ins_vert_min # extents
    ins_scale = np.sqrt(np.sum(ins_extents**2)) # scale # 
    cur_sample_vertices = cur_sample_vertices / ins_scale


    cur_sample_mesh_dict = data_utils.process_mesh(cur_sample_vertices, cur_sample_faces, quantization_bits=quantization_bits, recenter_mesh=True)

    processed_vertices, processed_faces = cur_sample_mesh_dict['vertices'], cur_sample_mesh_dict['faces']
    nn_processed_faces_indices = processed_faces.shape[0]

    if not (processed_vertices.shape[0] <= nn_max_permite_vertices and nn_processed_faces_indices <= nn_max_permite_faces):
      continue

    mesh_dict = {}
    mesh_dict = cur_sample_mesh_dict
    mesh_dict['class_label'] = 0

    mesh_dicts.append(mesh_dict)

  return mesh_dicts, part_tree


# mesh dict multi part
def get_mesh_dict_list_multi_part(dataset_root_folder, part_names, valid_indices=None):
  # max permit vertices
  nn_max_permite_vertices = opt.dataset.max_permit_vertices
  nn_max_permite_faces = opt.dataset.max_permit_faces

  # n_mask_samples = opt.model.n_mask_samples
  part_tree = {'idx': 0}

  # summary_obj_fn = "summary.obj"
  quantization_bits = opt.model.quantization_bits
  recenter_mesh = opt.model.recenter_mesh
  # tree_traverse, child_to_parent = data_utils.mid_traverse_tree(part_tree)

  mesh_dicts = []

  samples_fn = os.path.join(dataset_root_folder, part_names[0])
  samples_list = os.listdir(samples_fn)

  samples_list = [fn for fn in samples_list if fn.endswith(".obj")]

  n_max_samples = 200
  n_valid_samples = 0

  for i_s, sample_nm in enumerate(samples_list):
    scale_idx = sample_nm.split('.')[0].split('_')[-1]
    scale_idx = int(scale_idx)
    if scale_idx > 100:
      continue
    shp_idx = sample_nm.split("_")[0]
    shp_idx = int(shp_idx)
    # if shp_idx not in [1, 2, 5, 7, 9, 10, 12, 11]:
    #   continue
    if valid_indices is not None:
      valid = False
      for valid_indi in valid_indices:
        if valid_indi in sample_nm:
          valid = True
          break
      if not valid:
        continue
    
    vertices_list, faces_list = [], []
    for part_nm in part_names:
      cur_sample_cur_part_fn = os.path.join(dataset_root_folder, part_nm, sample_nm)
      # print(cur_sample_cur_part_fn)
      cur_sample_vertices, cur_sample_faces = data_utils.read_obj(cur_sample_cur_part_fn)
      # print("cc")
      # print(f"cur_sample_vertices: {cur_sample_vertices.shape}, cur_sample_faces: {len(cur_sample_faces)}")
      vertices_list.append(cur_sample_vertices)
      faces_list.append(cur_sample_faces)
    cur_sample_vertices, cur_sample_faces = data_utils.merge_meshes(vertices_list, faces_list)
    nn_face_indices = sum([len(sub_face) for sub_face in cur_sample_faces])

    # print(f"n_vertices: {cur_sample_vertices.shape[0]}, nn_face_indices: {nn_face_indices}")
    
    # if not (cur_sample_vertices.shape[0] <= nn_max_permite_vertices and nn_face_indices <= nn_max_permite_faces):
    #   continue
    
    ### centralize vertice  s ###
    ins_vert_min = cur_sample_vertices.min(axis=0) # vert_min
    ins_vert_max = cur_sample_vertices.max(axis=0) # vert_max
    ins_vert_center = 0.5 * (ins_vert_min + ins_vert_max)
    cur_sample_vertices = cur_sample_vertices - ins_vert_center
    ins_vert_min = cur_sample_vertices.min(axis=0) # vert_min
    ins_vert_max = cur_sample_vertices.max(axis=0) # vert_max
    ins_extents = ins_vert_max - ins_vert_min # extents
    ins_scale = np.sqrt(np.sum(ins_extents**2)) # scale # 
    cur_sample_vertices = cur_sample_vertices / ins_scale

    cur_sample_mesh_dict = data_utils.process_mesh(cur_sample_vertices, cur_sample_faces, quantization_bits=quantization_bits, recenter_mesh=True)
    mesh_dict = cur_sample_mesh_dict
    mesh_dict['class_label'] = 0

    mesh_dict = {}
    mesh_dict = cur_sample_mesh_dict
    mesh_dict['class_label'] = 0
    vertices, faces = mesh_dict['vertices'], mesh_dict['faces']

    nn_vertices = vertices.shape[0]
    nn_faces = faces.shape[0]
    if not (nn_vertices <= nn_max_permite_vertices and nn_faces <= nn_max_permite_faces):
      continue

    mesh_dicts.append(mesh_dict)

    # if len(mesh_dicts) >= n_max_samples:
    #   break

  return mesh_dicts, part_tree


# mesh dict multi part
def get_mesh_dict_list_multi_part_part_first_coarse_to_fine(dataset_root_folder_coarse, dataset_root_folder_fine, part_names, valid_indices=None, category_name="eyeglasses", ret_part_name_to_mesh_idx=False, pc_cond=False):
  # max permit vertices
  nn_max_permite_vertices = opt.dataset.max_permit_vertices
  nn_max_permite_faces = opt.dataset.max_permit_faces

  # n_mask_samples = opt.model.n_mask_samples
  part_tree = {'idx': 0}

  # summary_obj_fn = "summary.obj"
  quantization_bits = opt.model.quantization_bits
  recenter_mesh = opt.model.recenter_mesh
  debug = opt.model.debug # if we use the debug setting...
  # tree_traverse, child_to_parent = data_utils.mid_traverse_tree(part_tree)

  mesh_dicts = []

  if part_names is None:
    part_names = os.listdir(dataset_root_folder_coarse)
    part_names = [fnn for fnn in part_names if os.path.isdir(os.path.join(dataset_root_folder_coarse, fnn)) and os.path.isdir(os.path.join(dataset_root_folder_fine, fnn))]

  samples_fn = os.path.join(dataset_root_folder_coarse, part_names[0])
  samples_list = os.listdir(samples_fn)
  # samples list
  samples_list = [fn for fn in samples_list if fn.endswith(".obj")]

  fine_samples_fn = os.path.join(dataset_root_folder_fine, part_names[0])
  fine_samples_list = os.listdir(fine_samples_fn)
  fine_samples_list = [fn for fn in fine_samples_list if fn.endswith(".obj")]

  samples_list = [fn for fn in samples_list if fn in fine_samples_list]

  

  n_max_samples = 1000 if not debug else opt.loss.batch_size
  print(f"Debug: {debug}, n_max_samples: {n_max_samples}")

  shp_idx_to_not_ok = {}

  print(f"Loading data... Category name: {category_name}, part_names: {part_names}")

  part_nm_to_mesh_dict_idxes = {}

  for part_nm in part_names:
    coarse_cur_part_objs_folder = os.path.join(dataset_root_folder_coarse, part_nm)
    samples_list = os.listdir(coarse_cur_part_objs_folder)
    samples_list = [fn for fn in samples_list if fn.endswith(".obj")]

    fine_cur_part_objs_folder = os.path.join(dataset_root_folder_fine, part_nm)
    fine_samples_list = os.listdir(fine_cur_part_objs_folder)
    fine_samples_list = [fn for fn in fine_samples_list if fn.endswith(".obj")]

    samples_list = [fn for fn in samples_list if fn in fine_samples_list]

    for i_s, sample_nm in enumerate(samples_list):
      
      try:
        shp_idx = sample_nm.split("_")[0] # shape idx
        shp_idx = int(shp_idx)
        if shp_idx in shp_idx_to_not_ok:
          continue
      except:
        shp_idx = sample_nm.split(".")[0] # shape idx
        shp_idx = int(shp_idx)
        if shp_idx in shp_idx_to_not_ok:
          continue
      # if shp_idx not in [1, 2, 5, 7, 9, 10, 12, 11]:
      #   continue
      if valid_indices is not None:
        valid = False
        for valid_indi in valid_indices:
          if valid_indi in sample_nm:
            valid = True
            break
        if not valid:
          continue
      
      vertices_list, faces_list = [], []

      # for part_nm in part_names:
      cur_sample_cur_part_fn = os.path.join(dataset_root_folder_coarse, part_nm, sample_nm) # sample_name
      
      cur_sample_vertices, cur_sample_faces = data_utils.read_obj(cur_sample_cur_part_fn)
      # print("cc")
      # print(f"cur_sample_vertices: {cur_sample_vertices.shape}, cur_sample_faces: {len(cur_sample_faces)}")

      if category_name in ["eyeglasses"] and part_nm != "none_motion"  and "normalized" not in dataset_root_folder_coarse:
        cur_sample_vertices = cur_sample_vertices[:, [1, 0, 2]]

      vertices_list.append(cur_sample_vertices)
      faces_list.append(cur_sample_faces)

      cur_sample_vertices, cur_sample_faces = data_utils.merge_meshes(vertices_list, faces_list)
      nn_face_indices = sum([len(sub_face) for sub_face in cur_sample_faces])

      if not (len(cur_sample_vertices.shape) == 2 and cur_sample_vertices.shape[0] > 1 and len(cur_sample_faces) > 0):
        print(f"current sample coarse fn: {cur_sample_cur_part_fn}")
        continue

      if pc_cond:
        permuted_sampled_vertices = cur_sample_vertices[:, [2, 0, 1]]
        cur_sample_sampled_pts = data_utils.sample_pts_from_mesh(permuted_sampled_vertices, cur_sample_faces, npoints=512)

      try:
        cur_sample_mesh_dict = data_utils.process_mesh(cur_sample_vertices, cur_sample_faces, quantization_bits=quantization_bits, recenter_mesh=True)
      except:
        print(f"Category: {category_name}, part_name: {part_nm}, sample_nm: {sample_nm}")
        continue
      mesh_dict = cur_sample_mesh_dict
      mesh_dict['class_label'] = 0

      if pc_cond:
        mesh_dict['points'] = cur_sample_sampled_pts

      vertices, faces = mesh_dict['vertices'], mesh_dict['faces']


      fine_cur_sample_cur_part_fn = os.path.join(dataset_root_folder_fine, part_nm, sample_nm) # sample_name
      print("coarse_part_fn:", cur_sample_cur_part_fn, "fine_cur_sample_cur_part_fn:", fine_cur_sample_cur_part_fn)
      # print(cur_sample_cur_part_fn)
      fine_cur_sample_vertices, fine_cur_sample_faces = data_utils.read_obj(fine_cur_sample_cur_part_fn)

      if not (len(fine_cur_sample_vertices.shape) == 2 and fine_cur_sample_vertices.shape[0] > 1 and len(fine_cur_sample_faces) > 0):
        continue

      if category_name in ["eyeglasses"] and part_nm != "none_motion"    and "normalized" not in dataset_root_folder_fine:
        fine_cur_sample_vertices = fine_cur_sample_vertices[:, [1, 0, 2]]

      fine_cur_sample_mesh_dict = data_utils.process_mesh(fine_cur_sample_vertices, fine_cur_sample_faces, quantization_bits=quantization_bits, recenter_mesh=True)

      mesh_dict['fine_vertices'] = fine_cur_sample_mesh_dict['vertices']
      mesh_dict['fine_faces'] = fine_cur_sample_mesh_dict['faces'] # fine_cur_sample_mesh_dict

      nn_coarse_vertices, nn_fine_vertices = vertices.shape[0], fine_cur_sample_mesh_dict['vertices'].shape[0]
      nn_fine_faces = fine_cur_sample_mesh_dict['faces'].shape[0]
      print(f"Current sample, nn_coarse_vertices: {nn_coarse_vertices}, nn_fine_vertices: {nn_fine_vertices}")

      nn_vertices = vertices.shape[0]
      nn_faces = faces.shape[0]
      if not (nn_vertices <= nn_max_permite_vertices and nn_faces <= nn_max_permite_faces and nn_fine_vertices <= nn_max_permite_vertices and nn_fine_faces <= nn_max_permite_faces):
        shp_idx_to_not_ok[shp_idx] = 1
        continue
      
      if part_nm not in part_nm_to_mesh_dict_idxes:
        part_nm_to_mesh_dict_idxes[part_nm] = [len(mesh_dicts)]
      else:
        part_nm_to_mesh_dict_idxes[part_nm].append(len(mesh_dicts))

      mesh_dicts.append(mesh_dict)

      if debug and len(mesh_dicts) >= n_max_samples:
        break
    if debug and len(mesh_dicts) >= n_max_samples:
      break
  print(f"Category {category_name} loaded with {len(mesh_dicts)} instances.")
  if not ret_part_name_to_mesh_idx:
    return mesh_dicts, part_tree
  else:
    return mesh_dicts, part_tree, part_nm_to_mesh_dict_idxes



# load meshes from the dataset_root_folder (the category root folder)
# mesh dict multi part
def get_mesh_dict_list_multi_part_part_first(dataset_root_folder, part_names, valid_indices=None, category_name="eyeglasses", ret_part_name_to_mesh_idx=False, remove_du=True, split=None):
  # max permit vertices
  nn_max_permite_vertices = opt.dataset.max_permit_vertices
  nn_max_permite_faces = opt.dataset.max_permit_faces

  # n_mask_samples = opt.model.n_mask_samples
  part_tree = {'idx': 0}

  # summary_obj_fn = "summary.obj"
  quantization_bits = opt.model.quantization_bits
  recenter_mesh = opt.model.recenter_mesh
  debug = opt.model.debug # if we use the debug setting...
  # tree_traverse, child_to_parent = data_utils.mid_traverse_tree(part_tree)

  mesh_dicts = []

  if part_names is None:
    part_names = os.listdir(dataset_root_folder)
    part_names = [fnn for fnn in part_names if os.path.isdir(os.path.join(dataset_root_folder, fnn))]

  samples_fn = os.path.join(dataset_root_folder, part_names[0])
  samples_list = os.listdir(samples_fn)

  # samples list
  samples_list = [fn for fn in samples_list if fn.endswith(".obj")]

  n_max_samples = 1000 if not debug else 10 #  opt.loss.batch_size

  shp_idx_to_not_ok = {}

  print(f"Loading data... Category name: {category_name}, part_names: {part_names}")

  part_nm_to_mesh_dict_idxes = {}

  for part_nm in part_names:
    cur_part_objs_folder = os.path.join(dataset_root_folder, part_nm)
    samples_list = os.listdir(cur_part_objs_folder)
    samples_list = [fn for fn in samples_list if fn.endswith(".obj")]

    if split is not None:
      if split == 'training':
        if len(samples_list) <= 2:
          cur_split_nn_samples = min(1, len(samples_list))
        else:
          cur_split_nn_samples = int(len(samples_list) * 0.9)
        samples_list = samples_list[: cur_split_nn_samples]
      else:
        if len(samples_list) <= 2:
          cur_split_nn_samples = min(1, len(samples_list))
        else:
          cur_split_nn_samples = int(len(samples_list) * 0.9)
        samples_list = samples_list[cur_split_nn_samples: ]


    for i_s, sample_nm in enumerate(samples_list):

      # scale_idx = sample_nm.split('.')[0].split('_')[-1]
      # scale_idx = int(scale_idx)
      # if scale_idx > 100:
      #   continue
      try:
        shp_idx = sample_nm.split("_")[0] # shape idx
        shp_idx = int(shp_idx)
        if shp_idx in shp_idx_to_not_ok:
          continue
      except:
        shp_idx = sample_nm.split(".")[0] # shape idx
        shp_idx = int(shp_idx)
        if shp_idx in shp_idx_to_not_ok:
          continue
      # if shp_idx not in [1, 2, 5, 7, 9, 10, 12, 11]:
      #   continue
      if valid_indices is not None:
        valid = False
        for valid_indi in valid_indices:
          if valid_indi in sample_nm:
            valid = True
            break
        if not valid:
          continue
      
      vertices_list, faces_list = [], []

      # for part_nm in part_names:
      cur_sample_cur_part_fn = os.path.join(dataset_root_folder, part_nm, sample_nm) # sample_name
      # print(cur_sample_cur_part_fn)
      cur_sample_vertices, cur_sample_faces = data_utils.read_obj(cur_sample_cur_part_fn)
      # print("cc")
      # print(f"cur_sample_vertices: {cur_sample_vertices.shape}, cur_sample_faces: {len(cur_sample_faces)}")

      if category_name in ["eyeglasses"] and part_nm != "none_motion" and "normalized" not in dataset_root_folder:
        cur_sample_vertices = cur_sample_vertices[:, [1, 0, 2]]

      vertices_list.append(cur_sample_vertices)
      faces_list.append(cur_sample_faces)


      # 
      cur_sample_vertices, cur_sample_faces = data_utils.merge_meshes(vertices_list, faces_list)
      nn_face_indices = sum([len(sub_face) for sub_face in cur_sample_faces])

      # print(f"n_vertices: {cur_sample_vertices.shape[0]}, nn_face_indices: {nn_face_indices}")
      
      # if not (cur_sample_vertices.shape[0] <= nn_max_permite_vertices and nn_face_indices <= nn_max_permite_faces):
      #   continue
      
      ### centralize vertice  s ###

      # ins_vert_min = cur_sample_vertices.min(axis=0) # vert_min
      # ins_vert_max = cur_sample_vertices.max(axis=0) # vert_max
      # ins_vert_center = 0.5 * (ins_vert_min + ins_vert_max)
      # cur_sample_vertices = cur_sample_vertices - ins_vert_center
      # ins_vert_min = cur_sample_vertices.min(axis=0) # vert_min
      # ins_vert_max = cur_sample_vertices.max(axis=0) # vert_max
      # ins_extents = ins_vert_max - ins_vert_min # extents
      # ins_scale = np.sqrt(np.sum(ins_extents**2)) # scale # 
      # cur_sample_vertices = cur_sample_vertices / ins_scale

      try: # process meshxxx
        cur_sample_mesh_dict = data_utils.process_mesh(cur_sample_vertices, cur_sample_faces, quantization_bits=quantization_bits, recenter_mesh=True, remove_du=remove_du)
      except:
        print(f"Category: {category_name}, part_name: {part_nm}, sample_nm: {sample_nm}")
        continue
      mesh_dict = cur_sample_mesh_dict
      mesh_dict['class_label'] = 0

      # mesh_dict = {}
      # mesh_dict = cur_sample_mesh_dict
      # mesh_dict['class_label'] = 0
      vertices, faces = mesh_dict['vertices'], mesh_dict['faces']
      # print(f"vertices size: {vertices.shape}")

      if not (np.max(vertices).item() < 2 ** quantization_bits and np.min(vertices).item() >= 0):
        continue

      nn_vertices = vertices.shape[0]
      nn_faces = faces.shape[0]
      if not (nn_vertices <= nn_max_permite_vertices and nn_faces <= nn_max_permite_faces): # load and test whether they...
        shp_idx_to_not_ok[shp_idx] = 1
        continue
        
      if part_nm not in part_nm_to_mesh_dict_idxes: # part_nm
        part_nm_to_mesh_dict_idxes[part_nm] = [len(mesh_dicts)]
      else:
        part_nm_to_mesh_dict_idxes[part_nm].append(len(mesh_dicts))

      mesh_dicts.append(mesh_dict)

      if debug and len(mesh_dicts) >= n_max_samples:
        break
    if debug and len(mesh_dicts) >= n_max_samples:
      break
  print(f"Category {category_name} loaded with {len(mesh_dicts)} instances.")
  if not ret_part_name_to_mesh_idx:
    return mesh_dicts, part_tree
  else:
    return mesh_dicts, part_tree, part_nm_to_mesh_dict_idxes



def get_mesh_dict_list_multi_part_vox_meta_info(dataset_root_folder, part_names, ret_part_name_to_mesh_idx=False, remove_du=True, split=None, use_inst=False):
  print(f"Loaing meta info from root folder {dataset_root_folder} with part_names: {part_names}. Split: {split}")
  nn_max_permite_vertices = opt.dataset.max_permit_vertices # not 
  meta_info_fn = os.path.join(dataset_root_folder, "meta_infos.npy")
  cat_meta_info = np.load(meta_info_fn, allow_pickle=True).item()
  # inst: 
  # scaled voxelize ---> the whole object; # 

  mesh_dicts = []
  part_name_to_mesh_idx = {}
  for shp_inst in cat_meta_info:
    vox_fn_to_pts_nn = cat_meta_info[shp_inst] # vox_fn: 
    # part_
    for vox_fn in vox_fn_to_pts_nn:
      cur_vox_pts_nn = vox_fn_to_pts_nn[vox_fn]
      if cur_vox_pts_nn > nn_max_permite_vertices:
        continue
      part_nm_info = vox_fn.split("/")[-1].split(".")[0] # xxx.obj.binvox
      if "_ns_" in dataset_root_folder:
        part_nm_split_info = part_nm_info.split("_")
        part_nm = "_".join(part_nm_split_info[:-2])
        
      else:
        part_nm_split_info = part_nm_info.split("_")
        part_nm = "_".join(part_nm_split_info[:])
      if part_names is not None and part_nm not in part_names:
        continue
      # part_scale_idx = int(part_nm_split_info[-1])

      if use_inst:
        part_nm = part_nm + "_" + shp_inst

      cur_tot_vox_fn = os.path.join(dataset_root_folder, shp_inst, vox_fn)
      
      # pts = pts[:, [2, 0, 1]]
      # mesh_dict = {}
      # mesh_dict['vertices'] = pts
      # mesh_dict['class_label'] = 0
      if part_nm not in part_name_to_mesh_idx:
        part_name_to_mesh_idx[part_nm] = [len(mesh_dicts)]
      else:
        part_name_to_mesh_idx[part_nm].append(len(mesh_dicts))
      # mesh_dicts.append(mesh_dict)
      mesh_dicts.append(cur_tot_vox_fn)

  if split is not None:
    new_part_name_to_mesh_idx = {}
    tot_remaining_nn_meshes = 0
    old_mesh_idx_to_new_mesh_idx = {}
    for part_nm in part_name_to_mesh_idx:
      cur_part_mesh_idxes = part_name_to_mesh_idx[part_nm]
      if split == 'train':
        cur_split_nn = int(len(cur_part_mesh_idxes) * 0.90)
        cur_part_mesh_idxes = cur_part_mesh_idxes[:cur_split_nn]
        tot_remaining_nn_meshes += len(cur_part_mesh_idxes)
      else:
        cur_split_nn = int(len(cur_part_mesh_idxes) * 0.90)
        cur_part_mesh_idxes = cur_part_mesh_idxes[cur_split_nn: ]
        tot_remaining_nn_meshes += len(cur_part_mesh_idxes)
      new_part_name_to_mesh_idx[part_nm] = cur_part_mesh_idxes
    for part_nm in new_part_name_to_mesh_idx:
      cur_new_part_mesh_idxes = new_part_name_to_mesh_idx[part_nm]
      cur_new_new_part_mesh_idxes = []
      for cur_mesh_idx in cur_new_part_mesh_idxes:
        if cur_mesh_idx not in old_mesh_idx_to_new_mesh_idx:
          old_mesh_idx_to_new_mesh_idx[cur_mesh_idx] = len(old_mesh_idx_to_new_mesh_idx)
        cur_new_mesh_idx = old_mesh_idx_to_new_mesh_idx[cur_mesh_idx]
        cur_new_new_part_mesh_idxes.append(cur_new_mesh_idx)
      new_part_name_to_mesh_idx[part_nm] = cur_new_new_part_mesh_idxes
    # new_part_name_to_mesh_idx
    part_name_to_mesh_idx = new_part_name_to_mesh_idx
    new_mesh_dicts = [None for _ in range(len(old_mesh_idx_to_new_mesh_idx))]
    for cur_old_mesh_idx in old_mesh_idx_to_new_mesh_idx:
      cur_new_mesh_idx = old_mesh_idx_to_new_mesh_idx[cur_old_mesh_idx]
      new_mesh_dicts[cur_new_mesh_idx] = mesh_dicts[cur_old_mesh_idx]
    mesh_dicts = new_mesh_dicts ##### mesh_dicts, part_name_to_mesh_idx

  part_tree = {'idx': 0} # set class to zero...
  return mesh_dicts, part_tree, part_name_to_mesh_idx


def get_mesh_meta_info(dataset_root_folder, split=None):
  ##### dataset utils #####
  nn_max_permite_vertices = opt.dataset.max_permit_vertices # not 
  nn_max_permite_faces = opt.dataset.max_permit_faces
  # {shp_idx: {shp_part_nm: nn_pts}}

  meta_info_fn = os.path.join(dataset_root_folder, "meta_infos.npy")
  print(f"Meta-infos from {meta_info_fn} loaded!!!")
  # 
  cat_meta_info = np.load(meta_info_fn, allow_pickle=True).item()

  mesh_dicts = []
  part_name_to_mesh_idx = {}

  # for shp_inst in cat_meta_info:
  #   # vox_fn_to_pts_nn = cat_meta_info[shp_inst] # vox_fn: 
  #   #### pts 
    
  #   vox_fn_to_pts_nn = cat_meta_info

  #   scale_idx_to_voxel_part_fns = {}
  #   # for vox_fn in vox_fn_to_pts_nn:
  #   if "_ns_" in dataset_root_folder:
  #     for vox_fn in vox_fn_to_pts_nn:
  #       part_nm_info = vox_fn.split("/")[-1].split(".")[0] # part_nm_info
  #       part_nm_split_info = part_nm_info.split("_") # split("_")
  #       part_nm = "_".join(part_nm_split_info[:-2])
  #       scale_idx = int(part_nm_split_info[-1])
  #       if scale_idx not in scale_idx_to_voxel_part_fns:
  #         scale_idx_to_voxel_part_fns[scale_idx] = [vox_fn]
  #       else:
  #         scale_idx_to_voxel_part_fns[scale_idx].append(vox_fn)
  #     for scale_idx in scale_idx_to_voxel_part_fns: # scale_idx
  #       cur_scale_voxel_part_fns = sorted(scale_idx_to_voxel_part_fns[scale_idx])
  #       tot_n_pts_nns = sum([vox_fn_to_pts_nn[cur_vox_fn] for cur_vox_fn in cur_scale_voxel_part_fns]) # pts_nn of all parts

  #       if tot_n_pts_nns > nn_max_permite_vertices or tot_n_pts_nns < 50:
  #         continue

  #       # cur_first_voxel_part_fn = cur_scale_voxel_part_fns[0].split("/")[-1].split(".")[0]
  #       # part_nm_split_info = cur_first_voxel_part_fn.split("_")
  #       # part_nm = "_".join(part_nm_split_info[:-2])

  #       part_nm = "summary"
  #       if part_nm not in part_name_to_mesh_idx:
  #         part_name_to_mesh_idx[part_nm] = [len(mesh_dicts)]
  #       else:
  #         part_name_to_mesh_idx[part_nm].append(len(mesh_dicts))
  #       # mesh_dicts.append(mesh_dict)
  #       mesh_dicts.append(cur_scale_voxel_part_fns)
  #   else:
  
  vox_fn_to_pts_nn = cat_meta_info
  for vox_fn in vox_fn_to_pts_nn:
    # part_nm_info = vox_fn.split("/")[-1].split(".")[0] # part_nm_info
    # part_nm_split_info = part_nm_info.split("_") # split("_")
    # part_nm = "_".join(part_nm_split_info[:]) ### part_nm and split meta-info
    # scale_idx = int(part_nm_split_info[-1])
    # scale_idx = 0
    # if scale_idx not in scale_idx_to_voxel_part_fns:
    #   scale_idx_to_voxel_part_fns[scale_idx] = [vox_fn]
    # else:
    #   scale_idx_to_voxel_part_fns[scale_idx].append(vox_fn)
    # for scale_idx in scale_idx_to_voxel_part_fns: # scale_idx
    # cur_scale_voxel_part_fns = sorted(scale_idx_to_voxel_part_fns[scale_idx])
    # print(cur_scale_voxel_part_fns)
    cur_scale_voxel_part_fns = [vox_fn]
    # print([vox_fn_to_pts_nn[cur_vox_fn] for cur_vox_fn in cur_scale_voxel_part_fns])
    tot_n_pts_nns = sum([vox_fn_to_pts_nn[cur_vox_fn][0] for cur_vox_fn in cur_scale_voxel_part_fns]) # pts_nn of all parts
    tot_n_faces_nns = sum([vox_fn_to_pts_nn[cur_vox_fn][1] for cur_vox_fn in cur_scale_voxel_part_fns])

    print(f"tot_n_pts_nns: {tot_n_pts_nns}, tot_n_faces_nns: {tot_n_faces_nns}")
    if tot_n_pts_nns > nn_max_permite_vertices or tot_n_pts_nns < 10 or tot_n_faces_nns > nn_max_permite_faces:
      continue

    # cur_first_voxel_part_fn = cur_scale_voxel_part_fns[0].split("/")[-1].split(".")[0]
    # part_nm_split_info = cur_first_voxel_part_fn.split("_")
    # part_nm = "_".join(part_nm_split_info[:-2])

    part_nm = "summary" #### nop partname here...
    if part_nm not in part_name_to_mesh_idx:
      part_name_to_mesh_idx[part_nm] = [len(mesh_dicts)]
    else:
      part_name_to_mesh_idx[part_nm].append(len(mesh_dicts))
    # mesh_dicts.append(mesh_dict)
    mesh_dicts.append(vox_fn) 
      
  part_tree = {'idx': 0} # set class to zero...

  ##### get mesh_dicts and part_name information for a specific split #####
  if split is not None:
    if split == "train":
      cur_split_mesh_nn = int(len(mesh_dicts) * 0.9)
      cur_split_mesh_nn = max(cur_split_mesh_nn, 1)
      mesh_dicts = mesh_dicts[:cur_split_mesh_nn]
      for part_nm in part_name_to_mesh_idx:
        part_name_to_mesh_idx[part_nm] = [idxx for idxx in part_name_to_mesh_idx[part_nm] if idxx < len(mesh_dicts)]
    elif split == "validation":
      cur_split_mesh_nn = int(len(mesh_dicts) * 0.9)
      cur_split_mesh_nn = max(cur_split_mesh_nn, 1)
      if cur_split_mesh_nn == len(mesh_dicts):
        mesh_dicts = []
        part_name_to_mesh_idx = {}
      else:
        existing_mesh_dict_idxes = [iii for iii in range(cur_split_mesh_nn, len(mesh_dicts))]
        existing_mesh_dict_idxes = {existing_mesh_dict_idxes[ii]: ii for ii in range(len(existing_mesh_dict_idxes))}
        for part_nm in part_name_to_mesh_idx:
          ##### calculate part_name_to_mesh_idx #####
          part_name_to_mesh_idx[part_nm] = [existing_mesh_dict_idxes[idxx] for idxx in part_name_to_mesh_idx[part_nm] if idxx in existing_mesh_dict_idxes]
        ##### calculate new dict #####
        new_part_name_to_mesh_idx = {part_nm: part_name_to_mesh_idx[part_nm] for part_nm in part_name_to_mesh_idx if len(part_name_to_mesh_idx[part_nm]) > 0}
        part_name_to_mesh_idx = new_part_name_to_mesh_idx
        ##### the new mesh dicts list #####
        mesh_dicts = mesh_dicts[cur_split_mesh_nn: ] 

  return mesh_dicts, part_tree, part_name_to_mesh_idx
  



def get_mesh_dict_list_obj_vox_meta_info(dataset_root_folder, part_names, ret_part_name_to_mesh_idx=False, remove_du=True, split=None, use_inst=False):
  ##### dataset utils #####
  nn_max_permite_vertices = opt.dataset.max_permit_vertices # not 
  # {shp_idx: {shp_part_nm: nn_pts}}
  meta_info_fn = os.path.join(dataset_root_folder, "meta_infos.npy")
  # 
  cat_meta_info = np.load(meta_info_fn, allow_pickle=True).item()
  print(f"Length of meta info: {len(cat_meta_info)}...")
  # inst: 
  # scaled voxelize ---> the whole object; # 

  mesh_dicts = []
  part_name_to_mesh_idx = {}
  for shp_inst in cat_meta_info:
    vox_fn_to_pts_nn = cat_meta_info[shp_inst] # vox_fn: 
    #### pts 
    
    scale_idx_to_voxel_part_fns = {}
    # for vox_fn in vox_fn_to_pts_nn:
    if "_ns_" in dataset_root_folder:
      for vox_fn in vox_fn_to_pts_nn:
        part_nm_info = vox_fn.split("/")[-1].split(".")[0] # part_nm_info
        part_nm_split_info = part_nm_info.split("_") # split("_")
        part_nm = "_".join(part_nm_split_info[:-2])
        scale_idx = int(part_nm_split_info[-1])
        if scale_idx not in scale_idx_to_voxel_part_fns:
          scale_idx_to_voxel_part_fns[scale_idx] = [vox_fn]
        else:
          scale_idx_to_voxel_part_fns[scale_idx].append(vox_fn)
      for scale_idx in scale_idx_to_voxel_part_fns: # scale_idx
        cur_scale_voxel_part_fns = sorted(scale_idx_to_voxel_part_fns[scale_idx])
        tot_n_pts_nns = sum([vox_fn_to_pts_nn[cur_vox_fn] for cur_vox_fn in cur_scale_voxel_part_fns]) # pts_nn of all parts

        if tot_n_pts_nns > nn_max_permite_vertices or tot_n_pts_nns < 50:
          continue

        # cur_first_voxel_part_fn = cur_scale_voxel_part_fns[0].split("/")[-1].split(".")[0]
        # part_nm_split_info = cur_first_voxel_part_fn.split("_")
        # part_nm = "_".join(part_nm_split_info[:-2])

        part_nm = "summary"
        if part_nm not in part_name_to_mesh_idx:
          part_name_to_mesh_idx[part_nm] = [len(mesh_dicts)]
        else:
          part_name_to_mesh_idx[part_nm].append(len(mesh_dicts))
        # mesh_dicts.append(mesh_dict)
        mesh_dicts.append(cur_scale_voxel_part_fns)
    else:
      for vox_fn in vox_fn_to_pts_nn:
        part_nm_info = vox_fn.split("/")[-1].split(".")[0] # part_nm_info
        part_nm_split_info = part_nm_info.split("_") # split("_")
        part_nm = "_".join(part_nm_split_info[:])
        # scale_idx = int(part_nm_split_info[-1])
        scale_idx = 0
        if scale_idx not in scale_idx_to_voxel_part_fns:
          scale_idx_to_voxel_part_fns[scale_idx] = [vox_fn]
        else:
          scale_idx_to_voxel_part_fns[scale_idx].append(vox_fn)
      for scale_idx in scale_idx_to_voxel_part_fns: # scale_idx
        cur_scale_voxel_part_fns = sorted(scale_idx_to_voxel_part_fns[scale_idx])
        tot_n_pts_nns = sum([vox_fn_to_pts_nn[cur_vox_fn] for cur_vox_fn in cur_scale_voxel_part_fns]) # pts_nn of all parts

        if tot_n_pts_nns > nn_max_permite_vertices or tot_n_pts_nns < 50:
          continue

        # cur_first_voxel_part_fn = cur_scale_voxel_part_fns[0].split("/")[-1].split(".")[0]
        # part_nm_split_info = cur_first_voxel_part_fn.split("_")
        # part_nm = "_".join(part_nm_split_info[:-2])

        part_nm = "summary"
        if part_nm not in part_name_to_mesh_idx:
          part_name_to_mesh_idx[part_nm] = [len(mesh_dicts)]
        else:
          part_name_to_mesh_idx[part_nm].append(len(mesh_dicts))
        # mesh_dicts.append(mesh_dict)
        mesh_dicts.append(cur_scale_voxel_part_fns) 
      
  part_tree = {'idx': 0} # set class to zero...

  ##### get mesh_dicts and part_name information for a specific split #####
  if split is not None:
    if split == "train":
      cur_split_mesh_nn = int(len(mesh_dicts) * 0.9)
      cur_split_mesh_nn = max(cur_split_mesh_nn, 1)
      mesh_dicts = mesh_dicts[:cur_split_mesh_nn]
      for part_nm in part_name_to_mesh_idx:
        part_name_to_mesh_idx[part_nm] = [idxx for idxx in part_name_to_mesh_idx[part_nm] if idxx < len(mesh_dicts)]
    elif split == "validation":
      cur_split_mesh_nn = int(len(mesh_dicts) * 0.9)
      cur_split_mesh_nn = max(cur_split_mesh_nn, 1)
      if cur_split_mesh_nn == len(mesh_dicts):
        mesh_dicts = []
        part_name_to_mesh_idx = {}
      else:
        existing_mesh_dict_idxes = [iii for iii in range(cur_split_mesh_nn, len(mesh_dicts))]
        existing_mesh_dict_idxes = {existing_mesh_dict_idxes[ii]: ii for ii in range(len(existing_mesh_dict_idxes))}
        for part_nm in part_name_to_mesh_idx:
          ##### calculate part_name_to_mesh_idx #####
          part_name_to_mesh_idx[part_nm] = [existing_mesh_dict_idxes[idxx] for idxx in part_name_to_mesh_idx[part_nm] if idxx in existing_mesh_dict_idxes]
        ##### calculate new dict #####
        new_part_name_to_mesh_idx = {part_nm: part_name_to_mesh_idx[part_nm] for part_nm in part_name_to_mesh_idx if len(part_name_to_mesh_idx[part_nm]) > 0}
        part_name_to_mesh_idx = new_part_name_to_mesh_idx
        ##### the new mesh dicts list #####
        mesh_dicts = mesh_dicts[cur_split_mesh_nn: ] 

  return mesh_dicts, part_tree, part_name_to_mesh_idx


def get_mesh_dict_part_vox_meta_info(dataset_root_folder, part_names, ret_part_name_to_mesh_idx=False, remove_du=True, split=None, use_inst=False):
  ##### dataset utils #####
  nn_max_permite_vertices = opt.dataset.max_permit_vertices # not 
  # {shp_idx: {shp_part_nm: nn_pts}}
  meta_info_fn = os.path.join(dataset_root_folder, "meta_infos.npy")
  # 
  cat_meta_info = np.load(meta_info_fn, allow_pickle=True).item()
  print(f"Length of meta info: {len(cat_meta_info)}...")
  # inst: 
  # scaled voxelize ---> the whole object; # 

  mesh_dicts = []
  part_name_to_mesh_idx = {}
  for shp_inst in cat_meta_info:
    vox_fn_to_pts_nn = cat_meta_info[shp_inst] # vox_fn: 
    #### pts 
    
    # scale_idx_to_voxel_part_fns = {}
    # for vox_fn in vox_fn_to_pts_nn:
    if "_ns_" in dataset_root_folder:
      for vox_fn in vox_fn_to_pts_nn:
        part_nm_info = vox_fn.split("/")[-1].split(".")[0] # part_nm_info
        part_nm_split_info = part_nm_info.split("_") # split("_")
        part_nm = "_".join(part_nm_split_info[:-2])
        # scale_idx = int(part_nm_split_info[-1])
        # if scale_idx not in scale_idx_to_voxel_part_fns:
        #   scale_idx_to_voxel_part_fns[scale_idx] = [vox_fn]
        # else:
        #   scale_idx_to_voxel_part_fns[scale_idx].append(vox_fn)
        
        cur_part_nn_pts = vox_fn_to_pts_nn[vox_fn]
        if cur_part_nn_pts > nn_max_permite_vertices or cur_part_nn_pts < 50:
          continue
        if part_nm not in part_name_to_mesh_idx:
          part_name_to_mesh_idx[part_nm] = [len(mesh_dicts)]
        else:
          part_name_to_mesh_idx[part_nm].append(len(mesh_dicts))
        mesh_dicts.append([vox_fn])

      # for scale_idx in scale_idx_to_voxel_part_fns: # scale_idx
      #   cur_scale_voxel_part_fns = sorted(scale_idx_to_voxel_part_fns[scale_idx])
      #   tot_n_pts_nns = sum([vox_fn_to_pts_nn[cur_vox_fn] for cur_vox_fn in cur_scale_voxel_part_fns]) # pts_nn of all parts

      #   if tot_n_pts_nns > nn_max_permite_vertices or tot_n_pts_nns < 50:
      #     continue

      #   # cur_first_voxel_part_fn = cur_scale_voxel_part_fns[0].split("/")[-1].split(".")[0]
      #   # part_nm_split_info = cur_first_voxel_part_fn.split("_")
      #   # part_nm = "_".join(part_nm_split_info[:-2])

      #   part_nm = "summary"
      #   if part_nm not in part_name_to_mesh_idx:
      #     part_name_to_mesh_idx[part_nm] = [len(mesh_dicts)]
      #   else:
      #     part_name_to_mesh_idx[part_nm].append(len(mesh_dicts))
      #   # mesh_dicts.append(mesh_dict)
      #   mesh_dicts.append(cur_scale_voxel_part_fns)
    else:
      for vox_fn in vox_fn_to_pts_nn:
        part_nm_info = vox_fn.split("/")[-1].split(".")[0] # part_nm_info
        part_nm_split_info = part_nm_info.split("_") # split("_")
        part_nm = "_".join(part_nm_split_info[:])
        # scale_idx = int(part_nm_split_info[-1])
        scale_idx = 0
        
        cur_part_nn_pts = vox_fn_to_pts_nn[vox_fn]
        if cur_part_nn_pts > nn_max_permite_vertices or cur_part_nn_pts < 50:
          continue
        if part_nm not in part_name_to_mesh_idx:
          part_name_to_mesh_idx[part_nm] = [len(mesh_dicts)]
        else:
          part_name_to_mesh_idx[part_nm].append(len(mesh_dicts))
        mesh_dicts.append([vox_fn])


      #   if scale_idx not in scale_idx_to_voxel_part_fns:
      #     scale_idx_to_voxel_part_fns[scale_idx] = [vox_fn]
      #   else:
      #     scale_idx_to_voxel_part_fns[scale_idx].append(vox_fn)
      # for scale_idx in scale_idx_to_voxel_part_fns: # scale_idx
      #   cur_scale_voxel_part_fns = sorted(scale_idx_to_voxel_part_fns[scale_idx])
      #   tot_n_pts_nns = sum([vox_fn_to_pts_nn[cur_vox_fn] for cur_vox_fn in cur_scale_voxel_part_fns]) # pts_nn of all parts

      #   if tot_n_pts_nns > nn_max_permite_vertices or tot_n_pts_nns < 50:
      #     continue

      #   # cur_first_voxel_part_fn = cur_scale_voxel_part_fns[0].split("/")[-1].split(".")[0]
      #   # part_nm_split_info = cur_first_voxel_part_fn.split("_")
      #   # part_nm = "_".join(part_nm_split_info[:-2])

      #   part_nm = "summary"
      #   if part_nm not in part_name_to_mesh_idx:
      #     part_name_to_mesh_idx[part_nm] = [len(mesh_dicts)]
      #   else:
      #     part_name_to_mesh_idx[part_nm].append(len(mesh_dicts))
      #   # mesh_dicts.append(mesh_dict)
      #   mesh_dicts.append(cur_scale_voxel_part_fns) 
      
  part_tree = {'idx': 0} # set class to zero...

  ##### get mesh_dicts and part_name information for a specific split #####
  if split is not None:
    if split == "train":
      cur_split_mesh_nn = int(len(mesh_dicts) * 0.9)
      cur_split_mesh_nn = max(cur_split_mesh_nn, 1)
      mesh_dicts = mesh_dicts[:cur_split_mesh_nn]
      new_part_name_to_mesh_idx = {}
      for part_nm in part_name_to_mesh_idx:
        part_name_to_mesh_idx[part_nm] = [idxx for idxx in part_name_to_mesh_idx[part_nm] if idxx < len(mesh_dicts)]
        if len(part_name_to_mesh_idx[part_nm]) > 0:
          new_part_name_to_mesh_idx[part_nm] = part_name_to_mesh_idx[part_nm]
      part_name_to_mesh_idx = new_part_name_to_mesh_idx
    elif split == "validation":
      cur_split_mesh_nn = int(len(mesh_dicts) * 0.9)
      cur_split_mesh_nn = max(cur_split_mesh_nn, 1)
      if cur_split_mesh_nn == len(mesh_dicts):
        mesh_dicts = []
        part_name_to_mesh_idx = {}
      else:
        existing_mesh_dict_idxes = [iii for iii in range(cur_split_mesh_nn, len(mesh_dicts))]
        existing_mesh_dict_idxes = {existing_mesh_dict_idxes[ii]: ii for ii in range(len(existing_mesh_dict_idxes))}
        for part_nm in part_name_to_mesh_idx:
          ##### calculate part_name_to_mesh_idx #####
          part_name_to_mesh_idx[part_nm] = [existing_mesh_dict_idxes[idxx] for idxx in part_name_to_mesh_idx[part_nm] if idxx in existing_mesh_dict_idxes]
        ##### calculate new dict #####
        new_part_name_to_mesh_idx = {part_nm: part_name_to_mesh_idx[part_nm] for part_nm in part_name_to_mesh_idx if len(part_name_to_mesh_idx[part_nm]) > 0}
        part_name_to_mesh_idx = new_part_name_to_mesh_idx
        ##### the new mesh dicts list #####
        mesh_dicts = mesh_dicts[cur_split_mesh_nn: ] 

  return mesh_dicts, part_tree, part_name_to_mesh_idx


#### dataset root folder ####
def get_mesh_dict_part_subd_mesh_meta_info(dataset_root_folder, part_names, ret_part_name_to_mesh_idx=False, remove_du=True, split=None, use_inst=False):
  ##### dataset utils #####
  # max_permit_vertices
  nn_max_permite_vertices = opt.dataset.max_permit_vertices
  nn_max_permite_faces = opt.dataset.max_permit_faces
  # {shp_idx: {shp_part_nm: nn_pts}}
  meta_info_fn = os.path.join(dataset_root_folder, "meta_infos.npy")

  permite_obj_idx = None
  # permite_obj_idx = [10, 39, 114, 128, 103, 102, 70, 71, 98, 162, 149, 45, 51, 50, 53, 42, 81, 155, 41, 197, 130, 122, 37]
  
  st_subd_idx = opt.dataset.st_subd_idx
  subdn = opt.dataset.subdn
  part_name_to_mesh_idx = {}
  # mesh_dicts, part_tree, part_name_to_mesh_idx
  if not os.path.exists(meta_info_fn):
    subd_folders = os.listdir(dataset_root_folder)
    subd_folders = [fn for fn in subd_folders if os.path.isdir(os.path.join(dataset_root_folder, fn))]
    if permite_obj_idx is not None:
      subd_folders = [fn for fn in subd_folders if int(fn) in permite_obj_idx] # valid obj idxes
    # mesh_dicts = subd_folders

    tot_subd_folders  = []
    for cur_subd_folder in subd_folders:
      cur_tot_subd_folder = os.path.join(dataset_root_folder, cur_subd_folder)
      cur_tot_subd_files = os.listdir(cur_tot_subd_folder)
      cur_tot_subd_files = [fnn for fnn in cur_tot_subd_files if fnn.endswith(".obj")]
      if len(cur_tot_subd_files) > subdn:
        tot_subd_folders.append(cur_subd_folder)
    subd_folders = tot_subd_folders #### subd folders
    mesh_dicts = [os.path.join(dataset_root_folder, fn) for fn in subd_folders]
# 
    # tot_mesh_dicts = []
    
    if split is not None:
      tot_nn_meshes = len(mesh_dicts)
      train_nn = int(0.9 * tot_nn_meshes)
      if split == 'train':
        # mesh_dicts = mesh_dicts[:100]
        mesh_dicts = mesh_dicts[:train_nn]
        # mesh_dicts = mesh_dicts[:13]
      else:
        # mesh_dicts = mesh_dicts[-10:]
        mesh_dicts = mesh_dicts[train_nn:]
        # mesh_dicts = mesh_dicts[-100:]
        # mesh_dicts = mesh_dicts[:100]
        # mesh_dicts = mesh_dicts[:2]
        # mesh_dicts = mesh_dicts[:100]
        # mesh_dicts = mesh_dicts[-10:] ##### mesh_dicts
        # mesh_dicts = mesh_dicts[-1:]
    # mesh_dicts = mesh_dicts[:10]
    part_name_to_mesh_idx['summary'] = range(len(mesh_dicts))
  else:
    print(f"Loading meatinfo file from{meta_info_fn}!")
    meta_info = np.load(meta_info_fn, allow_pickle=True).item()
    mesh_dicts = []
    for cur_subd_folder in meta_info:
      cur_inst_subd_info = meta_info[cur_subd_folder]
      subd_prefix_nm = "subd"
      if not (st_subd_idx in cur_inst_subd_info and subdn - 1 in cur_inst_subd_info):
        continue
      st_subd_info = cur_inst_subd_info[st_subd_idx]
      ed_subd_info = cur_inst_subd_info[subdn - 1]
      # print(f"curinstsubdinfo: {cur_inst_subd_info}")
      if st_subd_info[0] > 0 and st_subd_info[1] > 0 and ed_subd_info[0] > 0 and ed_subd_info[1] > 0 and ed_subd_info[0] < nn_max_permite_vertices and ed_subd_info[1] < nn_max_permite_faces:
        mesh_dicts.append(cur_subd_folder)
      # cur_subd_pts_nn = meta_info[cur_subd_folder]
      # if cur_subd_pts_nn <= nn_max_permite_vertices:
      #   mesh_dicts.append(cur_subd_folder)
    part_name_to_mesh_idx['summary'] = range(len(mesh_dicts))

    # mesh_dicts = 
  # 
  # cat_meta_info = np.load(meta_info_fn, allow_pickle=True).item()
  # print(f"Length of meta info: {len(cat_meta_info)}...")
  # # inst: 
  # scaled voxelize ---> the whole object; # 

  # mesh_dicts = []
  # part_name_to_mesh_idx = {}
  # for shp_inst in cat_meta_info:
  #   vox_fn_to_pts_nn = cat_meta_info[shp_inst] # vox_fn: 
  #   #### pts 
    
  #   # scale_idx_to_voxel_part_fns = {}
  #   # for vox_fn in vox_fn_to_pts_nn:
  #   if "_ns_" in dataset_root_folder:
  #     for vox_fn in vox_fn_to_pts_nn:
  #       part_nm_info = vox_fn.split("/")[-1].split(".")[0] # part_nm_info
  #       part_nm_split_info = part_nm_info.split("_") # split("_")
  #       part_nm = "_".join(part_nm_split_info[:-2])
  #       # scale_idx = int(part_nm_split_info[-1])
  #       # if scale_idx not in scale_idx_to_voxel_part_fns:
  #       #   scale_idx_to_voxel_part_fns[scale_idx] = [vox_fn]
  #       # else:
  #       #   scale_idx_to_voxel_part_fns[scale_idx].append(vox_fn)
        
  #       cur_part_nn_pts = vox_fn_to_pts_nn[vox_fn]
  #       if cur_part_nn_pts > nn_max_permite_vertices or cur_part_nn_pts < 50:
  #         continue
  #       if part_nm not in part_name_to_mesh_idx:
  #         part_name_to_mesh_idx[part_nm] = [len(mesh_dicts)]
  #       else:
  #         part_name_to_mesh_idx[part_nm].append(len(mesh_dicts))
  #       mesh_dicts.append([vox_fn])

      # for scale_idx in scale_idx_to_voxel_part_fns: # scale_idx
      #   cur_scale_voxel_part_fns = sorted(scale_idx_to_voxel_part_fns[scale_idx])
      #   tot_n_pts_nns = sum([vox_fn_to_pts_nn[cur_vox_fn] for cur_vox_fn in cur_scale_voxel_part_fns]) # pts_nn of all parts

      #   if tot_n_pts_nns > nn_max_permite_vertices or tot_n_pts_nns < 50:
      #     continue

      #   # cur_first_voxel_part_fn = cur_scale_voxel_part_fns[0].split("/")[-1].split(".")[0]
      #   # part_nm_split_info = cur_first_voxel_part_fn.split("_")
      #   # part_nm = "_".join(part_nm_split_info[:-2])

      #   part_nm = "summary"
      #   if part_nm not in part_name_to_mesh_idx:
      #     part_name_to_mesh_idx[part_nm] = [len(mesh_dicts)]
      #   else:
      #     part_name_to_mesh_idx[part_nm].append(len(mesh_dicts))
      #   # mesh_dicts.append(mesh_dict)
      #   mesh_dicts.append(cur_scale_voxel_part_fns)
    # else:
    #   for vox_fn in vox_fn_to_pts_nn:
    #     part_nm_info = vox_fn.split("/")[-1].split(".")[0] # part_nm_info
    #     part_nm_split_info = part_nm_info.split("_") # split("_")
    #     part_nm = "_".join(part_nm_split_info[:])
    #     # scale_idx = int(part_nm_split_info[-1])
    #     scale_idx = 0
        
    #     cur_part_nn_pts = vox_fn_to_pts_nn[vox_fn]
    #     if cur_part_nn_pts > nn_max_permite_vertices or cur_part_nn_pts < 50:
    #       continue
    #     if part_nm not in part_name_to_mesh_idx:
    #       part_name_to_mesh_idx[part_nm] = [len(mesh_dicts)]
    #     else:
    #       part_name_to_mesh_idx[part_nm].append(len(mesh_dicts))
    #     mesh_dicts.append([vox_fn])


      #   if scale_idx not in scale_idx_to_voxel_part_fns:
      #     scale_idx_to_voxel_part_fns[scale_idx] = [vox_fn]
      #   else:
      #     scale_idx_to_voxel_part_fns[scale_idx].append(vox_fn)
      # for scale_idx in scale_idx_to_voxel_part_fns: # scale_idx
      #   cur_scale_voxel_part_fns = sorted(scale_idx_to_voxel_part_fns[scale_idx])
      #   tot_n_pts_nns = sum([vox_fn_to_pts_nn[cur_vox_fn] for cur_vox_fn in cur_scale_voxel_part_fns]) # pts_nn of all parts

      #   if tot_n_pts_nns > nn_max_permite_vertices or tot_n_pts_nns < 50:
      #     continue

      #   # cur_first_voxel_part_fn = cur_scale_voxel_part_fns[0].split("/")[-1].split(".")[0]
      #   # part_nm_split_info = cur_first_voxel_part_fn.split("_")
      #   # part_nm = "_".join(part_nm_split_info[:-2])

      #   part_nm = "summary"
      #   if part_nm not in part_name_to_mesh_idx:
      #     part_name_to_mesh_idx[part_nm] = [len(mesh_dicts)]
      #   else:
      #     part_name_to_mesh_idx[part_nm].append(len(mesh_dicts))
      #   # mesh_dicts.append(mesh_dict)
      #   mesh_dicts.append(cur_scale_voxel_part_fns) 
      
  part_tree = {'idx': 0} # set class to zero...

  # ##### get mesh_dicts and part_name information for a specific split #####
  # if split is not None:
  #   if split == "train":
  #     cur_split_mesh_nn = int(len(mesh_dicts) * 0.9)
  #     cur_split_mesh_nn = max(cur_split_mesh_nn, 1)
  #     mesh_dicts = mesh_dicts[:cur_split_mesh_nn]
  #     new_part_name_to_mesh_idx = {}
  #     for part_nm in part_name_to_mesh_idx:
  #       part_name_to_mesh_idx[part_nm] = [idxx for idxx in part_name_to_mesh_idx[part_nm] if idxx < len(mesh_dicts)]
  #       if len(part_name_to_mesh_idx[part_nm]) > 0:
  #         new_part_name_to_mesh_idx[part_nm] = part_name_to_mesh_idx[part_nm]
  #     part_name_to_mesh_idx = new_part_name_to_mesh_idx
  #   elif split == "validation":
  #     cur_split_mesh_nn = int(len(mesh_dicts) * 0.9)
  #     cur_split_mesh_nn = max(cur_split_mesh_nn, 1)
  #     if cur_split_mesh_nn == len(mesh_dicts):
  #       mesh_dicts = []
  #       part_name_to_mesh_idx = {}
  #     else:
  #       existing_mesh_dict_idxes = [iii for iii in range(cur_split_mesh_nn, len(mesh_dicts))]
  #       existing_mesh_dict_idxes = {existing_mesh_dict_idxes[ii]: ii for ii in range(len(existing_mesh_dict_idxes))}
  #       for part_nm in part_name_to_mesh_idx:
  #         ##### calculate part_name_to_mesh_idx #####
  #         part_name_to_mesh_idx[part_nm] = [existing_mesh_dict_idxes[idxx] for idxx in part_name_to_mesh_idx[part_nm] if idxx in existing_mesh_dict_idxes]
  #       ##### calculate new dict #####
  #       new_part_name_to_mesh_idx = {part_nm: part_name_to_mesh_idx[part_nm] for part_nm in part_name_to_mesh_idx if len(part_name_to_mesh_idx[part_nm]) > 0}
  #       part_name_to_mesh_idx = new_part_name_to_mesh_idx
  #       ##### the new mesh dicts list #####
  #       mesh_dicts = mesh_dicts[cur_split_mesh_nn: ] 

  return mesh_dicts, part_tree, part_name_to_mesh_idx


def read_binvox_to_pts(binvox_fn):
  with open(binvox_fn, 'rb') as f:
    model = binvox_rw.read_as_3d_array(f)
  pts = []
  dim_x, dim_y, dim_z = model.dims
  for i_x in range(dim_x):
    for i_y in range(dim_y):
        for i_z in range(dim_z):
            cur_model_data = int(model.data[i_x, i_y, i_z].item())
            if cur_model_data > 0:
                pts.append([i_x, i_y, i_z])
  pts = np.array(pts, dtype=np.long) # n_pts x 3
  pts = pts[:, [2, 0, 1]]
  mesh_dict = {}
  mesh_dict['vertices'] = pts
  mesh_dict['class_label'] = 0
  return mesh_dict
  
def read_binvox_list_to_pts(binvox_fns):
  tot_pts = []
  # part_idx_to_pts_idxes = 
  pts_idxes = []
  tot_n_pts = 0
  for binvox_fn in binvox_fns:
    with open(binvox_fn, 'rb') as f:
      model = binvox_rw.read_as_3d_array(f)
      pts = []
      dim_x, dim_y, dim_z = model.dims
      for i_x in range(dim_x):
        for i_y in range(dim_y):
            for i_z in range(dim_z):
                cur_model_data = int(model.data[i_x, i_y, i_z].item())
                if cur_model_data > 0:
                    pts.append([i_x, i_y, i_z])
    pts = np.array(pts, dtype=np.long) # n_pts x 3
    pts = pts[:, [2, 0, 1]]
    tot_pts.append(pts)
    cur_part_pts_idxes = np.arange(tot_n_pts, tot_n_pts + pts.shape[0], step=1)
    pts_idxes.append(cur_part_pts_idxes)
    tot_n_pts += pts.shape[0]
  tot_pts = np.concatenate(tot_pts, axis=0)
  mesh_dict = {}
  mesh_dict['vertices'] = tot_pts
  mesh_dict['class_label'] = 0
  return mesh_dict, pts_idxes

def read_edges(faces):
  edges_list = []
  edges_to_exist = {}
  
  for i_f, cur_f in enumerate(faces):
    for i0, v0 in enumerate(cur_f):
      i1 = (i0 + 1) % len(cur_f)
      v1 = cur_f[i1]
      edge_pair = (v0, v1) if v0 < v1 else (v1, v0)
      if v0 != v1 and edge_pair not in edges_to_exist:
        edges_to_exist[edge_pair] = len(edges_list)
        edges_list.append(edge_pair)
        
  return edges_list, edges_to_exist
        
def upsample_vertices(vertices, edges): # start from 1 # reform faces using vertices and edges, edge_pair_to_idx
  upsample_vertices = []
  edges = sorted(edges, key=lambda ii: ii, reverse=False)
  tot_nn_verts = vertices.shape[0] #### ori_nn_verts
  edge_pair_to_vert_idx = {}
  
  for i_e, cur_e in enumerate(edges):
    v0, v1 = cur_e
    vert0, vert1 = vertices[v0 - 1], vertices[v1 - 1]
    upsample_vert = (vert0 + vert1) / float(2.)
    upsample_vertices.append(upsample_vert)
    edge_pair_to_vert_idx[cur_e] = tot_nn_verts + 1
    tot_nn_verts += 1
  return upsample_vertices, edge_pair_to_vert_idx


def upsample_faces(edge_pair_to_vert_idx, faces):
  new_faces = []
  new_face_rel_idxes = [[0, 3, 5], [1, 4, 3], [2, 5, 4], [3, 4, 5]]
  
  for i_f, cur_f in enumerate(faces):
    new_vert_idxes = []
    for i0, v0 in enumerate(cur_f):
      i1 = (i0 + 1) % len(cur_f)
      v1 = cur_f[i1]
      edge_pair = (v0, v1) if v0 < v1 else (v1, v0)
      cur_edge_vert_idx = edge_pair_to_vert_idx[edge_pair]
      new_vert_idxes.append(cur_edge_vert_idx)
    tot_vert_idxes = cur_f + new_vert_idxes
    new_face_abs_idxes = []
    for i_rel_face, rel_face in enumerate(new_face_rel_idxes):
      new_abs_face = [tot_vert_idxes[ii] for ii in rel_face]
      new_face_abs_idxes.append(new_abs_face)
    new_faces += new_face_abs_idxes
  return new_faces
    

maxx_scale = 1e-5

def from_faces_to_half_flaps(faces, vertices, edges_to_midpoint_idx):
  ### 
  nn_even_vertices = vertices.shape[0]
  edges_to_face_idxes = {}
  for i_f, cur_f in enumerate(faces):
    for i0, v0 in enumerate(cur_f):
      i1 = (i0 + 1) % len(cur_f)
      v1 = cur_f[i1]
      v_first, v_second = v0 if v0 < v1 else v1, v1 if v0 < v1 else v0
      cur_half_edge = (v_first, v_second)
      if cur_half_edge not in edges_to_face_idxes:
        edges_to_face_idxes[cur_half_edge] = [i_f]
      else:
        edges_to_face_idxes[cur_half_edge].append(i_f)
  half_flaps = []
  half_flaps_mid_points_idx = []
  for cur_half_edge in edges_to_face_idxes:
    cur_half_faces = [faces[f_idx] for f_idx in edges_to_face_idxes[cur_half_edge]]
    if len(cur_half_faces) == 1:
      cur_half_faces = cur_half_faces + cur_half_faces 
    cur_half_vert_idx_to_exist = {vert_idx : 1 for vert_idx in cur_half_edge}
    remain_vert_idx = [[ii for ii in cur_f if ii not in cur_half_vert_idx_to_exist] for cur_f in cur_half_faces]
    remain_vert_idx = [ii[0] for ii in remain_vert_idx] ### expected length = 2
    cur_half_flap_vert_idx = [cur_half_edge[0], remain_vert_idx[0], cur_half_edge[1], remain_vert_idx[1]]
    half_flaps.append(cur_half_flap_vert_idx)
    cur_half_flap_vert_idx = [cur_half_edge[0], remain_vert_idx[1], cur_half_edge[1], remain_vert_idx[0]]
    half_flaps.append(cur_half_flap_vert_idx)
    
    cur_midpoint_idx = edges_to_midpoint_idx[cur_half_edge] + nn_even_vertices
    half_flaps_mid_points_idx.append(cur_midpoint_idx)
    half_flaps_mid_points_idx.append(cur_midpoint_idx)
  half_flaps = np.array(half_flaps, dtype=np.long) - 1 ### (n_edges x 2) x 4
  half_flaps_mid_points_idx = np.array(half_flaps_mid_points_idx, dtype=np.long)
  return half_flaps, half_flaps_mid_points_idx


# TODO: test this function
# - when to stop ---> stop sampling when sampling an invalid value; invalid xxx or other things...
def sort_vertices(vertices, faces):
  sort_inds = np.lexsort(vertices.T) # 
  vertices = vertices[sort_inds]

  ### 
  # Re-index faces and tris to re-ordered vertices. ### np.argsort(sort_inds)[f]
  faces = [np.argsort(sort_inds)[f] for f in faces]
  faces = [f.tolist() for f in faces]
  return vertices, faces

def read_subd_meshes(obj_folder):
  fake_upsample = opt.dataset.fake_upsample
  global maxx_scale
  subd_meshes = os.listdir(obj_folder)
  subd_meshes = [fn for fn in subd_meshes if os.path.exists(os.path.join(obj_folder, fn))]
  subd_meshes = [(fn.split(".")[0].split("_")[0], int(fn.split(".")[0].split("_")[1])) for fn in subd_meshes]
  subd_meshes = sorted(subd_meshes, key=lambda ii: ii[1], reverse=False)
  subd_idx_to_mesh = {}

  ar_subd_idx = opt.dataset.ar_subd_idx

  subdn = opt.dataset.subdn
  for fn_pair in subd_meshes:
    fn_indicator, subd_idx = fn_pair # subd_idx ---> subd_idx
    if subd_idx >= subdn: # larger than subdn...
      continue
    cur_mesh_fn = fn_indicator + "_" + str(subd_idx) + ".obj" ### cur_mesh_fn
    cur_mesh_fn = os.path.join(obj_folder, cur_mesh_fn)
    try:
      vertices, faces = data_utils.read_obj_file_ours(cur_mesh_fn)
      vertices = vertices[:, [2, 0, 1]] # 
    except:
      continue
    # vertices, 
    # local patch feature;   the concept of half flap  
    # local patch features; 
    # even points and odd points; 
    # local patch ---> idx0, idx1, idx2, idx3; xyz embedding features; local pathc features; half flap features
    if fake_upsample and subd_idx > 0 and vertices.shape[0] == subd_idx_to_mesh[1 - 1]['vertices'].shape[0]:
      vertices = subd_idx_to_mesh[subd_idx - 1]['upsampled_vertices']
      faces = subd_idx_to_mesh[subd_idx - 1]['upsampled_faces']

    if subd_idx > 0:
      cur_subd_fake_upsample = subd_idx_to_mesh[subd_idx - 1]['fake_upsampled_vertices']
      cur_subd_fake_faces = subd_idx_to_mesh[subd_idx - 1]['fake_upsampled_faces']
    else:
      cur_subd_fake_upsample = vertices
      cur_subd_fake_faces = faces
    


    edges_list, edges_to_exist = read_edges(faces)
    #### from faces to half flaps ####
    half_flaps, half_flaps_mid_points_idx = from_faces_to_half_flaps(faces, vertices, edges_to_exist)
    # upsample_vertices
    up_verts, edge_pair_to_vert_idx = upsample_vertices(vertices, edges_list) ### upsample vertices

    fake_edges_list, _ = read_edges(cur_subd_fake_faces)
    nex_cur_subd_fake_upsample, nex_edge_pair_to_vert_idx = upsample_vertices(cur_subd_fake_upsample, fake_edges_list)
    cur_subd_fake_upsample = np.concatenate([cur_subd_fake_upsample, nex_cur_subd_fake_upsample], axis=0)
    cur_subd_fake_faces_upsample = upsample_faces(nex_edge_pair_to_vert_idx, cur_subd_fake_faces)

    up_verts = np.concatenate([vertices, up_verts], axis=0)
    up_faces = upsample_faces(edge_pair_to_vert_idx, faces)
    # subd_idx_to_mesh[subd_idx] = {
    #   'vertices': vertices,  'faces': faces
    # }
    # vertices = data_utils.quantize_verts(vertices, n_bits=opt.dataset.)
    subd_idx_to_mesh[subd_idx] = {
      'vertices': vertices, 'upsampled_vertices': up_verts, 'faces': faces, 'upsampled_faces': up_faces, 
      'half_flaps': half_flaps, 'half_flaps_mid_points_idx': half_flaps_mid_points_idx, 'fake_upsampled_vertices': cur_subd_fake_upsample, 'fake_upsampled_faces': cur_subd_fake_faces_upsample
    }
    # subd_idx_to_mesh[subd_idx] = subd_idx_to_mesh
    # subd_idx_to_mesh
  
  try:
    subd_last_vertices = subd_idx_to_mesh[subdn - 1]['vertices']
  except:
    print(f"obj_folder: {obj_folder}")
    raise ValueError("zz")
  subd_last_vertices_min = subd_last_vertices.min(axis=0)
  subd_last_vertices_max = subd_last_vertices.max(axis=0)
  subd_last_vertices_center = 0.5 * (subd_last_vertices_min + subd_last_vertices_max)
  subd_idx_to_vertices_scale = {}

  maxx_subd_vertices_scale = 1e-5


  #### ar_vertices: n_verts x 3; ar_faces: list of faces
  zero_st_faces = subd_idx_to_mesh[ar_subd_idx]['faces']
  zero_st_faces = [[cur_f - 1 for cur_f in sub_f] for sub_f in zero_st_faces]
  ar_vertices, ar_faces = np.zeros_like(subd_idx_to_mesh[ar_subd_idx]['vertices']), zero_st_faces # sample faces / upsample faces
  ar_vertices[:, :] = subd_idx_to_mesh[ar_subd_idx]['vertices'][:, :]
  ar_vertices, ar_faces = sort_vertices(ar_vertices, ar_faces)
  #### ar_faces --> flatten_ar_faces
  ar_faces_flatten = data_utils.flatten_faces(ar_faces) # ar_faces
  subd_idx_to_mesh['ar'] = {
    'vertices': ar_vertices, 'faces': ar_faces_flatten, 'faces_unflatten': ar_faces
  }

  
  

  for i_subd in subd_idx_to_mesh:
    if not isinstance(i_subd, int):
      continue
    cur_subd_vertices = subd_idx_to_mesh[i_subd]['vertices']
    cur_subd_vertices_centered = cur_subd_vertices - np.reshape(subd_last_vertices_center, (1, 3))
    cur_subd_vertices_scale = data_utils.get_vertices_scale(cur_subd_vertices_centered).item()
    maxx_subd_vertices_scale = max(maxx_subd_vertices_scale, cur_subd_vertices_scale)

    cur_subd_vertices_upsampled = subd_idx_to_mesh[i_subd]['upsampled_vertices']
    cur_subd_vertices_upsampled_centered = cur_subd_vertices_upsampled - np.reshape(subd_last_vertices_center, (1, 3))
    cur_subd_vertices_upsampled_scale = data_utils.get_vertices_scale(cur_subd_vertices_upsampled_centered).item()
    maxx_subd_vertices_scale = max(maxx_subd_vertices_scale, cur_subd_vertices_upsampled_scale)

    # maxx_subd_vertices_scale = 83.0
    # maxx_scale = max(maxx_scale, maxx_subd_vertices_scale)
    # print(f"maxx_scale: {maxx_scale}")

  for i_subd in subd_idx_to_mesh:
    if not isinstance(i_subd, int):
      continue
    cur_subd_vertices = subd_idx_to_mesh[i_subd]['vertices']
    cur_subd_vertices_upsampled = subd_idx_to_mesh[i_subd]['upsampled_vertices']
    # print(f"subd_vertices: {cur_subd_vertices[:10]}")
    cur_subd_vertices = (cur_subd_vertices - np.reshape(subd_last_vertices_center, (1, 3))) / maxx_subd_vertices_scale
    cur_subd_vertices_upsampled = (cur_subd_vertices_upsampled - np.reshape(subd_last_vertices_center, (1, 3))) / maxx_subd_vertices_scale
    
    # cur_subd_vertices = data_utils.quantize_verts(cur_subd_vertices, n_bits=opt.model.quantization_bits)
    # cur_subd_vertices_upsampled = data_utils.quantize_verts(cur_subd_vertices_upsampled, n_bits=opt.model.quantization_bits)
    subd_idx_to_mesh[i_subd]['vertices'] = cur_subd_vertices
    subd_idx_to_mesh[i_subd]['upsampled_vertices'] = cur_subd_vertices_upsampled # upsampled vertices
  # subdn = opt.dataset.subdn
  # for i_subd in range(subdn)

  

  return subd_idx_to_mesh
    
    


# load meshes from the dataset_root_folder (the category root folder)
# mesh dict multi part
def get_mesh_dict_list_multi_part_vox(dataset_root_folder, part_names, ret_part_name_to_mesh_idx=False, remove_du=True, split=None):
  # max permit vertices
  nn_max_permite_vertices = opt.dataset.max_permit_vertices # not 

  # part_name_to_shp_idx = {}
  
  shp_insts = os.listdir(dataset_root_folder)
  shp_insts = [fn for fn in shp_insts if os.path.isdir(os.path.join(dataset_root_folder, fn))]
  
  mesh_dicts = []
  part_name_to_mesh_idx = {}
  for shp_inst in shp_insts:
    cur_shp_folder = os.path.join(dataset_root_folder, shp_inst)
    cur_shp_voxs = os.listdir(cur_shp_folder)
    cur_shp_voxs = [fn for fn in cur_shp_voxs if fn.endswith(".binvox")]

    if split is not None:
      if split == 'training':
        if len(cur_shp_voxs) <= 2: # number of samples and voxels
          cur_split_nn_samples = min(1, len(cur_shp_voxs))
        else:
          cur_split_nn_samples = int(len(cur_shp_voxs) * 0.8)
        cur_shp_voxs = cur_shp_voxs[: cur_split_nn_samples]
      else:
        if len(cur_shp_voxs) <= 2:
          cur_split_nn_samples = min(1, len(cur_shp_voxs))
        else:
          cur_split_nn_samples = int(len(cur_shp_voxs) * 0.8)
        cur_shp_voxs = cur_shp_voxs[cur_split_nn_samples: ]

    
    for vox_fn in cur_shp_voxs:
      part_nm = vox_fn.split(".")[0] # xxx.obj.binvox
      
      if part_names is not None and part_nm not in part_names:
        continue
      cur_vox_fn = os.path.join(cur_shp_folder, vox_fn)
      print(f"loading binvox file: {cur_vox_fn}...")
      with open(cur_vox_fn, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)
      pts = []
      dim_x, dim_y, dim_z = model.dims
      for i_x in range(dim_x):
        for i_y in range(dim_y):
            for i_z in range(dim_z):
                cur_model_data = int(model.data[i_x, i_y, i_z].item())
                if cur_model_data > 0:
                    pts.append([i_x, i_y, i_z])
      pts = np.array(pts, dtype=np.long) # n_pts x 3
      if pts.shape[0] > nn_max_permite_vertices:
        continue
      # permute pts xyzs
      pts = pts[:, [2, 0, 1]]
      mesh_dict = {}
      mesh_dict['vertices'] = pts
      mesh_dict['class_label'] = 0
      if part_nm not in part_name_to_mesh_idx:
        part_name_to_mesh_idx[part_nm] = [len(mesh_dicts)]
      else:
        part_name_to_mesh_idx[part_nm].append(len(mesh_dicts))
      mesh_dicts.append(mesh_dict)
  
  part_tree = {'idx': 0}

  #### part name to mesh idx ####

  if not ret_part_name_to_mesh_idx:
    return mesh_dicts, part_tree
  else:
    return mesh_dicts, part_tree, part_name_to_mesh_idx


# load meshes from the dataset_root_folder (the category root folder)
# mesh dict multi part
def get_mesh_dict_list_multi_part_mesh_vertices(dataset_root_folder, part_names, ret_part_name_to_mesh_idx=False, remove_du=True, split=None):
  # max permit vertices
  nn_max_permite_vertices = opt.dataset.max_permit_vertices # not  # part names and
  quantization_bits = opt.model.quantization_bits

  # part_name_to_shp_idx = {}
  
  shp_insts = os.listdir(dataset_root_folder)
  shp_insts = [fn for fn in shp_insts if os.path.isdir(os.path.join(dataset_root_folder, fn))]
  
  mesh_dicts = []
  part_name_to_mesh_idx = {}
  for shp_inst in shp_insts:
    cur_shp_folder = os.path.join(dataset_root_folder, shp_inst)
    cur_shp_voxs = os.listdir(cur_shp_folder)
    cur_shp_voxs = [fn for fn in cur_shp_voxs if fn.endswith(".obj")]

    if split is not None:
      if split == 'training':
        if len(cur_shp_voxs) <= 2: # number of samples and voxels
          cur_split_nn_samples = min(1, len(cur_shp_voxs))
        else:
          cur_split_nn_samples = int(len(cur_shp_voxs) * 0.8)
        cur_shp_voxs = cur_shp_voxs[: cur_split_nn_samples]
      else:
        if len(cur_shp_voxs) <= 2:
          cur_split_nn_samples = min(1, len(cur_shp_voxs))
        else:
          cur_split_nn_samples = int(len(cur_shp_voxs) * 0.8)
        cur_shp_voxs = cur_shp_voxs[cur_split_nn_samples: ]

    for vox_fn in cur_shp_voxs:
      part_nm = vox_fn.split(".")[0] # xxx.obj.binvox
      if part_names is not None and part_nm not in part_names:
        continue
      cur_vox_fn = os.path.join(cur_shp_folder, vox_fn)
      cur_sample_vertices, cur_sample_faces = data_utils.read_obj(cur_vox_fn)
      if cur_sample_vertices.shape[0] <= 0: # no vertices in the obj file
        print(f"No vertices in file {cur_vox_fn}, skipping...")
        continue
      try:
        cur_sample_mesh_dict = data_utils.process_mesh(cur_sample_vertices, cur_sample_faces, quantization_bits=quantization_bits, recenter_mesh=True, remove_du=remove_du)
      except:
        print(f"Something bad happended when processing file {cur_vox_fn}... Skipping...")
        continue

      vertices = cur_sample_mesh_dict['vertices'] # 
      # print(f"vertices size: {vertices.shape}")

      if not (np.max(vertices).item() < 2 ** quantization_bits and np.min(vertices).item() >= 0):
        continue
      
      nn_vertices = vertices.shape[0]
      # nn_faces = faces.shape[0]
      if not (nn_vertices <= nn_max_permite_vertices): 
        continue

      mesh_dict = {}
      mesh_dict['vertices'] = vertices
      mesh_dict['class_label'] = 0
      if part_nm not in part_name_to_mesh_idx:
        part_name_to_mesh_idx[part_nm] = [len(mesh_dicts)]
      else:
        part_name_to_mesh_idx[part_nm].append(len(mesh_dicts))
      mesh_dicts.append(mesh_dict)
  
  part_tree = {'idx': 0}

  #### part name to mesh idx ####
  if not ret_part_name_to_mesh_idx:
    return mesh_dicts, part_tree
  else:
    return mesh_dicts, part_tree, part_name_to_mesh_idx



# mesh dict multi part
def get_mesh_dict_list_multi_part_part_first_pc_cond(dataset_root_folder, part_names, valid_indices=None, category_name="eyeglasses"):
  # max permit vertices
  nn_max_permite_vertices = opt.dataset.max_permit_vertices
  nn_max_permite_faces = opt.dataset.max_permit_faces
  debug = opt.model.debug

  # n_mask_samples = opt.model.n_mask_samples # eyeglasses name
  part_tree = {'idx': 0}

  # summary_obj_fn = "summary.obj"
  quantization_bits = opt.model.quantization_bits
  recenter_mesh = opt.model.recenter_mesh
  # tree_traverse, child_to_parent = data_utils.mid_traverse_tree(part_tree)

  mesh_dicts = []

  if part_names is None:
    part_names = os.listdir(dataset_root_folder)
    part_names = [fnn for fnn in part_names if os.path.isdir(os.path.join(dataset_root_folder, fnn))]

  samples_fn = os.path.join(dataset_root_folder, part_names[0])
  samples_list = os.listdir(samples_fn)

  samples_list = [fn for fn in samples_list if fn.endswith(".obj")]

  n_max_samples = 1000 if not debug else 10
  # n_valid_samples = 0

  print(f"Loading data... Category name: {category_name}, part_names: {part_names}")

  for part_nm in part_names:
    cur_part_objs_folder = os.path.join(dataset_root_folder, part_nm)
    samples_list = os.listdir(cur_part_objs_folder)

    samples_list = [fn for fn in samples_list if fn.endswith(".obj")]

    

    for i_s, sample_nm in enumerate(samples_list):

      # scale_idx = sample_nm.split('.')[0].split('_')[-1]
      # scale_idx = int(scale_idx)
      # if scale_idx > 100:
      #   continue
      shp_idx = sample_nm.split("_")[0] # shape idx
      shp_idx = int(shp_idx)
      # if shp_idx not in [1, 2, 5, 7, 9, 10, 12, 11]:
      #   continue
      if valid_indices is not None:
        valid = False
        for valid_indi in valid_indices:
          if valid_indi in sample_nm:
            valid = True
            break
        if not valid:
          continue
      
      vertices_list, faces_list = [], []

      # for part_nm in part_names:
      cur_sample_cur_part_fn = os.path.join(dataset_root_folder, part_nm, sample_nm) # sample_name
      # print(cur_sample_cur_part_fn)
      cur_sample_vertices, cur_sample_faces = data_utils.read_obj(cur_sample_cur_part_fn)
      # print("cc")
      # print(f"cur_sample_vertices: {cur_sample_vertices.shape}, cur_sample_faces: {len(cur_sample_faces)}")
      if category_name in ["eyeglasses"] and part_nm != "none_motion":
        cur_sample_vertices = cur_sample_vertices[:, [1, 0, 2]]
      vertices_list.append(cur_sample_vertices)
      faces_list.append(cur_sample_faces)


      cur_sample_vertices, cur_sample_faces = data_utils.merge_meshes(vertices_list, faces_list)
      nn_face_indices = sum([len(sub_face) for sub_face in cur_sample_faces])

      # print(f"n_vertices: {cur_sample_vertices.shape[0]}, nn_face_indices: {nn_face_indices}")
      
      # if not (cur_sample_vertices.shape[0] <= nn_max_permite_vertices and nn_face_indices <= nn_max_permite_faces):
      #   continue
      
      ### centralize vertice  s ###
      ins_vert_min = cur_sample_vertices.min(axis=0) # vert_min
      ins_vert_max = cur_sample_vertices.max(axis=0) # vert_max
      ins_vert_center = 0.5 * (ins_vert_min + ins_vert_max)
      cur_sample_vertices = cur_sample_vertices - ins_vert_center
      ins_vert_min = cur_sample_vertices.min(axis=0) # vert_min
      ins_vert_max = cur_sample_vertices.max(axis=0) # vert_max
      ins_extents = ins_vert_max - ins_vert_min # extents
      ins_scale = np.sqrt(np.sum(ins_extents**2)) # scale # 
      cur_sample_vertices = cur_sample_vertices / ins_scale

      # sample
      cur_sample_vertices_permuted = cur_sample_vertices[:, [2, 0, 1]] # sampled vertices
      cur_sample_sampled_pts = data_utils.sample_pts_from_mesh(cur_sample_vertices_permuted, cur_sample_faces, npoints=512)

      cur_sample_mesh_dict = data_utils.process_mesh(cur_sample_vertices, cur_sample_faces, quantization_bits=quantization_bits, recenter_mesh=True)
      mesh_dict = cur_sample_mesh_dict
      mesh_dict['class_label'] = 0

      mesh_dict = {}
      mesh_dict = cur_sample_mesh_dict
      mesh_dict['class_label'] = 0
      mesh_dict['points'] = cur_sample_sampled_pts
      vertices, faces = mesh_dict['vertices'], mesh_dict['faces']
      if not (np.max(vertices).item() < 2 ** quantization_bits and np.min(vertices).item() >= 0):
        continue
      

      nn_vertices = vertices.shape[0]
      nn_faces = faces.shape[0]
      if not (nn_vertices <= nn_max_permite_vertices and nn_faces <= nn_max_permite_faces): # load and test whether they...
        continue

      mesh_dicts.append(mesh_dict)

      if len(mesh_dicts) >= n_max_samples:
        break

  return mesh_dicts, part_tree


# mesh dict multi part
def get_mesh_dict_list_multi_part_part_first_paired_finetuning(dataset_root_folder, part_names, valid_indices=None, category_name="eyeglasses"):
  # max permit vertices
  nn_max_permite_vertices = opt.dataset.max_permit_vertices
  nn_max_permite_faces = opt.dataset.max_permit_faces

  # n_mask_samples = opt.model.n_mask_samples
  part_tree = {'idx': 0}

  # summary_obj_fn = "summary.obj"
  quantization_bits = opt.model.quantization_bits
  recenter_mesh = opt.model.recenter_mesh
  # tree_traverse, child_to_parent = data_utils.mid_traverse_tree(part_tree)

  mesh_dicts = []

  if part_names is None:
    part_names = os.listdir(dataset_root_folder)
    part_names = [fnn for fnn in part_names if os.path.isdir(os.path.join(dataset_root_folder, fnn))]

  # samples_fn = os.path.join(dataset_root_folder, part_names[0])
  # samples_list = os.listdir(samples_fn)

  # samples_list = [fn for fn in samples_list if os.path.isdir(os.path.join(dataset_root_folder, part_nm, fn))]

  n_max_samples = 1000
  # n_valid_samples = 0
  backbone_obj_fn = "backbone.obj"

  print(f"Loading data... Category name: {category_name}, part_names: {part_names}")
  # but we have no part now...
  part_names = [part_names[0]]

  for part_nm in part_names:
    # cur_part_objs_folder = os.path.join(dataset_root_folder, part_nm)
    cur_part_objs_folder = dataset_root_folder # root_folder, 
    samples_list = os.listdir(cur_part_objs_folder)
    samples_list = [fn for fn in samples_list if os.path.isdir(os.path.join(cur_part_objs_folder, fn))]

    for i_s, sample_nm in enumerate(samples_list):
      # TODO recover such information?
      # scale_idx = sample_nm.split('.')[0].split('_')[-1]
      # scale_idx = int(scale_idx)
      # if scale_idx > 100:
      #   continue
      # shp_idx = sample_nm.split("_")[0]
      # shp_idx = int(shp_idx)
      # if shp_idx not in [1, 2, 5, 7, 9, 10, 12, 11]:
      #   continue
      # if valid_indices is not None:
      #   valid = False
      #   for valid_indi in valid_indices:
      #     if valid_indi in sample_nm:
      #       valid = True
      #       break
      #   if not valid:
      #     continue
      
      # vertices_list, faces_list = [], []


      # for part_nm in part_names:
      # cur_sample_cur_part_folder = os.path.join(dataset_root_folder, part_nm, sample_nm) # sample_name
      cur_sample_cur_part_folder = os.path.join(cur_part_objs_folder, sample_nm)
      # print(cur_sample_cur_part_fn)
      
      # backbone file name
      cur_sample_cur_part_backbone_fn = os.path.join(cur_sample_cur_part_folder, backbone_obj_fn)
      # for other objs' names
      tot_obj_fns = os.listdir(cur_sample_cur_part_folder)
      cur_sample_cur_part_details_fn = [fn for fn in tot_obj_fns if fn != backbone_obj_fn]
      
      cur_backbone_vertices, cur_backbone_faces = data_utils.read_obj(cur_sample_cur_part_backbone_fn)
      if category_name in ["eyeglasses"] and part_nm != "none_motion":
        cur_sample_vertices = cur_sample_vertices[:, [1, 0, 2]]

      nn_backbone_verts = cur_backbone_vertices.shape[0]

      for i_de_sample, cur_de_fn in enumerate(cur_sample_cur_part_details_fn):
        tot_cur_de_fn = os.path.join(cur_sample_cur_part_folder, cur_de_fn)
        cur_de_vertices, cur_de_faces = data_utils.read_obj(tot_cur_de_fn)
        
        cur_sample_mesh_dict = data_utils.process_mesh_list([cur_backbone_vertices, cur_de_vertices], [cur_backbone_faces, cur_de_faces], quantization_bits=quantization_bits, recenter_mesh=True)
        mesh_dict = cur_sample_mesh_dict
        mesh_dict['class_label'] = 0
        mesh_dict['nn_backbone_verts'] = nn_backbone_verts
      
      # mesh_dict = {}
      # mesh_dict = cur_sample_mesh_dict
      # mesh_dict['class_label'] = 0
        vertices, faces = mesh_dict['vertices'], mesh_dict['faces']

        nn_vertices = vertices.shape[0]
        nn_faces = faces.shape[0]
        if not (nn_vertices <= nn_max_permite_vertices and nn_faces <= nn_max_permite_faces): # load and test whether they...
          continue

        mesh_dicts.append(mesh_dict)

        if len(mesh_dicts) >= n_max_samples:
          break

  return mesh_dicts, part_tree

