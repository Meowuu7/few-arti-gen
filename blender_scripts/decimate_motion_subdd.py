import bpy
import os
import numpy as np

def delete_all():
  for object in bpy.data.objects:
    object.select = True
  bpy.ops.object.delete()
# decimate ratio
r = 0.5
delete_all()

# part_name = "none_motion"
# part_name = "04379243"
# part_name = "dof_rootd_Aa001_r"
# destination_path = "/home/xueyi/gen/polygen_2/data/MotionDataset/eyeglasses_remerge_rnd_deci_3_iter_1"
# destination_path = "/home/xueyi/gen/polygen_2/data/MotionDataset/eyeglasses_remerge_rnd_deci_3_tst"
# destination_path = "/data/datasets/ShapeNet_processed_deci"
# objs_path = "/data/datasets/ShapeNet_processed"
# os.makedirs(destination_path, exist_ok=True)
# destination_path = os.path.join(destination_path, part_name)
# os.makedirs(destination_path, exist_ok=True)
# objs_path = os.path.join(objs_path, part_name)

# motion_dataset_root = "/nas/datasets/gen/datasets/MotionDataset_processed"
motion_dataset_root = "/nas/datasets/gen/datasets/subd_mesh"
# motion_dataset_root = "/nas/datasets/gen/datasets/MotionDataset_processed_deci_part_first_30_normalized"
# motion_cat = "eyeglasses"
# tot_motion_cats = os.listdir(motion_dataset_root)
# tot_motion_cats = [fnn for fnn in tot_motion_cats if os.path.isdir(os.path.join(motion_dataset_root, fnn)) and fnn != "eyeglasses"]
# tot_motion_cats = [fnn for fnn in tot_motion_cats if os.path.isdir(os.path.join(motion_dataset_root, fnn))]
# tot_motion_cats = ["lamp", "screwdriver", "seesaw", "pen"]

motion_cat = "screwdriver"
motion_cat = "screw_1"

part_name = "dof_rootd_Aa001_r"


decimate_stra = "planar"
# decimate_stra = "collapse"

collapse_ratio = [0.1, 0.25, 0.5]
subd_folder_nm = ["subd0", "subd1", "subd2"]

# sv_subd_root_folder = os.path.join("/home/xueyi", "gen", "neuralSubdiv", "data_meshes", "{}-{}-valid".format(motion_cat, part_name))

# os.makedirs(sv_subd_root_folder, exist_ok=True)

sv_subd_root_folder = os.path.join(motion_dataset_root, motion_cat + "_planar")
os.makedirs(sv_subd_root_folder, exist_ok=True)



# decimate_stra = "planar"
# decimate_stra = "collapse"
fr_motion_folder = os.path.join(motion_dataset_root, motion_cat)
inst_nms = os.listdir(fr_motion_folder)
inst_nms = [fn for fn in inst_nms if os.path.isdir(os.path.join(fr_motion_folder, fn))]



n_samples = 1

n_samples_max = 10


# for collapse_rt, subd_folder in zip(collapse_ratio, subd_folder_nm):
#   cur_subd_sv_folder = os.path.join(sv_subd_root_folder, subd_folder)
#   os.makedirs(cur_subd_sv_folder, exist_ok=True)

tot_n_samples = 0

for inst_nm in inst_nms:
  
  cur_inst_folder = os.path.join(fr_motion_folder, inst_nm)
  cur_inst_sv_folder = os.path.join(sv_subd_root_folder, inst_nm)
  os.makedirs(cur_inst_sv_folder, exist_ok=True)

  print("Processing current inst folder:", cur_inst_folder, "; sv_folder:", cur_inst_sv_folder)
  
  cur_objs = os.listdir(cur_inst_folder)
  cur_objs = [fn for fn in cur_objs if fn.endswith(".obj")]
  
  for cur_obj in cur_objs:
    cur_obj_fn = os.path.join(cur_inst_folder, cur_obj)
    imported_object = bpy.ops.import_scene.obj(filepath=cur_obj_fn)

    for object in bpy.data.objects:
      bpy.context.scene.objects.active = object
      bpy.ops.object.modifier_add(type="DECIMATE")

      # cur_rnd_angle = np.random.randint(low=5, high=50, size=(1,))
      if decimate_stra == "planar":
        bpy.context.object.modifiers["Decimate"].decimate_type = 'DISSOLVE'
        # cur_rnd_angle = np.random.randint(low=5, high=25, size=(1,))
        # cur_rnd_angle = float(cur_rnd_angle.item())
        # cur_rnd_angle = 15. if n_samples == 1 else cur_rnd_angle
        cur_rnd_angle = float(5.) # float(angle_limit) if n_samples == 1 else cur_rnd_angle
        rand_angle_limit = float(cur_rnd_angle / 180. * np.pi)
        bpy.context.object.modifiers["Decimate"].angle_limit=rand_angle_limit
      else:
        bpy.context.object.modifiers["Decimate"].decimate_type = 'COLLAPSE'
        rand_ratio = collapse_rt
        bpy.context.object.modifiers["Decimate"].ratio = collapse_rt # decimate withangels 
      #### decimate by specifying the angle limit ####
    
    cur_sample_obj_sv_fn = os.path.join(cur_inst_sv_folder, cur_obj)
    tot_n_samples += 1
    
    # cur_sample_obj_sv_fn = os.path.join(cur_sample_sv_folder, cur_sample_obj)
    bpy.ops.export_scene.obj(filepath = cur_sample_obj_sv_fn, use_selection = False)
    delete_all()
    # if tot_n_samples >= n_samples_max:
    #   break
