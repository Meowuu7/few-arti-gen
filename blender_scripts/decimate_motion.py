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

motion_dataset_root = "/nas/datasets/gen/datasets/MotionDataset_processed"
motion_dataset_root = "/nas/datasets/gen/datasets/MotionDataset_processed_deci_part_first_30_normalized"
# motion_cat = "eyeglasses"
tot_motion_cats = os.listdir(motion_dataset_root)
# tot_motion_cats = [fnn for fnn in tot_motion_cats if os.path.isdir(os.path.join(motion_dataset_root, fnn)) and fnn != "eyeglasses"]
tot_motion_cats = [fnn for fnn in tot_motion_cats if os.path.isdir(os.path.join(motion_dataset_root, fnn))]
# tot_motion_cats = ["lamp", "screwdriver", "seesaw", "pen"]


decimate_stra = "planar"
decimate_stra = "collapse"

collapse_ratio = 0.2

angle_limit = 15
angle_limit = 30
angle_limit = 50

if decimate_stra == "planar":
  # motion dataset save root
  motion_dataset_sv_root = "/nas/datasets/gen/datasets/MotionDataset_processed_deci_" + str(angle_limit)
else:
  motion_dataset_sv_root = "/nas/datasets/gen/datasets/MotionDataset_processed_deci_" + str(collapse_ratio)
os.makedirs(motion_dataset_sv_root, exist_ok=True)
tot_motion_cats_sv = os.listdir(motion_dataset_sv_root)
tot_motion_cats_sv = [fn for fn in tot_motion_cats_sv if os.path.isdir(os.path.join(motion_dataset_sv_root, fn))]

# #### decimate motions #### #
tot_motion_cats = [fn for fn in tot_motion_cats if fn not in tot_motion_cats_sv]


# decimate_stra = "planar"
# decimate_stra = "collapse"

for motion_cat in tot_motion_cats:
  # motion_dataset_root = "/data/datasets/MotionDataset_processed"
  motion_category_folder = os.path.join(motion_dataset_root, motion_cat)

  filelist = os.listdir(motion_category_folder)
  filelist = [fn for fn in filelist if os.path.isdir(os.path.join(motion_category_folder, fn))]

  
  motion_category_sv_folder = os.path.join(motion_dataset_sv_root, motion_cat)
  os.makedirs(motion_category_sv_folder, exist_ok=True)


  n_samples = 1

  # filelist = os.listdir(objs_path)
  # filelist = [fn for fn in filelist if os.path.isdir(os.path.join(objs_path, fn))]

  for file in filelist: # file in the 
    cur_sample_folder = os.path.join(motion_category_folder, file)
    cur_sample_obj_fns = os.listdir(cur_sample_folder)
    cur_sample_obj_fns = [cur_fn for cur_fn in cur_sample_obj_fns if cur_fn.endswith(".obj")]

    cur_sample_sv_folder = os.path.join(motion_category_sv_folder, file)
    os.makedirs(cur_sample_sv_folder, exist_ok=True)
    for cur_sample_obj in cur_sample_obj_fns:
      tot_cur_sample_obj_fn = os.path.join(cur_sample_folder, cur_sample_obj)
      # import objs from obj files
      imported_object = bpy.ops.import_scene.obj(filepath = tot_cur_sample_obj_fn)

      for object in bpy.data.objects:
        bpy.context.scene.objects.active = object
        bpy.ops.object.modifier_add(type="DECIMATE")

        
        # cur_rnd_angle = np.random.randint(low=5, high=50, size=(1,))
        if decimate_stra == "planar":
          bpy.context.object.modifiers["Decimate"].decimate_type = 'DISSOLVE'
          cur_rnd_angle = np.random.randint(low=5, high=25, size=(1,))
          cur_rnd_angle = float(cur_rnd_angle.item())
          # cur_rnd_angle = 15. if n_samples == 1 else cur_rnd_angle
          cur_rnd_angle = float(angle_limit) if n_samples == 1 else cur_rnd_angle
          rand_angle_limit = float(cur_rnd_angle / 180. * np.pi)
          bpy.context.object.modifiers["Decimate"].angle_limit=rand_angle_limit
        else:
          bpy.context.object.modifiers["Decimate"].decimate_type = 'COLLAPSE'
          rand_ratio = collapse_ratio
          bpy.context.object.modifiers["Decimate"].ratio = rand_ratio # decimate withangels 
        #### decimate by specifying the angle limit ####
      
      cur_sample_obj_sv_fn = os.path.join(cur_sample_sv_folder, cur_sample_obj)
      bpy.ops.export_scene.obj(filepath = cur_sample_obj_sv_fn, use_selection = False)
      delete_all()


