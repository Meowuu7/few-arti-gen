import bpy
import os
import numpy as np


# from mathutils import Vector

def delete_all():
  for object in bpy.data.objects:
    object.select = True
  bpy.ops.object.delete()
# decimate ratio
r = 0.5
delete_all()


obj_fns_list = ["/Users/meow/Study/_2021_autumn/gen/projects/polygen/polygen/lamp-n-shot-4/train_ep_0_iter_0_bsz_0_i_s_7_src_merged_def_mesh.obj", "/Users/meow/Study/_2021_autumn/gen/projects/polygen/polygen/lamp-n-shot-4/train_ep_0_iter_0_bsz_0_i_s_6_src_merged_def_mesh.obj"] 
dx, dy = 0.2
for i, obj_fn in enumerate(obj_fns_list):
  imported_object = bpy.ops.import_scene.obj(filepath = obj_fn, )
  imported_object.location = [dx * i, dy * i, 0.]
  
  