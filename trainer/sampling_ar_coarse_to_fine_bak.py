# from glob import glob
from inspect import getmodule
import os
from unicodedata import category
# import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import os.path
import json
import numpy as np
import math
import sys
import torch

from torch.utils import data
import utils.data_utils_torch as data_utils
from utils.trainer_utils import *
# from datasets.urdf_dataset_pretraining_xl import URDFDataset as URDFDataset_xl
# from datasets.urdf_dataset_pretraining_xl import URDFDataset as URDFDataset_xl
from datasets.urdf_dataset_pretraining_ar_coarse_to_fine import URDFDataset as URDFDataset_ar
# from datasets.urdf_dataset_pretraining_cus_perm_xl import URDFDataset as URDFDataset_cus_perm_xl
# from datasets.urdf_dataset_pretraining_cus_perm_xl_2 import URDFDataset as URDFDataset_cus_perm_xl
from modules.modules_pretraining_ar_coarse_to_fine import VertexModel, FaceModel
# from modules.modules_pretraining_xl import VertexModel, FaceModel
# from modules.modules_pretraining_xl_2 import VertexModel, FaceModel


from tqdm import tqdm
import time
import torch.optim as optim
from options.options import opt
from utils.constants import *


epoch_counter = 0
test_epoch_counter = 0
# use_cuda = True
step_counter = 0




def step_sample(test_dataset, test_dataset_iter, cur_step=0):
    global test_epoch_counter # 
    try:
        data = next(test_dataset_iter)
        if data['class_label'].size(0) < batch_size:
            raise StopIteration
    except StopIteration:
        test_epoch_counter += 1
        # print("[Test DataLoader]: At Epoch %d!" % test_epoch_counter)
        # reset dataset iterator
        test_dataset_iter = iter(test_dataset)
        # get data for the next iteration
        data = next(test_dataset_iter)
    sample(data, cur_step=cur_step)
    return test_dataset_iter


def sample(test_data, cur_step=0):
    # global nn_max_vertices
    # global nn_max_faces
    
    ##### Get vertices and faces here #####
    nn_max_vertices = opt.vertex_model.max_vertices
    nn_max_faces = opt.face_model.max_faces
    bsz = opt.loss.batch_size
    bsz = min(bsz, test_data['coarse']['vertices'].size(0))
    ##### Get vertices and faces here #####

    vertex_model.eval()
    face_model.eval()

    fine_vertex_model.eval()
    fine_face_model.eval()

    with torch.no_grad():
        
        coarse_test_data, fine_test_data = test_data['coarse'], test_data['fine']
        for k in coarse_test_data:
          coarse_test_data[k] = coarse_test_data[k].cuda(non_blocking=True)
        for k in fine_test_data:
          fine_test_data[k] = fine_test_data[k].cuda(non_blocking=True)


        # coarse_vertex_model_pred_dict = vertex_model(coarse_test_data)
        coarse_vertex_model_pred_dict, global_feature = vertex_model(coarse_test_data, return_global_feature=True)
        print(f"here vertex model coarse data: {global_feature.size()}")
        coarse_vertices_global_feature = global_feature
        fine_test_data['global_context'] = coarse_vertices_global_feature


        vertex_sample_dict = fine_vertex_model.sample(bsz, context=fine_test_data, max_sample_length=nn_max_vertices, temperature=1., top_k=0, top_p=1.0, recenter_verts=True, only_return_complete=False)

        # class_label
        # recenter_verts
        # vertex_sample_dict['class_label'] = sample_context['class_label'] # class labels

        # perform face sampling # 
        face_sample_dict = fine_face_model.sample(vertex_sample_dict, max_sample_length=nn_max_faces, temperature=1., top_k=0,  top_p=1.0,  only_return_complete=False)

        print("Face sampling finished!")

        ### convert torch tensors to numpy values ####
        for k in vertex_sample_dict: # sample dict
            if isinstance(vertex_sample_dict[k], dict):
                for sub_k in vertex_sample_dict[k]:
                    vertex_sample_dict[k][sub_k] = vertex_sample_dict[k][sub_k].detach().cpu().numpy()
            else:
                vertex_sample_dict[k] = vertex_sample_dict[k].detach().cpu().numpy()
        for k in face_sample_dict:
            if isinstance(face_sample_dict[k], dict):
                for sub_k in face_sample_dict[k]:
                    face_sample_dict[k][sub_k] = face_sample_dict[k][sub_k].detach().cpu().numpy()
            else:
                face_sample_dict[k] = face_sample_dict[k].detach().cpu().numpy()
        test_data_np = {} # numpys
        for k in coarse_test_data:
            if isinstance(coarse_test_data[k], dict):
                test_data_np[k] = {}
                for sub_k in coarse_test_data[k]:
                    test_data_np[k][sub_k] = coarse_test_data[k][sub_k].detach().cpu().numpy()
            else:
                test_data_np[k] = coarse_test_data[k].detach().cpu().numpy()
        print("Ploting sampled meshes...")

        ### plot sampled meshes to the obj file ### # 
        # plot sampled
        # try:
        # data_utils.plot_sampled_meshes(v_sample=vertex_sample_dict, f_sample=face_sample_dict, sv_mesh_folder="./samples/eyeglasses/meshes/", cur_step=cur_step, predict_joint=predict_joint) # predict joint variable... 
        data_utils.plot_sampled_meshes_single_part_for_pretraining(v_sample=vertex_sample_dict, f_sample=face_sample_dict, context=test_data_np, sv_mesh_folder=sv_mesh_folder, cur_step=cur_step, predict_joint=False) # predict joint variable...
        # except:
        #     pass


# train iter
def test_iter(test_dataset, training_steps):
    # dataset_iter
    # iter dataset
    # dataset_iter = iter(test_dataset)
    test_dataset_iter = iter(test_dataset)
    for i in range(training_steps): # training_steps;  # test datas
        test_dataset_iter = step_sample(test_dataset, test_dataset_iter, cur_step=i) # 



# sometiems more easier than eyeglasses?
def get_model_descriptor():
    motion_dataset_nm = opt.dataset.dataset_name
    # motion_dataset_nm
    # motion_cat = "eyeglasses"
    motion_cat = opt.dataset.category
    batch_size = opt.loss.batch_size
    nn_max_vertices = opt.vertex_model.max_vertices
    nn_max_faces = opt.face_model.max_faces
    nn_max_permite_vertices = opt.dataset.max_permit_vertices
    nn_max_permite_faces = opt.dataset.max_permit_faces
    # mask_low_ratio = opt.dataset.mask_low_ratio
    # mask_high_ratio = opt.dataset.mask_high_ratio
    cus_perm_xl = opt.dataset.cus_perm_xl # whether to use the customized permutation for the permutation based auto-regressive model
    cross = opt.model.cross
    exp_flag = opt.common.exp_flag
    exp_mode = opt.common.exp_mode
    apply_random_shift = opt.dataset.apply_random_shift
    apply_random_flipping = opt.dataset.apply_random_flipping
    category_part_cat = opt.dataset.category_part_indicator
    use_light_weight = opt.model.use_light_weight
    model_descriptor = f"{exp_mode}_ar_ctof_{exp_flag}_cat_part_indi_{category_part_cat}light_{use_light_weight}_shift_{apply_random_shift}_filp_{apply_random_flipping}_{motion_dataset_nm}_bsz_{batch_size}_max_verts_{nn_max_vertices}_faces_{nn_max_faces}_max_permit_verts_{nn_max_permite_vertices}_faces_{nn_max_permite_faces}"
    return model_descriptor


# resume from
def resume_from_ckpts():
    vertex_model_state_dicts = torch.load(vertex_model_path, map_location="cpu")
    face_model_state_dicts = torch.load(face_model_path, map_location="cpu")

    fine_vertex_model_state_dicts = torch.load(fine_vertex_model_path, map_location="cpu")
    fine_face_model_state_dicts = torch.load(fine_face_model_path, map_location="cpu")

    vertex_model.load_state_dict(vertex_model_state_dicts)
    face_model.load_state_dict(face_model_state_dicts)
    fine_vertex_model.load_state_dict(fine_vertex_model_state_dicts)
    fine_face_model.load_state_dict(fine_face_model_state_dicts)
    print(f"Vertex model loaded from {vertex_model_path}.")
    print(f"Face model loaded from {face_model_path}.")
    print(f"Fine vertex model loaded from {fine_vertex_model_path}.")
    print(f"Fine face model loaded from {fine_face_model_path}.")


        
if __name__=='__main__':

    # motion_dataset_nm = "MotionDataset"
    motion_dataset_nm = opt.dataset.dataset_name
    # motion_dataset_nm
    # motion_cat = "eyeglasses"
    motion_cat = opt.dataset.category

    dataset_root_path = opt.dataset.dataset_root_path # root path
    debug = opt.model.debug

    print(f"Debug: {debug}")
    
    if THU_LAB_FLAG in dataset_root_path:
        opt.dataset.lab = "thu"
    else:
        opt.dataset.lab = "meg"
    
    # root_folder for the dataset
    root_folder = os.path.join(dataset_root_path, motion_dataset_nm, motion_cat)
    
    # process recenter mesh # recenter mesh
    # process_recenter_mesh = True

    recenter_mesh = opt.model.recenter_mesh
    process_recenter_mesh = opt.model.recenter_mesh

    # quantization_bits = 8
    quantization_bits = opt.model.quantization_bits

    batch_size = opt.loss.batch_size
    num_thread = 10
    plot_sample_step = 2000

    # nn_max_permite_vertices = 400
    # nn_max_permite_faces = 2000

    nn_max_permite_vertices = opt.dataset.max_permit_vertices
    nn_max_permite_faces = opt.dataset.max_permit_faces

    # nn_max_vertices = 400
    # nn_max_faces = 2000
    nn_max_vertices = opt.vertex_model.max_vertices
    nn_max_faces = opt.face_model.max_faces

    cus_perm_xl = opt.dataset.cus_perm_xl
    corss = opt.model.cross

    model_descriptor = get_model_descriptor()

    if motion_cat == "eyeglasses":
      #### for eyegalsse ####
      # train_ins_nns = ["%.4d" % iii for iii in [3, 8, 23, 13, 6]]
    #   train_ins_nns = [ "%.4d" % iii for iii in [3, 9, 4, 21]]
      train_ins_nns = None
      # train_ins_nns = ["%.4d" % iii for iii in [1, 2]]
      # test_ins_nns = ["%.4d" % iii for iii in [1, 2]]
      test_ins_nns = None
      # train_part_names = ["dof_rootd_Aa001_r", "dof_rootd_Aa002_r"]
      # test_part_names = ["none_motion"]
      #### for eyeglasses ####
    else:
      #### for polygen samples ###
      train_ins_nns = [str(iii) for iii in range(3000)]
      test_ins_nns = [str(iii) for iii in range(3000, 4001)]
      # train_part_names = None
      # test_part_names = None

    # if not cus_perm_xl:
    URDFDataset = URDFDataset_ar
    
    debug = opt.model.debug

    if not debug:
        # categories for the training set...
    #   cur_dataset_root_path = os.path.join(opt.dataset.dataset_root_path, opt.dataset.dataset_name)
    #   train_categories = os.listdir(cur_dataset_root_path)
    #   train_categories = [fn for fn in train_categories if os.path.isdir(os.path.join(cur_dataset_root_path, fn))]
    #   category_name = train_categories
    #   category_names_to_part_names = {cat: None for cat in category_name}
    #   category_names = category_name

    # #### for train dataset ####
    #   category_name = ["eyeglasses"]
    #   category_names_to_part_names = {
    #     "eyeglasses": ["dof_rootd_Aa001_r", "dof_rootd_Aa002_r"]
    #   }
    #   category_names = list(category_names_to_part_names.keys())
    # #### for train dataset ####

      # category_name = ["globe", "screwdriver"]
      # category_names_to_part_names = {cat: None for cat in category_name}
      # category_names = category_name

    #   category_name = ["globe", "screwdriver"]
    #   category_names_to_part_names = {cat: None for cat in category_name}
    #   category_names = category_name
    
      category_name = ["screwdriver", "eyeglasses",  "pen", "scissors", "stapler", "closestool",  "globe"]
      category_names_to_part_names = {cat: None for cat in category_name}
      category_names_to_part_names["eyeglasses"] = ["dof_rootd_Aa001_r", "dof_rootd_Aa002_r"]
      category_names = category_name
      

      # #### for train dataset ####
      # category_name = ["eyeglasses", ]
      # category_names_to_part_names = {
      #     "eyeglasses": ["dof_rootd_Aa001_r", "dof_rootd_Aa002_r"], 
      #     # "pen": None, "scissors": None, 
      # }
      # category_names = list(category_names_to_part_names.keys())
      # #### for train dataset ####
    else:
       #### for train dataset ####
      category_name = ["eyeglasses", "pen", "scissors" ]
      category_names_to_part_names = {
          "eyeglasses": ["none_motion"], "pen": None, "scissors": None, 
      }
      category_names = list(category_names_to_part_names.keys())
      #### for train dataset ####

    print(f"motion cat: {motion_cat}")

    # dataset_train = URDFDataset(root_folder=root_folder, quantization_bits=quantization_bits, nn_max_vertices=nn_max_vertices, nn_max_faces=nn_max_faces, nn_max_permite_vertices=nn_max_permite_vertices, nn_max_permite_faces=nn_max_permite_faces, category_name=category_names, instance_nns=train_ins_nns, part_names=None, category_nm_to_part_nm=category_names_to_part_names, is_training=True)

    # dataset_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=num_thread)

    # if corss:
    #     print(f"Cross part zero-shot generation.")
    #     opt.dataset.dataset_name = "MotionDataset_processed_deci_scaled"
    #     opt.dataset.category = "eyeglasses"
    #     motion_cat = "eyeglasses"
    #     test_ins_nns = ["%.4d" % iii for iii in [1, 2]]
    #     root_folder = os.path.join(dataset_root_path, opt.dataset.dataset_name, opt.dataset.category)

    #### for train dataset ####
    category_name = ["eyeglasses"]
    category_names_to_part_names = {
        "eyeglasses": ["none_motion"]
    }
    category_names = list(category_names_to_part_names.keys())
    #### for train dataset ####

    category_name = ["screwdriver", "eyeglasses",  "pen", "scissors", "stapler", "closestool",  "globe"]
    category_names_to_part_names = {cat: None for cat in category_name}
    # category_names_to_part_names["eyeglasses"] = ["dof_rootd_Aa001_r", "dof_rootd_Aa002_r"]
    category_names = category_name

    # #### for train dataset ####
    # category_name = "eyeglasses"
    # # category_names_to_part_names = {
    # #     "eyeglasses": ["none_motion"]
    # # }
    # # category_names = list(category_names_to_part_names.keys())
    # motion_cat = category_name
    # root_folder = os.path.join(dataset_root_path, opt.dataset.dataset_name, opt.dataset.category)
    # #### for train dataset ####

    dataset_test = URDFDataset(root_folder=root_folder, quantization_bits=quantization_bits, nn_max_vertices=nn_max_vertices, nn_max_faces=nn_max_faces, nn_max_permite_vertices=nn_max_permite_vertices, nn_max_permite_faces=nn_max_permite_faces, category_name=category_name, instance_nns=test_ins_nns, part_names=None, category_nm_to_part_nm=category_names_to_part_names, is_training=False)

    # shuffle = False
    # dataet for train...
    dataset_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_thread)


    ##### Model configs #####
    vertex_module_config, face_module_config = get_model_configs()
    vertex_model = VertexModel(**vertex_module_config)
    face_model = FaceModel(**face_module_config)

    fine_vertex_model = VertexModel(**vertex_module_config)
    fine_face_model = FaceModel(**face_module_config)

    vertex_model = vertex_model.cuda()
    face_model = face_model.cuda()

    fine_vertex_model = fine_vertex_model.cuda()
    fine_face_model = fine_face_model.cuda()

    vertex_model_path = opt.model.vertex_model_path
    face_model_path = opt.model.face_model_path
    fine_vertex_model_path = opt.model.fine_vertex_model_path
    fine_face_model_path = opt.model.fine_face_model_path

    resume_from_ckpts()

    print("Dataset constructed!")

    # Optimization settings
    # learning_rate = 5e-4
    # training_steps = 2000 * 500
    # check_step = 5

    learning_rate = opt.loss.learning_rate
    training_steps = opt.loss.training_steps
    check_step = opt.loss.check_step

    #### Setup optimizers ####
    vertex_optimizer = optim.Adam(vertex_model.parameters(), lr=learning_rate)
    face_optimizer = optim.Adam(face_model.parameters(), lr=learning_rate)

    fine_vertex_optimizer = optim.Adam(fine_vertex_model.parameters(), lr=learning_rate)
    fine_face_optimizer = optim.Adam(fine_face_model.parameters(), lr=learning_rate)
    #### Setup optimizers ####

    # if os.path.exists("/nas/datasets/gen"):
    if opt.dataset.lab == "thu": # save samples
        sv_mesh_folder = os.path.join("/nas/datasets/gen", "samples")
        os.makedirs(sv_mesh_folder, exist_ok=True)
        # sv_mesh_folder = os.path.join(sv_mesh_folder, "tables_large_model")
        sv_mesh_folder = os.path.join(sv_mesh_folder, model_descriptor)
        os.makedirs(sv_mesh_folder, exist_ok=True) # sv_mesh_folder

        sv_ckpts_folder = os.path.join("/nas/datasets/gen", "ckpts")
        os.makedirs(sv_ckpts_folder, exist_ok=True)
        sv_ckpts_folder = os.path.join(sv_ckpts_folder, model_descriptor)
        os.makedirs(sv_ckpts_folder, exist_ok=True)
    else:
        sv_mesh_folder = "./samples/tables_large_model/"
        os.makedirs("./samples", exist_ok=True)
        os.makedirs(sv_mesh_folder, exist_ok=True)

        sv_ckpts_folder = "./ckpts"
        os.makedirs(sv_ckpts_folder, exist_ok=True)
        sv_ckpts_folder = os.path.join(sv_ckpts_folder, model_descriptor)
        os.makedirs(sv_ckpts_folder, exist_ok=True)

    # train iters...
    # train_iter(dataset_train, dataset_test, vertex_model, face_model, vertex_optimizer, face_optimizer, training_steps, batch_size)
    test_iter(dataset_test, training_steps)
