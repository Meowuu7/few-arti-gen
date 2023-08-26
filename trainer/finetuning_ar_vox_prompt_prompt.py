# from glob import glob
from inspect import getmodule
import os
from turtle import exitonclick
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
from datasets.urdf_dataset_pretraining_ar_grid_vox import URDFDataset as URDFDataset_ar
# from datasets.urdf_dataset_pretraining_cus_perm_xl import URDFDataset as URDFDataset_cus_perm_xl
# from datasets.urdf_dataset_pretraining_cus_perm_xl_2 import URDFDataset as URDFDataset_cus_perm_xl
from modules.modules_pretraining_vox_prompt import VertexModel
# from modules.modules_pretraining_xl import VertexModel, FaceModel
# from modules.modules_pretraining_xl_2 import VertexModel, FaceModel
from modules.basic_modules import PromptValuesPrompt


from tqdm import tqdm
import time
import torch.optim as optim
from options.options import opt
from utils.constants import *


epoch_counter = 0
test_epoch_counter = 0
# use_cuda = True
step_counter = 0

# nn_max_vertices = 2000
# nn_max_faces = 2000

# todo: add logger to beautify the logging

def optimize(data, vertex_model, adapter_modules,  adapter_modules_optimizer):
    # global  use_cuda
    global step_counter

    vertex_model.eval()
    adapter_modules.train()
    # face_model.train()

    for k in data:
        data[k] = data[k].cuda(non_blocking=True)
        # print(f"key: {k}, shape: {data[k].size()}")

    # try:
    # vertex_model_pred_dict
    pred_dist_grid_xyz, pred_dist_grid_values = vertex_model(data, adapter_modules=adapter_modules)
    #
    # face_model_pred_dict = face_model(data)

    # grid_xyz_prediction_loss: bsz x grid_length x 3
    grid_xyz_prediction_loss = -torch.sum( # ientifiers
        pred_dist_grid_xyz.log_prob(data['grid_xyzs']) * data['grid_xyzs_mask'], dim=-1
    ) 
    grid_xyz_prediction_loss = grid_xyz_prediction_loss.sum(-1)
    # grid_xyz_prediction_loss = grid_xyz_prediction_loss.
    
    if pred_dist_grid_values is not None:
        grid_values_prediction_loss = -pred_dist_grid_values.log_prob(data['grid_content']) * data['grid_content_mask'] 
        # -torch.sum( # ientifiers
        #     pred_dist_grid_values.log_prob(data['grid_content']) * data['grid_content_mask'], dim=-1
        # ) 
        grid_values_prediction_loss = grid_values_prediction_loss.sum(-1).sum(-1).sum(-1).sum(-1)
    else:
        grid_values_prediction_loss = torch.zeros((1,), dtype=torch.float32).cuda()

    # # vertices loss and face loss
    # vertex_prediction_loss = -torch.sum( # ientifiers
    #     vertex_model_pred_dict.log_prob(data['vertices_flat']) * data['vertices_flat_mask'], dim=-1
    # ) 
    vertex_prediction_loss = torch.mean(grid_xyz_prediction_loss + grid_values_prediction_loss) #

    # face prediction loss
    # face_prediction_loss = -torch.sum(
    #     face_model_pred_dict.log_prob(data['faces']) * data['faces_mask'], dim=-1
    # ) 

    # face_prediction_loss = torch.sum(face_prediction_loss)

    loss = vertex_prediction_loss # + face_prediction_loss 
    # face_optimizer.zero_grad() # zero_grad
    # vertex_optimizer.zero_grad()
    adapter_modules_optimizer.zero_grad()
    loss.backward()
    # face_optimizer.step()
    adapter_modules_optimizer.step()
    # except:
    #     print(f"Something bad happended, continuing...")

    if step_counter % check_step == 0:
      print(f"Step {step_counter}")
      print(f"Vertex Loss {vertex_prediction_loss.item()}, grid_xyz_prediction_loss: {grid_xyz_prediction_loss.mean().item()}, grid_values_prediction_loss: {grid_values_prediction_loss.mean().item()}")
      # print(f"Face Loss {face_prediction_loss.item()}")
    step_counter += 1

    return



def step(train_dataset, vertex_model, adapter_modules, adapter_modules_optimizer, batch_size, dataset_iter):
    global epoch_counter
    try:
        data = next(dataset_iter)
        if data['class_label'].size(0) < batch_size:
            raise StopIteration
    except StopIteration:
        epoch_counter += 1
        # print("[DataLoader]: At Epoch %d!" % epoch_counter)
        # reset dataset iterator
        dataset_iter = iter(train_dataset)
        # get data for the next iteration
        data = next(dataset_iter)
    optimize(data, vertex_model, adapter_modules, adapter_modules_optimizer)
    return dataset_iter


def step_sample(test_dataset, vertex_model, adapter_modules, test_dataset_iter):
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
    sample(data, vertex_model, adapter_modules)
    return test_dataset_iter


def sample(test_data, vertex_model, adapter_modules, cur_step=0):
    # global nn_max_vertices
    # global nn_max_faces
    
    ##### Get vertices and faces here #####
    nn_max_vertices = opt.vertex_model.max_vertices
    # nn_max_faces = opt.face_model.max_faces
    bsz = opt.loss.batch_size
    ##### Get vertices and faces here #####

    vertex_model.eval()
    adapter_modules.eval()
    # face_model.eval()
    class_label = opt.model.sample_class_idx
    print(f"Sampling class {class_label}...")

    with torch.no_grad():

        for k in test_data:
          test_data[k] = test_data[k].cuda(non_blocking=True)
        # print(f"key: {k}, shape: {data[k].size()}")
        # construct sampling context
        # class_label 
        # sample_context = {
        #     'class_label': torch.full(size=(bsz,), fill_value=class_label, dtype=torch.long).cuda()
        # }
        # perform vertex sampling #
        vertex_sample_dict = vertex_model.sample(bsz, context=test_data, max_sample_length=nn_max_vertices, temperature=1., top_k=0, top_p=1.0, recenter_verts=True, only_return_complete=False, adapter_modules=adapter_modules)

        # # class_label
        # # recenter_verts
        # # vertex_sample_dict['class_label'] = sample_context['class_label'] # class labels

        # # perform face sampling
        # face_sample_dict = face_model.sample(vertex_sample_dict, max_sample_length=nn_max_faces, temperature=1., top_k=0,  top_p=1.0,  only_return_complete=False)

        print("Face sampling finished!")

        ### convert torch tensors to numpy values ####
        for k in vertex_sample_dict: # sample dict
            if isinstance(vertex_sample_dict[k], dict):
                for sub_k in vertex_sample_dict[k]:
                    vertex_sample_dict[k][sub_k] = vertex_sample_dict[k][sub_k].detach().cpu().numpy()
            else:
                vertex_sample_dict[k] = vertex_sample_dict[k].detach().cpu().numpy()
        # for k in face_sample_dict:
        #     if isinstance(face_sample_dict[k], dict):
        #         for sub_k in face_sample_dict[k]:
        #             face_sample_dict[k][sub_k] = face_sample_dict[k][sub_k].detach().cpu().numpy()
        #     else:
        #         face_sample_dict[k] = face_sample_dict[k].detach().cpu().numpy()
        # test_data_np = {} # numpys
        # for k in test_data:
        #     if isinstance(test_data[k], dict):
        #         test_data_np[k] = {}
        #         for sub_k in test_data[k]:
        #             test_data_np[k][sub_k] = test_data[k][sub_k].detach().cpu().numpy()
        #     else:
        #         test_data_np[k] = test_data[k].detach().cpu().numpy()
        print("Ploting sampled meshes...")
        data_utils.plot_grids_for_pretraining(v_sample=vertex_sample_dict, sv_mesh_folder=sv_mesh_folder, cur_step=cur_step) 



def save_ckpts(vertex_model, adapter_modules, cur_step):
    vertex_model_sv_folder = os.path.join(sv_ckpts_folder, "vertex_model")
    os.makedirs(vertex_model_sv_folder, exist_ok=True)

    adapter_modules_sv_folder = os.path.join(sv_ckpts_folder, "adapter_modules")
    os.makedirs(adapter_modules_sv_folder, exist_ok=True)

    # face_model_sv_folder = os.path.join(sv_ckpts_folder, "face_model")
    # os.makedirs(face_model_sv_folder, exist_ok=True)
    model_sv_path = f'{cur_step}.pth'
    vertex_model_sv_pth = os.path.join(vertex_model_sv_folder, model_sv_path)
    vertex_model_params = vertex_model.state_dict()
    torch.save(vertex_model_params, vertex_model_sv_pth)

    adapter_modules_sv_pth = os.path.join(adapter_modules_sv_folder, model_sv_path)
    adapter_modules_params = adapter_modules.state_dict()
    torch.save(adapter_modules_params, adapter_modules_sv_pth)

    # model_sv_path = f'{cur_step}.pth'
    # face_model_sv_pth = os.path.join(face_model_sv_folder, model_sv_path)
    # face_model_params = face_model.state_dict()
    # torch.save(face_model_params, face_model_sv_pth) # face model path...
    print(f"Vertex model at step {cur_step} saved to {vertex_model_sv_pth}.")
    print(f"Adapter modules at step {cur_step} saved to {adapter_modules_sv_pth}.")
    # print(f"Face model at step {cur_step} saved to {face_model_sv_pth}.")



# train iter
# 
def train_iter(train_dataset, test_dataset, vertex_model, adapter_modules, adapter_modules_optimizer, training_steps, batch_size):
    # dataset_iter
    # iter dataset
    dataset_iter = iter(train_dataset)
    test_dataset_iter = iter(test_dataset)
    for i in range(training_steps):
        dataset_iter = step(train_dataset, vertex_model, adapter_modules, adapter_modules_optimizer, batch_size, dataset_iter)

        if i % check_step == 0: # then sample from the model and save
            # sample(num_samples=batch_size, test_dataset=test_dataset, vertex_model=vertex_model, face_model=face_model, class_label=0, cur_step=i)
            test_dataset_iter = step_sample(test_dataset, vertex_model, adapter_modules, test_dataset_iter)
            save_ckpts(vertex_model, adapter_modules, cur_step=i)


# sometiems more easier than eyeglasses?
def get_model_descriptor():
    motion_dataset_nm = opt.dataset.dataset_name
    # motion_dataset_nm
    # motion_cat = "eyeglasses"
    motion_cat = opt.dataset.category
    batch_size = opt.loss.batch_size
    # nn 
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
    use_light_weight = opt.model.use_light_weight
    apply_random_shift = opt.dataset.apply_random_shift
    apply_random_flipping = opt.dataset.apply_random_flipping
    category_part_cat = opt.dataset.category_part_indicator
    face_class_conditional = opt.face_model.face_class_conditional
    random_scaling = opt.dataset.apply_random_scaling
    specify_light_weight = opt.model.specify_light_weight
    nn_vertices_predict_ratio = opt.dataset.nn_vertices_predict_ratio
    use_eyeglasses_frame = opt.dataset.use_eyeglasses_frame
    context_window = opt.model.context_window
    grid_size = opt.vertex_model.grid_size
    max_num_grids = opt.vertex_model.max_num_grids
    prefix_key_len = opt.model.prefix_key_len # prefix_key_len for ...
    model_descriptor = f"{exp_mode}_ar_vox_prompt_{exp_flag}_indi_{category_part_cat}_ctx_{context_window}_gs_{max_num_grids}_scale_{random_scaling}_{motion_dataset_nm}_bsz_{batch_size}_max_verts_{nn_max_vertices}_max_permit_verts_{nn_max_permite_vertices}_key_len_{prefix_key_len}"
    return model_descriptor


def safe_load_ckpt_common(model, state_dicts):
    ori_dict = state_dicts
    part_dict = dict()
    model_dict = model.state_dict()
    tot_params_n = 0
    tot_nn_params = 0
    for k in ori_dict:
        if k in model_dict:
            v = ori_dict[k]
            part_dict[k] = v
            tot_params_n += 1
            tot_nn_params += v.nelement()
    model_dict.update(part_dict)
    model.load_state_dict(model_dict)
    # self.logger.log('Setup', f"Resume glb-backbone finished!! Total number of parameters: {tot_params_n}.")
    print(f"Resume from the checkpoint with {tot_params_n} parameters and total size {tot_nn_params}.")
    #


def resume_from_ckpts(vertex_model, vertex_model_path):

    vertex_model_state_dicts = torch.load(vertex_model_path, map_location="cpu")
    # face_model_state_dicts = torch.load(face_model_path, map_location="cpu")

    safe_load_ckpt_common(vertex_model, vertex_model_state_dicts)
    # safe_load_ckpt_common(face_model, face_model_state_dicts)


    # vertex_model.load_state_dict(vertex_model_state_dicts)
    # face_model.load_state_dict(face_model_state_dicts)
    print(f"Vertex model loaded from {vertex_model_path}.")
    # print(f"Face model loaded from {face_model_path}.")


        
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
    dataset_root_folder = os.path.join(dataset_root_path, motion_dataset_nm)
    
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

    # if motion_cat == "eyeglasses":
    train_ins_nns = None
    test_ins_nns = None
    # else:
    #   train_ins_nns = [str(iii) for iii in range(3000)]
    #   test_ins_nns = [str(iii) for iii in range(3000, 4001)]

    # if not cus_perm_xl:
    URDFDataset = URDFDataset_ar
    # else:
    #     URDFDataset = URDFDataset_cus_perm_xl

    debug = opt.model.debug

    if not debug:

    # #### for train dataset ####
    #   category_name = ["eyeglasses"]
    #   category_names_to_part_names = {
    #     "eyeglasses": ["dof_rootd_Aa001_r", "dof_rootd_Aa002_r"]
    #   }
    #   category_names = list(category_names_to_part_names.keys())
    # #### for train dataset #### # bucket lighter "faucet", "globe",  "closestool"
     # "wine_bottle", "clock", "cabinet", "lamp", "oven", "seesaw", "washing_machine"

      # category_name = ["screwdriver", "eyeglasses",  "pen", "scissors", "stapler",  "globe",  "closestool", "faucet", "lighter", "bucket", "wine_bottle", "clock", "cabinet", "lamp", "oven", "seesaw", "washing_machine"]

      # category_name = os.listdir(dataset_root_path)
      # category_name = os.listdir(dataset_root_folder)
      # category_name = [fn for fn in category_name if os.path.isdir(os.path.join(dataset_root_folder, fn))]
      # print(f"Categories: {category_name}")
      # category_name = ["screwdriver", "eyeglasses",  "pen", "scissors", "stapler",]
      # category_names_to_part_names = {cat: None for cat in category_name}
      # if (not opt.model.specify_light_weight) and (not opt.dataset.use_eyeglasses_frame): # then wes
      #     category_names_to_part_names["eyeglasses"] = ["dof_rootd_Aa001_r", "dof_rootd_Aa002_r"]
      # category_names = category_name
      # ###### valve

        # #### for train dataset ####
        # category_name = ["eyeglasses"]
        # category_names_to_part_names = {
        #     "eyeglasses": ["none_motion"]
        # }
        # category_names = list(category_names_to_part_names.keys())
        # #### for train dataset ####

        #### for train dataset ####
        category_name = ["globe"]
        category_names_to_part_names = {
            "globe": None
        }
        category_names = list(category_names_to_part_names.keys())
        #### for train dataset ####


        # #### for train dataset ####
        # category_name = ["closestool"]
        # category_names_to_part_names = {
        #     "closestool": None
        # }
        # category_names = list(category_names_to_part_names.keys())
        # #### for train dataset ####
    else:
    #   #### for train dataset ####
    #   category_name = ["eyeglasses",]
    #   category_names_to_part_names = {
    #       "eyeglasses": ["dof_rootd_Aa001_r"], 
    #   }
    #   category_names = list(category_names_to_part_names.keys())
    #   #### for train dataset ####

      #### for train dataset ####
        category_name = ["valve"]
        category_names_to_part_names = {
            "valve": None
        }
        category_names = list(category_names_to_part_names.keys())
        #### for train dataset ####

      # category_name = ["wine_bottle"]
      # category_names_to_part_names = {
      #     "wine_bottle": None
      # }
      # category_names = list(category_names_to_part_names.keys())

      # #### for train dataset ####
      # category_name = ["eyeglasses"]
      # category_names_to_part_names = {
      #     "eyeglasses": ["dof_rootd_Aa001_r"]
      # }
      # category_names = list(category_names_to_part_names.keys())
      # #### for train dataset ####

    print(f"motion cat: {motion_cat}")

    dataset_train = URDFDataset(root_folder=root_folder, quantization_bits=quantization_bits, nn_max_vertices=nn_max_vertices, nn_max_faces=nn_max_faces, nn_max_permite_vertices=nn_max_permite_vertices, nn_max_permite_faces=nn_max_permite_faces, category_name=category_names, instance_nns=train_ins_nns, part_names=None, category_nm_to_part_nm=category_names_to_part_names, is_training=True)

    dataset_train_category_part_indicator_to_idxes = dataset_train.category_part_indicator_to_idxes

    dataset_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=num_thread)
    
    if not debug:
        #### for train dataset ####
        category_name = ["eyeglasses"]
        category_names_to_part_names = {
            "eyeglasses": ["none_motion"]
        }
        category_names = list(category_names_to_part_names.keys())
        #### for train dataset ####

        # #### for train dataset ####
        # category_name = ["water_bottle"]
        # category_names_to_part_names = {
        #     "eyeglasses": ["none_motion"]
        # }
        # category_names = list(category_names_to_part_names.keys())
        # #### for train dataset ####

        #### for train dataset ####
        category_name = ["globe"]
        category_names_to_part_names = {
            "globe": None
        }
        category_names = list(category_names_to_part_names.keys())
        #### for train dataset ####
    else:
        #### for train dataset ####
        category_name = ["eyeglasses"]
        category_names_to_part_names = {
            "eyeglasses": ["dof_rootd_Aa001_r"]
        }
        category_names = list(category_names_to_part_names.keys())
        #### for train dataset ####

    

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

    dataset_test_category_part_indicator_to_idxes = dataset_test.category_part_indicator_to_idxes

    # shuffle = False
    # dataet for train...
    dataset_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_thread)


    ##### Model configs #####
    if opt.model.specify_light_weight:
        vertex_module_config, face_module_config = get_model_configs(use_light_weight=True)
    else:
        vertex_module_config, face_module_config = get_model_configs()

    adapter_modules_config = get_prompt_prompt_value_config()
    adapter_modules = PromptValuesPrompt(**adapter_modules_config)
    
    vertex_model = VertexModel(**vertex_module_config)
    # face_model = FaceModel(**face_module_config)

    vertex_model_path = opt.model.vertex_model_path
    # face_model_path = opt.model.face_model_path

    if len(vertex_model_path) > 0:
        resume_from_ckpts(vertex_model, vertex_model_path)

    vertex_model = vertex_model.cuda()
    # face_model = face_model.cuda()
    adapter_modules = adapter_modules.cuda()

    print("Dataset constructed!")

    # Optimization settings
    # learning_rate = 5e-4
    # training_steps = 2000 * 500
    # check_step = 5

    learning_rate = opt.loss.learning_rate
    training_steps = opt.loss.training_steps
    check_step = opt.loss.check_step

    #### Setup optimizers ####
    # vertex_optimizer = optim.Adam(vertex_model.parameters(), lr=learning_rate)
    adapter_modules_optimizer = optim.Adam(adapter_modules.parameters(), lr=learning_rate)
    # face_optimizer = optim.Adam(face_model.parameters(), lr=learning_rate)
    #### Setup optimizers ####

    # if os.path.exists("/nas/datasets/gen"):
    # if opt.dataset.lab == "thu": # save samples
    sv_mesh_folder = os.path.join("/nas/datasets/gen", "samples")
    os.makedirs(sv_mesh_folder, exist_ok=True)
    # sv_mesh_folder = os.path.join(sv_mesh_folder, "tables_large_model")
    sv_mesh_folder = os.path.join(sv_mesh_folder, model_descriptor)
    os.makedirs(sv_mesh_folder, exist_ok=True)

    sv_ckpts_folder = os.path.join("/nas/datasets/gen", "ckpts")
    os.makedirs(sv_ckpts_folder, exist_ok=True)
    sv_ckpts_folder = os.path.join(sv_ckpts_folder, model_descriptor)
    os.makedirs(sv_ckpts_folder, exist_ok=True)
    # else:
    #     sv_mesh_folder = "./samples/tables_large_model/"
    #     os.makedirs("./samples", exist_ok=True)
    #     os.makedirs(sv_mesh_folder, exist_ok=True)

    #     sv_ckpts_folder = "./ckpts"
    #     os.makedirs(sv_ckpts_folder, exist_ok=True)
    #     sv_ckpts_folder = os.path.join(sv_ckpts_folder, model_descriptor)
    #     os.makedirs(sv_ckpts_folder, exist_ok=True)

    ### save dataset train related information ###
    dataset_train_part_indicator_to_idx_fn = os.path.join(sv_ckpts_folder, "dataset_train_part_indicator_to_idx.npy")
    np.save(dataset_train_part_indicator_to_idx_fn, dataset_train_category_part_indicator_to_idxes)
    ### save dataset test related information ###
    dataset_test_part_indicator_to_idx_fn = os.path.join(sv_ckpts_folder, "dataset_test_part_indicator_to_idx.npy")
    np.save(dataset_test_part_indicator_to_idx_fn, dataset_test_category_part_indicator_to_idxes)

    # train iters...
    # train_iter(dataset_train, dataset_test, vertex_model, face_model, vertex_optimizer, face_optimizer, training_steps, batch_size)
    train_iter(dataset_train, dataset_test, vertex_model, adapter_modules, adapter_modules_optimizer, training_steps, batch_size)
