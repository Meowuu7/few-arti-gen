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
from datasets.urdf_dataset_pretraining_ar import URDFDataset as URDFDataset_ar
# from datasets.urdf_dataset_pretraining_cus_perm_xl import URDFDataset as URDFDataset_cus_perm_xl
# from datasets.urdf_dataset_pretraining_cus_perm_xl_2 import URDFDataset as URDFDataset_cus_perm_xl
# from modules.modules_with_adapters import VertexModel, FaceModel
from modules.modules_with_prompt import VertexModel, FaceModel
# from modules.basic_modules import AdapterTuning
from modules.basic_modules import PrefixKeyValues
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

# nn_max_vertices = 2000
# nn_max_faces = 2000

# todo: add logger to beautify the logging

def optimize(data, vertex_model, face_model, vertex_optimizer, face_optimizer):
    # global  use_cuda
    global step_counter

    vertex_model.train()
    face_model.train()

    # vertex_model.eval()
    # face_model.eval()

    encoder_adapter.train()
    decoder_adapter_vertex.train()
    decoder_adapter_face.train()

    for k in data:
        data[k] = data[k].cuda(non_blocking=True)
        # print(f"key: {k}, shape: {data[k].size()}")

    # vertex_model_pred_dict
    vertex_model_pred_dict = vertex_model(data, adapter_module=decoder_adapter_vertex)
    #
    # print("After forwarding vertex model...")
    face_model_pred_dict = face_model(data, encoder_adapter=encoder_adapter, decoder_adapter=decoder_adapter_face)
    # print("After forwarding face model...")

    # vertices loss and face loss
    vertex_prediction_loss = -torch.sum( # ientifiers
        vertex_model_pred_dict.log_prob(data['vertices_flat']) * data['vertices_flat_mask'], dim=-1
    )

    vertex_prediction_loss = torch.sum(vertex_prediction_loss) #

    # face prediction loss
    face_prediction_loss = -torch.sum(
        face_model_pred_dict.log_prob(data['faces']) * data['faces_mask'], dim=-1
    ) 

    face_prediction_loss = torch.sum(face_prediction_loss)

    loss = vertex_prediction_loss + face_prediction_loss 
    # face_optimizer.zero_grad() # zero_grad
    # vertex_optimizer.zero_grad()

    encoder_adapter_optimizer.zero_grad()
    decoder_adapter_vertex_optimizer.zero_grad()
    decoder_adapter_face_optimizer.zero_grad()

    loss.backward()
    # face_optimizer.step()
    # vertex_optimizer.step()

    encoder_adapter_optimizer.step()
    decoder_adapter_vertex_optimizer.step()
    decoder_adapter_face_optimizer.step()

    if step_counter % check_step == 0:
      print(f"Step {step_counter}")
      print(f"Vertex Loss {vertex_prediction_loss.item()}")
      print(f"Face Loss {face_prediction_loss.item()}")
    step_counter += 1

    return



def step(train_dataset, vertex_model, face_model, vertex_optimizer, face_optimizer, batch_size, dataset_iter):
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
    optimize(data, vertex_model, face_model, vertex_optimizer, face_optimizer)
    return dataset_iter

def step_sample(test_dataset, vertex_model, face_model, test_dataset_iter):
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
    sample(data, vertex_model, face_model)
    return test_dataset_iter


def sample(test_data, vertex_model, face_model, cur_step=0):
    # global nn_max_vertices
    # global nn_max_faces
    
    ##### Get vertices and faces here #####
    nn_max_vertices = opt.vertex_model.max_vertices
    nn_max_faces = opt.face_model.max_faces
    bsz = opt.loss.batch_size
    ##### Get vertices and faces here #####

    vertex_model.eval()
    face_model.eval()

    encoder_adapter.eval()
    decoder_adapter_vertex.eval()
    decoder_adapter_face.eval()

    with torch.no_grad():

        for k in test_data:
          test_data[k] = test_data[k].cuda(non_blocking=True)
        # print(f"key: {k}, shape: {data[k].size()}")
        # construct sampling context
        # sample_context = {
        #     'class_label': torch.full(size=(num_samples,), fill_value=class_label, dtype=torch.long).cuda()
        # }
        # perform vertex sampling #
        vertex_sample_dict = vertex_model.sample(bsz, context=test_data, max_sample_length=nn_max_vertices, temperature=1., top_k=0, top_p=1.0, recenter_verts=True, only_return_complete=False, adapter_module=decoder_adapter_vertex)

        # class_label
        # recenter_verts
        # vertex_sample_dict['class_label'] = sample_context['class_label'] # class labels

        # perform face sampling
        face_sample_dict = face_model.sample(vertex_sample_dict, max_sample_length=nn_max_faces, temperature=1., top_k=0,  top_p=1.0,  only_return_complete=False, encoder_adapter=encoder_adapter, decoder_adapter=decoder_adapter_face)

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
        for k in test_data:
            if isinstance(test_data[k], dict):
                test_data_np[k] = {}
                for sub_k in test_data[k]:
                    test_data_np[k][sub_k] = test_data[k][sub_k].detach().cpu().numpy()
            else:
                test_data_np[k] = test_data[k].detach().cpu().numpy()
        print("Ploting sampled meshes...")
        ### plot sampled meshes to the obj file ###
        # plot sampled
        # try:
        # data_utils.plot_sampled_meshes(v_sample=vertex_sample_dict, f_sample=face_sample_dict, sv_mesh_folder="./samples/eyeglasses/meshes/", cur_step=cur_step, predict_joint=predict_joint) # predict joint variable... 
        data_utils.plot_sampled_meshes_single_part_for_pretraining(v_sample=vertex_sample_dict, f_sample=face_sample_dict, context=test_data_np, sv_mesh_folder=sv_mesh_folder, cur_step=cur_step, predict_joint=False) # predict joint variable...
        # except:
        #     pass



def save_ckpts(vertex_model, face_model, cur_step):
    vertex_model_sv_folder = os.path.join(sv_ckpts_folder, "vertex_model")
    os.makedirs(vertex_model_sv_folder, exist_ok=True)
    face_model_sv_folder = os.path.join(sv_ckpts_folder, "face_model")
    os.makedirs(face_model_sv_folder, exist_ok=True)
    encoder_adapter_sv_folder = os.path.join(sv_ckpts_folder, "encoder_adapter")
    os.makedirs(encoder_adapter_sv_folder, exist_ok=True)
    decoder_adapter_face_sv_folder = os.path.join(sv_ckpts_folder, "decoder_adapter_face")
    os.makedirs(decoder_adapter_face_sv_folder, exist_ok=True)
    decoder_adapter_vertex_sv_folder = os.path.join(sv_ckpts_folder, "decoder_adapter_vertex")
    os.makedirs(decoder_adapter_vertex_sv_folder, exist_ok=True)

    model_sv_path = f'{cur_step}.pth'

    vertex_model_sv_pth = os.path.join(vertex_model_sv_folder, model_sv_path)
    vertex_model_params = vertex_model.state_dict()
    torch.save(vertex_model_params, vertex_model_sv_pth)

    adapter_encoder_sv_pth = os.path.join(encoder_adapter_sv_folder, model_sv_path)
    adapter_encoder_params = encoder_adapter.state_dict()
    torch.save(adapter_encoder_params, adapter_encoder_sv_pth)

    adapter_decoder_face_sv_pth = os.path.join(decoder_adapter_face_sv_folder, model_sv_path)
    decoder_adapter_face_params = decoder_adapter_face.state_dict()
    torch.save(decoder_adapter_face_params, adapter_decoder_face_sv_pth)

    adapter_decoder_vertex_sv_pth = os.path.join(decoder_adapter_vertex_sv_folder, model_sv_path)
    decoder_adapter_vertex_params = decoder_adapter_vertex.state_dict()
    torch.save(decoder_adapter_vertex_params, adapter_decoder_vertex_sv_pth)

    # model_sv_path = f'{cur_step}.pth'
    face_model_sv_pth = os.path.join(face_model_sv_folder, model_sv_path)
    face_model_params = face_model.state_dict()
    torch.save(face_model_params, face_model_sv_pth) # face model path...
    print(f"Vertex model at step {cur_step} saved to {vertex_model_sv_pth}.")
    print(f"Face model at step {cur_step} saved to {face_model_sv_pth}.")
    print(f"Adapter encoder  at step {cur_step} saved to {adapter_encoder_sv_pth}.")
    print(f"Adapter decoder face  at step {cur_step} saved to {adapter_decoder_face_sv_pth}.")
    print(f"Adapter decdoer vertex at step {cur_step} saved to {adapter_decoder_vertex_sv_pth}.")



# train iter
def train_iter(train_dataset, test_dataset, vertex_model, face_model, vertex_optimizer, face_optimizer, training_steps, batch_size):
    # dataset_iter
    # iter dataset
    dataset_iter = iter(train_dataset)
    test_dataset_iter = iter(test_dataset)
    for i in range(training_steps):
        dataset_iter = step(train_dataset, vertex_model, face_model, vertex_optimizer, face_optimizer, batch_size, dataset_iter)
        if i % check_step == 0: # then sample from the model and save
            # sample(num_samples=batch_size, test_dataset=test_dataset, vertex_model=vertex_model, face_model=face_model, class_label=0, cur_step=i)
            test_dataset_iter = step_sample(test_dataset, vertex_model, face_model, test_dataset_iter)
            save_ckpts(vertex_model, face_model, cur_step=i)


# TODO notice that `exp-mode` should be set to `finetuning`
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
    context_window = opt.model.context_window
    apply_random_scaling = opt.dataset.apply_random_scaling
    model_descriptor = f"{exp_mode}_ar_prompt_{exp_flag}_ctx_{context_window}_cat_part_indi_{category_part_cat}_face_cond_{face_class_conditional}_light_{use_light_weight}_shift_{apply_random_shift}_scale_{apply_random_scaling}_{motion_dataset_nm}_bsz_{batch_size}_max_verts_{nn_max_vertices}_faces_{nn_max_faces}_max_permit_verts_{nn_max_permite_vertices}_faces_{nn_max_permite_faces}"
    return model_descriptor



def safe_load_ckpt_common(model, state_dicts):
    ori_dict = state_dicts
    part_dict = dict()
    model_dict = model.state_dict()
    tot_params_n = 0
    for k in ori_dict:
        if k in model_dict:
            v = ori_dict[k]
            part_dict[k] = v
            tot_params_n += 1
    model_dict.update(part_dict)
    model.load_state_dict(model_dict)
    # self.logger.log('Setup', f"Resume glb-backbone finished!! Total number of parameters: {tot_params_n}.")
    #


def resume_from_ckpts(vertex_model, face_model, vertex_model_path, face_model_path):
    vertex_model_state_dicts = torch.load(vertex_model_path, map_location="cpu")
    face_model_state_dicts = torch.load(face_model_path, map_location="cpu")

    safe_load_ckpt_common(vertex_model, vertex_model_state_dicts)
    safe_load_ckpt_common(face_model, face_model_state_dicts)

    # vertex_model.load_state_dict(vertex_model_state_dicts) # 
    # face_model.load_state_dict(face_model_state_dicts)
    print(f"Vertex model loaded from {vertex_model_path}.")
    print(f"Face model loaded from {face_model_path}.")


        
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

    train_ins_nns = None
    test_ins_nns = None

    # if not cus_perm_xl:
    URDFDataset = URDFDataset_ar
    # else:
    #     URDFDataset = URDFDataset_cus_perm_xl
  
    debug = opt.model.debug

    # #### for train dataset #### # we still use 
    category_name = ["eyeglasses"]
    category_names_to_part_names = {
        "eyeglasses": ["none_motion"]
    }
    category_names = list(category_names_to_part_names.keys())
    #### for train dataset ####

    #  #### for train dataset ####
    # category_name = ["water_bottle"]
    # category_names_to_part_names = {
    #     "water_bottle": ["none_motion"]
    # }
    # category_names = list(category_names_to_part_names.keys())
    # #### for train dataset ####

    # #### for train dataset ####
    # category_name = ["oven"]
    # category_names_to_part_names = {
    #     "oven": ["none_motion"]
    # }
    # category_names = list(category_names_to_part_names.keys())
    # #### for train dataset ####
    
    print(f"motion cat: {motion_cat}")

    dataset_train = URDFDataset(root_folder=root_folder, quantization_bits=quantization_bits, nn_max_vertices=nn_max_vertices, nn_max_faces=nn_max_faces, nn_max_permite_vertices=nn_max_permite_vertices, nn_max_permite_faces=nn_max_permite_faces, category_name=category_names, instance_nns=train_ins_nns, part_names=None, category_nm_to_part_nm=category_names_to_part_names, is_training=True)

    dataset_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=num_thread)

    #### for train dataset ####
    category_name = ["eyeglasses"]
    category_names_to_part_names = {
        "eyeglasses": ["none_motion"]
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

    # #### for train dataset ####
    # category_name = ["water_bottle"]
    # category_names_to_part_names = {
    #     "water_bottle": ["none_motion"]
    # }
    # category_names = list(category_names_to_part_names.keys())
    # #### for train dataset ####

    # #### for train dataset ####
    # category_name = ["oven"]
    # category_names_to_part_names = {
    #     "oven": ["none_motion"]
    # }
    # category_names = list(category_names_to_part_names.keys())
    # #### for train dataset ####

    dataset_test = URDFDataset(root_folder=root_folder, quantization_bits=quantization_bits, nn_max_vertices=nn_max_vertices, nn_max_faces=nn_max_faces, nn_max_permite_vertices=nn_max_permite_vertices, nn_max_permite_faces=nn_max_permite_faces, category_name=category_name, instance_nns=test_ins_nns, part_names=None, category_nm_to_part_nm=category_names_to_part_names, is_training=False)

    # shuffle = False
    # dataet for train...
    dataset_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_thread)


    ##### Model configs #####
    vertex_module_config, face_module_config = get_model_configs()
    vertex_model = VertexModel(**vertex_module_config)
    face_model = FaceModel(**face_module_config)

    vertex_model_path = opt.model.vertex_model_path
    face_model_path = opt.model.face_model_path

    resume_from_ckpts(vertex_model, face_model, vertex_model_path, face_model_path)
    # resume_from_ckpts ---> 

    # adapter_module_config = get_adapter_config()

    prefix_key_value_config = get_prefix_key_value_config()

    # encoder_adapter = AdapterTuning(**adapter_module_config)
    # decoder_adapter_vertex = AdapterTuning(**adapter_module_config)
    # decoder_adapter_face = AdapterTuning(**adapter_module_config)

    encoder_adapter = PrefixKeyValues(**prefix_key_value_config)
    decoder_adapter_vertex = PrefixKeyValues(**prefix_key_value_config) # adapter ver
    decoder_adapter_face = PrefixKeyValues(**prefix_key_value_config)

    vertex_model = vertex_model.cuda()
    face_model = face_model.cuda()

    encoder_adapter = encoder_adapter.cuda()
    decoder_adapter_vertex = decoder_adapter_vertex.cuda()
    decoder_adapter_face = decoder_adapter_face.cuda()

    


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

    encoder_adapter_optimizer = optim.Adam(encoder_adapter.parameters(), lr=learning_rate)
    decoder_adapter_vertex_optimizer = optim.Adam(decoder_adapter_vertex.parameters(), lr=learning_rate)
    decoder_adapter_face_optimizer = optim.Adam(decoder_adapter_face.parameters(), lr=learning_rate)
    #### Setup optimizers ####

    # if os.path.exists("/nas/datasets/gen"):
    if opt.dataset.lab == "thu": #
        sv_mesh_folder = os.path.join("/nas/datasets/gen", "samples")
        os.makedirs(sv_mesh_folder, exist_ok=True)
        # sv_mesh_folder = os.path.join(sv_mesh_folder, "tables_large_model")
        sv_mesh_folder = os.path.join(sv_mesh_folder, model_descriptor)
        os.makedirs(sv_mesh_folder, exist_ok=True)

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
    train_iter(dataset_train, dataset_test, vertex_model, face_model, vertex_optimizer, face_optimizer, training_steps, batch_size)
