# from glob import glob
from inspect import getmodule
import os
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
from datasets.urdf_dataset_pretraining import URDFDataset
# from modules.modules import VertexModel, VertexModelPart, FaceModelPart, FaceModel
from modules.modules_pretraining import VertexModel, FaceModel


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


# distribution of the vertex and distribution of 

def get_model_configs():
    # use_light_weight_model = True
    use_light_weight_model = opt.model.use_light_weight
    # use_light_weight_model = False
    memory_efficient = True
    part_tree = {'idx': 0}
  
    vertex_module_config = dict(
        encoder_config=dict(
        hidden_size=512,
        fc_size=2048,
        num_heads=8,
        layer_norm=True,
        num_layers=10,
        dropout_rate=0.2,
        re_zero=True, # 
        ),
        decoder_config=dict(
        hidden_size=512,
        fc_size=2048,
        num_heads=8,
        layer_norm=True,
        num_layers=24,
        dropout_rate=0.4,
        re_zero=True,
        ),
        num_classes=opt.model.num_classes,
        quantization_bits=opt.model.quantization_bits,
        class_conditional=opt.vertex_model.vertex_class_conditional,
        max_num_input_verts=opt.vertex_model.max_vertices,
        inter_part_auto_regressive=opt.vertex_model.inter_part_auto_regressive,
        predict_joint=opt.vertex_model.predict_joint,
        use_discrete_embeddings=opt.model.use_discrete_embeddings,
        # part_tree=part_tree
    )
    #### vertex module config ####

    #### vertex module config light ####
    vertex_module_config_light = dict(
        encoder_config=dict(
        hidden_size=128,
        fc_size=512,
        num_heads=4,
        layer_norm=True,
        num_layers=3,
        dropout_rate=0.2,
        re_zero=True,
        ),
        decoder_config=dict(
        hidden_size=128,
        fc_size=512,
        num_heads=4,
        layer_norm=True,
        num_layers=3,
        dropout_rate=0.4,
        re_zero=True,
        ),
        num_classes=opt.model.num_classes,
        quantization_bits=opt.model.quantization_bits,
        class_conditional=opt.vertex_model.vertex_class_conditional,
        max_num_input_verts=opt.vertex_model.max_vertices,
        inter_part_auto_regressive=opt.vertex_model.inter_part_auto_regressive,
        predict_joint=opt.vertex_model.predict_joint,
        use_discrete_embeddings=opt.model.use_discrete_embeddings,
        # part_tree=part_tree
    )
    #### vertex module config light ####

    #### face module config ####
    face_module_config=dict(
        encoder_config=dict(
            hidden_size=512,
            fc_size=2048,
            num_heads=8,
            layer_norm=True,
            num_layers=10,
            dropout_rate=0.2,
            re_zero=True,
        ),
        decoder_config=dict(
            hidden_size=512,
            fc_size=2048,
            num_heads=8,
            layer_norm=True,
            num_layers=14,
            dropout_rate=0.2,
            re_zero=True,
        ),
        class_conditional=opt.face_model.face_class_conditional,
        quantization_bits=opt.model.quantization_bits,
        decoder_cross_attention=opt.face_model.decoder_cross_attention,
        use_discrete_vertex_embeddings=opt.model.use_discrete_embeddings,
        max_seq_length=opt.face_model.max_faces,
        # part_tree={'idx': 0}
    )
    #### face module config ####

    #### face module config light ####
    face_module_config_light = dict(
        encoder_config=dict(
            hidden_size=128,
            fc_size=512,
            num_heads=4,
            layer_norm=True,
            num_layers=3,
            dropout_rate=0.2,
            re_zero=True,
        ),
        decoder_config=dict(
            hidden_size=128,
            fc_size=512,
            num_heads=4,
            layer_norm=True,
            num_layers=3,
            dropout_rate=0.2,
            re_zero=True,
        ),
        class_conditional=opt.face_model.face_class_conditional,
        quantization_bits=opt.model.quantization_bits,
        decoder_cross_attention=opt.face_model.decoder_cross_attention,
        use_discrete_vertex_embeddings=opt.model.use_discrete_embeddings,
        max_seq_length=opt.face_model.max_faces,
        # part_tree={'idx': 0}
    )
    vertex_module_config = vertex_module_config_light if use_light_weight_model else vertex_module_config
    face_module_config = face_module_config_light if use_light_weight_model else face_module_config
    return vertex_module_config, face_module_config


def optimize(data, vertex_model, face_model, vertex_optimizer, face_optimizer):
    # global  use_cuda
    global step_counter

    vertex_model.train()
    face_model.train()

    for k in data:
        data[k] = data[k].cuda(non_blocking=True)
        # print(f"key: {k}, shape: {data[k].size()}")

    # vertex_model_pred_dict
    vertex_model_pred_dict = vertex_model(data)
    #
    face_model_pred_dict = face_model(data)

    # vertices loss and face loss
    vertex_prediction_loss = -torch.sum( # ientifiers
        vertex_model_pred_dict.log_prob(data['vertices_flat']) * data['vertices_mask_identifier'], dim=-1
    ) 

    vertex_prediction_loss = torch.sum(vertex_prediction_loss) #

    # face prediction loss
    face_prediction_loss = -torch.sum(
        face_model_pred_dict.log_prob(data['faces']) * data['faces_mask'], dim=-1
    ) 

    face_prediction_loss = torch.sum(face_prediction_loss)

    loss = vertex_prediction_loss + face_prediction_loss
    face_optimizer.zero_grad() # zero_grad
    vertex_optimizer.zero_grad()
    loss.backward()
    face_optimizer.step()
    vertex_optimizer.step()

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

def step_sample(test_dataset, vertex_model, face_model, dataset_iter, cur_step=0):
    global test_epoch_counter # 
    try:
        data = next(dataset_iter)
        if data['class_label'].size(0) < batch_size:
            raise StopIteration
    except StopIteration:
        test_epoch_counter += 1
        # print("[Test DataLoader]: At Epoch %d!" % test_epoch_counter)
        # reset dataset iterator
        dataset_iter = iter(test_dataset)
        # get data for the next iteration
        data = next(dataset_iter)
    sample(data, vertex_model, face_model, cur_step=cur_step)
    return dataset_iter


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

    with torch.no_grad():

        for k in test_data:
          test_data[k] = test_data[k].cuda(non_blocking=True)
        # print(f"key: {k}, shape: {data[k].size()}")
        # construct sampling context
        # sample_context = {
        #     'class_label': torch.full(size=(num_samples,), fill_value=class_label, dtype=torch.long).cuda()
        # }
        # perform vertex sampling # vertex model sampling
        vertex_sample_dict = vertex_model.sample(bsz, context=test_data, max_sample_length=nn_max_vertices, temperature=1., top_k=0, top_p=1.0, recenter_verts=True, only_return_complete=False)

        # class_label
        # recenter_verts
        # vertex_sample_dict['class_label'] = sample_context['class_label'] # class labels

        # perform face sampling
        face_sample_dict = face_model.sample(vertex_sample_dict, max_sample_length=nn_max_faces, temperature=1., top_k=0,  top_p=1.0,  only_return_complete=False)

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
        data_utils.plot_sampled_meshes_single_part_for_pretraining(v_sample=vertex_sample_dict, f_sample=face_sample_dict, context=test_data_np, sv_mesh_folder=sv_mesh_folder, cur_step=cur_step, predict_joint=predict_joint) # predict joint variable...
        # except:
        #     pass

def save_ckpts(vertex_model, face_model, cur_step):
    vertex_model_sv_folder = os.path.join(sv_ckpts_folder, "vertex_model")
    os.makedirs(vertex_model_sv_folder, exist_ok=True)
    face_model_sv_folder = os.path.join(sv_ckpts_folder, "face_model")
    os.makedirs(face_model_sv_folder, exist_ok=True)
    model_sv_path = f'{cur_step}.pth'
    vertex_model_sv_pth = os.path.join(vertex_model_sv_folder, model_sv_path)
    vertex_model_params = vertex_model.state_dict()
    torch.save(vertex_model_params, vertex_model_sv_pth)

    # model_sv_path = f'{cur_step}.pth'
    face_model_sv_pth = os.path.join(face_model_sv_folder, model_sv_path)
    face_model_params = face_model.state_dict()
    torch.save(face_model_params, face_model_sv_pth) # face model path...
    print(f"Vertex model at step {cur_step} saved to {vertex_model_sv_pth}.")
    print(f"Face model at step {cur_step} saved to {face_model_sv_pth}.")
    
    


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
            step_sample(test_dataset, vertex_model, face_model, test_dataset_iter)
            save_ckpts(vertex_model, face_model, cur_step=i)


# train iter
def sample_iter(test_dataset, vertex_model, face_model, sampling_steps):
    # dataset_iter
    # iter dataset
    # dataset_iter = iter(test_dataset)
    test_dataset_iter = iter(test_dataset)
    for i in range(sampling_steps):
        step_sample(test_dataset, vertex_model, face_model, test_dataset_iter, cur_step=i)




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
    mask_low_ratio = opt.dataset.mask_low_ratio
    mask_high_ratio = opt.dataset.mask_high_ratio
    mask_vertices_type = opt.dataset.mask_vertices_type
    cross = opt.model.cross
    # cross = opt.model.cross
    model_descriptor = f"pretraining_cross_{cross}_{motion_dataset_nm}_cat_{motion_cat}_bsz_{batch_size}_max_verts_{nn_max_vertices}_faces_{nn_max_faces}_max_permit_verts_{nn_max_permite_vertices}_faces_{nn_max_permite_faces}_mask_low_{mask_low_ratio}_high_{mask_high_ratio}_mask_{mask_vertices_type}"
    return model_descriptor


def resume_from_ckpts(vertex_model, face_model, vertex_model_path, face_model_path):
    vertex_model_state_dicts = torch.load(vertex_model_path, map_location="cpu")
    face_model_state_dicts = torch.load(face_model_path, map_location="cpu")

    vertex_model.load_state_dict(vertex_model_state_dicts)
    face_model.load_state_dict(face_model_state_dicts)
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
    

    root_folder = os.path.join(dataset_root_path, motion_dataset_nm, motion_cat)
    
    # process recenter mesh # recenter mesh
    # process_recenter_mesh = True

    recenter_mesh = opt.model.recenter_mesh
    process_recenter_mesh = opt.model.recenter_mesh

    # quantization_bits = 8
    quantization_bits = opt.model.quantization_bits

    # n_max_instances = 19 * 50
    # n_max_instances = 4
    # n_max_instances = 2 * 50

    # inter_part_auto_regressive = False # 
    inter_part_auto_regressive = opt.vertex_model.inter_part_auto_regressive
    # predict_joint = False
    predict_joint = opt.vertex_model.predict_joint
    # batch_size = 8
    batch_size = opt.loss.batch_size
    # batch_size = 4 # 8
    # batch_size = 2
    # batch_size = 1
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

    model_descriptor = get_model_descriptor()

    if motion_cat == "eyeglasses":
      #### for eyegalsse ####
      # train_ins_nns = ["%.4d" % iii for iii in [3, 8, 23, 13, 6]]
      train_ins_nns = [ "%.4d" % iii for iii in [3, 9, 4, 21]]
      test_ins_nns = ["%.4d" % iii for iii in [1, 2]]
      #### for eyeglasses ####
    else:
      #### for polygen samples ###
      train_ins_nns = [str(iii) for iii in range(3000)]
      test_ins_nns = [str(iii) for iii in range(3000, 4001)]

    dataset_train = URDFDataset(root_folder=root_folder, quantization_bits=quantization_bits, nn_max_vertices=nn_max_vertices, nn_max_faces=nn_max_faces, nn_max_permite_vertices=nn_max_permite_vertices, nn_max_permite_faces=nn_max_permite_faces, category_name=motion_cat, instance_nns=train_ins_nns)

    # shuffle = False
    # dataet for train...
    dataset_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=num_thread)

    
    if opt.model.cross: # sampling for 
        # set dataset name and dataset category for cross-category testing
        opt.dataset.dataset_name = "MotionDataset_processed_deci_scaled"
        opt.dataset.category = "eyeglasses"
        motion_cat = "eyeglasses"
        test_ins_nns = ["%.4d" % iii for iii in [1, 2]]
        root_folder = os.path.join(dataset_root_path, opt.dataset.dataset_name, opt.dataset.category)

    dataset_test = URDFDataset(root_folder=root_folder, quantization_bits=quantization_bits, nn_max_vertices=nn_max_vertices, nn_max_faces=nn_max_faces, nn_max_permite_vertices=nn_max_permite_vertices, nn_max_permite_faces=nn_max_permite_faces, category_name=motion_cat, instance_nns=test_ins_nns)

    # shuffle = False
    # dataet for train...
    dataset_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_thread)


    ##### Model configs ##### # 
    vertex_module_config, face_module_config = get_model_configs()
    vertex_model = VertexModel(**vertex_module_config)
    face_model = FaceModel(**face_module_config)

    # vertex_model_path = opt.model.vertex_model_path
    # face_model_path = opt.model.face_model_path
    # resume_from_ckpts(vertex_model, face_model, vertex_model_path, face_model_path)


    vertex_model = vertex_model.cuda()
    face_model = face_model.cuda()

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
    #### Setup optimizers ####

    # if os.path.exists("/nas/datasets/gen"):
    if opt.dataset.lab == "thu": # save samples
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

    # sampling_steps = 100
    # sample_iter(dataset_test, vertex_model, face_model, sampling_steps)
    # train iters...
    train_iter(dataset_train, dataset_test, vertex_model, face_model, vertex_optimizer, face_optimizer, training_steps, batch_size)
