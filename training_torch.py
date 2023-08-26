# from glob import glob
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
import data_utils_torch as data_utils
from urdf_dataset import URDFDatasetTwoParts
from modules_torch import VertexModelPart, FaceModelPart


from tqdm import tqdm
import time
import torch.optim as optim


epoch_counter = 0
use_cuda = True
step_counter = 0

nn_max_vertices = 2000
nn_max_faces = 2000


# distribution of the vertex and distribution of 

def optimize(data, vertex_model, face_model, vertex_optimizer, face_optimizer):
    global  use_cuda
    global step_counter

    vertex_model.train()
    face_model.train()

    # left_vertices = data['left_vertices'].cuda(non_blocking=True)
    # rgt_vertices = data['rgt_vertices'].cuda(non_blocking=True)
    # left_faces = data['left_faces'].cuda(non_blocking=True)
    # rgt_faces = data['rgt_faces'].cuda(non_blocking=True)
    # class_label = data['class_label'].cuda(non_blocking=True)
    # dir = data['dir'].cuda(non_blocking=True)
    # pvp = data['pvp'].cuda(non_blocking=True)

    # todo: vertex flat?

    for k in data:
        data[k] = data[k].cuda(non_blocking=True)
        # print(f"key: {k}, shape: {data[k].size()}")

    # vertex_model_pred_dict
    vertex_model_pred_dict = vertex_model(data)
    #
    face_model_pred_dict = face_model(data)

    # print(f"")
    # print(f"vertex_model_pred_dict, left: {vertex_model_pred_dict['left']}, right: {}")

    vertex_prediction_loss = -torch.sum(
        vertex_model_pred_dict['left'].log_prob(data['left_vertices_flat']) * data['left_vertices_flat_mask'], dim=-1
    ) - torch.sum(
        vertex_model_pred_dict['rgt'].log_prob(data['rgt_vertices_flat']) * data['rgt_vertices_flat_mask'], dim=-1
    )
    vertex_prediction_loss = torch.mean(vertex_prediction_loss) #


    face_prediction_loss = -torch.sum(
        face_model_pred_dict['left'].log_prob(data['left_faces']) * data['left_faces_mask'], dim=-1
    ) - torch.sum(
        face_model_pred_dict['rgt'].log_prob(data['rgt_faces']) * data['rgt_faces_mask'], dim=-1
    )
    face_prediction_loss = torch.mean(face_prediction_loss)

    loss = vertex_prediction_loss # + face_prediction_loss
    face_optimizer.zero_grad()
    vertex_optimizer.zero_grad()
    loss.backward()
    face_optimizer.step()
    vertex_optimizer.step()

    print(f"Step {step_counter}")
    print(f"Vertex Loss {vertex_prediction_loss.item()}")
    print(f"Face Loss {face_prediction_loss.item()}")
    step_counter += 1

    return



def step(train_dataset, vertex_model, face_model, vertex_optimizer, face_optimizer, batch_size, dataset_iter):
    # epoch counter

    global epoch_counter
    try:
        data = next(dataset_iter)
        if data['class_label'].size(0) < batch_size:
            raise StopIteration
    except StopIteration:
        epoch_counter += 1
        print("[DataLoader]: At Epoch %d!" % epoch_counter)
        # reset dataset iterator
        dataset_iter = iter(train_dataset)
        # get data for the next iteration
        data = next(dataset_iter)
    optimize(data, vertex_model, face_model, vertex_optimizer, face_optimizer)
    return dataset_iter


def sample(num_samples, vertex_model, face_model, class_label, cur_step=0):
    global nn_max_vertices
    global nn_max_faces

    vertex_model.eval()
    face_model.eval()

    with torch.no_grad():
        # construct sampling context
        sample_context = {
            'class_label': torch.full(size=(num_samples,), fill_value=class_label, dtype=torch.long).cuda()
        }
        # perform vertex sampling #
        vertex_sample_dict = vertex_model.sample(num_samples, context=sample_context, max_sample_length=nn_max_vertices // 3, temperature=1., top_k=0, top_p=0.95, recenter_verts=False, only_return_complete=False)

        # class_label
        
        vertex_sample_dict['class_label'] = sample_context['class_label']
        vertex_sample_dict['left']['class_label'] = sample_context['class_label']
        vertex_sample_dict['rgt']['class_label'] = sample_context['class_label']

        # perform face sampling #
        face_sample_dict = face_model.sample(vertex_sample_dict, max_sample_length=nn_max_faces, temperature=1., top_k=0,  top_p=0.95,  only_return_complete=False)

        ### convert torch tensors to numpy values ###
        for k in vertex_sample_dict:
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
        
        ### plot sampled meshes to the obj file ###
        # plot sampled 
        try:
            data_utils.plot_sampled_meshes(v_sample=vertex_sample_dict, f_sample=face_sample_dict, sv_mesh_folder="./samples/eyeglasses/meshes/", cur_step=cur_step, predict_joint=predict_joint) # predict joint variable...
        except:
            pass



    


def train_iter(train_dataset, vertex_model, face_model, vertex_optimizer, face_optimizer, training_steps, batch_size):
    # dataset_iter
    dataset_iter = iter(train_dataset)
    for i in range(training_steps):
        dataset_iter = step(train_dataset, vertex_model, face_model, vertex_optimizer, face_optimizer, batch_size, dataset_iter)
        if i % plot_sample_step == 0:
            sample(num_samples=2, vertex_model=vertex_model, face_model=face_model, class_label=0, cur_step=i)

        


# todo: add config files
if __name__=='__main__':
    # global nn_max_faces
    # global nn_max_vertices

    motion_dataset_nm = "MotionDataset"
    # motion_dataset_nm
    motion_cat = "eyeglasses"
    # flag = "remerge_2"
    # flag = "remerge_rnd_deci_2"
    # flag = "remerge_rnd_deci_2_permuted"
    flag = "remerge_scaled_2_permuted"

    # set part name for this category
    part_names = ["dof_rootd_Aa001_r", "none_motion"]

    # set root folder
    # root folder and statistic folder
    root_folder = f"./data/{motion_dataset_nm}/{motion_cat}_{flag}"
    statistic_folder = f"./data/{motion_dataset_nm}/{motion_cat}_remerge/statistics"
    
    motion_dataset_nm = "toyexamples"
    motion_cat = "two_pieces_mesh"
    flag = "100_part"
    part_names = ["two_pieces_mesh_100_part_1", "two_pieces_mesh_100_part_2"]
    root_folder = f"./data/{motion_dataset_nm}/{motion_cat}_{flag}"

    # process recenter mesh
    process_recenter_mesh = False

    quantization_bits = 8

    # n_max_instances = 19 * 50
    # n_max_instances = 4
    n_max_instances = 2 * 50

    inter_part_auto_regressive = False
    predict_joint = False #
    # batch_size = 4
    batch_size = 2
    # batch_size = 1
    num_thread = 10
    plot_sample_step = 50

    dataset_train = URDFDatasetTwoParts(root_folder=root_folder, part_names=part_names, statistic_folder=statistic_folder, n_max_instance=n_max_instances, quantization_bits=quantization_bits, nn_max_vertices=nn_max_vertices, nn_max_faces=nn_max_faces, category_name=motion_cat)

    # shuffle = False...
    dataset_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=num_thread)

    #### Setup models: vertex model and face model ####
    vertex_model = VertexModelPart(
        encoder_config={
            'hidden_size': 128,
            'fc_size': 512,
            'num_layers': 3,
            'dropout_rate': 0.,
            're_zero': True
        },
        decoder_config={
            'hidden_size': 128,
            'fc_size': 512,
            'num_layers': 3,
            'dropout_rate': 0.,
            're_zero': True
        },
        quantization_bits=quantization_bits,
        class_conditional=True,
        num_classes=55,
        max_num_input_verts=nn_max_vertices,
        use_discrete_embeddings=True,
        inter_part_auto_regressive=inter_part_auto_regressive,
        predict_joint=predict_joint
    )

    face_model = FaceModelPart(
        encoder_config={
            'hidden_size': 128,
            'fc_size': 512,
            'num_layers': 3,
            'dropout_rate': 0.,
            're_zero': True
        },
        decoder_config={
            'hidden_size': 128,
            'fc_size': 512,
            'num_layers': 3,
            'dropout_rate': 0.,
            're_zero': True
        },
        class_conditional=True,
        num_classes=55,
        decoder_cross_attention=True,
        use_discrete_vertex_embeddings=True,
        quantization_bits=quantization_bits,
        max_seq_length=nn_max_faces # 
    )

    vertex_model = vertex_model.cuda()
    face_model = face_model.cuda()

    # Optimization settings
    learning_rate = 5e-4
    training_steps = 2000 * 5
    check_step = 5

    #### Setup optimizers ####
    vertex_optimizer = optim.Adam(vertex_model.parameters(), lr=learning_rate)
    face_optimizer = optim.Adam(face_model.parameters(), lr=learning_rate)
    #### Setup optimizers ####

    train_iter(dataset_train, vertex_model, face_model, vertex_optimizer, face_optimizer, training_steps, batch_size)


