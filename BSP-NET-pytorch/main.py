import os
import numpy as np

from modelAE import BSP_AE
from modelSVR import BSP_SVR

import argparse
import h5py

parser = argparse.ArgumentParser()
parser.add_argument("--phase", action="store", dest="phase", default=1, type=int, help="phase 0 = continuous, phase 1 = hard discrete, phase 2 = hard discrete with L_overlap, phase 3 = soft discrete, phase 4 = soft discrete with L_overlap [1]")
#phase 0 continuous for better convergence
#phase 1 hard discrete for bsp
#phase 2 hard discrete for bsp with L_overlap
#phase 3 soft discrete for bsp
#phase 4 soft discrete for bsp with L_overlap
#use [phase 0 -> phase 1] or [phase 0 -> phase 2] or [phase 0 -> phase 3] or [phase 0 -> phase 4]
parser.add_argument("--epoch", action="store", dest="epoch", default=0, type=int, help="Epoch to train [0]")
parser.add_argument("--iteration", action="store", dest="iteration", default=0, type=int, help="Iteration to train. Either epoch or iteration need to be zero [0]")
parser.add_argument("--learning_rate", action="store", dest="learning_rate", default=0.0001, type=float, help="Learning rate for adam [0.0001]")
parser.add_argument("--beta1", action="store", dest="beta1", default=0.5, type=float, help="Momentum term of adam [0.5]")
parser.add_argument("--dataset", action="store", dest="dataset", default="all_vox256_img", help="The name of dataset")
parser.add_argument("--checkpoint_dir", action="store", dest="checkpoint_dir", default="checkpoint", help="Directory name to save the checkpoints [checkpoint]")
parser.add_argument("--data_dir", action="store", dest="data_dir", default="./data/all_vox256_img/", help="Root directory of dataset [data]")
parser.add_argument("--sample_dir", action="store", dest="sample_dir", default="./samples/", help="Directory name to save the image samples [samples]")
parser.add_argument("--sample_vox_size", action="store", dest="sample_vox_size", default=64, type=int, help="Voxel resolution for coarse-to-fine training [64]")
parser.add_argument("--batch_size", action="store", dest="batch_size", default=24, type=int, help="Voxel resolution for coarse-to-fine training [64]")
parser.add_argument("--train", action="store_true", dest="train", default=False, help="True for training, False for testing [False]")
parser.add_argument("--start", action="store", dest="start", default=0, type=int, help="In testing, output shapes [start:end]")
parser.add_argument("--end", action="store", dest="end", default=16, type=int, help="In testing, output shapes [start:end]")
parser.add_argument("--ae", action="store_true", dest="ae", default=False, help="True for ae [False]")
parser.add_argument("--svr", action="store_true", dest="svr", default=False, help="True for svr [False]")
parser.add_argument("--getz", action="store_true", dest="getz", default=False, help="True for getting latent codes [False]")
parser.add_argument("--gpu", action="store", dest="gpu", default="0", help="Which GPU to use [0]")


parser.add_argument("--c_dim", action="store", dest="c_dim", default=256, type=int, help="Voxel resolution for coarse-to-fine training [64]")

parser.add_argument("--model_flag", action="store", dest="model_flag", default="", help="Directory name to save the image samples [samples]")

# parser.add_argument("--sample_dir", action="store", dest="sample_dir", default="./samples/", help="Directory name to save the image samples [samples]")

parser.add_argument("--sample_flag", action="store", dest="sample_flag", default="regular", help="Directory name to save the image samples [samples]")

FLAGS = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu



if not os.path.exists(FLAGS.sample_dir):
	os.makedirs(FLAGS.sample_dir, exist_ok=True)
 
# FLAGS.sample_dir = os.path.join(FLAGS.sample_dir, )

FLAGS.checkpoint_dir = "/data2/sim/ckpts/BSP-Net/"
FLAGS.sample_dir = "/data2/sim/samples/BSP-Net/"
FLAGS.data_dir = "/data2/sim/smplh_data_gathered"
FLAGS.dataset = "female"
FLAGS.model_flag = "cdim_512_female"

### get sample dir, data dir, checkpoint dir ###
# export SAMPLE_DIR=/data2/sim/samples/BSP-Net/
# export DATA_DIR=/data2/sim/smplh_data_gathered # data dir  # gather the smpl data ##
# export CHECKPOINT_DIR=/data2/sim/ckpts/BSP-Net ## category dir

# export BATCH_SIZE=2
# export C_DIM=512
# export CATEGORY=female
# export DATASET=${CATEGORY}

# export MODEL_FLAG=cdim_${C_DIM}_${CATEGORY} ### 


FLAGS.c_dim = 512
if not os.path.exists(FLAGS.checkpoint_dir):
  os.makedirs(FLAGS.checkpoint_dir, exist_ok=True)

# curr_model_ckpt_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.model_dir)
# os.makedirs(curr_model_ckpt_dir, exist_ok=True)


if FLAGS.ae:
	bsp_ae = BSP_AE(FLAGS)
 
	model_dir = bsp_ae.model_dir
 
	curr_model_ckpt_dir = os.path.join(FLAGS.checkpoint_dir, model_dir)
	os.makedirs(curr_model_ckpt_dir, exist_ok=True)



	if FLAGS.train:
		bsp_ae.train(FLAGS)
	elif FLAGS.getz:
		bsp_ae.get_z(FLAGS)
	else:
		if FLAGS.phase==0:
			bsp_ae.test_mesh_obj_material(FLAGS)
			# bsp_ae.test_mesh_point(FLAGS)
			# bsp_ae.test_dae3(FLAGS)
		else:
			#bsp_ae.test_bsp(FLAGS)
			# bsp_ae.test_mesh_point(FLAGS)
			# if FLAGS.sample_flag == "cvx":
			print(f"sample_flag: {FLAGS.sample_flag} for cvx decompostion")
			bsp_ae.test_mesh_obj_material_2_for_cvx_gen(FLAGS)
			# elif FLAGS.sample_flag == "deform":
			# 	bsp_ae.test_mesh_obj_material_2_for_deform(FLAGS)
			# else:
			# 	bsp_ae.test_mesh_obj_material_2(FLAGS)
elif FLAGS.svr:
	bsp_svr = BSP_SVR(FLAGS)

	if FLAGS.train:
		bsp_svr.train(FLAGS)
	else:
		#bsp_svr.test_bsp(FLAGS)
		bsp_svr.test_mesh_point(FLAGS)
		#bsp_svr.test_mesh_obj_material(FLAGS)
else:
	print("Please specify an operation: ae or svr?")
