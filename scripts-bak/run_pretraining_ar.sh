#!/bin/sh
# Copyright 2020 Deepmind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#### python 
# python3 -m venv polygen
# source polygen/bin/activate
# pip3 install .
# python3 model_test.py
# deactivate

# eval $(curl -s deploy.i.brainpp.cn/httpproxy)
# export PATH=/data/cuda/cuda-11.2/cuda/bin:$PATH
# export LD_LIBRARY_PATH=/data/cuda/cuda-11.2/cuda/lib64:/data/cuda/cuda-11.2/cudnn/v8.1.0/lib64:$LD_LIBRARY_PATH
# export CUDA_HOME=/data/cuda/cuda-11.2/cuda
# export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME

# export all_proxy=http://proxy.i.brainpp.cn:3128
# export no_proxy=.brainpp.cn,.brainpp.ml,.megvii-inc.com,.megvii-op.org,127.0.0.1,localhost
# export http_proxy=$all_proxy https_proxy=$all_proxy

export PYTHONPATH=.

# ### tabel, PolyGen Samples ####
# CUDA_VISIBLE_DEVICES=7 python trainer/pretraining_1_xl.py  --use-light-weight=True  --max-permit-vertices=400 --max-permit-faces=2000 --dataset-name="PolyGen_Samples" --category="04379243" --recenter-mesh=True --apply-random-shift=True --dataset-root-path='/nas/datasets/gen/datasets' --max-vertices=400 --max-faces=2000 --batch-size=16  --mask-low-ratio=0.30 --mask-high-ratio=0.50  --check-step=200 --cus-perm-xl=True --cross=True # --debug=True 

# ### small model ###
# # MotionDataset_processed_deci_part_first_30_normalized_reform
# CUDA_VISIBLE_DEVICES=1 python trainer/pretraining_ar.py  --exp-mode="pretraining"  --max-permit-vertices=400 --max-permit-faces=2000 --dataset-name="MotionDataset_processed_voxelized_part_first" --category="eyeglasses" --recenter-mesh=True --dataset-root-path='/nas/datasets/gen/datasets' --max-vertices=400 --max-faces=2000 --batch-size=2  --check-step=2000 --exp-flag="some_parts_v9" --num-classes=120 --nn-vertices-predict-ratio=1.0  --context-window=750    --not-remove-du=True --category-part-indicator=True  # --debug=True

#### small context windwo 
# CUDA_VISIBLE_DEVICES=6 python trainer/pretraining_ar.py  --exp-mode="pretraining"  --max-permit-vertices=400 --max-permit-faces=2000 --dataset-name="MotionDataset_processed_voxelized_part_first" --category="eyeglasses" --recenter-mesh=True --dataset-root-path='/nas/datasets/gen/datasets' --max-vertices=400 --max-faces=2000 --batch-size=2  --check-step=2000 --exp-flag="some_parts_v0" --num-classes=120 --nn-vertices-predict-ratio=1.0  --context-window=600    --not-remove-du=True # --category-part-indicator=True  # --debug=True

# CUDA_VISIBLE_DEVICES=1 python trainer/pretraining_ar.py  --exp-mode="pretraining"  --max-permit-vertices=600 --max-permit-faces=3000 --dataset-name="MotionDataset_processed_voxelized_part_first" --category="eyeglasses" --recenter-mesh=True --dataset-root-path='/nas/datasets/gen/datasets' --max-vertices=600 --max-faces=3000 --batch-size=1  --check-step=2000 --exp-flag="some_parts_v0" --num-classes=120 --nn-vertices-predict-ratio=1.0  --context-window=600    --not-remove-du=True # --category-part-indicator=True  # --debug=True

# export DATASET_NAME="screw_1"
# export DATASET_NAME="screw_1_planar"
# export DATASET_NAME="screw_2"
export DATA_ROOT_PATH="/nas/datasets/gen/datasets/bsp_abstraction" #  "/nas/datasets/gen/datasets/subd_mesh"
# export DATASET_NAME="eyeglasses_1_merged" # summary
export DATASET_NAME="eyeglasses"
export EXP_FLAG="test_scale_simple"
export CONTEXT_WINDOW=610 # 800 # 600 # 3000 # 800 # 2410 # 500 # 350 # 400 #500
export MAX_NUM_GRIDS=3000 # 3000 # 800 # 2410 # 500 # 400 # 800 # 400 #500
export MAX_VERTICES=610 # 800 # 600
export MAX_FACES=3000


CUDA_VISIBLE_DEVICES=3  python trainer/pretraining_ar.py  --exp-mode="pretraining"  --max-permit-vertices=${MAX_VERTICES} --max-permit-faces=${MAX_FACES} --dataset-name=${DATASET_NAME} --category="eyeglasses" --recenter-mesh=True --dataset-root-path=${DATA_ROOT_PATH} --max-vertices=${MAX_VERTICES} --max-faces=${MAX_FACES} --batch-size=1  --check-step=2000 --exp-flag=${EXP_FLAG} --num-classes=120 --nn-vertices-predict-ratio=1.0  --context-window=600    --not-remove-du=True  --apply-random-scaling=0  --category-part-indicator=True  # --debug=True


# run pretraining ar;
# we can still use xyz for grid position regresssion;
# xyz for grid position regression


# --vertex-model-path="" --face-model-path=""
# --use-eyeglasses-frame=True
# --context-window=900
# --use-eyeglasses-frame=True
# --category-part-indicator=True


#### predict ratio = 0.5
# --vertex-model-path="/nas/datasets/gen/ckpts/pretraining_ar_some_parts_v0_cond_cat_part_indi_True_face_cond_False_light_False_spec_False_vpred_0.5_shift_False_scale_0_MotionDataset_processed_deci_part_first_30_normalized_bsz_2_max_verts_400_faces_2000_max_permit_verts_400_faces_2000/vertex_model/6600.pth" --face-model-path="/nas/datasets/gen/ckpts/pretraining_ar_some_parts_v0_cond_cat_part_indi_True_face_cond_False_light_False_spec_False_vpred_0.5_shift_False_scale_0_MotionDataset_processed_deci_part_first_30_normalized_bsz_2_max_verts_400_faces_2000_max_permit_verts_400_faces_2000/face_model/6600.pth"  # --debug=True
# --debug=True --category-part-indicator=True --use-light-weight=True 
# --num-classes=177 --category-part-indicator=True # --debug=True
# --debug=Trues
# --cus-perm-xl=True # --cross=True # --debug=True 
# --apply-random-shift=True 
# --face-class-conditional=True 
# --apply-random-scaling=1 



# # ### small model ###
# CUDA_VISIBLE_DEVICES=4 python trainer/pretraining_ar.py  --exp-mode="pretraining"  --max-permit-vertices=400 --max-permit-faces=2000 --dataset-name="MotionDataset_processed_deci_part_first_0.2_normalized" --category="eyeglasses" --recenter-mesh=True --dataset-root-path='/nas/datasets/gen/datasets' --max-vertices=400 --max-faces=2000 --batch-size=8  --check-step=200 --exp-flag="some_parts_v0_coarse" --num-classes=120 --specify-light-weight=True  # --debug=True




# 
##### table polygen ssamples #####
# --recenter-mesh=True --memory-efficient=True --apply-random-shift=True --batch-size=1 --sv-samples-dir='./samples/tables_small_net_less_permit' --dataset-name="PolyGen_Samples" --category="04379243" --max-permit-vertices=800 --max-permit-faces=2800 --apply-random-shift=True 


# CUDA_VISIBLE_DEVICES=4 python trainer/pretraining_1_xl.py  --use-light-weight=True  --max-permit-vertices=400 --max-permit-faces=2000 --dataset-name="MotionDataset_processed_deci_scaled" --category="eyeglasses" --recenter-mesh=True --apply-random-shift=True --dataset-root-path='/nas/datasets/gen/datasets' --max-vertices=400 --max-faces=2000 --batch-size=16  --mask-low-ratio=0.30 --mask-high-ratio=0.50  --check-step=200 --cus-perm-xl=True --cross=True 