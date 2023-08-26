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


# export DATASET_NAME="screw_1_planar"
# export DATASET_NAME="screw_2"
# export DATASET_NAME="eyeglasses_1_merged"
export DATASET_NAME="eyeglasses"
export DATASET_NAME="scissors"
export EXP_FLAG="test_cvx_merged_1_sci_bsp_5"
# export DATA_ROOT_PATH="/nas/datasets/gen/datasets/subd_mesh"
# export DATA_ROOT_PATH="/nas/datasets/gen/datasets/subd_mesh"
# export DATA_ROOT_PATH="/nas/datasets/gen/datasets/bsp_abstraction_2"
# export DATA_ROOT_PATH="/nas/datasets/gen/datasets/bsp_abstraction_3"
# export DATA_ROOT_PATH="/nas/datasets/gen/datasets/bsp_abstraction_4"
# export DATA_ROOT_PATH="/nas/datasets/gen/datasets/bsp_abstraction_5"
export DATA_ROOT_PATH="/share/xueyi/datasets/gen/datasets/bsp_abstraction_5"
export MAX_VERTICES=400 # 800 # 600 ## 400
export MAX_FACES=700 #2000 # 3000 ## 2000


# ### small model ###
CUDA_VISIBLE_DEVICES=3,4,5,6,7 TORCH_DISTRIBUTED_DEBUG=INFO python -m torch.distributed.launch --nproc_per_node=5  \
trainer/pretraining_ar_ddp.py  --exp-mode="pretraining"  --max-permit-vertices=${MAX_VERTICES} --max-permit-faces=${MAX_FACES} --dataset-name=${DATASET_NAME} --category="eyeglasses" --recenter-mesh=True --dataset-root-path=${DATA_ROOT_PATH} --max-vertices=${MAX_VERTICES} --max-faces=${MAX_FACES} --batch-size=1  --check-step=200 --exp-flag=${EXP_FLAG} --num-classes=120 --category-part-indicator=True --apply-random-scaling=0 --prefix-key-len=256 --prefix-value-len=256
# --debug=True --category-part-indicator=True --use-light-weight=True 
# --num-classes=177 --category-part-indicator=True # --debug=True
# --debug=Trues
# --cus-perm-xl=True # --cross=True # --debug=True 
# --apply-random-shift=True 
# --face-class-conditional=True 
# --apply-random-scaling=1 




# 
##### table polygen ssamples #####
# --recenter-mesh=True --memory-efficient=True --apply-random-shift=True --batch-size=1 --sv-samples-dir='./samples/tables_small_net_less_permit' --dataset-name="PolyGen_Samples" --category="04379243" --max-permit-vertices=800 --max-permit-faces=2800 --apply-random-shift=True 


# CUDA_VISIBLE_DEVICES=4 python trainer/pretraining_1_xl.py  --use-light-weight=True  --max-permit-vertices=400 --max-permit-faces=2000 --dataset-name="MotionDataset_processed_deci_scaled" --category="eyeglasses" --recenter-mesh=True --apply-random-shift=True --dataset-root-path='/nas/datasets/gen/datasets' --max-vertices=400 --max-faces=2000 --batch-size=16  --mask-low-ratio=0.30 --mask-high-ratio=0.50  --check-step=200 --cus-perm-xl=True --cross=True 