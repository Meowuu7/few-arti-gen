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


export DATASET_NAME="screw_2"
export EXP_FLAG="test_convex_3_10000"
export CONTEXT_WINDOW=600 # 3000 # 800 # 2410 # 500 # 350 # 400 #500
# export MAX_NUM_GRIDS=3000 # 3000 # 800 # 2410 # 500 # 400 # 800 # 400 #500

# export MAX_VERTICES=600
# export MAX_FACES=3000

export MAX_VERTICES=500 # 800 # 600 ## 400
export MAX_FACES=800 #2000 # 3000 ## 2000


# CUDA_VISIBLE_DEVICES=3  python trainer/pretraining_ar.py  --exp-mode="pretraining"  --max-permit-vertices=${MAX_VERTICES} --max-permit-faces=${MAX_FACES} --dataset-name=${DATASET_NAME} --category="eyeglasses" --recenter-mesh=True --dataset-root-path='/nas/datasets/gen/datasets/subd_mesh' --max-vertices=${MAX_VERTICES} --max-faces=${MAX_FACES} --batch-size=1  --check-step=2000 --exp-flag=${EXP_FLAG} --num-classes=120 --nn-vertices-predict-ratio=1.0  --context-window=600    --not-remove-du=True # --apply-random-scaling=1  # --category-part-indicator=True  # --debug=True


# MotionDataset_processed_deci_part_first_30
# ### tabel, PolyGen Samples #### # polygen samples
CUDA_VISIBLE_DEVICES=2 python trainer/sampling_ar.py  --exp-mode="sampling"  --max-permit-vertices=${MAX_VERTICES} --max-permit-faces=${MAX_FACES} --dataset-name=${DATASET_NAME} --category="eyeglasses" --recenter-mesh=True  --dataset-root-path='/nas/datasets/gen/datasets/subd_mesh' --max-vertices=${MAX_VERTICES} --max-faces=${MAX_FACES} --batch-size=1  --check-step=200  --vertex-model-path="/nas/datasets/gen/ckpts/pretraining_ar_test_cvx_merged_1_eye_bsp_5_cat_part_indi_True_face_cond_False_light_False_shift_False_scale_0_eyeglasses_bsz_1_max_verts_500_faces_800_max_permit_verts_500_faces_800/vertex_model/10000.pth" --face-model-path="/nas/datasets/gen/ckpts/pretraining_ar_test_cvx_merged_1_eye_bsp_5_cat_part_indi_True_face_cond_False_light_False_shift_False_scale_0_eyeglasses_bsz_1_max_verts_500_faces_800_max_permit_verts_500_faces_800/face_model/10000.pth"  --exp-flag="convex_3" --num-classes=120 --category-part-indicator=True  --sample-class-idx=0 --context-window=${CONTEXT_WINDOW}  --exp-flag=${EXP_FLAG}  --not-remove-du=True  --apply-random-scaling=0 --prefix-key-len=256 --prefix-value-len=256 --training-steps=400 # --beam-search=True


# export MAX_VERTICES=500 # 800 # 600 ## 400
# export MAX_FACES=800 #2000 # 3000 ## 2000


# # ### small model ###
# CUDA_VISIBLE_DEVICES=4,5,6,7 TORCH_DISTRIBUTED_DEBUG=INFO python -m torch.distributed.launch --nproc_per_node=4  \
# trainer/pretraining_ar_ddp.py  --exp-mode="pretraining"  --max-permit-vertices=${MAX_VERTICES} --max-permit-faces=${MAX_FACES} --dataset-name=${DATASET_NAME} --category="eyeglasses" --recenter-mesh=True --dataset-root-path=${DATA_ROOT_PATH} --max-vertices=${MAX_VERTICES} --max-faces=${MAX_FACES} --batch-size=1  --check-step=200 --exp-flag=${EXP_FLAG} --num-classes=120 --category-part-indicator=True --apply-random-scaling=0 --prefix-key-len=256 --prefix-value-len=256


# --apply-random-shift=True --category-part-indicator=True 
# --use-light-weight=True 
# --context-window=750
###### 

# /nas/datasets/gen/ckpts/pretraining_ar_test_scale_simple_fr_False_indi_False_ctx_600_face_cond_False_light_False_spec_False_vpred_1.0_scale_0_screw_2_bsz_1_max_verts_600_faces_3000_max_permit_verts_600_faces_3000/vertex_model/430000.pth.
# Face model at step 430000 saved to /nas/datasets/gen/ckpts/pretraining_ar_test_scale_simple_fr_False_indi_False_ctx_600_face_cond_False_light_False_spec_False_vpred_1.0_scale_0_screw_2_bsz_1_max_verts_600_faces_3000_max_permit_verts_600_faces_3000/face_model/430000.pth.
