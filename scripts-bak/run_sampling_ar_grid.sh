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


# MotionDataset_processed_deci_part_first_30
# ### tabel, PolyGen Samples #### # polygen samples
CUDA_VISIBLE_DEVICES=1 python trainer/sampling_ar_grid.py  --exp-mode="sampling"  --max-permit-vertices=400 --max-permit-faces=2000 --dataset-name="MotionDataset_processed_voxelized_part_first" --category="eyeglasses" --recenter-mesh=True  --dataset-root-path='/nas/datasets/gen/datasets' --max-vertices=400 --max-faces=2000 --batch-size=2  --check-step=200  --vertex-model-path="/nas/datasets/gen/ckpts/pretraining_ar_grid_some_parts_v0_2d_fr_False_indi_True_ctx_600_face_cond_False_light_False_spec_False_vpred_1.0_scale_0_MotionDataset_processed_voxelized_part_first_bsz_2_max_verts_400_faces_2000_max_permit_verts_400_faces_2000/vertex_model/4000.pth" --exp-flag="v9_s" --num-classes=120 --category-part-indicator=True  --sample-class-idx=0   --not-remove-du=True --max-num-grids=400 --grid-size=4  # --beam-search=True

# --apply-random-shift=True --category-part-indicator=True 
# --use-light-weight=True 
# --context-window=750
###### 
