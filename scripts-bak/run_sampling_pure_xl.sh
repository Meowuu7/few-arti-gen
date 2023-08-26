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

# ### tabel, PolyGen Samples ####
# CUDA_VISIBLE_DEVICES=7 python trainer/pretraining_1_xl.py  --use-light-weight=True  --max-permit-vertices=400 --max-permit-faces=2000 --dataset-name="PolyGen_Samples" --category="04379243" --recenter-mesh=True --apply-random-shift=True --dataset-root-path='/nas/datasets/gen/datasets' --max-vertices=400 --max-faces=2000 --batch-size=16  --mask-low-ratio=0.30 --mask-high-ratio=0.50  --check-step=200 --cus-perm-xl=True --cross=True # --debug=True 
# ### tabel, PolyGen Samples ####

# CUDA_VISIBLE_DEVICES=3 python trainer/pretraining_1_xl.py  --use-light-weight=True  --max-permit-vertices=400 --max-permit-faces=2000 --dataset-name="MotionDataset_processed_deci_scaled" --category="eyeglasses" --recenter-mesh=True --apply-random-shift=True --dataset-root-path='/nas/datasets/gen/datasets' --max-vertices=400 --max-faces=2000 --mask-low-ratio=0.30 --mask-high-ratio=0.50  --batch-size=16 --check-step=200 --cus-perm-xl=True  # --debug=True


# ### tabel, PolyGen Samples ####
# CUDA_VISIBLE_DEVICES=3 python trainer/pretraining_1_xl.py  --use-light-weight=True  --max-permit-vertices=400 --max-permit-faces=2000 --dataset-name="MotionDataset_processed_deci_scaled" --category="oven" --recenter-mesh=True --apply-random-shift=True --dataset-root-path='/nas/datasets/gen/datasets' --max-vertices=400 --max-faces=2000 --batch-size=16  --mask-low-ratio=0.30 --mask-high-ratio=0.50  --check-step=200 --cus-perm-xl=True --cross=True # --debug=True 
# ### tabel, PolyGen Samples ####

# # ### tabel, PolyGen Samples ####
# CUDA_VISIBLE_DEVICES=4 python trainer/pretraining_1_xl.py  --use-light-weight=True  --max-permit-vertices=400 --max-permit-faces=2000 --dataset-name="MotionDataset_processed_deci_scaled" --category="eyeglasses" --recenter-mesh=True --apply-random-shift=True --dataset-root-path='/nas/datasets/gen/datasets' --max-vertices=400 --max-faces=2000 --batch-size=16  --mask-low-ratio=0.30 --mask-high-ratio=0.50  --check-step=200 --cus-perm-xl=True # --cross=True # --debug=True 
# # ### tabel, PolyGen Samples ####

# ### tabel, PolyGen Samples #### # polygen samples # 
CUDA_VISIBLE_DEVICES=2 python trainer/sampling_xl.py  --exp-mode="sampling" --use-light-weight=True  --max-permit-vertices=400 --max-permit-faces=2000 --dataset-name="MotionDataset_processed_deci_scaled" --category="eyeglasses" --recenter-mesh=True --apply-random-shift=True --dataset-root-path='/nas/datasets/gen/datasets' --max-vertices=400 --max-faces=2000 --batch-size=16  --mask-low-ratio=0.30 --mask-high-ratio=0.50  --check-step=200  --vertex-model-path="/nas/datasets/gen/ckpts/pretraining_pure_xl_oven_leg_scissors_pen_tot_cus_False_cross_False_4_MotionDataset_processed_deci_scaled_cat_eyeglasses_bsz_16_max_verts_400_faces_2000_max_permit_verts_400_faces_2000_mask_low_0.3_high_0.5/vertex_model/109800.pth" --face-model-path="/nas/datasets/gen/ckpts/pretraining_pure_xl_oven_leg_scissors_pen_tot_cus_False_cross_False_4_MotionDataset_processed_deci_scaled_cat_eyeglasses_bsz_16_max_verts_400_faces_2000_max_permit_verts_400_faces_2000_mask_low_0.3_high_0.5/face_model/109800.pth" --exp-flag="interp_scaled_leg_scissors_oven_pen"

###### 
# --vertex-model-path=/nas/datasets/gen/ckpts/pretraining_pure_xl_oven_leg_scissors_pen_tot_cus_False_cross_False_4_MotionDataset_processed_deci_scaled_cat_eyeglasses_bsz_16_max_verts_400_faces_2000_max_permit_verts_400_faces_2000_mask_low_0.3_high_0.5/vertex_model/144400.pth
# --face-model-path=/nas/datasets/gen/ckpts/pretraining_pure_xl_oven_leg_scissors_pen_tot_cus_False_cross_False_4_MotionDataset_processed_deci_scaled_cat_eyeglasses_bsz_16_max_verts_400_faces_2000_max_permit_verts_400_faces_2000_mask_low_0.3_high_0.5/face_model/144400.pth

# --vertex-model-path="/nas/datasets/gen/ckpts/pretraining_pure_xl_tot_cus_False_cross_False_4_MotionDataset_processed_deci_scaled_cat_eyeglasses_bsz_16_max_verts_400_faces_2000_max_permit_verts_400_faces_2000_mask_low_0.3_high_0.5/vertex_model/69200.pth" --face-model-path="/nas/datasets/gen/ckpts/pretraining_pure_xl_tot_cus_False_cross_False_4_MotionDataset_processed_deci_scaled_cat_eyeglasses_bsz_16_max_verts_400_faces_2000_max_permit_verts_400_faces_2000_mask_low_0.3_high_0.5/face_model/69200.pth" --exp-flag="interp_st_half_other"

# --vertex-model-path="/nas/datasets/gen/ckpts/pretraining_pure_xl_cus_False_cross_False_4_MotionDataset_processed_deci_scaled_cat_eyeglasses_bsz_16_max_verts_400_faces_2000_max_permit_verts_400_faces_2000_mask_low_0.3_high_0.5/vertex_model/89000.pth" --face-model-path="/nas/datasets/gen/ckpts/pretraining_pure_xl_cus_False_cross_False_4_MotionDataset_processed_deci_scaled_cat_eyeglasses_bsz_16_max_verts_400_faces_2000_max_permit_verts_400_faces_2000_mask_low_0.3_high_0.5/face_model/89000.pth" --exp-flag="interp_st_half" # \
# --debug=True 
# random shift
# ### tabel, PolyGen Samples ####

# /nas/datasets/gen/ckpts/pretraining_tot_nostop_xl_cus_True_cross_True_4_MotionDataset_processed_deci_scaled_cat_oven_bsz_16_max_verts_400_faces_2000_max_permit_verts_400_faces_2000_mask_low_0.3_high_0.5/vertex_model/11000.pth

# --recenter-mesh=True --memory-efficient=True --apply-random-shift=True --batch-size=1 --sv-samples-dir='./samples/tables_small_net_less_permit' --dataset-name="PolyGen_Samples" --category="04379243" --max-permit-vertices=800 --max-permit-faces=2800 --apply-random-shift=True 


# CUDA_VISIBLE_DEVICES=4 python trainer/pretraining_1_xl.py  --use-light-weight=True  --max-permit-vertices=400 --max-permit-faces=2000 --dataset-name="MotionDataset_processed_deci_scaled" --category="eyeglasses" --recenter-mesh=True --apply-random-shift=True --dataset-root-path='/nas/datasets/gen/datasets' --max-vertices=400 --max-faces=2000 --batch-size=16  --mask-low-ratio=0.30 --mask-high-ratio=0.50  --check-step=200 --cus-perm-xl=True --cross=True 