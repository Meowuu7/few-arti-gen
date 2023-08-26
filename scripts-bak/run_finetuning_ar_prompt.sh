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

# # ####### no conditional ########
# CUDA_VISIBLE_DEVICES=2 python trainer/finetuning_ar_prompt.py  --exp-mode="finetuning"  --max-permit-vertices=400 --max-permit-faces=2000 --dataset-name="MotionDataset_processed_deci_part_first_30_normalized" --category="eyeglasses" --recenter-mesh=True --dataset-root-path='/nas/datasets/gen/datasets' --max-vertices=400 --max-faces=2000 --batch-size=1  --check-step=200 --exp-flag="prompt_finetune_256" --num-classes=120 --vertex-model-path="/nas/datasets/gen/ckpts/pretraining_ar_some_parts_v1_cat_part_indi_False_face_cond_False_light_False_shift_False_filp_False_MotionDataset_processed_deci_part_first_30_normalized_bsz_2_max_verts_400_faces_2000_max_permit_verts_400_faces_2000/vertex_model/55800.pth" --face-model-path="/nas/datasets/gen/ckpts/pretraining_ar_some_parts_v1_cat_part_indi_False_face_cond_False_light_False_shift_False_filp_False_MotionDataset_processed_deci_part_first_30_normalized_bsz_2_max_verts_400_faces_2000_max_permit_verts_400_faces_2000/face_model/55800.pth"  --prefix-key-len=256 --prefix-value-len=256
# # --debug=True --category-part-indicator=True --use-light-weight=True 
# # --num-classes=177 --category-part-indicator=True # --debug=True
# # --debug=Trues
# # --cus-perm-xl=True # --cross=True # --debug=True 
# # --apply-random-shift=True 

# #######  conditional ######## # part-indicator=False
# cond pred
# CUDA_VISIBLE_DEVICES=2 python trainer/finetuning_ar_prompt.py  --exp-mode="finetuning"  --max-permit-vertices=400 --max-permit-faces=2000 --dataset-name="MotionDataset_processed_deci_part_first_30_normalized_reform" --category="eyeglasses" --recenter-mesh=True --dataset-root-path='/nas/datasets/gen/datasets' --max-vertices=400 --max-faces=2000 --batch-size=1  --check-step=2000 --exp-flag="prompt_ft_fr_512" --num-classes=120 --vertex-model-path="/nas/datasets/gen/ckpts/pretraining_ar_some_parts_v9_cond_fr_False_indi_True_ctx_750_face_cond_False_light_False_spec_False_vpred_1.0_scale_1_MotionDataset_processed_deci_part_first_30_normalized_reform_bsz_2_max_verts_400_faces_2000_max_permit_verts_400_faces_2000/vertex_model/164000.pth" --face-model-path="/nas/datasets/gen/ckpts/pretraining_ar_some_parts_v9_cond_fr_False_indi_True_ctx_750_face_cond_False_light_False_spec_False_vpred_1.0_scale_1_MotionDataset_processed_deci_part_first_30_normalized_reform_bsz_2_max_verts_400_faces_2000_max_permit_verts_400_faces_2000/face_model/164000.pth"   --prefix-key-len=512 --prefix-value-len=512  --context-window=750    --not-remove-du=True  --category-part-indicator=True --apply-random-scaling=1

CUDA_VISIBLE_DEVICES=4 python trainer/finetuning_ar_prompt.py  --exp-mode="finetuning"  --max-permit-vertices=400 --max-permit-faces=2000 --dataset-name="MotionDataset_processed_voxelized_part_first" --category="eyeglasses" --recenter-mesh=True --dataset-root-path='/nas/datasets/gen/datasets' --max-vertices=400 --max-faces=2000 --batch-size=1  --check-step=2000 --exp-flag="prompt_ft_fr_512" --num-classes=120 --vertex-model-path="/nas/datasets/gen/ckpts/pretraining_ar_some_parts_v0_fr_False_indi_False_ctx_600_face_cond_False_light_False_spec_False_vpred_1.0_scale_0_MotionDataset_processed_voxelized_part_first_bsz_2_max_verts_400_faces_2000_max_permit_verts_400_faces_2000/vertex_model/92000.pth" --face-model-path="/nas/datasets/gen/ckpts/pretraining_ar_some_parts_v0_fr_False_indi_False_ctx_600_face_cond_False_light_False_spec_False_vpred_1.0_scale_0_MotionDataset_processed_voxelized_part_first_bsz_2_max_verts_400_faces_2000_max_permit_verts_400_faces_2000/face_model/92000.pth"   --prefix-key-len=512 --prefix-value-len=512  --context-window=600    --not-remove-du=True  --category-part-indicator=True # --apply-random-scaling=1


# reform, ctx = 750, scale = 0
# --vertex-model-path="/nas/datasets/gen/ckpts/pretraining_ar_some_parts_v0_cond_fr_False_indi_True_ctx_750_face_cond_False_light_False_spec_False_vpred_1.0_scale_0_MotionDataset_processed_deci_part_first_30_normalized_reform_bsz_2_max_verts_400_faces_2000_max_permit_verts_400_faces_2000/vertex_model/127000.pth" --face-model-path="/nas/datasets/gen/ckpts/pretraining_ar_some_parts_v0_cond_fr_False_indi_True_ctx_750_face_cond_False_light_False_spec_False_vpred_1.0_scale_0_MotionDataset_processed_deci_part_first_30_normalized_reform_bsz_2_max_verts_400_faces_2000_max_permit_verts_400_faces_2000/face_model/127000.pth"   --prefix-key-len=512 --prefix-value-len=512  --context-window=750    --not-remove-du=True  --category-part-indicator=True

# ctx = 750, indi = True, 
# --vertex-model-path="/nas/datasets/gen/ckpts/pretraining_ar_some_parts_v0_cond_fr_False_indi_True_ctx_750_face_cond_False_light_False_spec_False_vpred_1.0_scale_0_MotionDataset_processed_deci_part_first_30_normalized_bsz_2_max_verts_400_faces_2000_max_permit_verts_400_faces_2000/vertex_model/288200.pth" --face-model-path="/nas/datasets/gen/ckpts/pretraining_ar_some_parts_v0_cond_fr_False_indi_True_ctx_750_face_cond_False_light_False_spec_False_vpred_1.0_scale_0_MotionDataset_processed_deci_part_first_30_normalized_bsz_2_max_verts_400_faces_2000_max_permit_verts_400_faces_2000/face_model/288200.pth"  --prefix-key-len=512 --prefix-value-len=512  --context-window=750   --category-part-indicator=True

# --vertex-model-path="/nas/datasets/gen/ckpts/pretraining_ar_some_parts_v0_cond_add_con_fr_True_indi_False_ctx_900_face_cond_False_light_False_spec_False_vpred_1.0_scale_0_MotionDataset_processed_deci_part_first_30_normalized_bsz_2_max_verts_400_faces_2000_max_permit_verts_400_faces_2000/vertex_model/235800.pth" --face-model-path="/nas/datasets/gen/ckpts/pretraining_ar_some_parts_v0_cond_add_con_fr_True_indi_False_ctx_900_face_cond_False_light_False_spec_False_vpred_1.0_scale_0_MotionDataset_processed_deci_part_first_30_normalized_bsz_2_max_verts_400_faces_2000_max_permit_verts_400_faces_2000/face_model/235800.pth"  --prefix-key-len=512 --prefix-value-len=512  --context-window=900 

### indi = False ###
# --vertex-model-path="/nas/datasets/gen/ckpts/pretraining_ar_some_parts_v0_uncond_fr_False_indi_False_ctx_900_face_cond_False_light_False_spec_False_vpred_1.0_scale_0_MotionDataset_processed_deci_part_first_30_normalized_bsz_2_max_verts_400_faces_2000_max_permit_verts_400_faces_2000/vertex_model/175200.pth" --face-model-path="/nas/datasets/gen/ckpts/pretraining_ar_some_parts_v0_uncond_fr_False_indi_False_ctx_900_face_cond_False_light_False_spec_False_vpred_1.0_scale_0_MotionDataset_processed_deci_part_first_30_normalized_bsz_2_max_verts_400_faces_2000_max_permit_verts_400_faces_2000/face_model/175200.pth"  --prefix-key-len=256 --prefix-value-len=256   --context-window=900 

# --vertex-model-path="/nas/datasets/gen/ckpts/pretraining_ar_some_parts_v0_cond_cat_part_indi_True_face_cond_False_light_False_shift_False_scale_0_MotionDataset_processed_deci_part_first_30_normalized_bsz_2_max_verts_400_faces_2000_max_permit_verts_400_faces_2000/vertex_model/101000.pth" --face-model-path="/nas/datasets/gen/ckpts/pretraining_ar_some_parts_v0_cond_cat_part_indi_True_face_cond_False_light_False_shift_False_scale_0_MotionDataset_processed_deci_part_first_30_normalized_bsz_2_max_verts_400_faces_2000_max_permit_verts_400_faces_2000/face_model/101000.pth"  --prefix-key-len=512 --prefix-value-len=512 
# --category-part-indicator=True # --debug=True
# 

#
##### table polygen ssamples #####
# --recenter-mesh=True --memory-efficient=True --apply-random-shift=True --batch-size=1 --sv-samples-dir='./samples/tables_small_net_less_permit' --dataset-name="PolyGen_Samples" --category="04379243" --max-permit-vertices=800 --max-permit-faces=2800 --apply-random-shift=True 


# CUDA_VISIBLE_DEVICES=4 python trainer/pretraining_1_xl.py  --use-light-weight=True  --max-permit-vertices=400 --max-permit-faces=2000 --dataset-name="MotionDataset_processed_deci_scaled" --category="eyeglasses" --recenter-mesh=True --apply-random-shift=True --dataset-root-path='/nas/datasets/gen/datasets' --max-vertices=400 --max-faces=2000 --batch-size=16  --mask-low-ratio=0.30 --mask-high-ratio=0.50  --check-step=200 --cus-perm-xl=True --cross=True 