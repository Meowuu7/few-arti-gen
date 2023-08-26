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


###### V1 ######
# ### tabel, PolyGen Samples ####s
# CUDA_VISIBLE_DEVICES=2 python trainer/pretraining_ar_coarse_to_fine.py  --exp-mode="pretraining"  --max-permit-vertices=300 --max-permit-faces=1500 --dataset-name="MotionDataset_processed_deci_part_first" --category="eyeglasses" --recenter-mesh=True --dataset-root-path='/nas/datasets/gen/datasets' --max-vertices=300 --max-faces=1500 --batch-size=1  --mask-low-ratio=0.30 --mask-high-ratio=0.50  --check-step=200 --exp-flag="some_parts_v2_to_frames" --num-classes=120  --coarse-angle-limit=0.2 --fine-angle-limit=30 --debug=True   # --debug=True 


# ### tabel, PolyGen Samples ####s
CUDA_VISIBLE_DEVICES=1 python trainer/pretraining_ar_coarse_to_fine_v2.py  --exp-mode="pretraining"  --max-permit-vertices=400 --max-permit-faces=2000 --dataset-name="MotionDataset_processed_deci_part_first" --category="eyeglasses" --recenter-mesh=True --dataset-root-path='/nas/datasets/gen/datasets' --max-vertices=400 --max-faces=2000 --batch-size=1  --mask-low-ratio=0.30 --mask-high-ratio=0.50  --check-step=200 --exp-flag="some_parts_v2_to_frames_res" --num-classes=120  --coarse-angle-limit=200 --fine-angle-limit=30 --vertex-model-path="/nas/datasets/gen/ckpts/pretraining_ar_ctof_v2_some_parts_v2_to_frames_cat_part_indi_Falselight_False_shift_False_filp_False_MotionDataset_processed_deci_part_first_bsz_1_max_verts_400_faces_2000_max_permit_verts_400_faces_2000/coarse_vertex_model/38800.pth" --face-model-path="/nas/datasets/gen/ckpts/pretraining_ar_ctof_v2_some_parts_v2_to_frames_cat_part_indi_Falselight_False_shift_False_filp_False_MotionDataset_processed_deci_part_first_bsz_1_max_verts_400_faces_2000_max_permit_verts_400_faces_2000/coarse_face_model/38800.pth" --fine-vertex-model-path="/nas/datasets/gen/ckpts/pretraining_ar_ctof_v2_some_parts_v2_to_frames_cat_part_indi_Falselight_False_shift_False_filp_False_MotionDataset_processed_deci_part_first_bsz_1_max_verts_400_faces_2000_max_permit_verts_400_faces_2000/vertex_model/38800.pth" --fine-face-model-path="/nas/datasets/gen/ckpts/pretraining_ar_ctof_v2_some_parts_v2_to_frames_cat_part_indi_Falselight_False_shift_False_filp_False_MotionDataset_processed_deci_part_first_bsz_1_max_verts_400_faces_2000_max_permit_verts_400_faces_2000/face_model/38800.pth"   # --debug=True --category-part-indicator=True
# --num-classes=177 --category-part-indicator=True # --debug=True
# --debug=Trues
# --cus-perm-xl=True # --cross=True # --debug=True 
# --apply-random-shift=True 
# ### tabel, PolyGen Samples ####
# --use-light-weight=True 

# --recenter-mesh=True --memory-efficient=True --apply-random-shift=True --batch-size=1 --sv-samples-dir='./samples/tables_small_net_less_permit' --dataset-name="PolyGen_Samples" --category="04379243" --max-permit-vertices=800 --max-permit-faces=2800 --apply-random-shift=True 


# CUDA_VISIBLE_DEVICES=4 python trainer/pretraining_1_xl.py  --use-light-weight=True  --max-permit-vertices=400 --max-permit-faces=2000 --dataset-name="MotionDataset_processed_deci_scaled" --category="eyeglasses" --recenter-mesh=True --apply-random-shift=True --dataset-root-path='/nas/datasets/gen/datasets' --max-vertices=400 --max-faces=2000 --batch-size=16  --mask-low-ratio=0.30 --mask-high-ratio=0.50  --check-step=200 --cus-perm-xl=True --cross=True 