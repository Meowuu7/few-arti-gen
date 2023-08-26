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



# CUDA_VISIBLE_DEVICES=2 python trainer/pretraining_ar_grid.py  --exp-mode="pretraining"  --max-permit-vertices=400 --max-permit-faces=2000 --dataset-name="MotionDataset_processed_voxelized_part_first" --category="eyeglasses" --recenter-mesh=True --dataset-root-path='/nas/datasets/gen/datasets' --max-vertices=400 --max-faces=2000 --batch-size=2  --check-step=200 --exp-flag="some_parts_v0_2d" --num-classes=120 --nn-vertices-predict-ratio=1.0  --context-window=600    --not-remove-du=True --max-num-grids=400 --grid-size=4 --category-part-indicator=True #   --debug=True # --category-part-indicator=True  # --debug=True

# grid size; quantization bits...
# grid size; quantization bits...
# ##### grid, context window, gridsize #####
# CUDA_VISIBLE_DEVICES=1 python trainer/pretraining_ar_grid_vox.py  --exp-mode="pretraining"  --max-permit-vertices=20000 --max-permit-faces=3000 --dataset-name="MotionDataset_processed_voxelized_64" --category="eyeglasses" --recenter-mesh=True --dataset-root-path='/nas/datasets/gen/datasets' --max-vertices=200000 --max-faces=3000 --batch-size=1  --check-step=200 --exp-flag="v1_vox_3_cv_5" --num-classes=120 --nn-vertices-predict-ratio=1.0  --not-remove-du=True --max-num-grids=250 --grid-size=4  --quantization-bits=6 --category-part-indicator=True --debug=True   #  --category-part-indicator=True # --debug=True   # --debug=True

# grid size; quantization bits...
# ##### grid, context window, gridsize #####
# CUDA_VISIBLE_DEVICES=4 python trainer/pretraining_ar_grid_vox_v2.py  --exp-mode="pretraining"  --max-permit-vertices=6400 --max-permit-faces=3000 --dataset-name="MotionDataset_processed_voxelized_64_v9_ns_50" --category="eyeglasses" --recenter-mesh=True --dataset-root-path='/nas/datasets/gen/datasets' --max-vertices=6400 --max-faces=3000 --batch-size=1  --check-step=200 --exp-flag="grid_vox_vocab_legs_inst" --num-classes=120 --nn-vertices-predict-ratio=1.0  --not-remove-du=True --max-num-grids=800 --grid-size=2  --quantization-bits=6 --vertex-model-path="/nas/datasets/gen/ckpts/pretraining_ar_grid_vocab_grid_vox_vocab_n3_indi_True_ctx_-1_gs_2_ngrids_700_scale_0_MotionDataset_processed_voxelized_64_v9_ns_50_bsz_1_max_verts_6400_max_permit_verts_6400_key_256/vertex_model/46400.pth"  --category-part-indicator=True --load-meta=True  --prefix-key-len=256 --prefix-value-len=256  # --use-inst=True #  --category-part-indicator=True # --debug=True   # --debug=True # --context-window=400

# grid size; quantization bits...
# ##### grid, context window, gridsize #####
# CUDA_VISIBLE_DEVICES=2 python trainer/pretraining_ar_grid_vox_v2.py  --exp-mode="pretraining"  --max-permit-vertices=6500 --max-permit-faces=3000 --dataset-name="MotionDataset_processed_voxelized_64_ns_50" --category="eyeglasses" --recenter-mesh=True --dataset-root-path='/nas/datasets/gen/datasets' --max-vertices=6500 --max-faces=3000 --batch-size=1  --check-step=200 --exp-flag="grid_obj_use_val" --num-classes=120 --nn-vertices-predict-ratio=1.0  --not-remove-du=True --max-num-grids=800 --grid-size=2  --quantization-bits=6  --category-part-indicator=True --load-meta=True  --prefix-key-len=256 --prefix-value-len=256 --num-parts=2  --context-window=450 --use-context-window-as-max=True  --sampling-max-num-grids=800 --ar-object=True --balance-classes=True # --use-inst=True #  --category-part-indicator=True # --debug=True   # --debug=True # --context-window=400

# export DATASET_NAME="PartNet_voxelized_64"
# export DATASET_NAME="ShapeNetCore.v2_voxelized_64"
# export DATASET_NAME="MotionDataset_processed_voxelized_64_ns_50"
# export DATASET_NAME="subd_mesh"
# export DATASET_NAME="PartNet" # /nas/datasets/gen/datasets/PartNet
# export DATASET_NAME="MotionDataset_processed_collapsed_sel"
# export DATASET_NAME="screw_1"
export DATASET_NAME="door_1"
export DATA_ROOT_PATH='/nas/datasets/gen/datasets/subd_mesh'
# export DATA_ROOT_PATH='/nas/datasets/gen/datasets/subd_mesh'
export EXP_MODE="pretraining"
# export EXP_MODE="sampling"
# export VERTEX_MODEL_PATH="/nas/datasets/gen/ckpts/pretraining_ar_grid_vocab_ctx_500_wedge_subd4_indi_False_ctx_500_gs_2_ngrids_800_nobj_3_scale_0_subd_mesh_bsz_1_max_verts_6500_max_permit_verts_6500_key_256/vertex_model/25200.pth"
# export VERTEX_MODEL_PATH="/nas/datasets/gen/ckpts/pretraining_ar_grid_vocab_bulk_ft_indi_False_ctx_350_gs_2_ngrids_400_nobj_3_scale_0_subd_mesh_bsz_1_max_verts_6500_max_permit_verts_6500_key_256/vertex_model/16800.pth"
export VERTEX_MODEL_PATH=""

export SUBDN=3 # 4, 3
export CONTEXT_WINDOW=1000 # 3000 # 800 # 2410 # 500 # 350 # 400 #500
export MAX_NUM_GRIDS=1000 # 3000 # 800 # 2410 # 500 # 400 # 800 # 400 #500
# export EXP_FLAG="bulk_ft" # "ctx_500_subd5"
export EXP_FLAG="collapse_sel_ddp" # "ctx_500_subd5"
export AR_SUBD_IDX=1 # 2



# 2,3,4,5,6,7     6
# a reald diffusion model...? a real diffusion model for mesh generation?

# # # ##### PartNet #####
# CUDA_VISIBLE_DEVICES=1 python  \
# trainer/pretraining_subd_meshes.py  --exp-mode="pretraining"  --max-permit-vertices=6500 --max-permit-faces=3000 --dataset-name=${DATASET_NAME} --category="eyeglasses" --recenter-mesh=True --dataset-root-path='/nas/datasets/gen/datasets' --max-vertices=6500 --max-faces=3000 --batch-size=1  --check-step=2000 --exp-flag="ctx_500_wedge_subd4_reg" --num-classes=120 --nn-vertices-predict-ratio=1.0  --not-remove-du=True --max-num-grids=800 --grid-size=2  --quantization-bits=6  --load-meta=True  --prefix-key-len=256 --prefix-value-len=256 --num-parts=2   --use-context-window-as-max=True   --sampling-max-num-grids=1600 --ar-object=True --balance-classes=True --num-objects=3 --subdn=4 --subdn-test=4 --context-window=500  --learning-rate=2e-3  # --use-part #  --use-inst=True  #  --category-part-indicator=True # --debug=True   # --debug=True # --context-window=400 --use-inst=True

### subd_meshes_ddp()
### 

### SSP_qslim_optimal_collapse_edge_callbacks
# # ##### PartNet ##### # 
CUDA_VISIBLE_DEVICES=4,5,6,7 TORCH_DISTRIBUTED_DEBUG=INFO  python  -m torch.distributed.launch --nproc_per_node=4 \
trainer/pretraining_ar_subd_meshes_ddp.py  --exp-mode=${EXP_MODE}  --max-permit-vertices=800 --max-permit-faces=1000 --dataset-name=${DATASET_NAME} --category="eyeglasses" --recenter-mesh=True --dataset-root-path=${DATA_ROOT_PATH} --max-vertices=800 --max-faces=1000 --batch-size=1  --check-step=200 --exp-flag=${EXP_FLAG} --num-classes=120 --nn-vertices-predict-ratio=1.0  --not-remove-du=True --max-num-grids=${MAX_NUM_GRIDS} --grid-size=2  --quantization-bits=6  --load-meta=True  --prefix-key-len=256 --prefix-value-len=256 --num-parts=2 --vertex-model-path=${VERTEX_MODEL_PATH}    --sampling-max-num-grids=1600 --ar-object=True --balance-classes=True --num-objects=3 --subdn=${SUBDN} --subdn-test=${SUBDN} --context-window=${CONTEXT_WINDOW} --use-context-window-as-max=True    --ar-subd-idx=${AR_SUBD_IDX} # --learning-rate=2e-3 # --use-part #  --use-inst=True  #  --category-part-indicator=True # --debug=True   # --debug=True # --context-window=400 --use-inst=True 

# ##### PartNet #####
# CUDA_VISIBLE_DEVICES=2 python  \
# trainer/pretraining_subd_meshes.py  --exp-mode="sampling"  --max-permit-vertices=6500 --max-permit-faces=3000 --dataset-name=${DATASET_NAME} --category="eyeglasses" --recenter-mesh=True --dataset-root-path='/nas/datasets/gen/datasets' --max-vertices=6500 --max-faces=3000 --batch-size=1  --check-step=200 --exp-flag="subd_exp_norm2" --num-classes=120 --nn-vertices-predict-ratio=1.0  --not-remove-du=True --max-num-grids=800 --grid-size=2  --quantization-bits=6  --load-meta=True  --prefix-key-len=256 --prefix-value-len=256 --num-parts=2  --context-window=800 --use-context-window-as-max=True   --sampling-max-num-grids=1600 --ar-object=True --balance-classes=True --num-objects=3 --subdn=2  --vertex-model-path="/nas/datasets/gen/ckpts/pretraining_ar_grid_vocab_subd_exp_indi_False_ctx_800_gs_2_ngrids_800_nobj_3_scale_0_subd_mesh_bsz_1_max_verts_6500_max_permit_verts_6500_key_256/vertex_model/31400.pth"  # --use-part #  --use-inst=True  #  --category-part-indicator=True # --debug=True   # --debug=True # --context-window=400 --use-inst=True

# # grid size; quantization bits...
# ##### grid, context window, gridsize;  #####
# CUDA_VISIBLE_DEVICES=0 python trainer/pretraining_ar_grid_vox_v2.py  --exp-mode="pretraining"  --max-permit-vertices=6500 --max-permit-faces=3000 --dataset-name="MotionDataset_processed_voxelized_64_ns_50" --category="eyeglasses" --recenter-mesh=True --dataset-root-path='/nas/datasets/gen/datasets' --max-vertices=6500 --max-faces=3000 --batch-size=1  --check-step=200 --exp-flag="grid_obj_corpus_v9" --num-classes=120 --nn-vertices-predict-ratio=1.0  --not-remove-du=True --max-num-grids=800 --grid-size=2  --quantization-bits=6   --load-meta=True  --prefix-key-len=256 --prefix-value-len=256 --num-parts=3  --context-window=800 --use-context-window-as-max=True  --sampling-max-num-grids=800 --balance-classes=True # --use-inst=True #  --category-part-indicator=True # --debug=True   # --debug=True # --context-window=400

# --grid-size=8
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