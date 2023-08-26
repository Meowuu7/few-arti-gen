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


export DATASET_ROOT_PATH="/nas/datasets/gen/datasets"
# export DATASET_ROOT_PATH="/nas/datasets/gen/samples"
# export DATASET_ROOT_PATH="/nas/datasets/gen"
# export DATASET_NAME="PartNet_voxelized_64"
# export DATASET_NAME="ShapeNetCore.v2_voxelized_64"
# export DATASET_NAME="MotionDataset_processed_voxelized_64_ns_50"
export DATASET_NAME="subd_mesh"
# export DATASET_NAME="samples"
export EXP_MODE="pretraining"
# export EXP_MODE="sampling"
# export VERTEX_MODEL_PATH="/nas/datasets/gen/ckpts/pretraining_ar_grid_vocab_nocontextual_screw_dquant1_indi_False_ctx_600_gs_2_ngrids_600_nobj_3_scale_0_subd_mesh_bsz_1_max_verts_6500_max_permit_verts_6500_key_256/vertex_model/1400.pth"
# export VERTEX_MODEL_PATH="/nas/datasets/gen/ckpts/pretraining_ar_grid_vocab_eyeglasses_no_delta_no_prob_ws_eye_dc_indi_False_ctx_1000_gs_2_ngrids_1000_nobj_3_scale_0_subd_mesh_bsz_1_max_verts_6500_max_permit_verts_6500_key_256/vertex_model/18800.pth"
export VERTEX_MODEL_PATH=""
# export SUBDN=5 # 4, 3
export SUBDN=2 # 3 # 4 # , 3
# export CONTEXT_WINDOW=350 # 400 #500
# export CONTEXT_WINDOW=10 # 400 #500
export CONTEXT_WINDOW=1000 # 800 # 600 # 500
# export MAX_NUM_GRIDS=400 # 800 # 400 #500
export MAX_NUM_GRIDS=1000 # 800 # 600 # 500
# export EXP_FLAG="bulk_ft" # "ctx_500_subd5"
# export EXP_FLAG="test_screw_nohf_0.2_all_eyeglasses" # "ctx_500_subd5" ### test half_flaps
# export EXP_FLAG="eyeglasses_ws_sc_bsp2" # "ctx_500_subd5" ### test half_flaps
export EXP_FLAG="eyeglasses_ws_more_subd_sc_wlap_mean" # "ctx_500_subd5" ### test half_flaps



# 2,3,4,5,6,7     6
# a reald diffusion model...? a real diffusion model for mesh generation?

# # # ##### PartNet #####
# CUDA_VISIBLE_DEVICES=1 python  \
# trainer/pretraining_subd_meshes.py  --exp-mode="pretraining"  --max-permit-vertices=6500 --max-permit-faces=3000 --dataset-name=${DATASET_NAME} --category="eyeglasses" --recenter-mesh=True --dataset-root-path='/nas/datasets/gen/datasets' --max-vertices=6500 --max-faces=3000 --batch-size=1  --check-step=2000 --exp-flag="ctx_500_wedge_subd4_reg" --num-classes=120 --nn-vertices-predict-ratio=1.0  --not-remove-du=True --max-num-grids=800 --grid-size=2  --quantization-bits=6  --load-meta=True  --prefix-key-len=256 --prefix-value-len=256 --num-parts=2   --use-context-window-as-max=True   --sampling-max-num-grids=1600 --ar-object=True --balance-classes=True --num-objects=3 --subdn=4 --subdn-test=4 --context-window=500  --learning-rate=2e-3  # --use-part #  --use-inst=True  #  --category-part-indicator=True # --debug=True   # --debug=True # --context-window=400 --use-inst=True


# # ##### PartNet #####
CUDA_VISIBLE_DEVICES=0 python  \
trainer/pretraining_subd_meshes_weak_sup.py  --exp-mode=${EXP_MODE}  --max-permit-vertices=6500 --max-permit-faces=3000 --dataset-name=${DATASET_NAME} --category="eyeglasses" --recenter-mesh=True --dataset-root-path=${DATASET_ROOT_PATH} --max-vertices=6500 --max-faces=3000 --batch-size=1  --check-step=200 --exp-flag=${EXP_FLAG} --num-classes=120 --nn-vertices-predict-ratio=1.0  --not-remove-du=True --max-num-grids=${MAX_NUM_GRIDS} --grid-size=2  --quantization-bits=10  --load-meta=True  --prefix-key-len=256 --prefix-value-len=256 --num-parts=2 --vertex-model-path=${VERTEX_MODEL_PATH}    --sampling-max-num-grids=1600 --ar-object=True --balance-classes=True --num-objects=3 --subdn=${SUBDN} --subdn-test=${SUBDN} --context-window=${CONTEXT_WINDOW} --use-context-window-as-max=True  --st-subd-idx=0  --use-prob --min-quant-range=-0.2 --max-quant-range=0.2 --fake-upsample --learning-rate=2e-3 --pred-delta  --ar-subd-idx=1 --with-laplacian  # --pred-prob #  --pred-delta #  --use-local-frame # --fake-upsample  # --use-part #  --use-inst=True  #  --category-part-indicator=True # --debug=True   # --debug=True # --context-window=400 --use-inst=True   --learning-rate=2e-3

### use_half_flaps
### pred_delta

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
