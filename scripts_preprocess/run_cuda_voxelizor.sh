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

# /data/datasets/genn/SAPIEN_Deform_voxelized_64_v2/Eyeglasses/link_2/dst
# export all_proxy=http://proxy.i.brainpp.cn:3128
# export no_proxy=.brainpp.cn,.brainpp.ml,.megvii-inc.com,.megvii-op.org,127.0.0.1,localhost
# export http_proxy=$all_proxy https_proxy=$all_proxy

export PYTHONPATH=.


# grid size; quantization bits...
##### grid, context window, gridsize #####
# CUDA_VISIBLE_DEVICES=2  python notebooks/process_data_sr.py --n-scales=50 --vox-size=64 # --exclude-existing #  
# CUDA_VISIBLE_DEVICES=2  python notebooks/process_data_sr.py --n-scales=50 --vox-size=32 --exclude-existing # v9
# CUDA_VISIBLE_DEVICES=2  python notebooks/process_data_sr.py --n-scales=1 --vox-size=64 # --exclude-existing #  
# CUDA_VISIBLE_DEVICES=2  python notebooks/process_data_sr.py --n-scales=50 --vox-size=64 --nprocs=50 # 
CUDA_VISIBLE_DEVICES=0 python notebooks/process_data_sr.py --n-scales=1 --vox-size=64 --nprocs=20 # --exclude-existing #  
