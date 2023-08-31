
# any flag for the model or just leave it blank ## 
export MODEL_FLAG=""


## number of convexes used for partition ##
export C_DIM=512
## sample directory ##
export SAMPLE_DIR=""
## source data directory ##
export DATA_DIR=""
## checkpoint directory ##
export CHECKPOINT_DIR=""


export BATCH_SIZE_16=8 ## batch_size
export BATCH_SIZE=2
export CUDA_ID=0



CUDA_VISIBLE_DEVICES=${CUDA_ID} python \
main.py --ae --train --phase 0 --iteration 200000 --sample_dir=${SAMPLE_DIR} --data_dir=${DATA_DIR} --dataset=${DATASET} --sample_vox_size 16 --batch_size=${BATCH_SIZE_16} --c_dim=${C_DIM} --model_flag=${MODEL_FLAG} --checkpoint_dir=${CHECKPOINT_DIR}

CUDA_VISIBLE_DEVICES=${CUDA_ID} python \
main.py --ae --train --phase 0 --iteration 200000 --sample_dir=${SAMPLE_DIR}  --data_dir=${DATA_DIR} --dataset=${DATASET} --sample_vox_size 32 --batch_size=${BATCH_SIZE} --c_dim=${C_DIM}  --model_flag=${MODEL_FLAG} --checkpoint_dir=${CHECKPOINT_DIR}

CUDA_VISIBLE_DEVICES=${CUDA_ID} python \
main.py --ae --train --phase 0 --iteration 400000 --sample_dir=${SAMPLE_DIR}  --data_dir=${DATA_DIR} --dataset=${DATASET} --sample_vox_size 64 --batch_size=${BATCH_SIZE} --c_dim=${C_DIM}  --model_flag=${MODEL_FLAG} --checkpoint_dir=${CHECKPOINT_DIR}

CUDA_VISIBLE_DEVICES=${CUDA_ID} python \
main.py --ae --train --phase 1 --iteration 400000 --sample_dir=${SAMPLE_DIR}  --data_dir=${DATA_DIR} --dataset=${DATASET} --sample_vox_size 64 --batch_size=${BATCH_SIZE} --c_dim=${C_DIM}  --model_flag=${MODEL_FLAG} --checkpoint_dir=${CHECKPOINT_DIR}
