
### get sample dir, data dir, checkpoint dir ###
export SAMPLE_DIR=/data2/sim/samples/BSP-Net/
export DATA_DIR=/data2/sim/smplh_data_gathered # data dir  # gather the smpl data ##
export CHECKPOINT_DIR=/data2/sim/ckpts/BSP-Net ## category dir

export BATCH_SIZE=2
export C_DIM=512
export CATEGORY=female
export DATASET=${CATEGORY}

export MODEL_FLAG=cdim_${C_DIM}_${CATEGORY} ### 


export CUDA_ID=0

CUDA_VISIBLE_DEVICES=${CUDA_ID} python \
main.py --ae --phase 1 --sample_dir=${SAMPLE_DIR}  --data_dir=${DATA_DIR} --dataset=${DATASET} --sample_vox_size 64 --batch_size=${BATCH_SIZE} --c_dim=${C_DIM}  --model_flag=${MODEL_FLAG} --checkpoint_dir=${CHECKPOINT_DIR} --start 0 --end 9000 --sample_flag=cvx

