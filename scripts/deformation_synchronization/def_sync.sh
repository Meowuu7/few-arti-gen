
export PYTHONPATH=.

# CUDA_VISIBLE_DEVICES=3 python \
# src/train_target_driven.py --batch_size=4 --epoch=30000 --learning_rate=0.0005 --gpu='0' --env_name='test' --num_basis=256 --display=100 --step_size=100 # --with-discriminator


# export NUM_BASIS=1024
export NUM_BASIS=64 # 16 # 64 ### number of basis here ###
export NUM_BASIS=16 # 16 # 64 ### number of basis here ###
# export NUM_BASIS=4 # 16 # 64 ### number of basis here ###
# export N_KETPOINTS=64
export N_KETPOINTS=256
# export N_KETPOINTS=1024
export N_NEIGHBORING=128


export BATCH_SIZE=1 # 2 # 4
export KL_WEIGHT=0.0001 # 1.0
export KL_WEIGHT=0.00001 # 1.0
export KL_WEIGHT=0.000001 # 1.0
export TAR_BASIS=0
# export TAR_BASIS=1
export PRED_TYPE="basis" ### ["basis", "offset"]

export SYM_AXIS=1
# export SYM_AXIS=0



# Eyeglasses -> link0/0 /data/datasets/genn/SAPIEN_Deform/Eyeglasses/link_2/dst
export def_version="v4"
export KL_WEIGHT=0.000001 # 1.0
export LAP_LAMBDA=4.0
export coef_multiplier=0.2
export dis_factor=2.5
export NUM_BASIS=16
export N_KETPOINTS=64
export DATA_DIR="/data/datasets/genn/SAPIEN_Deform/Eyeglasses/link_0"
export PRED_TYPE="basis"
export use_gt_cages="--use_gt_cages"
export cd_weight=0.0
export n_shots=30
# export EXP_FLAG=eye_link2_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_shots_${n_shots}_fr_table_
# # export EXP_FLAG=eye_link1_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_shots_${n_shots}_fr_self_
export EXP_FLAG=eye_link0_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_shots_${n_shots}_fr_self_
export SRC_FOLDER_FN="dst"
export DST_FOLDER_FN="dst"
export CVX_FOLDER_FN="dst"
export cvx_to_pts_sufix="_cvx_to_verts.npy" ## src_cvx_to_pts...
export src_cvx_to_pts_sufix="_cvx_to_verts.npy"
export dst_cvx_to_pts_sufix="_cvx_to_cvx_pts_n_512.npy"
export one_shp=""
export with_dis_loss="--with_dis_loss"
export pred_offst="--pred_offset"
export small_tar_nn=""
export lr=0.0001
export gravity_weight=1.0
export n_folds=3
export by_part="--by_part"
export optimize_coef="--optimize_coef"
export with_image_loss=""
export net_path=""
export optimize_basis="--optimize_basis"
export rnd_sample_nn=5
export glb_template="./data/sphere_template_4.obj"
export CUDA_ID=0
export cvx_list_filter=""
export src_data_dir="/data/datasets/genn/SAPIEN_Deform/Eyeglasses/link_2"
# export src_data_dir="/data/datasets/genn/SAPIEN_Deform/Eyeglasses/link_1"
export src_data_dir="/data/datasets/genn/SAPIEN_Deform/Eyeglasses/link_0"
export src_n_keypoints=64 ### nkeypoin ### 256 keypts ### 

# export dst_data_dir="/data/datasets/genn/ShapeNetCoreV2_Deform/03001627_c_512" # 04379243 ### dst_data_dir...
# export dst_n_keypoints=1024 ### dst_1024 n_keypoints ###

export dst_data_dir="/share/xueyi/datasets/gen/datasets/Shape2Motion_Deform/eyeglasses/none_motion"
# export dst_data_dir="/data/datasets/genn/ShapeNetCoreV2_Deform/04379243_c_512"
export dst_n_keypoints=1024
export dst_n_keypoints=64
# export src_n_keypoints=64 ### nkeypoin ### 256 keypts ### 

# export DST_FOLDER_FN="dst_def"
# export CVX_FOLDER_FN="src_bsp"
export DST_FOLDER_FN="dst"
export CVX_FOLDER_FN="dst"

export src_cvx_to_pts_sufix="_cvx_to_verts.npy"
export src_cvx_to_pts_sufix="_manifold_cvx_to_verts.npy"
# export dst_cvx_to_pts_sufix="_cvx_to_cvx_pts_n_512.npy"
export dst_cvx_to_pts_sufix="_cvx_to_verts.npy"
export dst_cvx_to_pts_sufix="_manifold_cvx_to_verts.npy"
export cvx_list_filter=""
export cvx_list_filter="--cvx_list_filter"
export src_index=-1






export TRAINER=def_sync.py





export CVX_DIM=128


export RECON_COND="cvx"
export RECON_COND="bbox"


# export NET_PATH=""
export with_glb=""

# export display=100
export display=20

export CUDA_ID=0

export BATCH_SIZE=1
export TAR_BASIS=0


CUDA_VISIBLE_DEVICES=${CUDA_ID} python \
src/deformation_synchronization/${TRAINER} --batch_size=${BATCH_SIZE} --epoch=30000 --learning_rate=${lr}  --env_name='test' --num_basis=${NUM_BASIS} --display=${display} --step_size=100 --kl-weight=${KL_WEIGHT} --exp_flag=${EXP_FLAG}  --data_dir=${DATA_DIR} --n_keypoints=${N_KETPOINTS} --tar_basis=${TAR_BASIS} --load_meta  --pred_type=${PRED_TYPE}   --use_pointnet2 --use_pp_tar_out_feat --neighbouring_k=${N_NEIGHBORING} --dataset_name=v2 --n_samples=${N_SAMPLES} --symmetry_axis=${SYM_AXIS}  --net_path=${net_path} --local_handle --hier_stage=${HIER_STAGE} --sv_ckpt_freq=${SV_CKPT_FREQ}   --with_cat_deform=0   --use_paired_data --wo_keypts_abs --lap_lambda=${LAP_LAMBDA}  --sym_lambda=${SYM_LAMBDA}  --use_cvx_feats --use_def_pc --use_trans_range  --use_vae_opt=${USE_VAE_OPT} --pred_positions --src_folder_fn=${SRC_FOLDER_FN} --dst_folder_fn=${DST_FOLDER_FN} --use_recon_as_cd --recon_cond=${RECON_COND} --cond_tar_pc --cvx_dim=${CVX_DIM} --only_tar  --cvx_to_pts_sufix=${cvx_to_pts_sufix} --coef_multiplier=${coef_multiplier}   ${with_glb} ${one_shp} ${with_dis_loss} --def_version=${def_version} --dis_factor=${dis_factor} --pred_offset ${small_tar_nn} --n_fold=${n_folds}  --gravity_weight=${gravity_weight} ${by_part} ${use_gt_cages} ${optimize_coef} ${with_image_loss} ${optimize_basis} --rnd_sample_nn=${rnd_sample_nn} --cvx_folder_fn=${CVX_FOLDER_FN} --glb_template=${glb_template} --n_shots=${n_shots} ${cvx_list_filter} --src_data_dir=${src_data_dir} --src_n_keypoints=${src_n_keypoints} --dst_data_dir=${dst_data_dir} --dst_n_keypoints=${dst_n_keypoints} --src_cvx_to_pts_sufix=${src_cvx_to_pts_sufix} --dst_cvx_to_pts_sufix=${dst_cvx_to_pts_sufix} --src_index=${src_index}

