
export PYTHONPATH=.


export display=100
export KL_WEIGHT=0.000001
export N_KETPOINTS=64
export TAR_BASIS=0
export PRED_TYPE="basis"
export N_NEIGHBORING=128
export N_SAMPLES=1024
export SYM_AXIS=1
export HIER_STAGE=1
export SV_CKPT_FREQ=100
export LAP_LAMBDA=4.0
export SYM_LAMBDA=0.0
export USE_VAE_OPT="flow_pc"
export one_shp=""
export with_dis_loss="--with_dis_loss"
export def_version="v4"
export dis_factor=2.5
export small_tar_nn=""
export n_folds=3
export gravity_weight=1.0
export by_part="--by_part"
export optimize_coef="--optimize_coef"
export optimize_basis="--optimize_basis"
export rnd_sample_nn=5
export CVX_FOLDER_FN="dst"
### render examples

# # TrashCan
export coef_multiplier=0.2
export src_index=-1
export n_shots=100
export n_shots=8
export n_shots=50
export EXP_FLAG=TrashCan_obj_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_n_shots_${n_shots}_rnd_src_index_${src_index}_shots_${n_shots}_
export cvx_to_pts_sufix="_cvx_to_cvx_pts_n_512.npy;_cvx_to_verts_cdim_128.npy"
export obj_data_root_folder="./data/SAPIEN_processed/TrashCan"
### data_dir for datas ###
export DATA_DIR="./data/SAPIEN_Deform/TrashCan/link_0;./data/SAPIEN_Deform/TrashCan/link_1"
export n_parts=2
export net_path="part1_net_path;part2_net_path;..."
export new_glb_template="/home/xueyi/gen/DeepMetaHandles/data/sphere_template_4.obj"
export glb_template="./data/sphere_V42_F80.off;./data/sphere_V42_F80.off"

export SRC_FOLDER_FN="dst;dst"
export DST_FOLDER_FN="dst;dst"
export CUDA_ID=5


export CVX_DIM=128


export RECON_COND="cvx"
export RECON_COND="bbox"


# export NET_PATH=""
export with_glb=""



export TRAINER=src/physicsaware_correction/train_def_obj.py


CUDA_VISIBLE_DEVICES=${CUDA_ID} python \
src/${TRAINER} --batch_size=${BATCH_SIZE} --epoch=30000 --learning_rate=${lr}  --env_name='test' --num_basis=${NUM_BASIS} --display=${display} --step_size=100 --kl-weight=${KL_WEIGHT} --exp_flag=${EXP_FLAG}  --data_dir=${DATA_DIR} --n_keypoints=${N_KETPOINTS} --tar_basis=${TAR_BASIS} --load_meta  --pred_type=${PRED_TYPE}   --use_pointnet2 --use_pp_tar_out_feat --neighbouring_k=${N_NEIGHBORING} --dataset_name=v2 --n_samples=${N_SAMPLES} --symmetry_axis=${SYM_AXIS}  --net_path=${net_path} --local_handle --hier_stage=${HIER_STAGE} --sv_ckpt_freq=${SV_CKPT_FREQ}   --with_cat_deform=0   --use_paired_data --wo_keypts_abs --lap_lambda=${LAP_LAMBDA}  --sym_lambda=${SYM_LAMBDA}  --use_cvx_feats --use_def_pc --use_trans_range  --use_vae_opt=${USE_VAE_OPT} --pred_positions --src_folder_fn=${SRC_FOLDER_FN} --dst_folder_fn=${DST_FOLDER_FN} --use_recon_as_cd --recon_cond=${RECON_COND} --cond_tar_pc --cvx_dim=${CVX_DIM} --only_tar  --cvx_to_pts_sufix=${cvx_to_pts_sufix} --coef_multiplier=${coef_multiplier}   ${with_glb} ${one_shp} ${with_dis_loss} --def_version=${def_version} --dis_factor=${dis_factor} --pred_offset ${small_tar_nn} --n_fold=${n_folds}  --gravity_weight=${gravity_weight} ${by_part} ${optimize_coef} ${optimize_basis} --rnd_sample_nn=${rnd_sample_nn} --cvx_folder_fn=${CVX_FOLDER_FN} --glb_template=${glb_template} --n_parts=${n_parts} --obj_data_root_folder=${obj_data_root_folder} --n_shots=${n_shots} --new_glb_template=${new_glb_template}

