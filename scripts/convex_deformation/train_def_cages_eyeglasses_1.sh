
export PYTHONPATH=.


#### Basic Configs ####
export CUDA_ID=0
export display=20
export KL_WEIGHT=0.000001 # 1.0
export BATCH_SIZE=1
export lr=0.0001
export NUM_BASIS=32
export N_LAYERS=5
export HIER_STAGE=1
export N_SAMPLES=1024
export LAP_LAMBDA=0.1
export SYM_LAMBDA=0.0
export SV_CKPT_FREQ=20
export coef_multiplier=0.1
export dis_factor=2.5
export with_dis_loss="--with_dis_loss"
export gravity_weight=1.0
export n_folds=3



#### Basics for the deformation ####
export NUM_BASIS=64
export CVX_DIM=128
export N_KETPOINTS=256
export N_NEIGHBORING=128
export PRED_TYPE="basis"
export pred_offst="--pred_offset"
export USE_VAE_OPT="flow_pc"
export TAR_BASIS=0
export SYM_AXIS=1
export NET_PATH=""
export def_version="v4"
export with_glb=""
export RECON_COND="bbox"


#### Basics for data loading ####
export src_data_dir="./data/Shape2Motion_Deform/eyeglasses/none_motion"
export dst_data_dir="./data/Shape2Motion_Deform/eyeglasses/none_motion"
export SRC_FOLDER_FN="dst"
export DST_FOLDER_FN="dst"
export CVX_FOLDER_FN="dst"
export src_cvx_to_pts_sufix="_manifold_cvx_to_verts.npy"
export dst_cvx_to_pts_sufix="_manifold_cvx_to_verts.npy"
export cvx_to_pts_sufix="_cvx_to_verts_cdim_128.npy"
export cvx_list_filter="--cvx_list_filter"
export src_index=-1 ## 
export one_shp=""


#### Flag for the experiment ####
export EXP_FLAG=cages_pred_type_${PRED_TYPE}_cd_weight_allshp_e_0.90_wdis_rnds_cons_rnd_cage_


#### Trainer file ####
export TRAINER=train_def_cages.py






CUDA_VISIBLE_DEVICES=${CUDA_ID} python \
src/convex_deformation/${TRAINER} --batch_size=${BATCH_SIZE} --epoch=30000 --learning_rate=${lr}  --env_name='test' --num_basis=${NUM_BASIS} --display=${display} --step_size=100 --kl-weight=${KL_WEIGHT} --exp_flag=${EXP_FLAG}  --data_dir=${DATA_DIR} --n_keypoints=${N_KETPOINTS} --tar_basis=${TAR_BASIS} --load_meta  --pred_type=${PRED_TYPE}   --use_pointnet2 --use_pp_tar_out_feat --neighbouring_k=${N_NEIGHBORING} --dataset_name=v2 --n_samples=${N_SAMPLES} --symmetry_axis=${SYM_AXIS}  --net_path=${net_path} --local_handle --hier_stage=${HIER_STAGE} --sv_ckpt_freq=${SV_CKPT_FREQ}   --with_cat_deform=0   --use_paired_data --wo_keypts_abs --lap_lambda=${LAP_LAMBDA}  --sym_lambda=${SYM_LAMBDA}  --use_cvx_feats --use_def_pc --use_trans_range  --use_vae_opt=${USE_VAE_OPT} --pred_positions --src_folder_fn=${SRC_FOLDER_FN} --dst_folder_fn=${DST_FOLDER_FN} --use_recon_as_cd --recon_cond=${RECON_COND} --cond_tar_pc --cvx_dim=${CVX_DIM} --only_tar  --cvx_to_pts_sufix=${cvx_to_pts_sufix} --coef_multiplier=${coef_multiplier}   ${with_glb} ${one_shp} ${with_dis_loss} --def_version=${def_version} --dis_factor=${dis_factor} --pred_offset --n_fold=${n_folds}  --gravity_weight=${gravity_weight} ${by_part}  ${optimize_coef} ${with_image_loss} ${optimize_basis}  --cvx_folder_fn=${CVX_FOLDER_FN} --glb_template=${glb_template} --n_shots=${n_shots} ${cvx_list_filter} --src_data_dir=${src_data_dir} --src_n_keypoints=${src_n_keypoints} --dst_data_dir=${dst_data_dir} --src_cvx_to_pts_sufix=${src_cvx_to_pts_sufix} --dst_cvx_to_pts_sufix=${dst_cvx_to_pts_sufix} --src_index=${src_index}

