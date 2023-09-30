
export PYTHONPATH=.



export NUM_BASIS=64
export N_KETPOINTS=256
export N_NEIGHBORING=128


export BATCH_SIZE=1 # 2 # 4
export KL_WEIGHT=0.000001 # 1.0
export TAR_BASIS=0
# export TAR_BASIS=1
export PRED_TYPE="basis" 

export SYM_AXIS=1
# export SYM_AXIS=0



export DATA_DIR="/data/datasets/genn/Shape2Motion_Deform_key_256/eyeglasses/none_motion"



export EXP_FLAG="eye_def_2_basis_8_sv_dicts_02_forward2_nb_16_w_dis_loss_"


export  NET_PATH=""


# export TRAINER=train_target_driven_prob_pair.py
# export TRAINER=train_target_driven_paired_data_hier_deform.py
# export TRAINER=train_target_driven_paired_data_hier_deform_v2.py
# export TRAINER=train_def_v6.py
# export TRAINER=train_target_driven_paired_data_hier_deform_v2.py
export TRAINER=train_def_cages.py


export N_LAYERS=5


export CUDA_ID=7

export HIER_STAGE=1

export N_SAMPLES=1024
# export HIER_STAGE=0

export COEF_MULTIPLIER=0.1
export COEF_MULTIPLIER=0.2
# export COEF_MULTIPLIER=1.0

export LAP_LAMBDA=0.2
export LAP_LAMBDA=0.1 #### origninal meshes...
# export LAP_LAMBDA=0.0 #### origninal meshes...

export SYM_LAMBDA=1.0
export SYM_LAMBDA=0.0

export SV_CKPT_FREQ=20
# export SV_CKPT_FREQ=100

export USE_VAE_OPT="flow_cvx"
export USE_VAE_OPT="flow_pc"
# export USE_VAE_OPT="diffusion"
export SRC_FOLDER_FN="src_bsp_v2"

export SRC_FOLDER_FN="dst"
export DST_FOLDER_FN="dst"

export CVX_FOLDER_FN=""

# Eyeglasses
### tttt# s
export def_version="v3"
export def_version="v4"
export LAP_LAMBDA=0.4
# export LAP_LAMBDA=0.0
export LAP_LAMBDA=4.0
# export LAP_LAMBDA=0.1
export coef_multiplier=0.5
# export coef_multiplier=0.05
export coef_multiplier=0.4
export coef_multiplier=0.1
export dis_factor=2.5
export NUM_BASIS=16
# export NUM_BASIS=8
export NUM_BASIS=32
# # export NUM_BASIS=8
# export NUM_BASIS=64
# export NUM_BASIS=4
# export N_KETPOINTS=1024 ### nkeypoints
export N_KETPOINTS=512 ### nkeypoints
export N_KETPOINTS=256 ### nkeypoin ### 256 keypts ###
export N_KETPOINTS=64 ### nkeypoin ### 256 keypts ###
export N_KETPOINTS=16 ### nkeypoin ### 256 keypts ###
# export N_KETPOINTS=8 ### nkeypoin ### 256 keypts ###
export DATA_DIR="/data/datasets/genn/Shape2Motion_Deform_key_256/eyeglasses/none_motion"
export PRED_TYPE="offset"
export PRED_TYPE="basis"
export use_gt_cages="--use_gt_cages"
# export use_gt_cages=""
export cd_weight=1.0
export cd_weight=0.0
########## one shape with ###########
# export EXP_FLAG=eye_02_nb_${NUM_BASIS}_nk_${N_KETPOINTS}_fr3_mult_coef_${coef_multiplier}_lap_${LAP_LAMBDA}_noneshp_wdis_ ###
# export EXP_FLAG=eye_02_nb_${NUM_BASIS}_nk_${N_KETPOINTS}_fr3_mult_coef_${coef_multiplier}_lap_${LAP_LAMBDA}_oneshp_wdis_
# export EXP_FLAG=eye_02_nb_${NUM_BASIS}_nk_${N_KETPOINTS}__mult_coef_${coef_multiplier}_lap_${LAP_LAMBDA}_noneshp_wdis_def_${def_version}_${dis_factor}nn_tst_basis_wdis_tar_coef_wk_reg_b_w_sm_tar_
# export EXP_FLAG=eye_02_nb_${NUM_BASIS}_nk_${N_KETPOINTS}__mult_coef_${coef_multiplier}_lap_${LAP_LAMBDA}_noneshp_wdis_def_${def_version}_${dis_factor}nn_tst_basis_wdis_tar_coef_kk_norm_
export EXP_FLAG=cages_pred_type_${PRED_TYPE}_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_allshp_e_0.90_wdis_rnds_cons_rnd_cage_
export SRC_FOLDER_FN="dst"
export DST_FOLDER_FN="dst"
export cvx_to_pts_sufix="_cvx_to_verts_cdim_128.npy"
export one_shp="--one_shp"
export one_shp=""
export with_dis_loss="--with_dis_loss"
# export with_dis_loss=""
export pred_offst="--pred_offset"
# export pred_offst="" ## pred_offsets
export small_tar_nn="--small_tar_nn"
export small_tar_nn=""
export lr=0.0001
export gravity_weight=1.0
export n_folds=3
export by_part="--by_part"
export optimize_coef="--optimize_coef"
# epxort by_part=""
export with_image_loss="--with_image_loss"
export with_image_loss=""
# export net_path="/share/xueyi/datasets/gen/ckpts/DeepMetaHandles/chair_cages_pred_type_basis_gt_cages__nn_pts_with_dist_offset_pred_optim_coef_only_reg_loss_cd_weight_0.0_allshp_e_0.90_wdis__0.0001_kl_1e-06_use_prob_0_src_prob_0_scaling_False_tar_basis_0_num_basis_8_n_keypoints_1024_neighbouring_k_128/net_0.pth"
export net_path="/share/xueyi/datasets/gen/ckpts/DeepMetaHandles/chair_cages_pred_type_basis_gt_cages__nn_pts_with_dist_offset_pred_optim_coef_only_reg_loss_cd_weight_0.0_allshp_e_0.90_wdis__0.0001_kl_1e-06_use_prob_0_src_prob_0_scaling_False_tar_basis_0_num_basis_8_n_keypoints_1024_neighbouring_k_128/net_60.pth"
export net_path=""
export optimize_basis="--optimize_basis"
# export optimize_basis=""
export rnd_sample_nn=5


# #### Chair datasest ####
# export DATA_DIR="/data/datasets/genn/ShapeNetCoreV2_Deform/03001627_c_512"
# export EXP_FLAG=chair_deform_nb_${NUM_BASIS}_nk_${N_KETPOINTS}_fr3_normed_glb_feats_1_nn_n_norm_7cvx_
# export SRC_FOLDER_FN="dst_def"
# export DST_FOLDER_FN="dst_def"
# export cvx_to_pts_sufix="_cvx_to_cvx_pts_n_512.npy"

### sample points from meshes ###

#### Drawer link 1 ####
# export def_version="v3"
# export def_version="v4"
# export LAP_LAMBDA=0.4
# # export LAP_LAMBDA=0.0
# export LAP_LAMBDA=4.0
# # export LAP_LAMBDA=0.1
# export coef_multiplier=0.5
# # export coef_multiplier=0.05
# export coef_multiplier=0.1
# export dis_factor=2.5
# # export NUM_BASIS=16
# export NUM_BASIS=8
# export NUM_BASIS=32
# export N_KETPOINTS=64 ### nkeypoin ### 256 keypts ###
# # export N_KETPOINTS=8 ### nkeypoin ### 256 keypts ###
# export DATA_DIR="/data/datasets/genn/Shape2Motion_Deform_key_256/eyeglasses/none_motion"
# export PRED_TYPE="offset"
# export PRED_TYPE="basis"
# export use_gt_cages="--use_gt_cages"
# # export use_gt_cages=""
# export cd_weight=1.0
# export cd_weight=0.0
# export EXP_FLAG=drawer_l1_pred_type_${PRED_TYPE}_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_allshp_e_0.90_wdis_rnd_v2_rnd_coef_
# export SRC_FOLDER_FN="dst"
# export DST_FOLDER_FN="dst"
# # /data/datasets/genn/SAPIEN_Deform/StorageFurniture_Style3/link_1/dst/inst_0_cvx_to_verts.npy 
# export cvx_to_pts_sufix="_cvx_to_verts.npy"  ### cvx_to_verts sufix ###
# export one_shp="--one_shp" ### 
# export one_shp=""
# export with_dis_loss="--with_dis_loss"
# export DATA_DIR="/data/datasets/genn/SAPIEN_Deform/StorageFurniture_Style3/link_1"
# export pred_offst="--pred_offset"
# # export pred_offst="" ## pred_offsets
# export small_tar_nn="--small_tar_nn"
# export small_tar_nn=""
# export lr=0.0001
# export gravity_weight=1.0
# export n_folds=3
# export by_part="--by_part"
# # epxort by_part=""
# export optimize_coef="--optimize_coef"
# export with_image_loss="--with_image_loss"
# export with_image_loss=""
# # export net_path="/share/xueyi/datasets/gen/ckpts/DeepMetaHandles/chair_cages_pred_type_basis_gt_cages__nn_pts_with_dist_offset_pred_optim_coef_only_reg_loss_cd_weight_0.0_allshp_e_0.90_wdis__0.0001_kl_1e-06_use_prob_0_src_prob_0_scaling_False_tar_basis_0_num_basis_8_n_keypoints_1024_neighbouring_k_128/net_0.pth"
# export net_path="/share/xueyi/datasets/gen/ckpts/DeepMetaHandles/chair_cages_pred_type_basis_gt_cages__nn_pts_with_dist_offset_pred_optim_coef_only_reg_loss_cd_weight_0.0_allshp_e_0.90_wdis__0.0001_kl_1e-06_use_prob_0_src_prob_0_scaling_False_tar_basis_0_num_basis_8_n_keypoints_1024_neighbouring_k_128/net_60.pth"
# export net_path=""
# export optimize_basis="--optimize_basis"
# # export optimize_basis=""
# export rnd_sample_nn=5





# # Chair
# export coef_multiplier=0.2
export coef_multiplier=0.2
# export CUDA_ID=5
export NUM_BASIS=16
export NUM_BASIS=8
export NUM_BASIS=32
export N_KETPOINTS=1024 ### nkeypoints
# export N_KETPOINTS=16 ### nkeypoin ### 256 keypts ###
# export N_KETPOINTS=512 ### nkeypoints
export PRED_TYPE="basis"
export use_gt_cages="--use_gt_cages"
# export use_gt_cages=""
export cd_weight=1.0
export cd_weight=0.0
export EXP_FLAG=chair_pred_type_${PRED_TYPE}_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_e_0.90_wdis_glb_def__rnd_v2_mult_0.95_coef_def_tem_v2_
export DATA_DIR="/data/datasets/genn/ShapeNetCoreV2_Deform/03001627_c_512"
# export EXP_FLAG=chair_deform_nb_${NUM_BASIS}_nk_${N_KETPOINTS}_fr3_normed_glb_feats_1_nn_n_norm_7cvx_
export SRC_FOLDER_FN="dst_def"
export DST_FOLDER_FN="dst_def"
export cvx_to_pts_sufix="_cvx_to_cvx_pts_n_512.npy"
export one_shp="--one_shp" ### 
export one_shp=""
export with_dis_loss="--with_dis_loss"
# export with_dis_loss=""
export pred_offst="--pred_offset" ### rped_offset 
# export pred_offst="" ## pred_offsets
export small_tar_nn="--small_tar_nn"
export small_tar_nn=""

export lr=0.0001
export gravity_weight=1.0
export n_folds=3
export by_part="--by_part"
export optimize_coef="--optimize_coef"

export with_image_loss="--with_image_loss"
export with_image_loss=""
export net_path="/share/xueyi/datasets/gen/ckpts/DeepMetaHandles/chair_cages_pred_type_basis_gt_cages__nn_pts_with_dist_offset_pred_optim_coef_only_reg_loss_cd_weight_0.0_allshp_e_0.90_wdis__0.0001_kl_1e-06_use_prob_0_src_prob_0_scaling_False_tar_basis_0_num_basis_8_n_keypoints_1024_neighbouring_k_128/net_0.pth"
export net_path=""
export optimize_basis="--optimize_basis"
# export optimize_basis=""
export rnd_sample_nn=5

# /data/datasets/genn/ShapeNetCoreV2_Deform/03636649_c_512

# Lamp
export coef_multiplier=0.2
export NUM_BASIS=32
export N_KETPOINTS=256
export PRED_TYPE="basis"
export use_gt_cages="--use_gt_cages"
export cd_weight=0.0
export EXP_FLAG=lamp_pred_type_${PRED_TYPE}_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_e_0.90_wdis_glb_def__rnd_v2_mult_0.95_coef_def_sv_ori_def_glb_cage_cv4_
export DATA_DIR="/data/datasets/genn/ShapeNetCoreV2_Deform/03636649_c_512"
# export EXP_FLAG=chair_deform_nb_${NUM_BASIS}_nk_${N_KETPOINTS}_fr3_normed_glb_feats_1_nn_n_norm_7cvx_
export SRC_FOLDER_FN="dst_def"
export DST_FOLDER_FN="dst_def"
export CVX_FOLDER_FN="src_bsp"
# /data/datasets/genn/ShapeNetCoreV2_Deform/03636649_c_512/src_bsp/1a9c1cbf1ca9ca24274623f5a5d0bcdc_cvx_to_verts.npy
# export cvx_to_pts_sufix="_cvx_to_cvx_pts_n_512.npy"
export cvx_to_pts_sufix="_cvx_to_verts.npy"
export one_shp="--one_shp" ### 
export one_shp=""
export with_dis_loss="--with_dis_loss"
# export with_dis_loss=""
export pred_offst="--pred_offset"
# export pred_offst=""
export small_tar_nn="--small_tar_nn"
export small_tar_nn=""
export lr=0.0001
export gravity_weight=1.0
export n_folds=3
export by_part="--by_part"
export optimize_coef="--optimize_coef"
export with_image_loss="--with_image_loss"
export with_image_loss=""
export net_path=""
export optimize_basis="--optimize_basis"
export rnd_sample_nn=5

# Table # /data/datasets/genn/ShapeNetCoreV2_Deform/04379243_c_512/dst_def
export coef_multiplier=0.2
export NUM_BASIS=32
export N_KETPOINTS=1024 ### nkeypoints
export PRED_TYPE="basis"
export use_gt_cages="--use_gt_cages"
export cd_weight=1.0
export cd_weight=0.0
export EXP_FLAG=table_pred_type_${PRED_TYPE}_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_e_0.90_wdis_glb_def__rnd_v2_mult_0.95_coef_def_tem_v2_
export DATA_DIR="/data/datasets/genn/ShapeNetCoreV2_Deform/04379243_c_512"
export SRC_FOLDER_FN="dst_def"
export DST_FOLDER_FN="dst_def"
export cvx_to_pts_sufix="_cvx_to_verts.npy"
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
export net_path="/share/xueyi/datasets/gen/ckpts/DeepMetaHandles/table_pred_type_basis_gt_cages_--use_gt_cages_cd_weight_0.0_e_0.90_wdis_glb_def__rnd_v2_mult_0.95_coef_def_tem_v2__0.0001_kl_1e-06_use_prob_0_src_prob_0_scaling_False_tar_basis_0_num_basis_32_n_keypoints_1024_neighbouring_k_128/net_100.pth"
export optimize_basis="--optimize_basis"
export rnd_sample_nn=5

# Eyeglasses - dof_rootd_Aa001_r --> /share/xueyi/datasets/gen/datasets/Shape2Motion_Deform/eyeglasses/dof_rootd_Aa001_r/dst
### tttt# s
export def_version="v4"
export LAP_LAMBDA=4.0
export coef_multiplier=0.2
export dis_factor=2.5
export NUM_BASIS=32
export N_KETPOINTS=64 ### nkeypoin ### 256 keypts ###
export DATA_DIR="/share/xueyi/datasets/gen/datasets/Shape2Motion_Deform/eyeglasses/dof_rootd_Aa001_r"
export DATA_DIR="/share/xueyi/datasets/gen/datasets/Shape2Motion_Deform/eyeglasses/none_motion"
export PRED_TYPE="basis"
export use_gt_cages="--use_gt_cages"
export cd_weight=1.0
export cd_weight=0.0
export n_shots=8
export EXP_FLAG=eye_legs_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_shots_${n_shots}_
export SRC_FOLDER_FN="dst"
export DST_FOLDER_FN="dst"
export CVX_FOLDER_FN="src_bsp_c_3"
export cvx_to_pts_sufix="_cvx_to_cvx_pts_n_512.npy"
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


# # drawer link 2 --> /data/datasets/genn/SAPIEN_Deform/StorageFurniture_Style3/link_2/dst
# export def_version="v4"
# export LAP_LAMBDA=4.0
# export coef_multiplier=0.2
# export dis_factor=2.5
# export NUM_BASIS=16
# export N_KETPOINTS=64 ### nkeypoin ### 256 keypts ###
# export DATA_DIR="/data/datasets/genn/SAPIEN_Deform/StorageFurniture_Style3/link_2"
# export PRED_TYPE="basis"
# export use_gt_cages="--use_gt_cages"
# export cd_weight=0.0
# export EXP_FLAG=darwer_link2_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_
# export SRC_FOLDER_FN="dst"
# export DST_FOLDER_FN="dst"
# export CVX_FOLDER_FN="dst"
# export cvx_to_pts_sufix="_manifold_cvx_to_verts.npy"
# export one_shp=""
# export with_dis_loss="--with_dis_loss"
# export pred_offst="--pred_offset"
# export small_tar_nn=""
# export lr=0.0001
# export gravity_weight=1.0
# export n_folds=3
# export by_part="--by_part"
# export optimize_coef="--optimize_coef"
# export with_image_loss=""
# export net_path=""
# export optimize_basis="--optimize_basis"
# export rnd_sample_nn=5

# # drawer link 1 --> /data/datasets/genn/SAPIEN_Deform/StorageFurniture_Style3/link_1/dst
# export def_version="v4"
# export LAP_LAMBDA=4.0
# export coef_multiplier=0.2
# export dis_factor=2.5
# export NUM_BASIS=16
# export N_KETPOINTS=64 ### nkeypoin ### 256 keypts ###
# export DATA_DIR="/data/datasets/genn/SAPIEN_Deform/StorageFurniture_Style3/link_1"
# export PRED_TYPE="basis"
# export use_gt_cages="--use_gt_cages"
# export cd_weight=0.0
# export EXP_FLAG=darwer_link1_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_
# export SRC_FOLDER_FN="dst"
# export DST_FOLDER_FN="dst"
# export CVX_FOLDER_FN="dst"
# export cvx_to_pts_sufix="_cvx_to_verts.npy"
# export one_shp=""
# export with_dis_loss="--with_dis_loss"
# export pred_offst="--pred_offset"
# export small_tar_nn=""
# export lr=0.0001
# export gravity_weight=1.0
# export n_folds=3
# export by_part="--by_part"
# export optimize_coef="--optimize_coef"
# export with_image_loss=""
# export net_path=""
# export optimize_basis="--optimize_basis"
# export rnd_sample_nn=5


# # drawer link 0 --> /data/datasets/genn/SAPIEN_Deform/StorageFurniture_Style3/link_0/dst
export def_version="v4"
export LAP_LAMBDA=4.0
export coef_multiplier=0.2
export dis_factor=2.5
export NUM_BASIS=16
export n_shots=8
export N_KETPOINTS=64 ### nkeypoin ### 256 keypts ###
export DATA_DIR="/data/datasets/genn/SAPIEN_Deform/StorageFurniture_Style3/link_0"
export DATA_DIR="/data/datasets/genn/SAPIEN_Deform/StorageFurniture_Style3/link_1"
export DATA_DIR="/data/datasets/genn/SAPIEN_Deform/StorageFurniture_Style3/link_2"
export PRED_TYPE="basis"
export use_gt_cages="--use_gt_cages"
export cd_weight=0.0
export EXP_FLAG=darwer_link0_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_shots_${n_shots}_
export EXP_FLAG=darwer_link1_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_shots_${n_shots}_
export EXP_FLAG=darwer_link2_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_shots_${n_shots}_
export SRC_FOLDER_FN="dst"
export DST_FOLDER_FN="dst"
export CVX_FOLDER_FN="dst"
export cvx_to_pts_sufix="_cvx_to_verts.npy"
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


# scissors -> link0/0 /data/datasets/genn/SAPIEN_Deform/Scissors/link_0/dst
# export def_version="v4"
# export LAP_LAMBDA=4.0
# export coef_multiplier=0.2
# export dis_factor=2.5
# export NUM_BASIS=16
# export N_KETPOINTS=64 ### nkeypoin ### 256 keypts ###
# export DATA_DIR="/data/datasets/genn/SAPIEN_Deform/Scissors/link_0"
# export DATA_DIR="/data/datasets/genn/SAPIEN_Deform/Scissors/link_1"
# export PRED_TYPE="basis"
# export use_gt_cages="--use_gt_cages"
# export cd_weight=0.0
# export n_shots=8
# # export EXP_FLAG=scissors_link0_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_shots_${n_shots}_
# export EXP_FLAG=scissors_link1_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_shots_${n_shots}_
# export SRC_FOLDER_FN="dst"
# export DST_FOLDER_FN="dst"
# export CVX_FOLDER_FN="dst"
# export cvx_to_pts_sufix="_cvx_to_verts.npy"
# export one_shp=""
# export with_dis_loss="--with_dis_loss"
# export pred_offst="--pred_offset"
# export small_tar_nn=""
# export lr=0.0001
# export gravity_weight=1.0
# export n_folds=3
# export by_part="--by_part"
# export optimize_coef="--optimize_coef"
# export with_image_loss=""
# export net_path=""
# export optimize_basis="--optimize_basis"
# export rnd_sample_nn=5

# Eyeglasses -> link0/0 /data/datasets/genn/SAPIEN_Deform/Eyeglasses/link_2/dst
# export def_version="v4"
# export LAP_LAMBDA=4.0
# export coef_multiplier=0.2
# export dis_factor=2.5
# export NUM_BASIS=16
# export N_KETPOINTS=64 ### nkeypoin ### 256 keypts ###
# export DATA_DIR="/data/datasets/genn/SAPIEN_Deform/Eyeglasses/link_2"
# # export DATA_DIR="/data/datasets/genn/SAPIEN_Deform/Eyeglasses/link_1"
# export PRED_TYPE="basis"
# export use_gt_cages="--use_gt_cages"
# export cd_weight=0.0
# export n_shots=8
# export EXP_FLAG=eye_link2_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_shots_${n_shots}
# # export EXP_FLAG=eye_link1_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_shots_${n_shots}
# export SRC_FOLDER_FN="dst"
# export DST_FOLDER_FN="dst"
# export CVX_FOLDER_FN="dst"
# export cvx_to_pts_sufix="_cvx_to_verts.npy"
# export one_shp=""
# export with_dis_loss="--with_dis_loss"
# export pred_offst="--pred_offset"
# export small_tar_nn=""
# export lr=0.0001
# export gravity_weight=1.0
# export n_folds=3
# export by_part="--by_part"
# export optimize_coef="--optimize_coef"
# export with_image_loss=""
# export net_path=""
# export optimize_basis="--optimize_basis"
# export rnd_sample_nn=5


# # /data/datasets/genn/SAPIEN_Deform/Oven/link_1


# Oven -> link0/0 /data/datasets/genn/SAPIEN_Deform/Eyeglasses/link_2/dst
export def_version="v4"
export n_shots=8
export LAP_LAMBDA=4.0
export coef_multiplier=0.2
export dis_factor=2.5
export NUM_BASIS=16
export N_KETPOINTS=64 ### nkeypoin ### 256 keypts ###
export DATA_DIR="/data/datasets/genn/SAPIEN_Deform/Oven/link_1"
# export DATA_DIR="/data/datasets/genn/SAPIEN_Deform/Oven/link_0"
export PRED_TYPE="basis"
export use_gt_cages="--use_gt_cages"
export cd_weight=0.0
# export EXP_FLAG=eye_link2_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_
export EXP_FLAG=oven_link1_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_n_shots_${n_shots}_
# export EXP_FLAG=oven_link0_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_n_shots_${n_shots}_
export SRC_FOLDER_FN="dst"
export DST_FOLDER_FN="dst"
export CVX_FOLDER_FN="dst"
export cvx_to_pts_sufix="_cvx_to_verts.npy"
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
export CUDA_ID=1



# # Micro -> /data/datasets/genn/SAPIEN_Deform/Microwave/link_1/dst
# export def_version="v4"
# export LAP_LAMBDA=4.0
# export coef_multiplier=0.2
# export dis_factor=2.5
# export NUM_BASIS=16
# export N_KETPOINTS=64 ### nkeypoin ### 256 keypts ###
# export DATA_DIR="/data/datasets/genn/SAPIEN_Deform/Microwave/link_1/"
# # export DATA_DIR="/data/datasets/genn/SAPIEN_Deform/Microwave/link_0/"
# export PRED_TYPE="basis"
# export use_gt_cages="--use_gt_cages"
# export cd_weight=0.0
# # export EXP_FLAG=eye_link2_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_
# export EXP_FLAG=micro_link1_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_
# # export EXP_FLAG=micro_link0_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_
# export SRC_FOLDER_FN="dst"
# export DST_FOLDER_FN="dst"
# export CVX_FOLDER_FN="dst"
# export cvx_to_pts_sufix="_cvx_to_verts.npy"
# export one_shp=""
# export with_dis_loss="--with_dis_loss"
# export pred_offst="--pred_offset"
# export small_tar_nn=""
# export lr=0.0001
# export gravity_weight=1.0
# export n_folds=3
# export by_part="--by_part"
# export optimize_coef="--optimize_coef"
# export with_image_loss=""
# export net_path=""
# export optimize_basis="--optimize_basis"
# export rnd_sample_nn=5


# # Lamp -> /data/datasets/genn/SAPIEN_Deform/Lamp/link_0/dst
# export def_version="v4"
# export LAP_LAMBDA=4.0
# export n_shots=8
# export coef_multiplier=0.2
# export dis_factor=2.5
# export NUM_BASIS=16
# export N_KETPOINTS=64 ### nkeypoin ### 256 keypts ###
# export DATA_DIR="/data/datasets/genn/SAPIEN_Deform/Lamp/link_0/"
# export DATA_DIR="/data/datasets/genn/SAPIEN_Deform/Lamp/link_1/"
# export DATA_DIR="/data/datasets/genn/SAPIEN_Deform/Lamp/link_2/"
# export DATA_DIR="/data/datasets/genn/SAPIEN_Deform/Lamp/link_3/"
# # export DATA_DIR="/data/datasets/genn/SAPIEN_Deform/Microwave/link_0/"
# export PRED_TYPE="basis"
# export use_gt_cages="--use_gt_cages"
# export cd_weight=0.0
# # export EXP_FLAG=eye_link2_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_
# export EXP_FLAG=lamp_link0_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_n_shots_${n_shots}_
# export EXP_FLAG=lamp_link1_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_n_shots_${n_shots}_
# export EXP_FLAG=lamp_link2_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_n_shots_${n_shots}_
# export EXP_FLAG=lamp_link3_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_n_shots_${n_shots}_
# # export EXP_FLAG=micro_link0_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_n_shots_${n_shots}_
# export SRC_FOLDER_FN="dst"
# export DST_FOLDER_FN="dst"
# export CVX_FOLDER_FN="dst"
# export cvx_to_pts_sufix="_cvx_to_verts.npy"
# export one_shp=""
# export with_dis_loss="--with_dis_loss"
# export pred_offst="--pred_offset"
# export small_tar_nn=""
# export lr=0.0001
# export gravity_weight=1.0
# export n_folds=3
# export by_part="--by_part"
# export optimize_coef="--optimize_coef"
# export with_image_loss=""
# export net_path=""
# export optimize_basis="--optimize_basis"
# export rnd_sample_nn=5
# # export glb_template="/home/xueyi/gen/DeepMetaHandles/data/sphere_template_4.obj"
# export glb_template="/home/xueyi/gen/DeepMetaHandles/data/sphere_V42_F80.off" # Oven link0, drawer link2


# # TrashCan -> /data/datasets/genn/SAPIEN_Deform/Lamp/link_0/dst
# export def_version="v4"
# export LAP_LAMBDA=4.0
# export n_shots=8
# export coef_multiplier=0.2
# export dis_factor=2.5
# export NUM_BASIS=16
# export N_KETPOINTS=64 ### nkeypoin ### 256 keypts ###
# export DATA_DIR="/data/datasets/genn/SAPIEN_Deform/TrashCan/link_0/"
# # export DATA_DIR="/data/datasets/genn/SAPIEN_Deform/TrashCan/link_1/"
# # export DATA_DIR="/data/datasets/genn/SAPIEN_Deform/Microwave/link_0/"
# export PRED_TYPE="basis"
# export use_gt_cages="--use_gt_cages"
# export cd_weight=0.0
# # export EXP_FLAG=eye_link2_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_
# export EXP_FLAG=TrashCan_link0_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_n_shots_${n_shots}_
# # export EXP_FLAG=TrashCan_link1_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_n_shots_${n_shots}_
# # export EXP_FLAG=micro_link0_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_n_shots_${n_shots}_
# export SRC_FOLDER_FN="dst"
# export DST_FOLDER_FN="dst"
# export CVX_FOLDER_FN="dst"
# export cvx_to_pts_sufix="_cvx_to_verts.npy"
# export one_shp=""
# export with_dis_loss="--with_dis_loss"
# export pred_offst="--pred_offset"
# export small_tar_nn=""
# export lr=0.0001
# export gravity_weight=1.0
# export n_folds=3
# export by_part="--by_part"
# export optimize_coef="--optimize_coef"
# export with_image_loss=""
# export net_path=""
# export optimize_basis="--optimize_basis"
# export rnd_sample_nn=5
# # export glb_template="/home/xueyi/gen/DeepMetaHandles/data/sphere_template_4.obj"
# export glb_template="/home/xueyi/gen/DeepMetaHandles/data/sphere_V42_F80.off" # Oven link0, drawer link2


# Table -> /data/datasets/genn/ShapeNetCoreV2_Deform/04379243_c_512/dst_def
export def_version="v4"
export LAP_LAMBDA=4.0
export n_shots=8
# export n_shots=2000
export coef_multiplier=0.2
export dis_factor=2.5
export NUM_BASIS=16
export N_KETPOINTS=1024 ### nkeypoin ### 256 keypts ###
# export DATA_DIR="/data/datasets/genn/SAPIEN_Deform/TrashCan/link_0/"
# export DATA_DIR="/data/datasets/genn/SAPIEN_Deform/TrashCan/link_1/"
# export DATA_DIR="/data/datasets/genn/SAPIEN_Deform/Microwave/link_0/"
export DATA_DIR="/data/datasets/genn/ShapeNetCoreV2_Deform/04379243_c_512"
export PRED_TYPE="basis"
export use_gt_cages="--use_gt_cages"
export cd_weight=0.0
# export EXP_FLAG=eye_link2_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_
export EXP_FLAG=Table_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_n_shots_${n_shots}_
# export EXP_FLAG=TrashCan_link1_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_n_shots_${n_shots}_
# export EXP_FLAG=micro_link0_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_n_shots_${n_shots}_
export SRC_FOLDER_FN="dst_def"
export DST_FOLDER_FN="dst_def"
export CVX_FOLDER_FN="src_bsp"
export cvx_to_pts_sufix="_cvx_to_verts.npy"
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
export glb_template="/home/xueyi/gen/DeepMetaHandles/data/sphere_template_4.obj"
# export glb_template="/home/xueyi/gen/DeepMetaHandles/data/sphere_V42_F80.off" # Oven link0, drawer link2
export CUDA_ID=0
export cvx_list_filter=""
export cvx_list_filter="--cvx_list_filter"



### src_cvx_to_sufix ###
# Chair -> /data/datasets/genn/ShapeNetCoreV2_Deform/03001627_c_512
export def_version="v4"
export LAP_LAMBDA=4.0
export n_shots=8
# export n_shots=2000
export coef_multiplier=0.2
export dis_factor=2.5
export NUM_BASIS=16
export N_KETPOINTS=1024 ### nkeypoin ### 256 keypts ###
export DATA_DIR="/data/datasets/genn/ShapeNetCoreV2_Deform/03001627_c_512"
export PRED_TYPE="basis"
export use_gt_cages="--use_gt_cages"
export cd_weight=0.0
# export EXP_FLAG=eye_link2_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_
export EXP_FLAG=Chair_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_n_shots_${n_shots}_
# export EXP_FLAG=TrashCan_link1_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_n_shots_${n_shots}_
# export EXP_FLAG=micro_link0_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_n_shots_${n_shots}_
export SRC_FOLDER_FN="dst_def"
export DST_FOLDER_FN="dst_def"
export CVX_FOLDER_FN="src_bsp"
# /data/datasets/genn/ShapeNetCoreV2_Deform/03001627_c_512/src_bsp/1a6f615e8b1b5ae4dbbc9440457e303e_cvx_to_cvx_pts_n_512.npy
export cvx_to_pts_sufix="_cvx_to_cvx_pts_n_512.npy"
export src_cvx_to_pts_sufix="_cvx_to_cvx_pts_n_512.npy"
### cvX_to_pts dst_models ###
export dst_cvx_to_pts_sufix="_cvx_to_verts.npy" # /data/datasets/genn/ShapeNetCoreV2_Deform/04379243_c_512/src_bsp/1a00aa6b75362cc5b324368d54a7416f_cvx_to_verts.npy
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
export glb_template="/home/xueyi/gen/DeepMetaHandles/data/sphere_template_4.obj" ## template v4
# export glb_template="/home/xueyi/gen/DeepMetaHandles/data/sphere_V42_F80.off" # Oven link0, drawer link2
export CUDA_ID=0
export cvx_list_filter=""
export cvx_list_filter="--cvx_list_filter"
export src_data_dir="/data/datasets/genn/ShapeNetCoreV2_Deform/03001627_c_512"
export src_n_keypoints=1024 ### nkeypoin ### 256 keypts ### 
# export dst_data_dir="/data/datasets/genn/ShapeNetCoreV2_Deform/03001627_c_512" # chair
export dst_data_dir="/data/datasets/genn/ShapeNetCoreV2_Deform/04379243_c_512" # 04379243
export dst_n_keypoints=1024 ### nkeypoin ### 256 keypts ###
export cvx_list_filter=""


# Lamp -> /data/datasets/genn/ShapeNetCoreV2_Deform/03636649_c_512
# export def_version="v4"
# export LAP_LAMBDA=4.0
# export n_shots=8
# # export n_shots=2000
# export coef_multiplier=0.2
# export dis_factor=2.5
# export NUM_BASIS=16
# export N_KETPOINTS=256
# export DATA_DIR="/data/datasets/genn/ShapeNetCoreV2_Deform/03636649_c_512"
# export PRED_TYPE="basis"
# export use_gt_cages="--use_gt_cages"
# export cd_weight=0.0
# # export EXP_FLAG=eye_link2_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_
# export EXP_FLAG=Lamp_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_n_shots_${n_shots}_
# export SRC_FOLDER_FN="dst_def"
# export DST_FOLDER_FN="dst_def"
# export CVX_FOLDER_FN="src_bsp"
# export cvx_to_pts_sufix="_cvx_to_verts.npy"
# export one_shp=""
# export with_dis_loss="--with_dis_loss"
# export pred_offst="--pred_offset"
# export small_tar_nn=""
# export lr=0.0001
# export gravity_weight=1.0
# export n_folds=3
# export by_part="--by_part"
# export optimize_coef="--optimize_coef"
# export with_image_loss=""
# export net_path=""
# export optimize_basis="--optimize_basis"
# export rnd_sample_nn=5
# export glb_template="/home/xueyi/gen/DeepMetaHandles/data/sphere_template_4.obj" ## template v4
# # export glb_template="/home/xueyi/gen/DeepMetaHandles/data/sphere_V42_F80.off" # Oven link0, drawer link2
# export CUDA_ID=1
# export cvx_list_filter=""
# export cvx_list_filter="--cvx_list_filter"


### src_cvx_to_sufix ###
# Airplane -> /data/datasets/genn/ShapeNetCoreV2_Deform/03001627_c_512
export def_version="v4"
export LAP_LAMBDA=4.0
export n_shots=8
# export n_shots=2000
export coef_multiplier=0.2
export dis_factor=2.5
export NUM_BASIS=16
export N_KETPOINTS=1024 ### nkeypoin ### 256 keypts ###
export DATA_DIR="/data/datasets/genn/ShapeNetCoreV2_Deform/03001627_c_512"
export PRED_TYPE="basis"
export use_gt_cages="--use_gt_cages"
export cd_weight=0.0
# export EXP_FLAG=eye_link2_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_
export EXP_FLAG=Airplane_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_n_shots_${n_shots}_
# export EXP_FLAG=TrashCan_link1_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_n_shots_${n_shots}_
# export EXP_FLAG=micro_link0_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_n_shots_${n_shots}_
export SRC_FOLDER_FN="dst_def"
export DST_FOLDER_FN="dst_def"
export CVX_FOLDER_FN="src_bsp"
# /data/datasets/genn/ShapeNetCoreV2_Deform/03001627_c_512/src_bsp/1a6f615e8b1b5ae4dbbc9440457e303e_cvx_to_cvx_pts_n_512.npy
export cvx_to_pts_sufix="_cvx_to_cvx_pts_n_512.npy"
export src_cvx_to_pts_sufix="_cvx_to_cvx_pts_n_512.npy"
### cvX_to_pts dst_models ###
export dst_cvx_to_pts_sufix="_cvx_to_verts.npy" # /data/datasets/genn/ShapeNetCoreV2_Deform/04379243_c_512/src_bsp/1a00aa6b75362cc5b324368d54a7416f_cvx_to_verts.npy
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
export glb_template="/home/xueyi/gen/DeepMetaHandles/data/sphere_template_4.obj" ## template v4
# export glb_template="/home/xueyi/gen/DeepMetaHandles/data/sphere_V42_F80.off" # Oven link0, drawer link2
export CUDA_ID=0
### cvx_list_filter ###
export cvx_list_filter=""
export cvx_list_filter="--cvx_list_filter"
export src_data_dir="/data/datasets/genn/ShapeNetCoreV2_Deform/02691156_c_256"
export src_n_keypoints=64 ### nkeypoin ### 256 keypts ### 
# export dst_data_dir="/data/datasets/genn/ShapeNetCoreV2_Deform/03001627_c_512" # chair
# export dst_data_dir="/data/datasets/genn/ShapeNetCoreV2_Deform/04379243_c_512" # 04379243
export dst_data_dir="/data/datasets/genn/ShapeNetCoreV2_Deform/02691156_c_256" # 04379243
export dst_n_keypoints=64 ### nkeypoin ### 256 keypts ###
export src_cvx_to_pts_sufix="_cvx_to_verts.npy"
export dst_cvx_to_pts_sufix="_cvx_to_verts.npy"
export cvx_list_filter=""



# Eyeglasses -> link0/0 /data/datasets/genn/SAPIEN_Deform/Eyeglasses/link_2/dst
export def_version="v4"
export LAP_LAMBDA=4.0
export coef_multiplier=0.2
export dis_factor=2.5
export NUM_BASIS=16
export N_KETPOINTS=64 ### nkeypoin ### 256 keypts ###
export DATA_DIR="/data/datasets/genn/SAPIEN_Deform/Eyeglasses/link_2"
# export DATA_DIR="/data/datasets/genn/SAPIEN_Deform/Eyeglasses/link_1"
# export DATA_DIR="/data/datasets/genn/SAPIEN_Deform/Eyeglasses/link_0"
export PRED_TYPE="basis"
export use_gt_cages="--use_gt_cages"
export cd_weight=0.0
export n_shots=8
export EXP_FLAG=eye_link2_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_shots_${n_shots}_fr_table_
# export EXP_FLAG=eye_link1_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_shots_${n_shots}_fr_self_
# export EXP_FLAG=eye_link0_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_shots_${n_shots}_fr_self_
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
export glb_template="/home/xueyi/gen/DeepMetaHandles/data/sphere_template_4.obj" ## template v4... ##
# export glb_template="/home/xueyi/gen/DeepMetaHandles/data/sphere_V42_F80.off" # Oven link0, drawer link2
#### train_def_cages #####
export CUDA_ID=0
export cvx_list_filter=""
export src_data_dir="/data/datasets/genn/SAPIEN_Deform/Eyeglasses/link_2"
# export src_data_dir="/data/datasets/genn/SAPIEN_Deform/Eyeglasses/link_1"
# export src_data_dir="/data/datasets/genn/SAPIEN_Deform/Eyeglasses/link_0"
# # export DATA_DIR="/data/datasets/genn/ShapeNetCoreV2_Deform/04379243_c_512"
# export src_data_dir="/data/datasets/genn/ShapeNetCoreV2_Deform/04379243_c_512"
export src_n_keypoints=64 ### nkeypoin ### 256 keypts ### 

# export dst_data_dir="/data/datasets/genn/ShapeNetCoreV2_Deform/03001627_c_512" # 04379243 ### dst_data_dir...
# export dst_n_keypoints=1024 ### dst_1024 n_keypoints ###

export dst_data_dir="/data/datasets/genn/SAPIEN_Deform/Eyeglasses/link_2" # 04379243 ### dst_data_dir...
export dst_data_dir="/data/datasets/genn/SAPIEN_Deform/Eyeglasses/link_1" # 04379243 ### dst_data_dir...
export dst_data_dir="/data/datasets/genn/SAPIEN_Deform/Eyeglasses/link_0" # 04379243 ### dst_data_dir...
export dst_data_dir="/data/datasets/genn/ShapeNetCoreV2_Deform/04379243_c_512"
export dst_n_keypoints=1024

export DST_FOLDER_FN="dst_def"
export CVX_FOLDER_FN="src_bsp"
# export DST_FOLDER_FN="dst"
# export CVX_FOLDER_FN="dst"

export src_cvx_to_pts_sufix="_cvx_to_verts.npy"
# export dst_cvx_to_pts_sufix="_cvx_to_cvx_pts_n_512.npy"
export dst_cvx_to_pts_sufix="_cvx_to_verts.npy"
export cvx_list_filter=""
# export cvx_list_filter="--cvx_list_filter"


# # TrashCan -> /data/datasets/genn/SAPIEN_Deform/Lamp/link_0/dst
# export def_version="v4"
# export LAP_LAMBDA=4.0
# export n_shots=8
# export coef_multiplier=0.2
# export dis_factor=2.5
# export NUM_BASIS=16
# export N_KETPOINTS=64 ### nkeypoin ### 256 keypts ###
# export DATA_DIR="/data/datasets/genn/SAPIEN_Deform/TrashCan/link_0/"
# # export DATA_DIR="/data/datasets/genn/SAPIEN_Deform/TrashCan/link_1/"
# # export DATA_DIR="/data/datasets/genn/SAPIEN_Deform/Microwave/link_0/"
# export PRED_TYPE="basis"
# export use_gt_cages="--use_gt_cages"
# export cd_weight=0.0
# # export EXP_FLAG=eye_link2_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_
# export EXP_FLAG=TrashCan_link0_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_n_shots_${n_shots}_
# # export EXP_FLAG=TrashCan_link1_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_n_shots_${n_shots}_
# # export EXP_FLAG=micro_link0_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_n_shots_${n_shots}_
# export SRC_FOLDER_FN="dst"
# export DST_FOLDER_FN="dst"
# export CVX_FOLDER_FN="dst"
# export cvx_to_pts_sufix="_cvx_to_verts.npy"
# export one_shp=""
# export with_dis_loss="--with_dis_loss"
# export pred_offst="--pred_offset"
# export small_tar_nn=""
# export lr=0.0001
# export gravity_weight=1.0
# export n_folds=3
# export by_part="--by_part"
# export optimize_coef="--optimize_coef"
# export with_image_loss=""
# export net_path=""
# export optimize_basis="--optimize_basis"
# export rnd_sample_nn=5
# # export glb_template="/home/xueyi/gen/DeepMetaHandles/data/sphere_template_4.obj"
# export glb_template="/home/xueyi/gen/DeepMetaHandles/data/sphere_V42_F80.off" # Oven link0, drawer link2


# Oven -> link0/0 /data/datasets/genn/SAPIEN_Deform/Eyeglasses/link_2/dst
# export def_version="v4"
# export n_shots=8
# export LAP_LAMBDA=4.0
# export coef_multiplier=0.2
# export dis_factor=2.5
# export NUM_BASIS=16
# export N_KETPOINTS=64 ### nkeypoin ### 256 keypts ###
# export DATA_DIR="/data/datasets/genn/SAPIEN_Deform/Oven/link_1"
# # export DATA_DIR="/data/datasets/genn/SAPIEN_Deform/Oven/link_0"
# export PRED_TYPE="basis"
# export use_gt_cages="--use_gt_cages"
# export cd_weight=0.0
# # export EXP_FLAG=eye_link2_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_
# export EXP_FLAG=Oven_link1_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_n_shots_${n_shots}_fr_self_ ## Oven_link1_
# # export EXP_FLAG=Oven_link0_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_n_shots_${n_shots}_fr_self_
# # export EXP_FLAG=oven_link0_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_n_shots_${n_shots}_
# export SRC_FOLDER_FN="dst"
# export DST_FOLDER_FN="dst"
# export CVX_FOLDER_FN="dst"
# export cvx_to_pts_sufix="_cvx_to_verts.npy"
# export one_shp=""
# export with_dis_loss="--with_dis_loss"
# export pred_offst="--pred_offset"
# export small_tar_nn=""
# export lr=0.0001
# export gravity_weight=1.0
# export n_folds=3
# export by_part="--by_part"
# export optimize_coef="--optimize_coef"
# export with_image_loss=""
# export net_path=""
# export optimize_basis="--optimize_basis"
# export rnd_sample_nn=5
# export CUDA_ID=1
# export cvx_list_filter=""
# export src_data_dir="/data/datasets/genn/SAPIEN_Deform/Oven/link_1"
# # export src_data_dir="/data/datasets/genn/SAPIEN_Deform/Oven/link_0"
# export src_n_keypoints=64 ### nkeypoin ### 256 keypts ### 
# # export dst_data_dir="/data/datasets/genn/ShapeNetCoreV2_Deform/03001627_c_512" # 04379243 ### dst_data_dir...
# # export dst_n_keypoints=1024 ### dst_1024 n_keypoints ###
# export dst_data_dir="/data/datasets/genn/SAPIEN_Deform/Oven/link_1" # 04379243 ### dst_data_dir...
# # export dst_data_dir="/data/datasets/genn/SAPIEN_Deform/Oven/link_0"
# export dst_n_keypoints=64
# # export DST_FOLDER_FN="dst_def"
# # export CVX_FOLDER_FN="src_bsp"
# export DST_FOLDER_FN="dst"
# export CVX_FOLDER_FN="dst"
# export src_cvx_to_pts_sufix="_cvx_to_verts.npy"
# # export dst_cvx_to_pts_sufix="_cvx_to_cvx_pts_n_512.npy"
# export dst_cvx_to_pts_sufix="_cvx_to_verts.npy"
# export cvx_list_filter=""
# export cvx_list_filter="--cvx_list_filter"



# Table -> /data/datasets/genn/ShapeNetCoreV2_Deform/04379243_c_512/dst_def
export def_version="v4"
export LAP_LAMBDA=4.0
# export n_shots=32
export n_shots=3000
export coef_multiplier=0.2
export dis_factor=2.5
export NUM_BASIS=16
export N_KETPOINTS=1024 ### nkeypoin ### 256 keypts ###
export DATA_DIR="/data/datasets/genn/ShapeNetCoreV2_Deform/04379243_c_512"
export PRED_TYPE="basis"
export use_gt_cages="--use_gt_cages"
export cd_weight=0.0
export EXP_FLAG=Table_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_n_shots_${n_shots}_basis_dict_
export SRC_FOLDER_FN="dst_def"
export DST_FOLDER_FN="dst_def"
export CVX_FOLDER_FN="src_bsp"
export cvx_to_pts_sufix="_cvx_to_verts.npy"
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
# export glb_template="/home/xueyi/gen/DeepMetaHandles/data/sphere_template_4.obj"
export glb_template="/home/xueyi/gen/DeepMetaHandles/data/sphere_V42_F80.off" # Oven link0, drawer link2
export cvx_list_filter=""
export cvx_list_filter="--cvx_list_filter"
export src_data_dir="/data/datasets/genn/ShapeNetCoreV2_Deform/04379243_c_512"
# export src_data_dir="/data/datasets/genn/SAPIEN_Deform/Oven/link_0"
export src_n_keypoints=1024 ### nkeypoin ### 256 keypts ### 
# export dst_data_dir="/data/datasets/genn/ShapeNetCoreV2_Deform/03001627_c_512" # 04379243 ### dst_data_dir...
# export dst_n_keypoints=1024 ### dst_1024 n_keypoints ###
export dst_data_dir="/data/datasets/genn/ShapeNetCoreV2_Deform/04379243_c_512" # 04379243 ### dst_data_dir...
# export dst_data_dir="/data/datasets/genn/SAPIEN_Deform/Oven/link_0"
export dst_n_keypoints=1024
# export DST_FOLDER_FN="dst_def"
# export CVX_FOLDER_FN="src_bsp"
export DST_FOLDER_FN="dst_def"
export CVX_FOLDER_FN="src_bsp"
export src_cvx_to_pts_sufix="_cvx_to_verts.npy"
export dst_cvx_to_pts_sufix="_cvx_to_verts.npy"
export cvx_list_filter=""
export cvx_list_filter="--cvx_list_filter"
export CUDA_ID=7
export src_index=0




# # Eyeglasses -> link0/0 /data/datasets/genn/SAPIEN_Deform/Eyeglasses/link_2/dst
# export def_version="v4"
# export LAP_LAMBDA=4.0
# export coef_multiplier=0.2
# export dis_factor=2.5
# export NUM_BASIS=16
# export N_KETPOINTS=64
# export DATA_DIR="/data/datasets/genn/SAPIEN_Deform/Eyeglasses/link_2"
# export PRED_TYPE="basis"
# export use_gt_cages="--use_gt_cages"
# export cd_weight=0.0
# export n_shots=8
# export EXP_FLAG=eye_link2_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_shots_${n_shots}_fr_self_
# export SRC_FOLDER_FN="dst"
# export DST_FOLDER_FN="dst"
# export CVX_FOLDER_FN="dst"
# export cvx_to_pts_sufix="_cvx_to_verts.npy" ## src_cvx_to_pts...
# export src_cvx_to_pts_sufix="_cvx_to_verts.npy"
# export dst_cvx_to_pts_sufix="_cvx_to_cvx_pts_n_512.npy"
# export one_shp=""
# export with_dis_loss="--with_dis_loss"
# export pred_offst="--pred_offset"
# export small_tar_nn=""
# export lr=0.0001
# export gravity_weight=1.0
# export n_folds=3
# export by_part="--by_part"
# export optimize_coef="--optimize_coef"
# export with_image_loss=""
# export net_path=""
# export optimize_basis="--optimize_basis"
# export rnd_sample_nn=5
# export glb_template="/home/xueyi/gen/DeepMetaHandles/data/sphere_template_4.obj"
# #### train_def_cages #####
# export CUDA_ID=0
# export cvx_list_filter=""
# export src_data_dir="/data/datasets/genn/SAPIEN_Deform/Eyeglasses/link_2"
# export src_n_keypoints=64
# export dst_data_dir="/data/datasets/genn/SAPIEN_Deform/Eyeglasses/link_2" # 04379243 ### dst_data_dir...
# # export dst_data_dir="/data/datasets/genn/SAPIEN_Deform/Eyeglasses/link_1" # 04379243 ### dst_data_dir...
# # export dst_data_dir="/data/datasets/genn/SAPIEN_Deform/Eyeglasses/link_0" # 04379243 ### dst_data_dir...
# # export dst_data_dir="/data/datasets/genn/ShapeNetCoreV2_Deform/04379243_c_512"
# export dst_n_keypoints=1024

# # export DST_FOLDER_FN="dst_def"
# # export CVX_FOLDER_FN="src_bsp"
# export DST_FOLDER_FN="dst"
# export CVX_FOLDER_FN="dst"

# export src_cvx_to_pts_sufix="_cvx_to_verts.npy"
# # export dst_cvx_to_pts_sufix="_cvx_to_cvx_pts_n_512.npy"
# export dst_cvx_to_pts_sufix="_cvx_to_verts.npy"
# export cvx_list_filter=""
# # export cvx_list_filter="--cvx_list_filter"


# Eyeglasses -> /data/datasets/genn/SAPIEN_Deform/Eyeglasses/link_2/dst
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
export EXP_FLAG=eye_link0_gt_cages_${use_gt_cages}_cd_weight_${cd_weight}_shots_${n_shots}_fr_self_
export SRC_FOLDER_FN="dst"
export DST_FOLDER_FN="dst"
export CVX_FOLDER_FN="dst"
export cvx_to_pts_sufix="_cvx_to_verts.npy"
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
export glb_template="/home/xueyi/gen/DeepMetaHandles/data/sphere_template_4.obj" ## template v4... ##
# export glb_template="/home/xueyi/gen/DeepMetaHandles/data/sphere_V42_F80.off" # Oven link0, drawer link2
#### train_def_cages #####
export CUDA_ID=0
export cvx_list_filter=""


export src_data_dir="/data/datasets/genn/SAPIEN_Deform/Eyeglasses/link_2"
# export src_data_dir="/data/datasets/genn/SAPIEN_Deform/Eyeglasses/link_1"
# export src_data_dir="/data/datasets/genn/SAPIEN_Deform/Eyeglasses/link_0"
# export src_data_dir="/share/xueyi/datasets/gen/datasets/Shape2Motion_Deform/eyeglasses/none_motion"
# # export DATA_DIR="/data/datasets/genn/ShapeNetCoreV2_Deform/04379243_c_512"
# export src_data_dir="/data/datasets/genn/ShapeNetCoreV2_Deform/04379243_c_512"
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



export src_data_dir="/share/xueyi/datasets/gen/datasets/Shape2Motion_Deform/eyeglasses/none_motion"
export dst_data_dir="/share/xueyi/datasets/gen/datasets/Shape2Motion_Deform/eyeglasses/none_motion"
export DST_FOLDER_FN="dst"
export CVX_FOLDER_FN="dst"
export src_cvx_to_pts_sufix="_manifold_cvx_to_verts.npy"
export dst_cvx_to_pts_sufix="_manifold_cvx_to_verts.npy"
export cvx_list_filter="--cvx_list_filter"
export src_index=-1 ## 


export TRAINER=train_def_cages.py


export CVX_DIM=128


export RECON_COND="cvx"
export RECON_COND="bbox"


# export NET_PATH=""
export with_glb=""

export display=20

export CUDA_ID=0

export BATCH_SIZE=1
export TAR_BASIS=0


CUDA_VISIBLE_DEVICES=${CUDA_ID} python \
src/convex_deformation/${TRAINER} --batch_size=${BATCH_SIZE} --epoch=30000 --learning_rate=${lr}  --env_name='test' --num_basis=${NUM_BASIS} --display=${display} --step_size=100 --kl-weight=${KL_WEIGHT} --exp_flag=${EXP_FLAG}  --data_dir=${DATA_DIR} --n_keypoints=${N_KETPOINTS} --tar_basis=${TAR_BASIS} --load_meta  --pred_type=${PRED_TYPE}   --use_pointnet2 --use_pp_tar_out_feat --neighbouring_k=${N_NEIGHBORING} --dataset_name=v2 --n_samples=${N_SAMPLES} --symmetry_axis=${SYM_AXIS}  --net_path=${net_path} --local_handle --hier_stage=${HIER_STAGE} --sv_ckpt_freq=${SV_CKPT_FREQ}   --with_cat_deform=0   --use_paired_data --wo_keypts_abs --lap_lambda=${LAP_LAMBDA}  --sym_lambda=${SYM_LAMBDA}  --use_cvx_feats --use_def_pc --use_trans_range  --use_vae_opt=${USE_VAE_OPT} --pred_positions --src_folder_fn=${SRC_FOLDER_FN} --dst_folder_fn=${DST_FOLDER_FN} --use_recon_as_cd --recon_cond=${RECON_COND} --cond_tar_pc --cvx_dim=${CVX_DIM} --only_tar  --cvx_to_pts_sufix=${cvx_to_pts_sufix} --coef_multiplier=${coef_multiplier}   ${with_glb} ${one_shp} ${with_dis_loss} --def_version=${def_version} --dis_factor=${dis_factor} --pred_offset ${small_tar_nn} --n_fold=${n_folds}  --gravity_weight=${gravity_weight} ${by_part} ${use_gt_cages} ${optimize_coef} ${with_image_loss} ${optimize_basis} --rnd_sample_nn=${rnd_sample_nn} --cvx_folder_fn=${CVX_FOLDER_FN} --glb_template=${glb_template} --n_shots=${n_shots} ${cvx_list_filter} --src_data_dir=${src_data_dir} --src_n_keypoints=${src_n_keypoints} --dst_data_dir=${dst_data_dir} --dst_n_keypoints=${dst_n_keypoints} --src_cvx_to_pts_sufix=${src_cvx_to_pts_sufix} --dst_cvx_to_pts_sufix=${dst_cvx_to_pts_sufix} --src_index=${src_index}

