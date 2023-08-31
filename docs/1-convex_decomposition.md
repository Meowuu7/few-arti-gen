# Convex Decomposition


We use [BSP-Net](https://github.com/czq142857/BSP-NET-pytorch) to conduct category-level part/object convex cosegmentation. 

The process of partitioning a set of part/object meshes from a same category into convexes consists of the following steps, 
- [Mesh voxelization](#mesh-voxelization) (`.obj` $\rightarrow$ `.binvox`)
- [Gathering voxels](#gathering-voxels) (gather a set of `.binvox` into a `.vox` file)
- [Convex partition](#convex-partition)

## Mesh Voxelization

We use [cuda_voxelizer](https://github.com/Forceflow/cuda_voxelizer) for efficiently converting meshes into voxels (with grid size as 64 in our experimenets). 

### Setup

Please follow the instructions for [building](https://github.com/Forceflow/cuda_voxelizer) to setup the tool. 

### Usage

Put all meshes to voxelize in the same folder, and run the script
```bash
CUDA_VISIBLE_DEVICES=${cuda_ids} python src/convex_decomposition/mesh_voxelization.py --data_folder=${data_folder} --voxelizer=${voxelizer}
```
where `data_folder` is the path to meshes to process and `voxelizer` is the path to the compiled binary excutable file `cuda_voxelizer`. 


## Gathering Voxels

Run the following script to gather all voxels to a single file
```bash
python src/convex_decomposition/gather_vox_from_binvox.py --data_root_folder=${data_root_folder} --sv_data_root_folder=${sv_data_root_folder}
```
where `data_root_folder` is the path to the folder of all voxels to gather and `sv_data_root_folder` is the destination folder. 



## Convex Partition

We use BSP-Net for co-segmentation. Navigate to its root directory,
```bash
cd BSP-Net-Pytorch
```
set up the module,
```bash
python setup.py build_ext --inplace
```
and run the following script to segment a set of meshes into convexes
```bash
CUDA_VISIBLE_DEVICES=${CUDA_ID} python \
main.py --ae --train --phase 0 --iteration 200000 --sample_dir=${SAMPLE_DIR} --data_dir=${DATA_DIR} --dataset=${DATASET} --sample_vox_size 16 --batch_size=${BATCH_SIZE_16} --c_dim=${C_DIM} --model_flag=${MODEL_FLAG} --checkpoint_dir=${CHECKPOINT_DIR}

CUDA_VISIBLE_DEVICES=${CUDA_ID} python \
main.py --ae --train --phase 0 --iteration 200000 --sample_dir=${SAMPLE_DIR}  --data_dir=${DATA_DIR} --dataset=${DATASET} --sample_vox_size 32 --batch_size=${BATCH_SIZE} --c_dim=${C_DIM}  --model_flag=${MODEL_FLAG} --checkpoint_dir=${CHECKPOINT_DIR}

CUDA_VISIBLE_DEVICES=${CUDA_ID} python \
main.py --ae --train --phase 0 --iteration 400000 --sample_dir=${SAMPLE_DIR}  --data_dir=${DATA_DIR} --dataset=${DATASET} --sample_vox_size 64 --batch_size=${BATCH_SIZE} --c_dim=${C_DIM}  --model_flag=${MODEL_FLAG} --checkpoint_dir=${CHECKPOINT_DIR}

CUDA_VISIBLE_DEVICES=${CUDA_ID} python \
main.py --ae --train --phase 1 --iteration 400000 --sample_dir=${SAMPLE_DIR}  --data_dir=${DATA_DIR} --dataset=${DATASET} --sample_vox_size 64 --batch_size=${BATCH_SIZE} --c_dim=${C_DIM}  --model_flag=${MODEL_FLAG} --checkpoint_dir=${CHECKPOINT_DIR}
```
where `DATA_DIR` should be the directory where the gathered voxels are saved, while `SAMPLE_DIR` and `CHECKPOINT_DIR` are directories to store samples on the fly and the checkpoints. 














