# Convex Decomposition


We use [BSP-Net](https://github.com/czq142857/BSP-NET-pytorch) to conduct category-level part/object convex cosegmentation. 

The process of partitioning a set of part/object meshes from a same category into convexes consists of the following steps, 
- [Mesh voxelization](#mesh-voxelization) (`.obj` $\rightarrow$ `.binvox`)
- [Gathering voxels](#gathering-voxels) (gather a set of `.binvox` into a `.vox` file)
- Convex partition

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







