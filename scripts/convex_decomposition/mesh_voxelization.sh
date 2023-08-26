## Excute scripts in the root folder of the project (i.e., xxxx/few-arti-gen)
export PYTHONPATH=.

## path to cuda_voxelizer
export voxelizer=""
## path to mesh data folder
export data_folder=""

export cuda_ids="1,2,5,7"

export script_to_excute="src/convex_decomposition/mesh_voxelization.py"


CUDA_VISIBLE_DEVICES=${cuda_ids} python ${script_to_excute} --data_folder=${data_folder} --voxelizer=${voxelizer}
