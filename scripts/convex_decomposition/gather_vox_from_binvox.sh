## Excute scripts in the root folder of the project (i.e., xxxx/few-arti-gen)
export PYTHONPATH=.

## path to cuda_voxelizer
export data_root_folder=""
## path to mesh data folder
export sv_data_root_folder=""


# /home/xueyi/gen/few-arti-gen/src/convex_decomposition/gather_vox_from_binvox.py
export script_to_excute="src/convex_decomposition/gather_vox_from_binvox.py"


python ${script_to_excute} --data_root_folder=${data_root_folder} --sv_data_root_folder=${sv_data_root_folder}
