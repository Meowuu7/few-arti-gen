import os
import argparse



def create_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-vox', '--voxelizer', type=str, default="", required=True, help='Path to cuda_voxelizer')
    parser.add_argument('-data', '--data_folder', type=str, default="", required=True, help='Path to mesh data folder')
    return parser


def get_binvox_data(args):
    data_root = args.data_folder
    obj_fns = os.listdir(data_root)
    obj_fns = [fn for fn in obj_fns if fn.endswith(".obj")]
    cuda_voxelizer_path = args.voxelizer
    vox_size = 64
    for obj_fn in obj_fns:
        cur_obj_fn = os.path.join(root, obj_fn)
        os.system(f"{cuda_voxelizer_path} -f {cur_obj_fn} -s {vox_size}")


if __name__=="__main__":
    
    parser = create_arguments()
    args = parser.parse_args()
    
    get_binvox_data(args)

