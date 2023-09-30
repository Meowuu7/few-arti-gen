# Convex Deformation


The convex deformation stage learns the deformation module via convex pairs in correspondence. 

The network aims at learning to predict the deformation bases from an input convex and is learned in a target-driven deformation fashion. 


## Usage

### Important arguments
- `src_data_dir`: Root folder of the deformation source part
- `dst_data_dir`: Root folder of the deformation target part
- `SRC_FOLDER_FN`: Secondary folder of the deformation source part
- `DST_FOLDER_FN`: Secondary folder of the deformation target part
- `src_cvx_to_pts_sufix`: Sufix of the convex-to-points dict file


### Usage
Currently, we need to specify important arguments in the script file to train on the corresponding category. 


For instance, run the script 
```bash
bash scripts/convex_deformation/train_def_cages_eyeglasses_1.sh
```
to train on the `none_motion` part of the eyeglasses category from the `Shape2Motion` dataset. 



**TODOs**
- [ ] Configs for each convex categroy
- [ ] Pre-trained weights

