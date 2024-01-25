## pytorch_points - a python library for learning point clouds on pytorch

This library implements and collects some useful functions commonly used for point cloud processing.
Many of the functions are adapted from the code base of amazing researchers. Thank you! (see [related repositories](#related_repositories) for a comprehensive list.)

### structures

- `_ext`: cuda extensions,
  - losses: "chamfer distance"
  - sampling: "farthest_sampling", "ball_query"
- `network`: common pytorch layers and operations for point cloud processing
  - operations: "group_KNN", "batch_normals"
  - layers
- `utils`: utility functions including functions for point cloud in/output etc
  - pc_utils

### install
```bash
# update conda
conda update -n base -c defaults conda
# requirements
conda create --name pytorch-all -f environment.yml  # to create a new conda environment
# or
conda config --add channels pytorch
conda config --add channels conda-forge
conda install --file requirements.txt

python setup.py install
```

### related repositories:
- [AtlasNet](https://github.com/ThibaultGROUEIX/AtlasNet): Thanks Thibault for your AtlasNet!!!
- [Pointnet-Pytorch](https://github.com/erikwijmans/Pointnet2_PyTorch): pytorch implementation of pointnet.
- [SO-Net](https://github.com/lijx10/SO-Net): CVPR 2018 spotlight paper
