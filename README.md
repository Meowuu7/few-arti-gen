<p align="center">

  <h1 align="center">Few-Shot Physically-Aware Articulated Mesh Generation via Hierarchical Deformation</h1>
  <!-- <h2 align="center">ICCV 2023</h2> -->
  <p align="center">
    <!-- <br> -->
    <!-- <br> -->
      <strong>ICCV 2023</strong>
      <br>
      <a href="https://meowuu7.github.io/few-arti-obj-gen/">Project page</a>
      |
      <a href="https://meowuu7.github.io/few-arti-obj-gen/static/pdfs/few-arti-gen.pdf">Paper</a>
      |
      <a href="https://meowuu7.github.io/few-arti-obj-gen/static/pdfs/few-arti-gen-supp.pdf">Supp</a>
      |
      <a href="https://youtu.be/p8x3GN3VSPE">Video</a>
  </p>

  <div align="center">
    <!-- <img src="./assets/teaser-2-cropped.gif" alt="Logo" width="100%"> -->
    <!-- <video id="teaser" autoplay muted loop width="100%">
      <source src="./assets/res-demo.mp4"
              type="video/mp4">
    </video> -->
    <!-- <video src="./assets/res-demo.mp4" width="100%"></video> -->
    
  </div>
  <!-- <br>
  <div align="center">
  </div>
  <strong>ICCV 2023</strong> -->
</p>
<!-- <video src="./assets/res-demo.mp4" width="100%"></video>
![video](./assets/res-demo.mp4) -->


https://github.com/Meowuu7/few-arti-gen/assets/50799886/96ec73a6-af99-4d0c-9365-c48d184fc33c

## Enviroment

```bash
conda env create -f environment.yml
```
This script will create an environment named `fewartigen`. 

## Data
Please refer to [`doc_data`](./docs/data.md) for datasets and pre-processed data-related information. 


## Instructions
The method consists of four stages:
- **Convex decomposition**: a pre-processing stage. Please refer to [`doc_convex_decomposition`](./docs/1-convex_decomposition.md) for details. 
- **Convex deformation**: learning the convex deformation module. Please refer to [`doc_convex_deformation`](./docs/2-convex_deformation.md) for details. 
- **Deformation synchronization**: synchronizing convex deformations for object-level deformations. Please refer to [`doc_deformation_synchronization`](./docs/3-deformation_synchronization.md) for details. 
- **Physics-aware correction**: deformation correction considering the physical validity of the generated articulated object. Please refer to [`doc_physics_correction`](./docs/4-physics_correction.md) for details. 

**TODOs (More to come, stay tuned!)**
- [ ] Data and checkpoitns
- [ ] More docs

## Citation

```bibtex
@inproceedings{liu2023fewshot,
      title={Few-Shot Physically-Aware Articulated Mesh Generation via Hierarchical Deformation},
      author={Liu, Xueyi and Wang, Bin and Wang, He and Yi, Li},
      booktitle={International Conference on Computer Vision (ICCV)},
      year={2023}
}
```

## Contact

Please contact xymeow7@gmail.com if you have any questions.

## Reference

Part of the code is taken from [BSP-Net-Pytorch](https://github.com/czq142857/BSP-NET-pytorch), [BAE-Net](https://github.com/czq142857/BAE-NET), [deep_cages](https://github.com/yifita/deep_cage), and [DeepMetaHandles](https://github.com/Colin97/DeepMetaHandles).  We thank the authors for their awesome code. 