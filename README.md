<p align="center">

  <h1 align="center">Few-Shot Physically-Aware Articulated Mesh Generation via Hierarchical Deformation</h1>
  <!-- <h2 align="center">ICCV 2023</h2> -->
  <p align="center">
    <!-- <br> -->
    <!-- <br> -->
      <strong>ICCV 2023</strong>
      <br>
      <a href="https://meowuu7.github.io/few-arti-obj-gen/">Webpage</a>
      |
      <a href="https://meowuu7.github.io/few-arti-obj-gen/static/pdfs/few-arti-gen.pdf">Paper</a>
      |
      <a href="https://meowuu7.github.io/few-arti-obj-gen/static/pdfs/few-arti-gen-supp.pdf">Supp</a>
      |
      <a href="https://youtu.be/p8x3GN3VSPE">Video</a>
  </p>

  <div align="center">
    <img src="./assets/teaser-2-cropped.gif" alt="Logo" width="100%">
    <!-- <video id="teaser" autoplay muted loop height="100%">
      <source src="./assets/teaser-2-cropped.mp4"
              type="video/mp4">
    </video> -->
  </div>
  <!-- <br>
  <div align="center">
  </div>
  <strong>ICCV 2023</strong> -->
</p>

## Instructions

The method consists of four stages:
- **Convex decomposition**: a pre-processing stage. Please refer to `./docs/1-convex_decomposition.md` for details. 
- **Convex deformation**: learning the convex deformation module. Please refer to `./docs/2-convex_deformation.md` for details. 
- **Deformation synchronization**: synchronizing convex deformations for object-level dformations. Please refer to `./docs/3-deformation_synchronization.md` for details. 
- **Physics-aware correction**: deformation correction considering the physical validity of the generated articulated object. Please refer to `./docs/4-physics_correction.md` for details. 

**TODOs (More to come, stay tuned!)**
- [ ] Remaining code for synchronization
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

Please contact xymeow7@gmail.com if you have any question.
