# Physics-Aware Deformation Correction


The physics-aware deformation correction stage takes deformation networks optimized for each part shape and reference part shapes with convex segments as input and synthesizes object shapes. It leverages a deformation correction guided by $\mathcal{L}_{proj}$ stated in the paper. 


## Usage
Change configs in `scripts/physicsaware_correction/train_def_cages_obj.sh` by specifying trained network paths for each part and data-related configurations. And run the script
```bash
bash scripts/physicsaware_correction/train_def_cages_obj.sh
```

**TODOs**
- [ ] Configs for each category
- [ ] Pre-trained weights
