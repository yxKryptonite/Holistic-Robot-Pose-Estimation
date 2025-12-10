# Holistic Robot Pose Estimation

Different features are implemented in different branches:

- `main`: Modified holistic robot pose estimation pipeline, shared across all other features.
- `depth`: Using depth information to improve generalizability to unseen scenes.
- `replicator-510`: Using Isaac Sim Replicator for large-scale synthetic data generation.
- `vit_backbone_experiment`: Using ViT backbone for holistic robot pose estimation.

- - -

**Development note:**

See original README [here](./README_original.md)

My installation notes:
- My environment: Ubuntu 20.04, cuda 11.8
- See `install.sh` for details

My tree structure: [tree.txt](./tree.txt)

Inference command:
```bash
python scripts/test.py -e experiments/panda_realsense --dataset panda-3cam_realsense --vis_skeleton
```