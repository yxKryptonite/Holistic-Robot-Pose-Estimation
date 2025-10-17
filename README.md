# Robot Pose Estimation

See original README [here](./README_original.md)

My installation notes:
- My environment: Ubuntu 20.04, cuda 11.8
- See `install.sh` for details

My tree structure: [tree.txt](./tree.txt)

Inference command:
```bash
python scripts/test.py -e experiments/panda_realsense --dataset panda-3cam_realsense --vis_skeleton
```