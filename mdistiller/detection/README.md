# MSCOCO object detection code

This code is based on [ReviewKD](https://github.com/dvlab-research/ReviewKD).

## Installation

1. Install Detectron2 following https://github.com/facebookresearch/detectron2.

2. Put the [MSCOCO](https://cocodataset.org/#download) dataset in datasets/.

3. Download the pretrained weights for both teacher and student models from the [ReviewKD](https://github.com/dvlab-research/ReviewKD/releases/) and place thm into `pretrained/`. The pretrained models they provided contains both teacher's and student's weights. The teacher's weights come from Detectron2's pretrained detector. The student's weights are ImageNet pretrained weights.


## Training

```
sh script_ER_KD.sh
```
