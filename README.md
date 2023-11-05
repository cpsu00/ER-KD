# Entropy-Reweighted Knowledge Distillation (ER-KD)

This repo is the official PyTorch implementation for "Entropy-Reweighted Knowledge Distillation".

## Framework
![ERKD](https://github.com/cpsu00/Entropy-Reweighted-Knowledge-Distillation/assets/85643374/535368d5-df66-4a56-9159-cdfc258f4eb8)

## Environments:

- Python 3.8
- PyTorch 2.2.0

## Usage

1. Place CIFAR100 and TinyImageNet datasets in `data/`.
2. Download pretrained teacher models for [CIFAR100](https://github.com/megvii-research/mdistiller/releases/download/checkpoints/cifar_teachers.tar) and [TinyImageNet]() and place them as follows:

```
download_ckpts/
├── cifar_teachers/             # Checkpoints for CIFAR100 teacher models
└── tinyimagenet_teachers/      # Checkpoints for TinyImageNet teacher models
```

3. Refer to the directories below for additional instructions to replicate our experiments. The original paper and code are also listed.

|ER Method|Directory|Original paper|Original paper code|
|:---:|:---:|:---:|:---:|
|ER-KD| [mdistiller](https://github.com/cpsu00/Entropy-Reweighted-Knowledge-Distillation/tree/main/mdistiller) |[Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)|https://github.com/megvii-research/mdistiller|
|ER-ReviewKD| [ER-ReviewKD](https://github.com/cpsu00/Entropy-Reweighted-Knowledge-Distillation/tree/main/ER-ReviewKD)  |[Distilling Knowledge via Knowledge Review](https://arxiv.org/abs/2104.09044)|https://github.com/dvlab-research/ReviewKD|
|ER-DKD| [mdistiller](https://github.com/cpsu00/Entropy-Reweighted-Knowledge-Distillation/tree/main/mdistiller) |[Decoupled Knowledge Distillation](https://arxiv.org/abs/2203.08679)|https://github.com/megvii-research/mdistiller|
|ER-MLD| [mdistiller](https://github.com/cpsu00/Entropy-Reweighted-Knowledge-Distillation/tree/main/mdistiller) |[Multi-level Logit Distillation](https://openaccess.thecvf.com/content/CVPR2023/papers/Jin_Multi-Level_Logit_Distillation_CVPR_2023_paper.pdf)|https://github.com/Jin-Ying/Multi-Level-Logit-Distillation|
|ER-FCFD| [ER-FCFD](https://github.com/cpsu00/Entropy-Reweighted-Knowledge-Distillation/tree/main/ER-FCFD) |[Function-Consistent Feature Distillation](https://arxiv.org/abs/2304.11832)|https://github.com/LiuDongyang6/FCFD|
|ER-CTKD| [ER-CTKD](https://github.com/cpsu00/Entropy-Reweighted-Knowledge-Distillation/tree/main/ER-CTKD)  |[Curriculum Temperature for Knowledge Distillation](https://arxiv.org/abs/2211.16231)|https://github.com/zhengli97/CTKD|


## Citation
If you find our method or this code useful, please consider citing our paper:

```bibtex
...
```

## Contact

For any inquiries or further discussions, feel free to reach out to me at cpsu00@outlook.com.


## Acknowledgement
We extend our sincere thanks to the contributors of [DKD(mdistiller)](<https://github.com/megvii-research/mdistiller>), [MLD](<https://github.com/Jin-Ying/Multi-Level-Logit-Distillation.git>), [ReviewKD](<https://github.com/dvlab-research/ReviewKD>), [FCFD](<https://github.com/LiuDongyang6/FCFD>) and [CTKD](<https://github.com/zhengli97/CTKD>) for their invaluable work, which has laid the foundation for our code.
