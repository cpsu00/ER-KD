
# Classification and detection code for ER-KD, ER-DKD, and ER-MLD

This code is based on [DKD(mdistiller)](<https://github.com/megvii-research/mdistiller>) and [MLD](<https://github.com/Jin-Ying/Multi-Level-Logit-Distillation.git>).

## Installation

To install the package, run:

```
python3 setup.py develop
```

## Training on CIFAR100 / TinyImageNet

```
sh script_ER.sh
```

## Training on MS-COCO

- See [detection.md](https://github.com/cpsu00/Entropy-Reweighted-Knowledge-Distillation/tree/main/mdistiller/detection)
