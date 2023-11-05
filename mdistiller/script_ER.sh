############ CIFAR100 ############
# ER-KD
python3 tools/train.py --cfg configs/cifar100/kd/res32x4_res8x4.yaml --er 1

# ER-DKD
python3 tools/train.py --cfg configs/cifar100/dkd/res32x4_res8x4.yaml --er 1

# ER-MLD
python3 tools/train_mld.py --cfg configs/cifar100/mld/res32x4_res8x4.yaml --er 1

############ TinyImageNet ############
# ER-KD
python3 tools/train.py --cfg configs/tinyimagenet200/kd/res32x4_res8x4.yaml --er 1
# ER-KD Transformer Teacher
python3 tools/train.py --cfg configs/tinyimagenet200/kd/vit_ResNet18.yaml --er 1

# ER-DKD
python3 tools/train.py --cfg configs/tinyimagenet200/dkd/res32x4_res8x4.yaml --er 1

# ER-MLD
python3 tools/train_mld.py --cfg configs/tinyimagenet200/mld/res32x4_res8x4.yaml --er 1