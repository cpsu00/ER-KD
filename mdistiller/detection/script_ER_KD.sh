# # Tea: R-101, Stu: R-18
python3 train_net.py --config-file configs/ERKD/ERKD-R18-R101.yaml --num-gpus 1

# # Tea: R-101, Stu: R-50
python3 train_net.py --config-file configs/ERKD/ERKD-R50-R101.yaml --num-gpus 1

# Tea: R-50, Stu: MV2
python3 train_net.py --config-file configs/ERKD/ERKD-MV2-R50.yaml --num-gpus 1