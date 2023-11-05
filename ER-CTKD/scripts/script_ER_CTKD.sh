############ CIFAR100 ############
# ER-CTKD
python3 train_student.py --path-t ../download_ckpts/cifar_teachers/resnet32x4_vanilla/ckpt_epoch_240.pth \
        --distill kd \
        --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --kd_T 4 \
        --batch_size 64 --learning_rate 0.05 \
        --have_mlp 1 --mlp_name 'global' \
        --cosine_decay 1 --decay_max 0 --decay_min -1 --decay_loops 10 \
        --save_model \
        --experiments_dir 'tea-res56-stu-res20/kd/global_T/your_experiment_name' \
        --experiments_name 'fold-1'

############ TinyImageNet ############
# ER-CTKD
python3 train_student.py --path-t ../mdistiller/download_ckpts/tinyimagenet_teachers/resnet32x4_vanilla/ckpt_epoch_240.pth \
        --distill kd \
        --dataset tinyimagenet200 \
        --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --kd_T 4 \
        --batch_size 64 --epochs 240 --learning_rate 0.05 \
        --have_mlp 1 --mlp_name 'global' \
        --cosine_decay 1 --decay_max 0 --decay_min -1 --decay_loops 10 \
        --save_model \
        --experiments_dir 'tinyimagenet200-tea-res32x4-stu-res8x4/ctkd_120/ctkd' \
        --experiments_name 'fold-1'