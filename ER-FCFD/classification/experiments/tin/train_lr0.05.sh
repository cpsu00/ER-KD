STUDENT="$1"
TEACHER="$2"
random="${3:-$RANDOM}"

exp_name="tin"
running_name="$STUDENT"-"$TEACHER"-"$random"
mkdir -p output/"$exp_name"/"$running_name"/log
python -u tin/main.py \
 --init_lr 0.05 --batch_size 64 \
 --output_dir=output/"$exp_name"/"$running_name" \
 --config=experiments/"$exp_name"/"$STUDENT"-"$TEACHER".yaml \
 --student="$STUDENT" --teacher="$TEACHER" \
 --teacher_ckpt=../../download_ckpts/tinyimagenet_teachers/"$TEACHER"_vanilla/ckpt_epoch_240.pth \
 --random_seed="$random" \
#  --is_train False --resume output/tin/resnet8x4-resnet32x4-1238/ckpt/model1_model_best.pth.tar