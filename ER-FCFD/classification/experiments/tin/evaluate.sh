STUDENT="$1"
ckpt="$2"

python -u tin/main.py \
 --is_train=false \
 --student="$STUDENT" \
 --resume="$ckpt"