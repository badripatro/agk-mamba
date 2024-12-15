#!/bin/bash

echo "Current path is $PATH"
echo "Running"
nvidia-smi
echo $CUDA_VISIBLE_DEVICES

python3 -m torch.distributed.launch \
   --nproc_per_node=8 \
   --nnodes=1 \
   --node_rank=0 \
   --master_addr="localhost" \
   --master_port=12346 \
   --use_env main.py --config configs/agk/agk_l.py --data-path ../../../../dataset/Image_net/imagenet --epochs 310 --batch-size 64 --drop-path 0.05 --weight-decay 0.05 --lr 1e-3 --num_workers 24\
   --token-label --token-label-size 7 --token-label-data ../../../../dataset/Image_net/imagenet_efficientnet_l2_sz475_top5/