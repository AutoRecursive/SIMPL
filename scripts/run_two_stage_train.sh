#!/bin/bash
source /home/u24/u24_Programs/AD/SIMPL/set_cudnn_path.sh

CUDA_VISIBLE_DEVICES=0 python train_two_stage.py \
  --features_dir data_av2/features/ \
  --pretrain_cfg_path config.simpl_av2_cfg \
  --pretrain_epochs 20 \
  --pretrain_batch_size 4 \
  --pretrain_val_batch_size 4 \
  --pretrain_val_interval 2 \
  --rl_cfg_path config.simpl_rl_cfg \
  --rl_epochs 10 \
  --rl_batch_size 4 \
  --rl_val_batch_size 4 \
  --rl_val_interval 1 \
  --data_aug \
  --use_cuda \
  --logger_writer
