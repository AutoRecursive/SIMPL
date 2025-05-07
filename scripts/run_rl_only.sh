#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_two_stage.py \
  --features_dir data_av2/features/ \
  --skip_pretrain \
  --pretrain_model_path saved_models/20250504-235305_pretrain_Simpl_ckpt_epoch16.tar \
  --rl_cfg_path config.simpl_rl_cfg \
  --rl_epochs 10 \
  --rl_batch_size 4 \
  --rl_val_batch_size 4 \
  --rl_val_interval 1 \
  --data_aug \
  --use_cuda \
  --logger_writer
