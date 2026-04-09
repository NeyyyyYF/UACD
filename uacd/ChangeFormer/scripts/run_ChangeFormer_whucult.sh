#!/usr/bin/env bash

#GPUs
gpus="0"
#Set paths
checkpoint_root=checkpoints
vis_root=vis
data_name=WHUCUL
img_size=256
batch_size=16
lr=0.0001
max_epochs=250
embed_dim=256
net_G=ChangeFormerV6

# net_G=ChangeFormerV6
lr_policy=linear
optimizer=adamw
loss=ce+uloss
# loss=uloss
# loss=u
# loss=ce0
multi_scale_train=False
multi_scale_infer=False
shuffle_AB=False
# pretrain=mit_b1.pth
# pretrain=checkpoints/CD_ChangeFormerV6_my_WHUCD_b16_lr0.0001_adamw_train_test_250_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256/best_ckpt.pt

pretrain=checkpoints/CD_ChangeFormerV6_S2Looking_b16_lr0.0001_adamw_train_test_250_linear_ce+uloss_multi_train_False_multi_infer_False_shuffle_AB_False_embed_dim_256/last_ckpt.pt
# pretrain=checkpoints/CD_ChangeFormerV6_my_WHUCUL_b16_lr0.0001_adamw_train_test_250_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256/best_ckpt.pt
split=train
split_val=test
project_name=CD_${net_G}_${data_name}_b${batch_size}_lr${lr}_${optimizer}_${split}_${split_val}_${max_epochs}_${lr_policy}_${loss}_multi_train_${multi_scale_train}_multi_infer_${multi_scale_infer}_shuffle_AB_${shuffle_AB}_embed_dim_${embed_dim}
CUDA_VISIBLE_DEVICES=0 python main_cd.py \
  --img_size ${img_size} --loss ${loss} --checkpoint_root ${checkpoint_root} \
  --vis_root ${vis_root} --lr_policy ${lr_policy} --optimizer ${optimizer} \
  --pretrain ${pretrain} --split ${split} --split_val ${split_val} \
  --net_G ${net_G} --multi_scale_train ${multi_scale_train} \
  --multi_scale_infer ${multi_scale_infer} --gpu_ids ${gpus} \
  --max_epochs ${max_epochs} --project_name ${project_name} \
  --batch_size ${batch_size} --shuffle_AB ${shuffle_AB} \
  --data_name ${data_name} --lr ${lr} --embed_dim ${embed_dim}

# scripts/run_rebuttal_whucult2.sh
