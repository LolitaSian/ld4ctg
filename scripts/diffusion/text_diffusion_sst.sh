export WANDB_MODE=offline
python train_text_diffusion.py \
  --learning_rate 0.0001 \
  --adam_weight_decay 0.01 \
  --dataset_name sst \
  --num_train_steps 150000 \
  --train_batch_size 256 \
  --eval_batch_size 256 \
  --tx_dim 768 \
  --tx_depth 12 \
  --objective pred_x0 \
  --enc_dec_model ./assist_model/bart-base \
  --num_samples 1000 \
  --normalize_latent \
  --scale_shift \
  --beta_schedule linear \
  --loss_type l2 \
  --class_conditional \
  --save_and_sample_every 10000 \
  --wandb_project denoising_diffusion \
  --wandb_name sst-bart-l2 \
  --resume_training \
  --resume_dir ./saved_models/sst

# 继续训练使用   --resume_training  --resume_dir <文件夹>