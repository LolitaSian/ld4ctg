export WANDB_MODE=offline
python train_text_diffusionp.py \
  --dataset_name e2e \
  --adam_weight_decay 0.01 \
  --learning_rate 1e-4 \
  --num_train_steps 100000 \
  --train_batch_size 256 \
  --eval_batch_size 256 \
  --tx_dim 768 \
  --tx_depth 12 \
  --objective pred_x0 \
  --enc_dec_model ./assist_model/bart-base/ \
  --num_samples 1000 \
  --normalize_latent \
  --scale_shift \
  --loss_type l1 \
  --beta_schedule linear \
  --sampling_timesteps 250 \
  --save_and_sample_every 10000 \
  --wandb_project denoising_diffusion \

