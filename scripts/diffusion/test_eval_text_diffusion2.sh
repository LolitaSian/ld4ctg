export WANDB_MODE=offline
python train_text_diffusion.py \
  --eval_test \
  --resume_dir saved_models/e2e2 \
  --sampling_timesteps 250 \
  --num_samples 1000 \
  --ddim_sampling_eta 1

