#!/bin/bash
# Resume additive noise sweep from 0.3 checkpoint, then continue 0.4, 0.5
# 750M timesteps each

set -euo pipefail

cd /home/talmolab/Desktop/SalkResearch/track-mjx
source /home/talmolab/Desktop/SalkResearch/mimic-mjx/bin/activate

CHECKPOINT="/home/talmolab/Desktop/SalkResearch/track-mjx/model_checkpoints/260325_081445_999447"

# Resume 0.3 from existing checkpoint
echo "=== Resuming run: additive noise std=0.3 ==="
python -m track_mjx.train \
  train_setup.checkpoint_to_restore="${CHECKPOINT}" \
  train_setup.train_config.num_timesteps=750000000

# Continue sweep with fresh runs
for noise in 0.4 0.5; do
  echo "=== Starting run: additive noise std=${noise} ==="
  python -m track_mjx.train \
    network_config.proprioception_noise_std="${noise}" \
    network_config.proprioception_noise_mode=additive \
    train_setup.train_config.num_timesteps=750000000 \
    logging_config.exp_name="noisy_add_${noise}" \
    logging_config.group_name=additive_noise_sweep
done
