#!/bin/bash
# Sweep additive proprioception noise: 0.1, 0.2, 0.3, 0.4, 0.5
# 750M timesteps each

set -euo pipefail

cd /home/talmolab/Desktop/SalkResearch/track-mjx
source /home/talmolab/Desktop/SalkResearch/mimic-mjx/bin/activate

for noise in 0.1 0.2 0.3 0.4 0.5; do
  echo "=== Starting run: additive noise std=${noise} ==="
  python -m track_mjx.train \
    network_config.proprioception_noise_std="${noise}" \
    network_config.proprioception_noise_mode=additive \
    train_setup.train_config.num_timesteps=750000000 \
    logging_config.exp_name="noisy_add_${noise}" \
    logging_config.group_name=additive_noise_sweep
done
