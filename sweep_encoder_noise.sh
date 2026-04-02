#!/bin/bash
# Sweep encoder noise levels: 0.1, 0.2, 0.3
# Proprioception noise fixed at 0.0 (isolate encoder noise effect)
set -e

source /home/talmolab/Desktop/SalkResearch/mimic-mjx/bin/activate
cd /home/talmolab/Desktop/SalkResearch/track-mjx

for NOISE in 0.1 0.2 0.3; do
    echo "================================================"
    echo "Starting encoder noise sweep: encoder_noise_std=${NOISE}"
    echo "================================================"

    python scripts/train.py \
        --config-name=rodent-full-clips \
        network_config.encoder_noise_std=${NOISE} \
        network_config.proprioception_noise_std=0.0 \
        train_setup.train_config.num_timesteps=750000000 \
        logging_config.exp_name="noisy_enc_${NOISE}" \
        logging_config.group_name=encoder_noise_sweep \
        logging_config.model_path=model_checkpoints/

    echo "Completed encoder noise ${NOISE}"
done

echo "All encoder noise sweep runs complete!"
