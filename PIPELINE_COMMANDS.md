# Combined Imitation → SCAMPER → Transfer Pipeline

## Step 1: Imitation (already queued)
```
!cd /home/talmolab/Desktop/SalkResearch/track-mjx && source /home/talmolab/Desktop/SalkResearch/mimic-mjx/bin/activate && python -m track_mjx.train --config-name=rodent-full-clips walker_config.reference_data_path=/home/talmolab/Desktop/SalkResearch/track-mjx/data/rodent/rodent_combined_reference_clips.h5 logging_config.exp_name=combined_gap_v1 logging_config.group_name=combined_imitation
```

When done, find the checkpoint:
```bash
ls -td /home/talmolab/Desktop/SalkResearch/track-mjx/model_checkpoints/2604* | head -1
```
Note the run_id (e.g., `260406_123456_789012`).

## Step 2: SCAMPER Prior Distillation
Replace `CKPT_PATH` with the imitation checkpoint path from Step 1.
Add this line to `job_queue.txt`:
```
!cd /home/talmolab/Desktop/SalkResearch/SCAMPER && source /home/talmolab/Desktop/SalkResearch/mimic-mjx/bin/activate && python -m scamper.train_prior --config-name=rodent-prior-warp teacher_config.checkpoint_path=CKPT_PATH env_config.reference_data_path=/home/talmolab/Desktop/SalkResearch/track-mjx/data/rodent/rodent_combined_reference_clips.h5 env_config.clip_length=250 env_config.domain_randomization.use_domain_randomization=true +env_config.reference_stride=1 logging_config.exp_name=combined_gap_prior logging_config.group_name=prior_distillation logging_config.model_path=model_checkpoints/
```

When done, find the SCAMPER checkpoint:
```bash
ls -td /home/talmolab/Desktop/SalkResearch/SCAMPER/model_checkpoints/2604* | head -1
```

## Step 3: Transfer with Variable Speed [0.5, 1.0]
Replace `PRIOR_CKPT_PATH` with the SCAMPER checkpoint path from Step 2.
Add this line to `job_queue.txt`:
```
rodent_run_gap/velocity_range_only transfer.prior_checkpoint_path=PRIOR_CKPT_PATH logging_config.exp_name=vel_range_0.5_1.0_combined_prior logging_config.group_name=combined_transfer
```
