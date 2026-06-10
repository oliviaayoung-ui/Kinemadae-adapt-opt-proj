#!/bin/bash
# Fresh start: B-adaptive + align_projection L2 + diffusion loss + align_stop_grad + TEACHER FROZEN PRETRAINED
# Base: launch_train_diffusion_baselineLoRAinit_stopgrad.sh
# ADDED:
#   --teacher_frozen_pretrained : teacher forward(align target)를 init pretrained Wan I2V 14B 로 고정.
#     teacher/student 가 dit 객체 공유 → student diffusion 이 LoRA/block_norms 학습하면 teacher align target 도 drift.
#     이 플래그 on 이면 teacher forward 순간에만 (a) LoRA off + (b) block_norms/mod → pretrained snapshot swap → 직후 원복.
#     별도 dit copy 불필요 (메모리 ≈ block_norms snapshot ~5MB). student/diffusion forward 는 학습 weight 그대로.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
NUM_GPUS=8
PY=/home/kaist_peta/miniconda3/envs/kinemadae-fa4/bin/python
BS=${BS:-2}
EXP_SUFFIX=${EXP_SUFFIX:-}
WAN_CKPT=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/checkpoints_persistent/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth
DIT_CKPT_DIR=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/checkpoints_persistent/Wan2.1-I2V-14B-480P
VIDEO_TRAIN=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/KinemaDAE-kk4aiq-to-lora/panda70m_train.txt
VIDEO_EVAL=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/KinemaDAE-kk4aiq-to-lora/panda70m_eval.txt
CAPTION_META=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/KinemaDAE-kk4aiq-to-lora/data/panda70m_metadata_captioned.jsonl
export KINEMADAE_DIFFSYNTH_PATH=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/KinemaDAE-kk4aiq/external/DiffSynth-Studio
export KINEMADAE_PROBING_PATH=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/KinemaDAE-kk4aiq/external/dit_probing
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
$PY -m torch.distributed.run --nproc_per_node=$NUM_GPUS --standalone train_causalvae_geoprior_dit_align.py \
    --exp_name "kinemadae_lora_bn_clamp1e7_align40_bs${BS}${EXP_SUFFIX}_b_proj_diffusion_baselineLoRAinit_stopgrad_teacherfrozen" \
    --pretrained_model_name_or_path "$WAN_CKPT" \
    --add_encoder_stages '[{"mode":"downsample3d","num_res_blocks":2,"init":"zero"}]' \
    --add_decoder_before_head_stages '[{"mode":"upsample3d","num_res_blocks":2}]' \
    --z_dim 16 \
    --no_expand_conv2 \
    --unfreeze_decoder \
    --subsample_mode bilinear \
    --video_path "$VIDEO_TRAIN" \
    --eval_video_path "$VIDEO_EVAL" \
    --num_frames 17 \
    --resolution 256 \
    --batch_size $BS \
    --lr 8e-5 \
    --patchify_lr 1e-4 \
    --epochs 50 \
    --kl_weight 3e-6 \
    --perceptual_weight 3.0 \
    --disc_weight 0.5 \
    --disc_start 9999999 \
    --gan_last_layer decoder_head \
    --save_ckpt_step 500 \
    --eval_steps 500 \
    --log_steps 1 \
    --mix_precision bf16 \
    --ema \
    --ema_decay 0.999 \
    --eval_lpips \
    --find_unused_parameters \
    --dit_ckpt_dir "$DIT_CKPT_DIR" \
    --align_weight 1.0 \
    --align_loss_type cosine \
    --align_layers all \
    --align_agg sum \
    --align_adaptive_weight \
    --adaptive_max_weight 10000000 \
    --grad_accum_steps ${GRAD_ACCUM:-2} \
    --patchify_init zero \
    --patchify_mask_init copy4_zero4 \
    --normalize_zprior \
    --freeze_patchify_zprior \
    --caption_metadata "$CAPTION_META" \
    --align_num_blocks 40 \
    --use_b_adaptive \
    --log_adaptive_weight \
    --seed 1234 \
    --dataset_num_worker 5 \
    --text_fsdp2 \
    --use_lora \
    --lora_rank 512 \
    --lora_target_modules q,k,v,o,k_img,v_img,ffn.0,ffn.2 \
    --normalize_zmain_bn \
    --bn_momentum 0.1 \
    --zmain_bn_init zprior \
    --no_fused_align \
    --use_align_projection \
    --align_projection_init zero \
    --align_proj_bottleneck_dim 16 \
    --use_diffusion_loss \
    --diffusion_loss_weight 1.0 \
    --diffusion_max_timestep_boundary 1.0 \
    --diffusion_min_timestep_boundary 0.0 \
    --diffusion_unfreeze_zprior_patchify \
    --diffusion_unfreeze_block_norms_mod \
    --init_lora_safetensors /NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/KinemaDAE-dit-lora-bn-stage2/results/dit/baseline_pure_wan_i2v_256/step-500.safetensors \
    --align_stop_grad_dit \
    --teacher_frozen_pretrained
