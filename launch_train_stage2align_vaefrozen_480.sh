#!/bin/bash
# Stage2 + align @ 480x832x81 — stage2 학습환경 정확 매칭 (VAE frozen, align→DiT만).
# Base: launch_train_stage2align_vaefrozen_zmainpatchzero.sh (256) 에서 480 매칭 위해 아래만 변경:
#   [stage2 정확 매칭 3-flag] (모두 opt-in, default off=기존 stage1)
#     --resolution_hw 480x832 + --num_frames 81 : non-square 481-frame
#     --dataset_stage2_preprocess               : stage2 ImageCropAndResize(cover-resize+center-crop) 매칭 (distort 아님)
#     --sample_rate 1                           : stage2 contiguous stride-1 (기존 기본 2=stride-2 와 다름)
#     --diffusion_zmain_use_mu                  : diffusion denoise 대상 z_main=mu (stage2 single_encode 동일, 샘플링 제거)
#   [메모리] --use_grad_checkpoint (전 40블록) + BS=1 GRAD_ACCUM=16 (effective 128 = stage2 bs16)
#   [정규화] --zmain_stats_path → 480x832x81 통계 (loraFreeze step5500 n200)
# 나머지(VAE freeze, align_weight 1.0, diffusion loss, LoRA512, zmain patchify zero-init fresh start)는 256 과 동일.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
NUM_GPUS=8
PY=/home/kaist_peta/miniconda3/envs/kinemadae-fa4/bin/python
BS=${BS:-1}
EXP_SUFFIX=${EXP_SUFFIX:-}
WAN_CKPT=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/checkpoints_persistent/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth
DIT_CKPT_DIR=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/checkpoints_persistent/Wan2.1-I2V-14B-480P
VIDEO_TRAIN=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/KinemaDAE-kk4aiq-to-lora/panda70m_train.txt
VIDEO_EVAL=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/KinemaDAE-kk4aiq-to-lora/panda70m_eval.txt
CAPTION_META=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/KinemaDAE-kk4aiq-to-lora/data/panda70m_metadata_captioned.jsonl
ZMAIN_STATS=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/KinemaDAE-dit-lora-bn-stage2/scripts/zmain_stats_loraFreeze_step5500_480x832x81f_n200.json
RESUME_CKPT=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/Kinemadae-adaptive-Bfix/results/kinemadae_stage1_bn_lora_align40_bs8_loraFreeze_matchS2_clamp1e5_b_proj_teacherfrozen-lr8.00e-05-bs8-rs256-sr2-fr17/checkpoint-5500.ckpt
export KINEMADAE_DIFFSYNTH_PATH=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/KinemaDAE-kk4aiq/external/DiffSynth-Studio
export KINEMADAE_PROBING_PATH=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/KinemaDAE-kk4aiq/external/dit_probing
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
$PY -m torch.distributed.run --nproc_per_node=$NUM_GPUS --standalone train_causalvae_geoprior_dit_align.py \
    --exp_name "kinemadae_stage2align_vaefrozen_zmainPatchZero_stage2match_480_bs${BS}${EXP_SUFFIX}" \
    --pretrained_model_name_or_path "$WAN_CKPT" \
    --add_encoder_stages '[{"mode":"downsample3d","num_res_blocks":2,"init":"zero"}]' \
    --add_decoder_before_head_stages '[{"mode":"upsample3d","num_res_blocks":2}]' \
    --z_dim 16 \
    --no_expand_conv2 \
    --subsample_mode bilinear \
    --video_path "$VIDEO_TRAIN" \
    --eval_video_path "$VIDEO_EVAL" \
    --num_frames 81 \
    --resolution 256 \
    --resolution_hw 480x832 \
    --dataset_stage2_preprocess \
    --sample_rate 1 \
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
    --eval_steps 999999 \
    --log_steps 1 \
    --mix_precision bf16 \
    --ema \
    --ema_decay 0.999 \
    --eval_lpips \
    --find_unused_parameters \
    --dit_ckpt_dir "$DIT_CKPT_DIR" \
    --align_weight 1.0 \
    --max_grad_norm 2.0 \
    --weight_decay 1e-2 \
    --align_loss_type cosine \
    --align_layers all \
    --align_agg sum \
    --align_adaptive_weight \
    --adaptive_max_weight 10000000 \
    --grad_accum_steps ${GRAD_ACCUM:-16} \
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
    --normalize_zmain \
    --zmain_stats_path "$ZMAIN_STATS" \
    --no_fused_align \
    --use_align_projection \
    --align_projection_init zero \
    --align_proj_bottleneck_dim 16 \
    --use_diffusion_loss \
    --diffusion_loss_weight 1.0 \
    --diffusion_max_timestep_boundary 1.0 \
    --diffusion_min_timestep_boundary 0.0 \
    --diffusion_zmain_use_mu \
    --diffusion_unfreeze_zprior_patchify \
    --diffusion_unfreeze_block_norms_mod \
    --use_grad_checkpoint \
    --skip_recon_when_frozen \
    --dit_fsdp2 \
    --teacher_frozen_pretrained \
    --freeze_encoder \
    --freeze_decoder \
    --resume_from_checkpoint "$RESUME_CKPT" \
    --no_resume_patchify \
    --resume_reset_step
