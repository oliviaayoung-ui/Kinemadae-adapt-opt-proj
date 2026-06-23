#!/bin/bash
# [decoder-only] aligned latent(z_main) 고정, decoder만 학습해서 recon 개선.
#   train_causalvae_geoprior_decoder_only.py = align/DiT 전부 제거된 VAE-only 스크립트.
#   스크립트가 encoder+conv1+conv2 를 무조건 freeze (z_main 고정) → decoder(Decoder3d)만 학습.
#   base = 1x52jvfx resume 최신 ckpt (--init_vae_from, weights-only 로드 → step0 fresh).
#   VAE arch flag 는 1x52jvfx 와 동일해야 weight strict=False 로드가 맞물림.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
NUM_GPUS=8
PY=/home/kaist_peta/miniconda3/envs/kinemadae-fa4/bin/python
BS=${BS:-8}                 # DiT 없어 메모리 여유 큼 → 더 키울 수 있음 (16/32 시도 가능)
WAN_CKPT=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/checkpoints_persistent/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth
VIDEO_TRAIN=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/KinemaDAE-kk4aiq-to-lora/panda70m_train.txt
VIDEO_EVAL=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/KinemaDAE-kk4aiq-to-lora/panda70m_eval.txt
# base aligned VAE — 1x52jvfx resume 최신 저장완료 ckpt (override: INIT_CKPT=... bash ...)
INIT_CKPT=${INIT_CKPT:-/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/Kinemadae-adaptive-Bfix/results/kinemadae_stage1_bn_lora_align40_bs8_clamp1e5fix_b_proj_teacherfrozen-lr8.00e-05-bs8-rs256-sr2-fr17/checkpoint-8500.ckpt}
export KINEMADAE_DIFFSYNTH_PATH=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/KinemaDAE-kk4aiq/external/DiffSynth-Studio
export KINEMADAE_PROBING_PATH=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/KinemaDAE-kk4aiq/external/dit_probing
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
$PY -m torch.distributed.run --nproc_per_node=$NUM_GPUS --standalone train_causalvae_geoprior_decoder_only.py \
    --exp_name "decoder_only_from1x52jvfx_256_bs${BS}" \
    --pretrained_model_name_or_path "$WAN_CKPT" \
    --add_encoder_stages '[{"mode":"downsample3d","num_res_blocks":2,"init":"zero"}]' \
    --add_decoder_before_head_stages '[{"mode":"upsample3d","num_res_blocks":2}]' \
    --z_dim 16 \
    --no_expand_conv2 \
    --unfreeze_decoder \
    --subsample_mode bilinear \
    --normalize_zprior \
    --video_path "$VIDEO_TRAIN" \
    --eval_video_path "$VIDEO_EVAL" \
    --num_frames 17 \
    --resolution 256 \
    --batch_size $BS \
    --lr 8e-5 \
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
    --no_log_grad \
    --seed 1234 \
    --dataset_num_worker 5 \
    --init_vae_from "$INIT_CKPT"
