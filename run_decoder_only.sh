#!/bin/bash
# Decoder-only finetuning of a geoprior Wan-VAE.
#   Encoder(+conv1) is FROZEN -> z_main latent fixed. Only the DECODER is trained
#   to improve reconstruction on a frozen, already-aligned latent.
#   val_ema = raw(frozen) encoder + EMA decoder (clean recon metric).
#
# ─── EDIT THESE PATHS (required) ─────────────────────────────────────────────
WAN_CKPT=${WAN_CKPT:-/path/to/Wan2.1_VAE.pth}                 # pretrained Wan2.1 VAE
VIDEO_TRAIN=${VIDEO_TRAIN:-/path/to/train_videos.txt}        # one video path per line
VIDEO_EVAL=${VIDEO_EVAL:-/path/to/eval_videos.txt}           # one video path per line
# Start point: choose ONE of the two below.
#   INIT_CKPT   = weights-only warm start (fresh optimizer, step 0)   -> --init_vae_from
#   RESUME_CKPT = full resume (optimizer+step+EMA+sampler restored)   -> --resume_from_checkpoint
INIT_CKPT=${INIT_CKPT:-/path/to/your_trained_geoprior_vae.ckpt}
RESUME_CKPT=${RESUME_CKPT:-}                                  # set this to resume instead of init
# ─────────────────────────────────────────────────────────────────────────────
NUM_GPUS=${NUM_GPUS:-8}
PY=${PY:-python}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/src"

# 256x256x81 trains at per-GPU batch 2 max (decoder upsample backward hits int32
# INT_MAX above that; not a memory limit). Use grad_accum for a larger effective batch.
BS=${BS:-2}
GRAD_ACCUM=${GRAD_ACCUM:-4}
LR=${LR:-8e-5}

# pick start-point flag
if [ -n "$RESUME_CKPT" ]; then START_FLAG=(--resume_from_checkpoint "$RESUME_CKPT");
else START_FLAG=(--init_vae_from "$INIT_CKPT"); fi

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$PY -m torch.distributed.run --nproc_per_node=$NUM_GPUS --standalone train_causalvae_geoprior_decoder_only.py \
    --exp_name "decoder_only_${EXP_TAG:-run}" \
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
    --num_frames 81 \
    --resolution 256 \
    --batch_size $BS \
    --grad_accum_steps $GRAD_ACCUM \
    --lr $LR \
    --epochs 50 \
    --kl_weight 3e-6 \
    --perceptual_weight 3.0 \
    --disc_weight 0.5 \
    --disc_start 9999999 \
    --gan_last_layer decoder_head \
    --save_ckpt_step 500 \
    --eval_steps ${EVAL_STEPS:-500} \
    --log_steps 1 \
    --mix_precision bf16 \
    --ema --ema_decay 0.999 \
    --eval_lpips \
    --eval_noise_robust --noise_robust_sigmas "0.1,0.2,0.3" \
    --eval_subset_size 100 \
    --eval_hd_subset_size 16 \
    --eval_resolutions_hd "480x832x81" \
    --eval_sample_rate 1 \
    --eval_batch_size 4 \
    --find_unused_parameters \
    --no_log_grad \
    --seed 1234 \
    --dataset_num_worker 5 \
    "${START_FLAG[@]}"
