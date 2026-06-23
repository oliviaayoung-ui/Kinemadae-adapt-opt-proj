#!/bin/bash
# [step3 동치검증] stage1 align dump — 고정 fixture(stage2 생성) 로드 → features/loss/grad dump.
# base = launch_train_stage2align_vaefrozen_loraFreeze5500.sh (checkpoint-5500 resume, 동일 align config).
# 1 GPU/batch1. dump 은 compute_alignment_loss 직후 exit → diffusion/optimizer 미실행.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PY=/home/kaist_peta/miniconda3/envs/kinemadae-fa4/bin/python
WAN_CKPT=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/checkpoints_persistent/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth
DIT_CKPT_DIR=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/checkpoints_persistent/Wan2.1-I2V-14B-480P
VIDEO_TRAIN=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/KinemaDAE-kk4aiq-to-lora/panda70m_train.txt
VIDEO_EVAL=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/KinemaDAE-kk4aiq-to-lora/panda70m_eval.txt
CAPTION_META=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/KinemaDAE-kk4aiq-to-lora/data/panda70m_metadata_captioned.jsonl
export KINEMADAE_DIFFSYNTH_PATH=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/KinemaDAE-kk4aiq/external/DiffSynth-Studio
export KINEMADAE_PROBING_PATH=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/KinemaDAE-kk4aiq/external/dit_probing
EQUIV_DIR=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/_align_equiv

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
ALIGN_EQUIV_FIXTURE="$EQUIV_DIR/fixture.pt" \
ALIGN_EQUIV_DUMP_OUT="$EQUIV_DIR/dump_stage1.pt" \
$PY -m torch.distributed.run --nproc_per_node=1 --standalone train_causalvae_geoprior_dit_align.py \
    --exp_name "kinemadae_equiv_dump_stage1" \
    --pretrained_model_name_or_path "$WAN_CKPT" \
    --add_encoder_stages '[{"mode":"downsample3d","num_res_blocks":2,"init":"zero"}]' \
    --add_decoder_before_head_stages '[{"mode":"upsample3d","num_res_blocks":2}]' \
    --z_dim 16 \
    --no_expand_conv2 \
    --subsample_mode bilinear \
    --video_path "$VIDEO_TRAIN" \
    --eval_video_path "$VIDEO_EVAL" \
    --num_frames 17 \
    --resolution 256 \
    --batch_size 1 \
    --lr 8e-5 \
    --patchify_lr 1e-4 \
    --epochs 50 \
    --kl_weight 3e-6 \
    --perceptual_weight 3.0 \
    --disc_weight 0.5 \
    --disc_start 9999999 \
    --gan_last_layer decoder_head \
    --save_ckpt_step 9999999 \
    --eval_steps 9999999 \
    --log_steps 1 \
    --mix_precision bf16 \
    --find_unused_parameters \
    --dit_ckpt_dir "$DIT_CKPT_DIR" \
    --align_weight 1.0 \
    --align_loss_type cosine \
    --align_layers all \
    --align_agg sum \
    --align_adaptive_weight \
    --adaptive_max_weight 10000000 \
    --grad_accum_steps 1 \
    --patchify_init zero \
    --patchify_mask_init copy4_zero4 \
    --normalize_zprior \
    --freeze_patchify_zprior \
    --caption_metadata "$CAPTION_META" \
    --align_num_blocks 40 \
    --use_b_adaptive \
    --seed 1234 \
    --dataset_num_worker 2 \
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
    --teacher_frozen_pretrained \
    --freeze_encoder \
    --freeze_decoder \
    --resume_from_checkpoint /NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/Kinemadae-adaptive-Bfix/results/kinemadae_stage1_bn_lora_align40_bs8_loraFreeze_matchS2_clamp1e5_b_proj_teacherfrozen-lr8.00e-05-bs8-rs256-sr2-fr17/checkpoint-5500.ckpt 2>&1
