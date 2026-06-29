#!/bin/bash
# [decoder-only 256x81 from 11500] aligned latent(z_main) 고정, decoder만 학습해서 recon 개선.
#   목적: decoder-only finetune(배치 큼)이 full finetune(clipfix, align+encoder+decoder, BS2)만큼 recon 올리나 비교.
#   base = preserved_ckpt11500 (clipfix와 동일 출발점, --init_vae_from weights-only → step0 fresh).
#   eval 섭셋/개수 = clipfix와 무조건 통일 (base 100=104패딩, HD 16, 480x832x81).
#   train_causalvae_geoprior_decoder_only.py = align/DiT 전부 제거, encoder freeze, decoder만 + save_checkpoint rotation 없음(전부 보존).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
NUM_GPUS=8
PY=/home/kaist_peta/miniconda3/envs/kinemadae-fa4/bin/python
BS=${BS:-4}                 # 메모리 테스트로 최종 확정 (256x81은 17보다 무거움; override: BS=N bash ...)
WAN_CKPT=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/checkpoints_persistent/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth
VIDEO_TRAIN=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/KinemaDAE-kk4aiq-to-lora/panda70m_train.txt
VIDEO_EVAL=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/KinemaDAE-kk4aiq-to-lora/panda70m_eval.txt
# base = 11500 완전본 (clipfix와 동일 출발점)
INIT_CKPT=${INIT_CKPT:-/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/Kinemadae-adaptive-Bfix/results/preserved_ckpt11500_for_clipfix.ckpt}
export KINEMADAE_DIFFSYNTH_PATH=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/KinemaDAE-kk4aiq/external/DiffSynth-Studio
export KINEMADAE_PROBING_PATH=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/KinemaDAE-kk4aiq/external/dit_probing
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
$PY -m torch.distributed.run --nproc_per_node=$NUM_GPUS --standalone train_causalvae_geoprior_decoder_only.py \
    --exp_name "decoder_only_from11500_256x81_bs${BS}${EXP_TAG:-}" \
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
    --grad_accum_steps ${GRAD_ACCUM:-1} \
    --lr ${LR:-8e-5} \
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
    --ema \
    --ema_decay 0.999 \
    --eval_lpips \
    --eval_subset_size 100 \
    --eval_hd_subset_size 16 \
    --eval_resolutions_hd "480x832x81" \
    --eval_sample_rate 1 \
    --eval_batch_size 4 \
    --find_unused_parameters \
    --no_log_grad \
    --seed 1234 \
    --dataset_num_worker 5 \
    --init_vae_from "$INIT_CKPT"
