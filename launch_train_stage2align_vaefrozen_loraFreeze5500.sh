#!/bin/bash
# Stage1 align + BN + LoRA + teacher_frozen + DIFFUSION LOSS (align_stop_grad 없음)
# Base: launch_train_stage1_bn_lora_teacherfrozen.sh 에 diffusion loss 추가.
# 설계:
#   - diffusion loss (flow matching, REPA-E): run_student_diffusion_forward 가 z_cat.detach() 로 시작
#     → diffusion grad 는 VAE 로 안 흐름 (= 요청한 "diffusion 만 VAE 차단", z.detach 로 자동).
#   - align_stop_grad_dit 없음: align 은 VAE + 공유 DiT(LoRA/patchify/block_norms) 둘 다로 흐름 (충돌 감수).
#   - teacher_frozen_pretrained 유지: teacher forward 시 LoRA off + block_norms swap → align target = pretrained Wan 고정.
#   - LoRA = Wan I2V 480 pretrained random init (lora_B=0). baseline 256 init 안 함 (init_lora_safetensors 없음).
# diffusion trainable: student_patchify z_prior(unfreeze) + davae_head + LoRA + block_norms/mod(unfreeze).
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
    --exp_name "kinemadae_stage2align_vaefrozen_bs${BS}${EXP_SUFFIX}_loraFreeze5500" \
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
    --teacher_frozen_pretrained \
    --freeze_encoder \
    --freeze_decoder \
    --resume_from_checkpoint /NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/Kinemadae-adaptive-Bfix/results/kinemadae_stage1_bn_lora_align40_bs8_loraFreeze_matchS2_clamp1e5_b_proj_teacherfrozen-lr8.00e-05-bs8-rs256-sr2-fr17/checkpoint-5500.ckpt
