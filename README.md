# Decoder-only finetuning for geoprior Wan-VAE

Finetune **only the decoder** of an already-trained geoprior Wan-VAE, with the
**encoder frozen** (so the `z_main` latent stays fixed). This polishes
reconstruction quality on a frozen, already-aligned latent without disturbing the
latent's structure (diffusability / alignment) that downstream diffusion depends on.

- Encoder + `conv1` are frozen → `z_main` unchanged.
- Only the decoder trains (≈77M params), driven by pure VAE recon loss (L1 + LPIPS).
- `val_ema` = **raw (frozen) encoder + EMA decoder** → clean recon metric, latent stats constant.
- Self-contained: no external repo needed (DiffSynth/dit_probing are **not** required).

## Setup

```bash
conda create -n decvae python=3.11 -y && conda activate decvae
pip install -r requirements.txt
```

## What you need to provide

| Item | Notes |
|---|---|
| **Wan2.1 VAE** (`Wan2.1_VAE.pth`) | pretrained Wan2.1-I2V VAE weights, used as architecture init |
| **Start checkpoint** | your trained geoprior-VAE checkpoint (the one whose latent you want to keep). Used either as a weights-only warm start (`INIT_CKPT`) or full resume (`RESUME_CKPT`) |
| **Video lists** | `train_videos.txt` / `eval_videos.txt`: one absolute video path per line |

Checkpoints are large and are **not** included here — point the launch script at your own copies.

## Run

Edit the paths at the top of `run_decoder_only.sh`, then:

```bash
# fresh start from a checkpoint (weights only, step 0):
INIT_CKPT=/path/to/vae.ckpt bash run_decoder_only.sh

# OR full resume (optimizer/step/EMA restored):
RESUME_CKPT=/path/to/checkpoint-N.ckpt bash run_decoder_only.sh
```

Common overrides (env vars): `NUM_GPUS`, `BS`, `GRAD_ACCUM`, `LR`, `EVAL_STEPS`, `EXP_TAG`.

## Important constraints

- **Batch size cap at 256×256×81: per-GPU `BS=2`.** The decoder spatial upsample
  (`upsample_nearest_nhwc`) backward uses int32 indexing; `B·T·192·256·256` must stay
  under `INT_MAX` (2.15e9). `BS=2`→2.04e9 OK, `BS≥3`→fail. This is a kernel limit, not
  memory. Use `GRAD_ACCUM` for a larger effective batch (model has 0 BatchNorm → grad
  accum is mathematically equal to a true larger batch).
- **EMA** only swaps trainable params (the decoder). The frozen encoder is never EMA-
  swapped, so `val_ema` = raw encoder + EMA decoder.

## Logged metrics (wandb)

- `val_ema/psnr`, `val_ema/recon`, `val_ema/lpips` — base (256×17)
- `val_ema_480x832x81/psnr`, `.../lpips` — HD (480×832×81)
- `val_ema/diffusability_pr`, `.../diffusability_lowfreq` — latent structure (constant: encoder frozen)
- `val_ema/noise_robust_s{0.1,0.2,0.3}` — noisy-decode vs clean-decode (decoder stability)
- `val_ema/noise_robust_gt_s{...}` — noisy-decode vs GT input (recon under per-channel latent noise)

## Layout

```
src/                                       # all code (self-contained)
  train_causalvae_geoprior_decoder_only.py # entry point
  kinemadae.py / kinemadae_geoprior.py     # Wan VAE + geoprior VAE
  perceptual_loss.py lpips_local.py discriminator.py
  ema_model.py ddp_sampler.py video_dataset.py video_utils.py
  transform.py distrib_utils.py adaptive_weighted_causal_conv_3d.py taming_download.py
run_decoder_only.sh                        # launch
requirements.txt
```
