# Inference Benchmarks

These tables collect inference benchmarks for Cosmos3. **Generator** sections measure visual-generation and world-model latency across PyTorch, vLLM-Omni, Diffusers, and NIM, including image and video generation, forward and inverse dynamics, and policy generation. **Reasoner** sections measure VLM serving and token-generation performance for text, image, and video inputs through vLLM and Hugging Face Transformers, with NIM and vLLM results compared side by side for Nano and Super.

Generator results are published incrementally from internal benchmark runs. **Empty cells mean that no result is published for that combination** — not that it is unsupported. In the refreshed NIM tables, failed runs also remain blank. See the notes under each table for workload details and data-source definitions.

## Table of Contents

- [Cosmos3-Edge Generator](#cosmos3-edge-generator)
  - [vLLM-Omni](#vllm-omni)
  - [PyTorch](#pytorch)
- [Cosmos3-Nano Generator](#cosmos3-nano-generator)
  - [Text-to-Video (t2v)](#text-to-video-t2v)
  - [Image-to-Video (i2v)](#image-to-video-i2v)
  - [Text-to-Image (t2i)](#text-to-image-t2i)
- [Cosmos3-Super Generator](#cosmos3-super-generator)
  - [Text-to-Video (t2v)](#text-to-video-t2v-1)
  - [Image-to-Video (i2v)](#image-to-video-i2v-1)
  - [Text-to-Image (t2i)](#text-to-image-t2i-1)
- [Cosmos3-Edge Reasoner](#cosmos3-edge-reasoner)
  - [RTX PRO 4500 Blackwell Server Edition](#rtx-pro-4500-blackwell-server-edition)
  - [RTX PRO 6000 Blackwell Server Edition](#rtx-pro-6000-blackwell-server-edition)
  - [Embedded-Platform Eager Transformers](#embedded-platform-eager-transformers)
- [Cosmos3-Nano Reasoner](#cosmos3-nano-reasoner)
  - [RTX PRO 6000 Blackwell](#rtx-pro-6000-blackwell)
  - [H20](#h20)
  - [H100 NVL](#h100-nvl)
  - [H200 NVL](#h200-nvl)
  - [H100 80GB HBM3 (SXM)](#h100-80gb-hbm3-sxm)
  - [H200 141GB HBM3](#h200-141gb-hbm3)
  - [B200](#b200)
  - [B300](#b300)
- [Cosmos3-Super Reasoner](#cosmos3-super-reasoner)
  - [RTX PRO 6000 Blackwell](#rtx-pro-6000-blackwell-1)
  - [H20](#h20-1)
  - [H100 NVL](#h100-nvl-1)
  - [H200 NVL](#h200-nvl-1)
  - [H100 80GB HBM3 (SXM)](#h100-80gb-hbm3-sxm-1)
  - [H200 141GB HBM3](#h200-141gb-hbm3-1)
  - [B200](#b200-1)
  - [B300](#b300-1)

## Cosmos3-Edge Generator

These tables report **Cosmos3-Edge** Generator latency in seconds for **image-to-video (i2v)**. Measurements use one GPU or one integrated computing platform. Lower latency is better, and empty cells indicate that a run has not been completed.

Unless otherwise noted, visual-generation benchmarks use **480p resolution**. Video benchmarks generate **121 frames**. vLLM-Omni values report end-to-end latency, while PyTorch values report average generation latency.

### vLLM-Omni

| GPU or Platform | Image-to-Video |
|---|---:|
| B200 SXM 192 GB | 7.09 |
| B300 | 8.74 |
| H200 SXM 141 GB | 13.13 |
| H200 NVL | 14.15 |
| H100 SXM 80 GB | 13.33 |
| H100 NVL 96 GB | 17.33 |
| H20 SXM 96 GB | 53.94 |
| RTX PRO 6000 Blackwell Server Edition |  |
| DGX Station | 7.13 |
| DGX Spark | 89.41 |
| Jetson AGX Thor T5000, 128 GB, MAXN |  |
| Jetson T3000, 32 GB, 1100 MHz |  |
| Jetson T2000, 16 GB, 702 MHz, THOR_NANO |  |

### PyTorch

| GPU or Platform | Image-to-Video |
|---|---:|
| B200 SXM 192 GB | 7.45 |
| B300 | 6.84 |
| H200 SXM 141 GB | 12.31 |
| H200 NVL | 14.07 |
| H100 SXM 80 GB | 12.68 |
| H100 NVL 96 GB | 16.42 |
| H20 SXM 96 GB | 52.83 |
| RTX PRO 6000 Blackwell Server Edition | 21.92 |
| DGX Station | 6.31 |
| DGX Spark | 103.36 |
| Jetson AGX Thor T5000, 128 GB, MAXN |  |
| Jetson T3000, 32 GB, 1100 MHz |  |

<sub>Notes:
1. All measurements use one GPU or one integrated computing platform.
2. Values are average end-to-end or generation latency in seconds; lower is better.
3. Unless otherwise specified, visual-generation measurements use **480p resolution**.
4. Video measurements (i2v) generate **121 output frames**.
5. PyTorch values report average generation latency rather than diffusion-only latency.</sub>

## Cosmos3-Nano Generator

These tables report **Cosmos3-Nano** generator latency in seconds for **image-to-video (i2v)**, **text-to-image (t2i)**, and **text-to-video (t2v)** - the primary vision-generation modes of the omni-model. OSS benchmarks use BF16 precision, batch size 1, and identical prompts, seeds, and sampler settings across engines where noted below. Video workloads follow the standard Cosmos3 generation profile (189 frames at 24 FPS unless a resolution tier limits frame count).

Four integration paths are compared. **PyTorch** reports average generation (sampling) time from OSS reference inference with CUDA Graphs enabled where supported. **vLLM-Omni** reports total pipeline time at **720p** on supported GPUs. **Diffusers** reports end-to-end generation time through the Hugging Face `Cosmos3OmniPipeline` without custom CUDA graphs at **256p/1**, **480p/1**, and **720p/1** (320×192, 832×480, and 1280×720). **NIM** reports `Avg. Generation Time (s)` from FP8 latency-profile runs with offload disabled, excluding request overhead such as MP4 encoding and response delivery. Empty cells indicate a run has not been completed for that GPU, engine, resolution, or tensor-parallel width - tables are filled in as benchmark campaigns finish.

### Text-to-Video (t2v)

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 13.95 |  | 4.90 | 180.81 |  |  | 786.37 | 225.45 | 127.57 |
| | vLLM-Omni | 10.65 | 5.06 | 3.78 | 105.61 | 35.93 | 23.76 | 369.67 | 114.30 | 68.66 |
| | Diffusers | 11.20 | | | 112.00 | | | 392.00 | | |
| | NIM | 7.24 | 3.45 | 2.83 | 81.35 | 25.41 | 17.04 | 313.63 | 90.81 | 49.33 |
| **H20** | PyTorch | 30.57 |  |  | 257.51 |  |  | 931.39 | 268.88 | 157.71 |
| | vLLM-Omni | 28.58 | 10.20 | 7.70 | 256.97 | 77.42 | 47.53 | 929.81 | 260.75 | 148.46 |
| | Diffusers | 30.20 | | | 258.00 | | | 926.00 | | |
| | NIM | 18.78 | 5.88 | 3.84 | 195.35 | 53.77 | 28.51 | 776.37 | 203.35 | 105.11 |
| **H100 NVL** | PyTorch | 10.03 | 4.27 | 3.95 | 84.12 | 29.18 | 21.46 | 297.27 | 94.15 | 61.63 |
| | vLLM-Omni | 9.25 | 3.68 | 3.15 | 80.75 | 27.48 | 18.77 | 311.13 | 88.25(*) | 54.01(*) |
| | Diffusers | 11.00 | | | 90.00 | | | 324.20 | | |
| | NIM | 6.70 | 2.50 | 2.46 | 67.09 | 24.35 | 12.96 | 252.22 | 102.48 | 48.50 |
| **H200 NVL** | PyTorch | 8.17 |  |  | 69.79 |  |  | 244.39 | 77.35 | 45.70 |
| | vLLM-Omni | 7.44 | 3.27 | 2.33 | 64.58 | 21.31 | 12.92 | 240.05 | 69.63 | 39.17 |
| | Diffusers | 9.00 | | | 74.00 | | | 276.20 | | |
| | NIM | 5.42 | 2.62 | 2.14 | 56.85 | 16.95 | 10.30 | 227.34 | 62.77 | 33.53 |
| **H100 80GB HBM3** | PyTorch | 7.61 | 3.50 | 3.17 | 59.83 | 21.23 | 14.37 | 207.78 | 66.94 | 41.81 |
| | vLLM-Omni | 6.97 | 3.45 | 3.49 | 58.17 | 19.95 | 13.46 | 202.29 | 62.82 | 37.80 |
| | Diffusers | 9.00 | | | 68.00 | | | 240.00 | | |
| | NIM | 5.35 | 2.24 | 1.81 | 46.08 | 13.78 | 7.77 | 173.20 | 48.35 | 26.14 |
| **H200 141GB HBM3** | PyTorch | 7.53 | 3.34 | 3.19 | 60.18 | 20.84 | 13.97 | 214.28 | 67.48 | 41.26 |
| | vLLM-Omni | 6.79 | 3.25 | 3.42 | 58.14 | 19.77 | 12.97 | 208.36 | 63.27 | 37.49 |
| | Diffusers | 9.00 | | | 67.00 | | | 239.60 | | |
| | NIM | 5.02 | 2.12 | 1.85 | 46.93 | 13.65 | 7.56 | 177.23 | 48.75 | 25.78 |
| **B200** | PyTorch | 4.56 | 2.78 | 2.79 | 33.20 | 13.20 | 9.69 | 114.85 | 39.75 | 26.27 |
| | vLLM-Omni | 4.03 | 2.43 | 3.49 | 32.04 | 12.63 | 10.09 | 107.84 | 35.29 | 22.87 |
| | Diffusers | 7.00 | | | 36.80 | | | 117.00 | | |
| | NIM | 2.94 | 1.57 | 1.56 | 25.03 | 7.54 | 4.41 | 91.62 | 25.97 | 14.50 |
| **B300** | PyTorch | | | | | | | | | |
| | vLLM-Omni | 4.46 | 4.11 | 5.44 | 32.18 | 13.83 | 11.57 | 102.10 | 35.68 | 24.33 |
| | Diffusers | 39.40 | | | 63.40 | | | 139.40 | | |
| | NIM | 2.86 | 2.48 | 2.50 | 23.67 | 7.36 | 4.82 | 85.52 | 24.41 | 13.80 |

### Image-to-Video (i2v)

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch |  |  |  | 182.14 |  |  | 788.80 | 226.25 | 127.79 |
| | vLLM-Omni | 11.04 | 5.48 | 4.24 | 107.77 | 38.05 | 25.95 | 375.01 | 119.27 | 73.57 |
| | Diffusers | 12.00 | | | 112.00 | | | 397.00 | | |
| | NIM | 7.27 | 3.50 | 2.98 | 81.50 | 25.54 | 16.03 | 313.64 | 90.98 | 49.97 |
| **H20** | PyTorch | 31.36 |  |  | 257.10 |  |  | 933.07 | 268.99 | 158.10 |
| | vLLM-Omni | 29.50 | 11.26 | 8.64 | 261.56 | 81.93 | 52.06 | 940.16 | 271.37 | 158.76 |
| | Diffusers | 31.00 | | | 258.00 | | | 925.00 | | |
| | NIM | 18.69 | 5.97 | 3.96 | 195.87 | 53.73 | 28.66 | 776.48 | 203.76 | 105.06 |
| **H100 NVL** | PyTorch | 10.19 | 4.31 | 3.99 | 84.50 | 28.69 | 21.52 | 298.57 | 95.76 | 60.58 |
| | vLLM-Omni | 9.62 | 4.11 | 3.63 | 82.61 | 29.35 | 20.73 | 286.33 | 92.23(*) | 58.02(*) |
| | Diffusers | 11.00 | | | 91.00 | | | 325.20 | | |
| | NIM | 6.76 | 2.61 | 2.53 | 67.50 | 24.29 | 13.18 | 251.21 | 99.98 | 48.97 |
| **H200 NVL** | PyTorch | 8.27 |  |  | 69.99 |  |  | 246.62 | 77.69 | 45.99 |
| | vLLM-Omni | 7.83 | 3.69 | 2.78 | 66.39 | 22.93 | 14.58 | 243.52 | 73.26 | 42.86 |
| | Diffusers | 9.00 | | | 74.00 | | | 275.20 | | |
| | NIM | 5.43 | 2.72 | 2.32 | 57.02 | 17.07 | 10.58 | 226.12 | 62.81 | 33.63 |
| **H100 80GB HBM3** | PyTorch | 7.64 | 3.47 | 3.21 | 59.95 | 21.40 | 14.43 | 207.87 | 67.52 | 41.66 |
| | vLLM-Omni | 7.37 | 3.81 | 3.97 | 59.77 | 21.68 | 15.12 | 205.97 | 66.52 | 41.51 |
| | Diffusers | 9.00 | | | 68.00 | | | 239.80 | | |
| | NIM | 5.37 | 2.29 | 1.93 | 46.34 | 13.88 | 7.92 | 173.14 | 48.33 | 26.37 |
| **H200 141GB HBM3** | PyTorch | 7.65 | 3.37 | 3.17 | 60.51 | 21.01 | 14.07 | 214.80 | 67.14 | 41.00 |
| | vLLM-Omni | 7.28 | 3.63 | 3.83 | 59.64 | 21.35 | 14.67 | 209.65 | 66.65 | 40.77 |
| | Diffusers | 9.00 | | | 67.20 | | | 240.00 | | |
| | NIM | 5.07 | 2.19 | 1.95 | 46.95 | 13.72 | 7.69 | 177.57 | 48.66 | 26.00 |
| **B200** | PyTorch | 4.60 | 2.77 | 2.81 |  | 13.07 | 9.66 | 113.90 | 40.01 | 26.58 |
| | vLLM-Omni | 4.33 | 2.77 | 3.84 | 33.09 | 13.79 | 11.39 | 110.19 | 37.76 | 25.68 |
| | Diffusers | | | | | | | 116.00 | | |
| | NIM | 2.98 | 1.62 | 1.71 | 25.59 | 7.62 | 4.59 | 91.65 | 26.13 | 14.50 |
| **B300** | PyTorch | | | | | | | | | |
| | vLLM-Omni | 5.61 | 4.67 | 5.90 | 33.45 | 15.06 | 13.13 | 104.75 | 38.27 | 26.87 |
| | Diffusers | 28.60 | | | 65.60 | | | 139.60 | | |
| | NIM | 2.88 | 2.48 | 2.62 | 23.74 | 7.45 | 4.97 | 85.50 | 24.56 | 14.20 |

### Text-to-Image (t2i)

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 2.99 |  |  | 4.51 |  |  | 7.12 | 3.18 | 2.70 |
| | vLLM-Omni | 1.59 | 1.54 | 1.59 | 2.87 | 1.55 | 1.81 | 4.99 | 2.32 | 1.96 |
| | Diffusers | 2.00 | | | 4.00 | | | 5.00 | | |
| | NIM | 1.01 | 0.90 | 0.89 | 1.33 | 0.92 | 0.94 | 2.25 | 1.13 | 1.00 |
| **H20** | PyTorch | 3.06 |  |  | 6.51 |  |  | 12.31 | 4.28 | 3.06 |
| | vLLM-Omni | 1.73 | 2.46 | 3.22 | 4.92 | 2.57 | 7.24 | 10.73 | 4.24 | 3.59 |
| | Diffusers | 3.00 | | | 6.00 | | | 10.00 | | |
| | NIM | 1.18 | 1.06 | 1.09 | 2.54 | 1.13 | 2.98 | 4.70 | 1.61 | 1.18 |
| **H100 NVL** | PyTorch | 2.77 | 2.45 | 2.57 | 2.83 | 2.56 | 2.51 | 4.21 | 2.57 | 2.64 |
| | vLLM-Omni | 1.55 | 1.75 | 1.91 | 1.92 | 1.81 | 10.82 | 3.44 | 1.83 | 1.90 |
| | Diffusers | 3.00 | | | 3.00 | | | 4.00 | | |
| | NIM | 1.14 | 0.96 | 0.97 | 1.15 | 0.98 | 2.72 | 1.72 | 1.00 | 1.00 |
| **H200 NVL** | PyTorch |  |  |  | 2.85 |  |  | 3.58 | 2.62 | 2.64 |
| | vLLM-Omni | 1.53 | 2.01 | 1.96 | 1.58 | 1.91 | 17.71 | 2.81 | 1.94 | 1.94 |
| | Diffusers | 3.00 | | | 3.00 | | | 4.00 | | |
| | NIM | 1.12 | 0.92 | 1.03 | 1.16 | 0.98 | 2.81 | 1.46 | 0.98 | 1.04 |
| **H100 80GB HBM3** | PyTorch | 3.01 | 2.66 | 2.56 | 3.01 | 2.59 | 2.75 | 3.45 | 2.73 | 2.77 |
| | vLLM-Omni | 1.61 | 2.45 | 3.18 | 1.53 | 2.35 | 7.02 | 2.61 | 2.45 | 3.03 |
| | Diffusers | 3.00 | | | 3.00 | | | 4.00 | | |
| | NIM | 1.10 | 0.98 | 1.02 | 1.24 | 1.00 | 2.93 | 1.51 | 1.08 | 1.07 |
| **H200 141GB HBM3** | PyTorch | 2.96 | 2.59 | 2.70 | 3.04 | 2.78 | 2.77 | 3.28 | 2.84 | 2.77 |
| | vLLM-Omni | 1.57 | 2.38 | 3.16 | 1.52 | 2.37 | 7.05 | 2.60 | 2.33 | 3.20 |
| | Diffusers | 3.00 | | | 3.00 | | | 4.00 | | |
| | NIM | 1.17 | 1.04 | 1.09 | 1.18 | 0.99 | 2.93 | 1.42 | 1.06 | 1.10 |
| **B200** | PyTorch |  | 2.39 | 2.59 | 2.75 | 2.43 | 2.56 | 2.87 | 2.58 | 2.62 |
| | vLLM-Omni | 1.49 | 2.21 | 3.27 | 1.20 | 2.05 | 7.58 | 1.77 | 2.20 | 3.41 |
| | Diffusers | | | | | | | 3.00 | | |
| | NIM | 0.99 | 0.87 | 0.90 | 1.06 | 0.91 | 1.02 | 1.02 | 0.89 | 0.91 |
| **B300** | PyTorch | | | | | | | | | |
| | vLLM-Omni | 1.97 | 4.52 | 5.82 | 1.81 | 4.16 | 71.19 | 2.34 | 4.09 | 5.62 |
| | Diffusers | 36.20 | | | | | | 41.00 | | |
| | NIM | 1.45 | 1.36 | 1.38 | 1.50 | 1.40 | 1.45 | 1.48 | 1.37 | 1.47 |

<sub>Notes:
1. OSS times use matched workloads (same seed, sampler settings, prompt).
2. 4×/8× GPU configurations use tensor parallelism.
3. vLLM-Omni numbers are for the upcoming public release in the vLLM-Omni repo; subject to change before GA. Values marked with (*) are pre-release vLLM-Omni measurements on H100 NVL and may change before GA.
4. Diffusers numbers use the HuggingFace `diffusers` integration without custom CUDA graphs; reported at 256p/1, 480p/1, and 720p/1 (single-GPU only).
5. PyTorch numbers report average generation (sampling) time from OSS inference benchmarking.
6. At 256p, multi-GPU configurations on B300 may underperform single-GPU due to small-workload TP overhead; single-GPU is recommended at this resolution.
7. NIM values report average generation time in seconds, excluding request overhead and MP4 encoding. Runs use FP8 latency profiles with offload disabled, concurrency 1, and three measured requests. Video workloads generate 189 frames; image workloads generate one frame.</sub>

## Cosmos3-Super Generator

These tables report **Cosmos3-Super** generator latency in seconds for **image-to-video (i2v)**, **text-to-image (t2i)**, and **text-to-video (t2v)**. The 32B checkpoint targets higher-quality world generation; expect longer runtimes than Nano at the same resolution. OSS benchmarks use BF16 precision, batch size 1, and matched prompts, seeds, and sampler settings. Video workloads follow the standard Cosmos3 profile (189 frames at 24 FPS where applicable).

As with Nano, four engines are tracked: **PyTorch** (OSS generation/sampling time), **vLLM-Omni** (total pipeline time at 720p on supported GPUs), **Diffusers** (Hugging Face `Cosmos3OmniPipeline` end-to-end time at **256p/1**, **480p/1**, and **720p/1** — 320×192, 832×480, and 1280×720), and **NIM** (`Avg. Generation Time (s)` using FP8 latency profiles with offload disabled, excluding request overhead such as MP4 encoding and response delivery). Super coverage is narrower than Nano in early releases - for example, vLLM-Omni and Diffusers runs exist primarily on B200 and select H200 configurations. **Empty cells indicate missing or failed measurements**, not unsupported configurations.

### Text-to-Video (t2v)

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch |  |  |  |  |  |  |  | 789.03 | 427.16 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| | NIM | 26.91 | 10.40 | 7.27 | 291.97 | 89.56 | 49.69 | 1152.93 | 319.95 | 171.15 |
| **H20** | PyTorch | | | | | | | | | 492.41 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| | NIM | 67.93 | 20.46 | 11.67 | 711.10 | 191.68 | 98.96 | 2815.02 | 733.55 | 370.48 |
| **H100 NVL** | PyTorch |  |  | 16.83 |  | 101.27 | 64.14 |  | 330.04 | 186.19 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| | NIM | 24.54 | 8.13 | 6.19 | 267.92 | 106.45 | 47.20 | 937.49 | 384.12 | 182.00 |
| **H200 NVL** | PyTorch | | | | | | | | 258.34 | 139.37 |
| | vLLM-Omni | 27.54 | | 5.06 | 252.33 | | 36.66 | 911.49 | 245.51 | 123.85 |
| | Diffusers | 33.00 | | | 286.80 | | | 1036.00 | | |
| | NIM | 19.32 | 7.56 | 5.00 | 222.13 | 60.91 | 32.38 | 840.59 | 230.97 | 117.52 |
| **H100 80GB HBM3** | PyTorch | | | | | | | | | |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| | NIM |  |  |  |  |  |  |  |  |  |
| **H200 141GB HBM3** | PyTorch |  | 14.82 | 11.82 |  | 70.27 | 41.78 |  | 224.43 | 123.49 |
| | vLLM-Omni | 25.61 | | 5.87 | 219.11 | | 35.26 | 769.63 | 212.30 | 111.94 |
| | Diffusers | 31.00 | | | 251.60 | | | 886.20 | | |
| | NIM | 17.51 | 5.66 | 3.55 | 170.11 | 47.23 | 24.50 | 639.87 | 173.93 | 88.18 |
| **B200** | PyTorch |  | 5.59 | 4.09 | 114.38 | 35.73 | 21.39 | 407.50 | 118.38 | 65.93 |
| | vLLM-Omni | 13.84 | | 4.76 | 114.08 | | 22.09 | 390.28 | 113.31 | 62.11 |
| | Diffusers | | | | 127.20 | | | 414.40 | | |
| | NIM | 9.39 | 3.49 | 2.46 | 94.04 | 25.14 | 13.30 | 346.99 | 92.65 | 46.93 |
| **B300** | PyTorch | | | | | | | | | |
| | vLLM-Omni | 14.57 | | 6.68 | 109.03 | | 22.67 | 366.66 | 108.58 | 60.73 |
| | Diffusers | 54.20 | | | 155.40 | | | 424.80 | | |
| | NIM | 9.29 | 3.80 | 3.58 | 88.67 | 23.87 | 13.50 | 322.94 | 86.59 | 44.95 |

### Image-to-Video (i2v)

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch |  |  |  |  |  |  |  | 795.14 | 427.96 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| | NIM | 26.96 | 10.47 | 7.58 | 292.49 | 89.68 | 51.55 | 1157.15 | 320.55 | 171.79 |
| **H20** | PyTorch | | | | | | | | 931.74 | |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| | NIM | 68.26 | 20.53 | 11.80 | 712.78 | 191.45 | 99.06 | 2815.06 | 733.13 | 369.69 |
| **H100 NVL** | PyTorch |  | 20.85 | 16.96 |  | 99.56 |  |  | 331.40 | 186.47 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| | NIM | 24.59 | 8.21 | 6.23 | 266.77 | 105.38 | 46.21 | 939.10 | 386.91 | 183.17 |
| **H200 NVL** | PyTorch |  |  |  |  |  |  |  | 265.33 | 138.31 |
| | vLLM-Omni | 27.90 | | 5.52 | 254.29 | | 38.51 | 915.05 | 248.89 | 127.32 |
| | Diffusers | 33.00 | | | 287.20 | | | 1034.60 | | |
| | NIM | 19.42 | 7.57 | 5.14 | 222.95 | 60.97 | 32.61 | 840.66 | 230.28 | 117.43 |
| **H100 80GB HBM3** | PyTorch | | | | | | | | | |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| | NIM |  |  |  |  |  |  |  |  |  |
| **H200 141GB HBM3** | PyTorch |  | 14.87 | 11.80 |  | 70.45 | 42.10 |  | 224.36 | 123.57 |
| | vLLM-Omni | 25.47 | | 6.32 | 220.70 | | 36.90 | 766.33 | 215.03 | 117.52 |
| | Diffusers | 31.00 | | | 249.20 | | | 879.20 | | |
| | NIM | 17.53 | 5.76 | 3.70 | 170.44 | 47.14 | 24.71 | 639.64 | 173.77 | 88.45 |
| **B200** | PyTorch | 14.71 | 5.63 | 4.12 |  | 35.70 | 21.25 | 397.31 | 117.98 | 65.91 |
| | vLLM-Omni | 14.13 | | 5.31 | 115.17 | | 23.26 | 393.02 | 115.69 | 64.82 |
| | Diffusers | 19.20 | | | | | | 414.80 | | |
| | NIM | 9.42 | 3.53 | 2.58 | 92.08 | 25.12 | 13.56 | 340.06 | 93.12 | 46.21 |
| **B300** | PyTorch | | | | | | | | | |
| | vLLM-Omni | 14.14 | | 7.19 | 111.42 | | 23.91 | 368.73 | 111.41 | 63.25 |
| | Diffusers | 54.20 | | | 151.80 | | | 425.00 | | |
| | NIM | 9.27 | 3.89 | 3.68 | 90.00 | 24.26 | 13.73 | 318.53 | 86.95 | 44.87 |

### Text-to-Image (t2i)

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | | | | | | | | 92.61 | 93.11 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| | NIM | 2.04 | 1.46 | 1.51 | 4.02 | 2.15 | 1.75 | 7.37 | 3.25 | 2.60 |
| **H20** | PyTorch |  |  |  |  |  |  |  |  | 20.92 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| | NIM | 2.22 | 1.79 | 1.76 | 10.18 | 3.76 | 5.24 | 18.16 | 6.07 | 3.93 |
| **H100 NVL** | PyTorch | | 19.73 | 19.86 | | | | | 20.68 | 19.87 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| | NIM | 1.91 | 1.61 | 1.74 | 3.73 | 1.71 | 4.79 | 6.50 | 2.46 | 2.13 |
| **H200 NVL** | PyTorch |  |  |  |  |  |  |  | 32.64 | 33.16 |
| | vLLM-Omni | 2.73 | | 3.20 | 6.28 | | 17.71 | 11.02 | 4.22 | 3.26 |
| | Diffusers | 5.00 | | | 8.00 | | | 12.00 | | |
| | NIM | 1.93 | 1.62 | 1.66 | 3.25 | 1.63 | 4.78 | 5.15 | 2.43 | 1.89 |
| **H100 80GB HBM3** | PyTorch | | | | | | | | | |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| | NIM |  |  |  |  |  |  |  |  |  |
| **H200 141GB HBM3** | PyTorch |  | 13.62 | 13.48 |  | 13.33 | 13.53 |  | 13.78 | 13.50 |
| | vLLM-Omni | 2.83 | | 4.50 | 5.70 | | 7.16 | 10.24 | 4.23 | 4.43 |
| | Diffusers | 5.00 | | | 8.00 | | | 11.00 | | |
| | NIM | 1.98 | 1.77 | 1.78 | 3.08 | 1.76 | 5.15 | 4.79 | 1.86 | 1.79 |
| **B200** | PyTorch |  | 4.10 | 4.27 | 4.78 | 4.13 | 4.48 | 7.25 | 4.28 | 4.65 |
| | vLLM-Omni | 2.32 | | 4.58 | 3.29 | | 9.10 | 6.02 | 3.09 | 4.43 |
| | Diffusers | | | | | | | 8.00 | | |
| | NIM | 1.68 | 1.48 | 1.60 | 1.73 | 1.54 | 1.49 | 2.68 | 1.47 | 1.54 |
| **B300** | PyTorch | | | | | | | | | |
| | vLLM-Omni | 5.05 | | 7.62 | 3.79 | | 72.65 | 7.08 | 5.79 | 7.24 |
| | Diffusers | 38.80 | | | 39.40 | | | 40.40 | | |
| | NIM | 2.56 | 2.35 | 2.38 | 2.55 | 2.32 | 2.62 | 2.80 | 2.39 | 2.48 |

<sub>Notes:
1. OSS times use matched workloads (same seed, sampler settings, prompt).
2. 4×/8× GPU configurations use tensor parallelism.
3. vLLM-Omni numbers are for the upcoming public release in the vLLM-Omni repo; subject to change before GA. Current vLLM-Omni coverage is B200 at 720p.
4. Diffusers numbers use the HuggingFace `diffusers` integration without custom CUDA graphs; reported at 256p/1, 480p/1, and 720p/1 (single-GPU only).
5. At 256p, multi-GPU configurations on B300 may underperform single-GPU due to small-workload TP overhead; single-GPU is recommended at this resolution.
6. PyTorch numbers report average generation (sampling) time from OSS inference benchmarking.
7. NIM values report average generation time in seconds, excluding request overhead and MP4 encoding. Runs use FP8 latency profiles with offload disabled, concurrency 1, and three measured requests. Video workloads generate 189 frames; image workloads generate one frame.</sub>

### Additional Super Generator NIM profiles

Single-GPU H100 SXM 80GB generation times using FP8 latency profiles. Offload mode is shown separately; `unspecified` means it was not set explicitly.

| GPU | Offload mode | Modality | 256p/1 | 480p/1 | 720p/1 |
|---|---|---|---:|---:|---:|
| H100 80GB HBM3 | model | t2v | 23.98 | 174.92 | 631.25 |
| H100 80GB HBM3 | model | i2v |  |  | 631.04 |
| H100 80GB HBM3 | model | t2i | 7.98 |  | 10.55 |
| H100 80GB HBM3 | unspecified | t2v | 41.47 | 170.37 | 625.43 |
| H100 80GB HBM3 | unspecified | i2v | 41.50 | 170.60 | 625.66 |
| H100 80GB HBM3 | unspecified | t2i | 37.21 | 37.24 | 37.28 |

## Cosmos3-Edge Reasoner

These tables report **Cosmos3-Edge** reasoner serving performance through **vLLM**. Unlike the Generator benchmarks, Reasoner workloads produce autoregressively generated text and measure time to first token (TTFT), end-to-end request latency, request throughput, and output-token throughput. Lower is better for latency metrics; higher is better for throughput.

All vLLM runs use the **`nvidia/Cosmos3-Edge`** checkpoint with one GPU. Metrics were collected at client-side concurrency levels of 1, 64, 128, and 256. Each GPU section contains four workload tables that vary input sequence length, output sequence length, and video frame rate.

### RTX PRO 4500 Blackwell Server Edition

#### Input 50 / Output 1 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 165.79 | 8817.33 | 14702.20 | 29482.39 |
| Request Latency (ms) | 165.79 | 8817.33 | 14702.20 | 29482.39 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 6.00 | 6.55 | 6.55 | 6.52 |
| Output Token Throughput (Tok/s) | 6.00 | 6.55 | 6.55 | 6.52 |

#### Input 50 / Output 1 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 371.67 | 20375.98 | 33812.45 | 68201.55 |
| Request Latency (ms) | 371.67 | 20375.98 | 33812.45 | 68201.55 |
| Request Count (requests) | 50 | 313 | 249 | 492 |
| Request Throughput (Req/s) | 2.68 | 2.77 | 2.76 | 2.71 |
| Output Token Throughput (Tok/s) | 2.68 | 2.77 | 2.76 | 2.71 |

#### Input 50 / Output 100 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 166.86 | 6900.90 | 19625.83 | 45729.55 |
| Request Latency (ms) | 764.15 | 16667.01 | 29196.84 | 55749.62 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 1.31 | 3.73 | 3.74 | 3.70 |
| Output Token Throughput (Tok/s) | 130.63 | 372.40 | 373.98 | 369.87 |

#### Input 50 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 374.93 | 23526.65 | 47550.99 | 101553.31 |
| Request Latency (ms) | 1041.29 | 33712.54 | 57641.53 | 111895.20 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 0.96 | 1.79 | 1.79 | 1.78 |
| Output Token Throughput (Tok/s) | 95.74 | 178.73 | 178.89 | 178.15 |

### RTX PRO 6000 Blackwell Server Edition

#### Input 50 / Output 1 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 141.99 | 3213.91 | 5384.51 | 10792.72 |
| Request Latency (ms) | 141.99 | 3213.91 | 5384.51 | 10792.72 |
| Request Count (requests) | 50 | 320 | 254 | 512 |
| Request Throughput (Req/s) | 6.96 | 18.00 | 17.95 | 17.89 |
| Output Token Throughput (Tok/s) | 6.96 | 18.00 | 17.95 | 17.89 |

#### Input 50 / Output 1 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 239.86 | 7483.22 | 12552.69 | 25259.11 |
| Request Latency (ms) | 239.86 | 7483.22 | 12552.69 | 25259.11 |
| Request Count (requests) | 49 | 303 | 249 | 491 |
| Request Throughput (Req/s) | 4.06 | 7.28 | 7.49 | 7.34 |
| Output Token Throughput (Tok/s) | 4.06 | 7.28 | 7.49 | 7.34 |

#### Input 50 / Output 100 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 138.74 | 943.46 | 2680.17 | 11599.63 |
| Request Latency (ms) | 503.44 | 6188.90 | 13022.07 | 26388.89 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 1.98 | 10.27 | 9.57 | 8.95 |
| Output Token Throughput (Tok/s) | 197.75 | 1026.14 | 956.47 | 893.91 |

#### Input 50 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 239.24 | 1798.96 | 11644.84 | 33293.32 |
| Request Latency (ms) | 638.71 | 13599.89 | 26299.90 | 49165.91 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 1.56 | 4.66 | 4.50 | 4.45 |
| Output Token Throughput (Tok/s) | 155.93 | 465.28 | 449.57 | 444.17 |

### Embedded-Platform Eager Transformers

These preliminary measurements use raw Hugging Face Transformers in eager mode rather than vLLM. They are presented separately because their runtime, workloads, and metric definitions differ from the vLLM serving benchmarks above.

| Board | Specification | Input | Prompt Tokens | Prefill Throughput | Prefill Latency | Decode Throughput | End-to-End Latency |
|---|---|---|---:|---:|---:|---:|---:|
| Jetson AGX Thor T5000 | 128 GB / MAXN | Text | 1705 | 8717 Tok/s | 0.20 s | 37.3 Tok/s | 3.60 s |
| Jetson AGX Thor T5000 | 128 GB / MAXN | Image | 911 | 4845 Tok/s | 0.19 s | 42.6 Tok/s | 3.17 s |
| Jetson AGX Thor T5000 | 128 GB / MAXN | Video | 1263 | 6032 Tok/s | 0.21 s | 41.8 Tok/s | 3.25 s |
| Jetson AGX Thor T4000 | 64 GB / MAXN, 1530 MHz | Text | 1705 | 6519 Tok/s | 0.26 s | 34.1 Tok/s | 3.99 s |
| Jetson AGX Thor T4000 | 64 GB / MAXN, 1530 MHz | Image | 911 | 3471 Tok/s | 0.26 s | 40.3 Tok/s | 3.41 s |
| Jetson AGX Thor T4000 | 64 GB / MAXN, 1530 MHz | Video | 1263 | 4164 Tok/s | 0.30 s | 38.1 Tok/s | 3.64 s |
| Jetson Thor T3000 | 32 GB / 1100 MHz | Text | 1705 | 5230 Tok/s | 0.33 s | 29.7 Tok/s | 4.61 s |
| Jetson Thor T3000 | 32 GB / 1100 MHz | Image | 911 | 2710 Tok/s | 0.34 s | 36.3 Tok/s | 3.83 s |
| Jetson Thor T3000 | 32 GB / 1100 MHz | Video | 1263 | 3388 Tok/s | 0.37 s | 33.7 Tok/s | 4.14 s |
| Jetson Thor T2000 | 16 GB / 702 MHz, THOR_NANO | Text | 1705 | 2355 Tok/s | 0.72 s | 15.7 Tok/s | 8.80 s |
| Jetson Thor T2000 | 16 GB / 702 MHz, THOR_NANO | Image | 911 | 1233 Tok/s | 0.74 s | 19.6 Tok/s | 7.21 s |
| Jetson Thor T2000 | 16 GB / 702 MHz, THOR_NANO | Video | 1263 | 1543 Tok/s | 0.82 s | 18.0 Tok/s | 7.87 s |
| Jetson AGX Orin | 64 GB | Text | 1705 | 3260 Tok/s | 0.52 s | 12.3 Tok/s | 10.83 s |
| Jetson AGX Orin | 64 GB | Image | 911 | 1840 Tok/s | 0.50 s | 12.3 Tok/s | 10.81 s |
| Jetson AGX Orin | 64 GB | Video | 1263 | 2103 Tok/s | 0.60 s | 12.2 Tok/s | 10.97 s |

<sub>Notes:
1. Source: vLLM inference benchmarking for `nvidia/Cosmos3-Edge`; metrics were collected with one GPU at client-side concurrency levels of 1, 64, 128, and 256.
2. **Time To First Token (TTFT)** measures latency until the first output token is emitted. **Request Latency** is end-to-end time per request. For single-token outputs (Output 1), TTFT and request latency are identical.
3. **Request Throughput** is completed requests per second. **Output Token Throughput** is generated tokens per second. For Output 1 workloads, the two throughput values match.
4. Concurrency is the number of simultaneous client requests, not tensor-parallel GPU count.
5. Embedded-platform measurements use Hugging Face Transformers in eager mode and should not be compared directly with the vLLM serving results.</sub>

## Cosmos3-Nano Reasoner

These tables compare **Cosmos3-Nano** reasoner serving performance through **vLLM** and **NIM**, grouped by GPU and workload. Lower is better for latency; higher is better for throughput. Precision is shown for NIM; `—` indicates that the vLLM baseline did not specify precision.

NIM runs use one GPU, tensor parallelism 1, and synthetic 374×374, 30-second videos. Input and output lengths are token targets. Concurrency is simultaneous client requests, not GPU count. Blank cells indicate missing results; backend coverage differs.

### RTX PRO 6000 Blackwell

#### Input 50 / Output 1 / Video 1 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 187.59 | 187.59 | 5.29 | 5.29 | 50 |
| NIM | FP8 | 1 | 307.44 | 307.44 | 3.21 | 3.21 | 30 |
| NIM | NVFP4 | 1 | 283.55 | 283.55 | 3.50 | 3.50 | 30 |
| NIM | FP8 | 8 | 1276.21 | 1276.21 | 5.88 | 5.88 | 50 |
| NIM | NVFP4 | 8 | 1200.00 | 1200.00 | 6.31 | 6.31 | 50 |
| NIM | FP8 | 16 | 2487.85 | 2487.85 | 6.00 | 6.00 | 80 |
| NIM | NVFP4 | 16 | 2393.43 | 2393.43 | 6.27 | 6.27 | 80 |
| NIM | FP8 | 32 | 4706.60 | 4706.60 | 6.22 | 6.22 | 120 |
| NIM | NVFP4 | 32 | 4485.85 | 4485.85 | 6.60 | 6.60 | 120 |
| vLLM | — | 64 | 5826.84 | 5826.84 | 9.95 | 9.95 | 320 |
| NIM | FP8 | 64 | 9584.58 | 9584.58 | 5.83 | 5.83 | 160 |
| NIM | NVFP4 | 64 | 8834.77 | 8834.77 | 6.40 | 6.40 | 160 |
| vLLM | — | 128 | 9742.43 | 9742.43 | 9.97 | 9.97 | 256 |
| vLLM | — | 256 | 19541.84 | 19541.84 | 9.89 | 9.89 | 512 |

#### Input 50 / Output 1 / Video 2 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 316.90 | 316.90 | 3.14 | 3.14 | 50 |
| NIM | FP8 | 1 | 567.42 | 567.42 | 1.75 | 1.75 | 30 |
| NIM | NVFP4 | 1 | 528.03 | 528.03 | 1.88 | 1.88 | 30 |
| NIM | FP8 | 8 | 2133.97 | 2133.97 | 3.52 | 3.52 | 50 |
| NIM | NVFP4 | 8 | 1934.05 | 1934.05 | 3.89 | 3.89 | 50 |
| NIM | FP8 | 16 | 4137.62 | 4137.62 | 3.58 | 3.58 | 80 |
| NIM | NVFP4 | 16 | 3751.98 | 3751.98 | 3.94 | 3.94 | 80 |
| NIM | FP8 | 32 | 8064.26 | 8064.26 | 3.53 | 3.53 | 120 |
| NIM | NVFP4 | 32 | 7113.45 | 7113.45 | 4.04 | 4.04 | 120 |
| vLLM | — | 64 | 12223.00 | 12223.00 | 4.73 | 4.73 | 320 |
| NIM | FP8 | 64 | 15172.65 | 15172.65 | 3.51 | 3.51 | 160 |
| NIM | NVFP4 | 64 | 13780.29 | 13780.29 | 3.91 | 3.91 | 160 |
| vLLM | — | 128 | 20364.04 | 20364.04 | 4.75 | 4.75 | 256 |
| vLLM | — | 256 | 40929.42 | 40929.42 | 4.71 | 4.71 | 512 |

#### Input 50 / Output 1 / Video 4 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 1012.17 | 1012.17 | 0.98 | 0.98 | 30 |
| NIM | NVFP4 | 1 | 983.13 | 983.13 | 1.01 | 1.01 | 30 |
| NIM | FP8 | 8 | 4126.51 | 4126.51 | 1.81 | 1.81 | 50 |
| NIM | NVFP4 | 8 | 3584.01 | 3584.01 | 2.09 | 2.09 | 50 |
| NIM | FP8 | 16 | 8022.78 | 8022.78 | 1.82 | 1.82 | 80 |
| NIM | NVFP4 | 16 | 7022.27 | 7022.27 | 2.09 | 2.09 | 80 |
| NIM | FP8 | 32 | 15405.08 | 15405.08 | 1.83 | 1.83 | 120 |
| NIM | NVFP4 | 32 | 13579.22 | 13579.22 | 2.08 | 2.08 | 120 |
| NIM | FP8 | 64 | 29205.36 | 29205.36 | 1.80 | 1.80 | 160 |
| NIM | NVFP4 | 64 | 25713.53 | 25713.53 | 2.05 | 2.05 | 160 |

#### Input 50 / Output 1 / Video 8 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 1414.83 | 1414.83 | 0.70 | 0.70 | 30 |
| NIM | NVFP4 | 1 | 1288.33 | 1288.33 | 0.77 | 0.77 | 30 |
| NIM | FP8 | 8 | 5683.67 | 5683.67 | 1.32 | 1.32 | 50 |
| NIM | NVFP4 | 8 | 5023.21 | 5023.21 | 1.49 | 1.49 | 50 |
| NIM | FP8 | 16 | 11129.10 | 11129.10 | 1.31 | 1.31 | 80 |
| NIM | NVFP4 | 16 | 9753.71 | 9753.71 | 1.50 | 1.50 | 80 |
| NIM | FP8 | 32 | 21336.06 | 21336.06 | 1.32 | 1.32 | 120 |
| NIM | NVFP4 | 32 | 18807.17 | 18807.17 | 1.50 | 1.50 | 120 |
| NIM | FP8 | 64 | 39606.00 | 39606.00 | 1.32 | 1.32 | 160 |
| NIM | NVFP4 | 64 | 35094.96 | 35094.96 | 1.50 | 1.50 | 160 |

#### Input 50 / Output 100 / Video 1 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 186.46 | 1402.12 | 0.71 | 71.22 | 50 |
| NIM | FP8 | 1 | 324.95 | 995.94 | 1.00 | 99.86 | 30 |
| NIM | NVFP4 | 1 | 285.67 | 771.52 | 1.29 | 129.17 | 30 |
| NIM | FP8 | 8 | 1255.20 | 2291.58 | 3.33 | 332.22 | 50 |
| NIM | NVFP4 | 8 | 1051.23 | 1936.48 | 3.90 | 389.52 | 50 |
| NIM | FP8 | 16 | 2333.64 | 3871.55 | 3.90 | 388.93 | 80 |
| NIM | NVFP4 | 16 | 1965.52 | 3385.58 | 4.62 | 459.94 | 80 |
| NIM | FP8 | 32 | 4148.85 | 6734.79 | 4.39 | 437.37 | 120 |
| NIM | NVFP4 | 32 | 3868.21 | 5985.22 | 4.86 | 484.85 | 120 |
| vLLM | — | 64 | 2280.43 | 9309.93 | 6.85 | 684.76 | 320 |
| NIM | FP8 | 64 | 8200.08 | 12699.23 | 4.53 | 451.86 | 160 |
| NIM | NVFP4 | 64 | 7549.54 | 11630.59 | 5.01 | 498.75 | 160 |
| vLLM | — | 128 | 4627.08 | 18541.90 | 6.82 | 682.18 | 256 |
| vLLM | — | 256 | 14419.32 | 39202.74 | 6.22 | 622.49 | 512 |

#### Input 50 / Output 100 / Video 2 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 315.77 | 1553.53 | 0.64 | 64.28 | 50 |
| NIM | FP8 | 1 | 577.76 | 1369.39 | 0.73 | 72.55 | 30 |
| NIM | NVFP4 | 1 | 521.69 | 1141.17 | 0.87 | 87.30 | 30 |
| NIM | FP8 | 8 | 1915.69 | 3588.67 | 2.19 | 218.86 | 50 |
| NIM | NVFP4 | 8 | 1672.00 | 3046.47 | 2.48 | 247.15 | 50 |
| NIM | FP8 | 16 | 3455.81 | 6171.33 | 2.46 | 245.22 | 80 |
| NIM | NVFP4 | 16 | 3017.20 | 5380.10 | 2.94 | 293.48 | 80 |
| NIM | FP8 | 32 | 7385.83 | 12008.77 | 2.60 | 259.21 | 120 |
| NIM | NVFP4 | 32 | 7414.49 | 11039.29 | 2.76 | 275.05 | 120 |
| vLLM | — | 64 | 3248.72 | 18532.34 | 3.44 | 343.79 | 320 |
| NIM | FP8 | 64 | 14931.71 | 22156.15 | 2.62 | 261.53 | 160 |
| NIM | NVFP4 | 64 | 14104.13 | 20759.52 | 2.71 | 270.15 | 160 |
| vLLM | — | 128 | 13795.45 | 37994.05 | 3.22 | 322.21 | 256 |
| vLLM | — | 256 | 44476.55 | 71534.87 | 3.15 | 314.62 | 512 |

#### Input 50 / Output 100 / Video 4 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 1017.01 | 2053.09 | 0.49 | 48.48 | 30 |
| NIM | NVFP4 | 1 | 931.56 | 1910.15 | 0.52 | 52.12 | 30 |
| NIM | FP8 | 8 | 3159.38 | 6157.46 | 1.26 | 125.20 | 50 |
| NIM | NVFP4 | 8 | 2748.20 | 5486.38 | 1.39 | 137.85 | 50 |
| NIM | FP8 | 16 | 4707.44 | 10642.06 | 1.48 | 147.34 | 80 |
| NIM | NVFP4 | 16 | 4709.12 | 9922.28 | 1.57 | 156.08 | 80 |
| NIM | FP8 | 32 | 5782.71 | 19356.28 | 1.58 | 156.98 | 120 |
| NIM | NVFP4 | 32 | 6386.52 | 18137.28 | 1.71 | 169.98 | 120 |
| NIM | FP8 | 64 | 23785.08 | 37583.40 | 1.58 | 157.25 | 160 |
| NIM | NVFP4 | 64 | 21795.28 | 34310.41 | 1.72 | 170.89 | 160 |

#### Input 50 / Output 100 / Video 8 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 1398.72 | 2521.49 | 0.40 | 39.43 | 30 |
| NIM | NVFP4 | 1 | 1292.16 | 2500.06 | 0.40 | 39.77 | 30 |
| NIM | FP8 | 8 | 4310.53 | 8138.37 | 0.93 | 92.56 | 50 |
| NIM | NVFP4 | 8 | 3757.36 | 7448.78 | 1.01 | 100.48 | 50 |
| NIM | FP8 | 16 | 7390.11 | 14736.78 | 1.07 | 105.89 | 80 |
| NIM | NVFP4 | 16 | 7077.35 | 13551.34 | 1.12 | 111.15 | 80 |
| NIM | FP8 | 32 | 11527.85 | 27186.92 | 1.13 | 112.43 | 120 |
| NIM | NVFP4 | 32 | 8929.87 | 25904.76 | 1.21 | 120.53 | 120 |
| NIM | FP8 | 64 | 30079.29 | 52017.44 | 1.15 | 114.26 | 160 |
| NIM | NVFP4 | 64 | 27611.81 | 48537.99 | 1.22 | 121.32 | 160 |

### H20

#### Input 50 / Output 1 / Video 1 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 358.62 | 358.62 | 2.77 | 2.77 | 50 |
| NIM | FP8 | 1 | 547.35 | 547.35 | 1.82 | 1.82 | 30 |
| NIM | FP8 | 8 | 2317.17 | 2317.17 | 3.25 | 3.25 | 50 |
| NIM | FP8 | 16 | 4516.25 | 4516.25 | 3.38 | 3.38 | 80 |
| NIM | FP8 | 32 | 8770.01 | 8770.01 | 3.32 | 3.32 | 120 |
| vLLM | — | 64 | 14953.90 | 14953.90 | 3.88 | 3.88 | 320 |
| NIM | FP8 | 64 | 17051.33 | 17051.33 | 3.21 | 3.21 | 160 |
| vLLM | — | 128 | 25086.30 | 25086.30 | 3.87 | 3.87 | 256 |
| vLLM | — | 256 | 49549.94 | 49549.94 | 3.90 | 3.90 | 512 |

#### Input 50 / Output 1 / Video 2 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 648.48 | 648.48 | 1.54 | 1.54 | 50 |
| NIM | FP8 | 1 | 949.20 | 949.20 | 1.05 | 1.05 | 30 |
| NIM | FP8 | 8 | 4133.15 | 4133.15 | 1.83 | 1.83 | 50 |
| NIM | FP8 | 16 | 7891.95 | 7891.95 | 1.86 | 1.86 | 80 |
| NIM | FP8 | 32 | 15621.24 | 15621.24 | 1.82 | 1.82 | 120 |
| vLLM | — | 64 | 30604.91 | 30604.91 | 1.89 | 1.89 | 320 |
| NIM | FP8 | 64 | 29841.78 | 29841.78 | 1.79 | 1.79 | 160 |
| vLLM | — | 128 | 51364.19 | 51364.19 | 1.89 | 1.89 | 256 |
| vLLM | — | 256 | 101597.85 | 101597.85 | 1.90 | 1.90 | 512 |

#### Input 50 / Output 1 / Video 4 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 1694.18 | 1694.18 | 0.59 | 0.59 | 30 |
| NIM | FP8 | 8 | 7799.71 | 7799.71 | 0.96 | 0.96 | 50 |
| NIM | FP8 | 16 | 15224.42 | 15224.42 | 0.96 | 0.96 | 80 |
| NIM | FP8 | 32 | 30379.42 | 30379.42 | 0.93 | 0.93 | 120 |
| NIM | FP8 | 64 | 54671.46 | 54671.46 | 0.96 | 0.96 | 160 |

#### Input 50 / Output 1 / Video 8 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 2431.77 | 2431.77 | 0.41 | 0.41 | 30 |
| NIM | FP8 | 8 | 10937.37 | 10937.37 | 0.68 | 0.68 | 50 |
| NIM | FP8 | 16 | 21360.22 | 21360.22 | 0.68 | 0.68 | 80 |
| NIM | FP8 | 32 | 41043.00 | 41043.00 | 0.69 | 0.69 | 120 |
| NIM | FP8 | 64 | 76860.36 | 76860.36 | 0.68 | 0.68 | 160 |

#### Input 50 / Output 100 / Video 1 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 360.40 | 1026.97 | 0.97 | 97.14 | 50 |
| NIM | FP8 | 1 | 547.20 | 797.55 | 1.25 | 124.80 | 30 |
| NIM | FP8 | 8 | 2294.63 | 3204.08 | 2.39 | 238.45 | 50 |
| NIM | FP8 | 16 | 4020.67 | 6004.87 | 2.57 | 256.62 | 80 |
| NIM | FP8 | 32 | 7677.54 | 12053.30 | 2.48 | 247.68 | 120 |
| vLLM | — | 64 | 6607.80 | 18990.48 | 3.37 | 336.55 | 320 |
| NIM | FP8 | 64 | 14200.91 | 23084.83 | 2.58 | 257.29 | 160 |
| vLLM | — | 128 | 10973.05 | 37514.21 | 3.40 | 339.57 | 256 |
| vLLM | — | 256 | 29404.60 | 74287.25 | 3.33 | 332.93 | 512 |

#### Input 50 / Output 100 / Video 2 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 646.27 | 1331.00 | 0.75 | 75.00 | 50 |
| NIM | FP8 | 1 | 965.35 | 1245.98 | 0.80 | 79.85 | 30 |
| NIM | FP8 | 8 | 3576.01 | 5458.78 | 1.40 | 139.66 | 50 |
| NIM | FP8 | 16 | 6492.91 | 10751.62 | 1.48 | 147.29 | 80 |
| NIM | FP8 | 32 | 11598.74 | 19692.49 | 1.51 | 150.59 | 120 |
| vLLM | — | 64 | 8329.82 | 37577.68 | 1.70 | 170.03 | 320 |
| NIM | FP8 | 64 | 21626.10 | 40779.74 | 1.55 | 154.81 | 160 |
| vLLM | — | 128 | 29036.81 | 74291.62 | 1.67 | 167.36 | 256 |
| vLLM | — | 256 | 86145.80 | 136416.07 | 1.67 | 167.12 | 512 |

#### Input 50 / Output 100 / Video 4 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 1684.35 | 2065.71 | 0.48 | 48.14 | 30 |
| NIM | FP8 | 8 | 5479.71 | 9443.04 | 0.81 | 81.08 | 50 |
| NIM | FP8 | 16 | 10459.59 | 19051.24 | 0.84 | 83.41 | 80 |
| NIM | FP8 | 32 | 12873.93 | 35679.80 | 0.89 | 88.50 | 120 |
| NIM | FP8 | 64 | 42199.22 | 67363.86 | 0.89 | 88.38 | 160 |

#### Input 50 / Output 100 / Video 8 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 2368.53 | 2819.99 | 0.35 | 35.26 | 30 |
| NIM | FP8 | 8 | 7529.95 | 12964.19 | 0.59 | 58.99 | 50 |
| NIM | FP8 | 16 | 13940.32 | 25560.59 | 0.62 | 62.07 | 80 |
| NIM | FP8 | 32 | 22735.99 | 49942.89 | 0.63 | 63.04 | 120 |
| NIM | FP8 | 64 | 53281.25 | 94533.66 | 0.64 | 63.66 | 160 |

### H100 NVL

#### Input 50 / Output 1 / Video 1 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 170.69 | 170.69 | 5.81 | 5.81 | 50 |
| NIM | FP8 | 1 | 289.89 | 289.89 | 3.42 | 3.42 | 30 |
| NIM | FP8 | 8 | 1264.31 | 1264.31 | 5.98 | 5.98 | 50 |
| NIM | FP8 | 16 | 2499.14 | 2499.14 | 6.07 | 6.07 | 80 |
| NIM | FP8 | 32 | 4948.89 | 4948.89 | 6.00 | 6.00 | 120 |
| vLLM | — | 64 | 6527.13 | 6527.13 | 8.88 | 8.88 | 320 |
| NIM | FP8 | 64 | 9819.37 | 9819.37 | 5.83 | 5.83 | 160 |
| vLLM | — | 128 | 10726.80 | 10726.80 | 9.05 | 9.05 | 256 |
| vLLM | — | 256 | 21881.52 | 21881.52 | 8.83 | 8.83 | 512 |

#### Input 50 / Output 1 / Video 2 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 303.65 | 303.65 | 3.27 | 3.27 | 50 |
| NIM | FP8 | 1 | 507.70 | 507.70 | 1.96 | 1.96 | 30 |
| NIM | FP8 | 8 | 1937.43 | 1937.43 | 3.88 | 3.88 | 50 |
| NIM | FP8 | 16 | 3842.05 | 3842.05 | 3.86 | 3.86 | 80 |
| NIM | FP8 | 32 | 7624.59 | 7624.59 | 3.78 | 3.78 | 120 |
| vLLM | — | 64 | 13480.29 | 13480.29 | 4.29 | 4.29 | 320 |
| NIM | FP8 | 64 | 15057.61 | 15057.61 | 3.63 | 3.63 | 160 |
| vLLM | — | 128 | 22431.93 | 22431.93 | 4.31 | 4.31 | 256 |
| vLLM | — | 256 | 44352.53 | 44352.53 | 4.35 | 4.35 | 512 |

#### Input 50 / Output 1 / Video 4 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 932.11 | 932.11 | 1.07 | 1.07 | 30 |
| NIM | FP8 | 8 | 3496.80 | 3496.80 | 2.15 | 2.15 | 50 |
| NIM | FP8 | 16 | 6820.93 | 6820.93 | 2.15 | 2.15 | 80 |
| NIM | FP8 | 32 | 13388.08 | 13388.08 | 2.13 | 2.13 | 120 |
| NIM | FP8 | 64 | 25298.55 | 25298.55 | 2.11 | 2.11 | 160 |

#### Input 50 / Output 1 / Video 8 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 1382.23 | 1382.23 | 0.72 | 0.72 | 30 |
| NIM | FP8 | 8 | 5140.30 | 5140.30 | 1.46 | 1.46 | 50 |
| NIM | FP8 | 16 | 10048.76 | 10048.76 | 1.46 | 1.46 | 80 |
| NIM | FP8 | 32 | 19606.71 | 19606.71 | 1.45 | 1.45 | 120 |
| NIM | FP8 | 64 | 36509.47 | 36509.47 | 1.46 | 1.46 | 160 |

#### Input 50 / Output 100 / Video 1 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 172.66 | 867.35 | 1.15 | 115.03 | 50 |
| NIM | FP8 | 1 | 296.01 | 547.60 | 1.82 | 181.22 | 30 |
| NIM | FP8 | 8 | 1247.01 | 1828.94 | 4.18 | 417.79 | 50 |
| NIM | FP8 | 16 | 2411.04 | 3401.43 | 4.64 | 463.58 | 80 |
| NIM | FP8 | 32 | 4340.34 | 6428.16 | 4.72 | 470.89 | 120 |
| vLLM | — | 64 | 2890.81 | 9192.19 | 6.94 | 694.37 | 320 |
| NIM | FP8 | 64 | 8103.38 | 12063.48 | 4.78 | 477.62 | 160 |
| vLLM | — | 128 | 5022.58 | 18061.43 | 7.02 | 702.48 | 256 |
| vLLM | — | 256 | 13929.58 | 35151.09 | 6.95 | 695.12 | 512 |

#### Input 50 / Output 100 / Video 2 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 296.99 | 1009.41 | 0.99 | 98.87 | 50 |
| NIM | FP8 | 1 | 528.32 | 821.59 | 1.21 | 120.78 | 30 |
| NIM | FP8 | 8 | 1797.42 | 2786.26 | 2.75 | 273.60 | 50 |
| NIM | FP8 | 16 | 3355.38 | 5262.55 | 3.03 | 302.16 | 80 |
| NIM | FP8 | 32 | 6198.83 | 9746.74 | 3.03 | 301.84 | 120 |
| vLLM | — | 64 | 3572.31 | 18030.41 | 3.54 | 353.81 | 320 |
| NIM | FP8 | 64 | 11088.70 | 18497.46 | 3.13 | 311.66 | 160 |
| vLLM | — | 128 | 13808.03 | 35239.25 | 3.48 | 348.37 | 256 |
| vLLM | — | 256 | 41101.98 | 64485.08 | 3.50 | 350.07 | 512 |

#### Input 50 / Output 100 / Video 4 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 949.76 | 1341.06 | 0.74 | 74.10 | 30 |
| NIM | FP8 | 8 | 2800.73 | 4657.69 | 1.64 | 162.79 | 50 |
| NIM | FP8 | 16 | 5099.70 | 8796.76 | 1.81 | 180.27 | 80 |
| NIM | FP8 | 32 | 10181.08 | 17457.11 | 1.82 | 180.99 | 120 |
| NIM | FP8 | 64 | 20160.56 | 31434.75 | 1.90 | 189.48 | 160 |

#### Input 50 / Output 100 / Video 8 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 1392.53 | 1835.51 | 0.54 | 54.22 | 30 |
| NIM | FP8 | 8 | 3788.95 | 6374.80 | 1.20 | 119.52 | 50 |
| NIM | FP8 | 16 | 6778.80 | 11993.94 | 1.31 | 130.93 | 80 |
| NIM | FP8 | 32 | 17383.46 | 24728.71 | 1.27 | 126.87 | 120 |
| NIM | FP8 | 64 | 39363.81 | 47399.44 | 1.25 | 124.77 | 160 |

### H200 NVL

#### Input 50 / Output 1 / Video 1 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 142.79 | 142.79 | 6.92 | 6.92 | 50 |
| NIM | FP8 | 1 | 279.46 | 279.46 | 3.53 | 3.53 | 30 |
| NIM | FP8 | 8 | 1206.11 | 1206.11 | 6.36 | 6.36 | 50 |
| NIM | FP8 | 16 | 2328.11 | 2328.11 | 6.49 | 6.49 | 80 |
| NIM | FP8 | 32 | 4704.00 | 4704.00 | 6.41 | 6.41 | 120 |
| vLLM | — | 64 | 3614.37 | 3614.37 | 16.04 | 16.04 | 320 |
| NIM | FP8 | 64 | 9571.12 | 9571.12 | 6.04 | 6.04 | 160 |
| vLLM | — | 128 | 6050.58 | 6050.58 | 16.08 | 16.08 | 256 |
| vLLM | — | 256 | 12094.34 | 12094.34 | 15.96 | 15.96 | 512 |

#### Input 50 / Output 1 / Video 2 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 228.84 | 228.84 | 4.34 | 4.34 | 50 |
| NIM | FP8 | 1 | 469.04 | 469.04 | 2.12 | 2.12 | 30 |
| NIM | FP8 | 8 | 1842.86 | 1842.86 | 4.11 | 4.11 | 50 |
| NIM | FP8 | 16 | 3612.08 | 3612.08 | 4.13 | 4.13 | 80 |
| NIM | FP8 | 32 | 6991.58 | 6991.58 | 4.18 | 4.18 | 120 |
| vLLM | — | 64 | 7569.46 | 7569.46 | 7.62 | 7.62 | 320 |
| NIM | FP8 | 64 | 14536.40 | 14536.40 | 3.82 | 3.82 | 160 |
| vLLM | — | 128 | 12515.91 | 12515.91 | 7.71 | 7.71 | 256 |
| vLLM | — | 256 | 25646.99 | 25646.99 | 7.48 | 7.48 | 512 |

#### Input 50 / Output 1 / Video 4 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 864.04 | 864.04 | 1.15 | 1.15 | 30 |
| NIM | FP8 | 8 | 3211.82 | 3211.82 | 2.34 | 2.34 | 50 |
| NIM | FP8 | 16 | 6132.30 | 6132.30 | 2.38 | 2.38 | 80 |
| NIM | FP8 | 32 | 12101.69 | 12101.69 | 2.35 | 2.35 | 120 |
| NIM | FP8 | 64 | 21991.72 | 21991.72 | 2.44 | 2.44 | 160 |

#### Input 50 / Output 1 / Video 8 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 1240.07 | 1240.07 | 0.80 | 0.80 | 30 |
| NIM | FP8 | 8 | 4449.08 | 4449.08 | 1.68 | 1.68 | 50 |
| NIM | FP8 | 16 | 8897.99 | 8897.99 | 1.64 | 1.64 | 80 |
| NIM | FP8 | 32 | 16461.06 | 16461.06 | 1.72 | 1.72 | 120 |
| NIM | FP8 | 64 | 32000.01 | 32000.01 | 1.64 | 1.64 | 160 |

#### Input 50 / Output 100 / Video 1 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 142.23 | 770.15 | 1.30 | 129.53 | 50 |
| NIM | FP8 | 1 | 275.58 | 543.00 | 1.83 | 182.66 | 30 |
| NIM | FP8 | 8 | 1203.04 | 1742.24 | 4.29 | 428.19 | 50 |
| NIM | FP8 | 16 | 2243.61 | 3174.24 | 5.00 | 499.35 | 80 |
| NIM | FP8 | 32 | 4113.44 | 5727.84 | 5.26 | 525.10 | 120 |
| vLLM | — | 64 | 1948.06 | 5284.58 | 12.07 | 1206.86 | 320 |
| NIM | FP8 | 64 | 7966.62 | 10760.62 | 5.33 | 532.51 | 160 |
| vLLM | — | 128 | 3180.20 | 10054.55 | 12.60 | 1259.60 | 256 |
| vLLM | — | 256 | 5271.37 | 19831.69 | 12.71 | 1270.44 | 512 |

#### Input 50 / Output 100 / Video 2 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 227.40 | 862.92 | 1.16 | 115.63 | 50 |
| NIM | FP8 | 1 | 478.85 | 755.70 | 1.32 | 131.88 | 30 |
| NIM | FP8 | 8 | 1596.25 | 2529.57 | 3.01 | 299.91 | 50 |
| NIM | FP8 | 16 | 3038.27 | 4687.47 | 3.38 | 337.55 | 80 |
| NIM | FP8 | 32 | 5479.86 | 8440.24 | 3.53 | 352.12 | 120 |
| vLLM | — | 64 | 2718.13 | 10249.14 | 6.22 | 621.97 | 320 |
| NIM | FP8 | 64 | 13047.04 | 16817.90 | 3.53 | 352.68 | 160 |
| vLLM | — | 128 | 5522.05 | 19775.33 | 6.38 | 638.33 | 256 |
| vLLM | — | 256 | 17729.47 | 39089.75 | 6.18 | 618.21 | 512 |

#### Input 50 / Output 100 / Video 4 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 866.45 | 1242.97 | 0.80 | 79.97 | 30 |
| NIM | FP8 | 8 | 2579.74 | 4107.43 | 1.87 | 186.23 | 50 |
| NIM | FP8 | 16 | 4831.05 | 7714.63 | 2.05 | 204.14 | 80 |
| NIM | FP8 | 32 | 9851.76 | 14042.61 | 2.14 | 212.80 | 120 |
| NIM | FP8 | 64 | 24686.49 | 27818.11 | 2.06 | 205.13 | 160 |

#### Input 50 / Output 100 / Video 8 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 1245.06 | 1659.66 | 0.60 | 60.15 | 30 |
| NIM | FP8 | 8 | 3445.88 | 5487.71 | 1.38 | 137.84 | 50 |
| NIM | FP8 | 16 | 6246.50 | 10109.67 | 1.50 | 148.54 | 80 |
| NIM | FP8 | 32 | 15706.39 | 20668.61 | 1.49 | 148.12 | 120 |
| NIM | FP8 | 64 | 34906.40 | 41049.14 | 1.45 | 144.61 | 160 |

### H100 80GB HBM3 (SXM)

#### Input 50 / Output 1 / Video 1 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 145.52 | 145.52 | 6.78 | 6.78 | 50 |
| NIM | FP8 | 1 | 333.71 | 333.71 | 2.96 | 2.96 | 30 |
| NIM | FP8 | 8 | 1388.65 | 1388.65 | 5.54 | 5.54 | 50 |
| NIM | FP8 | 16 | 2708.71 | 2708.71 | 5.62 | 5.62 | 80 |
| NIM | FP8 | 32 | 5375.97 | 5375.97 | 5.59 | 5.59 | 120 |
| vLLM | — | 64 | 3332.72 | 3332.72 | 17.41 | 17.41 | 320 |
| NIM | FP8 | 64 | 10895.36 | 10895.36 | 5.35 | 5.35 | 160 |
| vLLM | — | 128 | 5608.41 | 5608.41 | 17.36 | 17.36 | 256 |
| vLLM | — | 256 | 11133.76 | 11133.76 | 17.38 | 17.38 | 512 |

#### Input 50 / Output 1 / Video 2 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 228.80 | 228.80 | 4.34 | 4.34 | 50 |
| NIM | FP8 | 1 | 531.35 | 531.35 | 1.87 | 1.87 | 30 |
| NIM | FP8 | 8 | 2005.38 | 2005.38 | 3.78 | 3.78 | 50 |
| NIM | FP8 | 16 | 3969.20 | 3969.20 | 3.78 | 3.78 | 80 |
| NIM | FP8 | 32 | 8251.93 | 8251.93 | 3.51 | 3.51 | 120 |
| vLLM | — | 64 | 6876.25 | 6876.25 | 8.42 | 8.42 | 320 |
| NIM | FP8 | 64 | 15941.29 | 15941.29 | 3.52 | 3.52 | 160 |
| vLLM | — | 128 | 11556.33 | 11556.33 | 8.38 | 8.38 | 256 |
| vLLM | — | 256 | 22836.32 | 22836.32 | 8.46 | 8.46 | 512 |

#### Input 50 / Output 1 / Video 4 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 976.19 | 976.19 | 1.01 | 1.01 | 30 |
| NIM | FP8 | 8 | 3362.95 | 3362.95 | 2.24 | 2.24 | 50 |
| NIM | FP8 | 16 | 6741.02 | 6741.02 | 2.19 | 2.19 | 80 |
| NIM | FP8 | 32 | 13483.12 | 13483.12 | 2.10 | 2.10 | 120 |
| NIM | FP8 | 64 | 25283.00 | 25283.00 | 2.09 | 2.09 | 160 |

#### Input 50 / Output 1 / Video 8 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 1307.03 | 1307.03 | 0.76 | 0.76 | 30 |
| NIM | FP8 | 8 | 4783.64 | 4783.64 | 1.55 | 1.55 | 50 |
| NIM | FP8 | 16 | 9481.04 | 9481.04 | 1.54 | 1.54 | 80 |
| NIM | FP8 | 32 | 17615.76 | 17615.76 | 1.60 | 1.60 | 120 |
| NIM | FP8 | 64 | 33515.53 | 33515.53 | 1.57 | 1.57 | 160 |

#### Input 50 / Output 100 / Video 1 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 143.36 | 865.56 | 1.15 | 115.24 | 50 |
| NIM | FP8 | 1 | 323.06 | 610.24 | 1.63 | 163.18 | 30 |
| NIM | FP8 | 8 | 1343.33 | 1943.65 | 3.97 | 396.62 | 50 |
| NIM | FP8 | 16 | 2702.08 | 3643.80 | 4.33 | 433.00 | 80 |
| NIM | FP8 | 32 | 4824.22 | 6364.58 | 4.72 | 471.19 | 120 |
| vLLM | — | 64 | 1720.73 | 5251.56 | 12.14 | 1213.61 | 320 |
| NIM | FP8 | 64 | 9569.93 | 12347.01 | 4.92 | 490.85 | 160 |
| vLLM | — | 128 | 2906.00 | 9818.73 | 12.87 | 1286.89 | 256 |
| vLLM | — | 256 | 9000.63 | 18353.75 | 12.83 | 1282.48 | 512 |

#### Input 50 / Output 100 / Video 2 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 231.81 | 967.49 | 1.03 | 103.12 | 50 |
| NIM | FP8 | 1 | 519.84 | 848.03 | 1.18 | 117.44 | 30 |
| NIM | FP8 | 8 | 1809.48 | 2701.04 | 2.83 | 282.42 | 50 |
| NIM | FP8 | 16 | 3350.29 | 4947.38 | 3.16 | 314.85 | 80 |
| NIM | FP8 | 32 | 7804.60 | 9581.29 | 3.20 | 318.70 | 120 |
| vLLM | — | 64 | 2295.44 | 9738.18 | 6.54 | 653.91 | 320 |
| NIM | FP8 | 64 | 15444.84 | 18209.27 | 3.29 | 328.40 | 160 |
| vLLM | — | 128 | 8767.78 | 18061.79 | 6.53 | 653.39 | 256 |
| vLLM | — | 256 | 23214.88 | 33190.08 | 6.60 | 659.92 | 512 |

#### Input 50 / Output 100 / Video 4 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 909.74 | 1310.33 | 0.76 | 75.75 | 30 |
| NIM | FP8 | 8 | 2926.55 | 4160.93 | 1.86 | 185.19 | 50 |
| NIM | FP8 | 16 | 6326.60 | 7632.23 | 1.97 | 196.21 | 80 |
| NIM | FP8 | 32 | 13405.44 | 14932.49 | 1.98 | 197.54 | 120 |
| NIM | FP8 | 64 | 26985.48 | 28225.63 | 1.93 | 192.58 | 160 |

#### Input 50 / Output 100 / Video 8 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 1320.00 | 1790.23 | 0.56 | 55.54 | 30 |
| NIM | FP8 | 8 | 4438.29 | 5731.39 | 1.34 | 133.03 | 50 |
| NIM | FP8 | 16 | 9662.24 | 11069.30 | 1.37 | 136.19 | 80 |
| NIM | FP8 | 32 | 19122.95 | 20622.58 | 1.40 | 139.93 | 120 |
| NIM | FP8 | 64 | 38722.10 | 39943.47 | 1.34 | 133.20 | 160 |

### H200 141GB HBM3

#### Input 50 / Output 1 / Video 1 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 142.45 | 142.45 | 6.93 | 6.93 | 50 |
| NIM | FP8 | 1 | 308.91 | 308.91 | 3.21 | 3.21 | 30 |
| NIM | FP8 | 8 | 1337.25 | 1337.25 | 5.77 | 5.77 | 50 |
| NIM | FP8 | 16 | 2632.74 | 2632.74 | 5.79 | 5.79 | 80 |
| NIM | FP8 | 32 | 5409.51 | 5409.51 | 5.55 | 5.55 | 120 |
| vLLM | — | 64 | 3363.01 | 3363.01 | 17.25 | 17.25 | 320 |
| NIM | FP8 | 64 | 10985.30 | 10985.30 | 5.25 | 5.25 | 160 |
| vLLM | — | 128 | 5656.58 | 5656.58 | 17.21 | 17.21 | 256 |
| vLLM | — | 256 | 11271.28 | 11271.28 | 17.17 | 17.17 | 512 |

#### Input 50 / Output 1 / Video 2 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 229.70 | 229.70 | 4.31 | 4.31 | 50 |
| NIM | FP8 | 1 | 509.00 | 509.00 | 1.95 | 1.95 | 30 |
| NIM | FP8 | 8 | 1973.29 | 1973.29 | 3.84 | 3.84 | 50 |
| NIM | FP8 | 16 | 3930.73 | 3930.73 | 3.81 | 3.81 | 80 |
| NIM | FP8 | 32 | 7919.12 | 7919.12 | 3.70 | 3.70 | 120 |
| vLLM | — | 64 | 6932.55 | 6932.55 | 8.35 | 8.35 | 320 |
| NIM | FP8 | 64 | 15346.39 | 15346.39 | 3.65 | 3.65 | 160 |
| vLLM | — | 128 | 11640.68 | 11640.68 | 8.32 | 8.32 | 256 |
| vLLM | — | 256 | 23173.10 | 23173.10 | 8.33 | 8.33 | 512 |

#### Input 50 / Output 1 / Video 4 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 961.91 | 961.91 | 1.03 | 1.03 | 30 |
| NIM | FP8 | 8 | 3424.23 | 3424.23 | 2.18 | 2.18 | 50 |
| NIM | FP8 | 16 | 6600.38 | 6600.38 | 2.22 | 2.22 | 80 |
| NIM | FP8 | 32 | 13194.50 | 13194.50 | 2.15 | 2.15 | 120 |
| NIM | FP8 | 64 | 24661.92 | 24661.92 | 2.17 | 2.17 | 160 |

#### Input 50 / Output 1 / Video 8 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 1262.94 | 1262.94 | 0.79 | 0.79 | 30 |
| NIM | FP8 | 8 | 4562.13 | 4562.13 | 1.64 | 1.64 | 50 |
| NIM | FP8 | 16 | 9037.93 | 9037.93 | 1.61 | 1.61 | 80 |
| NIM | FP8 | 32 | 17326.68 | 17326.68 | 1.62 | 1.62 | 120 |
| NIM | FP8 | 64 | 33841.14 | 33841.14 | 1.56 | 1.56 | 160 |

#### Input 50 / Output 100 / Video 1 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 143.25 | 711.77 | 1.40 | 140.02 | 50 |
| NIM | FP8 | 1 | 317.14 | 525.16 | 1.90 | 189.62 | 30 |
| NIM | FP8 | 8 | 1327.65 | 1842.00 | 4.11 | 410.63 | 50 |
| NIM | FP8 | 16 | 2504.12 | 3424.74 | 4.58 | 457.09 | 80 |
| NIM | FP8 | 32 | 5038.65 | 6463.08 | 4.85 | 484.33 | 120 |
| vLLM | — | 64 | 2060.05 | 4965.09 | 12.84 | 1284.38 | 320 |
| NIM | FP8 | 64 | 10281.75 | 12299.72 | 4.88 | 487.04 | 160 |
| vLLM | — | 128 | 2839.97 | 9364.20 | 13.53 | 1352.41 | 256 |
| vLLM | — | 256 | 4713.83 | 18325.25 | 13.75 | 1374.57 | 512 |

#### Input 50 / Output 100 / Video 2 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 228.24 | 807.07 | 1.24 | 123.55 | 50 |
| NIM | FP8 | 1 | 524.38 | 785.40 | 1.27 | 126.91 | 30 |
| NIM | FP8 | 8 | 1714.05 | 2560.71 | 2.96 | 296.11 | 50 |
| NIM | FP8 | 16 | 3560.16 | 4779.41 | 3.23 | 322.87 | 80 |
| NIM | FP8 | 32 | 7892.99 | 9416.87 | 3.30 | 328.67 | 120 |
| vLLM | — | 64 | 2715.85 | 9341.21 | 6.82 | 682.33 | 320 |
| NIM | FP8 | 64 | 15925.16 | 17726.90 | 3.32 | 330.65 | 160 |
| vLLM | — | 128 | 5015.19 | 18285.65 | 6.90 | 690.26 | 256 |
| vLLM | — | 256 | 15991.33 | 35043.34 | 6.90 | 689.50 | 512 |

#### Input 50 / Output 100 / Video 4 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 899.23 | 1231.12 | 0.81 | 80.97 | 30 |
| NIM | FP8 | 8 | 2934.41 | 3943.17 | 1.96 | 195.37 | 50 |
| NIM | FP8 | 16 | 6724.25 | 7647.45 | 1.94 | 193.26 | 80 |
| NIM | FP8 | 32 | 14215.19 | 15141.39 | 1.92 | 191.32 | 120 |
| NIM | FP8 | 64 | 26680.93 | 27830.02 | 1.97 | 196.94 | 160 |

#### Input 50 / Output 100 / Video 8 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 1282.17 | 1683.42 | 0.59 | 59.12 | 30 |
| NIM | FP8 | 8 | 4786.68 | 5733.87 | 1.32 | 131.82 | 50 |
| NIM | FP8 | 16 | 9743.44 | 10773.89 | 1.38 | 137.02 | 80 |
| NIM | FP8 | 32 | 18481.96 | 19860.44 | 1.46 | 145.04 | 120 |
| NIM | FP8 | 64 | 36665.23 | 37927.89 | 1.44 | 142.98 | 160 |

### B200

#### Input 50 / Output 1 / Video 1 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 115.55 | 115.55 | 8.53 | 8.53 | 50 |
| NIM | FP8 | 1 | 237.13 | 237.13 | 4.17 | 4.17 | 30 |
| NIM | NVFP4 | 1 | 242.09 | 242.09 | 4.06 | 4.06 | 30 |
| NIM | FP8 | 8 | 1120.39 | 1120.39 | 6.87 | 6.87 | 50 |
| NIM | NVFP4 | 8 | 1043.98 | 1043.98 | 7.37 | 7.37 | 50 |
| NIM | FP8 | 16 | 2127.37 | 2127.37 | 7.13 | 7.13 | 80 |
| NIM | NVFP4 | 16 | 2043.03 | 2043.03 | 7.42 | 7.42 | 80 |
| NIM | FP8 | 32 | 4300.34 | 4300.34 | 6.91 | 6.91 | 120 |
| NIM | NVFP4 | 32 | 4149.21 | 4149.21 | 7.17 | 7.17 | 120 |
| vLLM | — | 64 | 1661.57 | 1661.57 | 34.96 | 34.96 | 320 |
| NIM | FP8 | 64 | 8618.79 | 8618.79 | 6.62 | 6.62 | 160 |
| NIM | NVFP4 | 64 | 8322.73 | 8322.73 | 6.83 | 6.83 | 160 |
| vLLM | — | 128 | 2819.22 | 2819.22 | 34.72 | 34.72 | 256 |
| vLLM | — | 256 | 5550.74 | 5550.74 | 34.94 | 34.94 | 512 |

#### Input 50 / Output 1 / Video 2 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 168.95 | 168.95 | 5.86 | 5.86 | 50 |
| NIM | FP8 | 1 | 423.17 | 423.17 | 2.34 | 2.34 | 30 |
| NIM | NVFP4 | 1 | 391.79 | 391.79 | 2.53 | 2.53 | 30 |
| NIM | FP8 | 8 | 1584.07 | 1584.07 | 4.76 | 4.76 | 50 |
| NIM | NVFP4 | 8 | 1560.39 | 1560.39 | 4.81 | 4.81 | 50 |
| NIM | FP8 | 16 | 3302.85 | 3302.85 | 4.47 | 4.47 | 80 |
| NIM | NVFP4 | 16 | 2900.49 | 2900.49 | 5.09 | 5.09 | 80 |
| NIM | FP8 | 32 | 6281.30 | 6281.30 | 4.58 | 4.58 | 120 |
| NIM | NVFP4 | 32 | 5768.67 | 5768.67 | 4.98 | 4.98 | 120 |
| vLLM | — | 64 | 3410.27 | 3410.27 | 16.98 | 16.98 | 320 |
| NIM | FP8 | 64 | 12280.02 | 12280.02 | 4.42 | 4.42 | 160 |
| NIM | NVFP4 | 64 | 12117.13 | 12117.13 | 4.48 | 4.48 | 160 |
| vLLM | — | 128 | 5699.88 | 5699.88 | 17.01 | 17.01 | 256 |
| vLLM | — | 256 | 11422.16 | 11422.16 | 16.93 | 16.93 | 512 |

#### Input 50 / Output 1 / Video 4 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 755.89 | 755.89 | 1.31 | 1.31 | 30 |
| NIM | NVFP4 | 1 | 721.67 | 721.67 | 1.37 | 1.37 | 30 |
| NIM | FP8 | 8 | 2677.30 | 2677.30 | 2.79 | 2.79 | 50 |
| NIM | NVFP4 | 8 | 2833.32 | 2833.32 | 2.64 | 2.64 | 50 |
| NIM | FP8 | 16 | 5722.03 | 5722.03 | 2.53 | 2.53 | 80 |
| NIM | NVFP4 | 16 | 5955.06 | 5955.06 | 2.46 | 2.46 | 80 |
| NIM | FP8 | 32 | 10434.58 | 10434.58 | 2.70 | 2.70 | 120 |
| NIM | NVFP4 | 32 | 11655.43 | 11655.43 | 2.42 | 2.42 | 120 |
| NIM | FP8 | 64 | 21739.39 | 21739.39 | 2.39 | 2.39 | 160 |
| NIM | NVFP4 | 64 | 21340.43 | 21340.43 | 2.48 | 2.48 | 160 |

#### Input 50 / Output 1 / Video 8 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 1051.29 | 1051.29 | 0.94 | 0.94 | 30 |
| NIM | NVFP4 | 1 | 988.04 | 988.04 | 1.01 | 1.01 | 30 |
| NIM | FP8 | 8 | 4772.87 | 4772.87 | 1.56 | 1.56 | 50 |
| NIM | NVFP4 | 8 | 4389.71 | 4389.71 | 1.70 | 1.70 | 50 |
| NIM | FP8 | 16 | 8539.57 | 8539.57 | 1.71 | 1.71 | 80 |
| NIM | NVFP4 | 16 | 8788.79 | 8788.79 | 1.66 | 1.66 | 80 |
| NIM | FP8 | 32 | 16313.79 | 16313.79 | 1.73 | 1.73 | 120 |
| NIM | NVFP4 | 32 | 16113.36 | 16113.36 | 1.74 | 1.74 | 120 |
| NIM | FP8 | 64 | 31628.49 | 31628.49 | 1.63 | 1.63 | 160 |
| NIM | NVFP4 | 64 | 31464.74 | 31464.74 | 1.65 | 1.65 | 160 |

#### Input 50 / Output 100 / Video 1 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 115.27 | 553.01 | 1.80 | 180.16 | 50 |
| NIM | FP8 | 1 | 243.86 | 473.68 | 2.10 | 209.99 | 30 |
| NIM | NVFP4 | 1 | 238.95 | 431.18 | 2.31 | 230.27 | 30 |
| NIM | FP8 | 8 | 981.76 | 1393.50 | 5.57 | 556.03 | 50 |
| NIM | NVFP4 | 8 | 957.38 | 1316.29 | 5.91 | 589.68 | 50 |
| NIM | FP8 | 16 | 1998.00 | 2550.16 | 5.92 | 591.49 | 80 |
| NIM | NVFP4 | 16 | 2010.88 | 2416.13 | 6.33 | 631.73 | 80 |
| NIM | FP8 | 32 | 3914.27 | 4802.62 | 6.31 | 630.02 | 120 |
| NIM | NVFP4 | 32 | 4367.16 | 4782.00 | 6.33 | 631.67 | 120 |
| vLLM | — | 64 | 1106.35 | 2736.53 | 23.28 | 2328.01 | 320 |
| NIM | FP8 | 64 | 9220.44 | 9876.21 | 6.02 | 601.01 | 160 |
| NIM | NVFP4 | 64 | 9331.22 | 9706.46 | 6.03 | 600.81 | 160 |
| vLLM | — | 128 | 2111.97 | 5001.20 | 25.23 | 2523.07 | 256 |
| vLLM | — | 256 | 2549.79 | 9279.25 | 27.01 | 2701.08 | 512 |

#### Input 50 / Output 100 / Video 2 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 166.52 | 622.36 | 1.60 | 160.11 | 50 |
| NIM | FP8 | 1 | 437.45 | 698.55 | 1.43 | 142.12 | 30 |
| NIM | NVFP4 | 1 | 399.13 | 608.52 | 1.64 | 163.26 | 30 |
| NIM | FP8 | 8 | 1453.73 | 2032.94 | 3.79 | 378.34 | 50 |
| NIM | NVFP4 | 8 | 1517.28 | 1855.22 | 4.16 | 415.30 | 50 |
| NIM | FP8 | 16 | 3772.21 | 4202.02 | 3.59 | 357.77 | 80 |
| NIM | NVFP4 | 16 | 3284.47 | 3647.38 | 4.16 | 413.80 | 80 |
| NIM | FP8 | 32 | 7337.26 | 7898.21 | 3.74 | 373.70 | 120 |
| NIM | NVFP4 | 32 | 6999.54 | 7357.78 | 4.02 | 400.10 | 120 |
| vLLM | — | 64 | 1914.24 | 4881.38 | 13.04 | 1303.92 | 320 |
| NIM | FP8 | 64 | 13372.43 | 14036.55 | 4.04 | 403.63 | 160 |
| NIM | NVFP4 | 64 | 13375.96 | 13732.15 | 4.10 | 408.34 | 160 |
| vLLM | — | 128 | 2596.36 | 9220.30 | 13.63 | 1362.49 | 256 |
| vLLM | — | 256 | 7277.89 | 17548.01 | 13.87 | 1386.99 | 512 |

#### Input 50 / Output 100 / Video 4 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 793.86 | 1117.10 | 0.89 | 88.85 | 30 |
| NIM | NVFP4 | 1 | 721.04 | 993.06 | 1.00 | 99.84 | 30 |
| NIM | FP8 | 8 | 2916.88 | 3458.54 | 2.20 | 219.08 | 50 |
| NIM | NVFP4 | 8 | 2905.02 | 3358.08 | 2.27 | 195.15 | 50 |
| NIM | FP8 | 16 | 6177.39 | 6747.97 | 2.18 | 217.48 | 80 |
| NIM | NVFP4 | 16 | 5534.25 | 6325.50 | 2.35 | 116.91 | 80 |
| NIM | FP8 | 32 | 12023.87 | 12602.24 | 2.28 | 226.99 | 120 |
| NIM | NVFP4 | 32 | 12332.78 | 13252.44 | 2.17 | 56.50 | 120 |
| NIM | FP8 | 64 | 22694.09 | 23320.71 | 2.32 | 231.01 | 160 |
| NIM | NVFP4 | 64 | 22412.73 | 23494.52 | 2.31 | 30.04 | 160 |

#### Input 50 / Output 100 / Video 8 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 1099.88 | 1460.91 | 0.68 | 67.95 | 30 |
| NIM | NVFP4 | 1 | 968.75 | 1251.79 | 0.80 | 79.39 | 30 |
| NIM | FP8 | 8 | 4473.46 | 5082.69 | 1.48 | 147.96 | 50 |
| NIM | NVFP4 | 8 | 4403.32 | 4737.75 | 1.58 | 156.97 | 50 |
| NIM | FP8 | 16 | 9046.99 | 9678.10 | 1.53 | 152.13 | 80 |
| NIM | NVFP4 | 16 | 8699.01 | 9039.05 | 1.62 | 160.85 | 80 |
| NIM | FP8 | 32 | 16676.11 | 17379.85 | 1.64 | 163.17 | 120 |
| NIM | NVFP4 | 32 | 17937.84 | 18251.79 | 1.55 | 153.71 | 120 |
| NIM | FP8 | 64 | 34333.08 | 34986.52 | 1.51 | 150.32 | 160 |
| NIM | NVFP4 | 64 | 32737.11 | 33055.53 | 1.57 | 156.23 | 160 |

### B300

#### Input 50 / Output 1 / Video 1 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 80.70 | 80.70 | 12.24 | 12.24 | 50 |
| NIM | FP8 | 1 | 245.23 | 245.23 | 4.01 | 4.01 | 30 |
| NIM | NVFP4 | 1 | 271.39 | 271.39 | 3.64 | 3.64 | 30 |
| NIM | FP8 | 8 | 1128.67 | 1128.67 | 6.80 | 6.80 | 50 |
| NIM | NVFP4 | 8 | 1278.36 | 1278.36 | 5.98 | 5.98 | 50 |
| NIM | FP8 | 16 | 2540.54 | 2540.54 | 5.88 | 5.88 | 80 |
| NIM | NVFP4 | 16 | 2553.62 | 2553.62 | 5.86 | 5.86 | 80 |
| NIM | FP8 | 32 | 5425.43 | 5425.43 | 5.43 | 5.43 | 120 |
| NIM | NVFP4 | 32 | 4124.61 | 4124.61 | 7.25 | 7.25 | 120 |
| vLLM | — | 64 | 1617.12 | 1617.12 | 35.92 | 35.92 | 320 |
| NIM | FP8 | 64 | 10142.65 | 10142.65 | 5.52 | 5.52 | 160 |
| NIM | NVFP4 | 64 | 10122.37 | 10122.37 | 5.54 | 5.54 | 160 |
| vLLM | — | 128 | 2742.68 | 2742.68 | 35.69 | 35.69 | 256 |
| vLLM | — | 256 | 5421.22 | 5421.22 | 35.79 | 35.79 | 511 |

#### Input 50 / Output 1 / Video 2 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 126.25 | 126.25 | 7.86 | 7.86 | 50 |
| NIM | FP8 | 1 | 503.00 | 503.00 | 1.97 | 1.97 | 30 |
| NIM | NVFP4 | 1 | 425.36 | 425.36 | 2.33 | 2.33 | 30 |
| NIM | FP8 | 8 | 2057.91 | 2057.91 | 3.67 | 3.67 | 50 |
| NIM | NVFP4 | 8 | 2309.47 | 2309.47 | 3.24 | 3.24 | 50 |
| NIM | FP8 | 16 | 3604.95 | 3604.95 | 4.09 | 4.09 | 80 |
| NIM | NVFP4 | 16 | 3807.48 | 3807.48 | 3.86 | 3.86 | 80 |
| NIM | FP8 | 32 | 7161.68 | 7161.68 | 3.95 | 3.95 | 120 |
| NIM | NVFP4 | 32 | 7924.13 | 7924.13 | 3.62 | 3.62 | 120 |
| vLLM | — | 64 | 3304.76 | 3304.76 | 17.53 | 17.53 | 320 |
| NIM | FP8 | 64 | 15123.50 | 15123.50 | 3.58 | 3.58 | 160 |
| NIM | NVFP4 | 64 | 14149.35 | 14149.35 | 3.75 | 3.75 | 160 |
| vLLM | — | 128 | 5551.12 | 5551.12 | 17.49 | 17.49 | 256 |
| vLLM | — | 256 | 11054.47 | 11054.47 | 17.49 | 17.49 | 512 |

#### Input 50 / Output 1 / Video 4 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 933.40 | 933.40 | 1.06 | 1.06 | 30 |
| NIM | NVFP4 | 1 | 885.63 | 885.63 | 1.12 | 1.12 | 30 |
| NIM | FP8 | 8 | 4318.24 | 4318.24 | 1.73 | 1.73 | 50 |
| NIM | NVFP4 | 8 | 3771.42 | 3771.42 | 2.00 | 2.00 | 50 |
| NIM | FP8 | 16 | 7739.33 | 7739.33 | 1.86 | 1.86 | 80 |
| NIM | NVFP4 | 16 | 7659.64 | 7659.64 | 1.90 | 1.90 | 80 |
| NIM | FP8 | 32 | 15385.63 | 15385.63 | 1.83 | 1.83 | 120 |
| NIM | NVFP4 | 32 | 15467.89 | 15467.89 | 1.80 | 1.80 | 120 |
| NIM | FP8 | 64 | 26090.01 | 26090.01 | 1.96 | 1.96 | 160 |
| NIM | NVFP4 | 64 | 29232.28 | 29232.28 | 1.79 | 1.79 | 160 |

#### Input 50 / Output 1 / Video 8 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 1290.72 | 1290.72 | 0.77 | 0.77 | 30 |
| NIM | NVFP4 | 1 | 1144.52 | 1144.52 | 0.87 | 0.87 | 30 |
| NIM | FP8 | 8 | 6253.13 | 6253.13 | 1.19 | 1.19 | 50 |
| NIM | NVFP4 | 8 | 5341.59 | 5341.59 | 1.40 | 1.40 | 50 |
| NIM | FP8 | 16 | 11460.92 | 11460.92 | 1.27 | 1.27 | 80 |
| NIM | NVFP4 | 16 | 12047.88 | 12047.88 | 1.21 | 1.21 | 80 |
| NIM | FP8 | 32 | 24536.15 | 24536.15 | 1.14 | 1.14 | 120 |
| NIM | NVFP4 | 32 | 26044.15 | 26044.15 | 1.07 | 1.07 | 120 |
| NIM | FP8 | 64 | 44685.47 | 44685.47 | 1.17 | 1.17 | 160 |
| NIM | NVFP4 | 64 | 43449.45 | 43449.45 | 1.20 | 1.20 | 160 |

#### Input 50 / Output 100 / Video 1 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 83.19 | 490.11 | 2.03 | 203.29 | 50 |
| NIM | FP8 | 1 | 281.68 | 506.31 | 1.97 | 196.55 | 30 |
| NIM | NVFP4 | 1 | 277.62 | 528.63 | 1.89 | 187.72 | 30 |
| NIM | FP8 | 8 | 1156.76 | 1541.62 | 4.95 | 494.07 | 50 |
| NIM | NVFP4 | 8 | 1117.25 | 1550.91 | 5.08 | 504.55 | 50 |
| NIM | FP8 | 16 | 2101.69 | 2635.92 | 5.80 | 578.30 | 80 |
| NIM | NVFP4 | 16 | 2458.83 | 2891.94 | 5.30 | 528.19 | 80 |
| NIM | FP8 | 32 | 4124.11 | 4818.90 | 6.34 | 633.16 | 120 |
| NIM | NVFP4 | 32 | 4929.86 | 5376.46 | 5.58 | 555.53 | 120 |
| vLLM | — | 64 | 1070.93 | 2657.06 | 23.96 | 2396.35 | 320 |
| NIM | FP8 | 64 | 10858.55 | 11316.58 | 5.08 | 506.63 | 160 |
| NIM | NVFP4 | 64 | 10075.93 | 10425.89 | 5.60 | 557.56 | 160 |
| vLLM | — | 128 | 1444.68 | 4750.02 | 26.57 | 2657.14 | 256 |
| vLLM | — | 256 | 2739.50 | 8975.21 | 27.92 | 2791.79 | 512 |

#### Input 50 / Output 100 / Video 2 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 129.22 | 550.02 | 1.81 | 181.25 | 50 |
| NIM | FP8 | 1 | 449.01 | 744.83 | 1.34 | 133.82 | 30 |
| NIM | NVFP4 | 1 | 471.56 | 732.74 | 1.36 | 135.93 | 30 |
| NIM | FP8 | 8 | 1945.13 | 2370.19 | 3.22 | 321.04 | 50 |
| NIM | NVFP4 | 8 | 1792.81 | 2171.62 | 3.49 | 346.59 | 50 |
| NIM | FP8 | 16 | 4285.32 | 4701.31 | 3.17 | 315.98 | 80 |
| NIM | NVFP4 | 16 | 4549.43 | 4921.58 | 3.01 | 299.49 | 80 |
| NIM | FP8 | 32 | 8071.59 | 8555.39 | 3.47 | 346.71 | 120 |
| NIM | NVFP4 | 32 | 7894.35 | 8273.62 | 3.49 | 347.81 | 120 |
| vLLM | — | 64 | 1602.00 | 4684.78 | 13.59 | 1358.62 | 320 |
| NIM | FP8 | 64 | 17846.34 | 18285.84 | 3.00 | 299.31 | 160 |
| NIM | NVFP4 | 64 | 14351.63 | 14765.34 | 3.79 | 376.81 | 160 |
| vLLM | — | 128 | 2404.48 | 8813.83 | 14.25 | 1425.02 | 256 |
| vLLM | — | 256 | 6982.58 | 16813.61 | 14.47 | 1447.23 | 512 |

#### Input 50 / Output 100 / Video 4 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 832.33 | 1200.86 | 0.83 | 82.95 | 30 |
| NIM | NVFP4 | 1 | 844.04 | 1197.89 | 0.83 | 82.95 | 30 |
| NIM | FP8 | 8 | 3937.20 | 4350.67 | 1.72 | 171.76 | 50 |
| NIM | NVFP4 | 8 | 3593.25 | 4013.75 | 1.85 | 183.28 | 50 |
| NIM | FP8 | 16 | 6687.37 | 7246.72 | 2.05 | 203.91 | 80 |
| NIM | NVFP4 | 16 | 7538.13 | 7813.42 | 1.85 | 184.45 | 80 |
| NIM | FP8 | 32 | 16537.76 | 17024.77 | 1.66 | 165.94 | 120 |
| NIM | NVFP4 | 32 | 15375.12 | 15814.95 | 1.81 | 180.18 | 120 |
| NIM | FP8 | 64 | 29923.52 | 30444.23 | 1.74 | 172.94 | 160 |
| NIM | NVFP4 | 64 | 26448.62 | 26880.97 | 1.96 | 194.51 | 160 |

#### Input 50 / Output 100 / Video 8 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 1217.99 | 1565.16 | 0.64 | 63.67 | 30 |
| NIM | NVFP4 | 1 | 1117.03 | 1494.89 | 0.67 | 66.29 | 30 |
| NIM | FP8 | 8 | 5952.61 | 6383.50 | 1.17 | 116.02 | 50 |
| NIM | NVFP4 | 8 | 5540.38 | 5898.33 | 1.26 | 125.28 | 50 |
| NIM | FP8 | 16 | 12771.50 | 13201.76 | 1.10 | 108.96 | 80 |
| NIM | NVFP4 | 16 | 12578.13 | 12960.52 | 1.13 | 112.16 | 80 |
| NIM | FP8 | 32 | 18290.56 | 18957.08 | 1.49 | 148.73 | 120 |
| NIM | NVFP4 | 32 | 20538.69 | 20942.34 | 1.34 | 133.64 | 120 |
| NIM | FP8 | 64 | 45718.42 | 46182.77 | 1.13 | 112.44 | 160 |
| NIM | NVFP4 | 64 | 43374.82 | 43767.04 | 1.20 | 119.43 | 160 |

<sub>Notes:
1. Source: vLLM inference benchmarking for `nvidia/Cosmos3-Nano`; AIPerf client was used as the benchmarking tool for the vLLM results.
2. Hardware: results are grouped by GPU product (RTX PRO 6000 Blackwell, H20, H100 NVL, H200 NVL, H100 80GB HBM3 SXM, H200 141GB HBM3, B200, B300). Latency metrics are request averages; throughput metrics are aggregate rates. Request counts indicate measurement volume.
3. **Time To First Token (TTFT)** measures latency until the first output token is emitted. **Request Latency** is end-to-end time per request. For single-token outputs (Output 1), TTFT and request latency are identical.
4. **Request Throughput** is completed requests per second. **Output Token Throughput** is generated tokens per second (for Output 1 workloads, the two throughputs match).
5. Concurrency is the number of simultaneous client requests, not tensor-parallel GPU count. The vLLM requests were issued by AIPerf.
6. NIM FP8 and NVFP4 are separate profiles. The published vLLM baseline does not specify precision or establish identical runtime settings.</sub>

## Cosmos3-Super Reasoner

These tables compare **Cosmos3-Super** reasoner serving performance through **vLLM** and **NIM**, grouped by GPU and workload. Lower is better for latency; higher is better for throughput. Precision is shown for NIM; `—` indicates that the vLLM baseline did not specify precision.

NIM runs use one GPU, tensor parallelism 1, and synthetic 374×374, 30-second videos. Input and output lengths are token targets. Concurrency is simultaneous client requests, not GPU count. Blank cells indicate missing results; backend coverage differs.

### RTX PRO 6000 Blackwell

#### Input 50 / Output 1 / Video 1 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 534.73 | 534.73 | 1.86 | 1.86 | 50 |
| NIM | FP8 | 1 | 519.52 | 519.52 | 1.92 | 1.92 | 30 |
| NIM | NVFP4 | 1 | 411.73 | 411.73 | 2.41 | 2.41 | 30 |
| NIM | FP8 | 8 | 2660.63 | 2660.63 | 2.90 | 2.90 | 50 |
| NIM | NVFP4 | 8 | 1963.19 | 1963.19 | 3.88 | 3.88 | 50 |
| NIM | FP8 | 16 | 5214.39 | 5214.39 | 2.89 | 2.89 | 80 |
| NIM | NVFP4 | 16 | 3697.16 | 3697.16 | 4.05 | 4.05 | 80 |
| NIM | FP8 | 32 | 10081.63 | 10081.63 | 2.86 | 2.86 | 120 |
| NIM | NVFP4 | 32 | 7244.67 | 7244.67 | 3.99 | 3.99 | 120 |
| vLLM | — | 64 | 24781.47 | 24781.47 | 2.34 | 2.34 | 320 |
| NIM | FP8 | 64 | 19135.23 | 19135.23 | 2.80 | 2.80 | 160 |
| NIM | NVFP4 | 64 | 13967.90 | 13967.90 | 3.87 | 3.87 | 160 |
| vLLM | — | 128 | 41467.45 | 41467.45 | 2.34 | 2.34 | 256 |
| vLLM | — | 256 | 82626.36 | 82626.36 | 2.32 | 2.32 | 509 |

#### Input 50 / Output 1 / Video 2 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 978.78 | 978.78 | 1.02 | 1.02 | 50 |
| NIM | FP8 | 1 | 988.43 | 988.43 | 1.01 | 1.01 | 30 |
| NIM | NVFP4 | 1 | 786.15 | 786.15 | 1.27 | 1.27 | 30 |
| NIM | FP8 | 8 | 5360.41 | 5360.41 | 1.43 | 1.43 | 50 |
| NIM | NVFP4 | 8 | 3854.34 | 3854.34 | 1.97 | 1.97 | 50 |
| NIM | FP8 | 16 | 10428.30 | 10428.30 | 1.41 | 1.41 | 80 |
| NIM | NVFP4 | 16 | 7579.26 | 7579.26 | 1.95 | 1.95 | 80 |
| NIM | FP8 | 32 | 20243.60 | 20243.60 | 1.40 | 1.40 | 120 |
| NIM | NVFP4 | 32 | 14646.19 | 14646.19 | 1.93 | 1.93 | 120 |
| vLLM | — | 64 | 51145.61 | 51145.61 | 1.13 | 1.13 | 320 |
| NIM | FP8 | 64 | 36939.03 | 36939.03 | 1.41 | 1.41 | 160 |
| NIM | NVFP4 | 64 | 27173.90 | 27173.90 | 1.93 | 1.93 | 160 |
| vLLM | — | 128 | 85476.50 | 85476.50 | 1.13 | 1.13 | 256 |
| vLLM | — | 256 |  |  |  |  |  |

#### Input 50 / Output 1 / Video 4 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 1943.77 | 1943.77 | 0.51 | 0.51 | 30 |
| NIM | NVFP4 | 1 | 1578.30 | 1578.30 | 0.63 | 0.63 | 30 |
| NIM | FP8 | 8 | 11133.08 | 11133.08 | 0.67 | 0.67 | 50 |
| NIM | NVFP4 | 8 | 8426.33 | 8426.33 | 0.89 | 0.89 | 50 |
| NIM | FP8 | 16 | 21843.91 | 21843.91 | 0.67 | 0.67 | 80 |
| NIM | NVFP4 | 16 | 16514.70 | 16514.70 | 0.88 | 0.88 | 80 |
| NIM | FP8 | 32 | 42231.66 | 42231.66 | 0.66 | 0.66 | 120 |
| NIM | NVFP4 | 32 | 31941.03 | 31941.03 | 0.88 | 0.88 | 120 |
| NIM | FP8 | 64 | 78368.66 | 78368.66 | 0.66 | 0.66 | 160 |
| NIM | NVFP4 | 64 | 58956.11 | 58956.11 | 0.88 | 0.88 | 160 |

#### Input 50 / Output 1 / Video 8 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 2819.52 | 2819.52 | 0.35 | 0.35 | 30 |
| NIM | NVFP4 | 1 | 2338.34 | 2338.34 | 0.43 | 0.43 | 30 |
| NIM | FP8 | 8 | 16412.04 | 16412.04 | 0.45 | 0.45 | 50 |
| NIM | NVFP4 | 8 | 12742.83 | 12742.83 | 0.59 | 0.59 | 50 |
| NIM | FP8 | 16 | 31960.87 | 31960.87 | 0.46 | 0.46 | 80 |
| NIM | NVFP4 | 16 | 24884.35 | 24884.35 | 0.58 | 0.58 | 80 |
| NIM | FP8 | 32 | 61814.80 | 61814.80 | 0.45 | 0.45 | 120 |
| NIM | NVFP4 | 32 | 48083.54 | 48083.54 | 0.58 | 0.58 | 120 |
| NIM | FP8 | 64 | 115545.21 | 115545.21 | 0.45 | 0.45 | 160 |
| NIM | NVFP4 | 64 | 89259.66 | 89259.66 | 0.58 | 0.58 | 160 |

#### Input 50 / Output 100 / Video 1 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 530.47 | 5225.79 | 0.19 | 19.12 | 50 |
| NIM | FP8 | 1 | 523.82 | 2601.95 | 0.38 | 38.38 | 30 |
| NIM | NVFP4 | 1 | 414.56 | 1803.99 | 0.55 | 55.37 | 30 |
| NIM | FP8 | 8 | 1353.26 | 4963.21 | 1.57 | 156.94 | 50 |
| NIM | NVFP4 | 8 | 1608.50 | 4031.81 | 1.91 | 190.54 | 50 |
| NIM | FP8 | 16 | 2162.00 | 7955.26 | 1.91 | 191.12 | 80 |
| NIM | NVFP4 | 16 | 3087.23 | 6683.27 | 2.27 | 226.88 | 80 |
| NIM | FP8 | 32 | 5998.86 | 14261.37 | 2.08 | 208.13 | 120 |
| NIM | NVFP4 | 32 | 6310.08 | 11995.56 | 2.51 | 251.37 | 120 |
| vLLM | — | 64 | 25094.90 | 40064.22 | 1.51 | 151.27 | 320 |
| NIM | FP8 | 64 | 12592.34 | 28531.35 | 2.14 | 214.34 | 160 |
| NIM | NVFP4 | 64 | 13936.07 | 25117.49 | 2.49 | 249.03 | 160 |
| vLLM | — | 128 | 54400.80 | 69193.77 | 1.50 | 149.69 | 256 |
| vLLM | — | 256 | 117849.75 | 133019.50 | 1.51 | 151.15 | 512 |

#### Input 50 / Output 100 / Video 2 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 981.55 | 5716.49 | 0.17 | 17.49 | 50 |
| NIM | FP8 | 1 | 1002.18 | 3340.86 | 0.30 | 29.92 | 30 |
| NIM | NVFP4 | 1 | 789.56 | 2665.34 | 0.37 | 37.44 | 30 |
| NIM | FP8 | 8 | 2986.55 | 8395.26 | 0.91 | 90.52 | 50 |
| NIM | NVFP4 | 8 | 2118.18 | 6924.45 | 1.12 | 111.76 | 50 |
| NIM | FP8 | 16 | 3783.46 | 14299.83 | 1.09 | 109.16 | 80 |
| NIM | NVFP4 | 16 | 3709.07 | 12120.88 | 1.29 | 129.21 | 80 |
| NIM | FP8 | 32 | 9211.28 | 27108.39 | 1.15 | 115.03 | 120 |
| NIM | NVFP4 | 32 | 5230.37 | 22362.10 | 1.39 | 139.12 | 120 |
| vLLM | — | 64 |  |  |  |  |  |
| NIM | FP8 | 64 | 31913.35 | 49958.02 | 1.16 | 115.74 | 160 |
| NIM | NVFP4 | 64 | 23297.33 | 42789.99 | 1.38 | 137.85 | 160 |
| vLLM | — | 128 | 114704.46 | 130177.35 | 0.77 | 77.00 | 256 |
| vLLM | — | 256 |  |  |  |  |  |

#### Input 50 / Output 100 / Video 4 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 1973.87 | 4670.83 | 0.21 | 21.25 | 30 |
| NIM | NVFP4 | 1 | 1593.01 | 4082.89 | 0.24 | 24.47 | 30 |
| NIM | FP8 | 8 | 5395.03 | 15426.43 | 0.51 | 50.45 | 50 |
| NIM | NVFP4 | 8 | 3524.17 | 13322.08 | 0.58 | 58.21 | 50 |
| NIM | FP8 | 16 | 10142.42 | 28123.99 | 0.55 | 55.26 | 80 |
| NIM | NVFP4 | 16 | 5388.81 | 24486.95 | 0.64 | 63.90 | 80 |
| NIM | FP8 | 32 | 36091.26 | 54815.40 | 0.54 | 54.25 | 120 |
| NIM | NVFP4 | 32 | 25640.48 | 47028.63 | 0.64 | 64.37 | 120 |
| NIM | FP8 | 64 | 82154.16 | 101354.40 | 0.54 | 53.97 | 160 |
| NIM | NVFP4 | 64 | 66184.95 | 88267.36 | 0.63 | 63.22 | 160 |

#### Input 50 / Output 100 / Video 8 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 2845.90 | 5965.72 | 0.17 | 16.69 | 30 |
| NIM | NVFP4 | 1 | 2355.79 | 5359.24 | 0.19 | 18.58 | 30 |
| NIM | FP8 | 8 | 8051.05 | 21722.31 | 0.35 | 35.46 | 50 |
| NIM | NVFP4 | 8 | 6668.41 | 20127.90 | 0.39 | 38.69 | 50 |
| NIM | FP8 | 16 | 21211.03 | 41202.96 | 0.38 | 37.45 | 80 |
| NIM | NVFP4 | 16 | 14609.61 | 36534.72 | 0.42 | 42.30 | 80 |
| NIM | FP8 | 32 | 58041.75 | 78504.08 | 0.37 | 37.31 | 120 |
| NIM | NVFP4 | 32 | 47466.78 | 69928.51 | 0.42 | 42.24 | 120 |
| NIM | FP8 | 64 | 124218.56 | 144964.96 | 0.37 | 37.22 | 160 |
| NIM | NVFP4 | 64 | 106321.80 | 128927.32 | 0.42 | 42.27 | 160 |

### H20

#### Input 50 / Output 1 / Video 1 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 1241.58 | 1241.58 | 0.80 | 0.80 | 50 |
| NIM | FP8 | 1 | 1035.92 | 1035.92 | 0.96 | 0.96 | 30 |
| NIM | FP8 | 8 | 5915.69 | 5915.69 | 1.29 | 1.29 | 50 |
| NIM | FP8 | 16 | 11006.56 | 11006.56 | 1.36 | 1.36 | 80 |
| NIM | FP8 | 32 | 20912.99 | 20912.99 | 1.37 | 1.37 | 120 |
| vLLM | — | 64 |  |  |  |  |  |
| NIM | FP8 | 64 | 40550.50 | 40550.50 | 1.32 | 1.32 | 160 |
| vLLM | — | 128 | 108912.73 | 108912.73 | 0.89 | 0.89 | 256 |
| vLLM | — | 256 |  |  |  |  |  |

#### Input 50 / Output 1 / Video 2 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 2399.91 | 2399.91 | 0.42 | 0.42 | 50 |
| NIM | FP8 | 1 | 1902.04 | 1902.04 | 0.52 | 0.52 | 30 |
| NIM | FP8 | 8 | 11081.53 | 11081.53 | 0.68 | 0.68 | 50 |
| NIM | FP8 | 16 | 21664.62 | 21664.62 | 0.68 | 0.68 | 80 |
| NIM | FP8 | 32 | 41785.21 | 41785.21 | 0.68 | 0.68 | 120 |
| vLLM | — | 64 |  |  |  |  |  |
| NIM | FP8 | 64 | 77914.41 | 77914.41 | 0.67 | 0.67 | 160 |
| vLLM | — | 128 |  |  |  |  |  |
| vLLM | — | 256 |  |  |  |  |  |

#### Input 50 / Output 1 / Video 4 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 3679.95 | 3679.95 | 0.27 | 0.27 | 30 |
| NIM | FP8 | 8 | 22741.65 | 22741.65 | 0.33 | 0.33 | 50 |
| NIM | FP8 | 16 | 44312.07 | 44312.07 | 0.33 | 0.33 | 80 |
| NIM | FP8 | 32 | 85286.31 | 85286.31 | 0.33 | 0.33 | 120 |
| NIM | FP8 | 64 | 157161.46 | 157161.46 | 0.33 | 0.33 | 160 |

#### Input 50 / Output 1 / Video 8 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 5275.65 | 5275.65 | 0.19 | 0.19 | 30 |
| NIM | FP8 | 8 | 32751.71 | 32751.71 | 0.23 | 0.23 | 50 |
| NIM | FP8 | 16 | 63858.62 | 63858.62 | 0.23 | 0.23 | 80 |
| NIM | FP8 | 32 | 122971.24 | 122971.24 | 0.23 | 0.23 | 120 |
| NIM | FP8 | 64 | 227514.53 | 227514.53 | 0.23 | 0.23 | 160 |

#### Input 50 / Output 100 / Video 1 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 1230.95 | 3523.74 | 0.28 | 28.36 | 50 |
| NIM | FP8 | 1 | 1044.03 | 1826.70 | 0.55 | 54.65 | 30 |
| NIM | FP8 | 8 | 5246.58 | 7935.06 | 0.97 | 97.07 | 50 |
| NIM | FP8 | 16 | 7014.46 | 14653.98 | 1.04 | 104.09 | 80 |
| NIM | FP8 | 32 | 8438.72 | 29285.57 | 1.08 | 107.91 | 120 |
| vLLM | — | 64 |  |  |  |  |  |
| NIM | FP8 | 64 | 17867.35 | 54170.20 | 1.10 | 110.29 | 160 |
| vLLM | — | 128 | 108988.21 | 135784.46 | 0.78 | 77.81 | 256 |
| vLLM | — | 256 |  |  |  |  |  |

#### Input 50 / Output 100 / Video 2 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 2380.85 | 4707.46 | 0.21 | 21.22 | 50 |
| NIM | FP8 | 1 | 1927.69 | 2855.09 | 0.35 | 34.83 | 30 |
| NIM | FP8 | 8 | 5384.33 | 14177.50 | 0.56 | 55.56 | 50 |
| NIM | FP8 | 16 | 9968.51 | 27231.12 | 0.58 | 58.38 | 80 |
| NIM | FP8 | 32 | 14458.23 | 53756.43 | 0.59 | 58.86 | 120 |
| vLLM | — | 64 |  |  |  |  |  |
| NIM | FP8 | 64 | 39728.21 | 106407.08 | 0.59 | 58.51 | 160 |
| vLLM | — | 128 |  |  |  |  |  |
| vLLM | — | 256 |  |  |  |  |  |

#### Input 50 / Output 100 / Video 4 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 3716.46 | 4903.53 | 0.20 | 20.36 | 30 |
| NIM | FP8 | 8 | 10114.49 | 26040.43 | 0.29 | 29.34 | 50 |
| NIM | FP8 | 16 | 14416.86 | 50974.28 | 0.30 | 29.81 | 80 |
| NIM | FP8 | 32 | 37024.50 | 104912.22 | 0.30 | 29.92 | 120 |
| NIM | FP8 | 64 | 126555.26 | 196883.58 | 0.30 | 29.64 | 160 |

#### Input 50 / Output 100 / Video 8 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 5258.94 | 6440.22 | 0.16 | 15.49 | 30 |
| NIM | FP8 | 8 | 18324.20 | 38370.48 | 0.21 | 20.43 | 50 |
| NIM | FP8 | 16 | 28292.57 | 74840.29 | 0.21 | 20.99 | 80 |
| NIM | FP8 | 32 | 75050.46 | 148325.45 | 0.21 | 20.79 | 120 |
| NIM | FP8 | 64 | 221267.10 | 300764.60 | 0.19 | 18.83 | 160 |

### H100 NVL

#### Input 50 / Output 1 / Video 1 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 521.87 | 521.87 | 1.91 | 1.91 | 50 |
| NIM | FP8 | 1 | 473.34 | 473.34 | 2.10 | 2.10 | 30 |
| NIM | FP8 | 8 | 2557.64 | 2557.64 | 3.01 | 3.01 | 50 |
| NIM | FP8 | 16 | 5013.63 | 5013.63 | 2.99 | 2.99 | 80 |
| NIM | FP8 | 32 | 9948.04 | 9948.04 | 2.90 | 2.90 | 120 |
| vLLM | — | 64 | 27004.42 | 27004.42 | 2.15 | 2.15 | 320 |
| NIM | FP8 | 64 | 19271.04 | 19271.04 | 2.80 | 2.80 | 160 |
| vLLM | — | 128 | 45688.95 | 45688.95 | 2.13 | 2.13 | 256 |
| vLLM | — | 256 | 90353.80 | 90353.80 | 2.14 | 2.14 | 512 |

#### Input 50 / Output 1 / Video 2 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 993.48 | 993.48 | 1.00 | 1.00 | 50 |
| NIM | FP8 | 1 | 884.77 | 884.77 | 1.13 | 1.13 | 30 |
| NIM | FP8 | 8 | 5101.77 | 5101.77 | 1.48 | 1.48 | 50 |
| NIM | FP8 | 16 | 10101.90 | 10101.90 | 1.46 | 1.46 | 80 |
| NIM | FP8 | 32 | 19585.65 | 19585.65 | 1.45 | 1.45 | 120 |
| vLLM | — | 64 | 55409.83 | 55409.83 | 1.05 | 1.05 | 320 |
| NIM | FP8 | 64 | 36972.80 | 36972.80 | 1.42 | 1.42 | 160 |
| vLLM | — | 128 | 92484.18 | 92484.18 | 1.05 | 1.05 | 256 |
| vLLM | — | 256 |  |  |  |  |  |

#### Input 50 / Output 1 / Video 4 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 1882.74 | 1882.74 | 0.53 | 0.53 | 30 |
| NIM | FP8 | 8 | 10560.76 | 10560.76 | 0.71 | 0.71 | 50 |
| NIM | FP8 | 16 | 20543.14 | 20543.14 | 0.71 | 0.71 | 80 |
| NIM | FP8 | 32 | 39570.98 | 39570.98 | 0.71 | 0.71 | 120 |
| NIM | FP8 | 64 | 73425.23 | 73425.23 | 0.71 | 0.71 | 160 |

#### Input 50 / Output 1 / Video 8 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 2720.75 | 2720.75 | 0.37 | 0.37 | 30 |
| NIM | FP8 | 8 | 15255.03 | 15255.03 | 0.49 | 0.49 | 50 |
| NIM | FP8 | 16 | 29918.19 | 29918.19 | 0.49 | 0.49 | 80 |
| NIM | FP8 | 32 | 57588.44 | 57588.44 | 0.49 | 0.49 | 120 |
| NIM | FP8 | 64 | 106111.36 | 106111.36 | 0.49 | 0.49 | 160 |

#### Input 50 / Output 100 / Video 1 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 508.97 | 3119.76 | 0.32 | 32.03 | 50 |
| NIM | FP8 | 1 | 478.57 | 1220.30 | 0.82 | 81.76 | 30 |
| NIM | FP8 | 8 | 1920.73 | 3801.64 | 2.04 | 204.36 | 50 |
| NIM | FP8 | 16 | 4017.22 | 7124.66 | 2.12 | 211.76 | 80 |
| NIM | FP8 | 32 | 8115.67 | 14244.73 | 2.19 | 219.20 | 120 |
| vLLM | — | 64 | 26567.65 | 39090.16 | 1.54 | 153.87 | 320 |
| NIM | FP8 | 64 | 14820.28 | 26867.21 | 2.17 | 216.28 | 160 |
| vLLM | — | 128 | 54861.82 | 67203.81 | 1.55 | 154.58 | 256 |
| vLLM | — | 256 | 116733.31 | 129435.49 | 1.54 | 154.04 | 512 |

#### Input 50 / Output 100 / Video 2 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 999.10 | 3638.23 | 0.27 | 27.46 | 50 |
| NIM | FP8 | 1 | 881.00 | 1809.27 | 0.55 | 55.14 | 30 |
| NIM | FP8 | 8 | 3103.08 | 6746.28 | 1.14 | 114.20 | 50 |
| NIM | FP8 | 16 | 6763.75 | 13083.30 | 1.19 | 119.35 | 80 |
| NIM | FP8 | 32 | 12907.57 | 26057.90 | 1.22 | 121.83 | 120 |
| vLLM | — | 64 | 49069.05 | 59084.57 | 1.00 | 100.14 | 320 |
| NIM | FP8 | 64 | 22161.72 | 49737.74 | 1.25 | 125.08 | 160 |
| vLLM | — | 128 | 116178.34 | 128875.27 | 0.77 | 77.38 | 256 |
| vLLM | — | 256 |  |  |  |  |  |

#### Input 50 / Output 100 / Video 4 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 1869.75 | 2962.01 | 0.34 | 33.67 | 30 |
| NIM | FP8 | 8 | 6616.18 | 13089.52 | 0.59 | 58.41 | 50 |
| NIM | FP8 | 16 | 11244.22 | 25711.51 | 0.61 | 60.69 | 80 |
| NIM | FP8 | 32 | 18354.09 | 49039.72 | 0.64 | 63.68 | 120 |
| NIM | FP8 | 64 | 60872.65 | 92696.03 | 0.63 | 62.81 | 160 |

#### Input 50 / Output 100 / Video 8 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 2740.64 | 3895.23 | 0.26 | 25.56 | 30 |
| NIM | FP8 | 8 | 9090.06 | 18431.00 | 0.43 | 42.60 | 50 |
| NIM | FP8 | 16 | 17269.04 | 36088.31 | 0.44 | 43.76 | 80 |
| NIM | FP8 | 32 | 36803.38 | 69545.27 | 0.44 | 44.10 | 120 |
| NIM | FP8 | 64 | 95063.85 | 128105.55 | 0.44 | 44.20 | 160 |

### H200 NVL

#### Input 50 / Output 1 / Video 1 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 357.69 | 357.69 | 2.78 | 2.78 | 50 |
| NIM | FP8 | 1 | 415.24 | 415.24 | 2.40 | 2.40 | 30 |
| NIM | FP8 | 8 | 2043.23 | 2043.23 | 3.73 | 3.73 | 50 |
| NIM | FP8 | 16 | 4065.31 | 4065.31 | 3.72 | 3.72 | 80 |
| NIM | FP8 | 32 | 7570.39 | 7570.39 | 3.81 | 3.81 | 120 |
| vLLM | — | 64 | 16243.11 | 16243.11 | 3.56 | 3.56 | 319 |
| NIM | FP8 | 64 | 14873.85 | 14873.85 | 3.67 | 3.67 | 160 |
| vLLM | — | 128 | 26759.35 | 26759.35 | 3.58 | 3.58 | 254 |
| vLLM | — | 256 | 54470.31 | 54470.31 | 3.53 | 3.53 | 510 |

#### Input 50 / Output 1 / Video 2 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 641.05 | 641.05 | 1.56 | 1.56 | 50 |
| NIM | FP8 | 1 | 752.90 | 752.90 | 1.32 | 1.32 | 30 |
| NIM | FP8 | 8 | 3741.71 | 3741.71 | 2.03 | 2.03 | 50 |
| NIM | FP8 | 16 | 7156.14 | 7156.14 | 2.05 | 2.05 | 80 |
| NIM | FP8 | 32 | 14171.53 | 14171.53 | 1.99 | 1.99 | 120 |
| vLLM | — | 64 | 33640.68 | 33640.68 | 1.72 | 1.72 | 320 |
| NIM | FP8 | 64 | 27889.86 | 27889.86 | 1.89 | 1.89 | 160 |
| vLLM | — | 128 | 56090.59 | 56090.59 | 1.72 | 1.72 | 255 |
| vLLM | — | 256 | 111965.21 | 111965.21 | 1.71 | 1.71 | 510 |

#### Input 50 / Output 1 / Video 4 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 1515.40 | 1515.40 | 0.66 | 0.66 | 30 |
| NIM | FP8 | 8 | 7717.69 | 7717.69 | 0.97 | 0.97 | 50 |
| NIM | FP8 | 16 | 15335.47 | 15335.47 | 0.95 | 0.95 | 80 |
| NIM | FP8 | 32 | 30179.55 | 30179.55 | 0.93 | 0.93 | 120 |
| NIM | FP8 | 64 | 57173.59 | 57173.59 | 0.91 | 0.91 | 160 |

#### Input 50 / Output 1 / Video 8 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 2223.77 | 2223.77 | 0.45 | 0.45 | 30 |
| NIM | FP8 | 8 | 11712.64 | 11712.64 | 0.64 | 0.64 | 50 |
| NIM | FP8 | 16 | 23245.94 | 23245.94 | 0.63 | 0.63 | 80 |
| NIM | FP8 | 32 | 45678.38 | 45678.38 | 0.61 | 0.61 | 120 |
| NIM | FP8 | 64 | 84811.26 | 84811.26 | 0.61 | 0.61 | 160 |

#### Input 50 / Output 100 / Video 1 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 348.88 | 2385.95 | 0.42 | 41.87 | 50 |
| NIM | FP8 | 1 | 427.27 | 1103.48 | 0.90 | 90.33 | 30 |
| NIM | FP8 | 8 | 1985.05 | 3303.28 | 2.37 | 236.92 | 50 |
| NIM | FP8 | 16 | 3716.73 | 5997.93 | 2.62 | 261.61 | 80 |
| NIM | FP8 | 32 | 6697.30 | 10946.22 | 2.72 | 271.88 | 120 |
| vLLM | — | 64 | 5805.63 | 21240.62 | 3.01 | 300.56 | 320 |
| NIM | FP8 | 64 | 12076.58 | 20833.65 | 2.78 | 277.37 | 160 |
| vLLM | — | 128 | 16053.68 | 40187.13 | 3.06 | 305.57 | 256 |
| vLLM | — | 256 | 48411.16 | 75354.93 | 2.99 | 298.96 | 512 |

#### Input 50 / Output 100 / Video 2 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 640.14 | 2692.46 | 0.37 | 37.12 | 50 |
| NIM | FP8 | 1 | 762.70 | 1583.33 | 0.63 | 63.03 | 30 |
| NIM | FP8 | 8 | 2924.81 | 5290.10 | 1.44 | 143.82 | 50 |
| NIM | FP8 | 16 | 4781.70 | 9598.67 | 1.60 | 159.66 | 80 |
| NIM | FP8 | 32 | 10762.46 | 19240.50 | 1.57 | 156.99 | 120 |
| vLLM | — | 64 | 13800.40 | 41460.80 | 1.52 | 151.97 | 320 |
| NIM | FP8 | 64 | 14228.56 | 35599.40 | 1.62 | 161.53 | 160 |
| vLLM | — | 128 | 47514.83 | 74683.53 | 1.51 | 151.42 | 256 |
| vLLM | — | 256 | 110513.13 | 138991.66 | 1.52 | 152.07 | 512 |

#### Input 50 / Output 100 / Video 4 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 1518.16 | 2453.99 | 0.41 | 40.62 | 30 |
| NIM | FP8 | 8 | 5197.99 | 10100.80 | 0.79 | 78.60 | 50 |
| NIM | FP8 | 16 | 7135.34 | 18928.23 | 0.83 | 83.26 | 80 |
| NIM | FP8 | 32 | 17005.30 | 37835.14 | 0.83 | 82.99 | 120 |
| NIM | FP8 | 64 | 39854.66 | 73670.34 | 0.82 | 81.94 | 160 |

#### Input 50 / Output 100 / Video 8 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 2222.35 | 3345.14 | 0.30 | 29.83 | 30 |
| NIM | FP8 | 8 | 7754.43 | 14174.09 | 0.54 | 53.93 | 50 |
| NIM | FP8 | 16 | 14627.15 | 28167.89 | 0.57 | 56.42 | 80 |
| NIM | FP8 | 32 | 20887.21 | 54580.37 | 0.56 | 55.82 | 120 |
| NIM | FP8 | 64 | 56735.02 | 109609.70 | 0.55 | 55.31 | 160 |

### H100 80GB HBM3 (SXM)

#### Input 50 / Output 1 / Video 1 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 460.03 | 460.03 | 2.16 | 2.16 | 30 |
| NIM | FP8 | 8 | 2058.81 | 2058.81 | 3.68 | 3.68 | 50 |
| NIM | FP8 | 16 | 3804.90 | 3804.90 | 3.91 | 3.91 | 80 |
| NIM | FP8 | 32 | 7802.58 | 7802.58 | 3.74 | 3.74 | 120 |
| NIM | FP8 | 64 | 15257.30 | 15257.30 | 3.64 | 3.64 | 160 |

#### Input 50 / Output 1 / Video 2 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 835.19 | 835.19 | 1.19 | 1.19 | 30 |
| NIM | FP8 | 8 | 3524.90 | 3524.90 | 2.15 | 2.15 | 50 |
| NIM | FP8 | 16 | 6726.85 | 6726.85 | 2.19 | 2.19 | 80 |
| NIM | FP8 | 32 | 13413.96 | 13413.96 | 2.13 | 2.13 | 120 |
| NIM | FP8 | 64 | 25767.12 | 25767.12 | 2.08 | 2.08 | 160 |

#### Input 50 / Output 1 / Video 4 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 1456.89 | 1456.89 | 0.68 | 0.68 | 30 |
| NIM | FP8 | 8 | 6959.44 | 6959.44 | 1.07 | 1.07 | 50 |
| NIM | FP8 | 16 | 13517.56 | 13517.56 | 1.08 | 1.08 | 80 |
| NIM | FP8 | 32 | 26129.88 | 26129.88 | 1.08 | 1.08 | 120 |
| NIM | FP8 | 64 | 48952.46 | 48952.46 | 1.07 | 1.07 | 160 |

#### Input 50 / Output 1 / Video 8 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 2163.80 | 2163.80 | 0.46 | 0.46 | 30 |
| NIM | FP8 | 8 | 10255.24 | 10255.24 | 0.73 | 0.73 | 50 |
| NIM | FP8 | 16 | 19935.25 | 19935.25 | 0.73 | 0.73 | 80 |
| NIM | FP8 | 32 | 38563.87 | 38563.87 | 0.73 | 0.73 | 120 |
| NIM | FP8 | 64 | 72741.63 | 72741.63 | 0.72 | 0.72 | 160 |

#### Input 50 / Output 100 / Video 1 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 463.60 | 1240.64 | 0.80 | 80.42 | 30 |
| NIM | FP8 | 8 | 2097.28 | 3442.65 | 2.20 | 219.53 | 50 |
| NIM | FP8 | 16 | 3916.73 | 6146.71 | 2.57 | 257.01 | 80 |
| NIM | FP8 | 32 | 7134.59 | 11186.72 | 2.69 | 268.41 | 120 |
| NIM | FP8 | 64 | 12312.40 | 21412.87 | 2.86 | 286.21 | 160 |

#### Input 50 / Output 100 / Video 2 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 804.78 | 1779.56 | 0.56 | 56.07 | 30 |
| NIM | FP8 | 8 | 3051.80 | 5458.86 | 1.39 | 139.21 | 50 |
| NIM | FP8 | 16 | 5835.07 | 9859.19 | 1.54 | 154.29 | 80 |
| NIM | FP8 | 32 | 7243.81 | 17684.19 | 1.78 | 177.65 | 120 |
| NIM | FP8 | 64 | 20719.42 | 33628.89 | 1.78 | 177.33 | 160 |

#### Input 50 / Output 100 / Video 4 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 1465.55 | 2713.20 | 0.37 | 36.77 | 30 |
| NIM | FP8 | 8 | 4656.57 | 9327.28 | 0.83 | 82.74 | 50 |
| NIM | FP8 | 16 | 4624.26 | 16638.32 | 0.95 | 94.74 | 80 |
| NIM | FP8 | 32 | 18992.31 | 32150.19 | 0.95 | 94.40 | 120 |
| NIM | FP8 | 64 | 47940.43 | 60909.22 | 0.93 | 92.41 | 160 |

#### Input 50 / Output 100 / Video 8 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 2179.16 | 3471.71 | 0.29 | 28.77 | 30 |
| NIM | FP8 | 8 | 7322.19 | 13169.59 | 0.58 | 58.00 | 50 |
| NIM | FP8 | 16 | 11430.80 | 24349.15 | 0.64 | 63.90 | 80 |
| NIM | FP8 | 32 | 33871.22 | 47147.32 | 0.63 | 62.95 | 120 |
| NIM | FP8 | 64 | 71457.63 | 86033.50 | 0.64 | 64.01 | 160 |

### H200 141GB HBM3

#### Input 50 / Output 1 / Video 1 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 327.21 | 327.21 | 3.04 | 3.04 | 50 |
| NIM | FP8 | 1 | 437.69 | 437.69 | 2.27 | 2.27 | 30 |
| NIM | FP8 | 8 | 1980.54 | 1980.54 | 3.84 | 3.84 | 50 |
| NIM | FP8 | 16 | 3814.23 | 3814.23 | 3.95 | 3.95 | 80 |
| NIM | FP8 | 32 | 7192.42 | 7192.42 | 4.02 | 4.02 | 120 |
| vLLM | — | 64 | 14045.01 | 14045.01 | 4.14 | 4.14 | 320 |
| NIM | FP8 | 64 | 14708.70 | 14708.70 | 3.78 | 3.78 | 160 |
| vLLM | — | 128 | 23809.00 | 23809.00 | 4.09 | 4.09 | 256 |
| vLLM | — | 256 | 46893.25 | 46893.25 | 4.08 | 4.08 | 507 |

#### Input 50 / Output 1 / Video 2 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 592.19 | 592.19 | 1.68 | 1.68 | 50 |
| NIM | FP8 | 1 | 768.01 | 768.01 | 1.29 | 1.29 | 30 |
| NIM | FP8 | 8 | 3365.54 | 3365.54 | 2.25 | 2.25 | 50 |
| NIM | FP8 | 16 | 6626.61 | 6626.61 | 2.23 | 2.23 | 80 |
| NIM | FP8 | 32 | 12979.60 | 12979.60 | 2.21 | 2.21 | 120 |
| vLLM | — | 64 | 28769.68 | 28769.68 | 2.01 | 2.01 | 320 |
| NIM | FP8 | 64 | 24985.28 | 24985.28 | 2.15 | 2.15 | 160 |
| vLLM | — | 128 | 48595.04 | 48595.04 | 1.99 | 1.99 | 256 |
| vLLM | — | 256 | 95884.55 | 95884.55 | 2.01 | 2.01 | 512 |

#### Input 50 / Output 1 / Video 4 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 1469.42 | 1469.42 | 0.68 | 0.68 | 30 |
| NIM | FP8 | 8 | 6863.53 | 6863.53 | 1.09 | 1.09 | 50 |
| NIM | FP8 | 16 | 13483.40 | 13483.40 | 1.08 | 1.08 | 80 |
| NIM | FP8 | 32 | 25838.27 | 25838.27 | 1.09 | 1.09 | 120 |
| NIM | FP8 | 64 | 48861.54 | 48861.54 | 1.07 | 1.07 | 160 |

#### Input 50 / Output 1 / Video 8 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 2141.97 | 2141.97 | 0.47 | 0.47 | 30 |
| NIM | FP8 | 8 | 10274.00 | 10274.00 | 0.73 | 0.73 | 50 |
| NIM | FP8 | 16 | 19962.51 | 19962.51 | 0.73 | 0.73 | 80 |
| NIM | FP8 | 32 | 38433.78 | 38433.78 | 0.73 | 0.73 | 120 |
| NIM | FP8 | 64 | 72346.17 | 72346.17 | 0.73 | 0.73 | 160 |

#### Input 50 / Output 100 / Video 1 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 327.73 | 2254.07 | 0.44 | 44.31 | 50 |
| NIM | FP8 | 1 | 442.52 | 1099.72 | 0.91 | 90.75 | 30 |
| NIM | FP8 | 8 | 1873.39 | 3151.53 | 2.52 | 252.05 | 50 |
| NIM | FP8 | 16 | 3856.29 | 5864.72 | 2.71 | 270.51 | 80 |
| NIM | FP8 | 32 | 6594.23 | 10343.67 | 2.89 | 288.39 | 120 |
| vLLM | — | 64 | 5374.55 | 18553.56 | 3.44 | 344.02 | 320 |
| NIM | FP8 | 64 | 12846.71 | 20154.03 | 2.89 | 287.88 | 160 |
| vLLM | — | 128 | 14328.53 | 35613.84 | 3.44 | 344.42 | 256 |
| vLLM | — | 256 | 42108.40 | 65558.20 | 3.43 | 343.29 | 512 |

#### Input 50 / Output 100 / Video 2 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 592.97 | 2533.93 | 0.39 | 39.43 | 50 |
| NIM | FP8 | 1 | 780.88 | 1530.80 | 0.65 | 65.19 | 30 |
| NIM | FP8 | 8 | 3023.73 | 5139.24 | 1.50 | 150.04 | 50 |
| NIM | FP8 | 16 | 5471.43 | 9413.38 | 1.67 | 167.27 | 80 |
| NIM | FP8 | 32 | 9243.30 | 17395.21 | 1.74 | 173.55 | 120 |
| vLLM | — | 64 | 11969.34 | 36021.00 | 1.75 | 174.83 | 320 |
| NIM | FP8 | 64 | 19866.20 | 34045.79 | 1.72 | 171.27 | 160 |
| vLLM | — | 128 | 41372.57 | 65053.92 | 1.74 | 173.72 | 256 |
| vLLM | — | 256 | 95995.33 | 120751.01 | 1.75 | 174.85 | 512 |

#### Input 50 / Output 100 / Video 4 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 1430.03 | 2289.23 | 0.44 | 43.35 | 30 |
| NIM | FP8 | 8 | 5452.50 | 9461.08 | 0.81 | 81.05 | 50 |
| NIM | FP8 | 16 | 7624.47 | 16926.73 | 0.92 | 91.65 | 80 |
| NIM | FP8 | 32 | 15305.73 | 32149.35 | 0.94 | 93.78 | 120 |
| NIM | FP8 | 64 | 39727.31 | 62037.59 | 0.97 | 96.65 | 160 |

#### Input 50 / Output 100 / Video 8 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 2139.08 | 3184.35 | 0.31 | 31.25 | 30 |
| NIM | FP8 | 8 | 7196.67 | 12801.29 | 0.60 | 59.48 | 50 |
| NIM | FP8 | 16 | 13294.30 | 24287.50 | 0.64 | 64.12 | 80 |
| NIM | FP8 | 32 | 18665.48 | 45776.67 | 0.66 | 65.78 | 120 |
| NIM | FP8 | 64 | 48398.35 | 90907.86 | 0.67 | 66.64 | 160 |

### B200

#### Input 50 / Output 1 / Video 1 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 212.10 | 212.10 | 4.68 | 4.68 | 50 |
| NIM | FP8 | 1 | 340.92 | 340.92 | 2.90 | 2.90 | 30 |
| NIM | NVFP4 | 1 | 272.13 | 272.13 | 3.62 | 3.62 | 30 |
| NIM | FP8 | 8 | 1659.95 | 1659.95 | 4.57 | 4.57 | 50 |
| NIM | NVFP4 | 8 | 1144.10 | 1144.10 | 6.68 | 6.68 | 50 |
| NIM | FP8 | 16 | 3261.37 | 3261.37 | 4.58 | 4.58 | 80 |
| NIM | NVFP4 | 16 | 2267.90 | 2267.90 | 6.69 | 6.69 | 80 |
| NIM | FP8 | 32 | 6430.24 | 6430.24 | 4.51 | 4.51 | 120 |
| NIM | NVFP4 | 32 | 4526.50 | 4526.50 | 6.62 | 6.62 | 120 |
| vLLM | — | 64 | 6902.51 | 6902.51 | 8.41 | 8.41 | 320 |
| NIM | FP8 | 64 | 12546.18 | 12546.18 | 4.39 | 4.39 | 160 |
| NIM | NVFP4 | 64 | 9045.86 | 9045.86 | 6.40 | 6.40 | 160 |
| vLLM | — | 128 | 11412.35 | 11412.35 | 8.52 | 8.52 | 256 |
| vLLM | — | 256 | 22707.11 | 22707.11 | 8.52 | 8.52 | 512 |

#### Input 50 / Output 1 / Video 2 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 350.30 | 350.30 | 2.84 | 2.84 | 50 |
| NIM | FP8 | 1 | 653.69 | 653.69 | 1.52 | 1.52 | 30 |
| NIM | NVFP4 | 1 | 463.22 | 463.22 | 2.14 | 2.14 | 30 |
| NIM | FP8 | 8 | 2892.41 | 2892.41 | 2.61 | 2.61 | 50 |
| NIM | NVFP4 | 8 | 1733.84 | 1733.84 | 4.38 | 4.38 | 50 |
| NIM | FP8 | 16 | 5862.31 | 5862.31 | 2.53 | 2.53 | 80 |
| NIM | NVFP4 | 16 | 3440.28 | 3440.28 | 4.34 | 4.34 | 80 |
| NIM | FP8 | 32 | 10903.75 | 10903.75 | 2.64 | 2.64 | 120 |
| NIM | NVFP4 | 32 | 6690.53 | 6690.53 | 4.34 | 4.34 | 120 |
| vLLM | — | 64 | 13909.70 | 13909.70 | 4.16 | 4.16 | 320 |
| NIM | FP8 | 64 | 19380.11 | 19380.11 | 2.79 | 2.79 | 160 |
| NIM | NVFP4 | 64 | 12953.43 | 12953.43 | 4.28 | 4.28 | 160 |
| vLLM | — | 128 | 23275.71 | 23275.71 | 4.16 | 4.16 | 256 |
| vLLM | — | 256 | 46780.29 | 46780.29 | 4.10 | 4.10 | 510 |

#### Input 50 / Output 1 / Video 4 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 1242.78 | 1242.78 | 0.80 | 0.80 | 30 |
| NIM | NVFP4 | 1 | 894.79 | 894.79 | 1.11 | 1.11 | 30 |
| NIM | FP8 | 8 | 5487.43 | 5487.43 | 1.37 | 1.37 | 50 |
| NIM | NVFP4 | 8 | 3217.83 | 3217.83 | 2.34 | 2.34 | 50 |
| NIM | FP8 | 16 | 10305.89 | 10305.89 | 1.43 | 1.43 | 80 |
| NIM | NVFP4 | 16 | 6113.43 | 6113.43 | 2.41 | 2.41 | 80 |
| NIM | FP8 | 32 | 18957.64 | 18957.64 | 1.49 | 1.49 | 120 |
| NIM | NVFP4 | 32 | 12737.17 | 12737.17 | 2.24 | 2.24 | 120 |
| NIM | FP8 | 64 | 35583.23 | 35583.23 | 1.48 | 1.48 | 160 |
| NIM | NVFP4 | 64 | 23833.53 | 23833.53 | 2.24 | 2.24 | 160 |

#### Input 50 / Output 1 / Video 8 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 1865.04 | 1865.04 | 0.53 | 0.53 | 30 |
| NIM | NVFP4 | 1 | 1279.09 | 1279.09 | 0.78 | 0.78 | 30 |
| NIM | FP8 | 8 | 8643.57 | 8643.57 | 0.87 | 0.87 | 50 |
| NIM | NVFP4 | 8 | 4705.93 | 4705.93 | 1.60 | 1.60 | 50 |
| NIM | FP8 | 16 | 16703.76 | 16703.76 | 0.88 | 0.88 | 80 |
| NIM | NVFP4 | 16 | 9346.42 | 9346.42 | 1.58 | 1.58 | 80 |
| NIM | FP8 | 32 | 32864.78 | 32864.78 | 0.86 | 0.86 | 120 |
| NIM | NVFP4 | 32 | 18391.48 | 18391.48 | 1.53 | 1.53 | 120 |
| NIM | FP8 | 64 | 59836.19 | 59836.19 | 0.88 | 0.88 | 160 |
| NIM | NVFP4 | 64 | 33242.15 | 33242.15 | 1.60 | 1.60 | 160 |

#### Input 50 / Output 100 / Video 1 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 212.40 | 1552.87 | 0.64 | 64.30 | 50 |
| NIM | FP8 | 1 | 345.43 | 860.21 | 1.16 | 115.87 | 30 |
| NIM | NVFP4 | 1 | 274.71 | 672.49 | 1.48 | 148.28 | 30 |
| NIM | FP8 | 8 | 1642.76 | 2545.74 | 2.97 | 296.31 | 50 |
| NIM | NVFP4 | 8 | 1137.22 | 1873.08 | 4.07 | 406.63 | 50 |
| NIM | FP8 | 16 | 3152.39 | 4845.41 | 3.29 | 328.51 | 80 |
| NIM | NVFP4 | 16 | 2182.58 | 3272.66 | 4.80 | 480.25 | 80 |
| NIM | FP8 | 32 | 5346.05 | 8494.35 | 3.48 | 347.82 | 120 |
| NIM | NVFP4 | 32 | 4055.73 | 5880.42 | 5.19 | 518.09 | 120 |
| vLLM | — | 64 | 2723.57 | 9594.94 | 6.65 | 664.83 | 320 |
| NIM | FP8 | 64 | 10244.50 | 16621.96 | 3.49 | 348.76 | 160 |
| NIM | NVFP4 | 64 | 7719.72 | 10954.36 | 5.33 | 532.59 | 160 |
| vLLM | — | 128 | 5574.27 | 17572.41 | 7.21 | 721.29 | 256 |
| vLLM | — | 256 | 16228.42 | 34293.88 | 6.97 | 696.84 | 512 |

#### Input 50 / Output 100 / Video 2 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 325.21 | 1686.82 | 0.59 | 59.21 | 50 |
| NIM | FP8 | 1 | 651.94 | 1294.02 | 0.77 | 77.10 | 30 |
| NIM | NVFP4 | 1 | 487.44 | 1004.08 | 0.99 | 99.18 | 30 |
| NIM | FP8 | 8 | 2453.12 | 4091.45 | 1.86 | 186.15 | 50 |
| NIM | NVFP4 | 8 | 1587.30 | 3107.57 | 2.45 | 159.96 | 50 |
| NIM | FP8 | 16 | 4804.59 | 7951.28 | 1.98 | 197.76 | 80 |
| NIM | NVFP4 | 16 | 2968.54 | 4863.06 | 3.27 | 326.45 | 80 |
| NIM | FP8 | 32 | 8127.58 | 14438.91 | 2.07 | 193.61 | 120 |
| NIM | NVFP4 | 32 | 5840.94 | 9103.46 | 3.24 | 323.12 | 120 |
| vLLM | — | 64 | 3821.22 | 17970.15 | 3.55 | 354.77 | 320 |
| NIM | FP8 | 64 | 15680.15 | 30640.16 | 2.04 | 130.43 | 160 |
| NIM | NVFP4 | 64 | 10825.16 | 17300.70 | 3.28 | 248.01 | 160 |
| vLLM | — | 128 | 15872.01 | 34042.27 | 3.52 | 352.10 | 256 |
| vLLM | — | 256 | 42120.75 | 61444.78 | 3.60 | 360.15 | 512 |

#### Input 50 / Output 100 / Video 4 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 1262.78 | 1970.33 | 0.51 | 50.58 | 30 |
| NIM | NVFP4 | 1 | 883.21 | 1435.68 | 0.70 | 69.50 | 30 |
| NIM | FP8 | 8 | 4124.87 | 7394.77 | 1.03 | 73.59 | 50 |
| NIM | NVFP4 | 8 | 2633.14 | 4877.56 | 1.63 | 117.77 | 50 |
| NIM | FP8 | 16 | 7947.67 | 14115.86 | 1.08 | 44.47 | 80 |
| NIM | NVFP4 | 16 | 5692.33 | 9391.80 | 1.63 | 67.14 | 80 |
| NIM | FP8 | 32 | 7718.62 | 23743.08 | 1.23 | 25.01 | 120 |
| NIM | NVFP4 | 32 | 11245.43 | 18452.96 | 1.70 | 34.49 | 120 |
| NIM | FP8 | 64 | 28627.33 | 51007.23 | 1.20 | 15.59 | 160 |
| NIM | NVFP4 | 64 | 18219.26 | 32747.63 | 1.88 | 24.46 | 160 |

#### Input 50 / Output 100 / Video 8 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 1858.31 | 2632.57 | 0.38 | 37.92 | 30 |
| NIM | NVFP4 | 1 | 1309.22 | 1921.75 | 0.52 | 51.94 | 30 |
| NIM | FP8 | 8 | 5996.80 | 10572.47 | 0.73 | 72.47 | 50 |
| NIM | NVFP4 | 8 | 3562.03 | 6323.58 | 1.20 | 119.31 | 50 |
| NIM | FP8 | 16 | 10855.50 | 19966.80 | 0.80 | 79.57 | 80 |
| NIM | NVFP4 | 16 | 6674.12 | 11599.89 | 1.31 | 130.74 | 80 |
| NIM | FP8 | 32 | 12221.64 | 39012.47 | 0.81 | 80.55 | 120 |
| NIM | NVFP4 | 32 | 14486.34 | 22751.70 | 1.29 | 128.64 | 120 |
| NIM | FP8 | 64 | 46003.13 | 74078.22 | 0.80 | 80.07 | 160 |
| NIM | NVFP4 | 64 | 41034.44 | 48643.48 | 1.23 | 122.66 | 160 |

### B300

#### Input 50 / Output 1 / Video 1 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 176.24 | 176.24 | 5.64 | 5.64 | 50 |
| NIM | FP8 | 1 | 369.26 | 369.26 | 2.68 | 2.68 | 30 |
| NIM | NVFP4 | 1 | 317.31 | 317.31 | 3.13 | 3.13 | 30 |
| NIM | FP8 | 8 | 1483.57 | 1483.57 | 5.10 | 5.10 | 50 |
| NIM | NVFP4 | 8 | 1288.17 | 1288.17 | 5.91 | 5.91 | 50 |
| NIM | FP8 | 16 | 2962.05 | 2962.05 | 5.03 | 5.03 | 80 |
| NIM | NVFP4 | 16 | 2403.98 | 2403.98 | 6.31 | 6.31 | 80 |
| NIM | FP8 | 32 | 6006.73 | 6006.73 | 4.85 | 4.85 | 120 |
| NIM | NVFP4 | 32 | 4999.81 | 4999.81 | 5.82 | 5.82 | 120 |
| vLLM | — | 64 | 6665.86 | 6665.86 | 8.71 | 8.71 | 320 |
| NIM | FP8 | 64 | 11854.70 | 11854.70 | 4.69 | 4.69 | 160 |
| NIM | NVFP4 | 64 | 9820.14 | 9820.14 | 5.68 | 5.68 | 160 |
| vLLM | — | 128 | 11233.68 | 11233.68 | 8.67 | 8.67 | 256 |
| vLLM | — | 256 | 22238.90 | 22238.90 | 8.65 | 8.65 | 510 |

#### Input 50 / Output 1 / Video 2 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 301.49 | 301.49 | 3.31 | 3.31 | 50 |
| NIM | FP8 | 1 | 676.86 | 676.86 | 1.47 | 1.47 | 30 |
| NIM | NVFP4 | 1 | 540.11 | 540.11 | 1.84 | 1.84 | 30 |
| NIM | FP8 | 8 | 2738.84 | 2738.84 | 2.77 | 2.77 | 50 |
| NIM | NVFP4 | 8 | 2083.51 | 2083.51 | 3.60 | 3.60 | 50 |
| NIM | FP8 | 16 | 5270.22 | 5270.22 | 2.80 | 2.80 | 80 |
| NIM | NVFP4 | 16 | 4036.32 | 4036.32 | 3.63 | 3.63 | 80 |
| NIM | FP8 | 32 | 10215.50 | 10215.50 | 2.80 | 2.80 | 120 |
| NIM | NVFP4 | 32 | 8442.80 | 8442.80 | 3.36 | 3.36 | 120 |
| vLLM | — | 64 | 13570.83 | 13570.83 | 4.26 | 4.26 | 320 |
| NIM | FP8 | 64 | 20022.89 | 20022.89 | 2.69 | 2.69 | 160 |
| NIM | NVFP4 | 64 | 15402.19 | 15402.19 | 3.50 | 3.50 | 160 |
| vLLM | — | 128 | 22688.87 | 22688.87 | 4.26 | 4.26 | 255 |
| vLLM | — | 256 | 45276.68 | 45276.68 | 4.26 | 4.26 | 512 |

#### Input 50 / Output 1 / Video 4 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 1360.80 | 1360.80 | 0.73 | 0.73 | 30 |
| NIM | NVFP4 | 1 | 1099.68 | 1099.68 | 0.90 | 0.90 | 30 |
| NIM | FP8 | 8 | 5270.29 | 5270.29 | 1.42 | 1.42 | 50 |
| NIM | NVFP4 | 8 | 3974.45 | 3974.45 | 1.88 | 1.88 | 50 |
| NIM | FP8 | 16 | 10252.09 | 10252.09 | 1.43 | 1.43 | 80 |
| NIM | NVFP4 | 16 | 8440.91 | 8440.91 | 1.72 | 1.72 | 80 |
| NIM | FP8 | 32 | 19740.94 | 19740.94 | 1.43 | 1.43 | 120 |
| NIM | NVFP4 | 32 | 14567.13 | 14567.13 | 1.98 | 1.98 | 120 |
| NIM | FP8 | 64 | 36784.84 | 36784.84 | 1.43 | 1.43 | 160 |
| NIM | NVFP4 | 64 | 28374.37 | 28374.37 | 1.84 | 1.84 | 160 |

#### Input 50 / Output 1 / Video 8 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 2026.22 | 2026.22 | 0.49 | 0.49 | 30 |
| NIM | NVFP4 | 1 | 1481.76 | 1481.76 | 0.67 | 0.67 | 30 |
| NIM | FP8 | 8 | 7901.60 | 7901.60 | 0.95 | 0.95 | 50 |
| NIM | NVFP4 | 8 | 6578.33 | 6578.33 | 1.13 | 1.13 | 50 |
| NIM | FP8 | 16 | 15125.53 | 15125.53 | 0.97 | 0.97 | 80 |
| NIM | NVFP4 | 16 | 12234.80 | 12234.80 | 1.19 | 1.19 | 80 |
| NIM | FP8 | 32 | 29548.23 | 29548.23 | 0.96 | 0.96 | 120 |
| NIM | NVFP4 | 32 | 23407.31 | 23407.31 | 1.19 | 1.19 | 120 |
| NIM | FP8 | 64 | 56588.92 | 56588.92 | 0.93 | 0.93 | 160 |
| NIM | NVFP4 | 64 | 47764.11 | 47764.11 | 1.09 | 1.09 | 160 |

#### Input 50 / Output 100 / Video 1 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 175.83 | 1492.46 | 0.67 | 66.93 | 50 |
| NIM | FP8 | 1 | 370.12 | 881.81 | 1.13 | 113.12 | 30 |
| NIM | NVFP4 | 1 | 333.86 | 733.79 | 1.36 | 135.91 | 30 |
| NIM | FP8 | 8 | 1507.69 | 2500.99 | 3.17 | 317.19 | 50 |
| NIM | NVFP4 | 8 | 1133.91 | 1804.86 | 4.17 | 415.95 | 50 |
| NIM | FP8 | 16 | 2885.51 | 4432.29 | 3.44 | 344.33 | 80 |
| NIM | NVFP4 | 16 | 2502.90 | 3492.14 | 4.48 | 447.15 | 80 |
| NIM | FP8 | 32 | 6104.65 | 8838.80 | 3.58 | 357.42 | 120 |
| NIM | NVFP4 | 32 | 5480.19 | 6581.84 | 4.70 | 469.66 | 120 |
| vLLM | — | 64 | 2491.56 | 9254.00 | 6.89 | 689.29 | 320 |
| NIM | FP8 | 64 | 12061.58 | 17144.64 | 3.66 | 365.48 | 160 |
| NIM | NVFP4 | 64 | 11499.13 | 12641.63 | 4.71 | 470.34 | 160 |
| vLLM | — | 128 | 4999.71 | 17203.60 | 7.37 | 736.64 | 256 |
| vLLM | — | 256 | 8916.09 | 33189.36 | 7.61 | 761.12 | 512 |

#### Input 50 / Output 100 / Video 2 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | — | 1 | 303.52 | 1637.02 | 0.61 | 61.02 | 50 |
| NIM | FP8 | 1 | 637.82 | 1231.59 | 0.81 | 81.09 | 30 |
| NIM | NVFP4 | 1 | 535.20 | 1042.24 | 0.96 | 95.51 | 30 |
| NIM | FP8 | 8 | 2392.39 | 4055.64 | 1.88 | 187.75 | 50 |
| NIM | NVFP4 | 8 | 1623.68 | 3052.85 | 2.50 | 171.86 | 50 |
| NIM | FP8 | 16 | 4314.39 | 7348.30 | 2.17 | 216.38 | 80 |
| NIM | NVFP4 | 16 | 3853.79 | 5830.97 | 2.64 | 128.87 | 80 |
| NIM | FP8 | 32 | 8283.16 | 13736.23 | 2.16 | 216.20 | 120 |
| NIM | NVFP4 | 32 | 8812.95 | 10967.68 | 2.68 | 97.10 | 120 |
| vLLM | — | 64 | 3655.15 | 17223.78 | 3.70 | 370.09 | 320 |
| NIM | FP8 | 64 | 15608.72 | 26277.18 | 2.24 | 223.39 | 160 |
| NIM | NVFP4 | 64 | 19112.95 | 21342.81 | 2.66 | 82.26 | 160 |
| vLLM | — | 128 | 8924.49 | 33088.08 | 3.82 | 382.18 | 256 |
| vLLM | — | 256 | 30494.72 | 62798.50 | 3.83 | 383.31 | 512 |

#### Input 50 / Output 100 / Video 4 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 1288.11 | 1929.17 | 0.52 | 51.61 | 30 |
| NIM | NVFP4 | 1 | 1096.57 | 1658.17 | 0.60 | 60.18 | 30 |
| NIM | FP8 | 8 | 4063.94 | 6987.32 | 1.09 | 108.38 | 50 |
| NIM | NVFP4 | 8 | 3322.59 | 4715.98 | 1.63 | 162.89 | 50 |
| NIM | FP8 | 16 | 7378.50 | 13128.94 | 1.21 | 120.48 | 80 |
| NIM | NVFP4 | 16 | 7503.06 | 9045.63 | 1.64 | 163.65 | 80 |
| NIM | FP8 | 32 | 13346.93 | 24134.32 | 1.23 | 123.23 | 120 |
| NIM | NVFP4 | 32 | 18348.21 | 19465.50 | 1.46 | 146.03 | 120 |
| NIM | FP8 | 64 | 32905.66 | 47954.85 | 1.25 | 124.80 | 160 |
| NIM | NVFP4 | 64 | 29285.15 | 31467.85 | 1.76 | 175.47 | 160 |

#### Input 50 / Output 100 / Video 8 FPS

| Backend | Precision | Concurrency | Time To First Token (ms) | Request Latency (ms) | Request Throughput (Req/s) | Output Token Throughput (Tok/s) | Request Count (requests) |
|---|---|---:|---:|---:|---:|---:|---:|
| NIM | FP8 | 1 | 1808.94 | 2618.74 | 0.38 | 37.99 | 30 |
| NIM | NVFP4 | 1 | 1428.32 | 2034.36 | 0.49 | 48.97 | 30 |
| NIM | FP8 | 8 | 5873.78 | 10060.45 | 0.76 | 75.61 | 50 |
| NIM | NVFP4 | 8 | 5619.32 | 7222.29 | 1.05 | 104.42 | 50 |
| NIM | FP8 | 16 | 10417.50 | 18411.82 | 0.85 | 84.46 | 80 |
| NIM | NVFP4 | 16 | 12413.83 | 14003.19 | 1.06 | 106.17 | 80 |
| NIM | FP8 | 32 | 18783.03 | 35169.14 | 0.85 | 84.43 | 120 |
| NIM | NVFP4 | 32 | 25775.43 | 27277.41 | 1.06 | 105.47 | 120 |
| NIM | FP8 | 64 | 38989.71 | 70687.86 | 0.86 | 85.48 | 160 |
| NIM | NVFP4 | 64 | 47731.88 | 49550.98 | 1.10 | 109.34 | 160 |

<sub>Notes:
1. Source: vLLM inference benchmarking for `nvidia/Cosmos3-Super`; AIPerf client was used as the benchmarking tool for the vLLM results.
2. Hardware: results are grouped by GPU product (RTX PRO 6000 Blackwell, H20, H100 NVL, H200 NVL, H100 80GB HBM3 SXM, H200 141GB HBM3, B200, B300). Latency metrics are request averages; throughput metrics are aggregate rates. Request counts indicate measurement volume.
3. **Time To First Token (TTFT)** measures latency until the first output token is emitted. **Request Latency** is end-to-end time per request. For single-token outputs (Output 1), TTFT and request latency are identical.
4. **Request Throughput** is completed requests per second. **Output Token Throughput** is generated tokens per second (for Output 1 workloads, the two throughputs match).
5. Concurrency is the number of simultaneous client requests, not tensor-parallel GPU count. The vLLM requests were issued by AIPerf.
6. NIM FP8 and NVFP4 are separate profiles. The published vLLM baseline does not specify precision or establish identical runtime settings.</sub>
