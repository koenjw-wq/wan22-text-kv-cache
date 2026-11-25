# Wan 2.2 text K/V caching on RTX 3090

A per-render cache reduced median ComfyUI denoising time by about **4.25%** on two RTX 3090s with CPU weight streaming enabled. All 17 decoded frames matched bit for bit.

| GPU | Stock | Cache | Less denoising time |
| --- | ---: | ---: | ---: |
| RTX 3090 #0 | 9.8449 s | 9.4271 s | 4.24% |
| RTX 3090 #1 | 10.2025 s | 9.7675 s | 4.26% |

[Full report (PDF)](Prometheus-Wan22-findings.pdf) · [GPU 0 raw results](results/render-512-gpu0.jsonl) · [GPU 1 raw results](results/render-512-gpu1.jsonl)

## How it works

For fixed text conditioning and model weights, each cross-attention layer's text key/value projections can be reused across denoising steps. The runner stores those projections in pinned system memory. On a hit it uploads the smaller projected tensors and skips staging and evaluating the K/V projection weights. Query projections, attention, and the output projection still run every step.

The cache lasts for one sampling call. It compares exact conditioning snapshots, stores entries per layer, and limits cached projections to 256 MiB. The tested runs used 252 MiB and recorded 105 hits per job on layers with offloaded K/V weights. The runner temporarily wraps ComfyUI methods within its Python process. Closing the runner leaves the installed ComfyUI files intact.

## Reproduce the A/B benchmark

**1. Prepare ComfyUI.** Use an existing ComfyUI environment on an NVIDIA GPU with BF16 support. The runner requires at least 12 GiB of free VRAM before model load and was tested on 24 GiB cards. The reference setup used PyTorch 2.13.0+cu130, NVIDIA driver 595.99.02, and ComfyUI revision [`20ca544`](https://github.com/Comfy-Org/ComfyUI/commit/20ca544ee0436721d8eb5f544665e490609f72c8). If you use a separate checkout, install its dependencies using [ComfyUI's installation instructions](https://github.com/Comfy-Org/ComfyUI#installing). Set the path to that installation and Python environment.

```bash
COMFY="/absolute/path/to/ComfyUI"
PYTHON="$COMFY/.venv/bin/python"
```

**2. Place the model files.** Use these three files from [Comfy-Org's Wan 2.2 model repository](https://huggingface.co/Comfy-Org/Wan-2.2).

| File | Path relative to ComfyUI |
| --- | --- |
| [Wan 2.2 TI2V 5B FP16](https://huggingface.co/Comfy-Org/Wan-2.2/blob/main/split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors) | `models/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors` |
| [UMT5 XXL text encoder](https://huggingface.co/Comfy-Org/Wan-2.2/blob/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors) | `models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors` |
| [Wan 2.2 VAE](https://huggingface.co/Comfy-Org/Wan-2.2/blob/main/split_files/vae/wan2.2_vae.safetensors) | `models/vae/wan2.2_vae.safetensors` |

**3. Run from this repository.** Free the target GPU from any other workload first. Set `CUDA_VISIBLE_DEVICES=1` to repeat on a second card. One GPU is sufficient to run the A/B.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
  "$PYTHON" -B render_ab.py "$COMFY" \
  --steps 6 --warmups 1 --pairs 3 --width 512 --height 288 --frames 17 \
  > run-gpu0.jsonl
```

The runner selects BF16, Euler/simple, CFG 3.5, seed 3090, and forces CPU weight streaming with `--novram --disable-dynamic-vram --async-offload 2`. It warms up the stock path once, alternates test order, and emits one JSON line per measurement. The final line gives both medians, the percentage reduction, the frame digest, and `"bitwise_equal_frames": true` if the check passes. The run fails if the cache recorded no hits on CPU-offloaded K/V weights or if the stock and cached outputs differ.

## Interpreting the numbers

The timed region covers denoising and cache setup. Text encoding and VAE decoding run outside the timer. The reference hardware used an existing BF16 modification on both benchmark paths. The report covers this checkpoint and workload with forced weight streaming. Speedups with GPU-resident weights, dynamic VRAM, LoRAs, longer clips, and other resolutions can differ. Use the runner to measure your own setup.

The two optional scripts exercise an isolated layer. `benchmark.py` compares bytes and timing with different weight placements. `check_comfy_block.py` tests numerical equality against the installed ComfyUI Wan layer. Both accept `--help`.

## Credits and license

GPT-6.2 Sol (Codex) designed, implemented and tested the experiment. Local Qwen 3.8 27B reviewed the approach. The runner adapts the attention call from [ComfyUI's Wan implementation](https://github.com/Comfy-Org/ComfyUI/blob/20ca544ee0436721d8eb5f544665e490609f72c8/comfy/ldm/wan/model.py). The benchmark code is distributed under [GPL-3.0](LICENSE). Model weights are distributed by their respective upstream projects.

Exact text K/V reuse also appears in [Diffusers](https://huggingface.co/docs/diffusers/main/api/cache#diffusers.TextKVCacheConfig). ComfyUI has a separate [autoregressive Wan cache](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/wan/ar_model.py).

## Related Project Prometheus research

[Host-staged BF16 all-reduce on two RTX 3090s](https://github.com/koenjw-wq/rtx3090-host-staged-allreduce) — a separate CUDA communication experiment reporting 23–26% less execution time than the installed NCCL path for large standalone BF16 collectives, reproduced on single-socket and dual-socket systems. End-to-end Qwen improvements remain unmeasured. These results are separate from the Wan denoising benchmark above.
