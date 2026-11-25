"""Run stock and cached Wan denoising with ComfyUI's CPU weight streaming."""

import argparse
import gc
import hashlib
import json
import logging
import statistics
import sys
import time
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import torch


@dataclass
class Conditioning:
    snapshot: torch.Tensor
    projections: dict = field(default_factory=dict)


@contextmanager
def cache_cross_attention(model):
    from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention
    from comfy.ldm.wan.model import WanModel, WanT2VCrossAttention

    if type(model) is not WanModel:
        raise TypeError("This experiment supports the plain WanModel only.")
    blocks = [block.cross_attn for block in model.blocks]
    if any(type(layer) is not WanT2VCrossAttention for layer in blocks):
        raise TypeError("This experiment supports vanilla Wan text cross-attention only.")

    conditions = []
    active = None
    stats = {"hits": 0, "misses": 0, "bytes_cached": 0, "offloaded_kv_hits": 0}
    max_bytes = 256 * 1024**2
    original_model = model.forward_orig

    def scoped_forward(x, timestep, context, *args, **kwargs):
        nonlocal active
        options = kwargs.get("transformer_options") or {}
        eligible = (
            torch.is_inference_mode_enabled() and context.is_cuda and
            not args and kwargs.get("clip_fea") is None and
            not options.get("patches") and not options.get("patches_replace") and
            not options.get("wrappers")
        )
        if eligible:
            for item in conditions:
                previous = item.snapshot
                if (context.device == previous.device and context.dtype == previous.dtype and
                        context.shape == previous.shape and torch.equal(context, previous)):
                    active = item
                    break
            else:
                if len(conditions) < 4:
                    active = Conditioning(context.detach().clone())
                    conditions.append(active)
        try:
            return original_model(x, timestep, context, *args, **kwargs)
        finally:
            active = None

    def wrap_layer(layer, index):
        original = layer.forward

        def forward(x, context, transformer_options=None, **kwargs):
            if active is None:
                return original(x, context, transformer_options=transformer_options, **kwargs)
            entry = active.projections.get(index)
            query = layer.norm_q(layer.q(x))
            if entry is None:
                key = layer.norm_k(layer.k(context))
                value = layer.v(context)
                stats["misses"] += 1
                size = sum(t.numel() * t.element_size() for t in (key, value))
                if stats["bytes_cached"] + size <= max_bytes:
                    pinned = tuple(torch.empty_like(t, device="cpu", pin_memory=True) for t in (key, value))
                    for host, tensor in zip(pinned, (key, value)):
                        host.copy_(tensor, non_blocking=True)
                    ready = torch.cuda.Event()
                    ready.record()
                    active.projections[index] = pinned, ready
                    stats["bytes_cached"] += size
            else:
                pinned, ready = entry
                torch.cuda.current_stream(context.device).wait_event(ready)
                key, value = (tensor.to(device=context.device, non_blocking=True) for tensor in pinned)
                stats["hits"] += 1
                stats["offloaded_kv_hits"] += int(layer.k.weight.device.type == "cpu" and layer.v.weight.device.type == "cpu")
            return layer.o(optimized_attention(
                AttentionTensorContainer(query), AttentionTensorContainer(key), AttentionTensorContainer(value),
                heads=layer.num_heads, preferred_attention=layer.comfy_attention,
                transformer_options=transformer_options if transformer_options is not None else {},
            ))

        return forward

    with ExitStack() as scope:
        scope.enter_context(patch.object(model, "forward_orig", scoped_forward))
        for index, layer in enumerate(blocks):
            scope.enter_context(patch.object(layer, "forward", wrap_layer(layer, index)))
        try:
            yield stats
        finally:
            torch.cuda.synchronize()


def digest(tensor):
    data = tensor.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(data.numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comfy_root", type=Path)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--frames", type=int, default=17)
    args = parser.parse_args()
    if (args.steps < 2 or args.warmups < 0 or args.pairs < 1 or args.width < 32 or
            args.height < 32 or args.width % 32 or args.height % 32 or args.frames < 5 or args.frames % 4 != 1):
        parser.error("Need at least two steps, one pair, 32-aligned dimensions, and 4n+1 frames.")

    sys.path.insert(0, str(args.comfy_root))
    sys.argv = [sys.argv[0], "--disable-dynamic-vram", "--novram", "--async-offload", "2"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfy.model_management as mm
    import comfy.sample
    import comfy.sd
    import comfy.utils

    logging.basicConfig(level=logging.INFO)
    if torch.cuda.mem_get_info()[0] < 12 * 1024**3:
        raise RuntimeError("Expected a free GPU for the render. Stop Sonar before running this test.")
    models = args.comfy_root / "models"
    with torch.inference_mode():
        clip = comfy.sd.load_clip(
            ckpt_paths=[str(models / "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors")],
            clip_type=comfy.sd.CLIPType.WAN,
        )
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(
            "A small red sailboat crosses a quiet blue lake at dawn, fixed camera, gentle ripples."
        ))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize("flicker, distortion, unreadable text"))
        del clip
        mm.unload_all_models()
        gc.collect()
        mm.soft_empty_cache()
        model = comfy.sd.load_diffusion_model(
            str(models / "diffusion_models/wan2.2_ti2v_5B_fp16.safetensors"),
            model_options={"dtype": torch.bfloat16}, disable_dynamic=True,
        )
        latent = torch.zeros((1, 48, (args.frames - 1) // 4 + 1, args.height // 16, args.width // 16))
        noise = comfy.sample.prepare_noise(latent, 3090)

        def sample(use_cache):
            torch.cuda.synchronize()
            start = time.perf_counter()
            cache = cache_cross_attention(model.model.diffusion_model) if use_cache else nullcontext({})
            with cache as stats:
                result = comfy.sample.sample(
                    model, noise.clone(), args.steps, 3.5, "euler", "simple", positive, negative,
                    latent.clone(), disable_pbar=True, seed=3090,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if use_cache and not stats["offloaded_kv_hits"]:
                raise RuntimeError("The cache did not replace any host-offloaded K/V projections.")
            return result, elapsed, stats

        for _ in range(args.warmups):
            sample(False)
        baseline = candidate = None
        records = []
        for pair in range(args.pairs):
            for cached in (False, True) if pair % 2 == 0 else (True, False):
                result, elapsed, stats = sample(cached)
                if not torch.isfinite(result).all():
                    raise RuntimeError("Denoising returned non-finite values.")
                if cached:
                    candidate = result
                else:
                    baseline = result
                if baseline is not None and candidate is not None and not torch.equal(baseline, candidate):
                    raise RuntimeError(f"Latents differ, max absolute error {(baseline - candidate).abs().max().item()}")
                records.append({"cached": cached, "sampling_seconds": elapsed, "latent_sha256": digest(result), **stats})
                print(json.dumps(records[-1]), flush=True)

        mm.unload_all_models()
        del model
        gc.collect()
        mm.soft_empty_cache()
        vae_weights, metadata = comfy.utils.load_torch_file(
            str(models / "vae/wan2.2_vae.safetensors"), return_metadata=True,
        )
        vae = comfy.sd.VAE(sd=vae_weights, metadata=metadata)
        vae.throw_exception_if_invalid()
        del vae_weights
        original_frames = vae.decode(baseline)
        cached_frames = vae.decode(candidate)
        if not torch.isfinite(original_frames).all() or not torch.equal(original_frames, cached_frames):
            raise RuntimeError("Decoded video frames are non-finite or different.")
        stock = statistics.median(row["sampling_seconds"] for row in records if not row["cached"])
        optimized = statistics.median(row["sampling_seconds"] for row in records if row["cached"])
        print(json.dumps({
            "baseline_seconds": stock, "cache_seconds": optimized,
            "sampling_reduction_percent": 100 * (1 - optimized / stock),
            "frames_shape": list(original_frames.shape), "frame_sha256": digest(original_frames),
            "bitwise_equal_frames": True,
        }), flush=True)


if __name__ == "__main__":
    main()
