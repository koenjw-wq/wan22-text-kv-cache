"""Measure exact text K/V reuse in one Wan 2.2 cross-attention block."""

import argparse
import json
import random
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


MiB = 1024 * 1024
MODES = ("recompute", "resident_cache", "offloaded_recompute", "host_cache")
PROJECTION_NAMES = ("k.weight", "k.bias", "norm_k.weight", "v.weight", "v.bias")


def load_layer(checkpoint, block, device, dtype):
    prefix = f"blocks.{block}.cross_attn."
    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        weights = {
            key.removeprefix(prefix): source.get_tensor(key).to(device=device, dtype=dtype)
            for key in source.keys()
            if key.startswith(prefix)
        }
    expected = {f"{name}.{kind}" for name in ("q", "k", "v", "o") for kind in ("weight", "bias")}
    expected.update({"norm_k.weight", "norm_q.weight"})
    if weights.keys() != expected:
        raise ValueError("Expected the dense, unquantized Wan cross-attention block.")
    return weights


def linear(x, weights, name):
    return F.linear(x, weights[f"{name}.weight"], weights[f"{name}.bias"])


def project_context(context, weights):
    key = F.rms_norm(linear(context, weights, "k"), (context.shape[-1],), weights["norm_k.weight"], eps=1e-6)
    return key, linear(context, weights, "v")


def attend(x, kv, weights):
    key, value = kv
    query = F.rms_norm(linear(x, weights, "q"), (x.shape[-1],), weights["norm_q.weight"], eps=1e-6)
    query, key, value = (t.unflatten(-1, (-1, 128)).transpose(1, 2) for t in (query, key, value))
    out = F.scaled_dot_product_attention(query, key, value)
    return linear(out.transpose(1, 2).flatten(-2), weights, "o")


def event_ms(work, iterations):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        work()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def benchmark(weights, device, dtype, tokens, context_tokens, samples):
    width = weights["q.weight"].shape[0]
    shape = (1, context_tokens, width)
    x = torch.randn((1, tokens, width), device=device, dtype=dtype)
    context = torch.randn(shape, device=device, dtype=dtype)
    resident = project_context(context, weights)
    pinned = tuple(t.cpu().pin_memory() for t in resident)
    staging = tuple(torch.empty_like(t) for t in resident)
    pinned_weights = {name: weights[name].cpu().pin_memory() for name in PROJECTION_NAMES}
    staged_weights = {name: torch.empty_like(weights[name]) for name in PROJECTION_NAMES}

    def recompute():
        return attend(x, project_context(context, weights), weights)

    def resident_cache():
        return attend(x, resident, weights)

    def offloaded_recompute():
        for name in PROJECTION_NAMES:
            staged_weights[name].copy_(pinned_weights[name], non_blocking=True)
        return attend(x, project_context(context, staged_weights), weights)

    def host_cache():
        for target, source in zip(staging, pinned):
            target.copy_(source, non_blocking=True)
        return attend(x, staging, weights)

    operations = dict(zip(MODES, (recompute, resident_cache, offloaded_recompute, host_cache)))
    expected = recompute()
    for name, operation in operations.items():
        result = operation()
        if not torch.isfinite(result).all() or not torch.equal(expected, result):
            raise RuntimeError(f"{name} produced a non-finite or different output.")
        for _ in range(4):
            operation()
    changed_context = torch.randn_like(context)
    changed = attend(x, project_context(changed_context, weights), weights)
    if torch.equal(expected, changed):
        raise RuntimeError("The changed-context control did not change the result.")
    del expected, result, changed
    torch.cuda.synchronize()

    times = {name: [] for name in MODES}
    shuffle = random.Random(17)
    for _ in range(samples):
        order = list(MODES)
        shuffle.shuffle(order)
        for name in order:
            times[name].append(event_ms(operations[name], 8))

    return {
        "query_tokens": tokens,
        "context_tokens": context_tokens,
        "width": width,
        "bitwise_equal": True,
        "different_context_changes_output": True,
        "cache_mib_per_layer": sum(t.numel() * t.element_size() for t in resident) / MiB,
        "offloaded_weights_mib_per_layer": sum(t.numel() * t.element_size() for t in pinned_weights.values()) / MiB,
        "projection_ms": event_ms(lambda: project_context(context, weights), 24),
        "samples_ms": times,
        "median_ms": {name: statistics.median(values) for name, values in times.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--tokens", type=int, nargs="+", default=[512, 2048, 8192])
    parser.add_argument("--context-tokens", type=int, default=512)
    parser.add_argument("--samples", type=int, default=7)
    args = parser.parse_args()
    device = torch.device("cuda", args.gpu)
    dtype = getattr(torch, args.dtype)
    torch.cuda.set_device(device)
    if torch.cuda.mem_get_info()[0] < 1024 * MiB:
        parser.error("This probe needs at least 1 GiB of free VRAM.")
    torch.manual_seed(3090)
    with torch.inference_mode():
        weights = load_layer(args.checkpoint, args.block, device, dtype)
        report = {
            "checkpoint": args.checkpoint.name,
            "gpu_index": args.gpu,
            "gpu_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "dtype": args.dtype,
            "block": args.block,
            "measurements": [
                benchmark(weights, device, dtype, tokens, args.context_tokens, args.samples)
                for tokens in args.tokens
            ],
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
