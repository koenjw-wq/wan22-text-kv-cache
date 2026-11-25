"""Verify cached projections against ComfyUI's installed Wan cross-attention path."""

import argparse
import json
import sys
from pathlib import Path

import torch

from benchmark import load_layer


def check(comfy_root, checkpoint, block):
    sys.path.insert(0, str(comfy_root))
    from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention
    from comfy.ldm.wan.model import WanT2VCrossAttention

    device = torch.device("cuda", 0)
    dtype = torch.bfloat16
    torch.manual_seed(3090)
    with torch.inference_mode():
        weights = load_layer(checkpoint, block, device, dtype)
        width = weights["q.weight"].shape[0]
        layer = WanT2VCrossAttention(
            width, width // 128,
            operation_settings={"operations": torch.nn, "device": device, "dtype": dtype},
        ).eval()
        layer.load_state_dict(weights)
        context = torch.randn((1, 512, width), device=device, dtype=dtype)
        key = layer.norm_k(layer.k(context))
        value = layer.v(context)
        projector_snapshots = key.clone(), value.clone()
        for tokens in (512, 2048, 8192):
            x = torch.randn((1, tokens, width), device=device, dtype=dtype)
            expected = layer(x, context, transformer_options={})
            query = layer.norm_q(layer.q(x))
            result = layer.o(optimized_attention(
                AttentionTensorContainer(query), AttentionTensorContainer(key), AttentionTensorContainer(value),
                heads=layer.num_heads, preferred_attention=layer.comfy_attention, transformer_options={},
            ))
            if not torch.isfinite(result).all() or not torch.equal(result, expected):
                raise RuntimeError(f"ComfyUI outputs differed at {tokens} query tokens.")
            if not all(torch.equal(original, current) for original, current in zip(projector_snapshots, (key, value))):
                raise RuntimeError("The attention implementation modified the cached projections.")
            print(json.dumps({"block": block, "query_tokens": tokens, "bitwise_equal": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comfy_root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--block", type=int, default=14)
    args = parser.parse_args()
    check(args.comfy_root, args.checkpoint, args.block)
