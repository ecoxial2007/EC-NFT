#!/usr/bin/env python3
"""Verify the Hugging Face Qwen3-VL eager-attention hook contract on CPU."""

import argparse

import torch
import transformers
from packaging.version import Version
from transformers import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextAttention,
    eager_attention_forward,
)


class DummyAttention:
    num_key_value_groups = 1
    training = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum-version", default="4.57.0.dev0")
    args = parser.parse_args()

    actual_version = str(transformers.__version__).split("+")[0]
    if Version(actual_version) < Version(args.minimum_version):
        raise RuntimeError(
            "This frozen release requires transformers>="
            f"{args.minimum_version}, found {transformers.__version__}. "
            "Install requirements.txt before running."
        )

    query = torch.randn(1, 2, 1, 4, dtype=torch.float32)
    key = torch.randn(1, 2, 3, 4, dtype=torch.float32)
    value = torch.randn(1, 2, 3, 4, dtype=torch.float32)
    output, attention = eager_attention_forward(
        DummyAttention(),
        query,
        key,
        value,
        attention_mask=None,
        scaling=0.5,
        dropout=0.0,
    )
    if output.shape != (1, 1, 2, 4):
        raise RuntimeError(f"unexpected eager attention output shape: {output.shape}")
    if not isinstance(attention, torch.Tensor) or attention.shape != (1, 2, 1, 3):
        raise RuntimeError(
            "Qwen3-VL eager attention did not return a 4-D attention tensor: "
            f"{type(attention)} {getattr(attention, 'shape', None)}"
        )

    config = Qwen3VLTextConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
    )
    config._attn_implementation = "eager"
    module = Qwen3VLTextAttention(config, layer_idx=0).eval()
    hidden = torch.randn(1, 3, 8, dtype=torch.float32)
    cos = torch.ones(1, 3, 4, dtype=torch.float32)
    sin = torch.zeros(1, 3, 4, dtype=torch.float32)
    module_output, module_attention = module(
        hidden_states=hidden,
        position_embeddings=(cos, sin),
        attention_mask=None,
        past_key_values=None,
    )
    if module_output.shape != (1, 3, 8):
        raise RuntimeError(
            f"unexpected Qwen3VLTextAttention output shape: {module_output.shape}"
        )
    if not isinstance(module_attention, torch.Tensor) or module_attention.shape != (
        1,
        2,
        3,
        3,
    ):
        raise RuntimeError(
            "Qwen3VLTextAttention.forward no longer exposes attn_weights as "
            "its second 4-D module output; the rollout hook is incompatible: "
            f"{type(module_attention)} {getattr(module_attention, 'shape', None)}"
        )

    print(
        "[TRANSFORMERS-CHECK] "
        f"version={transformers.__version__}; eager attention returns "
        f"{tuple(attention.shape)} and Qwen3VLTextAttention exposes "
        f"{tuple(module_attention.shape)}"
    )


if __name__ == "__main__":
    main()
