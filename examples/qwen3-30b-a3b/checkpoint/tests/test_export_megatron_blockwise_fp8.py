import pathlib
import sys
from collections import OrderedDict

import pytest
import torch

CHECKPOINT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CHECKPOINT_DIR))

from export_megatron_blockwise_fp8 import (  # noqa: E402
    BLOCK_SIZE,
    quantize_model_state,
    scale_key_for,
    should_quantize_weight,
)


@pytest.mark.parametrize(
    "name,shape,expected",
    [
        ("decoder.layers.0.self_attention.linear_qkv.weight", (5120, 2048), True),
        ("decoder.layers.0.self_attention.linear_proj.weight", (2048, 4096), True),
        (
            "decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight",
            (1536, 2048),
            True,
        ),
        (
            "decoder.layers.0.mlp.experts.local_experts.0.linear_fc2.weight",
            (2048, 768),
            True,
        ),
        ("embedding.word_embeddings.weight", (151936, 2048), False),
        ("output_layer.weight", (151936, 2048), False),
        ("decoder.layers.0.mlp.router.weight", (128, 2048), False),
        (
            "decoder.layers.0.self_attention.linear_qkv.layer_norm_weight",
            (2048,),
            False,
        ),
    ],
)
def test_qwen3_export_selection(name, shape, expected):
    tensor = torch.empty(shape, device="meta", dtype=torch.bfloat16)
    assert should_quantize_weight(name, tensor) is expected


def test_quantize_model_state_adds_dequant_scale_and_preserves_other_entries():
    model = OrderedDict(
        [
            ("decoder.layers.0.self_attention.linear_proj.weight", torch.ones(256, 128)),
            ("decoder.layers.0.mlp.router.weight", torch.ones(128, 128)),
            (
                "decoder.layers.0.self_attention.linear_proj._extra_state",
                torch.empty(0, dtype=torch.uint8),
            ),
        ]
    )

    def fake_quantize(weight):
        scale_shape = (weight.shape[0] // BLOCK_SIZE, weight.shape[1] // BLOCK_SIZE)
        return weight.to(torch.float8_e4m3fn), torch.full(scale_shape, 0.25)

    exported, names = quantize_model_state(model, fake_quantize)
    weight_key = "decoder.layers.0.self_attention.linear_proj.weight"

    assert names == [weight_key]
    assert exported[weight_key].dtype == torch.float8_e4m3fn
    torch.testing.assert_close(
        exported[scale_key_for(weight_key)], torch.full((2, 1), 0.25)
    )
    assert exported["decoder.layers.0.mlp.router.weight"].dtype == torch.float32
    assert (
        exported["decoder.layers.0.self_attention.linear_proj._extra_state"].numel()
        == 0
    )


def test_unaligned_linear_weight_is_rejected():
    model = {"decoder.layers.0.self_attention.linear_proj.weight": torch.ones(129, 128)}

    with pytest.raises(ValueError, match="not divisible"):
        quantize_model_state(model, lambda weight: (weight, torch.ones(1, 1)))


def test_scale_key_matches_official_weight_scale_inv():
    assert scale_key_for("decoder.layers.0.mlp.weight") == (
        "decoder.layers.0.mlp.weight_scale_inv"
    )
    with pytest.raises(ValueError, match=r"\.weight"):
        scale_key_for("decoder.layers.0.mlp.bias")
