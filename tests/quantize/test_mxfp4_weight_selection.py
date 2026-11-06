"""MXFP4 Linear selection by fully-qualified weight name."""

import logging

import pytest
import torch.nn as nn

from lumen.config import LumenConfig
from lumen.quantize import _patch_linear_layers
from lumen.quantize.config import (
    MXFP4QuantWeightSelectionConfig,
    QuantConfig,
    QuantFormat,
    ScalingType,
)


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(8, 8)
        self.up_proj = nn.Linear(8, 8)
        self.down_proj = nn.Linear(8, 8)


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_a_proj = nn.Linear(8, 8)
        self.q_b_proj = nn.Linear(8, 8)
        self.kv_a_proj_with_mqa = nn.Linear(8, 8)
        self.kv_b_proj = nn.Linear(8, 8)
        self.o_proj = nn.Linear(8, 8)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention()
        self.mlp = _MLP()


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Block(), _Block()])
        self.fc = nn.Linear(8, 8)


def _config(
    *,
    mode: str = "all_linear",
    include: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
) -> QuantConfig:
    return QuantConfig(
        format=QuantFormat.MXFP4,
        scaling=ScalingType.BLOCKWISE,
        mxfp4_quant_weight_selection=MXFP4QuantWeightSelectionConfig(
            mode=mode,
            include_quant_weight=include,
            exclude_quant_weight=exclude,
        ),
    )


def _patch(model: nn.Module, config: QuantConfig) -> None:
    _patch_linear_layers(model, manager=None, backend="auto", config=config)


def _is_patched(module: nn.Module) -> bool:
    return getattr(module, "_quant_enabled", False)


def test_all_linear_preserves_existing_behavior() -> None:
    model = _Model()
    _patch(model, _config())

    assert all(_is_patched(module) for module in model.modules() if isinstance(module, nn.Linear))


def test_explicit_include_selects_only_matching_weights() -> None:
    model = _Model()
    _patch(
        model,
        _config(
            mode="explicit",
            include=("layers.*.mlp.gate_proj.weight",),
        ),
    )

    assert all(_is_patched(layer.mlp.gate_proj) for layer in model.layers)
    assert not any(_is_patched(layer.mlp.up_proj) for layer in model.layers)
    assert not _is_patched(model.fc)


def test_exclude_takes_priority_over_include() -> None:
    model = _Model()
    _patch(
        model,
        _config(
            mode="explicit",
            include=("layers.*.mlp.*.weight",),
            exclude=("layers.*.mlp.gate_proj.weight",),
        ),
    )

    assert not any(_is_patched(layer.mlp.gate_proj) for layer in model.layers)
    assert all(_is_patched(layer.mlp.up_proj) for layer in model.layers)
    assert all(_is_patched(layer.mlp.down_proj) for layer in model.layers)


def test_dspark_policy_selects_six_weights_per_layer() -> None:
    model = _Model()
    _patch(
        model,
        _config(
            mode="explicit",
            include=(
                "layers.*.mlp.gate_proj.weight",
                "layers.*.mlp.up_proj.weight",
                "layers.*.mlp.down_proj.weight",
                "layers.*.self_attn.q_b_proj.weight",
                "layers.*.self_attn.kv_b_proj.weight",
                "layers.*.self_attn.o_proj.weight",
            ),
            exclude=(
                "fc.weight",
                "layers.*.self_attn.q_a_proj.weight",
                "layers.*.self_attn.kv_a_proj_with_mqa.weight",
            ),
        ),
    )

    patched = [
        module._quant_tensor_id
        for module in model.modules()
        if _is_patched(module)
    ]
    assert len(patched) == 12
    assert not _is_patched(model.fc)
    assert not any(_is_patched(layer.self_attn.q_a_proj) for layer in model.layers)
    assert not any(
        _is_patched(layer.self_attn.kv_a_proj_with_mqa)
        for layer in model.layers
    )


def test_unmatched_include_raises_before_any_module_is_patched() -> None:
    model = _Model()

    with pytest.raises(ValueError, match="matched no eligible Linear weight"):
        _patch(
            model,
            _config(
                mode="explicit",
                include=("layers.*.self_attn.q_z_proj.weight",),
            ),
        )

    assert not any(_is_patched(module) for module in model.modules())


def test_include_matching_skipped_output_layer_raises() -> None:
    model = nn.Module()
    model.lm_head = nn.Linear(8, 8)

    with pytest.raises(
        ValueError, match="output/BF16-layer/LoRA/exclude skips"
    ):
        _patch(
            model,
            _config(
                mode="explicit",
                include=("lm_head.weight",),
            ),
        )

    assert not _is_patched(model.lm_head)


@pytest.mark.parametrize(
    ("pattern", "expect_warning", "fc_patched"),
    [
        ("does.not.exist.weight", True, True),
        ("fc.weight", False, False),
    ],
)
def test_exclude_match_logging(
    caplog, pattern: str, expect_warning: bool, fc_patched: bool,
) -> None:
    model = _Model()
    with caplog.at_level(logging.WARNING):
        _patch(model, _config(exclude=(pattern,)))

    warning = "exclude_quant_weight patterns matched no Linear weight"
    assert _is_patched(model.fc) is fc_patched
    assert (warning in caplog.text) is expect_warning


def test_selection_config_validation() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        MXFP4QuantWeightSelectionConfig(mode="explicit")
    with pytest.raises(ValueError, match="only valid"):
        MXFP4QuantWeightSelectionConfig(
            mode="all_linear",
            include_quant_weight=("layers.*.weight",),
        )
    with pytest.raises(ValueError, match="must be 'all_linear' or 'explicit'"):
        MXFP4QuantWeightSelectionConfig(mode="invalid")
    with pytest.raises(TypeError, match="not a single string"):
        MXFP4QuantWeightSelectionConfig(
            mode="explicit",
            include_quant_weight="layers.*.mlp.gate_proj.weight",
        )
    with pytest.raises(TypeError, match="not a single string"):
        MXFP4QuantWeightSelectionConfig(exclude_quant_weight="layers.0.*")


def test_lumen_config_passes_selection_to_quant_config() -> None:
    config = LumenConfig(
        format="mxfp4",
        scaling="blockwise",
        mxfp4_quant_weight_selection={
            "mode": "explicit",
            "include_quant_weight": ["layers.*.mlp.gate_proj.weight"],
        },
    )

    selection = config.quant_config.mxfp4_quant_weight_selection
    assert isinstance(selection, MXFP4QuantWeightSelectionConfig)
    assert selection.mode == "explicit"
    assert selection.include_quant_weight == ("layers.*.mlp.gate_proj.weight",)


def test_selection_is_rejected_for_non_mxfp4_format() -> None:
    selection = {
        "mode": "explicit",
        "include_quant_weight": ["layers.*.mlp.gate_proj.weight"],
    }

    with pytest.raises(ValueError, match="only be configured"):
        LumenConfig(
            format="fp8_e4m3",
            mxfp4_quant_weight_selection=selection,
        ).quant_config
    with pytest.raises(ValueError, match="only be configured"):
        QuantConfig(
            format=QuantFormat.FP8_E4M3,
            mxfp4_quant_weight_selection=selection,
        )


def test_selective_mxfp4_rejects_native_lumen_linear_path() -> None:
    with pytest.raises(ValueError, match="not supported together with lumen_linear"):
        LumenConfig(
            format="mxfp4",
            scaling="blockwise",
            lumen_linear=True,
            mxfp4_quant_weight_selection={
                "mode": "explicit",
                "include_quant_weight": ["layers.*.mlp.gate_proj.weight"],
            },
        )
