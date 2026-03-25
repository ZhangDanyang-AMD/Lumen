from pathlib import Path

import pytest
import torch

import benchmarks.bench_wgrad_delay as bench_wgrad_delay
from benchmarks.bench_wgrad_delay import (
    _make_bench_linear_config,
    _validate_megatron_style_world_size,
)


def test_make_bench_linear_config_tier1_defaults():
    cfg = _make_bench_linear_config(sequence_parallel=False, tp_size=1)

    assert cfg.params_dtype is torch.bfloat16
    assert cfg.perform_initialization is False
    assert cfg.use_cpu_initialization is False
    assert cfg.sequence_parallel is False
    assert cfg.tensor_model_parallel_size == 1
    assert cfg.expert_model_parallel_size == 1
    assert cfg.lumen_tp_comm_overlap is False


def test_make_bench_linear_config_tier2_matches_tp_size():
    cfg = _make_bench_linear_config(sequence_parallel=True, tp_size=8)

    assert cfg.sequence_parallel is True
    assert cfg.tensor_model_parallel_size == 8
    assert cfg.expert_model_parallel_size == 1


def test_validate_megatron_style_world_size_accepts_two_ranks():
    _validate_megatron_style_world_size(2)


def test_validate_megatron_style_world_size_rejects_single_rank():
    with pytest.raises(ValueError, match="world_size >= 2"):
        _validate_megatron_style_world_size(1)


def _distributed_wgrad_delay_block_source() -> str:
    """Source slice from RealComm through NCCL/SDMA/profile sections (Tier 1–2 overlap paths).

    Stops before ``TestGradAccumFusionQuantizedLinear``; excludes Tier 0 single-GPU
    ``_DeferredWgrad`` classes and grad-acc fusion.
    """
    text = Path(bench_wgrad_delay.__file__).read_text()
    start = text.index("class TestDeferredWgradRealComm")
    end = text.index("class TestGradAccumFusionQuantizedLinear")
    return text[start:end]


def test_tier1_real_api_helpers_exist():
    assert hasattr(bench_wgrad_delay, "_get_or_create_rank_local_group")
    assert hasattr(bench_wgrad_delay, "_build_real_api_single_layer")
    assert hasattr(bench_wgrad_delay, "_build_real_api_layer_stack")
    assert hasattr(bench_wgrad_delay, "_run_real_api_backward_then_queue")


def test_distributed_wgrad_delay_block_avoids_legacy_defer_weight_api():
    block = _distributed_wgrad_delay_block_source()

    assert "dwg.defer(" not in block
    assert "dwg2.defer(" not in block


def test_megatron_style_helpers_exist():
    assert hasattr(bench_wgrad_delay, "_build_megatron_style_stack")
    assert hasattr(bench_wgrad_delay, "_run_megatron_style_layerwise_backward")


def test_megatron_style_benchmark_class_exists():
    assert hasattr(bench_wgrad_delay, "TestMegatronStyleWgradDelay")


def test_bench_wgrad_delay_docstring_mentions_megatron_style_selector():
    source = Path(bench_wgrad_delay.__file__).read_text()

    assert "-k MegatronStyle" in source


def test_bench_wgrad_delay_docstring_describes_tiers():
    source = Path(bench_wgrad_delay.__file__).read_text()

    assert "Tier 0" in source
    assert "Tier 1" in source
    assert "Tier 2" in source


def test_benchmarks_readme_mentions_real_api_and_megatron_style():
    root = Path(__file__).resolve().parents[2]
    readme = (root / "benchmarks" / "README.md").read_text()

    assert "Tier 0" in readme
    assert "Tier 1" in readme
    assert "Tier 2" in readme
    assert "MegatronStyle" in readme
