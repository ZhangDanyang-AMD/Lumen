###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################
"""
Tests for deferred weight gradient computation.

Covers:
  - _DeferredWgrad unit tests (defer, execute, accumulate, main_grad)
  - Integration with LumenRowParallelLinear (execute_deferred_wgrad API)
  - Integration with LumenColumnParallelLinear (execute_deferred_wgrad API)
"""

from types import SimpleNamespace
from unittest import mock

import torch

# =========================================================================
# Unit tests for _DeferredWgrad
# =========================================================================


class TestDeferredWgrad:

    def test_defer_and_execute(self):
        from lumen.modules.parallel_linear import _DeferredWgrad

        dw = _DeferredWgrad()
        weight = torch.nn.Parameter(torch.randn(4, 4))
        weight.grad = torch.zeros(4, 4)
        grad_val = torch.ones(4, 4)

        dw.defer(weight, lambda: grad_val)
        assert dw.has_pending

        dw.execute()
        assert not dw.has_pending
        torch.testing.assert_close(weight.grad, grad_val)

    def test_defer_into_main_grad(self):
        from lumen.modules.parallel_linear import _DeferredWgrad

        dw = _DeferredWgrad()
        weight = torch.nn.Parameter(torch.randn(4, 4))
        weight.main_grad = torch.zeros(4, 4)
        grad_val = torch.ones(4, 4) * 2

        dw.defer(weight, lambda: grad_val)
        dw.execute()
        torch.testing.assert_close(weight.main_grad, grad_val)

    def test_no_pending_execute_is_noop(self):
        from lumen.modules.parallel_linear import _DeferredWgrad

        dw = _DeferredWgrad()
        dw.execute()  # should not raise
        assert not dw.has_pending

    def test_accumulates_on_existing_grad(self):
        from lumen.modules.parallel_linear import _DeferredWgrad

        dw = _DeferredWgrad()
        weight = torch.nn.Parameter(torch.randn(4, 4))
        weight.grad = torch.ones(4, 4)

        dw.defer(weight, lambda: torch.ones(4, 4) * 3)
        dw.execute()
        torch.testing.assert_close(weight.grad, torch.ones(4, 4) * 4)

    def test_defer_creates_grad_if_none(self):
        from lumen.modules.parallel_linear import _DeferredWgrad

        dw = _DeferredWgrad()
        weight = torch.nn.Parameter(torch.randn(4, 4))
        assert weight.grad is None

        grad_val = torch.ones(4, 4)
        dw.defer(weight, lambda: grad_val)
        dw.execute()
        torch.testing.assert_close(weight.grad, grad_val)

    def test_double_defer_overwrites(self):
        from lumen.modules.parallel_linear import _DeferredWgrad

        dw = _DeferredWgrad()
        weight = torch.nn.Parameter(torch.randn(4, 4))
        weight.grad = torch.zeros(4, 4)

        dw.defer(weight, lambda: torch.ones(4, 4))
        dw.defer(weight, lambda: torch.ones(4, 4) * 5)
        dw.execute()
        torch.testing.assert_close(weight.grad, torch.ones(4, 4) * 5)


# =========================================================================
# Integration with parallel linear modules
# =========================================================================


def _make_config(sequence_parallel=False):
    return SimpleNamespace(
        params_dtype=torch.bfloat16,
        perform_initialization=False,
        use_cpu_initialization=False,
        sequence_parallel=sequence_parallel,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
        lumen_tp_comm_overlap=False,
    )


_PARALLEL_LINEAR_PATCHES = [
    mock.patch("lumen.modules.parallel_linear._get_tp_group", return_value=None),
    mock.patch("lumen.modules.parallel_linear._pg_size", return_value=1),
    mock.patch("lumen.modules.parallel_linear._pg_rank", return_value=0),
    mock.patch("lumen.modules.parallel_linear._use_sdma_from_args", return_value=False),
    mock.patch("lumen.modules.parallel_linear.divide", side_effect=lambda a, b: a // b),
    mock.patch("lumen.modules.parallel_linear._initialize_affine_weight_gpu"),
    mock.patch("lumen.modules.parallel_linear.set_tensor_model_parallel_attributes"),
    mock.patch("lumen.modules.parallel_linear.make_sharded_tensors_for_checkpoint", return_value={}),
    mock.patch("lumen.modules.parallel_linear.copy_to_tensor_model_parallel_region", side_effect=lambda x, **kw: x),
    mock.patch("lumen.modules.parallel_linear.gather_from_sequence_parallel_region", side_effect=lambda x, **kw: x),
    mock.patch("lumen.modules.parallel_linear.gather_from_tensor_model_parallel_region", side_effect=lambda x, **kw: x),
    mock.patch("lumen.modules.parallel_linear.reduce_from_tensor_model_parallel_region", side_effect=lambda x, **kw: x),
    mock.patch(
        "lumen.modules.parallel_linear.reduce_scatter_to_sequence_parallel_region", side_effect=lambda x, **kw: x
    ),
]


def _apply_patches(func):
    """Apply all parallel linear patches to a function."""
    for p in reversed(_PARALLEL_LINEAR_PATCHES):
        func = p(func)
    return func


class TestDeferredWgradModuleIntegration:
    """Test that parallel linear modules expose the execute_deferred_wgrad API."""

    @_apply_patches
    def test_row_parallel_has_deferred_wgrad(self, *_):
        from lumen.modules.parallel_linear import LumenRowParallelLinear

        config = _make_config()
        m = LumenRowParallelLinear(
            64,
            64,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
            input_is_parallel=True,
        )
        assert hasattr(m, "_deferred_wgrad")
        assert hasattr(m, "execute_deferred_wgrad")
        assert callable(m.execute_deferred_wgrad)
        assert not m._deferred_wgrad.has_pending

    @_apply_patches
    def test_column_parallel_has_deferred_wgrad(self, *_):
        from lumen.modules.parallel_linear import LumenColumnParallelLinear

        config = _make_config()
        m = LumenColumnParallelLinear(
            64,
            128,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
        )
        assert hasattr(m, "_deferred_wgrad")
        assert hasattr(m, "execute_deferred_wgrad")

    @_apply_patches
    def test_execute_deferred_wgrad_noop_when_no_pending(self, *_):
        from lumen.modules.parallel_linear import LumenRowParallelLinear

        config = _make_config()
        m = LumenRowParallelLinear(
            64,
            64,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
            input_is_parallel=True,
        )
        m.execute_deferred_wgrad()  # should not raise

    @_apply_patches
    def test_manual_defer_and_execute_via_module(self, *_):
        """Manually defer a wgrad and execute it via the module API."""
        from lumen.modules.parallel_linear import LumenRowParallelLinear

        config = _make_config()
        m = LumenRowParallelLinear(
            64,
            64,
            config=config,
            init_method=lambda w: torch.nn.init.kaiming_uniform_(w),
            input_is_parallel=True,
        )
        torch.nn.init.kaiming_uniform_(m.weight)
        m.weight.grad = torch.zeros_like(m.weight)
        grad_val = torch.ones_like(m.weight)

        m._deferred_wgrad.defer(m.weight, lambda: grad_val)
        assert m._deferred_wgrad.has_pending

        m.execute_deferred_wgrad()
        assert not m._deferred_wgrad.has_pending
        torch.testing.assert_close(m.weight.grad, grad_val)
