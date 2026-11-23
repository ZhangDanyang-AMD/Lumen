"""IMPORT-phase patch: run tiny ``@jit_fuser`` regions of Megatron eagerly.

``jit_fuser`` is ``torch.compile``.  For graphs that lower to one or two kernels
the Dynamo guard check and AOT runtime wrappers cost ~100-150 us of host time per
call, several times the eager cost, and the MoE forward is host-bound.  Two such
regions run on every layer:

* ``bias_dropout_add_fused_train`` with no bias and ``p == 0``: the graph is a
  single ``residual + x``.  Eager bf16/fp32 addition rounds the same way as the
  generated Triton kernel, so the result is bit-identical.
* ``MoEFlexTokenDispatcher.dispatch_preprocess``: views, ``topk`` (Inductor falls
  back to ``aten.topk``) and integer masking; eager is the reference semantics.

``LUMEN_EAGER_TINY_REGIONS=0`` keeps the compiled versions.
"""

from __future__ import annotations

import logging
import os

from lumen.patches.registry import PatchPhase, register_patch

logger = logging.getLogger(__name__)


def _uncompiled(fn):
    return getattr(fn, "_torchdynamo_orig_callable", None) or getattr(fn, "__wrapped__", None)


def install_eager_bias_dropout_add() -> None:
    try:
        import megatron.core.fusions.fused_bias_dropout as bda
    except ImportError:
        return
    if getattr(bda, "_lumen_eager_bda_patched", False):
        return

    compiled = bda.bias_dropout_add_fused_train

    def bias_dropout_add_fused_train(x_with_bias, residual, prob):
        x, bias = x_with_bias
        if bias is None and prob == 0.0 and x.dtype == residual.dtype:
            return residual + x
        return compiled(x_with_bias, residual, prob)

    bda.bias_dropout_add_fused_train = bias_dropout_add_fused_train
    bda._lumen_eager_bda_patched = True
    logger.info("Megatron bias-dropout-add runs eagerly when it is a plain residual add")


def install_eager_flex_dispatch_preprocess() -> None:
    try:
        from megatron.core.transformer.moe.token_dispatcher import MoEFlexTokenDispatcher
    except ImportError:
        return
    if getattr(MoEFlexTokenDispatcher, "_lumen_eager_preprocess_patched", False):
        return
    eager = _uncompiled(MoEFlexTokenDispatcher.dispatch_preprocess)
    if eager is None:
        return
    MoEFlexTokenDispatcher.dispatch_preprocess = eager
    MoEFlexTokenDispatcher._lumen_eager_preprocess_patched = True
    logger.info("Megatron flex-dispatcher preprocess runs eagerly")


def install_eager_tiny_regions() -> None:
    install_eager_bias_dropout_add()
    install_eager_flex_dispatch_preprocess()


register_patch(
    "eager_tiny_regions",
    PatchPhase.IMPORT,
    description="Run single-kernel Megatron jit_fuser regions eagerly",
    enabled=lambda: os.environ.get("LUMEN_EAGER_TINY_REGIONS", "1") == "1",
    tags=frozenset({"core", "megatron"}),
)(install_eager_tiny_regions)
