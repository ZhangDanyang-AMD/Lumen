"""Exercise native Wgrad dispatch/fallback without importing GPU kernels."""

import importlib.util
import logging
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def op(monkeypatch):
    dispatch = ModuleType("lumen.ops.dispatch")
    dispatch.Backend = type(
        "Backend", (), {"TRITON": "triton", "GLUON": "gluon", "OPUS": "opus"}
    )
    for name in (
        "pack_pre_routed_weights",
        "pre_routed_w4a16",
        "pre_routed_a4w4",
        "pre_routed_a4w4_bf16_main_grad",
        "pre_routed_a4w4_wgrad",
        "pre_routed_a4w4_wgrad_gluon",
        "pre_routed_a4w4_wgrad_opus",
        "pre_routed_a8w4",
        "pre_routed_a8w4f",
        "pre_routed_mapped",
    ):
        setattr(dispatch, "_probe_aiter_" + name, lambda: None)
    dispatch.try_backends = lambda backends, *args, **kwargs: backends[0][1](
        *args, **{k: v for k, v in kwargs.items() if k != "op_name"}
    )
    monkeypatch.setitem(sys.modules, "lumen.ops.dispatch", dispatch)
    root = Path(
        os.environ.get(
            "R81_LUMEN_SOURCE", str(Path(__file__).resolve().parents[2] / "lumen")
        )
    )
    spec = importlib.util.spec_from_file_location(
        "r81_test_mxfp4_sonic", root / "ops/moe/mxfp4_sonic.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_uses_legacy_six_argument_api(op, monkeypatch):
    calls = []
    monkeypatch.setattr(
        op, "_probe_aiter_pre_routed_a4w4", lambda: lambda *a: calls.append(a) or "old"
    )
    assert op.pre_routed_a4w4(1, 2, 3, 4, 5, 6) == "old"
    assert calls == [(1, 2, 3, 4, 5, 6)]


def test_missing_native_capability_fails_closed(op):
    with pytest.raises(RuntimeError, match="API_VERSION=1"):
        op.pre_routed_a4w4(1, 2, 3, 4, 5, wgrad_mode="mxfp4")


def test_explicit_bf16_fallback_logs_and_has_distinct_dispatch_label(
    op, monkeypatch, caplog
):
    calls = []
    monkeypatch.setattr(
        op, "_probe_aiter_pre_routed_a4w4", lambda: lambda *a: "fallback"
    )
    original = op.try_backends

    def dispatch(backends, *args, **kwargs):
        calls.append(kwargs["op_name"])
        return original(backends, *args, **kwargs)

    monkeypatch.setattr(op, "try_backends", dispatch)
    with caplog.at_level(logging.WARNING):
        assert (
            op.pre_routed_a4w4(1, 2, 3, 4, 5, wgrad_mode="mxfp4", wgrad_fallback="bf16")
            == "fallback"
        )
    assert "explicit fallback to BF16 Wgrad" in caplog.text
    assert calls == ["moe_pre_routed_a4w4:explicit_bf16_wgrad"]


def test_native_forwards_main_grad_buffers_and_distinct_label(op, monkeypatch):
    calls = []

    def native(*args, **kwargs):
        calls.append((args, kwargs))
        return "native"

    monkeypatch.setattr(op, "_probe_aiter_pre_routed_a4w4_wgrad", lambda: native)
    labels = []
    original = op.try_backends

    def dispatch(backends, *args, **kwargs):
        labels.append(kwargs["op_name"])
        return original(backends, *args, **kwargs)

    monkeypatch.setattr(op, "try_backends", dispatch)
    buffers = (object(), object())
    assert (
        op.pre_routed_quantized(
            "a4w4", 1, 2, 3, 4, 5, 6, wgrad_mode="mxfp4", main_grads=buffers
        )
        == "native"
    )
    assert calls == [((1, 2, 3, 4, 5, 6), {"main_grads": buffers})]
    assert labels == ["moe_pre_routed_a4w4:native_wgrad"]


def test_bf16_wgrad_forwards_main_grad_buffers_and_distinct_label(op, monkeypatch):
    calls = []
    buffers = (object(), object())
    monkeypatch.setattr(
        op,
        "_probe_aiter_pre_routed_a4w4_bf16_main_grad",
        lambda: lambda *a, **kw: calls.append((a, kw)) or "bf16-direct",
    )
    labels = []
    original = op.try_backends

    def dispatch(backends, *args, **kwargs):
        labels.append(kwargs["op_name"])
        return original(backends, *args, **kwargs)

    monkeypatch.setattr(op, "try_backends", dispatch)
    assert op.pre_routed_a4w4(1, 2, 3, 4, 5, 6, main_grads=buffers) == "bf16-direct"
    assert calls == [((1, 2, 3, 4, 5, 6), {"main_grads": buffers})]
    assert labels == ["moe_pre_routed_a4w4:bf16_wgrad_main_grad"]


def test_missing_bf16_main_grad_capability_fails_closed(op):
    with pytest.raises(RuntimeError, match="BF16_WGRAD_MAIN_GRAD_API_VERSION=1"):
        op.pre_routed_a4w4(1, 2, 3, 4, 5, main_grads=(1, 2))


def test_native_runtime_failure_is_not_retried(op, monkeypatch):
    fallback_calls = []

    def fail(*args, **kwargs):
        raise RuntimeError("partial write")

    monkeypatch.setattr(op, "_probe_aiter_pre_routed_a4w4_wgrad", lambda: fail)
    monkeypatch.setattr(
        op, "_probe_aiter_pre_routed_a4w4", lambda: lambda *a: fallback_calls.append(a)
    )
    with pytest.raises(RuntimeError, match="partial write"):
        op.pre_routed_a4w4(1, 2, 3, 4, 5, wgrad_mode="mxfp4", wgrad_fallback="bf16")
    assert fallback_calls == []


@pytest.mark.parametrize("recipe", ["w4a16", "a8w4", "a8w4f"])
def test_other_recipe_cannot_silently_enable_native_wgrad(op, recipe):
    with pytest.raises(ValueError, match="requires the a4w4"):
        op.pre_routed_quantized(recipe, 1, 2, 3, 4, 5, wgrad_mode="mxfp4")


@pytest.mark.parametrize(
    "kwargs",
    [{"wgrad_mode": "fp8"}, {"wgrad_fallback": "auto"}],
)
def test_invalid_contract_rejected_before_call(op, kwargs):
    with pytest.raises(ValueError):
        op.pre_routed_a4w4(1, 2, 3, 4, 5, **kwargs)


def test_gluon_capability_missing_does_not_silently_use_triton(op, monkeypatch):
    called = []
    monkeypatch.setattr(
        op, "_probe_aiter_pre_routed_a4w4_wgrad", lambda: lambda *a: called.append(a)
    )
    with pytest.raises(RuntimeError, match="gluon"):
        op.pre_routed_a4w4(1, 2, 3, 4, 5, wgrad_mode="mxfp4", wgrad_backend="gluon")
    assert called == []


def test_gluon_forwards_buffers_and_uses_gluon_dispatch_label(op, monkeypatch):
    calls = []
    buffers = (object(), object())
    monkeypatch.setattr(
        op,
        "_probe_aiter_pre_routed_a4w4_wgrad_gluon",
        lambda: lambda *a, **kw: calls.append((a, kw)) or "gluon",
    )
    original = op.try_backends
    labels = []

    def dispatch(backends, *args, **kwargs):
        labels.append((backends[0][0], kwargs["op_name"]))
        return original(backends, *args, **kwargs)

    monkeypatch.setattr(op, "try_backends", dispatch)
    assert (
        op.pre_routed_quantized(
            "a4w4",
            1,
            2,
            3,
            4,
            5,
            6,
            wgrad_mode="mxfp4",
            wgrad_backend="gluon",
            main_grads=buffers,
        )
        == "gluon"
    )
    assert calls == [((1, 2, 3, 4, 5, 6), {"main_grads": buffers})]
    assert labels == [("gluon", "moe_pre_routed_a4w4:native_wgrad:gluon")]


def test_gluon_failure_does_not_retry_other_backends(op, monkeypatch):
    calls = []

    def fail(*args, **kwargs):
        raise RuntimeError("partial Gluon write")

    monkeypatch.setattr(op, "_probe_aiter_pre_routed_a4w4_wgrad_gluon", lambda: fail)
    monkeypatch.setattr(
        op,
        "_probe_aiter_pre_routed_a4w4_wgrad",
        lambda: lambda *a: calls.append("triton"),
    )
    monkeypatch.setattr(
        op, "_probe_aiter_pre_routed_a4w4", lambda: lambda *a: calls.append("bf16")
    )
    with pytest.raises(RuntimeError, match="partial Gluon"):
        op.pre_routed_a4w4(
            1,
            2,
            3,
            4,
            5,
            wgrad_mode="mxfp4",
            wgrad_backend="gluon",
            wgrad_fallback="bf16",
        )
    assert calls == []


def test_explicit_gluon_to_bf16_fallback_logs(op, monkeypatch, caplog):
    monkeypatch.setattr(op, "_probe_aiter_pre_routed_a4w4", lambda: lambda *a: "bf16")
    with caplog.at_level(logging.WARNING):
        assert (
            op.pre_routed_a4w4(
                1,
                2,
                3,
                4,
                5,
                wgrad_mode="mxfp4",
                wgrad_backend="gluon",
                wgrad_fallback="bf16",
            )
            == "bf16"
        )
    assert "(gluon) unavailable; explicit fallback to BF16 Wgrad" in caplog.text


def test_invalid_native_backend_rejected(op):
    with pytest.raises(ValueError, match="wgrad_backend"):
        op.pre_routed_a4w4(1, 2, 3, 4, 5, wgrad_mode="mxfp4", wgrad_backend="auto")


def test_opus_forwards_buffers_and_uses_opus_dispatch_label(op, monkeypatch):
    calls = []
    buffers = (object(), object())
    monkeypatch.setattr(
        op,
        "_probe_aiter_pre_routed_a4w4_wgrad_opus",
        lambda: lambda *a, **kw: calls.append((a, kw)) or "opus",
    )
    original = op.try_backends
    labels = []

    def dispatch(backends, *args, **kwargs):
        labels.append((backends[0][0], kwargs["op_name"]))
        return original(backends, *args, **kwargs)

    monkeypatch.setattr(op, "try_backends", dispatch)
    result = op.pre_routed_a4w4(
        1, 2, 3, 4, 5, 6, wgrad_mode="mxfp4", wgrad_backend="opus", main_grads=buffers
    )
    assert result == "opus"
    assert calls == [((1, 2, 3, 4, 5, 6), {"main_grads": buffers})]
    assert labels == [(op.Backend.OPUS, "moe_pre_routed_a4w4:native_wgrad:opus")]


def test_opus_capability_missing_does_not_silently_use_triton(op, monkeypatch):
    called = []
    monkeypatch.setattr(
        op, "_probe_aiter_pre_routed_a4w4_wgrad", lambda: lambda *a: called.append(a)
    )
    monkeypatch.setattr(op, "_probe_aiter_pre_routed_a4w4_wgrad_opus", lambda: None)
    with pytest.raises(RuntimeError, match="opus"):
        op.pre_routed_a4w4(1, 2, 3, 4, 5, wgrad_mode="mxfp4", wgrad_backend="opus")
    assert called == []


def test_mapped_rejects_opus_before_call(op, monkeypatch):
    monkeypatch.setattr(op, "_probe_aiter_pre_routed_mapped", lambda: pytest.fail)
    with pytest.raises(ValueError, match="opus"):
        op.pre_routed_quantized_mapped(
            "a4w4", 1, 2, (1,), 3, 4, 5, wgrad_mode="mxfp4", wgrad_backend="opus"
        )


def test_mapped_api_forwards_source_routing_and_native_options(op, monkeypatch):
    calls = []

    def mapped(*args, **kwargs):
        calls.append((args, kwargs))
        return "mapped"

    monkeypatch.setattr(op, "_probe_aiter_pre_routed_mapped", lambda: mapped)
    buffers = (object(), object())
    assert (
        op.pre_routed_quantized_mapped(
            "a4w4",
            "source",
            "scores",
            (2, 1),
            "gather",
            "w1",
            "w2",
            "packed",
            wgrad_mode="mxfp4",
            main_grads=buffers,
            wgrad_fallback="bf16",
            wgrad_backend="gluon",
        )
        == "mapped"
    )
    assert calls == [
        (
            ("source", "scores", (2, 1), "gather", "w1", "w2", "packed"),
            {
                "recipe": "a4w4",
                "wgrad_mode": "mxfp4",
                "main_grads": buffers,
                "wgrad_fallback": "bf16",
                "wgrad_backend": "gluon",
            },
        )
    ]


def test_missing_mapped_api_fails_closed(op):
    assert not op.mapped_pre_routed_available()
    with pytest.raises(RuntimeError, match="MAPPED_DUAL_QUANT_API_VERSION=1"):
        op.pre_routed_quantized_mapped("a4w4", 1, 2, (1,), 3, 4, 5, wgrad_mode="mxfp4")


def test_mapped_runtime_failure_is_not_retried(op, monkeypatch):
    calls = []

    def fail(*args, **kwargs):
        calls.append((args, kwargs))
        raise RuntimeError("mapped kernel failed")

    monkeypatch.setattr(op, "_probe_aiter_pre_routed_mapped", lambda: fail)
    with pytest.raises(RuntimeError, match="mapped kernel failed"):
        op.pre_routed_quantized_mapped("a8w4f", 1, 2, (1,), 3, 4, 5)
    assert len(calls) == 1
