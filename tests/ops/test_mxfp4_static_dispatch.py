import csv
import os

import pytest
import torch

from lumen.ops import dispatch as dispatch_mod
from lumen.ops.quantize import linear as linear_mod
from lumen.ops.quantize import mxfp4_autotune
from lumen.ops.quantize import ops as ops_mod

KEY = (1024, 4096, 14336)


class _Operand:
    def __init__(self, shape):
        self.shape = shape


def _operands(key=KEY):
    return _Operand((key[0], key[2] // 2)), _Operand((key[1], key[2] // 2))


@pytest.fixture(autouse=True)
def _fresh_autotune_state():
    mxfp4_autotune.clear()
    yield
    mxfp4_autotune.clear()


@pytest.fixture
def static_policy(monkeypatch):
    monkeypatch.setattr(mxfp4_autotune, "AUTOTUNE_ENABLED", False)
    monkeypatch.setattr(mxfp4_autotune, "_capturing", lambda: False)


def test_autotune_off_ignores_persisted_and_in_process_choices(
    monkeypatch, tmp_path, static_policy
):
    candidates = [
        ("asm", lambda: pytest.fail("disabled autotune ran a candidate")),
        ("plain", lambda: pytest.fail("disabled autotune ran a candidate")),
    ]
    monkeypatch.setattr(mxfp4_autotune, "_CACHE_PATH", str(tmp_path / "history.json"))
    monkeypatch.setattr(
        mxfp4_autotune,
        "_load_cache",
        lambda: pytest.fail("disabled autotune loaded its persisted cache"),
    )
    mxfp4_autotune._choice[KEY] = "asm"

    mxfp4_autotune.configure()
    assert mxfp4_autotune.cached(KEY) is None
    assert mxfp4_autotune.pick_backend(KEY, candidates, fallback="plain") == "plain"
    assert mxfp4_autotune.cached(KEY) == "plain"
    assert mxfp4_autotune._choice[KEY] == "asm"
    assert mxfp4_autotune._static_choice[KEY] == "plain"
    assert not mxfp4_autotune._cache_dirty, "static choice marked for persisting"


def test_static_policy_enables_first_use_shuffle(monkeypatch, static_policy):
    monkeypatch.setattr(
        linear_mod, "_mxfp4_static_backend_for_shape", lambda shape: "shuffled"
    )
    monkeypatch.setattr(
        ops_mod, "mxfp4_data_shuffle_supported", lambda rows, cols: True
    )

    assert linear_mod._mxfp4_can_fuse_b_shuffle(KEY, 4096, 7168)
    assert mxfp4_autotune.cached(KEY) == "shuffled"
    assert mxfp4_autotune.cached_static(KEY) == "shuffled"
    assert KEY not in mxfp4_autotune._choice


def test_static_policy_requires_plain_probe(monkeypatch, static_policy):
    monkeypatch.setattr(linear_mod, "_fast_mxfp4_asm_ok", True)
    monkeypatch.setattr(linear_mod, "_fast_mxfp4_preshuffle_ok", True)
    monkeypatch.setattr(linear_mod, "_mxfp4_asm_eligible_shape", lambda *_dims: True)
    monkeypatch.setattr(
        linear_mod, "_mxfp4_preshuffle_eligible_shape", lambda *_dims: True
    )
    monkeypatch.setattr(
        ops_mod, "mxfp4_data_shuffle_supported", lambda rows, cols: True
    )

    monkeypatch.setattr(linear_mod, "_mxfp4_probe_backends", lambda: True)
    assert linear_mod._mxfp4_static_backend_for_shape(KEY) == "asm"

    monkeypatch.setattr(linear_mod, "_mxfp4_probe_backends", lambda: False)
    assert linear_mod._mxfp4_static_backend_for_shape(KEY) is None
    assert not linear_mod._mxfp4_can_fuse_b_shuffle(KEY, 4096, 7168)
    assert mxfp4_autotune.cached_static(KEY) is None


@pytest.mark.parametrize(
    "probed,fast,plain_ok,expected",
    [
        (True, False, True, "plain"),
        (True, False, False, "dequant_bf16"),
        (False, True, True, "dequant_bf16"),
    ],
    ids=["plain_on_restored_b", "plain_also_rejected", "plain_probe_failed"],
)
def test_shuffled_b_recovers_through_rowmajor_kernels(
    monkeypatch, probed, fast, plain_ok, expected,
):
    a_fp4 = _Operand((1024, 7168))
    w_fp4 = linear_mod._mark_mxfp4_data_shuffled(_Operand((4096, 7168)))
    w_rowmajor = _Operand((4096, 7168))
    restores = []

    def _rejected(*_args):
        raise RuntimeError("backend rejected these operands in this test")

    def _rowmajor(w):
        assert w is w_fp4
        restores.append(w)
        return w_rowmajor

    def _plain(a, w, *_args):
        if not probed:
            pytest.fail("ran an FP4 kernel although the plain probe failed")
        assert w is w_rowmajor, "plain read the shuffled bytes in place"
        if not plain_ok:
            _rejected()
        return "plain"

    def _fallback(a, w, *_args):
        assert w is w_rowmajor, "dequantized the shuffled bytes in place"
        return "dequant_bf16"

    dispatch_mod._backend_cache.clear()
    monkeypatch.setattr(dispatch_mod, "_SKIP_BACKEND_SYNC", True)
    monkeypatch.setattr(linear_mod, "_FAST_QUANT_DISPATCH", fast)
    monkeypatch.setattr(linear_mod, "_mxfp4_probe_backends", lambda: probed)
    monkeypatch.setattr(linear_mod, "_mxfp4_choose_backend", lambda *_args: "asm")
    monkeypatch.setitem(linear_mod._MXFP4_BACKENDS, "asm", _rejected)
    monkeypatch.setitem(linear_mod._MXFP4_BACKENDS, "shuffled", _rejected)
    monkeypatch.setitem(linear_mod._MXFP4_BACKENDS, "plain", _plain)
    monkeypatch.setattr(linear_mod, "_mxfp4_rowmajor_b", _rowmajor)
    monkeypatch.setattr(linear_mod, "_gemm_mxfp4_fallback", _fallback)
    try:
        assert linear_mod.gemm_mxfp4_dispatch(a_fp4, w_fp4, None, None) == expected
    finally:
        dispatch_mod._backend_cache.clear()
    assert len(restores) == 1, "restored B more than once in one call"


@pytest.mark.parametrize(
    "legality,tried_expected",
    [((False, False), ["plain"]), ((False, True), ["plain", "shuffled"])],
    ids=["no_other_legal_kernel", "shuffled_legal"],
)
def test_fallback_chain_skips_illegal_backends(monkeypatch, legality, tried_expected):
    a_fp4, w_fp4 = _operands()
    tried = []

    def _rejected(label):
        def run(*_args):
            tried.append(label)
            raise RuntimeError("backend rejected these operands in this test")

        return run

    dispatch_mod._backend_cache.clear()
    monkeypatch.setattr(dispatch_mod, "_SKIP_BACKEND_SYNC", True)
    monkeypatch.setattr(linear_mod, "_FAST_QUANT_DISPATCH", False)
    monkeypatch.setattr(linear_mod, "_mxfp4_probe_backends", lambda: True)
    monkeypatch.setattr(linear_mod, "_mxfp4_choose_backend", lambda *_args: "plain")
    monkeypatch.setattr(linear_mod, "_mxfp4_backend_legality", lambda *_args: legality)
    for label in ("asm", "shuffled", "plain"):
        monkeypatch.setitem(linear_mod._MXFP4_BACKENDS, label, _rejected(label))
    monkeypatch.setattr(linear_mod, "_gemm_mxfp4_fallback", lambda *_args: "dequant_bf16")
    try:
        assert linear_mod.gemm_mxfp4_dispatch(a_fp4, w_fp4, None, None) == "dequant_bf16"
    finally:
        dispatch_mod._backend_cache.clear()
    assert tried == tried_expected


def test_static_dispatch_hit_skips_runtime_probes(monkeypatch):
    a_fp4, w_fp4 = _operands()
    monkeypatch.setattr(mxfp4_autotune, "_capturing", lambda: False)
    monkeypatch.setattr(
        linear_mod,
        "_mxfp4_probe_backends",
        lambda: pytest.fail("static dispatch repeated backend probes"),
    )

    mxfp4_autotune.remember_static(KEY, "plain")
    assert linear_mod._mxfp4_choose_backend(a_fp4, w_fp4, None, None) == "plain"


def test_producer_memoized_shape_reaches_shape_log(
    monkeypatch, tmp_path, static_policy
):
    a_fp4, w_fp4 = _operands()
    log = tmp_path / "shapes.csv"
    exit_hooks = []
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(mxfp4_autotune, "_SHAPE_LOG_PATH", str(log))
    monkeypatch.setattr(mxfp4_autotune, "_hooks_registered", False)
    monkeypatch.setattr(mxfp4_autotune.atexit, "register", exit_hooks.append)
    monkeypatch.setattr(
        linear_mod, "_mxfp4_static_backend_for_shape", lambda shape: "plain"
    )
    monkeypatch.setattr(
        linear_mod, "_mxfp4_backend_legality", lambda *_args: (True, False)
    )
    monkeypatch.setattr(
        linear_mod,
        "_mxfp4_probe_backends",
        lambda: pytest.fail("shape logging repeated backend probes"),
    )
    monkeypatch.setattr(
        linear_mod,
        "_mxfp4_backend_candidates",
        lambda *_args: pytest.fail("static hit rebuilt the candidates"),
    )

    assert not linear_mod._mxfp4_can_fuse_b_shuffle(KEY, 4096, 7168)
    assert mxfp4_autotune.cached_static(KEY) == "plain"
    for _ in range(2):
        assert linear_mod._mxfp4_choose_backend(a_fp4, w_fp4, None, None) == "plain"
    assert mxfp4_autotune._save_shape_log in exit_hooks, "nothing writes the shape log at exit"
    mxfp4_autotune._save_shape_log()

    assert list(csv.DictReader(log.open())) == [
        {
            "M": "1024",
            "N": "4096",
            "K": "14336",
            "tuned": "1",
            "backend": "plain",
            "calls": "2",
        }
    ]


@pytest.mark.parametrize(
    "remembered,shuffled_b,resolved",
    [("asm", False, "plain"), ("plain", True, "shuffled")],
    ids=["asm_no_longer_legal", "plain_for_shuffled_b"],
)
def test_illegal_static_backend_is_resolved_again(
    monkeypatch, static_policy, remembered, shuffled_b, resolved,
):
    a_fp4, w_fp4 = _operands()
    if shuffled_b:
        linear_mod._mark_mxfp4_data_shuffled(w_fp4)
    monkeypatch.setattr(
        linear_mod, "_mxfp4_backend_legality", lambda *_args: (False, False)
    )
    monkeypatch.setattr(
        linear_mod,
        "_mxfp4_backend_candidates",
        lambda *_args: (
            [(resolved, lambda: pytest.fail("static policy ran a candidate"))],
            resolved,
            False,
        ),
    )

    mxfp4_autotune.remember_static(KEY, remembered)
    assert linear_mod._mxfp4_choose_backend(a_fp4, w_fp4, None, None) == resolved
    assert mxfp4_autotune.cached_static(KEY) == resolved


def test_tuned_config_cannot_change_after_dispatch(monkeypatch, tmp_path):
    tuned = tmp_path / "late.csv"
    tuned.write_text("gfx\n")
    monkeypatch.delenv(mxfp4_autotune.AITER_TUNED_CONFIG_ENV, raising=False)
    monkeypatch.setattr(
        mxfp4_autotune, "_aiter_default_tuned_config", lambda: None
    )

    mxfp4_autotune.remember_static(KEY, "plain")
    with pytest.raises(RuntimeError, match="before the first backend decision"):
        mxfp4_autotune.configure(tuned_config=str(tuned))
    assert mxfp4_autotune.AITER_TUNED_CONFIG_ENV not in os.environ


def test_wgrad_target_rows(monkeypatch):
    rows = [1, 255, 256, 257, 542, 992, 1024, 1025, 32768]

    monkeypatch.setattr(linear_mod, "_MXFP4_WGRAD_POWER2_BUCKET", True)
    assert [linear_mod._mxfp4_wgrad_target_rows(value) for value in rows] == [
        256, 256, 256, 512, 1024, 1024, 1024, 2048, 32768,
    ]

    monkeypatch.setattr(linear_mod, "_MXFP4_WGRAD_POWER2_BUCKET", False)
    assert [linear_mod._mxfp4_wgrad_target_rows(value) for value in rows] == [
        32, 256, 256, 288, 544, 992, 1024, 1056, 32768,
    ]


@pytest.mark.parametrize(
    "rows,k,n,fits",
    [
        (9000, 4096, 151936, True),
        (15000, 4096, 151936, False),
        (98304, 18432, 7168, False),
        (1, 32, 16_777_248, False),
    ],
    ids=[
        "gradient_read_at_logical_rows",
        "logical_gradient_past_int32",
        "dgrad_output_padded_past_int32",
        "packed_gradient_padded_past_int32",
    ],
)
def test_int32_guard_counts_padded_rows(monkeypatch, rows, k, n, fits):
    monkeypatch.setattr(linear_mod, "_MXFP4_WGRAD_POWER2_BUCKET", True)
    x = torch.empty((rows, k), device="meta", dtype=torch.bfloat16)
    w = torch.empty((n, k), device="meta", dtype=torch.bfloat16)

    assert linear_mod._mxfp4_operands_fit_int32(x, w) is fits
