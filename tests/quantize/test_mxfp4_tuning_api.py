"""CPU integration of Lumen configuration with the public AITER tuning API."""

import os
import sys
import json
import types

import pytest

from lumen.ops import dispatch
from lumen.ops.quantize import mxfp4_autotune


@pytest.fixture
def local_mxfp4_tuning(monkeypatch):
    """Legacy env/default-table mode: the cache guardrails run without AITER."""
    for name, value in {
        "_CACHE_PATH": "", "_SHAPE_LOG_PATH": "", "_choice": {}, "_shape_log": {},
        "_cache_loaded": False, "_cache_dirty": False, "_started": False,
        "_loaded_fingerprint": None, "_execution_fingerprint": None, "_hooks_registered": True,
    }.items():
        monkeypatch.setattr(mxfp4_autotune, name, value)
    monkeypatch.setattr(dispatch, "_probe_aiter_mxfp4_tuning", lambda: False)
    monkeypatch.delenv("AITER_CONFIG_GEMM_A4W4", raising=False)
    monkeypatch.delenv("LUMEN_MXFP4_AUTOTUNE_CACHE", raising=False)
    monkeypatch.setattr(mxfp4_autotune, "_arch", lambda: "gfx950")
    return mxfp4_autotune


@pytest.fixture
def isolated_public_mxfp4_tuning(monkeypatch):
    api = pytest.importorskip("aiter.utility.mxfp4_tuning", reason="requires the matching AITER tuning API")
    for name in ("_configured_sources", "_frozen_sources", "_resolved_file", "_frozen_fingerprint", "_effective_fingerprint"):
        monkeypatch.setattr(api, name, None)
    for name, value in {
        "_CACHE_PATH": "", "_SHAPE_LOG_PATH": "", "_choice": {}, "_shape_log": {},
        "_cache_loaded": False, "_cache_dirty": False, "_started": False,
        "_loaded_fingerprint": None, "_execution_fingerprint": None, "_hooks_registered": True,
    }.items():
        monkeypatch.setattr(mxfp4_autotune, name, value)
    monkeypatch.setattr(dispatch, "_probe_aiter_mxfp4_tuning", lambda: True)
    monkeypatch.delenv("AITER_CONFIG_GEMM_A4W4", raising=False)
    monkeypatch.delenv("LUMEN_MXFP4_AUTOTUNE_CACHE", raising=False)
    monkeypatch.setattr(mxfp4_autotune, "_arch", lambda: "gfx950")
    return api


def _new_process_state(api=None):
    if api is not None:
        for name in ("_configured_sources", "_frozen_sources", "_resolved_file", "_frozen_fingerprint", "_effective_fingerprint"):
            setattr(api, name, None)
    mxfp4_autotune.clear()


def test_mxfp4_autotune_cache_roundtrip(tmp_path, local_mxfp4_tuning):
    cache = tmp_path / "autotune.json"
    mxfp4_autotune._CACHE_PATH = str(cache)
    record = {"arch": "gfx950", "tuned_tables": mxfp4_autotune._tuned_table_fingerprint(), "choices": {"8192,12288,4096": "asm"}}
    cache.write_text(json.dumps(record))
    mxfp4_autotune._load_cache()
    assert mxfp4_autotune.cached((8192, 12288, 4096)) == "asm"
    _new_process_state()
    record["arch"] = "gfx000-not-a-real-arch"
    cache.write_text(json.dumps(record))
    mxfp4_autotune._load_cache()
    assert mxfp4_autotune.cached((8192, 12288, 4096)) is None


def test_mxfp4_autotune_cache_rejects_a_different_tuned_table(tmp_path, local_mxfp4_tuning, monkeypatch):
    table = tmp_path / "tuned.csv"
    header = "gfx,cu_num,M,N,K,kernelName,splitK\n"
    table.write_text(header + "gfx950,256,8192,12288,4096,first,0\n")
    monkeypatch.setenv("AITER_CONFIG_GEMM_A4W4", str(table))
    cache = tmp_path / "autotune.json"
    mxfp4_autotune._CACHE_PATH = str(cache)
    first = mxfp4_autotune._tuned_table_fingerprint()
    cache.write_text(json.dumps({"arch": "gfx950", "tuned_tables": first, "choices": {"8192,12288,4096": "asm"}}))
    mxfp4_autotune._load_cache()
    assert mxfp4_autotune.cached((8192, 12288, 4096)) == "asm"
    # The live process keeps the choices it already measured; freezing the
    # source data itself is AITER's job (covered by the public-API tests).
    table.write_text(header + "gfx950,256,64,64,128,second,0\n")
    assert mxfp4_autotune._tuned_table_fingerprint() != first
    _new_process_state()
    mxfp4_autotune._load_cache()
    assert mxfp4_autotune.cached((8192, 12288, 4096)) is None
    second = mxfp4_autotune._tuned_table_fingerprint()
    cache.write_text(json.dumps({"arch": "gfx950", "tuned_tables": second, "choices": {"64,64,128": "plain"}}))
    _new_process_state()
    mxfp4_autotune._load_cache()
    assert mxfp4_autotune.cached((64, 64, 128)) == "plain"


def test_mxfp4_autotune_cache_write_cannot_tear_the_file(tmp_path, local_mxfp4_tuning, monkeypatch):
    cache = tmp_path / "autotune.json"
    good = '{"previous": true}'
    cache.write_text(good)
    mxfp4_autotune._CACHE_PATH = str(cache)
    mxfp4_autotune._choice[(4, 5, 6)] = "asm"
    mxfp4_autotune._cache_dirty = True

    def fail_write(*args, **kwargs):
        raise OSError("simulated full filesystem")

    with monkeypatch.context() as scoped:
        scoped.setattr(mxfp4_autotune.json, "dump", fail_write)
        mxfp4_autotune._save_cache()
    assert cache.read_text() == good
    assert not list(tmp_path.glob("*.tmp"))
    mxfp4_autotune._save_cache()
    assert json.loads(cache.read_text())["choices"]["4,5,6"] == "asm"


def test_mxfp4_configure_wires_tuned_table_and_cache(tmp_path, isolated_public_mxfp4_tuning):
    cache = tmp_path / "autotune.json"
    tuned = tmp_path / "tuned.csv"
    tuned.write_text("gfx,cu_num,M,N,K,kernelName,splitK\n")
    applied = mxfp4_autotune.configure(tuned_config=str(tuned), autotune_cache=str(cache))
    assert applied["tuned_config"].split(os.pathsep)[0] == str(tuned)
    assert applied["autotune_cache"] == str(cache)
    mxfp4_autotune.mark_started()
    assert mxfp4_autotune.configure(tuned_config=str(tuned))["tuned_config"] == applied["tuned_config"]
    other = tmp_path / "other.csv"
    other.write_text(tuned.read_text())
    with pytest.raises(RuntimeError, match="first GEMM"):
        mxfp4_autotune.configure(tuned_config=str(other))


def test_mxfp4_configure_keeps_table_order_and_rejects_missing(tmp_path, isolated_public_mxfp4_tuning):
    model = tmp_path / "model.csv"
    generic = tmp_path / "generic.csv"
    for path in (model, generic):
        path.write_text("gfx,cu_num,M,N,K,kernelName,splitK\n")
    applied = mxfp4_autotune.configure(tuned_config=[str(model), str(generic)])
    assert applied["tuned_config"].split(os.pathsep)[:2] == [str(model), str(generic)]
    with pytest.raises(FileNotFoundError):
        mxfp4_autotune.configure(tuned_config=[str(tmp_path / "missing.csv"), str(generic)])
    assert mxfp4_autotune._tuned_table_paths()[:2] == [str(model), str(generic)]


def _stub_public_api(monkeypatch, names):
    """Install a fake aiter.utility.mxfp4_tuning exposing only ``names``."""
    pkg = types.ModuleType("aiter")
    pkg.__path__ = []
    util = types.ModuleType("aiter.utility")
    util.__path__ = []
    module = types.ModuleType("aiter.utility.mxfp4_tuning")
    for name in names:
        setattr(module, name, lambda *args, **kwargs: [])
    monkeypatch.setitem(sys.modules, "aiter", pkg)
    monkeypatch.setitem(sys.modules, "aiter.utility", util)
    monkeypatch.setitem(sys.modules, "aiter.utility.mxfp4_tuning", module)
    return module


def test_probe_requires_the_whole_public_api(monkeypatch):
    _stub_public_api(monkeypatch, ("configure_mxfp4_tuning", "mxfp4_tuning_sources"))
    dispatch._probe_aiter_mxfp4_tuning.cache_clear()
    assert not dispatch._probe_aiter_mxfp4_tuning()  # partial install must not pass

    _stub_public_api(monkeypatch, (
        "configure_mxfp4_tuning", "get_mxfp4_config_file", "mxfp4_tuning_fingerprint",
        "mxfp4_tuning_sources", "resolve_mxfp4_tuning_sources",
    ))
    dispatch._probe_aiter_mxfp4_tuning.cache_clear()
    assert dispatch._probe_aiter_mxfp4_tuning()


def test_mark_started_survives_a_broken_tuning_api(monkeypatch, tmp_path):
    module = _stub_public_api(monkeypatch, (
        "configure_mxfp4_tuning", "get_mxfp4_config_file", "mxfp4_tuning_fingerprint",
        "mxfp4_tuning_sources", "resolve_mxfp4_tuning_sources",
    ))

    def broken(*args, **kwargs):
        raise RuntimeError("aiter tuning is broken")

    module.get_mxfp4_config_file = broken
    module.mxfp4_tuning_fingerprint = broken
    module.mxfp4_tuning_sources = lambda: []
    monkeypatch.setattr(dispatch, "_probe_aiter_mxfp4_tuning", lambda: True)
    monkeypatch.setattr(mxfp4_autotune, "_started", False)
    monkeypatch.setattr(mxfp4_autotune, "_execution_fingerprint", None)
    monkeypatch.setattr(mxfp4_autotune, "_cache_loaded", False)
    monkeypatch.delenv("AITER_CONFIG_GEMM_A4W4", raising=False)

    mxfp4_autotune.mark_started()  # must not abort the first GEMM
    assert mxfp4_autotune._started
    assert mxfp4_autotune._execution_fingerprint
