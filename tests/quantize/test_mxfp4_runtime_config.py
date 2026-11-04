"""CPU coverage of explicit process policy and the public AITER API boundary."""

import os
import sys
import json
from types import ModuleType

import pytest

from lumen.ops import dispatch
from lumen.ops.quantize import mxfp4_autotune as runtime


@pytest.fixture
def isolated_runtime(monkeypatch):
    for name, value in {
        "_started": False, "_CACHE_PATH": "", "_SHAPE_LOG_PATH": "",
        "_loaded_fingerprint": None, "_execution_fingerprint": None,
        "_cache_loaded": False, "_cache_dirty": False, "_hooks_registered": True,
        "_choice": {}, "_shape_log": {}, "AUTOTUNE_ENABLED": True,
    }.items():
        monkeypatch.setattr(runtime, name, value)
    for env in ("AITER_CONFIG_GEMM_A4W4", "LUMEN_MXFP4_AUTOTUNE_CACHE", "LUMEN_MXFP4_GEMM_SHAPE_LOG"):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr(dispatch, "_probe_aiter_mxfp4_tuning", lambda: False)
    monkeypatch.setattr(runtime, "_aiter_default_tuned_config", lambda: None)
    return runtime








def test_configure_degrades_gracefully_without_the_public_api(isolated_runtime, caplog):
    # A pinned AITER without the companion patch must not abort startup; the
    # profile is skipped with a warning and the default/env table is kept.
    result = runtime.configure(profile="qwen3_fsdp")
    assert result["tuned_config"] == ""
    assert "missing the public mxfp4_tuning API" in caplog.text
    assert runtime._tuned_table_paths() == []


def test_profile_delegates_to_public_api_and_rejects_late_change(isolated_runtime, monkeypatch, tmp_path):
    table = tmp_path / "case.csv"
    table.write_text("gfx,cu_num,M,N,K,kernelName,splitK\ngfx950,256,32,64,128,case,0\n")
    other = tmp_path / "other.csv"
    other.write_text(table.read_text())
    api = ModuleType("aiter.utility.mxfp4_tuning")
    state = {"sources": []}
    calls = []
    api.mxfp4_tuning_sources = lambda: tuple(state["sources"])
    api.mxfp4_tuning_fingerprint = lambda: "test-fingerprint"
    api.get_mxfp4_config_file = lambda: str(table)
    api.resolve_mxfp4_tuning_sources = lambda profile, paths, merge: (str(table if profile == "case" else other),)

    def configure(profile, paths, merge):
        calls.append((profile, paths, merge))
        state["sources"] = list(api.resolve_mxfp4_tuning_sources(profile, paths, merge))

    api.configure_mxfp4_tuning = configure
    monkeypatch.setitem(sys.modules, api.__name__, api)
    monkeypatch.setattr(dispatch, "_probe_aiter_mxfp4_tuning", lambda: True)
    result = runtime.configure(profile="case")
    assert result["tuned_config"] == str(table)
    assert calls == [("case", None, True)]
    runtime.mark_started()
    with pytest.raises(RuntimeError, match="first GEMM"):
        runtime.configure(profile="other")
    assert calls == [("case", None, True)]
    assert state["sources"] == [str(table)]




def test_cache_save_uses_execution_fingerprint_not_changed_source(isolated_runtime, monkeypatch, tmp_path):
    version = ["version-one"]
    monkeypatch.setattr(runtime, "_tuned_table_fingerprint", lambda: version[0])
    monkeypatch.setattr(runtime, "_arch", lambda: "gfx950")
    cache = tmp_path / "choices.json"
    runtime.configure(autotune_cache=str(cache))
    runtime.mark_started()
    runtime._choice[(32, 64, 128)] = "plain"
    runtime._cache_dirty = True
    version[0] = "version-two"
    runtime._save_cache()
    assert json.loads(cache.read_text())["tuned_tables"] == "version-one"


def test_preloaded_choice_is_revalidated_before_layout_consumes_it(isolated_runtime, monkeypatch, tmp_path):
    version = ["version-one"]
    monkeypatch.setattr(runtime, "_tuned_table_fingerprint", lambda: version[0])
    monkeypatch.setattr(runtime, "_arch", lambda: "gfx950")
    cache = tmp_path / "choices.json"
    cache.write_text(json.dumps({"arch": "gfx950", "tuned_tables": "version-one", "choices": {"32,64,128": "asm"}}))
    runtime.configure(autotune_cache=str(cache))
    assert runtime._choice[(32, 64, 128)] == "asm"
    version[0] = "version-two"
    assert runtime.cached((32, 64, 128)) is None
    assert runtime._execution_fingerprint == "version-two"


def test_real_public_profile_api_and_execution_fingerprint(isolated_runtime, monkeypatch, tmp_path):
    api = pytest.importorskip("aiter.utility.mxfp4_tuning")
    for name in ("_configured_sources", "_frozen_sources", "_resolved_file", "_frozen_fingerprint", "_effective_fingerprint"):
        monkeypatch.setattr(api, name, None)
    monkeypatch.setattr(dispatch, "_probe_aiter_mxfp4_tuning", lambda: True)
    monkeypatch.setattr(runtime, "_arch", lambda: "gfx950")
    cache = tmp_path / "choices.json"
    result = runtime.configure(profile="qwen3_fsdp", autotune_cache=str(cache))
    assert "qwen3_8b_a4w4" in result["tuned_config"]
    runtime.mark_started()
    runtime._choice[(16384, 4096, 6144)] = "asm"
    runtime._cache_dirty = True
    runtime._save_cache()
    assert json.loads(cache.read_text())["tuned_tables"] == api.mxfp4_tuning_fingerprint()[:16]
    assert runtime._execution_fingerprint == api.mxfp4_tuning_fingerprint()[:16]

def test_configure_wires_the_legacy_env_without_the_public_api(isolated_runtime, tmp_path, monkeypatch):
    table = tmp_path / "tuned.csv"
    table.write_text("gfx,cu_num,M,N,K,kernelName,splitK\n")
    monkeypatch.setattr(runtime, "_aiter_default_tuned_config", lambda: None)
    result = runtime.configure(tuned_config=str(table))
    assert os.environ.get(runtime.AITER_TUNED_CONFIG_ENV) == str(table.resolve())
    assert result["tuned_config"] == str(table.resolve())

def test_configure_warns_about_a_missing_tuned_config(isolated_runtime, tmp_path, caplog):
    missing = tmp_path / "nope.csv"
    result = runtime.configure(tuned_config=str(missing))
    assert "does not exist, ignoring" in caplog.text
    assert runtime.AITER_TUNED_CONFIG_ENV not in os.environ
    assert result["tuned_config"] == ""


def test_configure_warns_when_aiter_already_resolved(isolated_runtime, tmp_path, monkeypatch, caplog):
    table = tmp_path / "tuned.csv"
    table.write_text("gfx,cu_num,M,N,K,kernelName,splitK\n")
    monkeypatch.setattr(runtime, "_aiter_config_already_resolved", lambda: True)
    monkeypatch.setattr(runtime, "_aiter_default_tuned_config", lambda: None)
    runtime.configure(tuned_config=str(table))
    assert "ignored in this process" in caplog.text


def test_configure_rejects_a_late_legacy_table(isolated_runtime, tmp_path, monkeypatch):
    table = tmp_path / "tuned.csv"
    table.write_text("gfx,cu_num,M,N,K,kernelName,splitK\n")
    monkeypatch.setattr(runtime, "_aiter_default_tuned_config", lambda: None)
    runtime.mark_started()
    with pytest.raises(RuntimeError, match="first GEMM"):
        runtime.configure(tuned_config=str(table))


def test_configure_falls_back_when_the_public_api_is_broken(isolated_runtime, monkeypatch, caplog):
    module = ModuleType("aiter.utility.mxfp4_tuning")

    def broken(*args, **kwargs):
        raise RuntimeError("newer aiter rejected the profile")

    for name in ("configure_mxfp4_tuning", "resolve_mxfp4_tuning_sources",
                 "mxfp4_tuning_sources", "mxfp4_tuning_fingerprint", "get_mxfp4_config_file"):
        setattr(module, name, broken)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(dispatch, "_probe_aiter_mxfp4_tuning", lambda: True)
    monkeypatch.setattr(runtime, "_aiter_default_tuned_config", lambda: None)
    result = runtime.configure(profile="qwen3_fsdp")  # must not abort startup
    assert "rejected the MXFP4 tuning request" in caplog.text
    assert result["tuned_config"] == ""
