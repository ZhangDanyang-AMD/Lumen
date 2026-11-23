import importlib.util
import json
import sys
import types
from pathlib import Path

_PATH = Path(__file__).parents[2] / "lumen" / "modules" / "mori_expert_count_log.py"
_SPEC = importlib.util.spec_from_file_location("mori_expert_count_log", _PATH)
log = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(log)


class _Counts(list):
    is_cuda = False

    def tolist(self):
        return list(self)


def _layer(counts, calls):
    class _MoriManager:
        def _synced_tokens_per_expert(self):
            calls.append("sync")
            return counts

    manager = _MoriManager()
    manager.tokens_per_expert = object()
    dispatcher = type("Dispatcher", (), {})()
    dispatcher._comm_manager = manager
    layer = type("MoELayer", (), {})()
    layer.token_dispatcher, layer.layer_number = dispatcher, 7
    return layer, manager


def test_logs_each_dispatch_once_without_extra_sync(tmp_path, monkeypatch):
    monkeypatch.setenv("LUMEN_LOG_EXPERT_COUNTS", str(tmp_path))
    monkeypatch.setattr(log, "_STATE", {"file": None})
    dist = types.SimpleNamespace(is_available=lambda: True, is_initialized=lambda: True, get_rank=lambda: 2)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(distributed=dist))
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    calls = []
    counts = _Counts([3, 0, 5, 1])
    layer, manager = _layer(counts, calls)

    assert log.install_expert_count_log(layer, 0)
    assert manager._synced_tokens_per_expert() is counts
    assert manager._synced_tokens_per_expert() is counts
    manager.tokens_per_expert = object()
    manager._synced_tokens_per_expert()
    log._STATE["file"].flush()

    assert calls == ["sync"] * 3
    rows = [json.loads(x) for x in (tmp_path / "expert-counts-rank2.jsonl").read_text().splitlines()]
    assert [(r["call"], r["layer"], r["rank"], r["counts"]) for r in rows] == [
        (0, 7, 2, [3, 0, 5, 1]),
        (1, 7, 2, [3, 0, 5, 1]),
    ]


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("LUMEN_LOG_EXPERT_COUNTS", raising=False)
    layer, manager = _layer(_Counts([1]), [])
    assert not log.install_expert_count_log(layer, 0)
    assert "_synced_tokens_per_expert" not in vars(manager)
