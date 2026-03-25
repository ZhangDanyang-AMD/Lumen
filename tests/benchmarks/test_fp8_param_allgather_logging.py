import ast
import contextlib
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_CONFTST_PATH = ROOT / "benchmarks" / "conftest.py"
BENCH_FP8_PATH = ROOT / "benchmarks" / "bench_fp8_param_allgather.py"


def _load_functions(path: Path, *names: str, namespace: dict[str, object] | None = None):
    source = path.read_text()
    tree = ast.parse(source)
    ns = {} if namespace is None else dict(namespace)

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(ast.get_source_segment(source, node), ns)

    missing = [name for name in names if name not in ns]
    assert not missing, f"Missing functions: {missing}"
    return ns


def _make_config(*, verbose=2, capture="no", quiet=0):
    return SimpleNamespace(
        option=SimpleNamespace(
            no_header=False,
            no_summary=False,
            verbose=verbose,
            capture=capture,
            quiet=quiet,
        )
    )


def _make_report(*, failed=False, outcome="passed"):
    return SimpleNamespace(failed=failed, outcome=outcome)


def test_pytest_configure_hides_distributed_headers_on_rank0(monkeypatch):
    funcs = _load_functions(BENCHMARK_CONFTST_PATH, "_dist_rank", "pytest_configure", namespace={"os": os})
    monkeypatch.setenv("RANK", "0")
    config = _make_config()

    funcs["pytest_configure"](config)

    assert config.option.no_header is True
    assert config.option.no_summary is True
    assert config.option.verbose == 2
    assert config.option.capture == "no"


def test_pytest_configure_quiets_nonzero_ranks(monkeypatch):
    funcs = _load_functions(BENCHMARK_CONFTST_PATH, "_dist_rank", "pytest_configure", namespace={"os": os})
    monkeypatch.setenv("RANK", "3")
    config = _make_config()

    funcs["pytest_configure"](config)

    assert config.option.no_header is True
    assert config.option.no_summary is True
    assert config.option.verbose == 0
    assert config.option.capture == "fd"
    assert config.option.quiet >= 2


def test_pytest_report_teststatus_suppresses_nonzero_rank_passes(monkeypatch):
    funcs = _load_functions(
        BENCHMARK_CONFTST_PATH,
        "_dist_rank",
        "pytest_report_teststatus",
        namespace={"os": os},
    )
    monkeypatch.setenv("RANK", "5")

    status = funcs["pytest_report_teststatus"](_make_report(failed=False), None)

    assert status == ("", "", "")


def test_pytest_report_teststatus_keeps_nonzero_rank_errors_visible(monkeypatch):
    funcs = _load_functions(
        BENCHMARK_CONFTST_PATH,
        "_dist_rank",
        "pytest_report_teststatus",
        namespace={"os": os},
    )
    monkeypatch.setenv("RANK", "5")

    status = funcs["pytest_report_teststatus"](_make_report(failed=False, outcome="error"), None)

    assert status is None


def test_init_dist_suppresses_stderr_around_process_group_init(monkeypatch):
    fake_dist = SimpleNamespace(
        is_initialized=mock.Mock(return_value=False),
        init_process_group=mock.Mock(),
    )
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(set_device=mock.Mock()),
        device=lambda name: name,
    )
    funcs = _load_functions(
        BENCH_FP8_PATH,
        "_init_dist",
        namespace={"dist": fake_dist, "os": os, "torch": fake_torch},
    )
    entered = []

    @contextlib.contextmanager
    def _fake_suppress():
        entered.append("enter")
        yield

    funcs["_suppress_dist_stderr"] = _fake_suppress
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")

    funcs["_init_dist"]()

    assert entered == ["enter"]
    fake_dist.init_process_group.assert_called_once()


def test_new_subgroup_for_size_suppresses_stderr_around_group_creation():
    fake_dist = SimpleNamespace(
        get_world_size=mock.Mock(return_value=4),
        get_rank=mock.Mock(return_value=2),
        new_group=mock.Mock(side_effect=lambda ranks: tuple(ranks)),
    )
    funcs = _load_functions(
        BENCH_FP8_PATH,
        "_new_subgroup_for_size",
        namespace={"dist": fake_dist},
    )
    entered = []

    @contextlib.contextmanager
    def _fake_suppress():
        entered.append("enter")
        yield

    funcs["_suppress_dist_stderr"] = _fake_suppress

    subgroup = funcs["_new_subgroup_for_size"](2)

    assert entered == ["enter", "enter"]
    assert subgroup == (2, 3)
