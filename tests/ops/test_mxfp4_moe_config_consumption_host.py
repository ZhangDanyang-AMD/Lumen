"""Public-view consumption and canonical host accounting, not GPU execution."""

import ast
import importlib.util
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def load_host_modules():
    spec = importlib.util.spec_from_file_location(
        "mxfp4_host_imports", ROOT / "tests/mxfp4_host_imports.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_host_modules()


@contextmanager
def host_modules():
    yield load_host_modules()


def config_view(mode=1, dtype=1, salt=0):
    # Deliberately only the public metadata protocol; no private records attribute.
    return SimpleNamespace(
        w_output_mode=mode,
        w_output_dtype=dtype,
        primitive=lambda op, **kw: SimpleNamespace(
            config=(op, kw["variant"], kw.get("output_dtype", 0), salt)
        ),
        gemm=lambda pass_id: SimpleNamespace(config=(4, pass_id, salt)),
    )


def adapter(
    modules, *, counts=(0, 31, 1, 33), mode=1, salt=0, ledger=None, invocation=0
):
    ledger = ledger if ledger is not None else modules.ledger.ObligationLedger("a" * 64)
    scope = SimpleNamespace(
        rng_epoch=0,
        attempted_step=0,
        microbatch_ordinal=0,
        global_layer_ordinal=0,
        logical_invocation_id=invocation,
    )
    route = SimpleNamespace(counts=counts, sorted_rows=sum(counts), routing_epoch=0)
    prepared = SimpleNamespace(
        config_view=config_view(mode=mode, salt=salt),
        static_spec=(0, 0, 0, 0, 0, 4096, 2048),
    )
    return modules.execution_ledger.FCObligations(
        ledger, scope, 1, route, 0, (), prepared
    )


def test_diagnostic_uses_success_configs_for_returned_calls_only():
    with host_modules() as modules:
        for mode in (0, 1):
            for pass_id, replay in ((0, False), (0, True), (1, False), (2, False)):
                fc = adapter(modules, mode=mode)
                journal = modules.submission.SubmissionJournal(
                    {"expert": i, "logical_M": count}
                    for i, count in enumerate(fc.route.counts)
                )
                fc.attach_journal(pass_id, journal, replay=replay)
                assert fc.ledger._entry(fc.keys[pass_id]).known_sublaunches == ()
                for index in (0, 1):
                    journal.enter(index)
                    journal.returned(index)
                journal.enter(2)  # Opaque: no child/config claim for expert 2 or 3.
                journal.failure()
                fc.attach_journal(pass_id, journal, replay=replay)
                entry = fc.ledger._entry(fc.keys[pass_id])
                expected = modules.children.dense_children(
                    fc.prepared.config_view,
                    fc.route.counts,
                    pass_id,
                    replay=replay,
                    experts={0, 1},
                )
                assert entry.known_sublaunches == tuple(expected)
                assert all(
                    item["expert"] < 2 and "config_id" in item
                    for item in entry.known_sublaunches
                )
                assert entry.submission_facts[2]["submission"] == "ENTERED"
                assert entry.submission_facts[3]["submission"] == "NOT_ENTERED"
                assert not entry.attempts  # Attachment does not enqueue or complete.


def finish_fc(fc, *, diagnostics=False):
    # Event is an isolated host completion fixture, not proof of CUDA completion.
    event = SimpleNamespace(query=lambda: True)
    fc.begin(0)
    fc.enqueued(0, event)
    fc.completed(0)
    fc.begin(0, replay=True)
    fc.enqueued(0, event)
    fc.completed(0)
    for pass_id in (1, 2):
        fc.begin(pass_id)
        fc.enqueued(pass_id, event)
        fc.completed(pass_id)
    original, replay = fc.ledger._entry(fc.keys[0]).attempts
    assert all(item["op"] == "gemm" for item in replay.sublaunches)
    assert len(original.sublaunches) == 2 * len(replay.sublaunches)
    assert [
        item["config_id"] for item in original.sublaunches if item["op"] == "gemm"
    ] == [item["config_id"] for item in replay.sublaunches]
    if diagnostics:
        # Deliberately noncanonical host-only objects must never enter replay identity.
        fc.ledger.attach_submission(
            fc.keys[0],
            facts=({"address": object()},),
            known_sublaunches=({"time": object()},),
        )


def test_canonical_replay_stable_across_host_identity_and_restore():
    with host_modules() as modules:
        first = adapter(modules)
        second = adapter(modules)
        finish_fc(first)
        finish_fc(second, diagnostics=True)
        checkpoint = first.ledger.checkpoint(last_committed_step=0)
        assert checkpoint == second.ledger.checkpoint(last_committed_step=0)
        restored = modules.ledger.ObligationLedger.restore(
            checkpoint, manifest_digest="a" * 64
        )
        continuation = adapter(modules, ledger=first.ledger, invocation=1)
        resumed = adapter(modules, ledger=restored, invocation=1)
        finish_fc(continuation)
        finish_fc(resumed)
        assert first.ledger.checkpoint(last_committed_step=0) == restored.checkpoint(
            last_committed_step=0
        )
        changed = adapter(modules, salt=7)
        finish_fc(changed)
        assert (
            changed.ledger.checkpoint(last_committed_step=0)["frontier"]["chain"]
            != checkpoint["frontier"]["chain"]
        )
        empty = adapter(modules, counts=(0, 0), mode=0)
        finish_fc(empty)
        assert empty.ledger.counts()["EMPTY"] == 3
        assert [
            item["op"]
            for item in empty.ledger._entry(empty.keys[2]).attempts[0].sublaunches
        ] == ["zero", "zero"]


def test_materializer_journal_consumes_public_view_without_records():
    path = ROOT / "lumen/ops/mxfp4_moe_materialize.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "records"
        for node in ast.walk(tree)
    )
    with host_modules() as modules:
        obligations = SimpleNamespace()
        view = config_view()
        journal = modules.materialization.materialize_with_journal(
            lambda x, **kw: kw["progress"]["journal"],
            None,
            prepared=SimpleNamespace(config_view=view),
            bindings=(None, None),
            obligations=obligations,
        )
        assert journal is obligations.materialization_submission
        assert journal.states == ["NOT_ENTERED"] * 11
        for expert in (0, 1):
            slots = journal.descriptors[1 + 5 * expert : 6 + 5 * expert]
            assert [slot["config_id"] for slot in slots] == [
                "torch.copy_",
                view.primitive(1, variant=0).config,
                view.primitive(3, variant=0).config,
                view.primitive(5, variant=4, output_dtype=0).config,
                view.primitive(5, variant=4, output_dtype=0).config,
            ]
            assert all(slot["expert"] == expert for slot in slots)
        # The three actual call-site selectors match the descriptor selectors.
        inner = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_materialize_fc"
        )
        selectors = [
            node
            for node in ast.walk(inner)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "primitive"
        ]
        configs = [
            view.primitive(
                ast.literal_eval(node.args[0]),
                **{kw.arg: ast.literal_eval(kw.value) for kw in node.keywords},
            ).config
            for node in selectors
        ]
        assert sorted(configs) == sorted(
            [
                view.primitive(1, variant=0).config,
                view.primitive(3, variant=0).config,
                view.primitive(5, variant=4, output_dtype=0).config,
            ]
        )


if __name__ == "__main__":
    for test in (
        test_diagnostic_uses_success_configs_for_returned_calls_only,
        test_canonical_replay_stable_across_host_identity_and_restore,
        test_materializer_journal_consumes_public_view_without_records,
    ):
        test()
        print("PASS", test.__name__)
