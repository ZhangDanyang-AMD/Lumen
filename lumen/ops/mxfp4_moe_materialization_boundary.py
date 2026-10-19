"""Selected materialization journal and failure disposition, shared by production."""

from lumen.utils.mxfp4_moe_ledger import Mxfp4StateError


def materialize_with_journal(materialize, x, **kwargs):
    """A selected child submission can never become a candidate/prelaunch miss."""
    from lumen.utils.mxfp4_moe_submission import SubmissionJournal

    prepared = kwargs["prepared"]
    descriptors = [
        {"expert": None, "op": "status_init", "config_id": "torch.copy_", "ordinal": 0}
    ]
    quant = tuple(prepared.config_view.primitive(1, variant=0).config)
    transpose = tuple(prepared.config_view.primitive(3, variant=0).config)
    zero = tuple(prepared.config_view.primitive(5, variant=4, output_dtype=0).config)
    for expert in range(len(kwargs["bindings"])):
        for offset, (op, config) in enumerate(
            (
                ("cold_status_init", "torch.copy_"),
                ("weight_quant", quant),
                ("weight_transpose", transpose),
                ("Z_zero", zero),
                ("G_zero", zero),
            )
        ):
            descriptors.append(
                {
                    "expert": expert,
                    "op": op,
                    "config_id": config,
                    "ordinal": 1 + expert * 5 + offset,
                }
            )
    progress = {
        "journal": SubmissionJournal(descriptors),
        "control_pending": False,
        "numeric_status": None,
    }
    kwargs["obligations"].materialization_submission = progress["journal"]
    try:
        return materialize(x, progress=progress, **kwargs)
    except Exception as exc:
        control, obligations = kwargs["control"], kwargs["obligations"]
        journal = progress["journal"]
        obligations.materialization_failure = journal.failure()
        obligations.attach_journal(0, journal, materialization=True)
        # Only the known CPU vote outcome uses its typed reason; launch facts come from the journal.
        vote_failed_healthily = (
            isinstance(exc, Mxfp4StateError)
            and exc.reason == "COMPONENT_HEALTH_FAILURE"
        )
        unknown = journal.unknown() or (
            progress["control_pending"] and not vote_failed_healthily
        )
        phase = (
            "MATERIALIZATION_POSTLAUNCH"
            if journal.known_prefix() or unknown
            else "MATERIALIZATION_PRELAUNCH"
        )
        # fail(F) owns the causal abort of untouched D/W. Do not fail those terminals again.
        obligations.fail(
            0,
            exc,
            phase,
            asynchronous=unknown,
            numeric_status=progress["numeric_status"],
        )
        obligations.emit_failure(control, exc, phase, asynchronous=unknown)
        raise
