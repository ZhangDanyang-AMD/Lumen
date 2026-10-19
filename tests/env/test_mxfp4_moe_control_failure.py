"""Healthy existing-store failure entry; no GPU fault or worker kill is injected."""

import os

import pytest
import torch.distributed as dist

from lumen.utils.mxfp4_moe_control import ControlSequence, Mxfp4Control
from lumen.utils.mxfp4_moe_ledger import Mxfp4StateError


def test_component_rejects_single_rank_metadata_failure(tmp_path):
    if not dist.is_initialized() or dist.get_world_size() != 8:
        pytest.skip("authorized world8 fixture required")
    from megatron.core.distributed.distributed_data_parallel import (
        DistributedDataParallel,
    )

    from tests.env.mxfp4_moe_fixture import build_core_layer, signed_config

    config, ddp_config = signed_config(
        os.environ["MXFP4_SIGNED_CONTRACT"],
        ep=int(os.environ["MXFP4_TEST_EP"]),
        overlap=False,
    )
    real_core_ddp = DistributedDataParallel(
        config=config, ddp_config=ddp_config, module=build_core_layer(config)
    )
    control = Mxfp4Control(
        real_core_ddp,
        physical_run_id=os.environ["MXFP4_PHYSICAL_RUN_ID"],
        journal_path=tmp_path / ("control-" + str(dist.get_rank()) + ".jsonl"),
    )

    def validate():
        if dist.get_rank() == 3:
            raise Mxfp4StateError("CAPACITY_OR_INDEX_OVERFLOW")

    try:
        with control.active_scope("test-prelaunch-error"), pytest.raises(
            Mxfp4StateError, match="COMPONENT_HEALTH_FAILURE"
        ):
            control.metadata_check(ControlSequence(0, 0, 0, 0, "STARTUP", 0), validate)
        assert control.failed
    finally:
        control.close()
