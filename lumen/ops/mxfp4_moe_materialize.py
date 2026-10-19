# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0
"""Selected bind materialization; all allocations precede public dense commit."""

import torch

from lumen.ops.mxfp4_moe import (
    RoutedFCExecution,
    TorchRuntimeOps,
    _probe_aiter_mxfp4_training,
)
from lumen.ops.mxfp4_moe_materialization_boundary import materialize_with_journal
from lumen.utils.mxfp4_moe_control import ControlSequence
from lumen.utils.mxfp4_moe_ddp import ZeroLease
from lumen.utils.mxfp4_moe_ledger import Mxfp4StateError


def _materialize_fc(
    x,
    *,
    prepared,
    route,
    tokens,
    bindings,
    handoff,
    control,
    identity,
    allocate,
    resources,
    obligations,
    cache_refs=None,
    progress=None
):
    """allocate is the selected BudgetedStorage allocator, never an unmetered factory."""
    from aiter.ops.triton.quant.fast_transpose import transpose_packed_fp4_out_v1
    from aiter.ops.triton.quant.quant import dynamic_mxfp4_quant_blockscale_out_v1
    from aiter.ops.triton.utils.mxfp4_contract import buffer_shapes

    api = _probe_aiter_mxfp4_training()
    if api is None:
        raise Mxfp4StateError("STATIC_CAPABILITY_MISS")
    n, k = prepared.static_spec[5:7]
    if (
        len(bindings) != len(route.counts)
        or any(token.route is not route for token in tokens)
        or len(tokens) != 2
        or tuple(token.domain for token in tokens) != (1, 2)
        or any(token.fc != identity[3] for token in tokens)
    ):
        raise Mxfp4StateError("ROUTING_SCHEMA_MISMATCH")
    prefix = "fc/" + "/".join(map(str, identity)) + "/"
    device = x.device

    def alloc(name, shape, dtype, category="reusable_workspace_bytes", pinned=False):
        size = {torch.uint8: 1, torch.bfloat16: 2, torch.int32: 4}[dtype]
        elements = 1
        for value in shape:
            elements *= value
        return allocate(
            prefix + name,
            category,
            elements * size,
            lambda: progress["journal"].device_call(
                "allocation_factory",
                torch.empty,
                shape,
                dtype=dtype,
                device="cpu" if pinned else device,
                pin_memory=pinned,
            ),
        )

    # Pinned host bytes require the caller's separate host allocator/accounting.
    status = alloc("status", (1,), torch.int32)
    mirror = resources["mirror"]
    host_zero = resources["zero"]
    progress["journal"].call(0, status.copy_, host_zero, non_blocking=True)
    y = alloc("Y", (route.sorted_rows, n), torch.bfloat16)
    dx = alloc("dX", (route.sorted_rows, k), torch.bfloat16)
    workspace = alloc("workspace", (0,), torch.uint8)
    events = resources["events"][:4]
    bounds, sources, leases, candidate_caches = [], [], [], []
    start = 0
    for expert, (count, binding) in enumerate(zip(route.counts, bindings)):
        binding.validate(window=binding.window, weight_epoch=binding.weight_epoch)
        refs = cache_refs[expert] if cache_refs is not None else None
        if refs is None:
            refs = {}
            for name, shape in buffer_shapes(count, n, k)[:3]:
                refs[name] = alloc(
                    str(expert) + "/" + name,
                    shape,
                    torch.uint8,
                    "persistent_cache_bytes",
                )
            refs["wt_scales"] = refs["w_scales"].t()
            cold = alloc(str(expert) + "/cold-status", (1,), torch.int32)
            progress["journal"].call(
                1 + expert * 5, cold.copy_, host_zero, non_blocking=True
            )
            quant = prepared.config_view.primitive(1, variant=0).config
            progress["journal"].call(
                2 + expert * 5,
                dynamic_mxfp4_quant_blockscale_out_v1,
                binding.parameter,
                refs["w_fp4"],
                refs["w_scales"],
                cold,
                n,
                k,
                0,
                (1, 0, 1, 1, 1, 0),
                quant,
            )
            transpose = prepared.config_view.primitive(3, variant=0).config
            progress["journal"].call(
                3 + expert * 5,
                transpose_packed_fp4_out_v1,
                refs["w_fp4"],
                refs["wt_fp4"],
                n,
                k,
                (1, 0, 3, 0, 1, 0),
                transpose,
            )
            journal = progress["journal"]
            journal.device_call(
                "cold_status.copy", mirror.copy_, cold, non_blocking=True
            )
            stream = journal.device_call(
                "current_stream", torch.cuda.current_stream, device
            )
            journal.device_call("cold_status.event_record", events[3].record, stream)
            journal.device_call("cold_status.synchronize", events[3].synchronize)
            snapshot = int(mirror[0])
            progress["numeric_status"] = (progress["numeric_status"] or 0) | snapshot
            candidate_caches.append((refs, snapshot))
        else:
            candidate_caches.append((refs, 0))
        values = dict(refs)
        for name, shape in buffer_shapes(count, n, k)[4:12]:
            values[name] = alloc(str(expert) + "/" + name, shape, torch.uint8)
        values.update(
            y_out=y[start : start + count],
            dx_out=dx[start : start + count],
            w_out=binding.main_grad,
            workspace=workspace,
            status=status,
        )
        source = x[start : start + count]
        bound = api.mxfp4_dense_bind_v1(
            prepared,
            count,
            source,
            *(values[name] for name, _ in buffer_shapes(count, n, k)),
            (0, count, n, n, 1, 2, device.index),
            tokens[0].expert_wire(expert),
            tokens[1].expert_wire(expert)
        )
        z = alloc(str(expert) + "/Z", (n, k), torch.bfloat16)
        g = alloc(str(expert) + "/G", (n, k), torch.bfloat16)
        zero_config = prepared.config_view.primitive(
            5, variant=4, output_dtype=0
        ).config
        for zero_index, tensor in enumerate((z, g)):
            progress["journal"].call(
                4 + expert * 5 + zero_index,
                api.mxfp4_zero_output_out_v1,
                tensor,
                n,
                k,
                0,
                (1, 0, 5, 0, 0, 0),
                zero_config,
            )
        lease_events = resources["events"][4 + expert * 3 : 7 + expert * 3]
        for event in lease_events:
            stream = progress["journal"].device_call(
                "current_stream", torch.cuda.current_stream, device
            )
            progress["journal"].device_call("lease.event_record", event.record, stream)
        leases.append(ZeroLease(binding, z, g, *lease_events))
        bounds.append(bound)
        sources.append(source)
        start += count
    stream = progress["journal"].device_call(
        "current_stream", torch.cuda.current_stream, device
    )
    progress["journal"].device_call(
        "materialization.event_record", events[0].record, stream
    )
    progress["journal"].device_call(
        "materialization.synchronize", events[0].synchronize
    )
    code = max((code for _, code in candidate_caches), default=0)
    progress["numeric_status"] = progress["numeric_status"] or 0
    progress["control_pending"] = True
    votes = control.vote(
        ControlSequence(*identity, "MATERIALIZATION_HEALTH", 0),
        code=code,
        reason="OK" if not code else "MATERIALIZATION_POSTLAUNCH",
    )
    progress["control_pending"] = False
    execution = RoutedFCExecution(
        route,
        tuple(bounds),
        tuple(sources),
        tuple(leases),
        handoff,
        control,
        identity,
        y,
        dx,
        status,
        mirror,
        events[1],
        events[2],
        events[3],
        obligations,
        operations=TorchRuntimeOps(),
    )
    execution.materialization_votes = votes
    execution.materialization_event = events[0]
    execution.host_initialization = host_zero
    execution.candidate_caches = tuple(refs for refs, _ in candidate_caches)
    return execution


def materialize_fc(x, **kwargs):
    return materialize_with_journal(_materialize_fc, x, **kwargs)
