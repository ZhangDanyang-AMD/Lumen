"""GPU tests for the CUDA-graph replay of Megatron's TopKRouter.

The checks import the real Megatron, so they run in a subprocess to keep ``sys.modules``
clean for the stub-based patch tests that share the pytest session.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _check():
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.moe import moe_utils
    from megatron.core.transformer.moe.router import TopKRouter
    from megatron.core.transformer.transformer_config import TransformerConfig

    from lumen.patches.runtime import megatron_graphed_router as mgr

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29733")
    dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)
    parallel_state.initialize_model_parallel(1, 1)
    cfg = TransformerConfig(
        num_layers=4, hidden_size=256, num_attention_heads=4, num_moe_experts=32,
        moe_router_topk=4, moe_router_dtype="fp32", moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=1e-3, moe_ffn_hidden_size=128, add_bias_linear=False, bf16=True,
        params_dtype=torch.bfloat16,
    )
    pgc = ProcessGroupCollection.use_mpu_process_groups()
    mgr.install_graphed_router()
    eager = TopKRouter(cfg, pg_collection=pgc).cuda()
    graphed = TopKRouter(cfg, pg_collection=pgc).cuda()
    with torch.no_grad():
        eager.weight.normal_(0, 0.02)
        graphed.weight.copy_(eager.weight)
    for r in (eager, graphed):
        r.set_layer_number(2)
        r.train()
    eager.__dict__["_lumen_router_graph"] = mgr._RouterGraph()
    eager.__dict__["_lumen_router_graph"].disabled = True

    def step(router, x, g):
        moe_utils.clear_aux_losses_tracker()
        xi = x.clone().requires_grad_(True)
        router.weight.grad = None
        probs, routing_map = router(xi)
        torch.autograd.backward(probs, g)
        torch.cuda.synchronize()
        tracker = moe_utils.get_moe_layer_wise_logging_tracker()
        values = {k: v["values"].clone() for k, v in tracker.items()}
        return [probs.detach().clone(), routing_map.clone(), xi.grad.clone(), router.weight.grad.clone()], values

    torch.manual_seed(1)
    for call in range(6):
        moe_utils.MoEAuxLossAutoScaler.set_loss_scale(torch.tensor(0.25 * (call + 1), device="cuda"))
        x = torch.randn(512, 1, 256, device="cuda", dtype=torch.bfloat16)
        g = torch.randn(512, 32, device="cuda")
        ref, ref_tracker = step(eager, x, g)
        got, got_tracker = step(graphed, x, g)
        for a, b in zip(ref, got):
            assert torch.equal(a, b), call
        assert ref_tracker.keys() == got_tracker.keys()
        for k in ref_tracker:
            assert torch.equal(ref_tracker[k], got_tracker[k]), (call, k)
    assert graphed.__dict__["_lumen_router_graph"].forward is not None

    # Ineligible calls (no grad, another shape, eval mode) run the eager forward.
    x = torch.randn(512, 1, 256, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        for a, b in zip(eager(x), graphed(x)):
            assert torch.equal(a, b)
    other = torch.randn(256, 1, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    assert graphed(other)[0].shape == (256, 32)
    graphed.eval()
    assert graphed(x.clone().requires_grad_(True))[0].shape == (512, 32)
    print("GRAPHED_ROUTER_OK")


def test_graph_replay_is_bit_exact_and_falls_back():
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("megatron") is None:
        pytest.skip("needs torch and Megatron-Core")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA graphs need a GPU")
    root = str(Path(__file__).resolve().parents[2])
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([root, os.environ.get("PYTHONPATH", "")]))
    proc = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0 and "GRAPHED_ROUTER_OK" in proc.stdout, proc.stdout[-4000:] + proc.stderr[-4000:]


if __name__ == "__main__":
    _check()
