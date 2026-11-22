"""Row-id map prefetch: one-shot identity cache, and exact TE permute results on GPU."""

from types import SimpleNamespace

import pytest
import torch

from lumen.modules import mori_row_id_prefetch as prefetch_module


class _Map:
    def __init__(self, shape):
        self.shape = shape
        self._version = 0


def test_prefetch_is_one_shot_and_identity_keyed():
    calls = []
    prefetch = prefetch_module._Prefetch(lambda rm, t, e: calls.append((rm, t, e)) or ("map", rm))
    first, other = _Map((4, 2)), _Map((4, 2))
    prefetch.prefetch(first)
    prefetch.prefetch(first)
    assert len(calls) == 1
    assert prefetch.take(other, 4, 2) is None
    assert prefetch.take(first, 4, 3) is None
    assert prefetch.take(first, 4, 2) == ("map", first)
    assert prefetch.take(first, 4, 2) is None
    prefetch.prefetch(first)
    assert len(calls) == 1


def test_prefetch_rejects_mutated_routing_map():
    prefetch = prefetch_module._Prefetch(lambda rm, t, e: "map")
    routing_map = _Map((4, 2))
    prefetch.prefetch(routing_map)
    routing_map._version += 1
    assert prefetch.take(routing_map, 4, 2) is None


_GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@_GPU
def test_installed_prefetch_keeps_te_permute_exact(monkeypatch):
    te_permutation = pytest.importorskip("transformer_engine.pytorch.permutation")
    token_dispatcher = pytest.importorskip("megatron.core.transformer.moe.token_dispatcher")
    monkeypatch.delenv("LUMEN_MORI_ROW_ID_PREFETCH", raising=False)
    manager_type = token_dispatcher._MoriManager
    monkeypatch.setattr(manager_type, "_synced_tokens_per_expert", manager_type._synced_tokens_per_expert)
    triton_permutation = te_permutation.triton_permutation
    monkeypatch.setattr(triton_permutation, "make_row_id_map", triton_permutation.make_row_id_map)
    monkeypatch.setitem(prefetch_module._STATE, "installed", False)
    assert prefetch_module.install_mori_row_id_prefetch()

    torch.manual_seed(5)
    tokens, experts, hidden = 4099, 32, 256
    routing_map = torch.rand((tokens, experts), device="cuda") < 0.2
    routing_map[7] = False
    probs = torch.rand((tokens, experts), device="cuda") * routing_map
    x = torch.randn((tokens, hidden), dtype=torch.bfloat16, device="cuda")
    counts = routing_map.sum(0)
    manager = object.__new__(manager_type)
    manager.permute_fusion = True
    manager.config = SimpleNamespace(moe_router_padding_for_quantization=False)
    manager.dispatched_routing_map = routing_map
    manager.tokens_per_expert = counts
    manager._tokens_per_expert_event = None
    manager._tokens_per_expert_cpu = None
    num_out = int(manager._synced_tokens_per_expert().sum())
    assert prefetch_module._STATE["prefetch"].row_id_map is not None

    def run(rm):
        xi = x.clone().requires_grad_()
        pi = probs.clone().requires_grad_()
        out, out_probs, row_id_map = te_permutation.moe_permute_with_probs(xi, pi, rm, num_out_tokens=num_out)
        (out.float().square().sum() + out_probs.sum()).backward()
        return out, out_probs, row_id_map, xi.grad, pi.grad

    prefetched = run(routing_map)
    assert prefetch_module._STATE["prefetch"].row_id_map is None
    reference = run(routing_map.clone())
    for index in (0, 1, 3, 4):
        assert torch.equal(prefetched[index], reference[index])
    # TE fills only the first n_routed slots of each half plus the trailing count.
    got_map, want_map = prefetched[2], reference[2]
    n_routed = want_map[:, -1:]
    assert torch.equal(got_map[:, -1:], n_routed)
    slot = torch.arange(experts, device="cuda").unsqueeze(0)
    defined = slot < n_routed
    for half in (slice(0, experts), slice(experts, 2 * experts)):
        assert torch.equal(got_map[:, half][defined], want_map[:, half][defined])
