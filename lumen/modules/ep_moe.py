###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Expert-parallel MoE block with all-to-all token routing.

Wraps a HuggingFace MoE block (e.g. ``Qwen3MoeSparseMoeBlock``) and
distributes experts across an EP process group.  Each GPU keeps
``num_experts // ep_size`` local experts.  Forward does:

  1. Gate routing (softmax + top-k)
  2. All-to-all: send tokens to the GPU owning the target expert
  3. Local expert compute (replaceable via ``_compute_local_experts``)
  4. All-to-all: send results back
  5. Scatter-add weighted outputs

The local expert compute (step 3) is isolated in
``_compute_local_experts`` so that ``LumenConfig._patch_ep_moe_block``
can replace it with fused AITER kernels without duplicating routing.

The all-to-all communication is wrapped in ``_AllToAllFn`` so that
gradients flow correctly through the dispatch/combine steps during
training.
"""

import logging

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _rank0_print(msg: str) -> None:
    try:
        if dist.is_initialized() and dist.get_rank() != 0:
            return
    except Exception:
        pass
    logger.info(msg)


class _AllToAllFn(torch.autograd.Function):
    """Differentiable all-to-all: backward performs the inverse all-to-all."""

    @staticmethod
    def forward(ctx, input, send_counts, recv_counts, ep_group):
        ctx.send_counts = send_counts
        ctx.recv_counts = recv_counts
        ctx.ep_group = ep_group

        send_splits = send_counts.tolist()
        recv_splits = recv_counts.tolist()
        total_recv = sum(recv_splits)
        dim = input.shape[1]

        output = torch.empty(
            total_recv, dim, dtype=input.dtype, device=input.device,
        )
        send_list = list(input.contiguous().view(-1).split(
            [c * dim for c in send_splits]))
        recv_list = list(output.view(-1).split(
            [c * dim for c in recv_splits]))
        dist.all_to_all(recv_list, send_list, group=ep_group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse: send_counts ↔ recv_counts
        grad_input = _AllToAllFn.apply(
            grad_output, ctx.recv_counts, ctx.send_counts, ctx.ep_group,
        )
        return grad_input, None, None, None


class EPShardedMoeBlock(nn.Module):
    """Expert-parallel MoE block with all-to-all token routing.

    Each GPU keeps only its local slice of experts (num_experts / ep_size).
    Forward: route tokens via all-to-all within ep_group, compute local
    experts, all-to-all back.
    """

    def __init__(self, original_moe_block, ep_rank, ep_size, ep_group):
        super().__init__()
        self.num_experts = getattr(
            original_moe_block, "num_experts", original_moe_block.gate.num_experts
        )
        self.top_k = getattr(
            original_moe_block, "top_k", original_moe_block.gate.top_k
        )
        self.norm_topk_prob = getattr(
            original_moe_block, "norm_topk_prob", original_moe_block.gate.norm_topk_prob
        )
        self.gate = original_moe_block.gate
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.ep_group = ep_group

        experts_per_gpu = self.num_experts // ep_size
        start = ep_rank * experts_per_gpu
        end = start + experts_per_gpu
        self.local_expert_start = start
        self.local_expert_end = end
        self.experts_per_gpu = experts_per_gpu

        # Old HF layout: experts are a ModuleList.
        # New HF layout: experts are packed tensors in Qwen3MoeExperts.
        self.local_experts = None
        self.local_gate_up_proj = None
        self.local_down_proj = None
        self.act_fn = None
        experts = original_moe_block.experts
        if hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj"):
            self.local_gate_up_proj = nn.Parameter(
                experts.gate_up_proj[start:end].contiguous()
            )
            self.local_down_proj = nn.Parameter(
                experts.down_proj[start:end].contiguous()
            )
            self.act_fn = experts.act_fn
        else:
            self.local_experts = nn.ModuleList(
                [experts[i] for i in range(start, end)]
            )
        del original_moe_block.experts

    def _compute_local_experts(self, recv_hidden, recv_expert_ids, recv_weights):
        """Compute local expert outputs for received tokens.

        This is the hot loop that processes each local expert sequentially.
        ``LumenConfig._patch_ep_moe_block`` replaces this method with a
        fused kernel version (AITER ``fused_moe_triton``).

        Args:
            recv_hidden: Received token hidden states [total_recv, hidden_dim].
            recv_expert_ids: Local expert index per token [total_recv].
            recv_weights: Routing weight per token [total_recv].

        Returns:
            output_hidden: Weighted expert outputs [total_recv, hidden_dim].
        """
        output_hidden = torch.zeros_like(recv_hidden)
        for local_idx in range(self.experts_per_gpu):
            mask = recv_expert_ids == local_idx
            if not mask.any():
                continue
            expert_input = recv_hidden[mask]
            if self.local_experts is not None:
                expert_output = self.local_experts[local_idx](expert_input)
            else:
                gate_up = self.local_gate_up_proj[local_idx]
                gate, up = F.linear(expert_input, gate_up).chunk(2, dim=-1)
                expert_hidden = self.act_fn(gate) * up
                expert_output = F.linear(expert_hidden, self.local_down_proj[local_idx])
            output_hidden[mask] = expert_output * recv_weights[mask].unsqueeze(-1)
        return output_hidden

    def forward(self, hidden_states: torch.Tensor):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_dim)
        num_tokens = hidden_flat.shape[0]

        # --- Routing ---
        gate_out = self.gate(hidden_flat)
        if isinstance(gate_out, tuple) and len(gate_out) == 3:
            router_logits, topk_weights, topk_indices = gate_out
        else:
            router_logits = gate_out
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
            if self.norm_topk_prob:
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(hidden_flat.dtype)

        # --- Map expert indices to target GPU within EP group ---
        target_gpus = topk_indices // self.experts_per_gpu
        local_expert_ids = topk_indices % self.experts_per_gpu

        # --- Count tokens per (source_gpu -> dest_gpu) (vectorized) ---
        target_gpus_flat = target_gpus.reshape(-1)
        send_counts = torch.bincount(
            target_gpus_flat, minlength=self.ep_size
        ).to(torch.long)

        recv_counts = torch.zeros(self.ep_size, dtype=torch.long, device=hidden_flat.device)
        dist.all_to_all_single(recv_counts, send_counts, group=self.ep_group)

        # --- Prepare send buffers (vectorized argsort) ---
        sorted_order = torch.argsort(target_gpus_flat, stable=True)
        send_indices = sorted_order // self.top_k
        send_expert_ids_cat = local_expert_ids.reshape(-1)[sorted_order]
        send_weights_cat = topk_weights.reshape(-1)[sorted_order]

        # Build send buffer with autograd-tracked hidden and weights.
        # Expert IDs are discrete (no grad); hidden and weights need grad flow.
        send_hidden = hidden_flat[send_indices]
        send_weights_sorted = send_weights_cat

        # --- Differentiable all-to-all dispatch ---
        recv_hidden = _AllToAllFn.apply(
            send_hidden, send_counts, recv_counts, self.ep_group,
        )

        # Expert IDs: small integers, no grad needed — pack + all-to-all
        send_counts_list = send_counts.tolist()
        recv_counts_list = recv_counts.tolist()
        total_recv = sum(recv_counts_list)

        recv_expert_ids = torch.empty(
            total_recv, dtype=send_expert_ids_cat.dtype,
            device=hidden_flat.device,
        )
        send_eid_list = list(send_expert_ids_cat.split(send_counts_list))
        recv_eid_list = list(recv_expert_ids.split(recv_counts_list))
        dist.all_to_all(recv_eid_list, send_eid_list, group=self.ep_group)

        # Weights: need grad flow through all-to-all
        recv_weights = _AllToAllFn.apply(
            send_weights_sorted.unsqueeze(1),
            send_counts, recv_counts, self.ep_group,
        ).squeeze(1)

        # --- Compute local experts (replaceable by fused_moe patch) ---
        output_hidden = self._compute_local_experts(
            recv_hidden, recv_expert_ids, recv_weights
        )

        # --- Differentiable all-to-all combine (reverse direction) ---
        return_hidden = _AllToAllFn.apply(
            output_hidden, recv_counts, send_counts, self.ep_group,
        )

        # --- Scatter-add results back to original token positions ---
        final_output = torch.zeros(num_tokens, hidden_dim,
                                   dtype=hidden_flat.dtype, device=hidden_flat.device)
        final_output.index_add_(0, send_indices, return_hidden)

        return final_output.view(batch_size, seq_len, hidden_dim)


def shard_experts_ep(model, ep_size, ep_rank, ep_group):
    """Replace every MoE block with EPShardedMoeBlock.

    Walks ``model.model.layers`` and replaces any layer whose ``mlp``
    has an ``experts`` attribute (HuggingFace MoE convention).
    """
    count = 0
    for layer in model.model.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            layer.mlp = EPShardedMoeBlock(layer.mlp, ep_rank, ep_size, ep_group)
            count += 1
    _rank0_print(f"> EP sharding: replaced {count} MoE blocks, "
                 f"{model.config.num_experts // ep_size if hasattr(model.config, 'num_experts') else '?'}"
                 f" local experts/layer on ep_rank {ep_rank}")
    return model
