###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Install fused QKV, fused gate/up, cached RoPE, and residual wiring on Qwen3."""

from __future__ import annotations

from lumen.ops.rope_cached import make_cached_rope_fn
from lumen.ops.gemm.linear_autograd import make_fused_gate_up_fn, make_fused_qkv_fn
from lumen.ops.gemm.stable_tgemm_mm import stable_tgemm_mm


def _rank0_print(msg: str) -> None:
    from lumen.config import _rank0_print as _print

    _print(msg)


def patch_qwen3_qkv(model) -> None:
    """Replace Qwen3 q/k/v projections with one concatenated GEMM.

    ``q_proj`` runs ``tgemm.mm`` on ``cat(q, k, v)`` weights. ``k_proj`` and
    ``v_proj`` return the slices from that same result, so Q/K RMSNorm still
    see the tensors ATOM splits out of ``qkv_proj``.
    """
    from aiter.tuned_gemm import TunedGemm

    tgemm = TunedGemm()
    qkv_fn = make_fused_qkv_fn(tgemm)
    count = 0
    for module in model.modules():
        if type(module).__name__ != "Qwen3Attention":
            continue
        q_mod = getattr(module, "q_proj", None)
        k_mod = getattr(module, "k_proj", None)
        v_mod = getattr(module, "v_proj", None)
        if q_mod is None or k_mod is None or v_mod is None:
            continue
        pending = {}

        def q_forward(hidden, q_mod=q_mod, k_mod=k_mod, v_mod=v_mod, pending=pending):
            fused = qkv_fn.apply(
                hidden, q_mod.weight, k_mod.weight, v_mod.weight, q_mod.bias, k_mod.bias, v_mod.bias
            )
            q, k, v = fused.split(
                (q_mod.weight.shape[0], k_mod.weight.shape[0], v_mod.weight.shape[0]),
                dim=-1,
            )
            pending["k"] = k
            pending["v"] = v
            return q

        def k_forward(hidden, k_mod=k_mod, pending=pending):
            cached = pending.pop("k", None)
            if cached is not None:
                return cached
            return stable_tgemm_mm(
                tgemm, hidden, k_mod.weight, k_mod.bias, hidden.dtype
            )

        def v_forward(hidden, v_mod=v_mod, pending=pending):
            cached = pending.pop("v", None)
            if cached is not None:
                return cached
            return stable_tgemm_mm(
                tgemm, hidden, v_mod.weight, v_mod.bias, hidden.dtype
            )

        q_mod.forward = q_forward
        k_mod.forward = k_forward
        v_mod.forward = v_forward
        count += 1
    if count:
        _rank0_print(f"> qkv: {count} Qwen3 attention blocks use one tgemm.mm")


def patch_qwen3_gate_up(model) -> None:
    """Replace Qwen3 gate/up projections with one concatenated GEMM.

    ``gate_proj`` runs ``tgemm.mm`` on ``cat(gate, up)`` weights. ``up_proj``
    returns the second half of that result, so the MLP still sees the two
    slices ATOM splits out of ``gate_up_proj``.
    """
    from aiter.tuned_gemm import TunedGemm

    tgemm = TunedGemm()
    gate_up_fn = make_fused_gate_up_fn(tgemm)
    count = 0
    for module in model.modules():
        if type(module).__name__ != "Qwen3MLP":
            continue
        gate_mod = getattr(module, "gate_proj", None)
        up_mod = getattr(module, "up_proj", None)
        if gate_mod is None or up_mod is None:
            continue
        pending = {}

        def gate_forward(hidden, gate_mod=gate_mod, up_mod=up_mod, pending=pending):
            fused = gate_up_fn.apply(
                hidden, gate_mod.weight, up_mod.weight, gate_mod.bias, up_mod.bias
            )
            gate, up = fused.split((gate_mod.weight.shape[0], up_mod.weight.shape[0]), dim=-1)
            pending["up"] = up
            return gate

        def up_forward(hidden, up_mod=up_mod, pending=pending):
            cached = pending.pop("up", None)
            if cached is not None:
                return cached
            return stable_tgemm_mm(
                tgemm, hidden, up_mod.weight, up_mod.bias, hidden.dtype
            )

        gate_mod.forward = gate_forward
        up_mod.forward = up_forward
        count += 1
    if count:
        _rank0_print(f"> gate_up: {count} Qwen3 MLP blocks use one tgemm.mm")


def patch_qwen3_rope(model) -> None:
    """Apply ATOM's cached RoPE on Qwen3 Q and K, then keep HF's head layout.

    Q and K stay ``(batch, tokens, heads, dim)`` for the kernel. The result
    is transposed back to ``(batch, heads, tokens, dim)`` for the rest of
    the Hugging Face attention forward.
    """
    import torch
    from aiter.rotary_embedding import get_rope

    count = 0
    for module in model.modules():
        if type(module).__name__ != "Qwen3Attention":
            continue
        cfg = module.config
        rope_type = cfg.rope_parameters.get("rope_type", "default")
        if rope_type != "default":
            _rank0_print(f"> rope: skip {rope_type}")
            continue
        head_dim = int(module.head_dim)
        rotary = get_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=int(cfg.max_position_embeddings),
            base=cfg.rope_parameters["rope_theta"],
            is_neox_style=True,
            dtype=torch.bfloat16,
        )
        rope_fn = make_cached_rope_fn(rotary)

        def forward(
            hidden_states,
            position_embeddings=None,
            attention_mask=None,
            past_key_values=None,
            position_ids=None,
            module=module,
            rope_fn=rope_fn,
            **kwargs,
        ):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, module.head_dim)
            query = module.q_norm(module.q_proj(hidden_states).view(hidden_shape))
            key = module.k_norm(module.k_proj(hidden_states).view(hidden_shape))
            value = module.v_proj(hidden_states).view(hidden_shape)
            if position_ids is None:
                position_ids = torch.arange(
                    query.shape[1], device=query.device
                ).expand(query.shape[0], -1)
            positions = position_ids.reshape(-1).to(dtype=torch.long)
            query_tok = query.reshape(-1, query.shape[-2], query.shape[-1])
            key_tok = key.reshape(-1, key.shape[-2], key.shape[-1])
            query_tok, key_tok = rope_fn.apply(query_tok, key_tok, positions)
            kwargs = dict(kwargs)
            kwargs.pop("position_ids", None)
            query_states = query_tok.view_as(query).transpose(1, 2)
            key_states = key_tok.view_as(key).transpose(1, 2)
            value_states = value.transpose(1, 2)
            from transformers.models.qwen3.modeling_qwen3 import (
                ALL_ATTENTION_FUNCTIONS,
                eager_attention_forward,
            )

            if past_key_values is not None:
                key_states, value_states = past_key_values.update(
                    key_states, value_states, module.layer_idx
                )
            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                module.config._attn_implementation, eager_attention_forward
            )
            attn_output, attn_weights = attention_interface(
                module,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not module.training else module.attention_dropout,
                scaling=module.scaling,
                sliding_window=module.sliding_window,
                **kwargs,
            )
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = module.o_proj(attn_output)
            return attn_output, attn_weights

        module.forward = forward
        count += 1
    if count:
        _rank0_print(f"> rope: {count} Qwen3 attention blocks use rope_cached_positions_2c_fwd_inplace")


def patch_qwen3_residual_rmsnorm(model) -> None:
    """Carry ATOM's residual into the next RMSNorm instead of adding in PyTorch.

    Layer 0 input norm stays a plain ``rmsnorm2d_fwd``. Every later norm that
    ATOM fuses with the residual — post-attention, the next layer input, and
    the final norm — calls ``rmsnorm2d_fwd_with_add``.
    """
    backbone = getattr(model, "model", model)
    layers = getattr(backbone, "layers", None)
    norm = getattr(backbone, "norm", None)
    if not layers or norm is None or type(layers[0]).__name__ != "Qwen3DecoderLayer":
        _rank0_print("> residual_rmsnorm: Qwen3 decoder stack not found, skipped")
        return

    import torch

    # The residual used to live on backbone._atom_residual. Activation
    # checkpointing recomputes one layer at a time and does not replay that
    # attribute, so the recomputed layer took the no-residual branch and
    # saved a different tensor (3D hidden vs the 2D fused residual). The
    # residual now rides in the layer output, which is the next layer's
    # positional input and is what checkpoint saves. Qwen3 hidden states
    # are rank 3; rank 4 with a leading 2 is (hidden, residual).
    def _split_residual(hidden_states):
        if hidden_states.dim() == 4 and hidden_states.shape[0] == 2:
            return hidden_states[0], hidden_states[1]
        return hidden_states, None

    def layer_forward(layer):
        def forward(
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            use_cache=False,
            position_embeddings=None,
            **kwargs,
        ):
            hidden_states, carried = _split_residual(hidden_states)
            if carried is None:
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
            else:
                hidden_states, residual = layer.input_layernorm(hidden_states, carried)
            hidden_states, _ = layer.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
            hidden_states = layer.mlp(hidden_states)
            return torch.stack((hidden_states, residual), dim=0)

        return forward

    for layer in layers:
        layer.forward = layer_forward(layer)

    orig_norm = norm.forward
    orig_model_forward = backbone.forward

    def norm_forward(hidden_states, residual=None):
        if residual is None:
            hidden_states, residual = _split_residual(hidden_states)
        if residual is None:
            return orig_norm(hidden_states)
        out, _ = orig_norm(hidden_states, residual)
        return out

    def model_forward(*args, **kwargs):
        return orig_model_forward(*args, **kwargs)

    norm.forward = norm_forward
    backbone.forward = model_forward
    _rank0_print(
        f"> residual_rmsnorm: {len(layers)} Qwen3 layers use rmsnorm2d_fwd_with_add"
    )
