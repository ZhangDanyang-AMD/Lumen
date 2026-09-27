"""OPUS attention backend: availability, numerics vs csrc, and gating.

The OPUS forward is a gfx950 bf16-only kernel with no backward, so the backend
pairs it with the csrc (CK/asm) backward. These checks pin down the two things
that pairing depends on -- the forward matching csrc within bf16 tolerance and
the saved LSE being the quantity the csrc backward expects -- plus the fallback
behaviour for shapes OPUS does not cover.

Run inside the lumen:dev container:
    python tests/kernels/test_opus_attn_backend.py
"""

from __future__ import annotations

import torch

from lumen.kernels.attention.attention_impl import csrc_available, opus_available
from lumen.ops.attention.attention import attention

DEV = "cuda"


def _mk(b, sq, sk, hq, hk, d, dv=None, seed=0):
    torch.manual_seed(seed)
    dv = dv or d
    q = torch.randn(b, sq, hq, d, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(b, sk, hk, d, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(b, sk, hk, dv, device=DEV, dtype=torch.bfloat16)
    return q, k, v


def _run(q, k, v, backend, causal, seq_major_out=False, dout=None):
    """Run one attention call; pass `dout` to also get grads for that exact
    upstream gradient (both backends must see the same one to be comparable)."""
    q, k, v = (t.detach().clone().requires_grad_(dout is not None) for t in (q, k, v))
    out = attention(
        q,
        k,
        v,
        softmax_scale=q.shape[-1] ** -0.5,
        causal=causal,
        backend_type=backend,
        seq_major_out=seq_major_out,
    )
    if dout is None:
        return out, None
    out.backward(dout)
    return out, (q.grad, k.grad, v.grad)


def _report(tag, a, b):
    diff = (a.float() - b.float()).abs()
    rel = diff.max() / b.float().abs().max().clamp_min(1e-6)
    print(f"  {tag:<10} max_abs={diff.max():.4e}  mean_abs={diff.mean():.4e}  rel={rel:.4e}")
    return rel.item()


def main() -> None:
    gfx = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    print(f"arch={gfx}  torch={torch.__version__}")
    print(f"csrc fwd available : {csrc_available('flash_attn_fwd')}")
    print(f"opus fwd available : {csrc_available('fmha_fwd_bf16_opus_fwd')}")

    if not csrc_available("fmha_fwd_bf16_opus_fwd"):
        raise SystemExit("OPUS forward not exposed by aiter — nothing to test")

    # D=128 GQA, the shape llama2-7b / llama3-8b / qwen3-8b all attend with.
    q, k, v = _mk(2, 1024, 1024, 16, 4, 128)
    print(f"\neligible(D=128, GQA 16/4): {opus_available(q, k, v)}")

    failures = []
    for causal in (True, False):
        print(f"\n[forward] causal={causal}")
        o_opus, _ = _run(q, k, v, "aiter_opus", causal)
        o_csrc, _ = _run(q, k, v, "aiter_csrc", causal)
        rel = _report("out", o_opus, o_csrc)
        if rel > 2e-2:
            failures.append(f"forward causal={causal} rel={rel:.3e}")

    print("\n[backward] OPUS fwd + csrc bwd vs csrc fwd + csrc bwd (causal=True)")
    torch.manual_seed(1234)
    dout = torch.randn(q.shape[0], q.shape[1], q.shape[2], v.shape[3], device=DEV, dtype=torch.bfloat16)
    _, g_opus = _run(q, k, v, "aiter_opus", True, dout=dout)
    _, g_csrc = _run(q, k, v, "aiter_csrc", True, dout=dout)
    for name, a, b in zip(("dq", "dk", "dv"), g_opus, g_csrc):
        rel = _report(name, a, b)
        if rel > 5e-2:
            failures.append(f"backward {name} rel={rel:.3e}")

    print("\n[seq_major_out] output view is transpose-free")
    o_sm, _ = _run(q, k, v, "aiter_opus", True, seq_major_out=True)
    sm_ok = o_sm.permute(1, 0, 2, 3).is_contiguous()
    print(f"  permute(1,0,2,3).is_contiguous() = {sm_ok}")
    o_ref, _ = _run(q, k, v, "aiter_opus", True, seq_major_out=False)
    rel = _report("vs plain", o_sm, o_ref)
    if not sm_ok or rel > 1e-6:
        failures.append("seq_major_out mismatch")

    print("\n[gating] shapes OPUS does not cover must degrade to csrc, not raise")
    cases = {
        "fp16 dtype": tuple(t.half() for t in _mk(1, 256, 256, 8, 8, 128)),
        "hdim 64": _mk(1, 256, 256, 8, 8, 64),
    }
    for tag, (qq, kk, vv) in cases.items():
        elig = opus_available(qq, kk, vv)
        out, _ = _run(qq, kk, vv, "aiter_opus", True)
        print(f"  {tag:<12} eligible={elig}  ran={tuple(out.shape)}  dtype={out.dtype}")
        if elig:
            failures.append(f"{tag} should not be OPUS-eligible")

    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        raise SystemExit(1)
    print("all OPUS backend checks passed")


if __name__ == "__main__":
    main()
