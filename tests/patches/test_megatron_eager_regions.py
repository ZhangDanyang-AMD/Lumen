"""Tests for the eager replacements of tiny Megatron ``jit_fuser`` regions."""

import sys
import types
from pathlib import Path

import pytest

if "lumen" not in sys.modules:
    _lumen = types.ModuleType("lumen")
    _lumen.__path__ = [str(Path(__file__).resolve().parents[2] / "lumen")]
    sys.modules["lumen"] = _lumen

from lumen.patches import PatchPhase, PatchRegistry

torch = pytest.importorskip("torch")


def _load():
    for key in ("lumen.patches.runtime.megatron_eager_regions", "lumen.patches.runtime"):
        sys.modules.pop(key, None)
    from lumen.patches.runtime import megatron_eager_regions

    return megatron_eager_regions


def _stub_package(monkeypatch, name, module):
    parts = name.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent not in sys.modules:
            monkeypatch.setitem(sys.modules, parent, types.ModuleType(parent))
    monkeypatch.setitem(sys.modules, name, module)


class TestEagerTinyRegions:
    def setup_method(self):
        PatchRegistry.clear()
        self.mod = _load()

    def test_registered_as_default_import_patch(self, monkeypatch):
        spec = PatchRegistry.get("eager_tiny_regions")
        assert spec.phase is PatchPhase.IMPORT
        assert spec.default is True
        monkeypatch.setenv("LUMEN_EAGER_TINY_REGIONS", "0")
        assert not spec.enabled()
        monkeypatch.setenv("LUMEN_EAGER_TINY_REGIONS", "1")
        assert spec.enabled()

    def test_plain_residual_add_runs_eagerly(self, monkeypatch):
        calls = []
        bda = types.ModuleType("megatron.core.fusions.fused_bias_dropout")

        def compiled(x_with_bias, residual, prob):
            calls.append(prob)
            return residual

        bda.bias_dropout_add_fused_train = compiled
        _stub_package(monkeypatch, "megatron.core.fusions.fused_bias_dropout", bda)

        self.mod.install_eager_bias_dropout_add()
        self.mod.install_eager_bias_dropout_add()
        x = torch.randn(4, 2, 8, dtype=torch.bfloat16)
        residual = torch.randn(4, 2, 8, dtype=torch.bfloat16)

        out = bda.bias_dropout_add_fused_train((x, None), residual, 0.0)
        torch.testing.assert_close(out, residual + x, rtol=0, atol=0)
        assert calls == []

        bda.bias_dropout_add_fused_train((x, None), residual, 0.1)
        bda.bias_dropout_add_fused_train((x, torch.zeros(8, dtype=x.dtype)), residual, 0.0)
        bda.bias_dropout_add_fused_train((x, None), residual.float(), 0.0)
        assert calls == [0.1, 0.0, 0.0]

    def test_eager_add_matches_compiled_megatron_body(self, monkeypatch):
        """The eager path must equal Megatron's fused function bit for bit, values and grads."""
        if not torch.cuda.is_available():
            pytest.skip("Inductor Triton codegen needs a GPU")

        def body(x_with_bias, residual, prob):
            x, bias = x_with_bias
            out = torch.nn.functional.dropout(x, p=prob, training=True, inplace=False)
            return residual + out

        compiled = torch.compile(body)
        bda = types.ModuleType("megatron.core.fusions.fused_bias_dropout")
        bda.bias_dropout_add_fused_train = compiled
        _stub_package(monkeypatch, "megatron.core.fusions.fused_bias_dropout", bda)
        self.mod.install_eager_bias_dropout_add()
        eager = bda.bias_dropout_add_fused_train
        for dtype in (torch.bfloat16, torch.float32):
            x = torch.randn(4096, 1, 2048, device="cuda", dtype=dtype) * 7
            residual = torch.randn(4096, 1, 2048, device="cuda", dtype=dtype) * 3
            grad = torch.randn_like(x)
            outs = []
            for fn in (compiled, eager):
                xi = x.clone().requires_grad_(True)
                ri = residual.clone().requires_grad_(True)
                out = fn((xi, None), ri, 0.0)
                out.backward(grad)
                outs.append((out.detach(), xi.grad, ri.grad))
            for ref, got in zip(*outs):
                assert torch.equal(ref, got)

    def test_flex_dispatch_preprocess_uses_uncompiled_body(self, monkeypatch):
        td = types.ModuleType("megatron.core.transformer.moe.token_dispatcher")

        def dispatch_preprocess(self, hidden_states, routing_map, probs):
            return hidden_states, probs

        def compiled(*args, **kwargs):
            raise AssertionError("compiled region must not run")

        compiled._torchdynamo_orig_callable = dispatch_preprocess

        class MoEFlexTokenDispatcher:
            pass

        MoEFlexTokenDispatcher.dispatch_preprocess = compiled
        td.MoEFlexTokenDispatcher = MoEFlexTokenDispatcher
        _stub_package(monkeypatch, "megatron.core.transformer.moe.token_dispatcher", td)

        self.mod.install_eager_flex_dispatch_preprocess()
        assert MoEFlexTokenDispatcher.dispatch_preprocess is dispatch_preprocess
        self.mod.install_eager_flex_dispatch_preprocess()
        assert MoEFlexTokenDispatcher.dispatch_preprocess is dispatch_preprocess

    def test_missing_megatron_is_a_no_op(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _block(name, globals=None, locals=None, fromlist=(), level=0):
            if name.startswith("megatron"):
                raise ImportError(name)
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _block)
        self.mod.install_eager_tiny_regions()
