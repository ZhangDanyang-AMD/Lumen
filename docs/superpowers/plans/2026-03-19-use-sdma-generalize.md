# `--use-sdma` Generalization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote `--use-sdma` from an FP8-amax-only flag to a general SDMA switch that gates all supported SDMA communication paths (CP A2A, per-step amax reduction, TP comm).

**Architecture:** Move the `--use-sdma` CLI flag from the `linear-fp8` arg group to the `lumen` arg group in `megatron.py`. Add it to FSDP trainers. Wire it into `cp_param_bundle` for CP A2A SDMA, and add an SDMA branch in `ScalingManager.get_scale` for per-step amax reduction.

**Tech Stack:** Python, PyTorch, Lumen (AMD GPU library), mori SDMA

**Spec:** `docs/superpowers/specs/2026-03-19-use-sdma-generalize-design.md`

---

### Task 1: Move `--use-sdma` from `linear-fp8` to `lumen` group in `megatron.py`

**Files:**
- Modify: `lumen/models/megatron.py:1236-1242` (remove from lfp8), `lumen/models/megatron.py:~1175` (add to lumen group)

- [ ] **Step 1: Remove `--use-sdma` from `lfp8` group**

In `lumen/models/megatron.py`, delete lines 1236-1242:

```python
    safe_add_argument(
        lfp8,
        "--use-sdma",
        action="store_true",
        default=False,
        help="Use mori SDMA for amax all-reduce instead of torch.distributed.",
    )
```

- [ ] **Step 2: Add `--use-sdma` to `lumen` group**

In `lumen/models/megatron.py`, after the `--lumen-tp-comm-overlap-method` block (after line ~1175), add:

```python
    safe_add_argument(
        lumen,
        "--use-sdma",
        action="store_true",
        default=False,
        help="Use mori SDMA instead of torch.distributed for supported collectives "
        "(TP comm, amax all-reduce, CP all-to-all) when available.",
    )
```

- [ ] **Step 3: Verify no references break**

Run: `cd Lumen && python -c "from lumen.models.megatron import add_common_megatron_args; print('OK')"`
Expected: `OK` (no import error)

- [ ] **Step 4: Commit**

```bash
git add lumen/models/megatron.py
git commit -m "refactor: move --use-sdma from linear-fp8 to lumen arg group"
```

---

### Task 2: Add `--use-sdma` to FSDP trainers

**Files:**
- Modify: `lumen/models/fsdp.py:73-87` (add_common_fsdp_args, FSDP group)
- Modify: `lumen/models/llama2/fsdp/sft.py:417-428` (FSDP group)
- Modify: `lumen/models/llama31/fsdp/pretrain.py:375-386` (FSDP group)

- [ ] **Step 1: Add `--use-sdma` to `add_common_fsdp_args`**

In `lumen/models/fsdp.py`, in the FSDP arg group (after `--fsdp-version`, ~line 87), add:

```python
    f.add_argument(
        "--use-sdma",
        action="store_true",
        default=False,
        help="Use mori SDMA instead of torch.distributed for supported collectives "
        "(TP comm, amax all-reduce, CP all-to-all) when available.",
    )
```

- [ ] **Step 2: Add `--use-sdma` to LLaMA2 FSDP SFT trainer**

In `lumen/models/llama2/fsdp/sft.py`, in the FSDP arg group (after `--fsdp-version`, ~line 428), add:

```python
    f.add_argument(
        "--use-sdma",
        action="store_true",
        default=False,
        help="Use mori SDMA instead of torch.distributed for supported collectives "
        "(TP comm, amax all-reduce, CP all-to-all) when available.",
    )
```

- [ ] **Step 3: Add `--use-sdma` to LLaMA3.1 FSDP pretrain trainer**

In `lumen/models/llama31/fsdp/pretrain.py`, in the FSDP arg group (after `--fsdp-version`, ~line 386), add:

```python
    f.add_argument(
        "--use-sdma",
        action="store_true",
        default=False,
        help="Use mori SDMA instead of torch.distributed for supported collectives "
        "(TP comm, amax all-reduce, CP all-to-all) when available.",
    )
```

- [ ] **Step 4: Write test — `--use-sdma` parseable in FSDP trainers**

In `tests/models/test_fsdp.py`, add a test class (or add to existing) that verifies `add_common_fsdp_args` registers `--use-sdma`:

```python
class TestUseSdmaArg:
    def test_add_common_fsdp_args_has_use_sdma(self):
        import argparse
        from lumen.models.fsdp import add_common_fsdp_args

        parser = argparse.ArgumentParser()
        add_common_fsdp_args(parser)
        args = parser.parse_args([])
        assert args.use_sdma is False

    def test_add_common_fsdp_args_use_sdma_true(self):
        import argparse
        from lumen.models.fsdp import add_common_fsdp_args

        parser = argparse.ArgumentParser()
        add_common_fsdp_args(parser)
        args = parser.parse_args(["--use-sdma"])
        assert args.use_sdma is True
```

- [ ] **Step 5: Run tests**

Run: `cd Lumen && python -m pytest tests/models/test_fsdp.py::TestUseSdmaArg -v`
Expected: 2 PASSED

- [ ] **Step 6: Commit**

```bash
git add lumen/models/fsdp.py lumen/models/llama2/fsdp/sft.py lumen/models/llama31/fsdp/pretrain.py tests/models/test_fsdp.py
git commit -m "feat: add --use-sdma to FSDP trainers"
```

---

### Task 3: Wire CP A2A SDMA in `attention_megatron.py`

**Files:**
- Modify: `lumen/modules/attention_megatron.py:174-181`
- Test: `tests/module/test_attention_megatron_module.py`

- [ ] **Step 1: Add `"use_sdma"` to `cp_param_bundle` in `attention_megatron.py`**

In `lumen/modules/attention_megatron.py`, the current `cp_param_bundle` construction (~line 174-181):

```python
        cp_param_bundle = None
        if self.cp_size > 1:
            cp_group = parallel_state.get_context_parallel_group()
            cp_comm_type = self.cp_comm_type or getattr(get_args(), "lumen_cp_comm_type", "a2a")
            cp_param_bundle = {
                "cp_group": cp_group,
                "cp_comm_type": cp_comm_type,
            }
```

Replace with:

```python
        cp_param_bundle = None
        if self.cp_size > 1:
            cp_group = parallel_state.get_context_parallel_group()
            cp_comm_type = self.cp_comm_type or getattr(get_args(), "lumen_cp_comm_type", "a2a")
            cp_param_bundle = {
                "cp_group": cp_group,
                "cp_comm_type": cp_comm_type,
                "use_sdma": _use_sdma_from_args(),
            }
```

Also add the import at the top of the file (with the other imports from `lumen.modules`):

```python
from lumen.modules.parallel_linear import _use_sdma_from_args
```

- [ ] **Step 2: Write test — `cp_param_bundle` includes `use_sdma`**

In `tests/module/test_attention_megatron_module.py`, add a test that constructs `LumenDotProductAttention` with `cp_size > 1`, mocks `get_args()` to return `use_sdma=True`, mocks `parallel_state.get_context_parallel_group`, and verifies `cp_param_bundle["use_sdma"]` is passed through.

Since the `forward` method builds `cp_param_bundle` internally and passes it to `attention()`, mock `attention` to capture the kwargs:

```python
class TestCpParamBundleUseSdma:
    @mock.patch("lumen.modules.attention_megatron.AttnMaskType", _MockAttnMaskType)
    @mock.patch("lumen.modules.attention_megatron.divide", side_effect=lambda a, b: a // b)
    @mock.patch("lumen.modules.attention_megatron.MegatronModule.__init__", _patched_megatron_init)
    @mock.patch("lumen.modules.attention_megatron.get_args", return_value=_make_args())
    @mock.patch("lumen.modules.attention_megatron.parallel_state")
    def test_cp_param_bundle_includes_use_sdma(self, mock_ps, *_):
        mock_ps.get_context_parallel_group.return_value = "fake_cp_group"

        from lumen.modules.attention_megatron import LumenDotProductAttention

        config = _make_config()
        config.context_parallel_size = 2
        attn = LumenDotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=_MockAttnMaskType.causal,
            attention_type="self",
        )

        with mock.patch(
            "lumen.modules.attention_megatron._use_sdma_from_args", return_value=True
        ), mock.patch(
            "lumen.modules.attention_megatron.attention"
        ) as mock_attn:
            mock_attn.return_value = torch.randn(2, 4, 8, 64, device="cuda")
            sq, b, np_, hn = 4, 2, 8, 64
            q = torch.randn(sq, b, np_, hn, device="cuda")
            k = torch.randn(sq, b, 8, hn, device="cuda")
            v = torch.randn(sq, b, 8, hn, device="cuda")
            attn(q, k, v, attention_mask=None)

            _, kwargs = mock_attn.call_args
            bundle = kwargs["cp_param_bundle"]
            assert bundle is not None
            assert bundle["use_sdma"] is True
            assert bundle["cp_group"] == "fake_cp_group"
```

- [ ] **Step 3: Run tests**

Run: `cd Lumen && python -m pytest tests/module/test_attention_megatron_module.py::TestCpParamBundleUseSdma -v`
Expected: 1 PASSED

- [ ] **Step 4: Commit**

```bash
git add lumen/modules/attention_megatron.py tests/module/test_attention_megatron_module.py
git commit -m "feat: wire --use-sdma into CP A2A via cp_param_bundle (megatron attn)"
```

---

### Task 4: Wire CP A2A SDMA in `attention_mla.py`

**Files:**
- Modify: `lumen/modules/attention_mla.py:148-155`
- Test: `tests/module/test_attention_mla_module.py`

- [ ] **Step 1: Add `"use_sdma"` to `cp_param_bundle` in `attention_mla.py`**

In `lumen/modules/attention_mla.py`, the current `cp_param_bundle` construction (~line 148-155):

```python
        cp_param_bundle = None
        if self.cp_size > 1:
            cp_group = parallel_state.get_context_parallel_group()
            cp_comm_type = self.cp_comm_type or getattr(get_args(), "lumen_cp_comm_type", "a2a")
            cp_param_bundle = {
                "cp_group": cp_group,
                "cp_comm_type": cp_comm_type,
            }
```

Replace with:

```python
        cp_param_bundle = None
        if self.cp_size > 1:
            cp_group = parallel_state.get_context_parallel_group()
            cp_comm_type = self.cp_comm_type or getattr(get_args(), "lumen_cp_comm_type", "a2a")
            cp_param_bundle = {
                "cp_group": cp_group,
                "cp_comm_type": cp_comm_type,
                "use_sdma": _use_sdma_from_args(),
            }
```

Also add the import at the top of the file:

```python
from lumen.modules.parallel_linear import _use_sdma_from_args
```

- [ ] **Step 2: Write test — `cp_param_bundle` includes `use_sdma` in MLA**

In `tests/module/test_attention_mla_module.py`, add a similar test to Task 3. The MLA module has different constructor args (`k_head_dim`, `v_head_dim` via config). Look at the existing test file for the exact `_make_config` / mock pattern and add a `TestCpParamBundleUseSdma` class that mirrors Task 3's test structure, adapted for the MLA module's constructor and forward signature.

Key differences from megatron attn:
- MLA config needs `qk_rope_head_dim` (k_head_dim = kv_channels + qk_rope_head_dim)
- MLA pads V when k_head_dim != v_head_dim

- [ ] **Step 3: Run tests**

Run: `cd Lumen && python -m pytest tests/module/test_attention_mla_module.py::TestCpParamBundleUseSdma -v`
Expected: 1 PASSED

- [ ] **Step 4: Commit**

```bash
git add lumen/modules/attention_mla.py tests/module/test_attention_mla_module.py
git commit -m "feat: wire --use-sdma into CP A2A via cp_param_bundle (MLA attn)"
```

---

### Task 5: Add SDMA path to `ScalingManager.get_scale`

**Files:**
- Modify: `lumen/quantize/scaling_manager.py:166-199` (get_scale method)
- Test: `tests/quantize/test_scaling_manager.py`

- [ ] **Step 1: Add `_reduce_single_amax_sdma` method**

In `lumen/quantize/scaling_manager.py`, after the `_reduce_fp8_amax_dist` method (~line 353), add:

```python
    def _reduce_single_amax_sdma(self, amax: torch.Tensor) -> torch.Tensor:
        """Reduce a single amax scalar via SDMA (allgather + max)."""
        from lumen.ops.sdma import sdma_allgather_max

        packed = amax.float().unsqueeze(0)
        if self._sdma_allgather is None:
            from lumen.ops.sdma import SdmaAllgather

            self._sdma_allgather = SdmaAllgather()
        result = sdma_allgather_max(packed, self._sdma_allgather)
        return result[0]
```

- [ ] **Step 2: Modify `get_scale` to branch on SDMA**

In `lumen/quantize/scaling_manager.py`, in the `get_scale` method, replace the two `torch.distributed.all_reduce` blocks.

**Delayed branch** (~line 183-188):

Replace:

```python
            if self.config.reduce_amax and self._dp_group is not None:
                torch.distributed.all_reduce(
                    amax,
                    op=torch.distributed.ReduceOp.MAX,
                    group=self._dp_group,
                )
```

With:

```python
            if self.config.reduce_amax and self._dp_group is not None:
                if self._use_sdma:
                    amax = self._reduce_single_amax_sdma(amax)
                else:
                    torch.distributed.all_reduce(
                        amax,
                        op=torch.distributed.ReduceOp.MAX,
                        group=self._dp_group,
                    )
```

**Dynamic branch** (~line 193-198):

Replace:

```python
            if self.config.reduce_amax and self._dp_group is not None:
                torch.distributed.all_reduce(
                    amax,
                    op=torch.distributed.ReduceOp.MAX,
                    group=self._dp_group,
                )
```

With:

```python
            if self.config.reduce_amax and self._dp_group is not None:
                if self._use_sdma:
                    amax = self._reduce_single_amax_sdma(amax)
                else:
                    torch.distributed.all_reduce(
                        amax,
                        op=torch.distributed.ReduceOp.MAX,
                        group=self._dp_group,
                    )
```

- [ ] **Step 3: Write test — `get_scale` uses SDMA path when `_use_sdma=True`**

In `tests/quantize/test_scaling_manager.py`, add inside the existing `TestGetScale` class or create a new class:

```python
class TestGetScaleSdma:
    """Verify get_scale routes to SDMA when _use_sdma is True."""

    def test_delayed_uses_sdma_when_enabled(self):
        from unittest import mock as umock

        mgr = ScalingManager(recipe="delayed")
        mgr._use_sdma = True
        mgr._dp_group = "fake_dp_group"
        mgr.config.reduce_amax = True

        mgr.amax_history["x"].append(torch.tensor(2.0, device="cuda"))

        with umock.patch.object(
            mgr, "_reduce_single_amax_sdma", return_value=torch.tensor(2.0, device="cuda")
        ) as mock_sdma:
            scale = mgr.get_scale("x", torch.randn(4, 8, device="cuda"))
            mock_sdma.assert_called_once()
            assert scale is not None

    def test_dynamic_uses_sdma_when_enabled(self):
        from unittest import mock as umock

        mgr = ScalingManager(recipe="dynamic")
        mgr._use_sdma = True
        mgr._dp_group = "fake_dp_group"
        mgr.config.reduce_amax = True

        with umock.patch.object(
            mgr, "_reduce_single_amax_sdma", return_value=torch.tensor(2.0, device="cuda")
        ) as mock_sdma:
            scale = mgr.get_scale("x", torch.randn(4, 8, device="cuda"))
            mock_sdma.assert_called_once()
            assert scale is not None

    def test_delayed_uses_dist_when_sdma_disabled(self):
        from unittest import mock as umock

        mgr = ScalingManager(recipe="delayed")
        mgr._use_sdma = False
        mgr._dp_group = "fake_dp_group"
        mgr.config.reduce_amax = True

        mgr.amax_history["x"].append(torch.tensor(2.0, device="cuda"))

        with umock.patch("torch.distributed.all_reduce") as mock_dist:
            mgr.get_scale("x", torch.randn(4, 8, device="cuda"))
            mock_dist.assert_called_once()

    def test_reduce_single_amax_sdma_casts_to_float32(self):
        """Verify _reduce_single_amax_sdma casts input to float32."""
        from unittest import mock as umock

        mgr = ScalingManager(recipe="delayed")
        mgr._use_sdma = True

        fake_result = torch.tensor([3.0], device="cuda")
        with umock.patch(
            "lumen.ops.sdma.sdma_allgather_max", return_value=fake_result
        ) as mock_fn, umock.patch(
            "lumen.ops.sdma.SdmaAllgather"
        ):
            amax_bf16 = torch.tensor(2.0, device="cuda", dtype=torch.bfloat16)
            result = mgr._reduce_single_amax_sdma(amax_bf16)
            call_args = mock_fn.call_args[0][0]
            assert call_args.dtype == torch.float32
            assert result.item() == pytest.approx(3.0)
```

- [ ] **Step 4: Run tests**

Run: `cd Lumen && python -m pytest tests/quantize/test_scaling_manager.py::TestGetScaleSdma -v`
Expected: 4 PASSED

- [ ] **Step 5: Run existing `TestGetScale` to ensure no regressions**

Run: `cd Lumen && python -m pytest tests/quantize/test_scaling_manager.py::TestGetScale -v`
Expected: all PASSED

- [ ] **Step 6: Commit**

```bash
git add lumen/quantize/scaling_manager.py tests/quantize/test_scaling_manager.py
git commit -m "feat: add SDMA path to ScalingManager.get_scale for per-step amax reduction"
```

---

### Task 6: Final verification

- [ ] **Step 1: Run all affected test suites**

```bash
cd Lumen
python -m pytest tests/models/test_fsdp.py -v
python -m pytest tests/module/test_attention_megatron_module.py -v
python -m pytest tests/module/test_attention_mla_module.py -v
python -m pytest tests/quantize/test_scaling_manager.py -v
python -m pytest tests/ops/test_sdma.py -v
```

Expected: all PASSED

- [ ] **Step 2: Verify Megatron arg parse end-to-end (no duplicate `--use-sdma`)**

```bash
cd Lumen
python -c "
import argparse
from lumen.models.megatron import add_common_megatron_args
parser = argparse.ArgumentParser()
add_common_megatron_args(parser)
args = parser.parse_args(['--use-sdma'])
assert args.use_sdma is True
print('megatron --use-sdma OK')
"
```

Expected: `megatron --use-sdma OK`

- [ ] **Step 3: Verify FSDP arg parse end-to-end**

```bash
cd Lumen
python -c "
import argparse
from lumen.models.fsdp import add_common_fsdp_args
parser = argparse.ArgumentParser()
add_common_fsdp_args(parser)
args = parser.parse_args(['--use-sdma'])
assert args.use_sdma is True
print('fsdp --use-sdma OK')
"
```

Expected: `fsdp --use-sdma OK`
