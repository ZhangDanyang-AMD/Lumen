"""Patch Megatron files to fix FusedLayerNorm + RMSNorm incompatibility.

FusedLayerNorm (from apex) does not support RMSNorm normalization, but
several Megatron files default to FusedLayerNorm when apex is present.
This script:

1. Creates a custom MegatronFusedRMSNorm wrapper in megatron/core/transformer/
   that wraps apex's FusedRMSNorm with Megatron's interface and supports
   sequence_parallel.
2. Patches gpt_layer_specs.py to use MegatronFusedRMSNorm when RMSNorm is detected.
3. Patches transformer_block.py to use MegatronFusedRMSNorm instead of FusedLayerNorm.
"""

import pathlib
import sys

megatron_root = sys.argv[1]
MARKER = "# patched-lnimpl"

files_patched = 0

# --- Step 0: Create MegatronFusedRMSNorm wrapper ---
norm_dir = pathlib.Path(megatron_root) / "megatron" / "core" / "transformer"
wrapper_path = norm_dir / "megatron_fused_rmsnorm.py"
if not wrapper_path.exists():
    wrapper_path.write_text(
        '''\
"""Megatron-compatible wrapper around apex FusedRMSNorm.

Supports the TransformerConfig constructor signature and
sequence_parallel attribute on weights.
"""

import numbers
import torch
from torch.nn.parameter import Parameter
from torch.nn import init

from apex.normalization.fused_layer_norm import (
    fused_rms_norm_affine,
    fused_rms_norm,
    manual_rms_norm,
)

from megatron.core.transformer import TransformerConfig
from megatron.core.utils import make_viewless_tensor


class MegatronFusedRMSNorm(torch.nn.Module):
    """RMSNorm using apex fused kernels, compatible with Megatron layer specs.

    Accepts the same (config, hidden_size, eps, ...) signature that Megatron
    layer-spec build_module() passes, and sets sequence_parallel on weights
    so gradient all-reduce works correctly with SP.
    """

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
        persist_layer_norm: bool = True,
        zero_centered_gamma: bool = False,
        normalization: str = "RMSNorm",
    ):
        super().__init__()
        self.config = config
        self.zero_centered_gamma = config.layernorm_zero_centered_gamma

        if isinstance(hidden_size, numbers.Integral):
            hidden_size = (hidden_size,)
        self.hidden_size = torch.Size(hidden_size)
        self.eps = eps

        self.weight = Parameter(torch.empty(*self.hidden_size))
        self.reset_parameters()

        self.sequence_parallel = config.sequence_parallel
        setattr(self.weight, "sequence_parallel", self.sequence_parallel)

    def reset_parameters(self):
        if self.zero_centered_gamma:
            init.zeros_(self.weight)
        else:
            init.ones_(self.weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight + 1 if self.zero_centered_gamma else self.weight

        if not input.is_cuda:
            return manual_rms_norm(input, self.hidden_size, weight, self.eps)

        output = fused_rms_norm_affine(input, weight, self.hidden_size, self.eps, False)
        return make_viewless_tensor(
            inp=output, requires_grad=input.requires_grad, keep_graph=True
        )
'''
    )
    print("[Patch] Created megatron_fused_rmsnorm.py")
    files_patched += 1
else:
    print("[Patch] megatron_fused_rmsnorm.py: already exists")

# --- Patch 1: gpt_layer_specs.py ---
gls_path = pathlib.Path(megatron_root) / "megatron" / "core" / "models" / "gpt" / "gpt_layer_specs.py"
if gls_path.exists():
    src = gls_path.read_text()
    if MARKER not in src and "        layer_norm_impl = LNImpl" in src:
        src = src.replace(
            "        layer_norm_impl = LNImpl",
            "        from megatron.core.transformer.megatron_fused_rmsnorm import MegatronFusedRMSNorm as _MFRN\n"
            '        layer_norm_impl = _MFRN if normalization == "RMSNorm" else LNImpl  ' + MARKER,
            1,
        )
        gls_path.write_text(src)
        files_patched += 1
        print("[Patch] gpt_layer_specs.py: fixed LNImpl for RMSNorm")
    else:
        print("[Patch] gpt_layer_specs.py: already patched or target not found")

# --- Patch 2: transformer_block.py ---
tb_path = pathlib.Path(megatron_root) / "megatron" / "core" / "transformer" / "transformer_block.py"
if tb_path.exists():
    src = tb_path.read_text()
    if MARKER not in src:
        old_block = "elif HAVE_APEX:\n    LayerNormImpl = FusedLayerNorm"
        new_block = (
            "elif HAVE_APEX:\n"
            "    from megatron.core.transformer.megatron_fused_rmsnorm import MegatronFusedRMSNorm  " + MARKER + "\n"
            "    LayerNormImpl = MegatronFusedRMSNorm"
        )
        if old_block in src:
            src = src.replace(old_block, new_block, 1)
            tb_path.write_text(src)
            files_patched += 1
            print("[Patch] transformer_block.py: fixed LayerNormImpl for RMSNorm")
        else:
            print("[Patch] transformer_block.py: target not found, skipping")
    else:
        print("[Patch] transformer_block.py: already patched")

print(f"[Patch] {files_patched} file(s) patched")
