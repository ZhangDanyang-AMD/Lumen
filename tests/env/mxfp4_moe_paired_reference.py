"""Independent dense recipe edges inside the unchanged real Core MoELayer."""

import importlib.util
import sys
from pathlib import Path

import torch


def load_recipe():
    directory = Path(__file__).resolve().parents[1] / "ops"
    for name in ("mxfp4_moe_sr_reference", "mxfp4_moe_recipe_reference"):
        path = directory / (name + ".py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules["mxfp4_moe_recipe_reference"]


_RECIPE = load_recipe()


class RecipeLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, counts, tokens, mathematical, *weights):
        ctx.save_for_backward(x, *weights)
        ctx.counts, ctx.tokens, ctx.mathematical = tuple(counts), tokens, mathematical
        outputs, start = [], 0
        for count, weight in zip(counts, weights):
            source = x[start : start + count]
            if mathematical:
                value = source.float() @ weight.float().T
            else:
                value = (
                    _RECIPE.quantize_reference(source)
                    @ _RECIPE.quantize_reference(weight, 32).T
                )
            outputs.append(value.to(torch.bfloat16))
            start += count
        return torch.cat(outputs)

    @staticmethod
    def backward(ctx, dy):
        x, *weights = ctx.saved_tensors
        gradients, weight_gradients, start = [], [], 0
        for expert, (count, weight) in enumerate(zip(ctx.counts, weights)):
            source, grad = (
                x[start : start + count],
                dy[start : start + count].contiguous(),
            )
            if count == 0:
                dx, dw = torch.zeros_like(source), torch.zeros_like(
                    weight, dtype=torch.float32
                )
            elif ctx.mathematical:
                _, dx, dw = _RECIPE.bf16_recipe(source, weight, grad)
            else:
                row, transformed = ctx.tokens
                _, dx, dw = _RECIPE.quantized_recipe(
                    source,
                    weight,
                    grad,
                    row_seed=row.seed,
                    row_base=row.base_draw + row.route.prefixes(1)[expert],
                    h16_seed=transformed.seed,
                    h16_base=transformed.base_draw
                    + transformed.route.prefixes(2)[expert],
                )
            gradients.append(dx.to(torch.bfloat16))
            # Record FP32 before autograd casts the gradient for a BF16 Parameter.
            weight._paired_fp32_grad = dw.detach().clone()
            weight_gradients.append(dw.to(weight.dtype))
            start += count
        return torch.cat(gradients), None, None, None, *weight_gradients


def install_reference(layers, captures, *, mathematical):
    for index, layer in enumerate(layers):
        for fc, module in enumerate(
            (layer.experts.linear_fc1, layer.experts.linear_fc2), 1
        ):

            def forward(x, counts, *, owner=module, key=(index, fc)):
                record = captures[key]
                assert tuple(counts) == record["counts"], "paired route changed"
                weights = tuple(
                    getattr(owner, "weight" + str(slot))
                    for slot in range(owner.num_gemms)
                )
                return (
                    RecipeLinear.apply(
                        x, tuple(counts), record["tokens"], mathematical, *weights
                    ),
                    None,
                )

            module.forward = forward


def metric(reference, actual, floor, *, elements=False):
    ref, out = reference.float(), actual.float()
    assert torch.isfinite(out).all()
    error = (ref - out).square().sum()
    snr = (
        float("inf")
        if error == 0
        else float(10 * torch.log10(ref.square().sum() / error))
    )
    assert snr >= floor, (snr, floor)
    if elements and ref.numel():
        assert (
            float(((ref - out).abs() <= 0.5 + 0.02 * ref.abs()).float().mean()) >= 0.99
        )
    norm = float((ref - out).norm() / ref.norm().clamp_min(1e-30))
    maximum = float((ref - out).abs().max()) if ref.numel() else 0.0
    return {"snr_db": snr, "relative_norm_error": norm, "max_abs_error": maximum}
