"""Connect MXFP4 public enable to native linears; leave legacy FP8 enablement alone."""

import logging

logger = logging.getLogger(__name__)

_NATIVE_LINEAR_CLASSES = {
    ("lumen.modules.parallel_linear", "LumenColumnParallelLinear"),
    ("lumen.modules.parallel_linear", "LumenRowParallelLinear"),
    ("lumen.modules.layernorm_linear", "LumenLayerNormLinear"),
    ("lumen.modules.grouped_linear", "LumenGroupedLinear"),
}


def is_native_linear(module):
    # Inspect existing types so an nn.Linear-only model need not import Megatron.
    return any((base.__module__, base.__name__) in _NATIVE_LINEAR_CLASSES
               for base in type(module).__mro__)


def weight_parameters(module):
    weight = getattr(module, "weight", None)
    if weight is not None and weight.dim() == 2:
        return (weight,)
    return tuple(param for name, param in module._parameters.items()
                 if name.startswith("weight") and name[6:].isdigit()
                 and param is not None and param.dim() == 2)


def enable_mxfp4_native_linears(
    model, config, *, gradient_accumulation_fusion=False, delay_wgrad=False, fp8_mha=False,
):
    """Apply the shared recipe and layer policy, keeping a manager per native layer."""
    from lumen.quantize import ScalingManager, _build_bf16_skip_prefixes, _is_output_layer, is_under_bf16_prefix
    from lumen.quantize.config import MXFP4_BLOCK_SIZE, QuantFormat

    if config.format != QuantFormat.MXFP4 or not config.is_quantized:
        return 0
    targets = [(name, module) for name, module in model.named_modules() if is_native_linear(module)]
    if not targets:
        return 0
    from lumen.quantize import assert_mxfp4_arch_supported

    assert_mxfp4_arch_supported()
    prefixes = _build_bf16_skip_prefixes(model, config)
    count = 0
    for name, module in targets:
        if (is_under_bf16_prefix(name, prefixes)
                or (not config.quantize_output_layer and _is_output_layer(name))
                or "lora_" in name):
            continue
        if config.format == QuantFormat.MXFP4 and any(
            weight.shape[0] % MXFP4_BLOCK_SIZE for weight in weight_parameters(module)
        ):
            logger.warning("%s: MXFP4 output width must be a multiple of 32; keeping BF16", name)
            continue
        manager = ScalingManager(config)
        module.enable_fp8(scaling_manager=manager, scaling_type=config.recipe,
                          fp8_dtype=config.torch_dtype, block_size=config.block_size)
        module._lumen_mxfp4_native = True
        # Existing FSDP communication and cache consumers inspect these MX-only markers.
        module._quant_manager = manager
        module._quant_enabled = True
        module._quant_scaling_type = config.recipe
        module.gradient_accumulation_fusion = gradient_accumulation_fusion
        module.delay_wgrad = delay_wgrad
        count += 1
    if fp8_mha:
        from lumen.modules.attention_megatron import LumenDotProductAttention
        from lumen.modules.attention_mla import LumenDotProductAttentionMLA
        from lumen.quantize.scaling_manager import Blockwise2DScaleManager

        for module in model.modules():
            if isinstance(module, (LumenDotProductAttention, LumenDotProductAttentionMLA)):
                module.scale_manager = Blockwise2DScaleManager()
    if count:
        from lumen.models._swiglu_fp8_fuse import set_fused_swiglu_scaling

        set_fused_swiglu_scaling(config.recipe, config.block_size)
        logger.info("Enabled %s on %d native linear modules", config.recipe, count)
    return count
