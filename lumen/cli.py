"""Shared linear-quantization CLI contract, without framework imports.

Register the group on any argparse parser (or argument group)::

    from lumen.cli import add_linear_quant_args

    add_linear_quant_args(parser, fsdp_aliases=True)

Then resolve it once with ``LumenConfig.from_args(args)``. The generic knobs are::

    --linear-quant-format mxfp4 --linear-quant-scaling blockwise
    --linear-quant-format fp8_e4m3 --linear-quant-block-size 128

``--linear-fp4``/``--linear-fp8`` and the ``--linear-fp8-*`` names remain
compatibility aliases; giving one field two different explicit values is an
error, and parser defaults never override an explicit MXFP4 recipe.
"""

import argparse


LINEAR_QUANT_ARG_MAP = {
    "format": ("linear_quant_format", "linear_fp8_format", "lumen_fp8_format"),
    "scaling": ("linear_quant_scaling", "linear_fp8_scaling"),
    "block_size": ("linear_quant_block_size", "linear_fp8_block_size"),
    "amax_algo": ("linear_quant_amax_algo", "linear_fp8_amax_algo"),
    "margin": ("linear_quant_margin", "linear_fp8_margin"),
    "reduce_amax": ("linear_quant_reduce_amax", "linear_fp8_reduce_amax"),
    "history_len": ("linear_quant_amax_history", "linear_fp8_amax_history"),
    "quantize_activation": ("linear_quant_activation", "linear_fp8_activation"),
    "fp8_wgrad": ("linear_quant_wgrad", "linear_fp8_wgrad"),
    "quantize_output_layer": (
        "linear_quant_output_layer", "linear_fp8_quantize_output_layer", "quantize_output_layer",
    ),
}

FORMATS = ("fp8_e4m3", "fp8_e5m2", "hybrid", "mxfp8", "mxfp4")
SCALINGS = ("dynamic", "delayed", "blockwise", "blockwise2d", "per_token", "none")


def normalize_scaling(value):
    return {
        "current": "dynamic", "current_per_tensor": "dynamic",
        "delayed_per_tensor": "delayed", "no_quant": "none",
    }.get(value, value)


class _QuantArgument(argparse.Action):
    """Retain explicit values, including explicit defaults and negative booleans."""

    def __call__(self, parser, namespace, values, option_string=None):
        value = self.const if self.nargs == 0 else values
        explicit = dict(getattr(namespace, "_lumen_quant_explicit", {}))
        if self.dest in explicit and explicit[self.dest] != value:
            raise argparse.ArgumentError(self, f"conflicting repeated value for {option_string}")
        explicit[self.dest] = value
        namespace._lumen_quant_explicit = explicit
        setattr(namespace, self.dest, value)


class _TrackedExistingArgument(argparse.Action):
    """Delegate an embedding parser's action while retaining input provenance."""

    def __init__(self, original):
        self.__dict__.update(vars(original))
        self._original = original

    def __call__(self, parser, namespace, values, option_string=None):
        self._original(parser, namespace, values, option_string)
        value = getattr(namespace, self.dest, None)
        if self.dest.endswith("scaling"):
            value = normalize_scaling(value)
        explicit = dict(getattr(namespace, "_lumen_quant_explicit", {}))
        if self.dest in explicit and explicit[self.dest] != value:
            raise argparse.ArgumentError(self, f"conflicting repeated value for {option_string}")
        explicit[self.dest] = value
        namespace._lumen_quant_explicit = explicit

    def format_usage(self):
        return self._original.format_usage()


def _track_existing_quant_args(container, registered):
    names = {name for aliases in LINEAR_QUANT_ARG_MAP.values() for name in aliases}
    for action in tuple(container._actions):
        if action.dest not in names or isinstance(action, _QuantArgument):
            continue
        if isinstance(action, _TrackedExistingArgument):
            registered.append(action.dest)
            continue
        tracked = _TrackedExistingArgument(action)
        container._actions[container._actions.index(action)] = tracked
        owner = getattr(action, "container", None)
        groups = [owner, *container._mutually_exclusive_groups]
        for group in groups:
            members = getattr(group, "_group_actions", ())
            if action in members:
                members[members.index(action)] = tracked
        for option in action.option_strings:
            container._option_string_actions[option] = tracked
        registered.append(action.dest)


def add_linear_quant_args(parser_or_group, *, fsdp_aliases=False, fp8_format_default=None):
    """Register generic selectors and legacy aliases with explicit-value tracking.

    Existing actions retain their conversion and side effects through a tracking
    delegate, so their defaults are not mistaken for explicit alias inputs.
    """
    registered = list(parser_or_group.get_default("_lumen_quant_registered") or ())

    def add(container, *options, **kwargs):
        available = [o for o in options if o not in container._option_string_actions]
        if not available:
            return
        # Do not make a second action for an externally owned destination.
        dest = kwargs.get("dest", available[0].lstrip("-").replace("-", "_"))
        if any(a.dest == dest for a in container._actions):
            return
        container.add_argument(*available, action=_QuantArgument, **kwargs)
        registered.append(dest)

    gates = parser_or_group.add_mutually_exclusive_group()
    add(gates, "--linear-fp8", *(["--fp8-training"] if fsdp_aliases else []),
        dest="linear_fp8", nargs=0, const=True, default=False,
        help="Compatibility selector for FP8 linear quantization.")
    add(gates, "--linear-fp4", dest="linear_fp4", nargs=0, const=True, default=False,
        help="Compatibility alias for --linear-quant-format mxfp4.")

    specs = (
        ("format", str, fp8_format_default, FORMATS, "format"),
        ("scaling", normalize_scaling, "delayed", SCALINGS, "scaling"),
        ("block_size", int, 128, None, "block-size"),
        ("amax_algo", str, "max", ("max", "most_recent"), "amax-algo"),
        ("margin", int, 0, None, "margin"),
        ("history_len", int, 16, None, "amax-history"),
    )
    for field, kind, default, choices, suffix in specs:
        generic, legacy = LINEAR_QUANT_ARG_MAP[field][:2]
        add(parser_or_group, f"--linear-quant-{suffix}", dest=generic,
            type=kind, default=None, choices=choices)
        legacy_choices = tuple(f for f in FORMATS if f != "mxfp4") if field == "format" else choices
        add(parser_or_group, f"--linear-fp8-{suffix}",
            *([f"--fp8-{suffix}"] if fsdp_aliases else []), dest=legacy,
            type=kind, default=default, choices=legacy_choices)

    for field, suffix, default in (
        ("quantize_activation", "activation", True),
        ("fp8_wgrad", "wgrad", True),
        ("reduce_amax", "reduce-amax", False),
        ("quantize_output_layer", "output-layer", False),
    ):
        generic, legacy = LINEAR_QUANT_ARG_MAP[field][:2]
        for prefix, dest, initial in (("linear-quant", generic, None), ("linear-fp8", legacy, default)):
            old_suffix = "quantize-output-layer" if field == "quantize_output_layer" and prefix == "linear-fp8" else suffix
            options = [f"--{prefix}-{old_suffix}"]
            negative = [f"--no-{prefix}-{old_suffix}"]
            if fsdp_aliases and prefix == "linear-fp8" and field in ("quantize_activation", "reduce_amax"):
                options.append(f"--fp8-{suffix}")
                negative.append(f"--no-fp8-{suffix}")
            before = len(registered)
            add(parser_or_group, *options, dest=dest, nargs=0, const=True, default=initial)
            if len(registered) != before:
                parser_or_group.add_argument(*negative, dest=dest, action=_QuantArgument,
                                             nargs=0, const=False, default=argparse.SUPPRESS)
    _track_existing_quant_args(parser_or_group, registered)
    parser_or_group.set_defaults(_lumen_quant_registered=tuple(dict.fromkeys(registered)))
    return parser_or_group


def explicit_arg_values(args, names):
    """Yield explicit inputs; unregistered Namespace attributes have no defaults."""
    managed = getattr(args, "_lumen_quant_registered", ())
    explicit = getattr(args, "_lumen_quant_explicit", {})
    for name in names:
        value = getattr(args, name, None)
        if value is not None and (name not in managed or name in explicit):
            yield name, value
