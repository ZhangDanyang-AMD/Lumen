"""CPU-only shared CLI resolution; no tensors or GPU kernels are needed."""

import argparse
import contextlib
import io
import unittest

from lumen.cli import add_linear_quant_args
from lumen.config import LumenConfig, _ARG_MAP


class TestLinearQuantCLI(unittest.TestCase):
    def parser(self, **kwargs):
        parser = argparse.ArgumentParser()
        add_linear_quant_args(parser.add_argument_group("quantization"), **kwargs)
        return parser

    def config(self, *argv):
        return LumenConfig.from_args(self.parser().parse_args(argv))

    def test_mapping_has_generic_destinations(self):
        for field, dest in (("format", "linear_quant_format"),
                            ("scaling", "linear_quant_scaling"),
                            ("block_size", "linear_quant_block_size")):
            self.assertIn(dest, _ARG_MAP[field])
        self.assertNotIn("linear_fp4", LumenConfig.__dataclass_fields__)

    def test_generic_and_fp4_alias_equal_constructor(self):
        direct = LumenConfig(format="mxfp4")
        for argv in (("--linear-quant-format", "mxfp4"), ("--linear-fp4",),
                     ("--linear-fp4", "--linear-quant-format", "mxfp4")):
            with self.subTest(argv=argv):
                self.assertEqual(self.config(*argv), direct)

    def test_parser_defaults_do_not_override_format_defaults(self):
        parser = self.parser(fsdp_aliases=True, fp8_format_default="fp8_e4m3")
        args = parser.parse_args(["--linear-quant-format", "mxfp4"])
        self.assertEqual(args.linear_fp8_block_size, 128)
        self.assertEqual(LumenConfig.from_args(args).block_size, None)
        self.assertEqual(LumenConfig.from_args(args).format, "mxfp4")

    def test_disabled_and_legacy_fp8_defaults(self):
        self.assertEqual(self.config().scaling, "none")
        self.assertEqual(self.config("--linear-fp8"), LumenConfig())
        self.assertEqual(self.config("--linear-fp4", "--linear-quant-scaling", "none"),
                         LumenConfig(format="mxfp4", scaling="none"))

    def test_same_alias_values_allowed_in_both_orders(self):
        first = ["--linear-quant-scaling", "dynamic"]
        second = ["--linear-fp8-scaling", "dynamic"]
        for flags in (first + second, second + first):
            self.assertEqual(self.config("--linear-fp8", *flags).scaling, "dynamic")

    def test_conflicting_aliases_rejected_in_both_orders(self):
        cases = [
            (["--linear-quant-format", "mxfp4"], ["--linear-fp8-format", "hybrid"]),
            (["--linear-quant-scaling", "dynamic"], ["--linear-fp8-scaling", "delayed"]),
            (["--linear-quant-block-size", "32"], ["--linear-fp8-block-size", "128"]),
            (["--linear-quant-activation"], ["--no-linear-fp8-activation"]),
            (["--no-linear-quant-wgrad"], ["--linear-fp8-wgrad"]),
        ]
        for first, second in cases:
            for flags in (first + second, second + first):
                with self.subTest(flags=flags), self.assertRaisesRegex(ValueError, "Conflicting"):
                    self.config(*flags)

    def test_repeated_option_conflict_is_not_last_wins(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.parser().parse_args(["--linear-quant-format", "mxfp4",
                                      "--linear-quant-format", "hybrid"])
        self.assertEqual(self.config("--linear-quant-format", "mxfp4",
                                     "--linear-quant-format", "mxfp4").format, "mxfp4")

    def test_namespace_inputs_are_explicit(self):
        args = argparse.Namespace(linear_quant_block_size=32, linear_fp8_block_size=128)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            LumenConfig.from_args(args)
        self.assertEqual(LumenConfig.from_args(argparse.Namespace(
            linear_fp8=False, linear_quant_format="mxfp4")), LumenConfig(format="mxfp4"))

    def test_fp8_gate_conflicts_with_mxfp4_selector(self):
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.config("--linear-fp8", "--linear-quant-format", "mxfp4")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.parser().parse_args(["--linear-fp8", "--linear-fp4"])

    def test_parser_reuse_does_not_leak_explicit_values(self):
        parser = self.parser()
        parser.parse_args(["--linear-quant-format", "mxfp4"])
        self.assertEqual(LumenConfig.from_args(parser.parse_args([])).scaling, "none")

    def test_embedding_parser_defaults_are_preserved_for_fp8_only(self):
        parser = self.parser()
        parser.set_defaults(linear_fp8_scaling="dynamic", linear_fp8_block_size=64)
        cfg = LumenConfig.from_args(parser.parse_args(["--linear-fp8"]))
        self.assertEqual((cfg.scaling, cfg.block_size), ("dynamic", 64))
        cfg = LumenConfig.from_args(parser.parse_args(["--linear-fp4"]))
        self.assertEqual(cfg, LumenConfig(format="mxfp4"))
        cfg = LumenConfig.from_args(parser.parse_args([
            "--linear-fp8", "--linear-quant-scaling", "blockwise",
            "--linear-quant-block-size", "32",
        ]))
        self.assertEqual((cfg.scaling, cfg.block_size), ("blockwise", 32))

    def test_generic_parser_default_can_select_mxfp4(self):
        parser = self.parser()
        parser.set_defaults(linear_quant_format="mxfp4", linear_quant_block_size=128)
        cfg = LumenConfig.from_args(parser.parse_args([]))
        self.assertEqual(cfg.format, "mxfp4")
        self.assertEqual(cfg.block_size, 128)  # must reach recipe validation, not be erased

    def test_embedding_parser_actions_keep_explicit_default_provenance(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--linear-fp8-block-size", type=int, default=128)
        parser.add_argument("--quantize-output-layer", action=argparse.BooleanOptionalAction, default=False)
        add_linear_quant_args(parser.add_argument_group("quantization"))
        args = parser.parse_args(["--linear-quant-format", "mxfp4", "--linear-quant-output-layer"])
        cfg = LumenConfig.from_args(args)
        self.assertTrue(cfg.quantize_output_layer)
        self.assertIsNone(cfg.block_size)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            LumenConfig.from_args(parser.parse_args([
                "--linear-quant-block-size", "32", "--linear-fp8-block-size", "128",
            ]))
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            LumenConfig.from_args(parser.parse_args([
                "--linear-quant-output-layer", "--no-quantize-output-layer",
            ]))

    def test_embedding_action_side_effects_and_mutual_exclusion_are_preserved(self):
        class Uppercase(argparse.Action):
            def __call__(self, parser, namespace, values, option_string=None):
                namespace.touched = True
                setattr(namespace, self.dest, values.lower())

        parser = argparse.ArgumentParser()
        group = parser.add_mutually_exclusive_group()
        group.add_argument("--linear-fp8-format", action=Uppercase)
        group.add_argument("--other-mode", action="store_true")
        add_linear_quant_args(parser)
        args = parser.parse_args(["--linear-fp8", "--linear-fp8-format", "HYBRID"])
        self.assertTrue(args.touched)
        self.assertEqual(LumenConfig.from_args(args).format, "hybrid")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--linear-fp8-format", "hybrid", "--other-mode"])

    def test_parser_disables_unselected_recipe_but_direct_namespace_keeps_legacy_default(self):
        self.assertEqual(self.config("--linear-fp8-format", "hybrid").scaling, "none")
        self.assertEqual(LumenConfig.from_args(argparse.Namespace()).scaling, "delayed")
        self.assertEqual(LumenConfig.from_args(argparse.Namespace(linear_fp8_format="hybrid")).scaling, "delayed")

    def test_scaling_aliases_are_compared_canonically(self):
        self.assertEqual(self.config("--linear-fp8", "--linear-quant-scaling", "current",
                                     "--linear-fp8-scaling", "dynamic").scaling, "dynamic")

    def test_megatron_translated_format_is_fallback(self):
        args = self.parser().parse_args(["--linear-fp8", "--linear-quant-format", "hybrid"])
        args.lumen_fp8_format = "fp8_e4m3"
        self.assertEqual(LumenConfig.from_args(args).format, "hybrid")
        args = self.parser().parse_args(["--linear-fp8"])
        args.lumen_fp8_format = "hybrid"
        self.assertEqual(LumenConfig.from_args(args).format, "hybrid")


if __name__ == "__main__":
    unittest.main()
