"""Python attribute semantics plus source equivalence, not actual Torch compatibility."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace


class AttributeProbeContractTests(unittest.TestCase):
    def test_existing_and_missing_attributes_have_boolean_results(self):
        self.assertTrue(hasattr(SimpleNamespace(_scaled_mm=object()), "_scaled_mm"))
        self.assertFalse(hasattr(SimpleNamespace(), "_scaled_mm"))

    def test_unexpected_attribute_errors_are_not_capability_misses(self):
        failure = RuntimeError("unexpected attribute implementation")

        class BrokenAttributes:
            def __getattr__(self, name):
                raise failure

        with self.assertRaises(RuntimeError) as caught:
            hasattr(BrokenAttributes(), "_scaled_mm")
        self.assertIs(caught.exception, failure)

    def test_exact_legacy_probe_keeps_name_cache_and_direct_attribute_check(self):
        path = Path(__file__).resolve().parents[2] / "lumen/ops/dispatch.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_probe_hipblas_fp8_output"
        )
        statements = [node for node in function.body if not isinstance(node, ast.Expr)]
        self.assertEqual(len(statements), 1)
        self.assertIsInstance(statements[0], ast.Return)
        self.assertEqual(
            ast.unparse(statements[0].value), "hasattr(torch, '_scaled_mm')"
        )
        self.assertEqual(
            ast.unparse(function.decorator_list[0]), "functools.lru_cache(maxsize=1)"
        )


if __name__ == "__main__":
    unittest.main()
