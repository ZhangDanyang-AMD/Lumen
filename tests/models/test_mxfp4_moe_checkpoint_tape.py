"""Dependency-free tape semantics; no framework or runtime claim."""

import unittest
from pathlib import Path


class TestCheckpointTapeSource(unittest.TestCase):
    def test_context_tape_declares_original_and_single_replay(self):
        # Parse production AST instead of importing a fake Torch/Core framework.
        import ast

        path = Path(__file__).resolve().parents[2] / "lumen/models/mxfp4_moe.py"
        if not path.exists():
            path = Path(__file__).resolve().parents[3] / "lumen/models/mxfp4_moe.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        classes = {
            node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
        }
        self.assertIn("CheckpointReceiptTape", classes)
        self.assertIn("CoreExpertsScope", classes)
        methods = {
            node.name
            for node in classes["CheckpointReceiptTape"].body
            if isinstance(node, ast.FunctionDef)
        }
        self.assertEqual(methods, {"__init__", "retain", "next_replay", "wrap"})
        self.assertNotIn(
            "random", {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        )


if __name__ == "__main__":
    unittest.main()
