"""Static checks for H3's authored model preference ladders.

These tests intentionally avoid importing the package: metadata inspection must remain usable
without constructing a model or installing CUDA dependencies.
"""

import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]
EXPECTED_WIDTHS = (8, 7, 4, 2, 1)
EXPECTED_GPU_PATTERNS = ("H100", "H200", "B200", "Blackwell", "RTX PRO 6000", "5090")


def _source_ladder(name: str) -> list[dict[str, object]]:
    tree = ast.parse((ROOT / "h3.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing ladder assignment: {name}")


class H3LadderTests(unittest.TestCase):
    def test_base_and_turbo_ladders_cover_h100_widths(self) -> None:
        for name in ("_DEFAULT_MODEL_LADDER", "_DEFAULT_TURBO_LORA_LADDER"):
            ladder = _source_ladder(name)
            self.assertEqual(tuple(row["gpus"] for row in ladder[:5]), EXPECTED_WIDTHS)
            self.assertTrue(all(row["gpu"] == "H100" for row in ladder[:5]))
            self.assertEqual(tuple(row["gpus"] for row in ladder[:-1]), EXPECTED_WIDTHS * 6)
            self.assertEqual(tuple(row["gpu"] for row in ladder[::5][:6]), EXPECTED_GPU_PATTERNS)
            self.assertEqual(ladder[-1]["gpu"], "*")
            self.assertEqual(ladder[-1]["gpus"], 1)

    def test_interface_matches_width_ladder_and_text_is_single_gpu(self) -> None:
        interface = json.loads((ROOT / "metadata/package-interface.json").read_text())
        for entrypoint in interface["entrypoints"]:
            for model in entrypoint.get("models", []):
                ladder = model.get("default_ladder")
                if not ladder:
                    continue
                widths = tuple(row["gpus"] for row in ladder)
                if model.get("components") == ["text_encoder"]:
                    self.assertEqual(widths, (1,) * 7)
                else:
                    self.assertEqual(widths, EXPECTED_WIDTHS * 6 + (1,))
                repeats = 1 if model.get("components") == ["text_encoder"] else 5
                expected_patterns = tuple(
                    pattern for pattern in EXPECTED_GPU_PATTERNS for _ in range(repeats)
                )
                self.assertEqual(tuple(row["gpu"] for row in ladder[:-1]), expected_patterns)
                self.assertEqual(ladder[-1]["gpu"], "*")


if __name__ == "__main__":
    unittest.main()
