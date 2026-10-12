"""CPU-only package discovery regression tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).parents[1]


class CpuImportTests(unittest.TestCase):
    def _cpu_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        return env

    def test_package_import_registers_normalize_without_cuda(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", "import seedvr2; print(sorted(seedvr2.app._registry))"],
            cwd=ROOT,
            env=self._cpu_env(),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("'normalize'", result.stdout)

    def test_cpu_conformance_describes_normalize(self) -> None:
        result = subprocess.run(
            [
                str(Path(sys.executable).with_name("cozy-runtime")),
                "--json",
                "--conformance",
                "--dir",
                str(ROOT),
                "describe",
            ],
            cwd=ROOT,
            env=self._cpu_env(),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        interface = json.loads(result.stdout)
        self.assertEqual([job["name"] for job in interface["jobs"]], ["normalize"])


if __name__ == "__main__":
    unittest.main()
