from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_dataset", ROOT / "scripts" / "validate_dataset.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ReleaseTest(unittest.TestCase):
    def test_dataset_integrity(self) -> None:
        result = MODULE.validate(ROOT / "data" / "finance")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["episodes"], 4800)
        self.assertEqual(result["streams"], 1800)


if __name__ == "__main__":
    unittest.main()
