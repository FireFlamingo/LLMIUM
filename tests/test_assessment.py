import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from llmium.core import Checkpoint, repair, scan
from llmium.demo import run_demo
from llmium.mcp import handle


class AssessmentTests(unittest.TestCase):
    def test_forward_recovery_and_exact_localization(self):
        with tempfile.TemporaryDirectory() as folder:
            result = run_demo(folder)
            row = next(t for t in result["before"]["tensors"] if t["name"] == result["target"])
            self.assertEqual(row["nonfinite"], 1)
            self.assertEqual(row["changed_elements"], 1)
            self.assertEqual(row["first_changed_indices"], [459])
            self.assertEqual(result["forward_pass"]["candidate"]["nonfinite_outputs"], 8)
            self.assertTrue(result["exact_forward_recovery"])
            self.assertEqual(result["after"]["findings"], [])

    def test_shards_empty_scalars_and_integer_precision(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            save_file({"empty": np.zeros((0,), dtype=np.float32), "scalar": np.array(2, dtype=np.float32)}, path / "a.safetensors")
            save_file({"integer": np.array([2**62, 2**62+1], dtype=np.int64)}, path / "b.safetensors")
            result = scan(path, path)
            self.assertEqual(len(result["tensors"]), 3)
            self.assertTrue(all(t["changed_elements"] == 0 for t in result["tensors"]))

    def test_reject_invalid_bounds_and_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.safetensors"
            for header in [b'{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}', b'{"x":{},"x":{}}']:
                path.write_bytes(struct.pack("<Q", len(header)) + header)
                with self.assertRaises(ValueError):
                    Checkpoint(path)

    def test_repair_refuses_overwrite_and_schema_drift(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            a, b = path / "a.safetensors", path / "b.safetensors"
            save_file({"x": np.ones(3, dtype=np.float32)}, a)
            save_file({"x": np.ones(4, dtype=np.float32)}, b)
            with self.assertRaises(ValueError):
                repair(a, b, path / "new", ["x"])
            with self.assertRaises(ValueError):
                repair(a, a, path, ["x"])
            self.assertFalse((path / "new").exists())

    def test_mcp_lists_tools_and_reports_errors(self):
        self.assertEqual(len(handle({"method": "tools/list"})["tools"]), 2)
        self.assertTrue(handle({"method": "tools/call", "params": {"name": "unknown"}})["isError"])


if __name__ == "__main__":
    unittest.main()
