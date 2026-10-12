"""配置边界测试：错字段不能悄悄运行，文件驱动应保持原有时间线。"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from simulator.config import load_experiment


ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def test_cli_from_other_directory_and_both_policies(self) -> None:
        # 从其他目录运行，验证 workload 路径确实相对配置，而非当前目录。
        with tempfile.TemporaryDirectory() as directory:
            for name, expected in (("basic", [8, 14, 16]), ("decode_first", [8, 10, 12])):
                output = Path(directory) / f"{name}.json"
                completed = subprocess.run(
                    [sys.executable, str(ROOT / "run_simulation.py"), "--config",
                     str(ROOT / "configs" / f"{name}.json"), "--output", str(output)],
                    cwd=directory, capture_output=True, text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                report = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(report["requests"][0]["token_times_ms"], expected)
                self.assertEqual(report["timing_source"], "teaching_assumption")
                self.assertEqual(report["summary"]["completed_requests"], 2)
                self.assertEqual(report["summary"]["total_output_tokens"], 5)

    def test_rejects_invalid_config_and_workload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            workload = Path(directory) / "workload.json"
            original = json.loads((ROOT / "configs/basic.json").read_text())
            original["workload"]["path"] = "workload.json"
            rows = json.loads((ROOT / "workloads/two_requests.json").read_text())
            workload.write_text(json.dumps(rows))
            for mutation in (
                {**original, "gpu": {}},
                {**original, "scheduler": {"policy": "typo"}},
                {**original, "timing": {"prefill_ms_per_input_token": 2}},
                {**original, "workload": {"path": "missing.json"}},
            ):
                config.write_text(json.dumps(mutation))
                with self.assertRaises(ValueError):
                    load_experiment(config)
            config.write_text(json.dumps(original))
            for invalid in ({}, [rows[0], rows[0]], [{**rows[0], "output_tokens": True}],
                            [{**rows[0], "extra": 1}]):
                workload.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    load_experiment(config)

    def test_rejects_duplicate_keys_and_nonfinite_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            for text in ('{"workload": {}, "workload": {}}', '{"x": NaN}', '{broken'):
                config.write_text(text)
                with self.assertRaises(ValueError):
                    load_experiment(config)


if __name__ == "__main__":
    unittest.main()
