"""显存换算手算、单位/整页边界和文件引用校验。"""

import json
from pathlib import Path
import tempfile
import unittest

from simulator.config import load_experiment
from simulator.memory_budget import MemoryDevice, MemoryModel, MemorySettings, resolve_budget


ROOT = Path(__file__).resolve().parents[1]


class MemoryBudgetTests(unittest.TestCase):
    def test_server_profile_and_derived_block_capacity(self) -> None:
        experiment = load_experiment(ROOT / "configs/rtx4090_memory.json")
        report = experiment.memory_budget
        self.assertEqual(report.device.memory_capacity_bytes, 24564 * 1024**2)
        self.assertEqual(report.model.num_layers, 36)
        self.assertEqual(report.model.weight_bytes, 16381470720)
        self.assertEqual(report.kv_bytes_per_token, 147456)
        self.assertEqual(report.block_bytes, 2359296)
        self.assertEqual(report.kv_budget_bytes, 4652544409)
        self.assertEqual(experiment.kv_config.total_blocks, 1972)
        self.assertEqual(report.unused_budget_bytes, 12697)
        self.assertIn("assumptions", experiment.memory_provenance)

    def test_capacity_and_predictor_model_must_match(self) -> None:
        config = json.loads((ROOT / "configs/predictor_mixed.json").read_text())
        config["workload"]["path"] = str(ROOT / "workloads/predictor_requests.json")
        config["kv_cache"] = {"gpu_id": "gpu_0", "block_tokens": 16,
            "memory_profile": str(ROOT / "configs/profiles/rtx4090_qwen3_8b_bf16.json")}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "不一致"):
                load_experiment(path)

    def test_formula_metadata_alignment_and_rounding(self) -> None:
        report = resolve_budget("toy", MemoryDevice("toy", 1000),
            MemoryModel("toy", 2, 1, 2, 2, 400), MemorySettings(0.9, 100, 1, 16), 3)
        self.assertEqual(report.usable_bytes, 900)
        self.assertEqual(report.kv_budget_bytes, 400)
        self.assertEqual(report.kv_bytes_per_token, 16)
        self.assertEqual(report.block_payload_bytes, 48)
        self.assertEqual(report.block_bytes, 64)
        self.assertEqual(report.total_blocks, 6)
        self.assertEqual(report.unused_budget_bytes, 16)

    def test_no_capacity_and_invalid_numbers(self) -> None:
        model = MemoryModel("toy", 2, 1, 2, 2, 400)
        for capacity in (400, 500, 510):
            with self.assertRaises(ValueError):
                resolve_budget("toy", MemoryDevice("toy", capacity), model, MemorySettings(1, 100), 3)
        for utilization in (0, True, -0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                MemorySettings(utilization, 0)
        with self.assertRaises(ValueError):
            MemoryModel("toy", 2, True, 2, 2, 400)
        with self.assertRaises(ValueError):
            MemorySettings(1, -1)

    def test_profile_relative_path_and_exclusive_modes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "profiles").mkdir()
            spec = {"device": {"name": "toy", "memory_capacity_bytes": 1000},
                "model": {"name": "toy", "num_layers": 2, "num_kv_heads": 1, "head_dim": 2,
                          "kv_bytes_per_element": 2, "weight_bytes": 400},
                "memory": {"utilization": 0.9, "runtime_reserve_bytes": 100,
                           "block_metadata_bytes": 1, "block_alignment_bytes": 16}}
            (root / "profiles/toy.json").write_text(json.dumps(spec))
            config = json.loads((ROOT / "configs/basic.json").read_text())
            config["workload"]["path"] = str(ROOT / "workloads/two_requests.json")
            config["kv_cache"] = {"gpu_id": "gpu_0", "block_tokens": 3, "memory_profile": "profiles/toy.json"}
            path = root / "config.json"
            path.write_text(json.dumps(config))
            experiment = load_experiment(path)
            self.assertEqual(experiment.kv_config.total_blocks, 6)
            self.assertEqual(Path(experiment.memory_budget.profile_path), root / "profiles/toy.json")
            config["kv_cache"]["total_blocks"] = 10
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_experiment(path)
            del config["kv_cache"]["total_blocks"]
            del config["kv_cache"]["memory_profile"]
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_experiment(path)


if __name__ == "__main__":
    unittest.main()
