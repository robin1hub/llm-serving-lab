"""核对公式、上下文口径、事件接入与配置边界。"""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from simulator.config import load_experiment
from simulator.engine import simulate_events
from simulator.models import Request, Timing
from simulator.predictor import (Calibration, DeviceProfile, ModelProfile, PredictorItem,
                                 RooflinePredictor, TeachingPredictor, build_predictor)


ROOT = Path(__file__).resolve().parents[1]


class PredictorTests(unittest.TestCase):
    def predictor(self, *, flops=1e6, bandwidth=1e6, calibration=Calibration()) -> RooflinePredictor:
        return RooflinePredictor(ModelProfile(1, 4, 8, 2, 1, 8),
                                 DeviceProfile("toy", flops, bandwidth), calibration)

    def test_hand_calculated_prefill_and_bottlenecks(self) -> None:
        items = (PredictorItem("A", "Prefill", 2, 0, False),)
        estimate = self.predictor(calibration=Calibration(batch_overhead_ms=0.1)).estimate(items)
        dense, attention = estimate.operators
        # matrices=32+16+96=144；2 token * 2 * 144=576 FLOPs。
        self.assertEqual((dense.flops, dense.memory_bytes), (576, 320))
        self.assertEqual((attention.flops, attention.memory_bytes), (48, 16))
        self.assertAlmostEqual(estimate.duration_ms, 0.724)
        self.assertEqual(dense.bottleneck, "compute")
        slow_memory = self.predictor(flops=1e9, bandwidth=1000).estimate(items)
        self.assertEqual(slow_memory.operators[0].bottleneck, "memory")
        self.assertAlmostEqual(slow_memory.duration_ms, 336)

    def test_context_batch_weights_and_calibration(self) -> None:
        short = PredictorItem("A", "Decode", 1, 3, True)
        long = PredictorItem("A", "Decode", 1, 30, True)
        predictor = self.predictor()
        first = predictor.estimate((short,))
        second = predictor.estimate((long,))
        self.assertGreater(second.operators[1].flops, first.operators[1].flops)
        self.assertGreater(second.operators[1].memory_bytes, first.operators[1].memory_bytes)
        batch = predictor.estimate((short, PredictorItem("B", "Decode", 1, 3, True)))
        # 共享权重，增加一成员只增加16字节的简化dense激活流量。
        self.assertEqual(batch.operators[0].memory_bytes - first.operators[0].memory_bytes, 16)
        scaled = self.predictor(calibration=Calibration(2, 3, 0.1)).estimate((short,))
        self.assertAlmostEqual(scaled.duration_ms,
            first.operators[0].duration_ms * 2 + first.operators[1].duration_ms * 3 + 0.1)

    def test_engine_context_and_hand_timeline(self) -> None:
        experiment = load_experiment(ROOT / "configs/predictor_roofline.json")
        run = simulate_events(experiment.requests, experiment.timing,
            scheduler_profile=experiment.scheduler_profile,
            predictor=build_predictor(experiment.predictor_settings, experiment.timing))
        self.assertEqual([tuple(i.context_tokens for i in e.items) for e in run.estimates],
                         [(0,), (2,), (0,), (3, 1), (4,)])
        self.assertEqual([tuple(i.produces_token for i in e.items) for e in run.estimates],
                         [(False,), (True,), (True,), (True, True), (True,)])
        for step, end in zip(run.batch_run.steps, (0.724, 1.24, 1.724, 2.624, 3.172)):
            self.assertAlmostEqual(step.end_ms, end)
        self.assertEqual(len(run.estimates), len(run.plans))
        for step, estimate in zip(run.batch_run.steps, run.estimates):
            self.assertAlmostEqual(step.end_ms - step.start_ms, estimate.duration_ms)
        self.assertEqual([len(r.token_times_ms) for r in run.batch_run.results], [3, 2])

    def test_default_teaching_compatibility_and_empty_workload(self) -> None:
        requests = (Request("A", 0, 4, 3), Request("B", 3, 2, 2))
        default = simulate_events(requests, Timing())
        explicit = simulate_events(requests, Timing(), predictor=TeachingPredictor(Timing()))
        self.assertEqual(default.batch_run, explicit.batch_run)
        self.assertEqual([r.token_times_ms for r in default.batch_run.results], [(8, 14, 16), (12, 14)])
        self.assertEqual(simulate_events((), Timing(), predictor=self.predictor()).estimates, ())

    def test_invalid_config_and_predictor_inputs(self) -> None:
        original = json.loads((ROOT / "configs/predictor_roofline.json").read_text())
        original["workload"]["path"] = str(ROOT / "workloads/predictor_requests.json")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            variants = [{**original, "timing": {}}, {**original, "predictor": {"kind": "typo"}}]
            for group, key, value in (("device", "peak_flops", 0), ("device", "peak_flops", True),
                ("device", "memory_bandwidth_bytes_per_s", float("inf")),
                ("device", "cuda_cores", 100), ("model", "num_kv_heads", 3),
                ("model", "hidden_size", 5), ("calibration", "dense_factor", -1),
                ("calibration", "batch_overhead_ms", -1)):
                variant = deepcopy(original)
                variant["predictor"][group][key] = value
                variants.append(variant)
            for variant in variants:
                path.write_text(json.dumps(variant))
                with self.subTest(config=variant), self.assertRaises(ValueError):
                    load_experiment(path)
        with self.assertRaises(ValueError):
            self.predictor().estimate(())
        with self.assertRaises(ValueError):
            PredictorItem("A", "Decode", 2, 0, True)

    def test_cli_exports_prediction_and_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            completed = subprocess.run([sys.executable, str(ROOT / "run_simulation.py"),
                "--config", str(ROOT / "configs/predictor_roofline.json"), "--output", str(output),
                "--summary"], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report["timing_source"], "uncalibrated_roofline")
            self.assertIsNone(report["timing"])
            self.assertEqual(report["predictor"]["device"]["name"], "toy-device-not-4090")
            self.assertEqual(len(report["estimates"]), 5)
            self.assertAlmostEqual(report["estimates"][0]["duration_ms"], 0.724)


if __name__ == "__main__":
    unittest.main()
