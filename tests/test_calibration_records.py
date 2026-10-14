"""验证采集描述与GPU安全边界；不需要GPU或vLLM依赖。"""

from types import SimpleNamespace as NS
import json
from pathlib import Path
import tempfile
import unittest

from calibration.scripts.batch_records import describe_work, select_device
from calibration.scripts.summarize_smoke import summarize
from calibration.scripts.compare_measured import compare_rows
from calibration.scripts.compare_senior_measured import SeniorReplayPredictor, replay_requests
from simulator.predictor import PredictorItem
from calibration.scripts.fit_timing import fit_nonnegative, training_basis
from calibration.scripts.matrix_plan import load_matrix_plan
from calibration.scripts.validate_fresh import summarize_errors


def schedule(new=(), cached=(), contexts=(), counts=None, resumed=()):
    return NS(scheduled_new_reqs=new,
              scheduled_cached_reqs=NS(req_ids=cached, num_computed_tokens=contexts,
                                       resumed_from_preemption=resumed),
              num_scheduled_tokens=counts or {})


class CalibrationRecordTests(unittest.TestCase):
    def test_validation_reports_worst_shape_and_does_not_hide_it_in_mean(self):
        report = summarize_errors([{"error": -0.01}, {"error": 0.39}], "error")
        self.assertAlmostEqual(report["mean_absolute_relative_error"], 0.2)
        self.assertEqual(report["max_absolute_relative_error"], 0.39)
        self.assertFalse(report["all_shapes_within_20_percent"])

    def test_fitter_recovers_positive_parameters_and_enforces_boundary(self):
        features = [[1, 0.2, 1], [2, 0.1, 1], [4, 2, 1], [8, 5, 1]]
        targets = [2 * x[0] + 0.5 * x[1] + 3 for x in features]
        for fitted, expected in zip(fit_nonnegative(features, targets), (2, 0.5, 3)):
            self.assertAlmostEqual(fitted, expected, places=7)
        boundary = fit_nonnegative(features, [x[0] + 2 for x in features])
        self.assertGreater(boundary[1], 0)
        self.assertLess(boundary[1], 1e-6)
        self.assertGreaterEqual(boundary[2], 0)

    def test_validation_values_do_not_enter_training_basis(self):
        summaries = [{"shape_id": "P-fit", "split": "fit", "measured_median_ms": 10},
                     {"shape_id": "P-heldout", "split": "validation", "measured_median_ms": 999999}]
        details = [{"shape_id": "P-fit", "sample_id": "a", "basis_ms": [1, 2]},
                   {"shape_id": "P-heldout", "sample_id": "b", "basis_ms": [999999, 999999]}]
        self.assertEqual(training_basis(summaries, details), ([[1, 2, 1]], [10], ["P-fit"]))

    def test_fresh_matrix_is_half_novel_and_separate_from_old_shapes(self):
        path = Path(__file__).resolve().parents[1] / "calibration/fresh_validation.json"
        plan = load_matrix_plan(path)
        old_report = json.loads((path.parent / "measurements/du-qwen25-3b-matrix-20261014.json").read_text())
        old = {row["shape_id"] for row in old_report["shapes"]}
        for shape in plan["shapes"]:
            shape_id = f'{shape["phase"]}-B{shape["batch_size"]}-L{shape["length"]}'
            self.assertEqual(shape_id in old, shape["split"] == "repeat")

    def test_senior_adapter_keeps_original_prompt_end_and_growing_context(self):
        item = PredictorItem("A", "Decode", 1, 145, True)
        rows = replay_requests((item,), {"A": 128})
        self.assertEqual((rows[0]["prompt_end"], rows[0]["context_length"],
                          rows[0]["range_start"], rows[0]["range_end"], rows[0]["logits_token_count"]),
                         (128, 145, 145, 146, 1))
        with self.assertRaises(ValueError):
            replay_requests((item, PredictorItem("B", "Prefill", 2, 0, True)), {"A": 128, "B": 2})

    def test_senior_cache_ignores_identity_but_not_workload(self):
        calls = []

        def replay(rows):
            calls.append(rows)
            return rows[0]["context_length"] / 10

        predictor = SeniorReplayPredictor({"A": 128, "B": 128}, replay)
        a = predictor.estimate((PredictorItem("A", "Decode", 1, 145, True),))
        b = predictor.estimate((PredictorItem("B", "Decode", 1, 145, True),))
        c = predictor.estimate((PredictorItem("B", "Decode", 1, 146, True),))
        self.assertEqual(len(calls), 2)
        self.assertEqual((a.duration_ms, b.duration_ms, c.duration_ms), (14.5, 14.5, 14.6))
        self.assertEqual(b.items[0].request_id, "B")

    def test_comparison_uses_actual_context_and_equal_repetition_weight(self):
        class ContextPredictor:
            def estimate(self, items):
                return NS(duration_ms=sum(item.context_tokens for item in items))

        def row(sample, context, measured, phase="Decode", warmup=False, count=1):
            return {"shape_id": "Decode-B1-L128", "sample_id": sample, "iteration_index": context,
                    "record_status": "measured", "timing_scope": "worker_execute_wall",
                    "duration_ms": measured, "preemptions": 0, "is_warmup": warmup,
                    "target_phase": "Decode", "split": "fit", "actual_batch_requests": count,
                    "items": [{"request_id": str(i), "phase": phase, "new_tokens": 1,
                               "context_tokens": context, "produces_token": True} for i in range(count)]}

        # 第二次生成的轮次多，也只占一次重复的权重；context没有用名称L128覆盖。
        data = [row("a", 10, 10), row("b", 20, 20), row("b", 20, 20), row("b", 20, 20),
                row("warm", 99, 99, warmup=True), row("input", 1, 99, phase="Prefill"),
                row("unexpected", 10, 99, count=2)]
        summary, details = compare_rows(data, ContextPredictor())
        self.assertEqual(summary[0]["repetitions"], 2)
        self.assertEqual(summary[0]["worker_rounds"], 4)
        self.assertEqual(summary[0]["measured_median_ms"], 15)
        self.assertEqual(summary[0]["predicted_median_ms"], 15)
        self.assertEqual([r["predicted_ms"] for r in details], [10, 20, 20, 20])
        self.assertEqual(summary[0]["excluded"], {"warmup": 1, "mixed_or_other_phase": 1, "partial_batch": 1})
        with self.assertRaises(ValueError):
            compare_rows([row("a", 10, float("nan"))], ContextPredictor())

    def test_uuid_maps_to_physical_index_and_rejects_busy(self):
        data = "3, GPU-B, 2441 MiB, 0 %\n7, GPU-A, 1 MiB, 0 %"
        self.assertEqual(select_device(data, "GPU-A"), 7)
        with self.assertRaises(RuntimeError):
            select_device(data, "GPU-B")
        with self.assertRaises(ValueError):
            select_device(data, "GPU-missing")
        with self.assertRaises(RuntimeError):
            select_device("7, GPU-A, 1 MiB, 100 %", "GPU-A")

    def test_partial_prefill_then_final_input_and_decode_history(self):
        lengths = {}
        new = NS(req_id="A", prompt_token_ids=[1] * 5, num_computed_tokens=0)
        row = describe_work(schedule(new=(new,), counts={"A": 2}), lengths)[0]
        self.assertEqual((row["phase"], row["context_tokens"], row["produces_token"]),
                         ("Prefill", 0, False))
        row = describe_work(schedule(cached=("A",), contexts=(2,), counts={"A": 3}), lengths)[0]
        self.assertEqual((row["phase"], row["context_tokens"], row["produces_token"]),
                         ("Prefill", 2, True))
        row = describe_work(schedule(cached=("A",), contexts=(7,), counts={"A": 1}), lengths)[0]
        self.assertEqual((row["phase"], row["context_tokens"], row["new_tokens"]),
                         ("Decode", 7, 1))

    def test_shared_trial_requires_explicit_memory_limit_and_zero_utilization(self):
        data = "0, GPU-A, 2441 MiB, 0 %"
        with self.assertRaises(RuntimeError):
            select_device(data, "GPU-A")
        self.assertEqual(select_device(data, "GPU-A", 3072), 0)
        with self.assertRaises(RuntimeError):
            select_device("0, GPU-A, 2441 MiB, 10 %", "GPU-A", 3072)
        with self.assertRaises(ValueError):
            select_device(data, "GPU-A", -1)

    def test_batch_keeps_member_contexts_and_rejects_unsupported_recovery(self):
        lengths = {"A": 5, "B": 8}
        rows = describe_work(schedule(cached=("A", "B"), contexts=(6, 10),
                                      counts={"A": 1, "B": 1}), lengths)
        self.assertEqual([r["context_tokens"] for r in rows], [6, 10])
        with self.assertRaises(RuntimeError):
            describe_work(schedule(cached=("A",), contexts=(0,), counts={"A": 5},
                                   resumed=(True,)), lengths)
        with self.assertRaises(ValueError):
            describe_work(schedule(cached=("A",), contexts=(6,), counts={"A": 2}), lengths)

    def test_smoke_summary_excludes_warmup_and_decode_input_prefill(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            metadata = {"record_status": "measured", "run_id": "trial", "shared_gpu_trial": True,
                        "gpu_uuid": "GPU-A", "software": {}, "hashes": {"config.json": "model-sha"},
                        "engine": {"gpu_memory_utilization": 0.8}}
            (run / "metadata.json").write_text(json.dumps(metadata))
            (run / "capacity.json").write_text(json.dumps({"record_status": "measured"}))
            rows, checks = [], []
            for target in ("Prefill", "Decode"):
                for repeat in range(3):
                    checks.append({"target_phase": target,
                                   "actual_output_tokens": [1 if target == "Prefill" else 3]})
                    phases = ["Prefill"] if target == "Prefill" else ["Prefill", "Decode", "Decode"]
                    for phase in phases:
                        rows.append({"record_status": "measured", "run_id": "trial", "split": "smoke",
                                     "timing_scope": "worker_execute_wall", "target_phase": target,
                                     "is_warmup": repeat == 0, "actual_batch_requests": 1,
                                     "duration_ms": 1000 if repeat == 0 or target != phase else 10,
                                     "items": [{"phase": phase, "new_tokens": 128 if phase == "Prefill" else 1,
                                                "context_tokens": 0 if phase == "Prefill" else 128}]})
            for name, records in (("batches.jsonl", rows), ("generation_checks.jsonl", checks)):
                (run / name).write_text("\n".join(json.dumps(row) for row in records))
            for name in ("COMPLETE", "gpu_before.txt", "startup.log"):
                (run / name).write_text("fixture")
            result = summarize(run)
            self.assertEqual([row["sample_count"] for row in result["batches"]], [2, 4])
            self.assertEqual([row["median_ms"] for row in result["batches"]], [10, 10])
            # 未完成和输出不全必须拒绝，避免把失败实验写成可用测量。
            (run / "COMPLETE").unlink()
            with self.assertRaises(ValueError):
                summarize(run)
            (run / "COMPLETE").write_text("fixture")
            checks[-1]["actual_output_tokens"] = [2]
            (run / "generation_checks.jsonl").write_text("\n".join(json.dumps(row) for row in checks))
            with self.assertRaises(ValueError):
                summarize(run)


if __name__ == "__main__":
    unittest.main()
