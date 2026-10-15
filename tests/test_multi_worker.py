"""多卡用手算时间线验收；不需要真实GPU、线程或推理框架。"""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from simulator.config import load_experiment
from simulator.engine import simulate_events, simulate_workers
from simulator.kv_cache import KVConfig
from simulator.models import Request, Timing
from simulator.scheduler import SchedulerConfig, SchedulerProfile
from simulator.worker import WorkerConfig


ROOT = Path(__file__).resolve().parents[1]
PROFILE = SchedulerProfile("fcfs_chunked", "round_robin", "prefill_first")


def worker(name: str, gpu: str, **kwargs) -> WorkerConfig:
    return WorkerConfig(name, gpu, PROFILE, **kwargs)


class MultiWorkerTests(unittest.TestCase):
    def test_two_cards_overlap_and_keep_each_cards_intervals_exclusive(self) -> None:
        config = load_experiment(ROOT / "configs/two_mix_workers.json")
        run = simulate_workers(config.requests, config.workers, dict(config.assignments))
        self.assertEqual([r.token_times_ms for r in run.batch_run.results], [(8, 10, 12), (4, 6)])
        self.assertEqual([(s.worker_id, s.start_ms, s.end_ms) for s in run.batch_run.steps],
                         [("worker_0", 0, 8), ("worker_1", 0, 4), ("worker_1", 4, 6),
                          ("worker_0", 8, 10), ("worker_0", 10, 12)])
        for local in run.workers:
            steps = local.run.batch_run.steps
            self.assertTrue(all(a.end_ms <= b.start_ms for a, b in zip(steps, steps[1:])))
            for record in local.run.kv_records:
                self.assertTrue(all(block.gpu_id == local.config.gpu_id for block in record.snapshot.blocks))
            self.assertEqual(local.run.kv_final.free_blocks, 4)
        for step, plan, estimate in zip(run.batch_run.steps, run.plans, run.estimates):
            self.assertEqual(step.request_ids, plan.request_ids)
            self.assertAlmostEqual(step.end_ms - step.start_ms, estimate.duration_ms)
        self.assertEqual(run, simulate_workers(config.requests, config.workers, dict(config.assignments)))

    def test_same_time_arrivals_and_all_completions_precede_any_wakeup(self) -> None:
        requests = (Request("A", 0, 2, 2), Request("B", 0, 2, 2), Request("C", 4, 1, 1))
        run = simulate_workers(requests, (worker("w0", "g0"), worker("w1", "g1")),
                               {"A": "w0", "B": "w1", "C": "w0"})
        at_four = [line for line in run.trace if "] 4ms 处理" in line]
        kinds = [line.split("处理 ")[1].split()[0] for line in at_four]
        self.assertEqual(kinds, ["arrival", "complete", "complete", "wakeup", "wakeup"])
        self.assertEqual([r.token_times_ms for r in run.batch_run.results], [(4, 8), (4, 6), (6,)])

    def test_capacity_block_on_one_card_does_not_stop_other_card(self) -> None:
        requests = (Request("A", 0, 2, 3), Request("B", 1, 1, 3))
        workers = (worker("w0", "g0", kv_config=KVConfig("g0", 1, 2)),
                   worker("w1", "g1", kv_config=KVConfig("g1", 4, 2)))
        run = simulate_workers(requests, workers, {"A": "w0", "B": "w1"})
        self.assertEqual(run.status, "blocked")
        self.assertEqual(run.pending_request_ids, ("A",))
        self.assertEqual(run.workers[0].run.pending_progress[0].token_times_ms, (4,))
        self.assertEqual(run.batch_run.results[0].token_times_ms, (3, 5, 7))
        self.assertEqual(run.workers[1].run.status, "completed")

    def test_recovery_and_scheduler_state_are_local_to_each_worker(self) -> None:
        config = load_experiment(ROOT / "configs/recompute.json")
        # 新卡正常执行时，旧卡仍按原来的保护/暂停/重算流程推进。
        requests = config.requests + (Request("C", 0, 1, 3),)
        run = simulate_workers(requests, (config.workers[0], worker("w1", "g1")),
                               {"A": "worker_0", "B": "worker_0", "C": "w1"})
        original = simulate_events(config.requests, config.timing, scheduler_profile=config.scheduler_profile,
                                   kv_config=config.kv_config, recovery_config=config.recovery_config)
        self.assertEqual(run.workers[0].run, original)
        self.assertEqual(run.workers[1].run.batch_run.results[0].token_times_ms, (2, 4, 6))
        self.assertEqual(run.workers[1].run.recovery_records, ())
        alternating = SchedulerProfile("fcfs_chunked", "round_robin", "alternating", SchedulerConfig(prefill_chunk_tokens=1))
        cards = tuple(WorkerConfig(f"w{i}", f"g{i}", alternating) for i in range(2))
        requests = (Request("A", 0, 2, 3), Request("B", 0, 3, 2),
                    Request("C", 1, 3, 2), Request("D", 2, 1, 2))
        owners = {"A": "w0", "B": "w1", "C": "w0", "D": "w1"}
        result = simulate_workers(requests, cards, owners)
        for local in result.workers:
            isolated = simulate_events(tuple(r for r in requests if owners[r.request_id] == local.config.worker_id),
                                       Timing(), scheduler_profile=alternating)
            self.assertEqual(local.run, isolated)

    def test_independent_predictors_and_idle_worker(self) -> None:
        requests = (Request("A", 0, 2, 2), Request("B", 0, 2, 2))
        cards = (worker("w0", "g0"), worker("w1", "g1", timing=Timing(1, 1)), worker("idle", "g2"))
        run = simulate_workers(requests, cards, {"A": "w0", "B": "w1"})
        self.assertEqual([r.token_times_ms for r in run.batch_run.results], [(4, 6), (2, 3)])
        self.assertEqual(run.workers[2].run.batch_run.results, ())
        empty = simulate_workers((), cards, {})
        self.assertEqual(empty.status, "completed")
        self.assertEqual(empty.batch_run.steps, ())

    def test_routing_does_not_implicitly_move_requests_to_idle_card(self) -> None:
        config = load_experiment(ROOT / "configs/two_mix_workers.json")
        run = simulate_workers(config.requests, config.workers, {"A": "worker_0", "B": "worker_0"})
        self.assertEqual(run.workers[1].run.batch_run.steps, ())
        # 共用卡0的4块KV时，A增长占满容量；B等A释放后才能继续Decode。
        self.assertEqual([r.token_times_ms for r in run.batch_run.results], [(8, 14, 16), (12, 18)])

    def test_config_rejects_ambiguous_resources_and_invalid_ownership(self) -> None:
        original = json.loads((ROOT / "configs/two_mix_workers.json").read_text())
        original["workload"]["path"] = str(ROOT / "workloads/two_mix_requests.json")
        mutations = []
        for owner in ({"A": "worker_0"}, {"A": "worker_0", "B": "missing"},
                      {"A": "worker_0", "B": "worker_1", "extra": "worker_1"}):
            config = deepcopy(original)
            config["routing"]["assignments"] = owner
            mutations.append(config)
        for field, value in (("id", "worker_0"), ("gpu_id", "gpu_0"), ("role", "D"),
                             ("gpu_id", ""), ("scheduler_profile", "missing")):
            config = deepcopy(original)
            config["workers"][1][field] = value
            mutations.append(config)
        config = deepcopy(original)
        config.pop("routing")
        mutations.append(config)
        config = deepcopy(original)
        config["workers"][1].pop("gpu_id")
        mutations.append(config)
        config = deepcopy(original)
        config["workers"][1]["kv_cache"]["gpu_id"] = "gpu_0"
        mutations.append(config)
        mutations.append({**original, "kv_cache": original["workers"][0]["kv_cache"]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for config in mutations:
                path.write_text(json.dumps(config))
                with self.subTest(config=config), self.assertRaises(ValueError):
                    load_experiment(path)
            original["workers"][1]["timing"] = {"prefill_ms_per_input_token": 1, "decode_ms_per_step": 1}
            path.write_text(json.dumps(original))
            parsed = load_experiment(path)
            self.assertEqual(parsed.workers[1].timing, Timing(1, 1))
            # 子级roofline整体替换教学耗时来源，不残留父级timing形成冲突。
            original["workers"][1].pop("timing")
            original["workers"][1]["predictor"] = json.loads(
                (ROOT / "configs/measure_a6000_qwen25_3b.json").read_text())["predictor"]
            path.write_text(json.dumps(original))
            parsed = load_experiment(path)
            self.assertEqual([w.predictor_settings.kind for w in parsed.workers], ["teaching", "roofline"])

    def test_multi_cli_reports_global_window_and_per_worker_kv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            result = subprocess.run([sys.executable, str(ROOT / "run_simulation.py"), "--summary", "--config",
                                     str(ROOT / "configs/two_mix_workers.json"), "--output", str(output)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report["schema_version"], 2)
            self.assertEqual(report["summary"]["window"]["duration_ms"], 12)
            self.assertAlmostEqual(report["summary"]["request_throughput_per_second"], 1000 / 6)
            self.assertEqual([w["busy_time_ms"] for w in report["workers"]], [12, 6])
            self.assertEqual([w["kv_final"]["gpu_id"] for w in report["workers"]], ["gpu_0", "gpu_1"])
            self.assertEqual([r["worker_id"] for r in report["requests"]], ["worker_0", "worker_1"])

    def test_blocked_cli_saves_other_cards_completed_requests_before_exit_two(self) -> None:
        config = json.loads((ROOT / "configs/two_mix_workers.json").read_text())
        config["workload"]["path"] = str(ROOT / "workloads/two_mix_requests.json")
        config["workers"][0]["kv_cache"]["total_blocks"] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            output = Path(directory) / "result.json"
            path.write_text(json.dumps(config))
            result = subprocess.run([sys.executable, str(ROOT / "run_simulation.py"), "--summary", "--config",
                                     str(path), "--output", str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "blocked")
            self.assertEqual(report["summary_scope"], "completed_requests_only")
            self.assertEqual(report["pending_request_ids"], ["A"])
            self.assertEqual(report["requests"][0]["request"]["request_id"], "B")
            self.assertEqual(report["workers"][1]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
