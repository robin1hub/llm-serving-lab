"""1P1D交接按手算与块生命周期验收，不依赖真实GPU。"""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from simulator.config import load_experiment
from simulator.engine import simulate_pd
from simulator.handoff import LinkConfig
from simulator.kv_cache import BlockPool, KVConfig
from simulator.models import Request
from simulator.scheduler import BatchPlan, ReadyTask


ROOT = Path(__file__).resolve().parents[1]


class PDTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_experiment(ROOT / "configs/one_p_one_d.json")

    def simulate(self, requests=None, **overrides):
        return simulate_pd(self.config.requests if requests is None else requests,
                           overrides.get("workers", self.config.workers),
                           overrides.get("pd", self.config.pd_config))

    def test_single_request_preserves_first_start_and_first_token(self) -> None:
        run = self.simulate()
        result = run.batch_run.results[0]
        self.assertEqual(result.start_ms, 0)
        self.assertEqual(result.token_times_ms, (8, 15, 17))
        self.assertEqual(result.inter_token_intervals_ms, (7, 2))
        transfer = run.transfers[0]
        self.assertEqual((transfer.ready_ms, transfer.start_ms, transfer.end_ms), (8, 8, 13))
        self.assertEqual((transfer.payload_tokens, transfer.payload_bytes), (4, 4096))
        self.assertEqual(transfer.status, "completed")
        self.assertEqual([(s.worker_id, s.phase, s.start_ms, s.end_ms) for s in run.batch_run.steps],
                         [("p0", "Prefill", 0, 8), ("d0", "Decode", 13, 15), ("d0", "Decode", 15, 17)])
        self.assertEqual([item.context_tokens for estimate in run.estimates[1:] for item in estimate.items], [4, 5])
        self.assertEqual(run.workers[0].run.pending_request_ids, ())

    def test_target_populating_and_source_held_until_transfer_finishes(self) -> None:
        run = self.simulate()
        source, target = [worker.run for worker in run.workers]
        held = next(r for r in source.kv_records if r.action == "transfer_source_held:A")
        self.assertEqual(held.snapshot.tables[0].cached_tokens, 4)
        self.assertTrue(all(b.state == "Ready" for b in held.snapshot.blocks))
        copying = next(r for r in target.kv_records if r.action == "transfer_populating:A")
        self.assertEqual(copying.time_ms, 8)
        self.assertEqual(copying.snapshot.tables, ())
        self.assertTrue(all(b.state == "Populating" and b.valid_end == b.valid_start for b in copying.snapshot.blocks))
        ready = next(r for r in target.kv_records if r.action == "transfer_ready:A")
        released = next(r for r in source.kv_records if r.action == "transfer_source_release:A")
        self.assertEqual((ready.time_ms, released.time_ms), (13, 13))
        self.assertEqual(ready.snapshot.tables[0].cached_tokens, 4)
        self.assertEqual(released.snapshot.blocks, ())
        self.assertTrue(all(w.run.kv_final.free_blocks == 16 for w in run.workers))

    def test_bandwidth_changes_gap_after_first_token_only(self) -> None:
        slower = replace(self.config.pd_config, link=LinkConfig(512000, 1, 1024))
        run = self.simulate(pd=slower)
        self.assertEqual(run.batch_run.results[0].token_times_ms, (8, 19, 21))
        self.assertEqual(run.transfers[0].end_ms, 17)

    def test_fifo_link_overlaps_compute_and_destination_waits_for_busy_batch(self) -> None:
        requests = (Request("A", 0, 4, 3), Request("B", 0, 2, 2))
        run = self.simulate(requests)
        self.assertEqual([(t.ready_ms, t.start_ms, t.end_ms) for t in run.transfers], [(8, 8, 13), (12, 13, 16)])
        self.assertEqual([r.token_times_ms for r in run.batch_run.results], [(8, 15, 17), (12, 19)])
        # B在13–16拷入时，D同时执行A；事务互不覆盖，B在16就绪也不能打断15–17的批。
        both = [r.snapshot for r in run.workers[1].run.kv_records if len(r.snapshot.reservations) == 2]
        self.assertTrue(both)
        self.assertTrue(all(t.status == "completed" for t in run.transfers))

    def test_single_output_needs_no_transfer_or_decode(self) -> None:
        run = self.simulate((Request("A", 0, 4, 1),))
        self.assertEqual(run.batch_run.results[0].token_times_ms, (8,))
        self.assertEqual(run.transfers, ())
        self.assertEqual(run.workers[1].run.batch_run.steps, ())
        self.assertEqual(run.workers[0].run.kv_final.free_blocks, 16)
        self.assertEqual(self.simulate(()).status, "completed")

    def test_source_waits_for_held_kv_then_wakes_when_copy_releases_it(self) -> None:
        p, d = self.config.workers
        p = replace(p, kv_config=KVConfig(p.gpu_id, 2, 2))
        run = self.simulate((Request("A", 0, 4, 2), Request("B", 0, 4, 2)), workers=(p, d))
        self.assertEqual([r.token_times_ms for r in run.batch_run.results], [(8, 15), (21, 28)])
        self.assertEqual(run.batch_run.results[1].start_ms, 13)
        self.assertEqual(run.status, "completed")

    def test_same_time_transfer_and_decode_completion_are_settled_before_dispatch(self) -> None:
        run = self.simulate((Request("A", 0, 4, 2), Request("B", 0, 1, 2)))
        self.assertEqual([r.token_times_ms for r in run.batch_run.results], [(8, 15), (10, 17)])
        events = [line for line in run.trace if "15ms 处理" in line]
        kinds = [line.split("处理 ")[1].split()[0] for line in events]
        self.assertLess(kinds.index("complete"), kinds.index("wakeup"))
        self.assertLess(kinds.index("transfer_complete"), kinds.index("wakeup"))

    def test_chunked_prefill_transfers_only_after_last_chunk(self) -> None:
        p, d = self.config.workers
        limits = replace(p.scheduler_profile.constraints, prefill_chunk_tokens=2)
        p = replace(p, scheduler_profile=replace(p.scheduler_profile, constraints=limits))
        run = self.simulate(workers=(p, d))
        self.assertEqual([r.token_times_ms for r in run.batch_run.results], [(8, 15, 17)])
        self.assertEqual(len(run.transfers), 1)
        self.assertEqual(run.transfers[0].ready_ms, 8)
        self.assertEqual([s.phase for s in run.batch_run.steps], ["Prefill", "Prefill", "Decode", "Decode"])

    def test_destination_shortage_is_rejected_not_silently_skipped(self) -> None:
        p, d = self.config.workers
        d = replace(d, kv_config=KVConfig(d.gpu_id, 1, 2))
        with self.assertRaisesRegex(ValueError, "目标KV容量不足"):
            self.simulate(workers=(p, d))

    def test_real_model_geometry_checks_transfer_bytes(self) -> None:
        settings = load_experiment(ROOT / "configs/measure_a6000_qwen25_3b.json").predictor_settings
        workers = tuple(replace(worker, predictor_settings=settings) for worker in self.config.workers)
        with self.assertRaisesRegex(ValueError, "Predictor模型结构"):
            self.config.pd_config.validate_workers(workers)
        # 3B示例: 2(K,V)×36层×2KV头×128维×2字节 = 36864 bytes/token。
        config = replace(self.config.pd_config, link=LinkConfig(1024000, 1, 36864))
        config.validate_workers(workers)

    def test_disjoint_transactions_allow_out_of_order_commit_and_rollback(self) -> None:
        pool = BlockPool(KVConfig("g", 4, 2))
        def plan(name):
            return BatchPlan((ReadyTask(name, "Prefill", 2),), "test")
        a, b = pool.reserve(plan("A")), pool.reserve(plan("B"))
        pool.begin_populating(a)
        pool.begin_populating(b)
        before = pool.snapshot()
        with self.assertRaisesRegex(ValueError, "同一请求"):
            pool.reserve(plan("A"))
        self.assertEqual(before, pool.snapshot())
        pool.complete(b)
        pool.rollback(a)
        self.assertEqual((pool.cached_tokens("A"), pool.cached_tokens("B"), pool.free_blocks), (0, 2, 3))
        self.assertEqual(len(before.reservations), 2)

    def test_invalid_pd_configuration_and_link_units(self) -> None:
        raw = json.loads((ROOT / "configs/one_p_one_d.json").read_text())
        raw["workload"]["path"] = str(ROOT / "workloads/pd_request.json")
        mutations = []
        for field, value in (("bandwidth_bytes_per_s", 0), ("bandwidth_bytes_per_s", True),
                             ("latency_ms", -1), ("kv_bytes_per_token", 0), ("kv_bytes_per_token", 1.5)):
            candidate = deepcopy(raw)
            candidate["pd"]["link"][field] = value
            mutations.append(candidate)
        candidate = deepcopy(raw)
        candidate.pop("pd")
        mutations.append(candidate)
        candidate = deepcopy(raw)
        candidate["workers"][1].pop("kv_cache")
        mutations.append(candidate)
        candidate = deepcopy(raw)
        candidate["workers"][0]["recovery"] = {"capacity_policy": "preempt"}
        mutations.append(candidate)
        candidate = deepcopy(raw)
        candidate["pd"]["decode_worker"] = "p0"
        mutations.append(candidate)
        candidate = deepcopy(raw)
        candidate["routing"] = {"policy": "fixed", "assignments": {"A": "p0"}}
        mutations.append(candidate)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for candidate in mutations:
                path.write_text(json.dumps(candidate))
                with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                    load_experiment(path)

    def test_cli_reports_transfer_and_end_to_end_metrics_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            completed = subprocess.run([sys.executable, str(ROOT / "run_simulation.py"), "--config",
                                        str(ROOT / "configs/one_p_one_d.json"), "--output", str(output)],
                                       capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report["execution_mode"], "fixed_1p1d")
            self.assertEqual(report["summary"]["completed_requests"], 1)
            self.assertEqual(report["summary"]["total_output_tokens"], 3)
            self.assertEqual(report["summary"]["window"]["duration_ms"], 17)
            self.assertEqual(report["requests"][0]["worker_path"], ["p0", "d0"])
            self.assertEqual(report["transfers"][0]["payload_bytes"], 4096)
            self.assertEqual([w["produced_output_tokens"] for w in report["workers"]], [1, 2])
            self.assertIn("8–13ms", completed.stdout)


if __name__ == "__main__":
    unittest.main()
