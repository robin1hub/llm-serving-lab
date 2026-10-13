"""KV身份、页边界、事务回滚、完成释放和容量阻塞的确定性验证。"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from simulator.config import load_experiment
from simulator.engine import simulate_events
from simulator.models import Request, Timing
from simulator.kv_cache import BlockPool, KVConfig
from simulator.scheduler import BatchPlan, ReadyTask


ROOT = Path(__file__).resolve().parents[1]


def prefill(request_id: str, count: int, start: int = 0) -> BatchPlan:
    return BatchPlan((ReadyTask(request_id, "Prefill", count, start),), "test")


def write(pool: BlockPool, plan: BatchPlan) -> None:
    reservation = pool.reserve(plan)
    pool.begin_populating(reservation)
    pool.complete(reservation)


class KVTests(unittest.TestCase):
    def test_identity_lifecycle_and_fragmentation(self) -> None:
        pool = BlockPool(KVConfig("gpu_0", 4, 4))
        reservation = pool.reserve(prefill("A", 6))
        reserved = pool.snapshot()
        self.assertEqual([b.state for b in reserved.blocks], ["Reserved", "Reserved"])
        self.assertEqual(pool.cached_tokens("A"), 0)
        self.assertEqual(reserved.free_blocks, 2)
        with self.assertRaises(ValueError):
            pool.complete(reservation)
        with self.assertRaises(ValueError):
            pool.release_request("A")
        pool.begin_populating(reservation)
        self.assertTrue(all(b.state == "Populating" for b in pool.snapshot().blocks))
        pool.complete(reservation)
        ready = pool.snapshot()
        self.assertEqual([(b.valid_start, b.valid_end) for b in ready.blocks], [(0, 4), (4, 6)])
        self.assertEqual(ready.internal_fragmentation_tokens, 2)
        self.assertEqual(ready.tables[0].block_ids, (1, 2))
        self.assertEqual(ready.blocks[0].gpu_id, "gpu_0")
        self.assertEqual(ready.blocks[0].request_refs, ("A",))
        write(pool, BatchPlan((ReadyTask("A", "Decode", 1),), "decode"))
        self.assertEqual(pool.snapshot().tables[0].block_ids, (1, 2))
        self.assertEqual(pool.cached_tokens("A"), 7)
        self.assertEqual(pool.snapshot().internal_fragmentation_tokens, 1)
        pool.release_request("A")
        self.assertEqual(pool.free_blocks, 4)
        self.assertEqual(pool.snapshot().blocks, ())
        # 冻结历史不会被后续完成或释放修改，旧事务不能再次完成。
        self.assertEqual([b.state for b in reserved.blocks], ["Reserved", "Reserved"])
        with self.assertRaises(ValueError):
            pool.complete(reservation)

    def test_atomic_failure_and_rollback(self) -> None:
        pool = BlockPool(KVConfig("gpu_0", 2, 2))
        write(pool, prefill("A", 1))
        before = pool.snapshot()
        too_large = BatchPlan((ReadyTask("A", "Decode", 1), ReadyTask("B", "Prefill", 4)), "test")
        with self.assertRaises(ValueError):
            pool.reserve(too_large)
        self.assertEqual(pool.snapshot(), before)
        reservation = pool.reserve(BatchPlan((ReadyTask("A", "Decode", 1),
                                             ReadyTask("B", "Prefill", 2)), "test"))
        pool.begin_populating(reservation)
        pool.rollback(reservation)
        after = pool.snapshot()
        self.assertEqual(after.blocks, before.blocks)
        self.assertEqual(after.tables, before.tables)
        self.assertEqual(after.free_blocks, before.free_blocks)
        with self.assertRaises(ValueError):
            pool.rollback(reservation)
        with self.assertRaises(ValueError):
            pool.reserve(prefill("A", 1, 0))

    def test_read_only_admission_shrinks_prefill_after_decode(self) -> None:
        pool = BlockPool(KVConfig("gpu_0", 2, 2))
        write(pool, prefill("A", 1))
        before = pool.snapshot()
        candidate = BatchPlan((ReadyTask("A", "Decode", 1), ReadyTask("B", "Prefill", 3)), "test")
        admitted = pool.admit(candidate)
        self.assertEqual([i.token_count for i in admitted.items], [1, 2])
        self.assertEqual(candidate.items[1].token_count, 3)
        self.assertEqual(pool.snapshot(), before)
        write(pool, admitted)
        self.assertEqual(pool.free_blocks, 0)
        # A正好满2个token，下一轮跨页；B尾块也满，均不能继续写。
        self.assertIsNone(pool.admit(BatchPlan((ReadyTask("A", "Decode", 1),), "test")))

    def test_capacity_timeline_and_release(self) -> None:
        experiment = load_experiment(ROOT / "configs/kv_blocks.json")
        run = simulate_events(experiment.requests, experiment.timing,
            scheduler_profile=experiment.scheduler_profile, kv_config=experiment.kv_config)
        self.assertEqual(run.status, "completed")
        self.assertEqual([r.token_times_ms for r in run.batch_run.results], [(4, 6, 8), (14, 16)])
        self.assertEqual(run.kv_final.free_blocks, 2)
        self.assertEqual(run.kv_final.peak_used_blocks, 2)
        self.assertEqual(run.kv_final.tables, ())
        self.assertIn("release:A", [record.action for record in run.kv_records])
        self.assertTrue(all(0 <= record.snapshot.free_blocks <= 2 for record in run.kv_records))

    def test_mixed_shrink_then_completed_request_reclaims(self) -> None:
        run = simulate_events((Request("A", 0, 1, 2), Request("B", 1, 3, 1)), Timing(),
            policy="mixed_decode_first", max_batch_requests=2, max_tokens_per_batch=4,
            kv_config=KVConfig("gpu_0", 2, 2))
        self.assertEqual([p.phase for p in run.plans], ["Prefill", "Mixed", "Prefill"])
        self.assertEqual(run.plans[1].items[-1].token_count, 2)
        self.assertEqual([r.token_times_ms for r in run.batch_run.results], [(2, 8), (10,)])
        self.assertEqual(run.kv_final.free_blocks, 2)

    def test_prefill_blocked_falls_back_to_decode(self) -> None:
        run = simulate_events((Request("A", 0, 1, 2), Request("B", 1, 2, 1)), Timing(),
            policy="prefill_first", kv_config=KVConfig("gpu_0", 1, 2))
        self.assertEqual(run.status, "completed")
        self.assertEqual([p.phase for p in run.plans], ["Prefill", "Decode", "Prefill"])
        self.assertIn("回退Decode", run.plans[1].reason)

    def test_blocked_report_and_cli_exit(self) -> None:
        experiment = load_experiment(ROOT / "configs/kv_blocked.json")
        run = simulate_events(experiment.requests, experiment.timing,
            scheduler_profile=experiment.scheduler_profile, kv_config=experiment.kv_config)
        self.assertEqual(run.status, "blocked")
        self.assertEqual(run.pending_request_ids, ("A", "B"))
        self.assertEqual(run.batch_run.results, ())
        self.assertEqual(run.pending_progress[0].token_times_ms, (4,))
        self.assertEqual(run.kv_final.tables[0].cached_tokens, 2)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "blocked.json"
            completed = subprocess.run([sys.executable, str(ROOT / "run_simulation.py"), "--config",
                str(ROOT / "configs/kv_blocked.json"), "--output", str(output), "--summary"],
                capture_output=True, text=True)
            self.assertEqual(completed.returncode, 2)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "blocked")
            self.assertEqual(report["summary"]["completed_requests"], 0)
            self.assertEqual(report["pending_progress"][0]["token_times_ms"], [4])

    def test_config_validation_and_disabled_compatibility(self) -> None:
        for cap, size in ((0, 2), (2, True), (-1, 2)):
            with self.assertRaises(ValueError):
                KVConfig("gpu_0", cap, size)
        requests = (Request("A", 0, 4, 3), Request("B", 3, 2, 2))
        old = simulate_events(requests, Timing())
        large = simulate_events(requests, Timing(), kv_config=KVConfig("gpu_0", 100, 2))
        self.assertEqual(old.batch_run, large.batch_run)
        self.assertIsNone(old.kv_final)
        self.assertEqual(old.kv_records, ())
        self.assertEqual(large.kv_final.free_blocks, 100)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config = json.loads((ROOT / "configs/basic.json").read_text())
            config["workload"]["path"] = str(ROOT / "workloads/two_requests.json")
            for value in (None, {"gpu_id": "gpu_0", "total_blocks": 2, "block_tokens": 2, "typo": 1}):
                config["kv_cache"] = value
                path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    load_experiment(path)


if __name__ == "__main__":
    unittest.main()
