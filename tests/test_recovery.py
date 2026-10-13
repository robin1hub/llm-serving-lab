"""选择策略、KV重建目标、输出不重复、反复抢占保护与配置边界。"""

import json
from pathlib import Path
import tempfile
import unittest

from simulator.config import load_experiment
from simulator.engine import simulate_events
from simulator.kv_cache import KVConfig
from simulator.models import Request, Timing
from simulator.predictor import Calibration, DeviceProfile, ModelProfile, RooflinePredictor
from simulator.recovery import RecoveryCandidate, RecoveryConfig, select_victim


ROOT = Path(__file__).resolve().parents[1]
PREEMPT = RecoveryConfig("preempt")


class RecoveryTests(unittest.TestCase):
    def test_victim_selection(self) -> None:
        candidates = (RecoveryCandidate("A", 1, 0), RecoveryCandidate("B", 3, 1))
        self.assertEqual(select_victim(PREEMPT, candidates), "A")
        self.assertEqual(select_victim(RecoveryConfig("preempt", "newest"), candidates), "B")
        self.assertIsNone(select_victim(PREEMPT, ()))

    def test_recompute_hand_timeline_and_block_default(self) -> None:
        experiment = load_experiment(ROOT / "configs/recompute.json")
        kwargs = {"scheduler_profile": experiment.scheduler_profile, "kv_config": experiment.kv_config}
        blocked = simulate_events(experiment.requests, experiment.timing, **kwargs)
        self.assertEqual(blocked.status, "blocked")
        run = simulate_events(experiment.requests, experiment.timing, recovery_config=PREEMPT, **kwargs)
        self.assertEqual(run.status, "completed")
        self.assertEqual([r.token_times_ms for r in run.batch_run.results], [(4, 10, 12), (8, 18, 20)])
        self.assertEqual([(r.action, r.request_id, r.time_ms) for r in run.recovery_records],
                         [("preempt", "B", 8), ("restore_start", "B", 12), ("resume", "B", 16)])
        replay = next(e for e in run.estimates if e.items[0].phase == "Recompute")
        self.assertEqual(replay.duration_ms, 4)
        self.assertFalse(replay.items[0].produces_token)
        self.assertEqual(run.kv_final.free_blocks, 2)
        self.assertEqual(run.recovery_pending, ())

    def test_generated_history_replayed_without_duplicate_tokens(self) -> None:
        run = simulate_events((Request("A", 0, 2, 5), Request("B", 0, 2, 3)), Timing(),
            max_batch_requests=2, prefill_chunk_tokens=2, kv_config=KVConfig("gpu_0", 3, 2),
            recovery_config=PREEMPT)
        self.assertEqual(run.status, "completed")
        self.assertEqual([r.token_times_ms for r in run.batch_run.results], [(4, 10, 12, 26, 28), (8, 14, 16)])
        self.assertEqual(run.recovery_records[0].request_id, "A")
        self.assertEqual(run.recovery_records[0].cached_tokens, 4)
        replays = [i for p in run.plans for i in p.items if i.phase == "Recompute"]
        self.assertEqual([(i.range_start, i.token_count) for i in replays], [(0, 2), (2, 2)])

    def test_partial_prefill_preserves_start_and_original_progress(self) -> None:
        run = simulate_events((Request("A", 0, 2, 3), Request("B", 0, 3, 2)), Timing(),
            prefill_chunk_tokens=2, kv_config=KVConfig("gpu_0", 2, 2), recovery_config=PREEMPT)
        self.assertEqual(run.status, "completed")
        self.assertEqual([r.token_times_ms for r in run.batch_run.results], [(4, 10, 12), (18, 20)])
        self.assertEqual(run.batch_run.results[1].start_ms, 4)
        self.assertEqual(run.recovery_records[0].cached_tokens, 2)
        self.assertEqual([i.range_start for p in run.plans for i in p.items
                          if i.request_id == "B" and i.phase == "Prefill"], [0, 2])

    def test_impossible_single_request_stops_without_self_preemption(self) -> None:
        run = simulate_events((Request("A", 0, 2, 3),), Timing(), kv_config=KVConfig("gpu_0", 1, 2),
                              recovery_config=PREEMPT)
        self.assertEqual(run.status, "blocked")
        self.assertEqual(run.recovery_records, ())
        self.assertEqual(run.pending_progress[0].token_times_ms, (4,))

    def test_roofline_recompute_has_no_logits(self) -> None:
        predictor = RooflinePredictor(ModelProfile(1, 4, 8, 2, 1, 8),
                                     DeviceProfile("toy", 1e6, 1e6), Calibration())
        run = simulate_events((Request("A", 0, 2, 3), Request("B", 0, 2, 3)), Timing(),
            kv_config=KVConfig("gpu_0", 2, 2), recovery_config=PREEMPT, predictor=predictor)
        self.assertEqual(run.status, "completed")
        replay = next(e for e in run.estimates if e.items[0].phase == "Recompute")
        self.assertFalse(replay.items[0].produces_token)
        self.assertEqual(replay.operators[0].flops, 576)
        self.assertEqual([len(r.token_times_ms) for r in run.batch_run.results], [3, 3])

    def test_fifo_restore_before_new_prefill_and_token_budget(self) -> None:
        requests = tuple(Request(item, 0, 2, 3) for item in "ABC")
        run = simulate_events(requests, Timing(), max_tokens_per_batch=1, prefill_chunk_tokens=1,
            kv_config=KVConfig("gpu_0", 2, 2), recovery_config=PREEMPT)
        self.assertEqual(run.status, "completed")
        self.assertEqual([r.token_times_ms for r in run.batch_run.results],
                         [(4, 10, 12), (8, 18, 20), (24, 26, 28)])
        self.assertEqual(sum(r.action == "preempt" for r in run.recovery_records), 1)
        self.assertTrue(all(sum(i.token_count for i in p.items) <= 1 for p in run.plans))
        self.assertEqual(run.kv_final.free_blocks, 2)

    def test_config_validation(self) -> None:
        for args in (("preempt", "typo", "recompute"), ("preempt", "newest", "swap"),
                     ("typo", "newest", "recompute")):
            with self.assertRaises(ValueError):
                RecoveryConfig(*args)
        with tempfile.TemporaryDirectory() as directory:
            config = json.loads((ROOT / "configs/basic.json").read_text())
            config["workload"]["path"] = str(ROOT / "workloads/two_requests.json")
            config["recovery"] = {"capacity_policy": "preempt"}
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_experiment(path)


if __name__ == "__main__":
    unittest.main()
