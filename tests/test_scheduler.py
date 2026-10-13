"""核对只读决策、轮转、人数限制和原有时间线兼容性。"""

import json
from pathlib import Path
import tempfile
import unittest

from simulator.config import load_experiment
from simulator.engine import simulate_events
from simulator.models import Request, Timing
from simulator.scheduler import ReadyTask, Scheduler, SchedulerConfig, SchedulingSnapshot


class SchedulerTests(unittest.TestCase):
    def test_read_only_selection_and_idle(self) -> None:
        snapshot = SchedulingSnapshot((), tuple(ReadyTask(item, "Decode", 1) for item in "ABCD"))
        scheduler = Scheduler(SchedulerConfig(max_batch_requests=2))
        first = scheduler.decide(snapshot)
        self.assertEqual(first.request_ids, ("A", "B"))
        self.assertEqual(scheduler.decide(snapshot), first)
        self.assertEqual(len(snapshot.ready_decode), 4)
        self.assertIsNone(scheduler.decide(SchedulingSnapshot((), ())))

    def test_four_request_rotation_and_completion(self) -> None:
        requests = tuple(Request(item, 0, 1, 3) for item in "ABCD")
        run = simulate_events(requests, Timing(), max_batch_requests=2)
        decode = [step for step in run.batch_run.steps if step.phase == "Decode"]
        self.assertEqual(tuple(step.request_ids for step in decode),
                         (("A", "B"), ("C", "D"), ("A", "B"), ("C", "D")))
        self.assertEqual(tuple(r.token_times_ms for r in run.batch_run.results),
                         ((2, 10, 14), (4, 10, 14), (6, 12, 16), (8, 12, 16)))
        self.assertTrue(all(len(plan.items) <= 2 for plan in run.plans))
        self.assertTrue(all(item.token_count == 1 for plan in run.plans for item in plan.items))

    def test_finished_member_removed_before_rotation(self) -> None:
        requests = (Request("A", 0, 1, 2), Request("B", 0, 1, 3), Request("C", 0, 1, 2))
        run = simulate_events(requests, Timing(), max_batch_requests=2)
        self.assertEqual(tuple(p.request_ids for p in run.plans if p.phase == "Decode"),
                         (("A", "B"), ("C", "B")))

    def test_limit_one_and_unlimited(self) -> None:
        requests = (Request("A", 0, 4, 3), Request("B", 3, 2, 2))
        old = simulate_events(requests, Timing())
        large = simulate_events(requests, Timing(), max_batch_requests=32)
        self.assertEqual(old.batch_run, large.batch_run)
        single = simulate_events(requests, Timing(), max_batch_requests=1)
        self.assertEqual(tuple(p.request_ids for p in single.plans if p.phase == "Decode"),
                         (("A",), ("B",), ("A",)))
        strict = simulate_events(requests, Timing(), policy="decode_first", max_batch_requests=1)
        self.assertEqual(tuple(r.token_times_ms for r in strict.batch_run.results), ((8, 10, 12), (16, 18)))

    def test_configuration_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "requests.json").write_text("[]")
            config = {"workload": {"path": "requests.json"},
                      "scheduler": {"policy": "prefill_first"},
                      "timing": {"prefill_ms_per_input_token": 2, "decode_ms_per_step": 2}}
            path = root / "config.json"
            for value in (0, -1, True, 1.5, "2"):
                config["scheduler"]["max_batch_requests"] = value
                path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    load_experiment(path)
                with self.assertRaises(ValueError):
                    simulate_events((), Timing(), max_batch_requests=value)
            for value in (None, 2):
                config["scheduler"]["max_batch_requests"] = value
                path.write_text(json.dumps(config))
                self.assertEqual(load_experiment(path).max_batch_requests, value)


if __name__ == "__main__":
    unittest.main()
