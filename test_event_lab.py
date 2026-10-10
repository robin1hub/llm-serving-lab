"""用轮次模型作为对照，核对事件驱动的边界和资源互斥。"""

import random
import unittest

from batching_lab import simulate_batched
from event_lab import simulate_events
from serving_lab import Request, Timing


class EventTests(unittest.TestCase):
    def test_equivalence_on_boundaries_and_seeded_workloads(self) -> None:
        rng = random.Random(42)
        workloads = [(), (Request("A", 0, 4, 3), Request("B", 3, 2, 2)),
                     (Request("A", 0, 1, 4), Request("B", 2, 1, 2), Request("C", 3, 1, 1)),
                     (Request("A", 5, 1, 1), Request("B", 20, 1, 2))]
        for _ in range(20):
            workloads.append(tuple(Request(str(i), rng.randint(0, 25), rng.randint(1, 6),
                                           rng.randint(1, 5)) for i in range(8)))
        for policy in ("prefill_first", "decode_first"):
            for requests in workloads:
                with self.subTest(policy=policy, requests=requests):
                    old = simulate_batched(requests, Timing(), policy=policy)
                    new = simulate_events(requests, Timing(), policy=policy).batch_run
                    self.assertEqual(new.steps, old.steps)
                    self.assertEqual(tuple((r.start_ms, r.token_times_ms) for r in new.results),
                                     tuple((r.start_ms, r.token_times_ms) for r in old.results))
                    for earlier, later in zip(new.steps, new.steps[1:]):
                        self.assertLessEqual(earlier.end_ms, later.start_ms)

    def test_arrival_is_processed_while_gpu_busy(self) -> None:
        run = simulate_events((Request("A", 0, 4, 3), Request("B", 3, 2, 2)), Timing())
        arrivals = next(i for i, line in enumerate(run.trace) if line == "3ms 处理 arrival ['B']")
        completion = next(i for i, line in enumerate(run.trace) if line == "8ms 处理 complete ['A']")
        self.assertLess(arrivals, completion)
        self.assertIn("GPU 忙碌，继续等待", run.trace[arrivals + 1])

    def test_invalid_policy_and_duplicate_ids(self) -> None:
        with self.assertRaises(ValueError):
            simulate_events((), Timing(), policy="unknown")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            simulate_events((Request("A", 0, 1, 1), Request("A", 1, 1, 1)), Timing())


if __name__ == "__main__":
    unittest.main()
