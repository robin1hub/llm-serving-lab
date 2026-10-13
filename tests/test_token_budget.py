"""手算分块进度、首 token 边界，并验证双预算约束与兼容性。"""

import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from simulator.engine import simulate_events
from simulator.models import Request, Timing
from simulator.scheduler import SchedulerConfig


class TokenBudgetTests(unittest.TestCase):
    def test_chunked_prefill_only_final_chunk_emits_first_token(self) -> None:
        run = simulate_events((Request("A", 0, 5, 3), Request("B", 3, 2, 2)), Timing(),
                              max_batch_requests=2, max_tokens_per_batch=3, prefill_chunk_tokens=2)
        prefills = [item for plan in run.plans for item in plan.items if item.phase == "Prefill"]
        self.assertEqual(tuple((i.request_id, i.range_start, i.token_count) for i in prefills),
                         (("A", 0, 2), ("A", 2, 2), ("A", 4, 1), ("B", 0, 2)))
        self.assertEqual(tuple(r.token_times_ms for r in run.batch_run.results), ((10, 16, 18), (14, 16)))
        self.assertEqual(run.batch_run.results[0].start_ms, 0)
        self.assertEqual(run.batch_run.results[1].start_ms, 10)
        self.assertEqual(run.batch_run.results[1].waiting_ms, 7)

    def test_token_budget_limits_decode_even_with_larger_request_limit(self) -> None:
        run = simulate_events(tuple(Request(i, 0, 1, 2) for i in "ABCD"), Timing(),
                              max_batch_requests=4, max_tokens_per_batch=2)
        self.assertEqual(tuple(p.request_ids for p in run.plans if p.phase == "Decode"),
                         (("A", "B"), ("C", "D")))

    def test_budget_one_and_single_output(self) -> None:
        run = simulate_events((Request("A", 5, 3, 1),), Timing(), max_tokens_per_batch=1)
        self.assertEqual(run.batch_run.results[0].token_times_ms, (11,))
        self.assertEqual(run.batch_run.results[0].start_ms, 5)
        self.assertEqual(len(run.plans), 3)
        self.assertTrue(all(p.phase == "Prefill" for p in run.plans))

    def test_random_progress_and_limits(self) -> None:
        rng = random.Random(42)
        requests = tuple(Request(str(i), rng.randint(0, 20), rng.randint(1, 10), rng.randint(1, 6)) for i in range(12))
        for policy in ("prefill_first", "decode_first"):
            run = simulate_events(requests, Timing(), policy=policy, max_batch_requests=3,
                                  max_tokens_per_batch=2, prefill_chunk_tokens=1)
            for plan in run.plans:
                self.assertLessEqual(len(plan.items), 3)
                self.assertLessEqual(sum(i.token_count for i in plan.items), 2)
            for result in run.batch_run.results:
                self.assertEqual(len(result.token_times_ms), result.request.output_tokens)
                ranges = [i for p in run.plans for i in p.items
                          if i.phase == "Prefill" and i.request_id == result.request.request_id]
                self.assertEqual(sum(i.token_count for i in ranges), result.request.input_tokens)
                self.assertEqual(tuple(i.range_start for i in ranges), tuple(range(result.request.input_tokens)))

    def test_unset_limits_and_large_limits_preserve_execution(self) -> None:
        requests = (Request("A", 0, 4, 3), Request("B", 3, 2, 2))
        for policy in ("prefill_first", "decode_first"):
            before = simulate_events(requests, Timing(), policy=policy)
            after = simulate_events(requests, Timing(), policy=policy, max_tokens_per_batch=99,
                                    prefill_chunk_tokens=99)
            self.assertEqual(before.batch_run, after.batch_run)

    def test_invalid_limits_and_config_cli(self) -> None:
        for field in ("max_tokens_per_batch", "prefill_chunk_tokens"):
            for value in (0, -1, True, 1.2, "2"):
                with self.assertRaises(ValueError):
                    SchedulerConfig(**{field: value})
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            result = subprocess.run([sys.executable, str(root / "run_simulation.py"), "--config",
                                     str(root / "configs/token_budget.json"), "--output", str(output)],
                                    cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report["scheduler"]["max_tokens_per_batch"], 3)
            self.assertEqual(report["scheduler"]["prefill_chunk_tokens"], 2)
            self.assertEqual(report["requests"][0]["token_times_ms"], [10, 16, 18])


if __name__ == "__main__":
    unittest.main()
