"""用手算时间和 batch 成员验证轮次边界与动态进出。"""

import unittest

from batching_lab import simulate_batched
from serving_lab import Request, Timing, simulate


class BatchingTests(unittest.TestCase):
    def test_shared_decode_step_and_dynamic_exit(self) -> None:
        run = simulate_batched((Request("A", 0, 4, 3), Request("B", 3, 2, 2)), Timing())
        self.assertEqual(tuple(item.token_times_ms for item in run.results), ((8, 14, 16), (12, 14)))
        self.assertEqual(
            tuple((step.phase, step.request_ids) for step in run.steps),
            (("Prefill", ("A",)), ("Prefill", ("B",)), ("Decode", ("A", "B")), ("Decode", ("A",))),
        )
        self.assertEqual(run.results[0].inter_token_intervals_ms, (6, 2))
        self.assertEqual(run.results[1].ttft_ms, 9)

    def test_arrival_during_decode_waits_until_next_boundary(self) -> None:
        # A 在 2–4ms Decode；3ms 到达的 B 不能中途插入。
        run = simulate_batched((Request("A", 0, 1, 4), Request("B", 3, 1, 2)), Timing())
        self.assertEqual(run.results[1].start_ms, 4)
        self.assertEqual(tuple(step.request_ids for step in run.steps),
                         (("A",), ("A",), ("B",), ("A", "B"), ("A",)))
        self.assertEqual(run.results[0].token_times_ms, (2, 4, 8, 10))

    def test_single_token_and_idle_gap(self) -> None:
        run = simulate_batched((Request("A", 0, 1, 1), Request("B", 20, 1, 1)), Timing())
        self.assertEqual(tuple(step.phase for step in run.steps), ("Prefill", "Prefill"))
        self.assertEqual(run.results[1].waiting_ms, 0)
        self.assertEqual(run.results[1].token_times_ms, (22,))

    def test_single_request_matches_original_model(self) -> None:
        request = Request("A", 5, 4, 3)
        result = simulate_batched((request,), Timing()).results[0]
        self.assertEqual(result.token_times_ms, simulate(request, Timing()).token_times_ms)

    def test_stable_arrival_order_empty_and_duplicate_ids(self) -> None:
        run = simulate_batched((Request("B", 2, 1, 2), Request("A", 0, 1, 2), Request("C", 2, 1, 2)), Timing())
        self.assertEqual(tuple(step.request_ids for step in run.steps[:3]), (("A",), ("B",), ("C",)))
        self.assertEqual(simulate_batched((), Timing()).results, ())
        self.assertEqual(simulate_batched((), Timing()).steps, ())
        with self.assertRaises(ValueError):
            simulate_batched((Request("A", 0, 1, 1), Request("A", 2, 1, 1)), Timing())


if __name__ == "__main__":
    unittest.main()
