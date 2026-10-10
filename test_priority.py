"""验证优先级确实改变时间分配，并保留原有默认行为。"""

import unittest

from batching_lab import simulate_batched
from serving_lab import Request, Timing, simulate_serial


class PriorityTests(unittest.TestCase):
    def test_decode_priority_protects_a_and_delays_b(self) -> None:
        requests = (Request("A", 0, 4, 3), Request("B", 3, 2, 2))
        timing = Timing()
        prefill = simulate_batched(requests, timing)
        decode = simulate_batched(requests, timing, policy="decode_first")
        self.assertEqual(prefill.results[0].token_times_ms, (8, 14, 16))
        self.assertEqual(decode.results[0].token_times_ms, (8, 10, 12))
        self.assertEqual(decode.results[1].token_times_ms, (16, 18))
        self.assertEqual(decode.results[1].ttft_ms, 13)
        self.assertEqual(tuple(result.token_times_ms for result in decode.results),
                         tuple(result.token_times_ms for result in simulate_serial(requests, timing)))
        self.assertEqual(tuple(step.phase for step in decode.steps),
                         ("Prefill", "Decode", "Decode", "Prefill", "Decode"))

    def test_decode_priority_starts_prefill_when_no_decode_exists(self) -> None:
        run = simulate_batched((Request("A", 5, 1, 1), Request("B", 20, 1, 2)),
                               Timing(), policy="decode_first")
        self.assertEqual(run.results[0].token_times_ms, (7,))
        self.assertEqual(run.results[1].token_times_ms, (22, 24))
        self.assertEqual(simulate_batched((), Timing(), policy="decode_first").steps, ())

    def test_unknown_policy_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            simulate_batched((), Timing(), policy="unknown")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
