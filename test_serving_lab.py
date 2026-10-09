"""验证时间语义，尤其是首 token 和 Decode 轮数的边界。"""

import unittest

from serving_lab import Request, Timing, simulate, simulate_serial


class SingleRequestTests(unittest.TestCase):
    def test_three_tokens_need_two_decode_steps(self) -> None:
        result = simulate(Request("A", 0.0, 4, 3), Timing())
        self.assertEqual(result.token_times_ms, (8.0, 10.0, 12.0))
        self.assertEqual(result.ttft_ms, 8.0)
        self.assertEqual(result.inter_token_intervals_ms, (2.0, 2.0))
        self.assertEqual(result.latency_ms, 12.0)

    def test_single_output_finishes_at_prefill_completion(self) -> None:
        result = simulate(Request("A", 5.0, 4, 1), Timing())
        self.assertEqual(result.token_times_ms, (13.0,))
        self.assertEqual(result.inter_token_intervals_ms, ())
        self.assertEqual(result.ttft_ms, 8.0)
        self.assertEqual(result.latency_ms, 8.0)

    def test_arrival_shifts_clock_but_not_latency(self) -> None:
        result = simulate(Request("A", 100.0, 4, 3), Timing())
        self.assertEqual(result.token_times_ms, (108.0, 110.0, 112.0))
        self.assertEqual(result.latency_ms, 12.0)

    def test_invalid_inputs_are_rejected(self) -> None:
        for bad_count in (0, -1, 1.5, True):
            with self.subTest(bad_count=bad_count):
                with self.assertRaises(ValueError):
                    Request("A", 0.0, 4, bad_count)
        for bad_time in (-1.0, float("nan"), float("inf"), True):
            with self.subTest(bad_time=bad_time):
                with self.assertRaises(ValueError):
                    Request("A", bad_time, 4, 3)
        with self.assertRaises(ValueError):
            Timing(decode_ms_per_step=0.0)


class SerialRequestTests(unittest.TestCase):
    def test_second_request_waits_for_first_to_finish(self) -> None:
        first, second = simulate_serial(
            (Request("A", 0.0, 4, 3), Request("B", 3.0, 2, 2)), Timing()
        )
        self.assertEqual(first.token_times_ms, (8.0, 10.0, 12.0))
        self.assertEqual(second.start_ms, 12.0)
        self.assertEqual(second.waiting_ms, 9.0)
        self.assertEqual(second.token_times_ms, (16.0, 18.0))
        self.assertEqual(second.ttft_ms, 13.0)
        self.assertEqual(second.latency_ms, 15.0)
        self.assertEqual(second.events[0].time_ms, 3.0)

    def test_idle_gap_does_not_create_waiting(self) -> None:
        _, second = simulate_serial(
            (Request("A", 0.0, 4, 3), Request("B", 20.0, 2, 2)), Timing()
        )
        self.assertEqual(second.start_ms, 20.0)
        self.assertEqual(second.waiting_ms, 0.0)
        self.assertEqual(second.token_times_ms, (24.0, 26.0))

    def test_arrival_order_and_stable_ties(self) -> None:
        results = simulate_serial(
            (Request("B", 3.0, 1, 1), Request("A", 0.0, 1, 1), Request("C", 3.0, 1, 1)),
            Timing(),
        )
        self.assertEqual(tuple(result.request.request_id for result in results), ("A", "B", "C"))
        self.assertEqual(tuple(result.start_ms for result in results), (0.0, 3.0, 5.0))

    def test_empty_workload_and_duplicate_ids(self) -> None:
        self.assertEqual(simulate_serial((), Timing()), ())
        with self.assertRaises(ValueError):
            simulate_serial((Request("A", 0.0, 1, 1), Request("A", 1.0, 1, 1)), Timing())


if __name__ == "__main__":
    unittest.main()
