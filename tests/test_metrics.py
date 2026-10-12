"""手算统计口径，避免把无样本、毫秒与秒、请求权重混淆。"""

import unittest

from simulator.engine import simulate_events
from simulator.metrics import distribution, summarize
from simulator.models import Request, Timing


class MetricsTests(unittest.TestCase):
    def test_hand_calculated_two_requests(self) -> None:
        run = simulate_events((Request("A", 0, 4, 3), Request("B", 3, 2, 2)), Timing())
        summary = summarize(run.batch_run)
        self.assertEqual(summary["total_input_tokens"], 6)
        self.assertEqual(summary["total_output_tokens"], 5)
        self.assertEqual(summary["ttft_ms"], {"count": 2, "mean": 8.5, "p50": 8, "p95": 9})
        self.assertEqual(summary["latency_ms"]["mean"], 13.5)
        self.assertAlmostEqual(summary["itl_ms"]["mean"], 10 / 3)
        self.assertEqual(summary["itl_ms"]["p95"], 6)
        self.assertEqual(summary["window"]["duration_ms"], 16)
        self.assertEqual(summary["request_throughput_per_second"], 125)
        self.assertEqual(summary["output_throughput_tokens_per_second"], 312.5)

    def test_empty_run_and_single_output_have_no_itl(self) -> None:
        empty = summarize(simulate_events((), Timing()).batch_run)
        self.assertEqual(empty["completed_requests"], 0)
        self.assertIsNone(empty["ttft_ms"]["mean"])
        self.assertIsNone(empty["output_throughput_tokens_per_second"])
        single = summarize(simulate_events((Request("A", 100, 4, 1),), Timing()).batch_run)
        self.assertIsNone(single["itl_ms"]["mean"])
        self.assertEqual(single["window"]["duration_ms"], 8)

    def test_nearest_rank_and_order_independence(self) -> None:
        self.assertEqual(distribution((4, 1, 3, 2)), {"count": 4, "mean": 2.5, "p50": 2, "p95": 4})

    def test_window_includes_idle_and_excludes_before_first_arrival(self) -> None:
        summary = summarize(simulate_events((Request("A", 100, 1, 1), Request("B", 200, 1, 1)), Timing()).batch_run)
        self.assertEqual(summary["window"], {"definition": "first_arrival_to_last_completion",
                                            "start_ms": 100, "end_ms": 202, "duration_ms": 102})
        self.assertAlmostEqual(summary["request_throughput_per_second"], 2 / 0.102)


if __name__ == "__main__":
    unittest.main()
