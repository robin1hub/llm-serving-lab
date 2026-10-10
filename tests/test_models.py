"""保留当前数据结构的输入约束和指标边界验证。"""

import unittest

from simulator.engine import simulate_events
from simulator.models import Request, Timing


class ModelTests(unittest.TestCase):
    def test_single_output_has_no_interval(self) -> None:
        result = simulate_events((Request("A", 5, 4, 1),), Timing()).batch_run.results[0]
        self.assertEqual(result.token_times_ms, (13,))
        self.assertEqual(result.ttft_ms, 8)
        self.assertEqual(result.inter_token_intervals_ms, ())

    def test_rejects_invalid_lengths(self) -> None:
        for count in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                Request("A", 0, 4, count)

    def test_rejects_invalid_times(self) -> None:
        for time in (-1, True, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                Request("A", time, 4, 3)
        with self.assertRaises(ValueError):
            Timing(decode_ms_per_step=0)


if __name__ == "__main__":
    unittest.main()
