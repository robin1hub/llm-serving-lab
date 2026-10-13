"""验证采集描述与GPU安全边界；不需要GPU或vLLM依赖。"""

from types import SimpleNamespace as NS
import unittest

from calibration.scripts.batch_records import describe_work, select_device


def schedule(new=(), cached=(), contexts=(), counts=None, resumed=()):
    return NS(scheduled_new_reqs=new,
              scheduled_cached_reqs=NS(req_ids=cached, num_computed_tokens=contexts,
                                       resumed_from_preemption=resumed),
              num_scheduled_tokens=counts or {})


class CalibrationRecordTests(unittest.TestCase):
    def test_uuid_maps_to_physical_index_and_rejects_busy(self):
        data = "3, GPU-B, 2441 MiB, 0 %\n7, GPU-A, 1 MiB, 0 %"
        self.assertEqual(select_device(data, "GPU-A"), 7)
        with self.assertRaises(RuntimeError):
            select_device(data, "GPU-B")
        with self.assertRaises(ValueError):
            select_device(data, "GPU-missing")
        with self.assertRaises(RuntimeError):
            select_device("7, GPU-A, 1 MiB, 100 %", "GPU-A")

    def test_partial_prefill_then_final_input_and_decode_history(self):
        lengths = {}
        new = NS(req_id="A", prompt_token_ids=[1] * 5, num_computed_tokens=0)
        row = describe_work(schedule(new=(new,), counts={"A": 2}), lengths)[0]
        self.assertEqual((row["phase"], row["context_tokens"], row["produces_token"]),
                         ("Prefill", 0, False))
        row = describe_work(schedule(cached=("A",), contexts=(2,), counts={"A": 3}), lengths)[0]
        self.assertEqual((row["phase"], row["context_tokens"], row["produces_token"]),
                         ("Prefill", 2, True))
        row = describe_work(schedule(cached=("A",), contexts=(7,), counts={"A": 1}), lengths)[0]
        self.assertEqual((row["phase"], row["context_tokens"], row["new_tokens"]),
                         ("Decode", 7, 1))

    def test_batch_keeps_member_contexts_and_rejects_unsupported_recovery(self):
        lengths = {"A": 5, "B": 8}
        rows = describe_work(schedule(cached=("A", "B"), contexts=(6, 10),
                                      counts={"A": 1, "B": 1}), lengths)
        self.assertEqual([r["context_tokens"] for r in rows], [6, 10])
        with self.assertRaises(RuntimeError):
            describe_work(schedule(cached=("A",), contexts=(0,), counts={"A": 5},
                                   resumed=(True,)), lengths)
        with self.assertRaises(ValueError):
            describe_work(schedule(cached=("A",), contexts=(6,), counts={"A": 2}), lengths)


if __name__ == "__main__":
    unittest.main()
