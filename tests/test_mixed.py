"""混合预算、完成屏障、分块/轮转及两种耗时模型的手算核对。"""

from pathlib import Path
import unittest

from simulator.config import load_experiment
from simulator.engine import simulate_events
from simulator.models import Request, Timing
from simulator.predictor import build_predictor
from simulator.scheduler import ReadyTask, Scheduler, SchedulerConfig, SchedulingSnapshot


ROOT = Path(__file__).resolve().parents[1]


class MixedTests(unittest.TestCase):
    def test_budget_and_read_only_selection(self) -> None:
        snapshot = SchedulingSnapshot((ReadyTask("P", "Prefill", 10, 2),),
                                     tuple(ReadyTask(item, "Decode", 1) for item in "ABC"))
        scheduler = Scheduler(SchedulerConfig("mixed_decode_first", 4, 8, 6))
        plan = scheduler.decide(snapshot)
        self.assertEqual(plan.phase, "Mixed")
        self.assertEqual([(i.request_id, i.phase, i.token_count) for i in plan.items],
                         [("A", "Decode", 1), ("B", "Decode", 1), ("C", "Decode", 1), ("P", "Prefill", 5)])
        self.assertEqual(plan.items[-1].range_start, 2)
        self.assertEqual(scheduler.decide(snapshot), plan)
        for cap, budget in ((3, 8), (4, 3), (1, 1)):
            limited = Scheduler(SchedulerConfig("mixed_decode_first", cap, budget)).decide(snapshot)
            self.assertEqual(limited.phase, "Decode")
            self.assertLessEqual(len(limited.items), cap)
            self.assertLessEqual(sum(i.token_count for i in limited.items), budget)

    def test_teaching_timeline_and_intermediate_chunks(self) -> None:
        experiment = load_experiment(ROOT / "configs/worker_mixed.json")
        run = simulate_events(experiment.requests, experiment.timing,
                              scheduler_profile=experiment.scheduler_profile)
        self.assertEqual([s.phase for s in run.batch_run.steps],
                         ["Prefill", "Mixed", "Mixed", "Mixed", "Decode"])
        self.assertEqual([r.token_times_ms for r in run.batch_run.results], [(2, 8, 14, 20), (20, 22)])
        self.assertEqual([e.duration_ms for e in run.estimates], [2, 6, 6, 6, 2])
        self.assertEqual([(i.range_start, i.token_count) for p in run.plans for i in p.items
                          if i.phase == "Prefill" and i.request_id == "B"], [(0, 2), (2, 2), (4, 2)])
        self.assertEqual(run.batch_run.results[1].start_ms, 2)

    def test_roofline_joint_estimate_and_rotation(self) -> None:
        experiment = load_experiment(ROOT / "configs/predictor_mixed.json")
        run = simulate_events(experiment.requests, experiment.timing,
            scheduler_profile=experiment.scheduler_profile,
            predictor=build_predictor(experiment.predictor_settings, experiment.timing))
        self.assertEqual([p.request_ids for p in run.plans], [("A",), ("A",), ("A", "B"), ("B", "A")])
        estimate = run.estimates[2]
        self.assertEqual([(i.phase, i.context_tokens) for i in estimate.items], [("Decode", 3), ("Prefill", 0)])
        self.assertEqual([(op.flops, op.memory_bytes) for op in estimate.operators], [(704, 384), (80, 40)])
        self.assertAlmostEqual(estimate.duration_ms, 0.884)
        for step, end in zip(run.batch_run.steps, (0.724, 1.240, 2.124, 3.040)):
            self.assertAlmostEqual(step.end_ms, end)

    def test_arrival_during_batch_waits_for_completion(self) -> None:
        run = simulate_events((Request("A", 0, 1, 3), Request("B", 1, 2, 2), Request("C", 3, 1, 1)),
                              Timing(), policy="mixed_decode_first", max_batch_requests=3,
                              max_tokens_per_batch=3, prefill_chunk_tokens=2)
        self.assertEqual(run.batch_run.steps[1].request_ids, ("A", "B"))
        self.assertEqual(run.batch_run.steps[1].end_ms, 8)
        c_result = next(r for r in run.batch_run.results if r.request.request_id == "C")
        self.assertEqual(c_result.start_ms, 8)
        for first, second in zip(run.batch_run.steps, run.batch_run.steps[1:]):
            self.assertLessEqual(first.end_ms, second.start_ms)
        self.assertEqual(len(c_result.token_times_ms), 1)

    def test_saturated_decode_then_prefill_makes_progress(self) -> None:
        requests = (Request("A", 0, 1, 3), Request("B", 1, 2, 1))
        run = simulate_events(requests, Timing(), policy="mixed_decode_first", max_batch_requests=1,
                              max_tokens_per_batch=2, prefill_chunk_tokens=1)
        self.assertNotIn("Mixed", [p.phase for p in run.plans])
        self.assertEqual([r.token_times_ms for r in run.batch_run.results], [(2, 4, 6), (10,)])
        # 有限Decode完成后才释放槽位；不是隐式为Prefill预留槽或抢占。
        self.assertEqual(run.batch_run.results[1].start_ms, 6)


if __name__ == "__main__":
    unittest.main()
