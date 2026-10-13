"""方案引用、角色能力和完成驱动交替；时间线用手算结果核对。"""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from simulator.config import load_experiment
from simulator.engine import simulate_events
from simulator.models import Timing
from simulator.scheduler import ReadyTask, Scheduler, SchedulerConfig, SchedulerProfile, SchedulingSnapshot


ROOT = Path(__file__).resolve().parents[1]


class ProfileTests(unittest.TestCase):
    def test_alternating_changes_only_after_completion(self) -> None:
        scheduler = Scheduler(SchedulerConfig(), profile=SchedulerProfile(
            "fcfs_chunked", "round_robin", "alternating"))
        snapshot = SchedulingSnapshot((ReadyTask("P", "Prefill", 2),),
                                      (ReadyTask("D", "Decode", 1),))
        first = scheduler.decide(snapshot)
        self.assertEqual(first.phase, "Prefill")
        self.assertEqual(scheduler.decide(snapshot), first)
        scheduler.on_batch_completed(first)
        second = scheduler.decide(snapshot)
        self.assertEqual(second.phase, "Decode")
        self.assertEqual(scheduler.decide(snapshot), second)
        scheduler.on_batch_completed(second)
        self.assertEqual(scheduler.decide(snapshot).phase, "Prefill")

    def test_one_stage_never_waits_for_turn(self) -> None:
        scheduler = Scheduler(SchedulerConfig(policy="alternating"))
        only_p = SchedulingSnapshot((ReadyTask("P", "Prefill", 2),), ())
        only_d = SchedulingSnapshot((), (ReadyTask("D", "Decode", 1),))
        for snapshot in (only_p, only_p, only_d, only_d):
            plan = scheduler.decide(snapshot)
            self.assertIsNotNone(plan)
            scheduler.on_batch_completed(plan)
        self.assertIsNone(scheduler.decide(SchedulingSnapshot((), ())))

    def test_role_capabilities(self) -> None:
        p = Scheduler(SchedulerConfig(), profile=SchedulerProfile(prefill_algorithm="fcfs_chunked"), role="P")
        d = Scheduler(SchedulerConfig(), profile=SchedulerProfile(decode_algorithm="round_robin"), role="D")
        only_p = SchedulingSnapshot((ReadyTask("P", "Prefill", 2),), ())
        only_d = SchedulingSnapshot((), (ReadyTask("D", "Decode", 1),))
        self.assertEqual(p.decide(only_p).phase, "Prefill")
        self.assertEqual(d.decide(only_d).phase, "Decode")
        with self.assertRaises(ValueError):
            p.decide(only_d)
        with self.assertRaises(ValueError):
            d.decide(only_p)
        with self.assertRaises(ValueError):
            Scheduler(SchedulerConfig(), profile=p.profile, role="Mix")

    def test_alternating_timeline_and_baseline(self) -> None:
        expected = {"worker_alternating": ((2, 4, 10, 16), (20, 22)),
                    "worker_prefill_first": ((2, 16, 18, 20), (14, 16))}
        for name, times in expected.items():
            experiment = load_experiment(ROOT / "configs" / f"{name}.json")
            run = simulate_events(experiment.requests, experiment.timing,
                                  scheduler_profile=experiment.scheduler_profile)
            self.assertEqual(tuple(r.token_times_ms for r in run.batch_run.results), times)
            for plan in run.plans:
                self.assertLessEqual(len(plan.items), 2)
                self.assertLessEqual(sum(task.token_count for task in plan.items), 3)
            chunks = [p.items[0] for p in run.plans if p.phase == "Prefill" and p.request_ids == ("B",)]
            self.assertEqual([(c.range_start, c.token_count) for c in chunks], [(0, 2), (2, 2), (4, 2)])

    def test_profile_configuration_errors(self) -> None:
        original = json.loads((ROOT / "configs/worker_alternating.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.json"
            original["workload"]["path"] = str(ROOT / "workloads/alternating_requests.json")
            mutations = []
            for field, value in (("scheduler_profile", "missing"), ("role", "P"), ("role", "unknown")):
                config = deepcopy(original)
                config["workers"][0][field] = value
                mutations.append(config)
            for field, value in (("mix_algorithm", "unknown"), ("decode_algorithm", None),
                                 ("constraints", {"policy": "decode_first"}),
                                 ("constraints", {"max_tokens_per_batch": 0})):
                config = deepcopy(original)
                config["scheduler_profiles"]["mix_default"][field] = value
                mutations.append(config)
            mutations.extend([{**original, "workers": original["workers"] * 2},
                              {**original, "scheduler": {"policy": "prefill_first"}}])
            for config in mutations:
                path.write_text(json.dumps(config))
                with self.subTest(config=config), self.assertRaises(ValueError):
                    load_experiment(path)

    def test_legacy_config_preserves_execution(self) -> None:
        modern = load_experiment(ROOT / "configs/basic.json")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            path.write_text(json.dumps({"workload": {"path": str(modern.workload_path)},
                "scheduler": {"policy": "prefill_first"},
                "timing": {"prefill_ms_per_input_token": 2, "decode_ms_per_step": 2}}))
            legacy = load_experiment(path)
            old = simulate_events(legacy.requests, Timing(), scheduler_profile=legacy.scheduler_profile)
            new = simulate_events(modern.requests, modern.timing, scheduler_profile=modern.scheduler_profile)
            self.assertEqual(old.batch_run, new.batch_run)


if __name__ == "__main__":
    unittest.main()
