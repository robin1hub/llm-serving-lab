"""配置驱动入口：读取 → 执行 → 输出；没有新的调度机制。"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from simulator.config import load_experiment
from simulator.engine import simulate_events
from simulator.metrics import print_summary, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description="单卡教学模拟器（耗时尚未校准）")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--summary", action="store_true", help="只打印摘要，完整轨迹仍写入结果文件")
    parser.add_argument("--output", type=Path, default=Path("output/result.json"),
                        help="结果文件，默认 output/result.json；相对当前工作目录")
    args = parser.parse_args()
    try:
        # 【重点 1】文件转为请求对象；事件内核继续使用原来的函数。
        experiment = load_experiment(args.config)
        run = simulate_events(experiment.requests, experiment.timing, policy=experiment.policy,
                              max_batch_requests=experiment.max_batch_requests)
        summary = summarize(run.batch_run)
        # 【重点 2】结果保留实际输入和参数，便于之后核对和复现。
        report = {
            "schema_version": 1,
            "timing_source": "teaching_assumption",
            "config_path": str(experiment.config_path),
            "workload_path": str(experiment.workload_path),
            "workload_metadata": experiment.workload_metadata,
            "policy": experiment.policy,
            "scheduler": {"policy": experiment.policy, "max_batch_requests": experiment.max_batch_requests,
                          "decode_order": "round_robin_after_completion"},
            "timing": asdict(experiment.timing),
            "summary": summary,
            "requests": [
                {"request": asdict(result.request), "start_ms": result.start_ms,
                 "waiting_ms": result.waiting_ms, "token_times_ms": result.token_times_ms,
                 "ttft_ms": result.ttft_ms, "itl_ms": result.inter_token_intervals_ms,
                 "latency_ms": result.latency_ms, "events": [asdict(e) for e in result.events]}
                for result in run.batch_run.results
            ],
            "steps": [asdict(step) for step in run.batch_run.steps],
            "plans": [asdict(plan) for plan in run.plans],
            "trace": run.trace,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
                               + "\n", encoding="utf-8")
    except (ValueError, OSError) as error:
        parser.exit(2, f"错误：{error}\n")
    for step in (() if args.summary else run.batch_run.steps):
        print(f"{step.start_ms:g}–{step.end_ms:g}ms {step.phase} {list(step.request_ids)}")
    for result in (() if args.summary else run.batch_run.results):
        print(f"{result.request.request_id}: TTFT {result.ttft_ms:g}ms，"
              f"ITL {result.inter_token_intervals_ms}ms，总延迟 {result.latency_ms:g}ms")
    print_summary(summary)
    print(f"结果已保存：{args.output.resolve()}（教学耗时，未校准）")


if __name__ == "__main__":
    main()
