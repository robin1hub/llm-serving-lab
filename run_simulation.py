"""配置驱动入口：读取 → 执行 → 输出；没有新的调度机制。"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from simulator.config import load_experiment
from simulator.engine import simulate_events
from simulator.metrics import print_summary, summarize
from simulator.predictor import build_predictor


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
                              max_batch_requests=experiment.max_batch_requests,
                              max_tokens_per_batch=experiment.max_tokens_per_batch,
                              prefill_chunk_tokens=experiment.prefill_chunk_tokens,
                              scheduler_profile=experiment.scheduler_profile,
                              predictor=build_predictor(experiment.predictor_settings, experiment.timing),
                              kv_config=experiment.kv_config)
        summary = summarize(run.batch_run)
        # 【重点 2】结果保留实际输入和参数，便于之后核对和复现。
        report = {
            "schema_version": 1,
            "status": run.status,
            "summary_scope": "completed_requests_only" if run.status == "blocked" else "all_requests",
            "pending_request_ids": run.pending_request_ids,
            "pending_progress": [asdict(progress) for progress in run.pending_progress],
            "kv_cache": asdict(experiment.kv_config) if experiment.kv_config is not None else None,
            "kv_final": asdict(run.kv_final) if run.kv_final is not None else None,
            "kv_records": [asdict(record) for record in run.kv_records],
            "timing_source": ("teaching_assumption" if experiment.predictor_settings.kind == "teaching"
                              else "uncalibrated_roofline"),
            "predictor": asdict(experiment.predictor_settings),
            "config_path": str(experiment.config_path),
            "workload_path": str(experiment.workload_path),
            "workload_metadata": experiment.workload_metadata,
            "policy": experiment.policy,
            "worker": {"id": experiment.worker_id, "role": experiment.worker_role,
                       "scheduler_profile": experiment.profile_name},
            # 只保存用户配置的约束，不把兼容旧接口的 policy 放入 constraints。
            "resolved_scheduler_profile": {
                "prefill_algorithm": experiment.scheduler_profile.prefill_algorithm,
                "decode_algorithm": experiment.scheduler_profile.decode_algorithm,
                "mix_algorithm": experiment.scheduler_profile.mix_algorithm,
                "constraints": {"max_batch_requests": experiment.max_batch_requests,
                                "max_tokens_per_batch": experiment.max_tokens_per_batch,
                                "prefill_chunk_tokens": experiment.prefill_chunk_tokens}},
            "scheduler": {"policy": experiment.policy, "max_batch_requests": experiment.max_batch_requests,
                          "max_tokens_per_batch": experiment.max_tokens_per_batch,
                          "prefill_chunk_tokens": experiment.prefill_chunk_tokens,
                          "decode_order": "round_robin_after_completion"},
            "timing": asdict(experiment.timing) if experiment.predictor_settings.kind == "teaching" else None,
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
            "estimates": [asdict(estimate) for estimate in run.estimates],
            "trace": run.trace,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
                               + "\n", encoding="utf-8")
    except (ValueError, OSError) as error:
        parser.exit(2, f"错误：{error}\n")
    for step in (() if args.summary else run.batch_run.steps):
        print(f"{step.start_ms:g}–{step.end_ms:g}ms {step.phase} {list(step.request_ids)}")
    if not args.summary:
        print("计划工作量（新 token 数；Prefill range_start 是输入偏移）：")
        for index, plan in enumerate(run.plans):
            print(f"  第 {index + 1} 轮：{[(item.request_id, item.phase, item.range_start, item.token_count) for item in plan.items]}")
    for result in (() if args.summary else run.batch_run.results):
        print(f"{result.request.request_id}: TTFT {result.ttft_ms:g}ms，"
              f"ITL {result.inter_token_intervals_ms}ms，总延迟 {result.latency_ms:g}ms")
    print_summary(summary)
    print(f"结果已保存：{args.output.resolve()}（{report['timing_source']}，未校准）")
    if run.status == "blocked":
        parser.exit(2, f"KV容量阻塞，未完成请求：{list(run.pending_request_ids)}；结果已保存，summary仅统计完成请求。\n")


if __name__ == "__main__":
    main()
