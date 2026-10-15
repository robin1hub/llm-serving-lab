"""配置驱动入口：读取 → 执行 → 输出；没有新的调度机制。"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from simulator.config import Experiment, load_experiment
from simulator.engine import ClusterRun, EventRun, simulate_events, simulate_workers, simulate_pd
from simulator.metrics import print_summary, summarize
from simulator.predictor import build_predictor


def single_report(experiment: Experiment, run: EventRun) -> dict:
    summary = summarize(run.batch_run)
    return {
        "schema_version": 1,
        "status": run.status,
        "summary_scope": "completed_requests_only" if run.status == "blocked" else "all_requests",
        "pending_request_ids": run.pending_request_ids,
        "pending_progress": [asdict(progress) for progress in run.pending_progress],
        "kv_cache": asdict(experiment.kv_config) if experiment.kv_config is not None else None,
        "memory_budget": asdict(experiment.memory_budget) if experiment.memory_budget is not None else None,
        "memory_provenance": experiment.memory_provenance,
        "recovery": asdict(experiment.recovery_config),
        "recovery_records": [asdict(record) for record in run.recovery_records],
        "recovery_pending": [asdict(target) for target in run.recovery_pending],
        "recovery_summary": {
            "preemptions": sum(record.action == "preempt" for record in run.recovery_records),
            "recomputed_tokens": sum(item.token_count for plan in run.plans for item in plan.items
                                      if item.phase == "Recompute")},
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


def cluster_report(experiment: Experiment, run: ClusterRun) -> dict:
    """【输出重点】集群只汇总请求和时间线；不同卡的KV快照保持分开。"""
    workers = []
    requests = []
    pending_progress = []
    for worker_run in run.workers:
        config = worker_run.config
        limits = config.scheduler_profile.constraints
        local = replace(experiment, timing=config.timing, policy=config.scheduler_profile.mix_algorithm,
                        max_batch_requests=limits.max_batch_requests, max_tokens_per_batch=limits.max_tokens_per_batch,
                        prefill_chunk_tokens=limits.prefill_chunk_tokens, scheduler_profile=config.scheduler_profile,
                        profile_name=config.profile_name, worker_id=config.worker_id, worker_role=config.role,
                        predictor_settings=config.predictor_settings, kv_config=config.kv_config,
                        memory_budget=config.memory_budget, memory_provenance=config.memory_provenance,
                        recovery_config=config.recovery_config)
        report = single_report(local, worker_run.run)
        requests.extend({**row, "worker_id": config.worker_id, "gpu_id": config.gpu_id} for row in report["requests"])
        pending_progress.extend({**row, "worker_id": config.worker_id} for row in report["pending_progress"])
        # 全局请求/执行轮另存一次；每卡保留资源、状态、KV、恢复和局部统计。
        for key in ("requests", "steps", "plans", "estimates", "trace", "config_path", "workload_path", "workload_metadata"):
            report.pop(key)
        report["worker"]["gpu_id"] = config.gpu_id
        report["busy_time_ms"] = sum(step.end_ms - step.start_ms for step in worker_run.run.batch_run.steps)
        if run.pd_config is not None:
            report["produced_output_tokens"] = sum(item.produces_token for estimate in worker_run.run.estimates
                                                   for item in estimate.items)
        workers.append(report)
    by_id = {row["request"]["request_id"]: row for row in requests}
    report = {
        "schema_version": 2, "execution_mode": "fixed_1p1d" if run.pd_config else "independent_mix_workers", "status": run.status,
        "summary_scope": "completed_requests_only" if run.status == "blocked" else "all_requests",
        "config_path": str(experiment.config_path), "workload_path": str(experiment.workload_path),
        "workload_metadata": experiment.workload_metadata,
        "routing": {"policy": "fixed", "assignments": dict(run.assignments)},
        "workers": workers, "timing_source": "per_worker",
        "pending_request_ids": run.pending_request_ids, "pending_progress": pending_progress,
        "summary": summarize(run.batch_run),
        "requests": [by_id[result.request.request_id] for result in run.batch_run.results],
        "steps": [asdict(step) for step in run.batch_run.steps],
        "plans": [asdict(plan) for plan in run.plans],
        "estimates": [asdict(estimate) for estimate in run.estimates], "trace": run.trace,
    }
    if run.pd_config is not None:
        report["pd"] = asdict(run.pd_config)
        report["routing"]["policy"] = "fixed_1p1d"
        report["transfers"] = [asdict(record) for record in run.transfers]
        for row in report["requests"]:
            row["worker_path"] = ([run.pd_config.prefill_worker, run.pd_config.decode_worker]
                                  if row["worker_id"] == run.pd_config.decode_worker
                                  else [run.pd_config.prefill_worker])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM serving教学事件模拟器")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--summary", action="store_true", help="只打印摘要，完整轨迹仍写入结果文件")
    parser.add_argument("--output", type=Path, default=Path("output/result.json"),
                        help="结果文件，默认 output/result.json；相对当前工作目录")
    args = parser.parse_args()
    try:
        # 【重点 1】文件转为请求对象；事件内核继续使用原来的函数。
        experiment = load_experiment(args.config)
        if experiment.pd_config is not None:
            run = simulate_pd(experiment.requests, experiment.workers, experiment.pd_config)
        elif len(experiment.workers) > 1:
            run = simulate_workers(experiment.requests, experiment.workers, dict(experiment.assignments))
        else:
            run = simulate_events(experiment.requests, experiment.timing, policy=experiment.policy,
                                  max_batch_requests=experiment.max_batch_requests,
                                  max_tokens_per_batch=experiment.max_tokens_per_batch,
                                  prefill_chunk_tokens=experiment.prefill_chunk_tokens,
                                  scheduler_profile=experiment.scheduler_profile,
                                  predictor=build_predictor(experiment.predictor_settings, experiment.timing),
                                  kv_config=experiment.kv_config, recovery_config=experiment.recovery_config)
        summary = summarize(run.batch_run)
        # 【重点 2】结果保留实际输入和参数，便于之后核对和复现。
        report = cluster_report(experiment, run) if isinstance(run, ClusterRun) else single_report(experiment, run)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
                               + "\n", encoding="utf-8")
    except (ValueError, OSError) as error:
        parser.exit(2, f"错误：{error}\n")
    for step in (() if args.summary else run.batch_run.steps):
        owner = f"[{step.worker_id}/{step.gpu_id}] " if isinstance(run, ClusterRun) else ""
        print(f"{owner}{step.start_ms:g}–{step.end_ms:g}ms {step.phase} {list(step.request_ids)}")
    if isinstance(run, ClusterRun) and not args.summary:
        for transfer in run.transfers:
            print(f"[KV {transfer.source_worker}→{transfer.target_worker}] "
                  f"{transfer.start_ms:g}–{transfer.end_ms:g}ms {transfer.request_id}，"
                  f"{transfer.payload_bytes}字节，链路等待{transfer.start_ms - transfer.ready_ms:g}ms")
    if not args.summary:
        print("计划工作量（本轮处理 token 数；Prefill/Recompute range_start 是序列偏移）：")
        for index, plan in enumerate(run.plans):
            print(f"  第 {index + 1} 轮：{[(item.request_id, item.phase, item.range_start, item.token_count) for item in plan.items]}")
    for result in (() if args.summary else run.batch_run.results):
        print(f"{result.request.request_id}: TTFT {result.ttft_ms:g}ms，"
              f"ITL {result.inter_token_intervals_ms}ms，总延迟 {result.latency_ms:g}ms")
    print_summary(summary)
    source = "各worker耗时来源分别记录；多卡请求级结果未校准" if isinstance(run, ClusterRun) else f"{report['timing_source']}，未校准"
    print(f"结果已保存：{args.output.resolve()}（{source}）")
    if run.status == "blocked":
        parser.exit(2, f"KV容量阻塞，未完成请求：{list(run.pending_request_ids)}；结果已保存，summary仅统计完成请求。\n")


if __name__ == "__main__":
    main()
