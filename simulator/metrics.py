"""仅统计已完成的运行记录，不改变调度或事件时间。"""

import math
import statistics

from .models import BatchRun


def distribution(values: tuple[float, ...]) -> dict:
    """最近秩分位数：排序后取 ceil(p*n) 项（从 1 开始编号）。"""
    if not values:
        # 没有观测与延迟为零不同，JSON 用 null 表示没有统计值。
        return {"count": 0, "mean": None, "p50": None, "p95": None}
    ordered = sorted(values)
    return {
        "count": len(ordered), "mean": statistics.mean(ordered),
        "p50": ordered[math.ceil(0.50 * len(ordered)) - 1],
        "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
    }


def summarize(run: BatchRun) -> dict:
    # 【重点 1】TTFT 和总延迟按请求统计；每个请求贡献一个样本。
    results = run.results
    # 【重点 2】ITL 按所有相邻 token 间隔汇总，长输出请求贡献更多样本。
    # 只有一个输出 token 的请求不贡献 ITL，不能塞一个 0 来稀释平均值。
    intervals = tuple(gap for result in results for gap in result.inter_token_intervals_ms)
    input_tokens = sum(result.request.input_tokens for result in results)
    output_tokens = sum(len(result.token_times_ms) for result in results)

    # 【重点 3】吞吐窗口从最早请求到达，到最后请求完成，包括空闲和排队。
    start_ms = min((result.request.arrival_ms for result in results), default=None)
    end_ms = max((result.token_times_ms[-1] for result in results), default=None)
    duration_ms = None if start_ms is None else end_ms - start_ms
    seconds = None if duration_ms is None or duration_ms <= 0 else duration_ms / 1000
    return {
        "completed_requests": len(results), "batch_steps": len(run.steps),
        "total_input_tokens": input_tokens, "total_output_tokens": output_tokens,
        "percentile_method": "nearest_rank",
        "itl_weighting": "all_inter_token_intervals",
        "ttft_ms": distribution(tuple(result.ttft_ms for result in results)),
        "latency_ms": distribution(tuple(result.latency_ms for result in results)),
        "itl_ms": distribution(intervals),
        "window": {"definition": "first_arrival_to_last_completion", "start_ms": start_ms,
                   "end_ms": end_ms, "duration_ms": duration_ms},
        "request_throughput_per_second": None if seconds is None else len(results) / seconds,
        "output_throughput_tokens_per_second": None if seconds is None else output_tokens / seconds,
    }


def print_summary(summary: dict) -> None:
    def number(value: float | None) -> str:
        return "无样本" if value is None else f"{value:.3f}"

    print(f"完成 {summary['completed_requests']} 请求，执行 {summary['batch_steps']} 轮；"
          f"输入 {summary['total_input_tokens']}、输出 {summary['total_output_tokens']} token")
    for key, label in (("ttft_ms", "TTFT"), ("latency_ms", "端到端延迟"), ("itl_ms", "ITL（按间隔）")):
        metric = summary[key]
        print(f"{label}: 样本 {metric['count']}，平均 {number(metric['mean'])} ms，"
              f"P50 {number(metric['p50'])} ms，P95 {number(metric['p95'])} ms")
    print(f"统计窗口 {number(summary['window']['duration_ms'])} ms（首请求到达到最后完成）")
    print(f"请求吞吐 {number(summary['request_throughput_per_second'])} req/s；"
          f"输出吞吐 {number(summary['output_throughput_tokens_per_second'])} token/s")
