"""打印事件处理顺序，并核对旧版的 GPU 轮次与 token 时间。"""

from batching_lab import simulate_batched
from event_lab import simulate_events
from serving_lab import Request, Timing


def main() -> None:
    requests = (Request("A", 0, 4, 3), Request("B", 3, 2, 2))
    timing = Timing()
    run = simulate_events(requests, timing)
    print("单卡事件实验：Prefill 优先，教学耗时；没有真实 GPU 计算。")
    for line in run.trace:
        print(line)
    print("\nGPU 执行区间：")
    for step in run.batch_run.steps:
        print(f"{step.start_ms:g}–{step.end_ms:g}ms {step.phase} {list(step.request_ids)}")
    for result in run.batch_run.results:
        print(f"{result.request.request_id}: token {result.token_times_ms}ms，TTFT {result.ttft_ms:g}ms")
    previous = simulate_batched(requests, timing)
    matches = run.batch_run.steps == previous.steps and all(
        new.token_times_ms == old.token_times_ms
        for new, old in zip(run.batch_run.results, previous.results)
    )
    print(f"与旧版执行区间、token 时间一致：{matches}")


if __name__ == "__main__":
    main()
