"""同一份 workload 对比串行与 Decode batching，打印每轮成员。"""

from batching_lab import simulate_batched
from serving_lab import Request, Timing, simulate_serial


def main() -> None:
    requests = (
        Request("A", 0.0, 4, 3),
        Request("B", 3.0, 2, 2),
    )
    timing = Timing()
    run = simulate_batched(requests, timing)
    print("教学规则：Prefill 优先；Decode 整轮固定 2ms，不随 batch 大小变化")
    print("GPU 执行轮次：")
    for step in run.steps:
        print(f"{step.start_ms:g}–{step.end_ms:g} ms | {step.phase} batch {list(step.request_ids)}")

    print("请求事件（相同时间保持原记录顺序）：")
    timeline = [(event, result.request.request_id) for result in run.results for event in result.events]
    for event, request_id in sorted(timeline, key=lambda item: item[0].time_ms):
        token = "" if event.token_index is None else f" #{event.token_index}"
        print(f"{event.time_ms:5g} ms | {request_id} | {event.description}{token}")

    # 对照实验保持请求和耗时参数一致，只改变执行规则。
    for label, results in (("串行", simulate_serial(requests, timing)), ("Batching", run.results)):
        print(f"{label} 指标：")
        for result in results:
            print(
                f"  {result.request.request_id}: TTFT {result.ttft_ms:g} ms，"
                f"ITL {result.inter_token_intervals_ms} ms，总延迟 {result.latency_ms:g} ms"
            )


if __name__ == "__main__":
    main()
