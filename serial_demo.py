"""第二步：两个请求共享 GPU，先观察串行执行造成的排队。"""

from serving_lab import Request, Timing, simulate_serial


def main() -> None:
    requests = (
        Request("A", arrival_ms=0.0, input_tokens=4, output_tokens=3),
        Request("B", arrival_ms=3.0, input_tokens=2, output_tokens=2),
    )
    results = simulate_serial(requests, Timing())
    print("串行实验：按到达顺序，前一请求完成后再执行下一请求")
    # 各请求记录合并后按时间打印；同一时刻保持原有记录顺序。
    timeline = [(event, result.request.request_id) for result in results for event in result.events]
    for event, request_id in sorted(timeline, key=lambda item: item[0].time_ms):
        token = "" if event.token_index is None else f" #{event.token_index}"
        print(f"{event.time_ms:5g} ms | {request_id} | {event.description}{token}")
    for result in results:
        print(
            f"{result.request.request_id}: 等待 {result.waiting_ms:g} ms，"
            f"TTFT {result.ttft_ms:g} ms，端到端 {result.latency_ms:g} ms"
        )


if __name__ == "__main__":
    main()
