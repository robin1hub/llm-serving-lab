"""第四步：同一组请求，只改变 Prefill / Decode 的优先级。"""

from batching_lab import simulate_batched
from serving_lab import Request, Timing


def main() -> None:
    requests = (Request("A", 0, 4, 3), Request("B", 3, 2, 2))
    timing = Timing()
    print("教学耗时：Prefill 每输入 token 2ms；Decode 整轮 2ms。")
    # 两次实验输入完全相同，差别仅是轮次边界选哪个阶段。
    for policy in ("prefill_first", "decode_first"):
        label = "Prefill 优先" if policy == "prefill_first" else "严格 Decode 优先"
        run = simulate_batched(requests, timing, policy=policy)
        print(f"\n{label}：")
        for step in run.steps:
            print(f"  {step.start_ms:g}–{step.end_ms:g}ms {step.phase} {list(step.request_ids)}")
        for result in run.results:
            print(
                f"  {result.request.request_id}: token 时间 {result.token_times_ms}ms；"
                f"TTFT {result.ttft_ms:g}ms；ITL {result.inter_token_intervals_ms}ms；"
                f"总延迟 {result.latency_ms:g}ms"
            )
    print("\n观察 8ms 的选择：先 Prefill B 会延后 A 的下一 token；先 Decode A 会延后 B 的首 token。")
    print("当前严格 Decode 优先会排空 active 再接纳新请求，因此本例退化为串行，没有多请求 Decode batch。")


if __name__ == "__main__":
    main()
