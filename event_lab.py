"""单卡离散事件实验；先读 EVENT_GUIDE.md，不必一次读懂全部代码。"""

from collections import deque
from dataclasses import dataclass, field
import heapq
from typing import Literal

from batching_lab import BatchRun, BatchStep
from serving_lab import Event, Request, Result, Timing


@dataclass(order=True, frozen=True)
class _ScheduledEvent:
    """未来通知单：只用时间、事件优先级、序号排序，不比较内容。"""

    time_ms: float
    priority: int
    sequence: int
    kind: str = field(compare=False)
    request_ids: tuple[str, ...] = field(compare=False)
    phase: str = field(default="", compare=False)


@dataclass(frozen=True)
class EventRun:
    batch_run: BatchRun
    trace: tuple[str, ...]


def simulate_events(
    requests: tuple[Request, ...], timing: Timing, *,
    policy: Literal["prefill_first", "decode_first"] = "prefill_first",
) -> EventRun:
    """保持原有 batch 规则，只更换时间推进方式。"""
    if policy not in ("prefill_first", "decode_first"):
        raise ValueError("policy 必须是 prefill_first 或 decode_first")
    if len({item.request_id for item in requests}) != len(requests):
        raise ValueError("request_id 必须唯一")
    ordered = sorted(requests, key=lambda item: item.arrival_ms)
    by_id = {item.request_id: item for item in ordered}
    token_times: dict[str, list[float]] = {item.request_id: [] for item in ordered}
    records: dict[str, list[Event]] = {item.request_id: [] for item in ordered}
    starts: dict[str, float] = {}

    # 【重点 1：三个容器各管什么？】
    # future 保存未来事件；waiting 只含已经到达的请求；active 只含待 Decode 请求。
    future: list[_ScheduledEvent] = []
    waiting: deque[str] = deque()
    active: list[str] = []
    steps: list[BatchStep] = []
    trace: list[str] = []
    busy = False
    wakeup_pending = False
    sequence = 0
    now_ms = 0.0

    def schedule(time_ms: float, kind: str, ids: tuple[str, ...] = (), phase: str = "") -> None:
        nonlocal sequence
        # 同时间先登记到达，再结算完成，最后调度；与 P/D 策略优先级不同。
        priority = {"arrival": 0, "complete": 1, "wakeup": 2}[kind]
        heapq.heappush(future, _ScheduledEvent(time_ms, priority, sequence, kind, ids, phase))
        sequence += 1

    def wake_scheduler() -> None:
        nonlocal wakeup_pending
        # 同时间多个到达只需要唤醒一次；忙碌时等完成事件再唤醒。
        if not busy and not wakeup_pending:
            schedule(now_ms, "wakeup")
            wakeup_pending = True

    for request in ordered:
        schedule(request.arrival_ms, "arrival", (request.request_id,))

    # 【重点 2：只有取出事件时才改变模拟时间。】
    while future:
        event = heapq.heappop(future)
        now_ms = event.time_ms
        trace.append(f"{now_ms:g}ms 处理 {event.kind} {list(event.request_ids)}")

        # 【重点 3：到达只是排队，不能打断正在执行的 batch。】
        if event.kind == "arrival":
            request_id = event.request_ids[0]
            waiting.append(request_id)
            records[request_id].append(Event(now_ms, "请求到达"))
            trace.append(f"  waiting={list(waiting)}；GPU {'忙碌，继续等待' if busy else '空闲'}")
            wake_scheduler()

        # 【重点 4：完成通知到来后，才记 token、退出请求并释放 GPU。】
        elif event.kind == "complete":
            for request_id in event.request_ids:
                token_times[request_id].append(now_ms)
                count = len(token_times[request_id])
                records[request_id].append(Event(now_ms, f"{event.phase} 完成，产出 token", count))
                if count == by_id[request_id].output_tokens:
                    records[request_id].append(Event(now_ms, "请求完成"))
                    if request_id in active:
                        active.remove(request_id)
                elif event.phase == "Prefill":
                    active.append(request_id)
            busy = False
            trace.append(f"  完成后 active={active}；GPU 空闲")
            wake_scheduler()

        else:  # wakeup：GPU 空闲时才有资格选择下一轮。
            wakeup_pending = False
            if busy or not (waiting or active):
                continue
            choose_prefill = bool(waiting) and (policy == "prefill_first" or not active)
            if choose_prefill:
                request_id = waiting.popleft()
                members = (request_id,)
                phase = "Prefill"
                duration_ms = by_id[request_id].input_tokens * timing.prefill_ms_per_input_token
                starts[request_id] = now_ms
                records[request_id].append(Event(now_ms, "开始 Prefill"))
            else:
                members = tuple(active)
                phase = "Decode"
                duration_ms = timing.decode_ms_per_step

            # 【重点 5：开始执行只注册未来完成，不把时钟跳到结束。】
            # members 固定本轮成员；busy 阻止其他到达事件启动重叠计算。
            end_ms = now_ms + duration_ms
            busy = True
            steps.append(BatchStep(now_ms, end_ms, phase, members))
            schedule(end_ms, "complete", members, phase)
            trace.append(f"  启动 {phase} {list(members)}；注册 {end_ms:g}ms 完成通知")

    # 辅助封装：只把记录转换为原有结果类型，方便与旧模型对照。
    results = tuple(
        Result(item, tuple(records[item.request_id]), tuple(token_times[item.request_id]), starts[item.request_id])
        for item in ordered
    )
    return EventRun(BatchRun(results, tuple(steps)), tuple(trace))
