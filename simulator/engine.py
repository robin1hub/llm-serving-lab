"""单卡离散事件实验；先读 docs/chapters/03-events.md，不必一次读懂全部代码。"""

from collections import deque
from dataclasses import dataclass, field
import heapq

from .models import BatchRun, BatchStep
from .models import Event, Request, Result, Timing, _validate_time
from .scheduler import BatchPlan, Policy, ReadyTask, Scheduler, SchedulerConfig, SchedulerProfile, SchedulingSnapshot, rotate_after_completion
from .predictor import BatchEstimate, Predictor, PredictorItem, TeachingPredictor
from .kv_cache import BlockPool, KVConfig, KVRecord, KVReservation, KVSnapshot


@dataclass(order=True, frozen=True)
class _ScheduledEvent:
    """未来通知单：只用时间、事件优先级、序号排序，不比较内容。"""

    time_ms: float
    priority: int
    sequence: int
    kind: str = field(compare=False)
    request_ids: tuple[str, ...] = field(compare=False)
    phase: str = field(default="", compare=False)
    plan: BatchPlan | None = field(default=None, compare=False)
    reservation: KVReservation | None = field(default=None, compare=False)


@dataclass(frozen=True)
class PendingProgress:
    request_id: str
    prefill_done: int
    token_times_ms: tuple[float, ...]
    start_ms: float | None


@dataclass(frozen=True)
class EventRun:
    batch_run: BatchRun
    trace: tuple[str, ...]
    plans: tuple[BatchPlan, ...]
    estimates: tuple[BatchEstimate, ...] = ()
    kv_records: tuple[KVRecord, ...] = ()
    kv_final: KVSnapshot | None = None
    status: str = "completed"
    pending_request_ids: tuple[str, ...] = ()
    pending_progress: tuple[PendingProgress, ...] = ()


def simulate_events(
    requests: tuple[Request, ...], timing: Timing, *,
    policy: Policy = "prefill_first",
    max_batch_requests: int | None = None,
    max_tokens_per_batch: int | None = None,
    prefill_chunk_tokens: int | None = None,
    scheduler_profile: SchedulerProfile | None = None,
    predictor: Predictor | None = None,
    kv_config: KVConfig | None = None,
) -> EventRun:
    """单卡事件执行；Scheduler 只提计划，内核负责提交和完成。"""
    scheduler = Scheduler(SchedulerConfig(policy, max_batch_requests, max_tokens_per_batch,
                                          prefill_chunk_tokens), profile=scheduler_profile)
    duration_predictor = predictor if predictor is not None else TeachingPredictor(timing)
    kv_pool = BlockPool(kv_config) if kv_config is not None else None
    kv_records: list[KVRecord] = []

    def record_kv(time_ms: float, action: str) -> None:
        if kv_pool is not None:
            kv_records.append(KVRecord(time_ms, action, kv_pool.snapshot()))

    record_kv(0, "initial")
    if len({item.request_id for item in requests}) != len(requests):
        raise ValueError("request_id 必须唯一")
    ordered = sorted(requests, key=lambda item: item.arrival_ms)
    by_id = {item.request_id: item for item in ordered}
    token_times: dict[str, list[float]] = {item.request_id: [] for item in ordered}
    records: dict[str, list[Event]] = {item.request_id: [] for item in ordered}
    starts: dict[str, float] = {}
    prefill_done = {item.request_id: 0 for item in ordered}

    # 【重点 1：三个容器各管什么？】
    # future 保存未来事件；waiting 只含已经到达的请求；active 只含待 Decode 请求。
    future: list[_ScheduledEvent] = []
    waiting: deque[str] = deque()
    active: list[str] = []
    steps: list[BatchStep] = []
    plans: list[BatchPlan] = []
    estimates: list[BatchEstimate] = []
    trace: list[str] = []
    busy = False
    wakeup_pending = False
    sequence = 0
    now_ms = 0.0

    def schedule(time_ms: float, kind: str, ids: tuple[str, ...] = (), phase: str = "",
                 plan: BatchPlan | None = None, reservation: KVReservation | None = None) -> None:
        nonlocal sequence
        # 同时间先登记到达，再结算完成，最后调度；与 P/D 策略优先级不同。
        priority = {"arrival": 0, "complete": 1, "wakeup": 2}[kind]
        heapq.heappush(future, _ScheduledEvent(time_ms, priority, sequence, kind, ids, phase, plan, reservation))
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
            assert event.plan is not None  # 完成的是启动时固定的计划，不重新选成员。
            if kv_pool is not None:
                assert event.reservation is not None
                kv_pool.complete(event.reservation)
                record_kv(now_ms, "ready")
            for item in event.plan.items:
                request_id = item.request_id
                if item.phase == "Prefill":
                    # 【分块重点 1】完成事件才累加已处理输入；中间块不产生输出。
                    assert item.range_start == prefill_done[request_id]
                    prefill_done[request_id] += item.token_count
                    assert prefill_done[request_id] <= by_id[request_id].input_tokens
                    if prefill_done[request_id] < by_id[request_id].input_tokens:
                        records[request_id].append(Event(now_ms, f"Prefill 分块完成，输入进度 "
                            f"{prefill_done[request_id]}/{by_id[request_id].input_tokens}，尚无首 token"))
                        # 当前 FCFS 继续优先服务队首分块；不隐式改变阶段策略。
                        waiting.appendleft(request_id)
                        continue
                token_times[request_id].append(now_ms)
                count = len(token_times[request_id])
                records[request_id].append(Event(now_ms, f"{item.phase} 完成，产出 token", count))
                if count == by_id[request_id].output_tokens:
                    records[request_id].append(Event(now_ms, "请求完成"))
                    if request_id in active:
                        active.remove(request_id)
                    if kv_pool is not None:
                        kv_pool.release_request(request_id)
                        record_kv(now_ms, f"release:{request_id}")
                elif item.phase == "Prefill":
                    active.append(request_id)
            # 【混合完成重点】只轮转本轮实际Decode成员，新进入active的Prefill不算已服务。
            served_decode = tuple(item.request_id for item in event.plan.items if item.phase == "Decode")
            if served_decode:
                active = list(rotate_after_completion(tuple(active), served_decode))
            # 【交替重点】进度提交完成后才通知策略；查询计划本身不能切换轮次。
            scheduler.on_batch_completed(event.plan)
            busy = False
            trace.append(f"  完成后 active={active}；GPU 空闲")
            wake_scheduler()

        else:  # wakeup：GPU 空闲时才有资格选择下一轮。
            wakeup_pending = False
            if busy or not (waiting or active):
                continue
            # 【选批重点：只读快照 → 候选计划】快照不暴露真实最终输出长度。
            snapshot = SchedulingSnapshot(
                tuple(ReadyTask(item, "Prefill", by_id[item].input_tokens - prefill_done[item],
                                prefill_done[item]) for item in waiting),
                tuple(ReadyTask(item, "Decode", 1) for item in active),
            )
            plan = scheduler.decide(snapshot)
            if plan is None:
                continue
            # 【KV重点1】选批只提候选，容量模块可缩块/删成员；未准入不修改队列。
            if kv_pool is not None:
                plan = kv_pool.admit(plan)
                if plan is None:
                    # 队首Prefill放不下时，尝试可增长的Decode释放空间；不抢占任何请求。
                    ready_decode = tuple(item for item in snapshot.ready_decode
                        if kv_pool.additional_blocks(item.request_id, item.token_count) <= kv_pool.free_blocks)
                    fallback = scheduler.decide(SchedulingSnapshot((), ready_decode))
                    plan = kv_pool.admit(fallback) if fallback is not None else None
                    if plan is not None:
                        plan = BatchPlan(plan.items, plan.reason + "；KV阻塞时回退Decode")
                if plan is None:
                    trace.append("  KV容量阻塞：无可准入计划；等待未来到达，否则结束为blocked")
                    record_kv(now_ms, "blocked")
                    continue
            members = plan.request_ids
            phase = plan.phase
            trace.append(f"  候选 Prefill={list(waiting)}，Decode={active}；"
                         f"选择 {phase} {list(members)}；{plan.reason}")
            trace.append(f"  工作量={[(item.request_id, item.token_count) for item in plan.items]}；"
                         f"本轮 token 总数={sum(item.token_count for item in plan.items)}")
            # 【预测重点1】只提供本轮工作与已知上下文，不提供未来输出长度。
            # 首token是最终Prefill产生的，尚未再次输入模型：Decode context因此减1。
            predictor_items = tuple(PredictorItem(
                item.request_id, item.phase, item.token_count,
                item.range_start if item.phase == "Prefill" else
                by_id[item.request_id].input_tokens + len(token_times[item.request_id]) - 1,
                item.phase == "Decode" or
                item.range_start + item.token_count == by_id[item.request_id].input_tokens,
            ) for item in plan.items)
            estimate = duration_predictor.estimate(predictor_items)
            duration_ms = estimate.duration_ms
            # 拒绝非法预测与无法推进浮点时钟的时间，避免无进展循环。
            _validate_time("预测耗时", duration_ms, allow_zero=False)
            end_ms = now_ms + duration_ms
            _validate_time("完成时间", end_ms, allow_zero=False)
            if end_ms <= now_ms:
                raise ValueError("预测耗时过小，无法推进模拟时钟")
            # 【KV重点2】预测成功后原子预留，在事件完成前不提交有效KV范围。
            reservation = None
            if kv_pool is not None:
                reservation = kv_pool.reserve(plan)
                record_kv(now_ms, "reserved")
                try:
                    kv_pool.begin_populating(reservation)
                except Exception:
                    kv_pool.rollback(reservation)
                    raise
                record_kv(now_ms, "populating")
            # 【提交重点】准入/预测/预留成功后，才取走请求。
            # 【混合提交重点】整批标签可能是Mixed，按成员阶段取走Prefill并记录首次开始。
            for item in plan.items:
                if item.phase != "Prefill":
                    continue
                request_id = waiting.popleft()
                assert request_id == item.request_id
                # 【分块重点 2】等待时间以第一块开始为准，不被后续块覆盖。
                if request_id not in starts:
                    starts[request_id] = now_ms
                    records[request_id].append(Event(now_ms, "开始 Prefill"))
                else:
                    records[request_id].append(Event(now_ms, "继续 Prefill 分块"))

            # 【重点 5：开始执行只注册未来完成，不把时钟跳到结束。】
            # members 固定本轮成员；busy 阻止其他到达事件启动重叠计算。
            busy = True
            steps.append(BatchStep(now_ms, end_ms, phase, members))
            plans.append(plan)
            estimates.append(estimate)
            schedule(end_ms, "complete", members, phase, plan, reservation)
            trace.append(f"  启动 {phase} {list(members)}；注册 {end_ms:g}ms 完成通知")

    # 辅助封装：只把记录转换为原有结果类型，方便与旧模型对照。
    completed = tuple(item for item in ordered if len(token_times[item.request_id]) == item.output_tokens)
    pending = tuple(item.request_id for item in ordered if len(token_times[item.request_id]) < item.output_tokens)
    # 容量满时保留已完成结果与块表，不假装所有请求完成，也不无限安排唤醒事件。
    results = tuple(
        Result(item, tuple(records[item.request_id]), tuple(token_times[item.request_id]), starts[item.request_id])
        for item in completed
    )
    return EventRun(BatchRun(results, tuple(steps)), tuple(trace), tuple(plans), tuple(estimates),
                    tuple(kv_records), kv_pool.snapshot() if kv_pool is not None else None,
                    "blocked" if pending else "completed", pending,
                    tuple(PendingProgress(item, prefill_done[item], tuple(token_times[item]), starts.get(item))
                          for item in pending))
