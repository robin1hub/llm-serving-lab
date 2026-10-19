"""共享事件队列与每卡独立运行状态；先读 docs/chapters/03-events.md，不必一次读懂全部代码。"""

from collections import deque
from dataclasses import dataclass, field
import heapq
from typing import Callable

from .worker import WorkerConfig
from .handoff import CapacityWaitRecord, HandoffController, PDConfig, TransferRecord
from .models import BatchRun, BatchStep
from .models import Event, Request, Result, Timing, _validate_time
from .scheduler import BatchPlan, Policy, ReadyTask, Scheduler, SchedulerConfig, SchedulerProfile, SchedulingSnapshot, rotate_after_completion
from .predictor import BatchEstimate, Predictor, PredictorItem, build_predictor
from .kv_cache import BlockPool, KVConfig, KVRecord, KVReservation, KVSnapshot
from .recovery import RecoveryCandidate, RecoveryConfig, RecoveryController, RecoveryRecord, RecoveryTarget, select_victim


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
    worker_id: str = field(default="worker_0", compare=False)


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
    recovery_records: tuple[RecoveryRecord, ...] = ()
    recovery_pending: tuple[RecoveryTarget, ...] = ()


@dataclass(frozen=True)
class WorkerStep(BatchStep):
    """多卡执行记录带身份；旧BatchStep和单卡输出保持不变。"""

    worker_id: str
    gpu_id: str


@dataclass(frozen=True)
class WorkerRun:
    config: WorkerConfig
    run: EventRun


@dataclass(frozen=True)
class ClusterRun:
    batch_run: BatchRun
    workers: tuple[WorkerRun, ...]
    plans: tuple[BatchPlan, ...]
    estimates: tuple[BatchEstimate, ...]
    trace: tuple[str, ...]
    assignments: tuple[tuple[str, str], ...]
    status: str
    pending_request_ids: tuple[str, ...]
    pd_config: PDConfig | None = None
    transfers: tuple[TransferRecord, ...] = ()
    capacity_waits: tuple[CapacityWaitRecord, ...] = ()


# schedule回调只负责把某张卡的未来通知放进全局堆，不推进时间。
Schedule = Callable[[str, float, str, tuple[str, ...], str, BatchPlan | None, KVReservation | None], None]


class _WorkerRuntime:
    """【多卡重点2】动态状态归一张卡所有，其他卡完成时不能清除本卡busy。"""

    def __init__(self, config: WorkerConfig, requests: tuple[Request, ...],
                 schedule_event: Schedule, predictor: Predictor | None = None) -> None:
        self.config = config
        self._schedule_event = schedule_event
        self.scheduler = Scheduler(config.scheduler_profile.constraints, profile=config.scheduler_profile, role=config.role)
        self.duration_predictor = (predictor if predictor is not None
                                   else build_predictor(config.predictor_settings, config.timing))
        self.kv_pool = BlockPool(config.kv_config) if config.kv_config is not None else None
        self.recovery = (RecoveryController(config.recovery_config)
                         if config.recovery_config.capacity_policy == "preempt" else None)
        self.kv_records: list[KVRecord] = []
        self.ordered = sorted(requests, key=lambda item: item.arrival_ms)
        self.by_id = {item.request_id: item for item in self.ordered}
        self.arrival_order = {item.request_id: index for index, item in enumerate(self.ordered)}
        self.token_times: dict[str, list[float]] = {item.request_id: [] for item in self.ordered}
        self.records: dict[str, list[Event]] = {item.request_id: [] for item in self.ordered}
        self.starts: dict[str, float] = {}
        self.prefill_done = {item.request_id: 0 for item in self.ordered}
        self.waiting: deque[str] = deque()
        self.active: list[str] = []
        self.steps: list[BatchStep] = []
        self.plans: list[BatchPlan] = []
        self.estimates: list[BatchEstimate] = []
        self.trace: list[str] = []
        self.busy = False
        self.departed: set[str] = set()
        self.on_prefill_ready: Callable[[str, float], None] | None = None
        self.wakeup_pending = False
        # 这是当前事件的时间副本；唯一时间来源仍是全局heap弹出的事件。
        self.now_ms = 0.0
        self.record_kv(0, "initial")

    def schedule(self, time_ms: float, kind: str, ids: tuple[str, ...] = (), phase: str = "",
                 plan: BatchPlan | None = None, reservation: KVReservation | None = None) -> None:
        self._schedule_event(self.config.worker_id, time_ms, kind, ids, phase, plan, reservation)

    def record_kv(self, time_ms: float, action: str) -> None:
        if self.kv_pool is not None:
            self.kv_records.append(KVRecord(time_ms, action, self.kv_pool.snapshot()))

    def wake_scheduler(self) -> None:
        # 同时间多个到达只需要唤醒一次；忙碌时等完成事件再唤醒。
        if not self.busy and not self.wakeup_pending:
            self.schedule(self.now_ms, "wakeup")
            self.wakeup_pending = True

    def try_preempt(self) -> bool:
        # 【恢复重点1】只在没有准入计划且GPU空闲时选择受害者；不读未来输出目标。
        if self.recovery is None or self.kv_pool is None:
            return False
        memory = self.kv_pool.snapshot()
        blocks = {block.block_id: block for block in memory.blocks}
        tables = {table.request_id: table for table in memory.tables}
        beneficiaries = (self.recovery.focus,) if self.recovery.focus else tuple(self.active) + tuple(self.waiting)
        for beneficiary in beneficiaries:
            needed = self.kv_pool.additional_blocks(beneficiary, 1)
            candidates = []
            for request_id in tuple(self.active) + tuple(self.waiting):
                if request_id == beneficiary or request_id not in tables:
                    continue
                owned = [blocks[block_id] for block_id in tables[request_id].block_ids]
                if any(block.execution_owner is not None for block in owned):
                    continue
                releasable = sum(block.request_refs == (request_id,) for block in owned)
                # 不仅检查“有KV”，还要确保释放后受益请求能至少前进一步。
                if releasable > 0 and self.kv_pool.free_blocks + releasable >= needed:
                    candidates.append(RecoveryCandidate(request_id, len(self.token_times[request_id]),
                                                         self.arrival_order[request_id]))
            victim = select_victim(self.recovery.config, tuple(candidates))
            if victim is None:
                continue
            cached = self.kv_pool.cached_tokens(victim)
            self.kv_pool.release_request(victim)
            if victim in self.active:
                self.active.remove(victim)
            if victim in self.waiting:
                self.waiting.remove(victim)
            self.recovery.suspend(victim, cached, beneficiary, self.now_ms)
            self.records[victim].append(Event(self.now_ms, f"抢占，释放{cached}个token的KV，等待重算"))
            self.trace.append(f"  抢占{victim}；保护{beneficiary}继续执行；恢复目标KV={cached}")
            self.record_kv(self.now_ms, f"preempt:{victim}")
            return True
        return False

    def handle(self, event: _ScheduledEvent) -> None:
        self.now_ms = event.time_ms
        self.trace.append(f"{self.now_ms:g}ms 处理 {event.kind} {list(event.request_ids)}")
        if event.kind == "arrival":
            self.arrive(event)
        elif event.kind == "complete":
            self.complete(event)
        else:
            self.dispatch()

    def arrive(self, event: _ScheduledEvent) -> None:
        request_id = event.request_ids[0]
        self.waiting.append(request_id)
        self.records[request_id].append(Event(self.now_ms, "请求到达"))
        self.trace.append(f"  waiting={list(self.waiting)}；GPU {'忙碌，继续等待' if self.busy else '空闲'}")
        self.wake_scheduler()

    def complete(self, event: _ScheduledEvent) -> None:
        assert event.plan is not None  # 完成的是启动时固定的计划，不重新选成员。
        if self.kv_pool is not None:
            assert event.reservation is not None
            self.kv_pool.complete(event.reservation)
            self.record_kv(self.now_ms, "ready")
        for item in event.plan.items:
            request_id = item.request_id
            if item.phase == "Recompute":
                # 【恢复重点2】KV进度由池提交；原输入/已输出token记录都不倒退或重复。
                assert self.recovery is not None and self.recovery.restoring is not None and self.kv_pool is not None
                rebuilt = self.kv_pool.cached_tokens(request_id)
                self.records[request_id].append(Event(self.now_ms, f"重算KV进度{rebuilt}/{self.recovery.restoring.cached_tokens}"))
                if rebuilt == self.recovery.restoring.cached_tokens:
                    self.recovery.finish_restore(self.now_ms)
                    if self.token_times[request_id]:
                        self.active.append(request_id)
                    else:
                        self.waiting.appendleft(request_id)
                continue
            if item.phase == "Prefill":
                # 【分块重点 1】完成事件才累加已处理输入；中间块不产生输出。
                assert item.range_start == self.prefill_done[request_id]
                self.prefill_done[request_id] += item.token_count
                assert self.prefill_done[request_id] <= self.by_id[request_id].input_tokens
                if self.prefill_done[request_id] < self.by_id[request_id].input_tokens:
                    self.records[request_id].append(Event(self.now_ms, f"Prefill 分块完成，输入进度 "
                        f"{self.prefill_done[request_id]}/{self.by_id[request_id].input_tokens}，尚无首 token"))
                    # 当前 FCFS 继续优先服务队首分块；不隐式改变阶段策略。
                    self.waiting.appendleft(request_id)
                    continue
            self.token_times[request_id].append(self.now_ms)
            count = len(self.token_times[request_id])
            self.records[request_id].append(Event(self.now_ms, f"{item.phase} 完成，产出 token", count))
            if count == self.by_id[request_id].output_tokens:
                self.records[request_id].append(Event(self.now_ms, "请求完成"))
                if request_id in self.active:
                    self.active.remove(request_id)
                if self.kv_pool is not None:
                    self.kv_pool.release_request(request_id)
                    self.record_kv(self.now_ms, f"release:{request_id}")
                if self.recovery is not None:
                    self.recovery.completed(request_id)
            elif item.phase == "Prefill":
                if self.config.role == "P":
                    # 【PD重点】首token已经记入原历史；剩余输出等KV传到D，不留在P的active。
                    assert self.on_prefill_ready is not None
                    self.on_prefill_ready(request_id, self.now_ms)
                else:
                    self.active.append(request_id)
        # 【混合完成重点】只轮转本轮实际Decode成员，新进入active的Prefill不算已服务。
        served_decode = tuple(item.request_id for item in event.plan.items if item.phase == "Decode")
        if served_decode:
            self.active = list(rotate_after_completion(tuple(self.active), served_decode))
        # 【交替重点】进度提交完成后才通知策略；查询计划本身不能切换轮次。
        if event.plan.phase != "Recompute":
            self.scheduler.on_batch_completed(event.plan)
        self.busy = False
        self.trace.append(f"  完成后 active={self.active}；GPU 空闲")
        self.wake_scheduler()

    def dispatch(self) -> None:
        self.wakeup_pending = False
        if self.busy or not (self.waiting or self.active or (self.recovery and (self.recovery.queue or self.recovery.restoring))):
            return
        # 【恢复重点3】先排空仍驻留请求，再FIFO恢复；新到达请求不抢恢复所需空间。
        if self.recovery is not None and self.kv_pool is not None:
            resident = any(self.kv_pool.cached_tokens(item) > 0 for item in tuple(self.waiting) + tuple(self.active))
            if self.recovery.queue and self.recovery.restoring is None and self.recovery.focus is None and not resident:
                self.recovery.start_restore(self.now_ms)
                self.record_kv(self.now_ms, f"restore_start:{self.recovery.focus}")
        focused_waiting = tuple(item for item in self.waiting if self.recovery is None or self.recovery.focus is None
                                or item == self.recovery.focus)
        focused_active = tuple(item for item in self.active if self.recovery is None or self.recovery.focus is None
                               or item == self.recovery.focus)
        # 【选批重点：只读快照 → 候选计划】快照不暴露真实最终输出长度。
        snapshot = SchedulingSnapshot(
            tuple(ReadyTask(item, "Prefill", self.by_id[item].input_tokens - self.prefill_done[item],
                            self.prefill_done[item]) for item in focused_waiting),
            tuple(ReadyTask(item, "Decode", 1) for item in focused_active),
        )
        if self.recovery is not None and self.recovery.restoring is not None:
            assert self.kv_pool is not None
            target = self.recovery.restoring
            cursor = self.kv_pool.cached_tokens(target.request_id)
            limits = [target.cached_tokens - cursor]
            limits.extend(limit for limit in (self.scheduler.config.max_tokens_per_batch,
                                               self.scheduler.config.prefill_chunk_tokens) if limit is not None)
            plan = BatchPlan((ReadyTask(target.request_id, "Recompute", min(limits), cursor),),
                             "FIFO重算恢复：只重建已提交KV，不产生输出")
        else:
            plan = self.scheduler.decide(snapshot)
        if plan is None:
            return
        # 【KV重点1】选批只提候选，容量模块可缩块/删成员；未准入不修改队列。
        if self.kv_pool is not None:
            plan = self.kv_pool.admit(plan)
            if plan is None:
                # 队首Prefill放不下时，尝试可增长的Decode释放空间；不抢占任何请求。
                ready_decode = tuple(item for item in snapshot.ready_decode
                    if self.kv_pool.additional_blocks(item.request_id, item.token_count) <= self.kv_pool.free_blocks)
                fallback = self.scheduler.decide(SchedulingSnapshot((), ready_decode))
                plan = self.kv_pool.admit(fallback) if fallback is not None else None
                if plan is not None:
                    plan = BatchPlan(plan.items, plan.reason + "；KV阻塞时回退Decode")
            if plan is None:
                if self.try_preempt():
                    self.wake_scheduler()
                    return
                self.trace.append("  KV容量阻塞：无可准入计划；等待未来到达，否则结束为blocked")
                self.record_kv(self.now_ms, "blocked")
                return
        members = plan.request_ids
        phase = plan.phase
        self.trace.append(f"  候选 Prefill={list(self.waiting)}，Decode={self.active}；"
                     f"选择 {phase} {list(members)}；{plan.reason}")
        self.trace.append(f"  工作量={[(item.request_id, item.token_count) for item in plan.items]}；"
                     f"本轮 token 总数={sum(item.token_count for item in plan.items)}")
        # 【预测重点1】只提供本轮工作与已知上下文，不提供未来输出长度。
        # 首token是最终Prefill产生的，尚未再次输入模型：Decode context因此减1。
        predictor_items = tuple(PredictorItem(
            item.request_id, item.phase, item.token_count,
            item.range_start if item.phase in ("Prefill", "Recompute") else
            self.by_id[item.request_id].input_tokens + len(self.token_times[item.request_id]) - 1,
            item.phase == "Decode" or (item.phase == "Prefill" and
            item.range_start + item.token_count == self.by_id[item.request_id].input_tokens),
        ) for item in plan.items)
        estimate = self.duration_predictor.estimate(predictor_items)
        duration_ms = estimate.duration_ms
        # 拒绝非法预测与无法推进浮点时钟的时间，避免无进展循环。
        _validate_time("预测耗时", duration_ms, allow_zero=False)
        end_ms = self.now_ms + duration_ms
        _validate_time("完成时间", end_ms, allow_zero=False)
        if end_ms <= self.now_ms:
            raise ValueError("预测耗时过小，无法推进模拟时钟")
        # 【KV重点2】预测成功后原子预留，在事件完成前不提交有效KV范围。
        reservation = None
        if self.kv_pool is not None:
            reservation = self.kv_pool.reserve(plan)
            self.record_kv(self.now_ms, "reserved")
            try:
                self.kv_pool.begin_populating(reservation)
            except Exception:
                self.kv_pool.rollback(reservation)
                raise
            self.record_kv(self.now_ms, "populating")
        # 【提交重点】准入/预测/预留成功后，才取走请求。
        # 【混合提交重点】整批标签可能是Mixed，按成员阶段取走Prefill并记录首次开始。
        for item in plan.items:
            if item.phase != "Prefill":
                continue
            request_id = self.waiting.popleft()
            assert request_id == item.request_id
            # 【分块重点 2】等待时间以第一块开始为准，不被后续块覆盖。
            if request_id not in self.starts:
                self.starts[request_id] = self.now_ms
                self.records[request_id].append(Event(self.now_ms, "开始 Prefill"))
            else:
                self.records[request_id].append(Event(self.now_ms, "继续 Prefill 分块"))

        # 【重点 5：开始执行只注册未来完成，不把时钟跳到结束。】
        # members 固定本轮成员；busy 阻止其他到达事件启动重叠计算。
        self.busy = True
        self.steps.append(BatchStep(self.now_ms, end_ms, phase, members))
        self.plans.append(plan)
        self.estimates.append(estimate)
        self.schedule(end_ms, "complete", members, phase, plan, reservation)
        self.trace.append(f"  启动 {phase} {list(members)}；注册 {end_ms:g}ms 完成通知")

    def finish(self) -> EventRun:
        owned = tuple(item for item in self.ordered if item.request_id not in self.departed)
        completed = tuple(item for item in owned if len(self.token_times[item.request_id]) == item.output_tokens)
        pending = tuple(item.request_id for item in owned if len(self.token_times[item.request_id]) < item.output_tokens)
        # 容量满时保留已完成结果与块表，不假装所有请求完成，也不无限安排唤醒事件。
        results = tuple(
            Result(item, tuple(self.records[item.request_id]), tuple(self.token_times[item.request_id]), self.starts[item.request_id])
            for item in completed
        )
        return EventRun(BatchRun(results, tuple(self.steps)), tuple(self.trace), tuple(self.plans), tuple(self.estimates),
                        tuple(self.kv_records), self.kv_pool.snapshot() if self.kv_pool is not None else None,
                        "blocked" if pending else "completed", pending,
                        tuple(PendingProgress(item, self.prefill_done[item], tuple(self.token_times[item]), self.starts.get(item))
                          for item in pending), tuple(self.recovery.records) if self.recovery is not None else (),
                        ((self.recovery.restoring,) if self.recovery.restoring is not None else ()) + tuple(self.recovery.queue)
                        if self.recovery is not None else ())


def _drive_workers(requests: tuple[Request, ...], configs: tuple[WorkerConfig, ...],
                   assignments: dict[str, str], predictor: Predictor | None = None,
                   pd_config: PDConfig | None = None) -> ClusterRun:
    """【多卡重点3】所有卡共用此处的future；不是分别跑完单卡后拼接结果。"""
    if not configs or len({worker.worker_id for worker in configs}) != len(configs):
        raise ValueError("workers必须非空且worker_id唯一")
    if len({worker.gpu_id for worker in configs}) != len(configs):
        raise ValueError("当前每个worker必须对应不同的独立gpu_id")
    if len({request.request_id for request in requests}) != len(requests):
        raise ValueError("request_id必须唯一")
    if set(assignments) != {request.request_id for request in requests}:
        raise ValueError("固定归属表必须恰好覆盖全部请求")
    worker_ids = {worker.worker_id for worker in configs}
    if any(not isinstance(owner, str) or owner not in worker_ids for owner in assignments.values()):
        raise ValueError("请求归属引用了不存在的worker")
    if pd_config is not None:
        pd_config.validate_workers(configs)
        if any(owner != pd_config.prefill_worker for owner in assignments.values()):
            raise ValueError("固定1P1D的请求必须先到P")
    elif any(config.role != "Mix" for config in configs):
        raise ValueError("独立多worker模式只接受Mix；P/D需要显式pd配置")
    future: list[_ScheduledEvent] = []
    sequence = 0
    execution_order: list[tuple[str, int]] = []

    def schedule(worker_id: str, time_ms: float, kind: str, ids: tuple[str, ...], phase: str,
                 plan: BatchPlan | None, reservation: KVReservation | None) -> None:
        nonlocal sequence
        priority = {"arrival": 0, "complete": 1, "transfer_complete": 1,
                    "wakeup": 2, "transfer_start": 2}[kind]
        heapq.heappush(future, _ScheduledEvent(time_ms, priority, sequence, kind, ids,
                                             phase, plan, reservation, worker_id))
        sequence += 1
        if kind == "complete":
            # 记录启动顺序，使全局steps/plans/estimates一一对应；完成顺序可能不同。
            execution_order.append((worker_id, len(states[worker_id].steps) - 1))

    states = {config.worker_id: _WorkerRuntime(config,
              tuple(request for request in requests if assignments[request.request_id] == config.worker_id),
              schedule, predictor) for config in configs}
    handoff = None
    if pd_config is not None:
        def notify(time_ms: float, kind: str, request_id: str) -> None:
            # start是链路唤醒，真正成员由FIFO队首决定，不能把触发者误当本次成员。
            ids = () if kind == "transfer_start" else (request_id,)
            schedule(pd_config.decode_worker, time_ms, kind, ids, "", None, None)
        handoff = HandoffController(pd_config, states[pd_config.prefill_worker],
                                     states[pd_config.decode_worker], notify)
        states[pd_config.prefill_worker].on_prefill_ready = handoff.prefill_ready
    for request in sorted(requests, key=lambda item: item.arrival_ms):
        schedule(assignments[request.request_id], request.arrival_ms, "arrival",
                 (request.request_id,), "", None, None)
    trace: list[str] = []
    now_ms = 0.0
    while future:
        event = heapq.heappop(future)
        now_ms = event.time_ms
        if event.kind in ("transfer_start", "transfer_complete"):
            assert handoff is not None
            trace.append(f"[link] {event.time_ms:g}ms 处理 {event.kind} {list(event.request_ids)}")
            if event.kind == "transfer_start":
                handoff.start_next(event.time_ms)
            else:
                handoff.complete(event.request_ids[0], event.time_ms)
            continue
        state = states[event.worker_id]
        previous = len(state.trace)
        free_before = state.kv_pool.free_blocks if state.kv_pool is not None else 0
        state.handle(event)
        trace.extend(f"[{event.worker_id}] {line}" for line in state.trace[previous:])
        # 【等待重点4】只响应目标池真实释放块；检查事件排在同刻全部完成事件之后。
        # 本阶段没有swap/缓存回收，D容量只会在请求完成释放时增加，无需轮询。
        if (handoff is not None and state is handoff.target and state.kv_pool is not None
                and state.kv_pool.free_blocks > free_before):
            handoff.capacity_released(now_ms)
    worker_runs = tuple(WorkerRun(config, states[config.worker_id].finish()) for config in configs)
    results_by_id = {result.request.request_id: result for worker in worker_runs for result in worker.run.batch_run.results}
    ordered = sorted(requests, key=lambda item: item.arrival_ms)
    results = tuple(results_by_id[request.request_id] for request in ordered if request.request_id in results_by_id)
    pending = tuple(request.request_id for request in ordered if request.request_id not in results_by_id)
    steps: list[BatchStep] = []
    plans: list[BatchPlan] = []
    estimates: list[BatchEstimate] = []
    for worker_id, index in execution_order:
        state = states[worker_id]
        step = state.steps[index]
        steps.append(WorkerStep(step.start_ms, step.end_ms, step.phase, step.request_ids,
                                worker_id, state.config.gpu_id))
        plans.append(state.plans[index])
        estimates.append(state.estimates[index])
    return ClusterRun(BatchRun(results, tuple(steps)), worker_runs, tuple(plans), tuple(estimates),
                      tuple(trace), tuple((request.request_id, assignments[request.request_id]) for request in ordered),
                      "blocked" if pending else "completed", pending, pd_config,
                      tuple(handoff.records) if handoff is not None else (),
                      handoff.snapshot_waits(now_ms) if handoff is not None else ())


def simulate_workers(requests: tuple[Request, ...], workers: tuple[WorkerConfig, ...],
                     assignments: dict[str, str]) -> ClusterRun:
    """显式固定归属；一张卡容量阻塞不会结束其他卡的事件。"""
    return _drive_workers(requests, workers, assignments)


def simulate_pd(requests: tuple[Request, ...], workers: tuple[WorkerConfig, ...], config: PDConfig) -> ClusterRun:
    """固定1P1D：请求从P进入，首token后跨链路到D；不进行动态路由。"""
    return _drive_workers(requests, workers, {request.request_id: config.prefill_worker for request in requests},
                          pd_config=config)


def simulate_events(
    requests: tuple[Request, ...], timing: Timing, *,
    policy: Policy = "prefill_first",
    max_batch_requests: int | None = None,
    max_tokens_per_batch: int | None = None,
    prefill_chunk_tokens: int | None = None,
    scheduler_profile: SchedulerProfile | None = None,
    predictor: Predictor | None = None,
    kv_config: KVConfig | None = None,
    recovery_config: RecoveryConfig = RecoveryConfig(),
) -> EventRun:
    """旧单卡入口：复用同一个事件内核，返回原有EventRun与原样单卡记录。"""
    limits = SchedulerConfig(policy, max_batch_requests, max_tokens_per_batch, prefill_chunk_tokens)
    profile = scheduler_profile or SchedulerProfile("fcfs_chunked", "round_robin", policy, limits)
    worker = WorkerConfig("worker_0", kv_config.gpu_id if kv_config else "gpu_0", profile,
                          timing=timing, kv_config=kv_config, recovery_config=recovery_config)
    cluster = _drive_workers(requests, (worker,), {request.request_id: worker.worker_id for request in requests}, predictor)
    return cluster.workers[0].run
