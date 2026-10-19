"""固定1P1D的串行链路、FIFO容量等待与事件驱动重试；见第23章。"""

from collections import deque
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable

from .kv_cache import KVReservation
from .models import Event, _validate_time
from .scheduler import BatchPlan, ReadyTask

if TYPE_CHECKING:
    from .engine import _WorkerRuntime
    from .worker import WorkerConfig


@dataclass(frozen=True)
class LinkConfig:
    bandwidth_bytes_per_s: float
    latency_ms: float
    kv_bytes_per_token: int

    def __post_init__(self) -> None:
        _validate_time("链路带宽", self.bandwidth_bytes_per_s, allow_zero=False)
        _validate_time("链路固定延迟", self.latency_ms, allow_zero=True)
        if type(self.kv_bytes_per_token) is not int or self.kv_bytes_per_token <= 0:
            raise ValueError("kv_bytes_per_token必须是正整数")

    def duration_ms(self, payload_bytes: int) -> float:
        # 【交接重点1】传有效KV载荷，不传页面空位；单位bytes/s转成ms。
        if type(payload_bytes) is not int or payload_bytes <= 0:
            raise ValueError("KV载荷必须是正整数字节数")
        try:
            duration = self.latency_ms + payload_bytes / self.bandwidth_bytes_per_s * 1000
        except OverflowError as error:
            raise ValueError("KV传输载荷过大") from error
        _validate_time("KV传输耗时", duration, allow_zero=False)
        return duration


@dataclass(frozen=True)
class PDConfig:
    prefill_worker: str
    decode_worker: str
    link: LinkConfig

    def validate_workers(self, workers: tuple["WorkerConfig", ...]) -> None:
        if (not isinstance(self.prefill_worker, str) or not isinstance(self.decode_worker, str)
                or self.prefill_worker == self.decode_worker):
            raise ValueError("固定1P1D需要两个不同worker")
        by_id = {worker.worker_id: worker for worker in workers}
        if (len(workers) != 2 or len(by_id) != 2 or len({w.gpu_id for w in workers}) != 2
                or self.prefill_worker not in by_id or self.decode_worker not in by_id):
            raise ValueError("pd必须恰好引用两张独立卡上的P/D worker")
        p, d = by_id[self.prefill_worker], by_id[self.decode_worker]
        if p.role != "P" or d.role != "D":
            raise ValueError("pd的源必须为P、目标必须为D")
        if p.kv_config is None or d.kv_config is None:
            raise ValueError("本阶段1P1D需要两卡各自启用KV池")
        if any(w.recovery_config.capacity_policy != "block" for w in workers):
            raise ValueError("本阶段PD交接不支持抢占；传输保护与恢复协议留待后续")
        models = []
        kv_shapes = []
        for worker in workers:
            model = worker.predictor_settings.model
            if worker.predictor_settings.kind == "roofline":
                expected = (2 * model.num_layers * model.num_kv_heads
                            * (model.hidden_size // model.num_attention_heads) * model.kv_bytes_per_element)
                if self.link.kv_bytes_per_token != expected:
                    raise ValueError("链路KV字节数与Predictor模型结构不一致")
                models.append(model)
                kv_shapes.append((model.num_layers, model.num_kv_heads,
                                  model.hidden_size // model.num_attention_heads, model.kv_bytes_per_element))
            if worker.memory_budget is not None:
                if self.link.kv_bytes_per_token != worker.memory_budget.kv_bytes_per_token:
                    raise ValueError("链路KV字节数与显存预算模型不一致")
                memory_model = worker.memory_budget.model
                kv_shapes.append((memory_model.num_layers, memory_model.num_kv_heads,
                                  memory_model.head_dim, memory_model.kv_bytes_per_element))
        if len(models) == 2 and models[0] != models[1]:
            raise ValueError("P/D必须使用同一Predictor模型结构与精度")
        if len(set(kv_shapes)) > 1:
            raise ValueError("P/D的KV结构与精度必须一致")


@dataclass(frozen=True)
class TransferRecord:
    request_id: str
    source_worker: str
    target_worker: str
    ready_ms: float
    start_ms: float
    end_ms: float
    payload_tokens: int
    payload_bytes: int
    status: str = "in_flight"


@dataclass(frozen=True)
class CapacityWaitRecord:
    """【等待重点1】等待不是传输；未恢复时end_ms为空，观察截止也单独保留。"""

    request_id: str
    start_ms: float
    end_ms: float | None
    observed_until_ms: float
    required_blocks: int
    free_blocks_at_start: int
    reason: str
    status: str = "waiting"

    @property
    def wait_ms(self) -> float:
        return self.observed_until_ms - self.start_ms


class HandoffController:
    """【交接重点2】队列等待与链路执行分开；传输期间请求不能进入D的active。"""

    def __init__(self, config: PDConfig, source: "_WorkerRuntime", target: "_WorkerRuntime",
                 notify: Callable[[float, str, str], None]) -> None:
        self.config, self.source, self.target = config, source, target
        self.notify = notify
        self.queue: deque[tuple[str, float]] = deque()
        self.records: list[TransferRecord] = []
        self.active: TransferRecord | None = None
        self.reservation: KVReservation | None = None
        self.wait_records: list[CapacityWaitRecord] = []
        self.waiting_index: int | None = None
        self.start_pending = False

    def wake_link(self, now_ms: float) -> None:
        # 【等待重点2】同刻多次释放只安排一次检查；链路忙时由传输完成负责接棒。
        if self.active is None and self.queue and not self.start_pending:
            self.start_pending = True
            self.notify(now_ms, "transfer_start", self.queue[0][0])

    def capacity_released(self, now_ms: float) -> None:
        # 只有已经容量阻塞的队首需要容量通知，普通链路排队无需额外事件。
        if (self.waiting_index is not None
                and self.wait_records[self.waiting_index].status == "waiting"):
            self.wake_link(now_ms)

    def snapshot_waits(self, now_ms: float) -> tuple[CapacityWaitRecord, ...]:
        # 堆耗尽不等于等待结束：仍保留waiting和空end_ms，不能伪造成功。
        return tuple(replace(record, observed_until_ms=now_ms)
                     if record.status == "waiting" else record for record in self.wait_records)

    def prefill_ready(self, request_id: str, now_ms: float) -> None:
        source, target = self.source, self.target
        request = source.by_id[request_id]
        # 只移动执行归属，完整请求的首开始/首token/事件记录保留同一条历史。
        target.ordered.append(request)
        target.by_id[request_id] = request
        target.arrival_order[request_id] = len(target.arrival_order)
        target.prefill_done[request_id] = source.prefill_done[request_id]
        target.starts[request_id] = source.starts[request_id]
        target.token_times[request_id] = source.token_times[request_id]
        target.records[request_id] = source.records[request_id]
        source.departed.add(request_id)
        target.records[request_id].append(Event(now_ms, "P完成，等待KV链路；D尚不可执行"))
        self.queue.append((request_id, now_ms))
        # 新请求到来不反复探测已经阻塞的队首；释放容量才可能改变结果。
        if self.waiting_index is None:
            self.wake_link(now_ms)

    def start_next(self, now_ms: float) -> None:
        self.start_pending = False
        if self.active is not None or not self.queue:
            return
        request_id, ready_ms = self.queue[0]
        source, target = self.source, self.target
        assert source.kv_pool is not None and target.kv_pool is not None
        tokens = source.kv_pool.cached_tokens(request_id)
        assert tokens == source.by_id[request_id].input_tokens
        required_blocks = target.kv_pool.additional_blocks(request_id, tokens)
        if required_blocks > target.kv_pool.free_blocks:
            if self.waiting_index is None:
                # 【等待重点3】不reserve、不取走队首、不安排定时轮询；源KV原样保留。
                impossible = required_blocks > target.kv_pool.config.total_blocks
                reason = "request_exceeds_target_capacity" if impossible else "temporary_capacity"
                self.waiting_index = len(self.wait_records)
                self.wait_records.append(CapacityWaitRecord(
                    request_id, now_ms, None, now_ms, required_blocks,
                    target.kv_pool.free_blocks, reason, "impossible" if impossible else "waiting"))
                message = ("目标KV总容量不足，无法交接" if impossible else "目标KV暂时不足，等待释放")
                target.records[request_id].append(Event(now_ms, message))
                target.record_kv(now_ms, f"transfer_capacity_wait:{request_id}")
                source.record_kv(now_ms, f"transfer_source_wait_held:{request_id}")
            return
        payload = tokens * self.config.link.kv_bytes_per_token
        end_ms = now_ms + self.config.link.duration_ms(payload)
        _validate_time("KV传输完成时间", end_ms, allow_zero=False)
        if end_ms <= now_ms:
            raise ValueError("KV传输时间过小，无法推进模拟时钟")
        # 复用块池的写入事务；这是数据拷入，不是再做一次Prefill计算。
        plan = BatchPlan((ReadyTask(request_id, "Prefill", tokens),), "KV传输目标预留")
        reservation = target.kv_pool.reserve(plan)
        target.record_kv(now_ms, f"transfer_reserved:{request_id}")
        try:
            target.kv_pool.begin_populating(reservation)
        except Exception:
            target.kv_pool.rollback(reservation)
            raise
        target.record_kv(now_ms, f"transfer_populating:{request_id}")
        if self.waiting_index is not None:
            record = self.wait_records[self.waiting_index]
            assert record.request_id == request_id and record.status == "waiting"
            self.wait_records[self.waiting_index] = replace(
                record, end_ms=now_ms, observed_until_ms=now_ms, status="resumed")
            self.waiting_index = None
            target.records[request_id].append(Event(now_ms, "目标容量满足，恢复KV交接"))
        self.queue.popleft()
        self.active = TransferRecord(request_id, source.config.worker_id, target.config.worker_id,
                                     ready_ms, now_ms, end_ms, tokens, payload)
        self.reservation = reservation
        self.records.append(self.active)
        target.records[request_id].append(Event(now_ms, f"开始传输{payload}字节KV，完成于{end_ms:g}ms"))
        source.record_kv(now_ms, f"transfer_source_held:{request_id}")
        self.notify(end_ms, "transfer_complete", request_id)

    def complete(self, request_id: str, now_ms: float) -> None:
        # 【交接重点3】先提交目标Ready，再释放源，最后让D看见请求。
        assert self.active is not None and self.active.request_id == request_id
        assert now_ms == self.active.end_ms and self.reservation is not None
        source, target = self.source, self.target
        assert target.kv_pool is not None and source.kv_pool is not None
        target.kv_pool.complete(self.reservation)
        target.record_kv(now_ms, f"transfer_ready:{request_id}")
        source.kv_pool.release_request(request_id)
        source.record_kv(now_ms, f"transfer_source_release:{request_id}")
        target.active.append(request_id)
        target.records[request_id].append(Event(now_ms, "KV交接完成，进入D队列"))
        self.records[-1] = replace(self.active, status="completed")
        self.active, self.reservation = None, None
        source.now_ms = target.now_ms = now_ms
        source.wake_scheduler()
        target.wake_scheduler()
        self.wake_link(now_ms)
