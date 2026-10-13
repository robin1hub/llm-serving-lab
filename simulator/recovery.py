"""可配置受害者选择与FIFO重算恢复；实际KV分配/事件仍由内核执行。"""

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class RecoveryConfig:
    capacity_policy: str = "block"
    victim_policy: str = "fewest_generated"
    recovery_action: str = "recompute"

    def __post_init__(self) -> None:
        if self.capacity_policy not in ("block", "preempt"):
            raise ValueError("capacity_policy必须是block或preempt")
        if self.victim_policy not in ("fewest_generated", "newest"):
            raise ValueError("victim_policy必须是fewest_generated或newest")
        if self.recovery_action != "recompute":
            raise ValueError("当前recovery_action只实现recompute，交换需要CPU池和传输")


@dataclass(frozen=True)
class RecoveryCandidate:
    request_id: str
    generated_tokens: int
    arrival_order: int


def select_victim(config: RecoveryConfig, candidates: tuple[RecoveryCandidate, ...]) -> str | None:
    # 【重点1：选谁与怎么恢复分开】调用者先筛掉不能释放/释放后仍无进展的请求。
    if not candidates:
        return None
    if config.victim_policy == "newest":
        return max(candidates, key=lambda item: item.arrival_order).request_id
    return min(candidates, key=lambda item: (item.generated_tokens, -item.arrival_order)).request_id


@dataclass(frozen=True)
class RecoveryTarget:
    request_id: str
    cached_tokens: int


@dataclass(frozen=True)
class RecoveryRecord:
    time_ms: float
    action: str
    request_id: str
    cached_tokens: int
    beneficiary: str | None = None


class RecoveryController:
    def __init__(self, config: RecoveryConfig) -> None:
        self.config = config
        self.queue: deque[RecoveryTarget] = deque()
        self.restoring: RecoveryTarget | None = None
        self.focus: str | None = None
        self.records: list[RecoveryRecord] = []

    def suspend(self, request_id: str, cached_tokens: int, beneficiary: str, time_ms: float) -> None:
        self.queue.append(RecoveryTarget(request_id, cached_tokens))
        # 【重点2：保护进展】先让受益请求独占直到结束，避免刚释放又立刻恢复形成循环。
        self.focus = beneficiary
        self.records.append(RecoveryRecord(time_ms, "preempt", request_id, cached_tokens, beneficiary))

    def start_restore(self, time_ms: float) -> None:
        self.restoring = self.queue.popleft()
        self.focus = self.restoring.request_id
        self.records.append(RecoveryRecord(time_ms, "restore_start", self.focus,
                                           self.restoring.cached_tokens))

    def finish_restore(self, time_ms: float) -> None:
        assert self.restoring is not None
        self.records.append(RecoveryRecord(time_ms, "resume", self.restoring.request_id,
                                           self.restoring.cached_tokens))
        self.restoring = None
        # focus仍保留，让恢复的请求继续生成并结束，而不是马上再被抢占。

    def completed(self, request_id: str) -> None:
        if self.focus == request_id:
            self.focus = None
