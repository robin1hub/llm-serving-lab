"""只读候选状态并提出 BatchPlan，不取走请求、不修改生成进度。"""

from dataclasses import dataclass
from typing import Literal


Phase = Literal["Prefill", "Decode"]
Policy = Literal["prefill_first", "decode_first"]


@dataclass(frozen=True)
class SchedulerConfig:
    policy: Policy = "prefill_first"
    max_batch_requests: int | None = None
    max_tokens_per_batch: int | None = None
    prefill_chunk_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.policy not in ("prefill_first", "decode_first"):
            raise ValueError("policy 必须是 prefill_first 或 decode_first")
        # None 表示不启用该限制，兼容旧实验；这些限制都不代表 KV 容量。
        for name in ("max_batch_requests", "max_tokens_per_batch", "prefill_chunk_tokens"):
            limit = getattr(self, name)
            if limit is not None and (type(limit) is not int or limit <= 0):
                raise ValueError(f"{name} 必须是正整数或 null")


@dataclass(frozen=True)
class ReadyTask:
    """一个候选的下一步工作；没有实际最终输出长度，避免选批依赖未来答案。"""

    request_id: str
    phase: Phase
    token_count: int
    range_start: int = 0  # Prefill 本轮输入范围起点；Decode 暂不使用此字段。


@dataclass(frozen=True)
class SchedulingSnapshot:
    waiting_prefill: tuple[ReadyTask, ...]
    ready_decode: tuple[ReadyTask, ...]


@dataclass(frozen=True)
class BatchPlan:
    """明确这一轮执行谁、哪个阶段、处理多少新 token；暂不混合阶段。"""

    items: tuple[ReadyTask, ...]
    reason: str

    @property
    def phase(self) -> Phase:
        return self.items[0].phase

    @property
    def request_ids(self) -> tuple[str, ...]:
        return tuple(item.request_id for item in self.items)


class Scheduler:
    def __init__(self, config: SchedulerConfig) -> None:
        self.config = config

    def decide(self, snapshot: SchedulingSnapshot) -> BatchPlan | None:
        # 【重点 1：先选阶段】空快照返回 Idle（None），没有修改任何队列。
        prefill = snapshot.waiting_prefill
        decode = snapshot.ready_decode
        if prefill and (self.config.policy == "prefill_first" or not decode):
            # 【预算重点 1】剩余输入、整轮预算、单请求 chunk 三者取最小值。
            task = prefill[0]
            limits = [task.token_count]
            limits.extend(limit for limit in (self.config.max_tokens_per_batch,
                                               self.config.prefill_chunk_tokens) if limit is not None)
            selected = ReadyTask(task.request_id, "Prefill", min(limits), task.range_start)
            reason = "Prefill 优先或没有 Decode 候选"
            if selected.token_count < task.token_count:
                reason += "；按 token 预算/chunk 分块"
            return BatchPlan((selected,), reason)
        if not decode:
            return None

        # 【重点 2：候选集合不等于本轮成员】只取轮转队列前 limit 个。
        # 【预算重点 2】每个 Decode 消耗 1 token，人数和 token 预算同时限制成员。
        limits = [len(decode)]
        limits.extend(limit for limit in (self.config.max_batch_requests,
                                           self.config.max_tokens_per_batch) if limit is not None)
        members = decode[:min(limits)]
        reason = "全部 Decode 候选入选" if len(members) == len(decode) else "请求数/token 上限，其余留待下一轮"
        return BatchPlan(members, reason)


def rotate_after_completion(active: tuple[str, ...], served: tuple[str, ...]) -> tuple[str, ...]:
    # 【重点 3：完成后才轮转】未服务的留在前面，已服务且未完成的移到尾部。
    # active 由内核先剔除完成者；不在决定计划时改变轮转位置。
    served_set = set(served)
    return tuple(item for item in active if item not in served_set) + tuple(
        item for item in active if item in served_set
    )
