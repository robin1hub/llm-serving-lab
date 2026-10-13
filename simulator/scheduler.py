"""只读候选状态并提出 BatchPlan，不取走请求、不修改生成进度。"""

from dataclasses import dataclass
from typing import Literal


Phase = Literal["Prefill", "Decode"]
Policy = Literal["prefill_first", "decode_first"]


@dataclass(frozen=True)
class SchedulerConfig:
    policy: Policy = "prefill_first"
    max_batch_requests: int | None = None

    def __post_init__(self) -> None:
        if self.policy not in ("prefill_first", "decode_first"):
            raise ValueError("policy 必须是 prefill_first 或 decode_first")
        limit = self.max_batch_requests
        # None 明确表示不限成员数，兼容之前的教学实验；不是显存无限。
        if limit is not None and (type(limit) is not int or limit <= 0):
            raise ValueError("max_batch_requests 必须是正整数或 null")


@dataclass(frozen=True)
class ReadyTask:
    """一个候选的下一步工作；没有实际最终输出长度，避免选批依赖未来答案。"""

    request_id: str
    phase: Phase
    token_count: int


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
            # 完整 Prefill 仍一次一个；本步不引入分块、token 预算和混合 batch。
            return BatchPlan(prefill[:1], "Prefill 优先或没有 Decode 候选")
        if not decode:
            return None

        # 【重点 2：候选集合不等于本轮成员】只取轮转队列前 limit 个。
        limit = self.config.max_batch_requests
        members = decode if limit is None else decode[:limit]
        reason = "全部 Decode 候选入选" if len(members) == len(decode) else "请求数上限，其余留待下一轮"
        return BatchPlan(members, reason)


def rotate_after_completion(active: tuple[str, ...], served: tuple[str, ...]) -> tuple[str, ...]:
    # 【重点 3：完成后才轮转】未服务的留在前面，已服务且未完成的移到尾部。
    # active 由内核先剔除完成者；不在决定计划时改变轮转位置。
    served_set = set(served)
    return tuple(item for item in active if item not in served_set) + tuple(
        item for item in active if item in served_set
    )
