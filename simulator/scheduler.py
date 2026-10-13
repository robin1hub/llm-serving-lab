"""只读候选状态并提出 BatchPlan，不取走请求、不修改生成进度。"""

from dataclasses import dataclass
from typing import Literal


Phase = Literal["Prefill", "Decode"]
Policy = Literal["prefill_first", "decode_first", "alternating"]
Role = Literal["P", "D", "Mix"]


@dataclass(frozen=True)
class SchedulerConfig:
    policy: Policy = "prefill_first"
    max_batch_requests: int | None = None
    max_tokens_per_batch: int | None = None
    prefill_chunk_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.policy not in ("prefill_first", "decode_first", "alternating"):
            raise ValueError("policy 必须是 prefill_first、decode_first 或 alternating")
        # None 表示不启用该限制，兼容旧实验；这些限制都不代表 KV 容量。
        for name in ("max_batch_requests", "max_tokens_per_batch", "prefill_chunk_tokens"):
            limit = getattr(self, name)
            if limit is not None and (type(limit) is not int or limit <= 0):
                raise ValueError(f"{name} 必须是正整数或 null")


@dataclass(frozen=True)
class SchedulerProfile:
    """按能力配置算法；P/D 方案可以独立定义，Mix 需要三个槽都存在。"""

    prefill_algorithm: str | None = None
    decode_algorithm: str | None = None
    mix_algorithm: str | None = None
    constraints: SchedulerConfig = SchedulerConfig()

    def __post_init__(self) -> None:
        for name, supported in (("prefill_algorithm", ("fcfs_chunked",)),
                                ("decode_algorithm", ("round_robin",)),
                                ("mix_algorithm", ("prefill_first", "decode_first", "alternating"))):
            value = getattr(self, name)
            if value is not None and value not in supported:
                raise ValueError(f"{name} 不支持算法 {value!r}")
        if all(value is None for value in (self.prefill_algorithm, self.decode_algorithm, self.mix_algorithm)):
            raise ValueError("调度方案至少指定一个算法")

    def validate_role(self, role: Role) -> None:
        if role not in ("P", "D", "Mix"):
            raise ValueError("worker.role 必须是 P、D 或 Mix")
        required = {"P": ("prefill_algorithm",), "D": ("decode_algorithm",),
                    "Mix": ("prefill_algorithm", "decode_algorithm", "mix_algorithm")}[role]
        for field in required:
            if getattr(self, field) is None:
                raise ValueError(f"{role} worker 的方案缺少 {field}")


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
    def __init__(
        self, config: SchedulerConfig, *, profile: SchedulerProfile | None = None,
        role: Role = "Mix",
    ) -> None:
        self.profile = profile or SchedulerProfile("fcfs_chunked", "round_robin", config.policy, config)
        self.profile.validate_role(role)
        self.config = self.profile.constraints
        self.role = role
        self.last_completed_phase: Phase | None = None

    def on_batch_completed(self, plan: BatchPlan) -> None:
        # 有状态策略仅在完成后更新；重复 decide 不会偷偷消耗交替机会。
        self.last_completed_phase = plan.phase

    def decide(self, snapshot: SchedulingSnapshot) -> BatchPlan | None:
        # 【重点 1：先选阶段】空快照返回 Idle（None），没有修改任何队列。
        prefill = snapshot.waiting_prefill
        decode = snapshot.ready_decode
        if (self.role == "P" and decode) or (self.role == "D" and prefill):
            raise ValueError("worker 收到了角色不支持的就绪阶段")
        # 【角色重点】只有一个阶段就绪时用该阶段算法；两者同时就绪才调用 Mix 槽。
        choose_prefill = bool(prefill) and not decode
        if prefill and decode:
            algorithm = self.profile.mix_algorithm
            choose_prefill = algorithm == "prefill_first" or (
                algorithm == "alternating" and self.last_completed_phase != "Prefill"
            )
        if choose_prefill:
            # 【预算重点 1】剩余输入、整轮预算、单请求 chunk 三者取最小值。
            task = prefill[0]
            limits = [task.token_count]
            limits.extend(limit for limit in (self.config.max_tokens_per_batch,
                                               self.config.prefill_chunk_tokens) if limit is not None)
            selected = ReadyTask(task.request_id, "Prefill", min(limits), task.range_start)
            reason = "仅 Prefill 就绪：fcfs_chunked"
            if decode:
                reason = f"fcfs_chunked；Mix={self.profile.mix_algorithm} 选择 Prefill"
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
        reason = (f"Mix={self.profile.mix_algorithm} 选择 Decode；" if prefill else "仅 Decode 就绪；") + reason
        return BatchPlan(members, reason)


def rotate_after_completion(active: tuple[str, ...], served: tuple[str, ...]) -> tuple[str, ...]:
    # 【重点 3：完成后才轮转】未服务的留在前面，已服务且未完成的移到尾部。
    # active 由内核先剔除完成者；不在决定计划时改变轮转位置。
    served_set = set(served)
    return tuple(item for item in active if item not in served_set) + tuple(
        item for item in active if item in served_set
    )
