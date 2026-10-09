"""第三步：在轮次边界组 Decode batch，使用 Prefill 优先的教学规则。"""

from dataclasses import dataclass, field

from serving_lab import Event, Request, Result, Timing


@dataclass(frozen=True)
class BatchStep:
    """一轮 GPU 执行记录；成员一旦确定，本轮中途不会改变。"""

    start_ms: float
    end_ms: float
    phase: str
    request_ids: tuple[str, ...]


@dataclass(frozen=True)
class BatchRun:
    results: tuple[Result, ...]
    steps: tuple[BatchStep, ...]


@dataclass
class _Progress:
    """函数内部的可变执行进度，与不可变请求参数分开。"""

    request: Request
    start_ms: float | None = None
    token_times: list[float] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)


def simulate_batched(requests: tuple[Request, ...], timing: Timing) -> BatchRun:
    """单 GPU；单请求完整 Prefill 优先，否则所有活跃请求一起 Decode。

    Decode batch 整轮耗时固定，不随成员数量变化，仅用于理解流程。
    没有容量限制、混合 P/D batch 或 Prefill 分块。
    """
    if len({request.request_id for request in requests}) != len(requests):
        raise ValueError("request_id 必须唯一")

    # 稳定排序保证同时到达时遵循输入顺序。
    pending = [
        _Progress(request, events=[Event(request.arrival_ms, "请求到达")])
        for request in sorted(requests, key=lambda item: item.arrival_ms)
    ]
    all_progress = tuple(pending)
    active: list[_Progress] = []
    steps: list[BatchStep] = []
    now_ms = 0.0

    while pending or active:
        # 无可执行请求时直接跳到下一次到达，不做真实等待。
        if not active and pending[0].request.arrival_ms > now_ms:
            now_ms = pending[0].request.arrival_ms

        if pending and pending[0].request.arrival_ms <= now_ms:
            # 本轮采用 Prefill 优先；已有 Decode 请求在这段时间暂停。
            progress = pending.pop(0)
            progress.start_ms = now_ms
            progress.events.append(Event(now_ms, "开始 Prefill"))
            end_ms = now_ms + progress.request.input_tokens * timing.prefill_ms_per_input_token
            steps.append(BatchStep(now_ms, end_ms, "Prefill", (progress.request.request_id,)))
            now_ms = end_ms
            progress.token_times.append(now_ms)
            progress.events.append(Event(now_ms, "Prefill 完成，产出 token", 1))
            if progress.request.output_tokens == 1:
                # 首 token 就是全部输出，不应加入 Decode 再生成一次。
                progress.events.append(Event(now_ms, "请求完成"))
            else:
                active.append(progress)
                progress.events.append(Event(now_ms, "加入活跃 Decode 集合"))
            continue

        # 在轮次开始时冻结成员：本轮执行期间到达的请求只能下轮被考虑。
        members = tuple(active)
        end_ms = now_ms + timing.decode_ms_per_step
        steps.append(BatchStep(
            now_ms, end_ms, "Decode", tuple(item.request.request_id for item in members)
        ))
        now_ms = end_ms
        for progress in members:
            # 时间只推进一次，所有成员各生成一个 token，而非逐成员累加耗时。
            progress.token_times.append(now_ms)
            token_index = len(progress.token_times)
            progress.events.append(Event(now_ms, "Decode 完成，产出 token", token_index))
            if token_index == progress.request.output_tokens:
                progress.events.append(Event(now_ms, "请求完成，退出活跃 Decode 集合"))

        # 本轮完成后重建活跃集合；下一轮的 batch 可以缩小或加入新请求。
        active = [
            item for item in active if len(item.token_times) < item.request.output_tokens
        ]

    results = []
    for progress in all_progress:
        assert progress.start_ms is not None  # 退出循环意味着每个请求均已执行完毕。
        results.append(Result(
            progress.request, tuple(progress.events), tuple(progress.token_times), progress.start_ms
        ))
    return BatchRun(tuple(results), tuple(steps))
