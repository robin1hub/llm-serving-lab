"""单请求教学实验：用虚拟时间描述 Prefill 和逐 token Decode。"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Request:
    """实验输入；output_tokens 是终止条件，不是调度器的预测。"""

    request_id: str
    arrival_ms: float
    input_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id 必须是非空字符串")
        _validate_time("arrival_ms", self.arrival_ms, allow_zero=True)
        for name, value in (
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} 必须是正整数")


@dataclass(frozen=True)
class Timing:
    """人为指定的教学耗时，单位统一为毫秒，不代表真实 GPU 性能。"""

    prefill_ms_per_input_token: float = 2.0
    decode_ms_per_step: float = 2.0

    def __post_init__(self) -> None:
        _validate_time(
            "prefill_ms_per_input_token",
            self.prefill_ms_per_input_token,
            allow_zero=False,
        )
        _validate_time("decode_ms_per_step", self.decode_ms_per_step, allow_zero=False)


def _validate_time(name: str, value: float, *, allow_zero: bool) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是数值")
    if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{name} 必须是有限的{'非负' if allow_zero else '正'}数")


@dataclass(frozen=True)
class Event:
    """一个已经发生的事件；token_index 为输出 token 的编号。"""

    time_ms: float
    description: str
    token_index: int | None = None


@dataclass(frozen=True)
class Result:
    """不可变的执行记录及由记录计算出的指标。"""

    request: Request
    events: tuple[Event, ...]
    token_times_ms: tuple[float, ...]
    start_ms: float

    @property
    def waiting_ms(self) -> float:
        return self.start_ms - self.request.arrival_ms

    @property
    def ttft_ms(self) -> float:
        return self.token_times_ms[0] - self.request.arrival_ms

    @property
    def inter_token_intervals_ms(self) -> tuple[float, ...]:
        return tuple(
            later - earlier
            for earlier, later in zip(self.token_times_ms, self.token_times_ms[1:])
        )

    @property
    def latency_ms(self) -> float:
        return self.token_times_ms[-1] - self.request.arrival_ms


def simulate(
    request: Request, timing: Timing, *, gpu_available_ms: float = 0.0
) -> Result:
    """GPU 可用且请求到达后才开始；开始后独占 GPU 直到完成。"""
    _validate_time("gpu_available_ms", gpu_available_ms, allow_zero=True)
    start_ms = max(request.arrival_ms, gpu_available_ms)
    now_ms = start_ms
    events = [
        Event(request.arrival_ms, "请求到达"),
        Event(start_ms, "开始 Prefill"),
    ]

    # Prefill 一次处理全部输入，并在完成时产出首 token。
    now_ms += request.input_tokens * timing.prefill_ms_per_input_token
    token_times = [now_ms]
    events.append(Event(now_ms, "Prefill 完成，产出 token", token_index=1))

    # 首 token 已经产生，剩余输出逐轮 Decode。
    for token_index in range(2, request.output_tokens + 1):
        now_ms += timing.decode_ms_per_step
        token_times.append(now_ms)
        events.append(Event(now_ms, "Decode 完成，产出 token", token_index))

    events.append(Event(now_ms, "请求完成"))
    return Result(request, tuple(events), tuple(token_times), start_ms)


def simulate_serial(requests: tuple[Request, ...], timing: Timing) -> tuple[Result, ...]:
    """按到达顺序串行执行；同时到达时保持输入顺序，不做抢占。"""
    request_ids = [request.request_id for request in requests]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("request_id 必须唯一")
    gpu_available_ms = 0.0
    results = []
    for request in sorted(requests, key=lambda item: item.arrival_ms):
        result = simulate(request, timing, gpu_available_ms=gpu_available_ms)
        results.append(result)
        # 下一请求只能在上一请求生成结束后使用 GPU。
        gpu_available_ms = result.token_times_ms[-1]
    return tuple(results)


def main() -> None:
    request = Request("A", arrival_ms=0.0, input_tokens=4, output_tokens=3)
    result = simulate(request, Timing())
    print("单请求实验（所有耗时均为教学假设）")
    for event in result.events:
        token = "" if event.token_index is None else f" #{event.token_index}"
        print(f"{event.time_ms:5g} ms | {request.request_id} | {event.description}{token}")
    print(f"首 token 延迟：{result.ttft_ms:g} ms")
    print(f"相邻 token 间隔：{result.inter_token_intervals_ms} ms")
    print(f"端到端延迟：{result.latency_ms:g} ms")


if __name__ == "__main__":
    main()
