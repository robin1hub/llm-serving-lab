"""当前模拟器的数据结构与校验。"""

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

