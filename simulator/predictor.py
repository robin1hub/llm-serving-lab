"""只读耗时预测：教学公式和简化 dense Transformer roofline。见第10章。"""

from dataclasses import dataclass
import math
from typing import Protocol

from .models import Timing, _validate_time
from .scheduler import Phase


@dataclass(frozen=True)
class ModelProfile:
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_kv_heads: int
    vocab_size: int
    weight_bytes_per_element: int = 2
    activation_bytes_per_element: int = 2
    kv_bytes_per_element: int = 2

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"model.{name} 必须是正整数")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size 必须是 num_attention_heads 的整数倍")
        if self.num_attention_heads % self.num_kv_heads:
            raise ValueError("attention heads 必须是 KV heads 的整数倍")


@dataclass(frozen=True)
class DeviceProfile:
    name: str
    peak_flops: float
    memory_bandwidth_bytes_per_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("device.name 必须非空")
        _validate_time("device.peak_flops", self.peak_flops, allow_zero=False)
        _validate_time("device.memory_bandwidth_bytes_per_s",
                       self.memory_bandwidth_bytes_per_s, allow_zero=False)


@dataclass(frozen=True)
class Calibration:
    # 第一版只分两组；系数乘在时间上，大于1会使预测更慢。
    dense_factor: float = 1.0
    attention_factor: float = 1.0
    batch_overhead_ms: float = 0.0

    def __post_init__(self) -> None:
        _validate_time("dense_factor", self.dense_factor, allow_zero=False)
        _validate_time("attention_factor", self.attention_factor, allow_zero=False)
        _validate_time("batch_overhead_ms", self.batch_overhead_ms, allow_zero=True)


@dataclass(frozen=True)
class PredictorSettings:
    kind: str = "teaching"
    model: ModelProfile | None = None
    device: DeviceProfile | None = None
    calibration: Calibration | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("teaching", "roofline"):
            raise ValueError("predictor.kind 必须是 teaching 或 roofline")
        if self.kind == "roofline" and any(
            value is None for value in (self.model, self.device, self.calibration)
        ):
            raise ValueError("roofline 需要 model/device/calibration")
        if self.kind == "teaching" and any(
            value is not None for value in (self.model, self.device, self.calibration)
        ):
            raise ValueError("teaching 不使用 model/device/calibration")


@dataclass(frozen=True)
class PredictorItem:
    request_id: str
    phase: Phase
    new_tokens: int
    context_tokens: int
    produces_token: bool

    def __post_init__(self) -> None:
        if self.phase not in ("Prefill", "Decode"):
            raise ValueError("predictor item 阶段无效")
        if type(self.new_tokens) is not int or self.new_tokens <= 0:
            raise ValueError("new_tokens 必须是正整数")
        if type(self.context_tokens) is not int or self.context_tokens < 0:
            raise ValueError("context_tokens 必须是非负整数")
        if type(self.produces_token) is not bool:
            raise ValueError("produces_token 必须是布尔值")
        if self.phase == "Decode" and (self.new_tokens != 1 or not self.produces_token):
            raise ValueError("Decode 每请求处理一个新 token 并产生一个输出")


@dataclass(frozen=True)
class OperatorEstimate:
    name: str
    flops: float
    memory_bytes: float
    compute_ms: float
    memory_ms: float
    duration_ms: float
    bottleneck: str


@dataclass(frozen=True)
class BatchEstimate:
    source: str
    items: tuple[PredictorItem, ...]
    duration_ms: float
    operators: tuple[OperatorEstimate, ...] = ()
    overhead_ms: float = 0.0


class Predictor(Protocol):
    def estimate(self, items: tuple[PredictorItem, ...]) -> BatchEstimate: ...


class TeachingPredictor:
    def __init__(self, timing: Timing) -> None:
        self.timing = timing

    def estimate(self, items: tuple[PredictorItem, ...]) -> BatchEstimate:
        if not items or len({item.phase for item in items}) != 1:
            raise ValueError("教学 Predictor 仅支持非空的单阶段 batch")
        # 【重点1：兼容旧实验】Decode整轮固定耗时，不能乘以成员数量。
        duration = (sum(item.new_tokens for item in items) * self.timing.prefill_ms_per_input_token
                    if items[0].phase == "Prefill" else self.timing.decode_ms_per_step)
        return BatchEstimate("teaching_assumption", items, duration)


class RooflinePredictor:
    def __init__(self, model: ModelProfile, device: DeviceProfile, calibration: Calibration) -> None:
        self.model, self.device, self.calibration = model, device, calibration

    def estimate(self, items: tuple[PredictorItem, ...]) -> BatchEstimate:
        if not items:
            raise ValueError("不能预测空 batch")
        m = self.model
        # 【重点2：工作量】Prefill上下文是已完成输入；Decode包含已写入KV的输入/输出。
        tokens = sum(item.new_tokens for item in items)
        logits = sum(item.produces_token for item in items)
        scores = sum(item.new_tokens * (item.context_tokens + (item.new_tokens + 1) / 2)
                     for item in items)
        head_dim = m.hidden_size // m.num_attention_heads
        kv_width = m.num_kv_heads * head_dim
        # Dense/GQA + SwiGLU近似：QKV、输出投影、三次MLP矩阵乘；不支持MoE/MLA。
        matrices = m.num_layers * (
            m.hidden_size * (m.hidden_size + 2 * kv_width)
            + m.hidden_size**2 + 3 * m.hidden_size * m.intermediate_size)
        dense_flops = 2 * tokens * matrices + 2 * logits * m.hidden_size * m.vocab_size
        # 【重点3：组批共享】权重每batch按读取一次估计，不能逐请求重复算。
        dense_bytes = (matrices + (m.hidden_size * m.vocab_size if logits else 0)) * m.weight_bytes_per_element
        dense_bytes += 2 * tokens * m.hidden_size * m.num_layers * m.activation_bytes_per_element
        attention_flops = 4 * m.num_attention_heads * head_dim * m.num_layers * scores
        # 简化KV流量：每成员已有KV读一次，新KV写一次；忽略缓存和分块重读。
        attention_bytes = 2 * kv_width * m.num_layers * m.kv_bytes_per_element * sum(
            item.context_tokens + item.new_tokens for item in items)
        operators = (
            self._operator("dense", dense_flops, dense_bytes, self.calibration.dense_factor),
            self._operator("attention", attention_flops, attention_bytes, self.calibration.attention_factor),
        )
        duration = sum(op.duration_ms for op in operators) + self.calibration.batch_overhead_ms
        _validate_time("预测耗时", duration, allow_zero=False)
        return BatchEstimate("uncalibrated_roofline", items, duration, operators,
                             self.calibration.batch_overhead_ms)

    def _operator(self, name: str, flops: float, memory_bytes: float, factor: float) -> OperatorEstimate:
        # 【重点4：时间公式】组内取max，组间相加；不是把整个batch只取一次max。
        compute = factor * flops / self.device.peak_flops * 1000
        memory = factor * memory_bytes / self.device.memory_bandwidth_bytes_per_s * 1000
        if not all(math.isfinite(value) for value in (compute, memory)):
            raise ValueError("预测算子耗时不是有限值")
        return OperatorEstimate(name, flops, memory_bytes, compute, memory, max(compute, memory),
                                "compute" if compute >= memory else "memory")


def build_predictor(settings: PredictorSettings, timing: Timing) -> Predictor:
    if settings.kind == "teaching":
        return TeachingPredictor(timing)
    assert settings.model is not None and settings.device is not None and settings.calibration is not None
    return RooflinePredictor(settings.model, settings.device, settings.calibration)
