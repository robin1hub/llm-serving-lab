"""把单GPU显存字节预算换成KV块数；不探测服务器，不部署模型。见第13章。"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR
import math


@dataclass(frozen=True)
class MemoryDevice:
    name: str
    memory_capacity_bytes: int

    def __post_init__(self) -> None:
        _name("device.name", self.name)
        _integer("memory_capacity_bytes", self.memory_capacity_bytes)


@dataclass(frozen=True)
class MemoryModel:
    name: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    kv_bytes_per_element: int
    weight_bytes: int

    def __post_init__(self) -> None:
        _name("model.name", self.name)
        for name in ("num_layers", "num_kv_heads", "head_dim", "kv_bytes_per_element", "weight_bytes"):
            _integer(name, getattr(self, name))


@dataclass(frozen=True)
class MemorySettings:
    utilization: float
    runtime_reserve_bytes: int
    block_metadata_bytes: int = 0
    block_alignment_bytes: int = 1

    def __post_init__(self) -> None:
        if (isinstance(self.utilization, bool) or not isinstance(self.utilization, (int, float))
                or not math.isfinite(self.utilization) or not 0 < self.utilization <= 1):
            raise ValueError("memory.utilization 必须是(0,1]有限数值")
        _integer("runtime_reserve_bytes", self.runtime_reserve_bytes, allow_zero=True)
        _integer("block_metadata_bytes", self.block_metadata_bytes, allow_zero=True)
        _integer("block_alignment_bytes", self.block_alignment_bytes)


def _name(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须为非空字符串")


def _integer(name: str, value: int, *, allow_zero: bool = False) -> None:
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} 必须为{'非负' if allow_zero else '正'}整数")


@dataclass(frozen=True)
class MemoryBudget:
    profile_path: str
    device: MemoryDevice
    model: MemoryModel
    settings: MemorySettings
    usable_bytes: int
    kv_budget_bytes: int
    kv_bytes_per_token: int
    block_payload_bytes: int
    block_bytes: int
    total_blocks: int
    unused_budget_bytes: int


def resolve_budget(profile_path: str, device: MemoryDevice, model: MemoryModel,
                   settings: MemorySettings, block_tokens: int) -> MemoryBudget:
    # 【重点1：先扣权重/预留】不使用当前空闲显存；其他进程占用属于部署条件。
    _integer("block_tokens", block_tokens)
    # 按配置十进制比例计算并向下取整，避免巨大字节整数经float丢失精度。
    usable = int((Decimal(device.memory_capacity_bytes) * Decimal(str(settings.utilization)))
                 .to_integral_value(rounding=ROUND_FLOOR))
    kv_budget = usable - model.weight_bytes - settings.runtime_reserve_bytes
    if kv_budget <= 0:
        raise ValueError("模型权重与运行时预留耗尽可用显存，无法分配KV")
    # 【重点2：单卡dense/GQA KV】K/V两份，所有层和KV头；不适用于MLA/滑动窗口。
    per_token = 2 * model.num_layers * model.num_kv_heads * model.head_dim * model.kv_bytes_per_element
    payload = block_tokens * per_token
    unaligned = payload + settings.block_metadata_bytes
    alignment = settings.block_alignment_bytes
    block_bytes = ((unaligned + alignment - 1) // alignment) * alignment
    # 【重点3：按整页收费】不足一整页的预算不能用于分配一个块。
    total_blocks = kv_budget // block_bytes
    if total_blocks == 0:
        raise ValueError("剩余KV预算不足一个完整块")
    return MemoryBudget(profile_path, device, model, settings, usable, kv_budget, per_token,
                        payload, block_bytes, total_blocks, kv_budget % block_bytes)
