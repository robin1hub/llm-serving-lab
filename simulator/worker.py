"""每张逻辑卡的不可变输入；可复用配置，不复用运行状态。"""

from dataclasses import dataclass

from .kv_cache import KVConfig
from .memory_budget import MemoryBudget
from .models import Timing
from .predictor import PredictorSettings
from .recovery import RecoveryConfig
from .scheduler import SchedulerProfile


@dataclass(frozen=True)
class WorkerConfig:
    """【多卡重点1】描述一张Mix/P/D逻辑卡；角色组合由运行入口进一步校验。"""

    worker_id: str
    gpu_id: str
    scheduler_profile: SchedulerProfile
    timing: Timing = Timing()
    predictor_settings: PredictorSettings = PredictorSettings()
    kv_config: KVConfig | None = None
    recovery_config: RecoveryConfig = RecoveryConfig()
    profile_name: str = "legacy"
    role: str = "Mix"
    memory_budget: MemoryBudget | None = None
    memory_provenance: dict | None = None

    def __post_init__(self) -> None:
        for name in ("worker_id", "gpu_id", "profile_name"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"worker.{name}必须是非空字符串")
        self.scheduler_profile.validate_role(self.role)
        if self.kv_config is not None and self.kv_config.gpu_id != self.gpu_id:
            raise ValueError("worker.gpu_id与其KV池gpu_id必须一致")
        if self.recovery_config.capacity_policy == "preempt" and self.kv_config is None:
            raise ValueError("抢占恢复需要该worker自己的KV池")
