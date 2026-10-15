"""配置边界：把 JSON 转成经过校验的请求、策略和耗时。"""

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import cast

from .models import Request, Timing
from .dataset import sha256_file
from .scheduler import Policy, SchedulerConfig, SchedulerProfile
from .predictor import Calibration, DeviceProfile, ModelProfile, PredictorSettings
from .kv_cache import KVConfig
from .memory_budget import MemoryBudget, MemoryDevice, MemoryModel, MemorySettings, resolve_budget
from .recovery import RecoveryConfig
from .worker import WorkerConfig


@dataclass(frozen=True)
class Experiment:
    """多卡以workers/assignments为准；旧单卡字段保留为首worker兼容视图。"""

    requests: tuple[Request, ...]
    timing: Timing
    policy: Policy
    config_path: Path
    workload_path: Path
    workload_metadata: dict | None = None
    max_batch_requests: int | None = None
    max_tokens_per_batch: int | None = None
    prefill_chunk_tokens: int | None = None
    scheduler_profile: SchedulerProfile | None = None
    profile_name: str = "legacy"
    worker_id: str = "worker_0"
    worker_role: str = "Mix"
    predictor_settings: PredictorSettings = PredictorSettings()
    kv_config: KVConfig | None = None
    memory_budget: MemoryBudget | None = None
    memory_provenance: dict | None = None
    recovery_config: RecoveryConfig = RecoveryConfig()
    workers: tuple[WorkerConfig, ...] = ()
    assignments: tuple[tuple[str, str], ...] = ()


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """拒绝重复键，避免 JSON 后面的值悄悄覆盖前面的值。"""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 重复字段：{key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"JSON 不接受非有限数：{value}")


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"),
                          object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (OSError, ValueError) as error:
        raise ValueError(f"读取 {path} 失败：{error}") from error


def _fields(value: object, expected: set[str], location: str, optional: set[str] | None = None) -> dict:
    """未知字段也报错，让拼写错误在运行前被发现。"""
    if not isinstance(value, dict):
        raise ValueError(f"{location} 必须是对象")
    missing = expected - value.keys()
    unknown = value.keys() - expected - (optional or set())
    if missing or unknown:
        raise ValueError(f"{location}：缺少字段 {sorted(missing)}；未知字段 {sorted(unknown)}")
    return value


def load_experiment(path: Path) -> Experiment:
    """【多卡重点4】逐卡解析资源；旧单卡字段保留，多卡执行必须使用workers。"""
    config_path = path.resolve()
    raw = _fields(_read_json(config_path), {"workload"}, "config",
                  {"timing", "predictor", "kv_cache", "recovery", "scheduler",
                   "scheduler_profiles", "workers", "routing"})
    if "scheduler" in raw and ("workers" in raw or "scheduler_profiles" in raw):
        raise ValueError("scheduler与scheduler_profiles/workers不能混用")
    if "predictor" in raw and isinstance(raw["predictor"], dict):
        if raw["predictor"].get("kind") == "roofline" and "timing" in raw:
            raise ValueError("roofline不能同时设置教学timing")
    entries = raw.get("workers")
    if entries is None and "scheduler" in raw:
        entries = [None]  # 旧配置在原解析器内自动产生唯一Mix worker。
    if not isinstance(entries, list) or not entries:
        raise ValueError("workers必须是非空数组")
    multiple = len(entries) > 1
    if multiple and ("kv_cache" in raw or "recovery" in raw):
        raise ValueError("多worker请将kv_cache/recovery放在各worker内，避免误用同一GPU池")
    parsed: list[WorkerConfig] = []
    first: Experiment | None = None
    for index, entry in enumerate(entries):
        local = {key: value for key, value in raw.items() if key != "routing"}
        gpu_id = None
        if entry is not None:
            spec = _fields(entry, {"id", "role", "scheduler_profile"}, f"workers[{index}]",
                           {"gpu_id", "timing", "predictor", "kv_cache", "recovery"})
            gpu_id = spec.get("gpu_id")
            if multiple and "gpu_id" not in spec:
                raise ValueError("多worker必须显式指定各自gpu_id")
            if "gpu_id" in spec and (not isinstance(gpu_id, str) or not gpu_id.strip()):
                raise ValueError("worker.gpu_id必须是非空字符串")
            local["workers"] = [{key: spec[key] for key in ("id", "role", "scheduler_profile")}]
            if "timing" in spec and "predictor" in spec:
                if not isinstance(spec["predictor"], dict) or spec["predictor"].get("kind") != "teaching":
                    raise ValueError("worker的roofline不能同时设置timing")
            # 耗时来源整体替换，不把父级roofline与子级教学timing拼在一起。
            if "timing" in spec:
                local.pop("predictor", None)
            if "predictor" in spec and isinstance(spec["predictor"], dict):
                if spec["predictor"].get("kind") == "roofline":
                    local.pop("timing", None)
            for key in ("timing", "predictor", "kv_cache", "recovery"):
                if key in spec:
                    local[key] = spec[key]
        experiment = _load_single_experiment(config_path, local, first)
        if first is None:
            first = experiment
        resolved_gpu = gpu_id if gpu_id is not None else (
            experiment.kv_config.gpu_id if experiment.kv_config is not None else "gpu_0")
        parsed.append(WorkerConfig(
            experiment.worker_id, resolved_gpu, experiment.scheduler_profile,
            experiment.timing, experiment.predictor_settings, experiment.kv_config,
            experiment.recovery_config, experiment.profile_name, experiment.worker_role,
            experiment.memory_budget, experiment.memory_provenance))
    if len({worker.worker_id for worker in parsed}) != len(parsed):
        raise ValueError("worker.id必须唯一")
    if len({worker.gpu_id for worker in parsed}) != len(parsed):
        raise ValueError("当前各worker必须使用不同gpu_id")
    assert first is not None
    if "routing" in raw:
        routing = _fields(raw["routing"], {"policy", "assignments"}, "routing")
        if routing["policy"] != "fixed" or not isinstance(routing["assignments"], dict):
            raise ValueError("当前routing只支持fixed与显式assignments对象")
        assignments = routing["assignments"]
    elif multiple:
        raise ValueError("多worker必须显式配置固定routing.assignments")
    else:
        assignments = {request.request_id: parsed[0].worker_id for request in first.requests}
    if set(assignments) != {request.request_id for request in first.requests}:
        raise ValueError("固定归属表必须恰好覆盖全部请求，不能遗漏或多写")
    worker_ids = {worker.worker_id for worker in parsed}
    if any(not isinstance(owner, str) or owner not in worker_ids for owner in assignments.values()):
        raise ValueError("固定归属表引用了不存在的worker")
    return replace(first, workers=tuple(parsed), assignments=tuple(assignments.items()))


def _load_single_experiment(path: Path, raw_config: dict,
                            workload_source: Experiment | None = None) -> Experiment:
    # 【重点 1：先读配置，不让执行内核解析 JSON。】
    config_path = path.resolve()
    raw = _fields(raw_config, {"workload"}, "config",
                  {"timing", "predictor", "kv_cache", "recovery", "scheduler", "scheduler_profiles", "workers"})
    workload = _fields(raw["workload"], {"path"}, "workload")
    recovery = _fields(raw.get("recovery", {"capacity_policy": "block"}), {"capacity_policy"},
                       "recovery", {"victim_policy", "recovery_action"})
    recovery_config = RecoveryConfig(**recovery)
    kv_config = None
    memory_budget = None
    memory_provenance = None
    if "kv_cache" in raw:
        # 【显存重点】人工块数和独立预算profile二选一；换算完复用原BlockPool。
        kv = _fields(raw["kv_cache"], {"gpu_id", "block_tokens"}, "kv_cache",
                     {"total_blocks", "memory_profile"})
        if ("total_blocks" in kv) == ("memory_profile" in kv):
            raise ValueError("kv_cache必须且只能设置total_blocks或memory_profile之一")
        if "total_blocks" in kv:
            kv_config = KVConfig(kv["gpu_id"], kv["total_blocks"], kv["block_tokens"])
        else:
            if not isinstance(kv["memory_profile"], str) or not kv["memory_profile"].strip():
                raise ValueError("memory_profile必须是非空路径")
            profile_path = (config_path.parent / kv["memory_profile"]).resolve()
            spec = _fields(_read_json(profile_path), {"device", "model", "memory"}, "memory_profile",
                           {"provenance"})
            device = _fields(spec["device"], {"name", "memory_capacity_bytes"}, "memory_profile.device")
            model = _fields(spec["model"], {"name", "num_layers", "num_kv_heads", "head_dim",
                            "kv_bytes_per_element", "weight_bytes"}, "memory_profile.model")
            settings = _fields(spec["memory"], {"utilization", "runtime_reserve_bytes"}, "memory_profile.memory",
                               {"block_metadata_bytes", "block_alignment_bytes"})
            memory_budget = resolve_budget(str(profile_path), MemoryDevice(**device), MemoryModel(**model),
                                           MemorySettings(**settings), kv["block_tokens"])
            kv_config = KVConfig(kv["gpu_id"], memory_budget.total_blocks, kv["block_tokens"])
            memory_provenance = spec.get("provenance")
            if memory_provenance is not None and not isinstance(memory_provenance, dict):
                raise ValueError("memory_profile.provenance必须是对象")
    if recovery_config.capacity_policy == "preempt" and kv_config is None:
        raise ValueError("preempt恢复策略需要kv_cache配置")
    # 【方案重点】兼容旧配置，但新旧写法不能混用，防止两个地方设置不同策略。
    if "scheduler" in raw:
        if "scheduler_profiles" in raw or "workers" in raw:
            raise ValueError("scheduler 与 scheduler_profiles/workers 不能混用")
        scheduler = _fields(raw["scheduler"], {"policy"}, "scheduler",
                            {"max_batch_requests", "max_tokens_per_batch", "prefill_chunk_tokens"})
        scheduler_config = SchedulerConfig(**scheduler)
        profile = SchedulerProfile("fcfs_chunked", "round_robin", scheduler_config.policy, scheduler_config)
        profile_name, worker_id, worker_role = "legacy", "worker_0", "Mix"
    else:
        profiles = raw.get("scheduler_profiles")
        if not isinstance(profiles, dict) or not profiles:
            raise ValueError("scheduler_profiles 必须是非空方案对象")
        parsed_profiles = {}
        for name, value in profiles.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("方案名必须非空")
            spec = _fields(value, {"constraints"}, f"profile.{name}",
                           {"prefill_algorithm", "decode_algorithm", "mix_algorithm"})
            limits = _fields(spec["constraints"], set(), f"profile.{name}.constraints",
                             {"max_batch_requests", "max_tokens_per_batch", "prefill_chunk_tokens"})
            parsed_profiles[name] = SchedulerProfile(
                spec.get("prefill_algorithm"), spec.get("decode_algorithm"), spec.get("mix_algorithm"),
                SchedulerConfig(**limits))
        workers = raw.get("workers")
        if not isinstance(workers, list) or len(workers) != 1:
            raise ValueError("内部单卡资源解析需要恰好一个已拆分worker")
        worker = _fields(workers[0], {"id", "role", "scheduler_profile"}, "workers[0]")
        worker_id, worker_role, profile_name = worker["id"], worker["role"], worker["scheduler_profile"]
        if not isinstance(worker_id, str) or not worker_id.strip():
            raise ValueError("worker.id 必须是非空字符串")
        if not isinstance(profile_name, str) or profile_name not in parsed_profiles:
            raise ValueError("worker 引用了不存在的 scheduler_profile")
        profile = parsed_profiles[profile_name]
        profile.validate_role(worker_role)
        if worker_role != "Mix":
            raise ValueError("当前完整请求运行仅支持 Mix worker；P/D 需要后续 P→D 交接机制")
        scheduler_config = profile.constraints
    # 【预测重点】算法不决定耗时来源；两个来源互斥，避免参数被悄悄忽略。
    predictor = _fields(raw.get("predictor", {"kind": "teaching"}), {"kind"}, "predictor",
                        {"model", "device", "calibration"})
    if predictor["kind"] == "teaching":
        _fields(predictor, {"kind"}, "predictor")
        timing = _fields(raw.get("timing"),
                         {"prefill_ms_per_input_token", "decode_ms_per_step"}, "timing")
        timing_config = Timing(**timing)
        predictor_settings = PredictorSettings()
    elif predictor["kind"] == "roofline":
        if "timing" in raw:
            raise ValueError("roofline 不能同时设置教学 timing")
        _fields(predictor, {"kind", "model", "device", "calibration"}, "predictor")
        model = _fields(predictor["model"],
                        {"num_layers", "hidden_size", "intermediate_size", "num_attention_heads",
                         "num_kv_heads", "vocab_size"}, "predictor.model",
                        {"weight_bytes_per_element", "activation_bytes_per_element", "kv_bytes_per_element"})
        device = _fields(predictor["device"],
                         {"name", "peak_flops", "memory_bandwidth_bytes_per_s"}, "predictor.device")
        calibration = _fields(predictor["calibration"], set(), "predictor.calibration",
                              {"dense_factor", "attention_factor", "batch_overhead_ms"})
        predictor_settings = PredictorSettings("roofline", ModelProfile(**model),
                                               DeviceProfile(**device), Calibration(**calibration))
        timing_config = Timing()  # 兼容已有Experiment字段；roofline不使用它，输出为null。
    else:
        raise ValueError("predictor.kind 必须是 teaching 或 roofline")
    if memory_budget is not None and predictor_settings.model is not None:
        # 同一模型的容量和耗时不能使用两套不同的KV结构。
        prediction_model = predictor_settings.model
        memory_model = memory_budget.model
        actual = (prediction_model.num_layers, prediction_model.num_kv_heads,
                  prediction_model.hidden_size // prediction_model.num_attention_heads,
                  prediction_model.kv_bytes_per_element)
        expected = (memory_model.num_layers, memory_model.num_kv_heads,
                    memory_model.head_dim, memory_model.kv_bytes_per_element)
        if actual != expected:
            raise ValueError("memory_profile与Predictor的KV模型结构/精度不一致")
    if not isinstance(workload["path"], str) or not workload["path"].strip():
        raise ValueError("workload.path 必须是非空路径字符串")
    policy = profile.mix_algorithm

    # 【重点 2：负载相对路径以配置文件所在目录为基准，不依赖运行目录。】
    workload_path = (config_path.parent / workload["path"]).resolve()
    if workload_source is None:
        rows = _read_json(workload_path)
        if not isinstance(rows, list):
            raise ValueError("负载文件必须是请求数组")
        requests = []
        for index, row in enumerate(rows):
            fields = _fields(row, {"request_id", "arrival_ms", "input_tokens", "output_tokens"},
                             f"workload[{index}]")
            try:
                requests.append(Request(**fields))
            except ValueError as error:
                raise ValueError(f"workload[{index}]：{error}") from error
        if len({item.request_id for item in requests}) != len(requests):
            raise ValueError("负载中的 request_id 必须唯一")

        # 生成负载的说明文件包含分词身份；校验它对应当前请求文件，避免沿用旧说明。
        manifest_path = workload_path.with_suffix(".manifest.json")
        metadata = None
        if manifest_path.exists():
            metadata = _read_json(manifest_path)
            if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
                raise ValueError("负载 manifest 版本或结构无效")
            if metadata.get("requests_sha256") != sha256_file(workload_path):
                raise ValueError("负载与 manifest 校验和不匹配，请重新准备")
            if metadata.get("selected_count") != len(requests):
                raise ValueError("负载与 manifest 请求数量不匹配")
    else:
        # 多卡共享同一份不可变负载，不重复读取/分词，也不共享执行进度。
        requests = workload_source.requests
        metadata = workload_source.workload_metadata
    # 【重点 3：复用原 Request / Timing 的校验，返回不可变实验输入。】
    return Experiment(tuple(requests), timing_config,
                      cast(Policy, policy),
                      config_path, workload_path, metadata, scheduler_config.max_batch_requests,
                      scheduler_config.max_tokens_per_batch, scheduler_config.prefill_chunk_tokens,
                      profile, profile_name, worker_id, worker_role, predictor_settings, kv_config,
                      memory_budget, memory_provenance, recovery_config)
