"""配置边界：把 JSON 转成经过校验的请求、策略和耗时。"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import cast

from .models import Request, Timing
from .dataset import sha256_file
from .scheduler import Policy, SchedulerConfig, SchedulerProfile


@dataclass(frozen=True)
class Experiment:
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
    # 【重点 1：先读配置，不让执行内核解析 JSON。】
    config_path = path.resolve()
    raw = _fields(_read_json(config_path), {"workload", "timing"}, "config",
                  {"scheduler", "scheduler_profiles", "workers"})
    workload = _fields(raw["workload"], {"path"}, "workload")
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
            raise ValueError("当前只支持一个逻辑 worker；多 worker 需要后续路由和交接")
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
    timing = _fields(raw["timing"],
                     {"prefill_ms_per_input_token", "decode_ms_per_step"}, "timing")
    if not isinstance(workload["path"], str) or not workload["path"].strip():
        raise ValueError("workload.path 必须是非空路径字符串")
    policy = profile.mix_algorithm

    # 【重点 2：负载相对路径以配置文件所在目录为基准，不依赖运行目录。】
    workload_path = (config_path.parent / workload["path"]).resolve()
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
    # 【重点 3：复用原 Request / Timing 的校验，返回不可变实验输入。】
    return Experiment(tuple(requests), Timing(**timing),
                      cast(Policy, policy),
                      config_path, workload_path, metadata, scheduler_config.max_batch_requests,
                      scheduler_config.max_tokens_per_batch, scheduler_config.prefill_chunk_tokens,
                      profile, profile_name, worker_id, worker_role)
