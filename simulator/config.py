"""配置边界：把 JSON 转成经过校验的请求、策略和耗时。"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal, cast

from .models import Request, Timing


@dataclass(frozen=True)
class Experiment:
    requests: tuple[Request, ...]
    timing: Timing
    policy: Literal["prefill_first", "decode_first"]
    config_path: Path
    workload_path: Path


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


def _fields(value: object, expected: set[str], location: str) -> dict:
    """未知字段也报错，让拼写错误在运行前被发现。"""
    if not isinstance(value, dict):
        raise ValueError(f"{location} 必须是对象")
    missing = expected - value.keys()
    unknown = value.keys() - expected
    if missing or unknown:
        raise ValueError(f"{location}：缺少字段 {sorted(missing)}；未知字段 {sorted(unknown)}")
    return value


def load_experiment(path: Path) -> Experiment:
    # 【重点 1：先读配置，不让执行内核解析 JSON。】
    config_path = path.resolve()
    raw = _fields(_read_json(config_path), {"workload", "scheduler", "timing"}, "config")
    workload = _fields(raw["workload"], {"path"}, "workload")
    scheduler = _fields(raw["scheduler"], {"policy"}, "scheduler")
    timing = _fields(raw["timing"],
                     {"prefill_ms_per_input_token", "decode_ms_per_step"}, "timing")
    if not isinstance(workload["path"], str) or not workload["path"].strip():
        raise ValueError("workload.path 必须是非空路径字符串")
    policy = scheduler["policy"]
    if policy not in ("prefill_first", "decode_first"):
        raise ValueError("scheduler.policy 必须是 prefill_first 或 decode_first")

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

    # 【重点 3：复用原 Request / Timing 的校验，返回不可变实验输入。】
    return Experiment(tuple(requests), Timing(**timing),
                      cast(Literal["prefill_first", "decode_first"], policy),
                      config_path, workload_path)
