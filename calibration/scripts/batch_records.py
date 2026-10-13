"""采集前的身份检查与工作量描述；仅标准库，可在本地验证。"""

from typing import Any


def select_device(snapshot: str, requested_uuid: str, max_existing_memory_mib: int = 64) -> int:
    """核对UUID及显式允许的既有占用，返回旧vLLM支持的数字卡号。"""
    if type(max_existing_memory_mib) is not int or max_existing_memory_mib < 0:
        raise ValueError("允许的既有显存占用必须是非负整数")
    for line in snapshot.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 4 or fields[1] != requested_uuid:
            continue
        used_mib = int(fields[2].split()[0])
        utilization = int(fields[3].split()[0])
        if used_mib > max_existing_memory_mib or utilization > 0:
            raise RuntimeError(f"候选卡已有占用：{used_mib}MiB，利用率{utilization}%")
        return int(fields[0])
    raise ValueError("未找到指定GPU UUID")


def describe_work(scheduler_output: Any, prompt_lengths: dict[str, int]) -> list[dict[str, Any]]:
    """用本轮真实上下文描述工作，不用最终目标输出长度推算进度。"""
    contexts: dict[str, int] = {}
    for item in scheduler_output.scheduled_new_reqs:
        prompt_lengths[item.req_id] = len(item.prompt_token_ids)
        contexts[item.req_id] = item.num_computed_tokens
    cached = scheduler_output.scheduled_cached_reqs
    if any(cached.resumed_from_preemption):
        raise RuntimeError("初轮耗时拟合不接受抢占恢复样本")
    contexts.update(zip(cached.req_ids, cached.num_computed_tokens))
    items = []
    for request_id, count in scheduler_output.num_scheduled_tokens.items():
        context = contexts[request_id]
        prompt_len = prompt_lengths[request_id]
        if context < 0 or count <= 0:
            raise ValueError("实际执行范围必须为非负上下文和正token数")
        phase = "Prefill" if context < prompt_len else "Decode"
        if phase == "Decode" and count != 1:
            raise ValueError("当前采集不支持投机Decode或多token Decode")
        items.append({"request_id": request_id, "phase": phase,
                      "new_tokens": count, "context_tokens": context,
                      "produces_token": context + count >= prompt_len})
    return items
