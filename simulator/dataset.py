"""离线准备 ShareGPT；分词与事件模拟分开，模拟时不需要 tokenizer。"""

from collections import Counter
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import random
import statistics
from typing import Iterator, Protocol

from .models import Request


class LengthTokenizer(Protocol):
    """这里只依赖长度接口，测试可以用小型替身，不加载真实模型。"""

    def lengths(self, prompt: str, answer: str) -> tuple[int, int]: ...


class ChatTokenizer:
    def __init__(self, path: Path) -> None:
        from transformers import AutoTokenizer

        # 只读本地文件，禁止隐式联网和执行模型仓库的自定义代码。
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False
        )
        if not self.tokenizer.chat_template:
            raise ValueError("tokenizer 必须包含 chat_template")

    def lengths(self, prompt: str, answer: str) -> tuple[int, int]:
        # 【重点 1】输入包含模型的角色/生成模板；回答只数文本 token，不加 EOS。
        # Qwen 模板可能自动补系统提示，因此不能用字符数代替输入长度。
        prompt_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True
        )
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)
        return len(prompt_ids), len(answer_ids)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_sharegpt(path: Path) -> Iterator[object]:
    import ijson

    # 【重点 2】流式读取顶层数组，避免一次载入约 640MiB 的原始文本。
    with path.open("rb") as stream:
        if next((byte.decode("ascii") for byte in iter(lambda: stream.read(1), b"")
                 if not byte.isspace()), None) != "[":
            raise ValueError("ShareGPT 文件必须是顶层 JSON 数组")
        stream.seek(0)
        yield from ijson.items(stream, "item")


def first_pair(row: object) -> tuple[str, str] | None:
    if not isinstance(row, dict):
        return None
    turns = row.get("conversations")
    if not isinstance(turns, list) or len(turns) < 2:
        return None
    user, assistant = turns[:2]
    if not isinstance(user, dict) or not isinstance(assistant, dict):
        return None
    if user.get("from") != "human" or assistant.get("from") != "gpt":
        return None
    prompt, answer = user.get("value"), assistant.get("value")
    if not isinstance(prompt, str) or not isinstance(answer, str):
        return None
    if not prompt.strip() or not answer.strip():
        return None
    return prompt, answer


def prepare_requests(
    rows: Iterator[object], tokenizer: LengthTokenizer, *, count: int,
    max_input_tokens: int, max_output_tokens: int,
    arrival_mode: str, requests_per_second: float, seed: int,
) -> tuple[list[dict], dict]:
    """选前 count 个有效请求，不截断文本；seed 只控制泊松到达。"""
    for name, value in (("count", count), ("max_input_tokens", max_input_tokens),
                        ("max_output_tokens", max_output_tokens)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} 必须是正整数")
    if type(seed) is not int:
        raise ValueError("seed 必须是整数")
    if arrival_mode not in ("fixed", "poisson"):
        raise ValueError("arrival.mode 必须是 fixed 或 poisson")
    if isinstance(requests_per_second, bool) or not isinstance(requests_per_second, (int, float)):
        raise ValueError("requests_per_second 必须是数值")
    if not math.isfinite(requests_per_second) or requests_per_second <= 0:
        raise ValueError("requests_per_second 必须是有限正数")

    rejected: Counter[str] = Counter()
    selected = []
    source_rows = []
    scanned = 0
    rng = random.Random(seed)
    arrival_ms = 0.0
    for index, row in enumerate(rows):
        scanned += 1
        pair = first_pair(row)
        if pair is None:
            rejected["invalid_first_pair"] += 1
            continue
        input_length, output_length = tokenizer.lengths(*pair)
        if input_length <= 0 or output_length <= 0:
            rejected["empty_tokens"] += 1
            continue
        if input_length > max_input_tokens or output_length > max_output_tokens:
            rejected["length_limit"] += 1
            continue

        # 【重点 3】负载到达与 GPU 执行无关；首请求在 0ms，之后按间隔到达。
        if selected:
            gap_s = (1 / requests_per_second if arrival_mode == "fixed"
                     else rng.expovariate(requests_per_second))
            arrival_ms += gap_s * 1000
        request = Request(f"sharegpt-{index:06d}", arrival_ms, input_length, output_length)
        selected.append(asdict(request))
        source_rows.append(index)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"有效请求不足：需要 {count}，得到 {len(selected)}；跳过 {dict(rejected)}")

    def describe(key: str) -> dict:
        lengths = sorted(row[key] for row in selected)
        return {"min": lengths[0], "max": lengths[-1], "mean": statistics.mean(lengths),
                "median": statistics.median(lengths),
                "p95": lengths[math.ceil(0.95 * len(lengths)) - 1]}

    return selected, {
        "selection": "first_valid", "scanned_records": scanned,
        "selected_count": len(selected), "source_row_indices": source_rows,
        "rejected_in_scanned_prefix": dict(rejected),
        "length_statistics": {key: describe(key) for key in ("input_tokens", "output_tokens")},
    }
