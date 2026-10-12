"""读取准备配置，离线分词后发布标准负载与可追溯说明。"""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import tempfile

from simulator.config import _fields, _read_json
from simulator.dataset import ChatTokenizer, iter_sharegpt, prepare_requests, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description="本地 ShareGPT 单轮负载准备")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        config_path = args.config.resolve()
        raw = _fields(_read_json(config_path),
                      {"source", "tokenizer", "selection", "arrival", "output_dir"}, "prepare")
        source = _fields(raw["source"], {"format", "path"}, "source")
        tokenizer = _fields(raw["tokenizer"], {"model_id", "path"}, "tokenizer")
        selection = _fields(raw["selection"],
                            {"count", "max_input_tokens", "max_output_tokens"}, "selection")
        arrival = _fields(raw["arrival"], {"mode", "requests_per_second", "seed"}, "arrival")
        if source["format"] != "sharegpt":
            raise ValueError("source.format 当前仅支持 sharegpt")
        if not isinstance(tokenizer["model_id"], str) or not tokenizer["model_id"].strip():
            raise ValueError("tokenizer.model_id 必须为非空说明字符串")

        def resolve(value: object) -> Path:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("路径必须是非空字符串")
            return (config_path.parent / value).resolve()

        source_path = resolve(source["path"])
        tokenizer_path = resolve(tokenizer["path"])
        output_dir = resolve(raw["output_dir"])
        if output_dir.exists():
            raise ValueError(f"输出目录已存在：{output_dir}；请换目录以保留已有实验")
        adapter = ChatTokenizer(tokenizer_path)
        rows, info = prepare_requests(
            iter_sharegpt(source_path), adapter, **selection,
            arrival_mode=arrival["mode"], requests_per_second=arrival["requests_per_second"],
            seed=arrival["seed"],
        )
        # 【重点 1】记录文件校验和、模板和依赖版本，不只记一个模型名字。
        manifest = {
            "schema_version": 1, "prepare_config": raw,
            "source_path": str(source_path), "source_sha256": sha256_file(source_path),
            "tokenizer_path": str(tokenizer_path),
            "tokenizer_files_sha256": {p.name: sha256_file(p) for p in sorted(tokenizer_path.glob("*.json"))},
            "chat_template": adapter.tokenizer.chat_template,
            "transformers_version": version("transformers"), "ijson_version": version("ijson"),
            "output_length_rule": "reference_answer_text_tokens_without_eos",
            "arrival_rule": "first_request_at_zero_then_interarrival_gaps",
            **info,
        }
        # 【重点 2】全部准备成功后再发布目录，不覆盖旧负载。
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output_dir.parent) as temp:
            staging = Path(temp)
            target = staging / "requests.json"
            target.write_text(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            manifest["requests_sha256"] = sha256_file(target)
            (staging / "requests.manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            staging.rename(output_dir)
        print(f"已准备 {len(rows)} 个请求：{output_dir / 'requests.json'}")
        print(json.dumps(info, ensure_ascii=False, indent=2))
    except (ValueError, OSError, ImportError) as error:
        parser.exit(2, f"准备失败：{error}\n")


if __name__ == "__main__":
    main()
