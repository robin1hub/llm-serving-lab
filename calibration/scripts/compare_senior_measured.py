"""只读调用师兄native Predictor，和同一实测矩阵/原始基线做三方对照。"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from calibration.scripts.compare_measured import compare_rows
from simulator.predictor import BatchEstimate, PredictorItem


def replay_requests(items: tuple[PredictorItem, ...], prompt_lengths: dict[str, int]) -> tuple[dict, ...]:
    """将实际工作量转成师兄的范围结构，不把增长的context误当原输入长度。"""
    if not items or len({item.phase for item in items}) != 1 or items[0].phase not in ("Prefill", "Decode"):
        raise ValueError("本次只对照纯Prefill/Decode")
    return tuple({"request_ordinal": index, "phase": item.phase.lower(),
                  "context_length": item.context_tokens, "range_start": item.context_tokens,
                  "range_end": item.context_tokens + item.new_tokens,
                  "prompt_end": prompt_lengths[item.request_id],
                  "logits_token_count": int(item.produces_token), "suppress_kv_write": False}
                 for index, item in enumerate(items))


class SeniorReplayPredictor:
    def __init__(self, prompt_lengths: dict[str, int], replay: Callable[[tuple[dict, ...]], float]) -> None:
        self.prompt_lengths, self.replay = prompt_lengths, replay
        self.cache: dict[str, float] = {}

    def estimate(self, items: tuple[PredictorItem, ...]) -> BatchEstimate:
        requests = replay_requests(items, self.prompt_lengths)
        # 【重点1】成本由工作量决定，与请求ID无关；相同形状复用，但不同context独立预测。
        key = json.dumps(requests, sort_keys=True)
        if key not in self.cache:
            duration = self.replay(requests)
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("师兄预测器返回无效耗时")
            self.cache[key] = duration
        return BatchEstimate("senior_native_default_analytic", items, self.cache[key])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--senior-root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    senior, run, baseline_path, out = (p.resolve() for p in
                                      (args.senior_root, args.run, args.baseline, args.output))
    baseline = json.loads(baseline_path.read_text())
    if not (run / "COMPLETE").is_file() or baseline["fit_performed"]:
        raise ValueError("要求完整实测和未拟合基线")
    # 【重点2】先核对同一份原始文件，禁止把两次实测混在一起比较。
    for name, digest in baseline["raw_artifact_sha256"].items():
        if hashlib.sha256((run / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"实测身份不符：{name}")
    metadata = json.loads((run / "metadata.json").read_text())
    if metadata["record_status"] != "measured" or metadata["run_id"] != baseline["run_id"]:
        raise ValueError("实测状态或运行身份不符")
    rows = [json.loads(line) for line in (run / "batches.jsonl").read_text().splitlines()]
    prompt_lengths: dict[str, int] = {}
    for row in rows:
        length = int(row["shape_id"].split("-")[2][1:])
        for item in row["items"]:
            request_id = item["request_id"]
            if request_id in prompt_lengths and prompt_lengths[request_id] != length:
                raise ValueError("同一请求对应不同原输入长度")
            prompt_lengths[request_id] = length
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(out)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(senior / "src"), str(senior / ".venv/lib/python3.13/site-packages")]
    # 与前次相同：-S关闭editable构建钩子，只读加载已有二进制。
    binary = senior / "build/cmake/python-editable/_native.cpython-313-x86_64-linux-gnu.so"
    spec = importlib.util.spec_from_file_location("router_simulator._native", binary)
    if spec is None or spec.loader is None:
        raise ValueError("缺少师兄native模块")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    import yaml
    from router_simulator.calibration.replay import IterationShape, replay_shape
    from router_simulator.config.loader import ConfigLoader

    model, device = metadata["model_config"], baseline["predictor"]["device"]
    raw = yaml.safe_load((senior / "examples/native-smoke.yaml").read_text())
    raw["simulation_name"] = "lab-qwen25-measured-shape-replay"
    raw["model_config"] = {
        "model_name": "Qwen2.5-3B-Instruct", "num_layers": model["num_hidden_layers"],
        "hidden_size": model["hidden_size"], "intermediate_size": model["intermediate_size"],
        "num_attention_heads": model["num_attention_heads"], "num_key_value_heads": model["num_key_value_heads"],
        "head_dim": model["hidden_size"] // model["num_attention_heads"], "vocab_size": model["vocab_size"],
        "dtype": "bfloat16", "kv_cache_dtype": "bfloat16", "tie_word_embeddings": model["tie_word_embeddings"],
    }
    raw["cluster"]["nodes"][0]["devices"] = [{
        "device_id": "gpu-0", "device_profile_name": device["name"], "hbm_capacity_bytes": 25757220864,
        "hbm_bandwidth_bytes_per_s": device["memory_bandwidth_bytes_per_s"],
        "peak_flops_by_dtype": {"bfloat16": device["peak_flops"]},
        "nvlink_bandwidth_bytes_per_s": 1e9, "pcie_bandwidth_bytes_per_s": 1e9}]
    raw["predictor"] = {"calibration_params": {}}
    raw["kv_cache"]["enable_prefix_cache"] = False
    raw["arrival"]["dataset_path"] = str(senior / "examples/workloads/native-smoke.jsonl")
    raw["workload"] = {"cache_dir": str(out / "workload_cache")}
    raw["output"]["output_dir"] = str(out / "unused_session")
    config_path = out / "senior_config.yaml"
    config_path.write_text(yaml.safe_dump(raw, allow_unicode=True))
    config = ConfigLoader(config_path).load()

    def replay(requests: tuple[dict, ...]) -> float:
        shape = IterationShape("observed", "gpu-0", requests[0]["phase"], 0, 0,
                               model["num_hidden_layers"], requests)
        return replay_shape(config, shape).duration_s * 1000

    predictor = SeniorReplayPredictor(prompt_lengths, replay)
    summaries, details = compare_rows(rows, predictor)
    own = {row["shape_id"]: row for row in baseline["shapes"]}
    if {row["shape_id"] for row in summaries} != set(own):
        raise ValueError("两边形状集合不符")
    for row in summaries:
        previous = own[row["shape_id"]]
        for key in ("repetitions", "worker_rounds", "measured_median_ms", "excluded", "context_min", "context_max"):
            if row[key] != previous[key]:
                raise ValueError(f"两边统计口径不符：{key}")
        row["lab_predicted_median_ms"] = previous["predicted_median_ms"]
        row["lab_relative_error"] = previous["relative_error"]
    report = {"schema_version": 1, "kind": "senior_analytic_vs_same_hardware_measurements",
              "run_id": metadata["run_id"], "fit_performed": False,
              "raw_artifact_sha256": baseline["raw_artifact_sha256"],
              "baseline_report_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
              "native_binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
              "senior_source_head": subprocess.check_output(["git", "-C", str(senior), "rev-parse", "HEAD"], text=True).strip(),
              "native_matches_source_head": "not_proven", "native_config": raw,
              "mode": "immutable_batch_replay_no_des_or_request_latency",
              "unique_replayed_shapes": len(predictor.cache), "matched_worker_rounds": len(details),
              "timing_scope_measured": "worker_execute_wall",
              "shapes": summaries,
              "senior_mean_absolute_shape_error": sum(abs(row["relative_error"]) for row in summaries) / len(summaries),
              "senior_max_absolute_shape_error": max(abs(row["relative_error"]) for row in summaries),
              "lab_mean_absolute_shape_error": baseline["mean_absolute_shape_error"]}
    (out / "comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    with (out / "matched_rounds.jsonl").open("w") as stream:
        for row in details:
            stream.write(json.dumps(row) + "\n")
    print(json.dumps({key: report[key] for key in ("senior_mean_absolute_shape_error", "senior_max_absolute_shape_error",
                                                 "matched_worker_rounds", "unique_replayed_shapes")}))


if __name__ == "__main__":
    main()
