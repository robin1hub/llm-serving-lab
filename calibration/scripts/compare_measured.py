"""按实际batch逐轮比较实测与原始Predictor；不调参，不改变默认配置。"""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simulator.config import load_experiment
from simulator.predictor import Predictor, PredictorItem, build_predictor


def compare_rows(rows: list[dict], predictor: Predictor) -> tuple[list[dict], list[dict]]:
    """每轮按真实上下文预测；只汇总指定阶段、完整人数、无恢复的纯batch。"""
    groups: dict[str, list[dict]] = {}
    rejected: dict[str, dict[str, int]] = {}
    for row in rows:
        shape_id = row["shape_id"]
        _, batch, _ = shape_id.split("-")
        expected_batch = int(batch[1:])
        counts = rejected.setdefault(shape_id, {"warmup": 0, "mixed_or_other_phase": 0,
                                                "partial_batch": 0})
        duration = row["duration_ms"]
        if row["record_status"] != "measured" or row["timing_scope"] != "worker_execute_wall":
            raise ValueError("测量状态或计时边界不符")
        if not math.isfinite(duration) or duration <= 0 or row["preemptions"]:
            raise ValueError("耗时不合法或发生恢复")
        if row["is_warmup"]:
            counts["warmup"] += 1
            continue
        items = tuple(PredictorItem(**item) for item in row["items"])
        if not items or any(item.phase != row["target_phase"] for item in items):
            counts["mixed_or_other_phase"] += 1
            continue
        if len(items) != row["actual_batch_requests"]:
            raise ValueError("实际成员数量与记录不符")
        if len(items) != expected_batch:
            counts["partial_batch"] += 1
            continue
        # 【重点1】不用shape名称中的初始长度替代实际context：Decode历史会增长。
        estimate = predictor.estimate(items)
        groups.setdefault(shape_id, []).append({
            "sample_id": row["sample_id"], "iteration_index": row["iteration_index"],
            "split": row["split"], "measured_ms": duration, "predicted_ms": estimate.duration_ms,
            "relative_error": estimate.duration_ms / duration - 1,
            "context_min": min(item.context_tokens for item in items),
            "context_max": max(item.context_tokens for item in items),
        })
    summaries, details = [], []
    for shape_id, records in groups.items():
        # 【重点2】先按一次生成实验汇总，避免把31轮Decode当31次独立重复。
        samples: dict[str, list[dict]] = {}
        for row in records:
            samples.setdefault(row["sample_id"], []).append(row)
        sample_medians = [median(row["measured_ms"] for row in sample) for sample in samples.values()]
        sample_predictions = [median(row["predicted_ms"] for row in sample) for sample in samples.values()]
        measured, predicted = median(sample_medians), median(sample_predictions)
        summaries.append({"shape_id": shape_id, "split": records[0]["split"],
                          "repetitions": len(samples), "worker_rounds": len(records),
                          "measured_median_ms": measured, "predicted_median_ms": predicted,
                          "relative_error": predicted / measured - 1,
                          "sample_median_min_ms": min(sample_medians),
                          "sample_median_max_ms": max(sample_medians),
                          "context_min": min(row["context_min"] for row in records),
                          "context_max": max(row["context_max"] for row in records),
                          "excluded": rejected[shape_id]})
        details.extend({"shape_id": shape_id, **row} for row in records)
    if set(groups) != set(rejected):
        raise ValueError("至少一个形状没有合格的完整batch")
    return summaries, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (args.run / "COMPLETE").is_file():
        raise ValueError("缺少COMPLETE，不能使用失败或未完成记录")
    metadata = json.loads((args.run / "metadata.json").read_text())
    if metadata["record_status"] != "measured":
        raise ValueError("metadata不是完成测量")
    experiment = load_experiment(args.config)
    settings = experiment.predictor_settings
    if settings.kind != "roofline":
        raise ValueError("本实验使用roofline，不将教学常数包装为硬件模型")
    model = metadata["model_config"]
    expected = {"num_layers": model["num_hidden_layers"], "hidden_size": model["hidden_size"],
                "intermediate_size": model["intermediate_size"], "num_attention_heads": model["num_attention_heads"],
                "num_kv_heads": model["num_key_value_heads"], "vocab_size": model["vocab_size"]}
    if any(getattr(settings.model, key) != value for key, value in expected.items()):
        raise ValueError("实测模型与预测模型结构不一致")
    rows = [json.loads(line) for line in (args.run / "batches.jsonl").read_text().splitlines()]
    checks = [json.loads(line) for line in (args.run / "generation_checks.jsonl").read_text().splitlines()]
    if len(checks) != 400 or any(row["run_id"] != metadata["run_id"] for row in rows):
        raise ValueError("完整矩阵的重复数或运行身份不符")
    for check in checks:
        _, batch, _ = check["shape_id"].split("-")
        expected_outputs = 1 if check["target_phase"] == "Prefill" else metadata["sampling"]["decode_output_tokens"]
        if check["actual_output_tokens"] != [expected_outputs] * int(batch[1:]):
            raise ValueError("生成输出数量不符")
    summaries, details = compare_rows(rows, build_predictor(settings, experiment.timing))
    if len(summaries) != 16 or any(row["repetitions"] != 20 for row in summaries):
        raise ValueError("缺少形状或没有达到20次有效重复")
    capacity = json.loads((args.run / "capacity.json").read_text())
    artifacts = ("metadata.json", "capacity.json", "batches.jsonl", "generation_checks.jsonl", "startup.log", "COMPLETE")
    report = {"schema_version": 1, "kind": "measured_matrix_baseline_comparison_not_fitted",
              "run_id": metadata["run_id"], "hardware_measured": True, "fit_performed": False,
              "gpu_uuid": metadata["gpu_uuid"],
              "predictor": asdict(settings), "software": metadata["software"],
              "model_config_sha256": metadata["hashes"]["config.json"],
              "capacity": capacity, "raw_worker_rounds": len(rows), "generation_checks": len(checks),
              "timing_scope": "worker_execute_wall", "shared_gpu_trial": metadata["shared_gpu_trial"],
              "requested_engine": metadata["engine"], "effective_engine": metadata.get("effective_engine"),
              "shapes": summaries,
              "mean_absolute_shape_error": sum(abs(row["relative_error"]) for row in summaries) / len(summaries),
              "max_absolute_shape_error": max(abs(row["relative_error"]) for row in summaries),
              "raw_artifact_sha256": {name: hashlib.sha256((args.run / name).read_bytes()).hexdigest() for name in artifacts}}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    with (args.output / "matched_rounds.jsonl").open("w") as stream:
        for row in details:
            stream.write(json.dumps(row) + "\n")
    print(json.dumps(summaries, ensure_ascii=False))


if __name__ == "__main__":
    main()
