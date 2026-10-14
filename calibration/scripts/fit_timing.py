"""只用fit形状拟合三个非负参数；标准库实现，不读取新测试结果调参。"""

import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from calibration.scripts.compare_measured import compare_rows
from simulator.config import load_experiment
from simulator.predictor import build_predictor


def solve_linear(matrix: list[list[float]], target: list[float]) -> list[float] | None:
    """最多3阶，带主元消元；不可识别的有效集合跳过，不伪造解。"""
    size = len(target)
    augmented = [list(row) + [value] for row, value in zip(matrix, target)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda index: abs(augmented[index][column]))
        if abs(augmented[pivot][column]) < 1e-14:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for index in range(size):
            if index != column:
                factor = augmented[index][column]
                augmented[index] = [a - factor * b for a, b in zip(augmented[index], augmented[column])]
    return [row[-1] for row in augmented]


def fit_nonnegative(features: list[list[float]], targets: list[float]) -> list[float]:
    """最小化各形状相对误差平方和；两个factor>0、开销>=0。"""
    if not features or len(features) != len(targets) or any(y <= 0 or not math.isfinite(y) for y in targets):
        raise ValueError("拟合需要正且有限的目标值")
    if any(len(row) != 3 or any(not math.isfinite(x) or x < 0 for x in row) for row in features):
        raise ValueError("需要三个非负且有限的基函数")
    lower = [1e-9, 1e-9, 0.0]
    scaled = [[x / y for x in row] for row, y in zip(features, targets)]
    adjusted = [1 - sum(x * b for x, b in zip(row, lower)) for row in scaled]
    candidates = []
    for size in range(4):
        for active in itertools.combinations(range(3), size):
            normal = [[sum(row[i] * row[j] for row in scaled) for j in active] for i in active]
            rhs = [sum(row[i] * y for row, y in zip(scaled, adjusted)) for i in active]
            solution = solve_linear(normal, rhs) if size else []
            if solution is None or any(x < 0 for x in solution):
                continue
            beta = lower.copy()
            for index, value in zip(active, solution):
                beta[index] += value
            loss = sum((sum(x * b for x, b in zip(row, beta)) - 1) ** 2 for row in scaled)
            candidates.append((loss, beta))
    return min(candidates, key=lambda candidate: candidate[0])[1]


def training_basis(summaries: list[dict], details: list[dict]) -> tuple[list[list[float]], list[float], list[str]]:
    features, targets, ids = [], [], []
    for shape in summaries:
        # 【重点1】明确过滤fit；原validation和未来fresh测试都不能进入求解。
        if shape["split"] != "fit":
            continue
        samples: dict[str, list[list[float]]] = {}
        for row in details:
            if row["shape_id"] == shape["shape_id"]:
                if len(row["basis_ms"]) != 2:
                    raise ValueError("当前只拟合dense与attention两组")
                samples.setdefault(row["sample_id"], []).append(row["basis_ms"])
        features.append([median(median(row[index] for row in sample) for sample in samples.values())
                         for index in range(2)] + [1.0])
        targets.append(shape["measured_median_ms"])
        ids.append(shape["shape_id"])
    return features, targets, ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.profile.exists():
        raise ValueError("使用全新输出和profile，保留原始拟合身份")
    original = json.loads(args.baseline_report.read_text())
    for name, digest in original["raw_artifact_sha256"].items():
        if hashlib.sha256((args.run / name).read_bytes()).hexdigest() != digest:
            raise ValueError("原始训练证据SHA不符")
    config = load_experiment(args.config)
    if asdict(config.predictor_settings) != original["predictor"]:
        raise ValueError("拟合配置与原始基线身份不符")
    rows = [json.loads(line) for line in (args.run / "batches.jsonl").read_text().splitlines()]
    summaries, details = compare_rows(rows, build_predictor(config.predictor_settings, config.timing))
    features, targets, ids = training_basis(summaries, details)
    if len(ids) != 12 or any(row["repetitions"] != 20 for row in summaries):
        raise ValueError("要求原12组fit及20有效重复")
    beta = fit_nonnegative(features, targets)
    parameters = dict(zip(("dense_factor", "attention_factor", "batch_overhead_ms"), beta))
    profile = json.loads(args.config.read_text())
    profile["predictor"]["calibration"] = parameters
    profile["predictor"]["device"]["name"] = "4090-Qwen25-3B-vllm011-eager-worker-wall-fitted"
    args.profile.parent.mkdir(parents=True, exist_ok=True)
    with args.profile.open("x", encoding="utf-8") as stream:
        json.dump(profile, stream, indent=2)
        stream.write("\n")
    # 【重点2】参数冻结之后才评估原validation；后续新采集也只读这个profile。
    calibrated = load_experiment(args.profile)
    evaluated, _ = compare_rows(rows, build_predictor(calibrated.predictor_settings, calibrated.timing))
    baseline = {row["shape_id"]: row for row in summaries}
    for row in evaluated:
        row["baseline_relative_error"] = baseline[row["shape_id"]]["relative_error"]
        row["baseline_predicted_ms"] = baseline[row["shape_id"]]["predicted_median_ms"]
    report = {"schema_version": 1, "parameters": parameters, "training_shape_ids": ids,
              "objective": "equal_shape_relative_squared_error", "fit_shape_count": 12,
              "validation_shape_count": 4, "source_run_id": original["run_id"],
              "raw_artifact_sha256": original["raw_artifact_sha256"],
              "model_config_sha256": original["model_config_sha256"],
              "software": original["software"], "effective_engine": original["effective_engine"],
              "timing_scope": original["timing_scope"], "profile_sha256": hashlib.sha256(args.profile.read_bytes()).hexdigest(),
              "scope": "Qwen2.5-3B BF16 TP1 RTX4090 vLLM0.11 eager synchronized Worker wall",
              "shapes": evaluated}
    args.output.mkdir(parents=True)
    (args.output / "fit_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"parameters": parameters, "validation_errors":
                      [row["relative_error"] for row in evaluated if row["split"] == "validation"]}))


if __name__ == "__main__":
    main()
