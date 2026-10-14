"""在冻结参数和新采集上验证；半数复测、半数新形状，不回写拟合结果。"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from calibration.scripts.compare_measured import compare_rows
from calibration.scripts.matrix_plan import load_matrix_plan
from simulator.config import load_experiment
from simulator.predictor import build_predictor


def summarize_errors(shapes: list[dict], error_key: str) -> dict:
    values = [abs(row[error_key]) for row in shapes]
    return {"shape_count": len(values), "mean_absolute_relative_error": sum(values) / len(values),
            "max_absolute_relative_error": max(values), "all_shapes_within_20_percent": all(x <= 0.2 for x in values)}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("run", "baseline", "profile", "fit_report", "plan", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    fit = json.loads(args.fit_report.read_text())
    # 【重点1】冻结参数SHA绑定到训练报告；新数据不能偷偷触发再拟合。
    if hashlib.sha256(args.profile.read_bytes()).hexdigest() != fit["profile_sha256"]:
        raise ValueError("校准profile在冻结后被修改")
    if not (args.run / "COMPLETE").is_file():
        raise ValueError("新采集尚未完成")
    metadata = json.loads((args.run / "metadata.json").read_text())
    plan = load_matrix_plan(args.plan)
    if (metadata["record_status"] != "measured" or metadata["experiment_matrix"] != plan
            or metadata["experiment_matrix_sha256"] != hashlib.sha256(args.plan.read_bytes()).hexdigest()):
        raise ValueError("新采集状态或矩阵身份不符")
    if metadata["hashes"]["config.json"] != fit["model_config_sha256"] or metadata["software"] != fit["software"]:
        raise ValueError("模型/框架已变化，不能套用原profile验证")
    if metadata["effective_engine"] != fit["effective_engine"] or metadata["timing_scope"] != fit["timing_scope"]:
        raise ValueError("运行参数或计时边界已变化")
    old = {row["shape_id"] for row in fit["shapes"]}
    expected = {f'{s["phase"]}-B{s["batch_size"]}-L{s["length"]}': s for s in plan["shapes"]}
    for key, shape in expected.items():
        if (key in old) != (shape["split"] == "repeat"):
            raise ValueError("复测/新形状划分与原始矩阵不符")
    checks = [json.loads(line) for line in (args.run / "generation_checks.jsonl").read_text().splitlines()]
    for key, shape in expected.items():
        sample = [row for row in checks if row["shape_id"] == key]
        tokens = 1 if shape["phase"] == "Prefill" else metadata["sampling"]["decode_output_tokens"]
        if (len(sample) != plan["warmups"] + plan["repeats"]
                or sum(row["is_warmup"] for row in sample) != plan["warmups"]
                or any(row["actual_output_tokens"] != [tokens] * shape["batch_size"] for row in sample)):
            raise ValueError("新采集重复数或输出长度不符")
    if len(checks) != len(expected) * (plan["warmups"] + plan["repeats"]):
        raise ValueError("出现未知采集形状")
    rows = [json.loads(line) for line in (args.run / "batches.jsonl").read_text().splitlines()]
    if any(row["run_id"] != metadata["run_id"] for row in rows):
        raise ValueError("轮次来自不同运行")
    baseline = load_experiment(args.baseline)
    calibrated = load_experiment(args.profile)
    if baseline.predictor_settings.model != calibrated.predictor_settings.model:
        raise ValueError("校准前后模型结构不符")
    baseline_shapes, _ = compare_rows(rows, build_predictor(baseline.predictor_settings, baseline.timing))
    shapes, details = compare_rows(rows, build_predictor(calibrated.predictor_settings, calibrated.timing))
    if {row["shape_id"] for row in shapes} != set(expected) or any(row["repetitions"] != plan["repeats"] for row in shapes):
        raise ValueError("新形状不全或有效重复不足")
    before = {row["shape_id"]: row for row in baseline_shapes}
    for row in shapes:
        original = before[row["shape_id"]]
        row["baseline_relative_error"] = original["relative_error"]
        row["baseline_predicted_ms"] = original["predicted_median_ms"]
    # 【重点2】复测、新形状分别报告，不能只给整体均值掩盖泛化失败。
    groups = {}
    for split in ("all", "repeat", "novel"):
        selected = [row for row in shapes if split == "all" or row["split"] == split]
        groups[split] = {"baseline": summarize_errors(selected, "baseline_relative_error"),
                         "calibrated": summarize_errors(selected, "relative_error")}
    artifacts = ("metadata.json", "capacity.json", "batches.jsonl", "generation_checks.jsonl", "startup.log", "COMPLETE")
    report = {"schema_version": 1, "kind": "frozen_calibration_fresh_half_novel_validation",
              "run_id": metadata["run_id"], "training_run_id": fit["source_run_id"],
              "refit_on_fresh_data": False, "scope": fit["scope"], "profile_sha256": fit["profile_sha256"],
              "parameters": fit["parameters"], "predictor": asdict(calibrated.predictor_settings),
              "matrix": plan, "groups": groups, "shapes": shapes,
              "raw_worker_rounds": len(rows), "generation_checks": len(checks),
              "software": metadata["software"], "effective_engine": metadata["effective_engine"],
              "raw_artifact_sha256": {name: hashlib.sha256((args.run / name).read_bytes()).hexdigest() for name in artifacts}}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "validation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    with (args.output / "matched_rounds.jsonl").open("w") as stream:
        for row in details:
            stream.write(json.dumps(row) + "\n")
    print(json.dumps(groups))


if __name__ == "__main__":
    main()
