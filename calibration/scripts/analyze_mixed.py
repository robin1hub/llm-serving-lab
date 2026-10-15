"""用冻结的纯P/D系数验证已有混合轮次；不重新拟合，不冒充请求级对照。"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simulator.config import load_experiment
from simulator.predictor import Predictor, PredictorItem, build_predictor
from calibration.scripts.batch_records import match_controlled_mixed


def validate_controlled_run(metadata: dict, rows: list[dict], checks: list[dict]) -> None:
    """独立验收每次投递，包括预热；未完成或少测的矩阵不能混进正式报告。"""
    matrix = metadata["matrix"]
    expected = {}
    for count, tokens in matrix["cases"]:
        shape = f"Mixed-D{count}-P{tokens}"
        for repeat in range(matrix["warmups"] + matrix["repeats"]):
            sample = f"{shape}-{repeat}"
            expected[sample] = (shape, count, tokens, repeat < matrix["warmups"])
    if len(checks) != len(expected) or {c["sample_id"] for c in checks} != set(expected):
        raise ValueError("受控矩阵缺少或重复生成验收记录")
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["sample_id"], []).append(row)
    if set(grouped) != set(expected):
        raise ValueError("实际执行记录与矩阵不一致")
    for check in checks:
        sample = check["sample_id"]
        shape, count, tokens, warm = expected[sample]
        residents = {f"{sample}-d{i}" for i in range(count)}
        new_id = f"{sample}-p"
        if (check["run_id"] != metadata["run_id"] or check["shape_id"] != shape
                or check["is_warmup"] != warm
                or check["actual_outputs"] != {**{key: 16 for key in residents}, new_id: 1}):
            raise ValueError("受控生成输出或样本身份不符")
        for row in grouped[sample]:
            if row["shape_id"] != shape or row["is_warmup"] != warm:
                raise ValueError("执行轮次标签与生成验收不一致")
        match = match_controlled_mixed(grouped[sample], residents, new_id, tokens)
        if check["matched_iteration"] != match["iteration_index"]:
            raise ValueError("验收轮次与实际混合执行不一致")


def controlled_target_rows(rows: list[dict], checks: list[dict]) -> list[dict]:
    """只比较注入P之后验收的目标轮；驻留请求入场也可能出现其他混合轮。"""
    targets = {(check["sample_id"], check["matched_iteration"]) for check in checks}
    return [row for row in rows if (row["sample_id"], row["iteration_index"]) in targets]


def summarize(records: list[dict]) -> dict:
    samples: dict[str, list[dict]] = {}
    for row in records:
        samples.setdefault(row["sample_id"], []).append(row)
    # 每次生成内的轮次相关，先取中位，再按独立生成重复等权。
    values = {key: median(median(row[key] for row in sample) for sample in samples.values())
              for key in ("measured_ms", "baseline_ms", "calibrated_ms")}
    return {**values, "repetitions": len(samples), "mixed_worker_rounds": len(records),
            "sample_median_min_ms": min(median(row["measured_ms"] for row in sample) for sample in samples.values()),
            "sample_median_max_ms": max(median(row["measured_ms"] for row in sample) for sample in samples.values()),
            "baseline_relative_error": values["baseline_ms"] / values["measured_ms"] - 1,
            "calibrated_relative_error": values["calibrated_ms"] / values["measured_ms"] - 1,
            "decode_context_min": min(row["decode_context_min"] for row in records),
            "decode_context_max": max(row["decode_context_max"] for row in records)}


def analyze_rows(rows: list[dict], baseline: Predictor, calibrated: Predictor) -> tuple[list[dict], list[dict], list[dict]]:
    details = []
    for row in rows:
        if row["is_warmup"]:
            continue
        phases = {item["phase"] for item in row["items"]}
        if phases != {"Prefill", "Decode"}:
            continue
        if (row["record_status"] != "measured" or row["timing_scope"] != "worker_execute_wall"
                or row["preemptions"] or not math.isfinite(row["duration_ms"]) or row["duration_ms"] <= 0):
            raise ValueError("混合测量状态/边界/时间不合法或含恢复")
        items = tuple(PredictorItem(**item) for item in row["items"])
        if len(items) != row["actual_batch_requests"]:
            raise ValueError("实际成员数不符")
        prefill = [item for item in items if item.phase == "Prefill"]
        decode = [item for item in items if item.phase == "Decode"]
        # 【重点1】一整批只调用一次estimate，不把单独P时间与D时间相加。
        details.append({"source_case": row["shape_id"], "sample_id": row["sample_id"],
                        "iteration_index": row["iteration_index"], "prefill_requests": len(prefill),
                        "decode_requests": len(decode), "prefill_new_tokens": sum(item.new_tokens for item in prefill),
                        "decode_context_min": min(item.context_tokens for item in decode),
                        "decode_context_max": max(item.context_tokens for item in decode),
                        "items": row["items"], "measured_ms": row["duration_ms"],
                        "baseline_ms": baseline.estimate(items).duration_ms,
                        "calibrated_ms": calibrated.estimate(items).duration_ms})
    if not details:
        raise ValueError("没有非预热混合轮次")
    cases: dict[str, list[dict]] = {}
    patterns: dict[tuple, list[dict]] = {}
    for row in details:
        cases.setdefault(row["source_case"], []).append(row)
        key = row["source_case"], row["prefill_requests"], row["decode_requests"], row["prefill_new_tokens"]
        patterns.setdefault(key, []).append(row)
    # 【重点2】原case是请求投递实验标签，不是混合成员数；另报观察到的P/D组成。
    case_summary = [{"source_case": key, **summarize(records)} for key, records in cases.items()]
    pattern_summary = [{"source_case": key[0], "prefill_requests": key[1], "decode_requests": key[2],
                        "prefill_new_tokens": key[3], **summarize(records)} for key, records in patterns.items()]
    return case_summary, pattern_summary, details


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("run", "baseline", "profile", "fit_report", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    fit = json.loads(args.fit_report.read_text())
    if hashlib.sha256(args.profile.read_bytes()).hexdigest() != fit["profile_sha256"]:
        raise ValueError("纯P/D参数在冻结后被修改")
    if not (args.run / "COMPLETE").is_file():
        raise ValueError("实测未完成")
    metadata = json.loads((args.run / "metadata.json").read_text())
    if (metadata["record_status"] != "measured" or metadata["software"] != fit["software"]
            or metadata["hashes"]["config.json"] != fit["model_config_sha256"]
            or metadata["effective_engine"] != fit["effective_engine"]):
        raise ValueError("模型/软件/有效参数与已校准范围不符")
    if fit.get("gpu_uuid") is not None and metadata["gpu_uuid"] != fit["gpu_uuid"]:
        raise ValueError("混合验证与纯P/D拟合不在同一张卡")
    rows = [json.loads(line) for line in (args.run / "batches.jsonl").read_text().splitlines()]
    if any(row["run_id"] != metadata["run_id"] for row in rows):
        raise ValueError("实测记录来自不同运行")
    controlled = metadata.get("purpose") == "mixed_validation_no_refit"
    if controlled:
        checks = [json.loads(line) for line in (args.run / "generation_checks.jsonl").read_text().splitlines()]
        validate_controlled_run(metadata, rows, checks)
        rows = controlled_target_rows(rows, checks)
    before, after = load_experiment(args.baseline), load_experiment(args.profile)
    if before.predictor_settings.model != after.predictor_settings.model:
        raise ValueError("基线和校准模型不一致")
    cases, patterns, details = analyze_rows(rows, build_predictor(before.predictor_settings, before.timing),
                                           build_predictor(after.predictor_settings, after.timing))
    overall = {"source_case_count": len(cases), "mixed_worker_rounds": len(details),
               "independent_generations": len({row["sample_id"] for row in details}),
               "baseline_mean_absolute_case_error": sum(abs(row["baseline_relative_error"]) for row in cases) / len(cases),
               "calibrated_mean_absolute_case_error": sum(abs(row["calibrated_relative_error"]) for row in cases) / len(cases),
               "calibrated_max_absolute_case_error": max(abs(row["calibrated_relative_error"]) for row in cases)}
    report = {"schema_version": 1, "kind": "controlled_mixed_with_frozen_pure_PD_profile" if controlled
              else "existing_mixed_transitions_with_frozen_pure_PD_profile",
              "run_id": metadata["run_id"], "refit_performed": False, "profile_sha256": fit["profile_sha256"],
              "gpu_uuid": metadata["gpu_uuid"], "software": metadata["software"],
              "model_config_sha256": metadata["hashes"]["config.json"],
              "effective_engine": metadata["effective_engine"], "calibration_scope": fit["scope"],
              "controlled_matrix": metadata.get("matrix") if controlled else None,
              "selection_rule": "validated matched_iteration per sample; warmups excluded" if controlled
              else "all observed non-warm mixed transitions",
              "parameters": fit["parameters"], "timing_scope": metadata["timing_scope"],
              "overall": overall, "source_cases": cases, "composition_patterns": patterns,
              "pattern_grouping": "source case + P/D counts + total P tokens; exact items used for each prediction",
              "raw_artifact_sha256": {name: hashlib.sha256((args.run / name).read_bytes()).hexdigest()
                                      for name in (("metadata.json", "batches.jsonl", "COMPLETE", "generation_checks.jsonl", "capacity.json", "startup.log")
                                                   if controlled else ("metadata.json", "batches.jsonl", "COMPLETE"))}}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "mixed_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    with (args.output / "matched_rounds.jsonl").open("w") as stream:
        for row in details:
            stream.write(json.dumps(row) + "\n")
    print(json.dumps(overall))


if __name__ == "__main__":
    main()
