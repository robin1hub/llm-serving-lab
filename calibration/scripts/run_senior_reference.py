"""同形状参照实验：借用师兄native core，只写本项目结果，不修改师兄仓库。"""

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--senior-root", type=Path, required=True)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    senior = args.senior_root.resolve()
    scenario_path = args.scenario.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    # 师兄日志使用当前目录；将默认日志也收进本次实验，避免污染任一仓库。
    os.chdir(out)
    lab = Path(__file__).resolve().parents[2]
    sys.dont_write_bytecode = True
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # 【重点1】-S不执行editable导入钩子；已有Python依赖/native二进制只读使用。
    sys.path[:0] = [str(lab), str(senior / "src"),
                   str(senior / ".venv/lib/python3.13/site-packages")]
    binary = senior / "build/cmake/python-editable/_native.cpython-313-x86_64-linux-gnu.so"
    spec = importlib.util.spec_from_file_location("router_simulator._native", binary)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    import numpy as np
    import yaml
    from router_simulator.calibration.replay import IterationShape, replay_shape
    from router_simulator.config.loader import ConfigLoader
    from router_simulator.experiments.io import iter_detail_records, read_json_object, read_jsonl_objects
    from router_simulator.output.native_writer import NativeOutputWriter
    from router_simulator.session import run_native_session
    from simulator.engine import simulate_events
    from simulator.models import Request, Timing
    from simulator.predictor import Calibration, DeviceProfile, ModelProfile, PredictorItem, RooflinePredictor

    scenario = json.loads(scenario_path.read_text())
    m, d = scenario["model"], scenario["device"]
    own_model = ModelProfile(m["num_hidden_layers"], m["hidden_size"], m["intermediate_size"],
                             m["num_attention_heads"], m["num_key_value_heads"], m["vocab_size"])
    own_device = DeviceProfile(d["name"], d["peak_flops"], d["memory_bandwidth_bytes_per_s"])
    base_predictor = RooflinePredictor(own_model, own_device, Calibration(1, 1, 0))
    # 原示例只取一个请求，避免选批算法/共享/恢复差异混入执行模型比较。
    fixture = json.loads((senior / "examples/workloads/native-smoke.jsonl").read_text().splitlines()[0])
    (out / "one_request.jsonl").write_text(json.dumps(fixture) + "\n")
    raw = {
        "simulation_name": "lab-reference-single-request", "seed": 42, "native_thread_count": 1,
        "model_config": {"model_name": "Qwen3-8B", "num_layers": m["num_hidden_layers"],
                         "hidden_size": m["hidden_size"], "intermediate_size": m["intermediate_size"],
                         "num_attention_heads": m["num_attention_heads"],
                         "num_key_value_heads": m["num_key_value_heads"], "vocab_size": m["vocab_size"],
                         "head_dim": m["head_dim"], "dtype": "bfloat16", "kv_cache_dtype": "bfloat16",
                         "tie_word_embeddings": m["tie_word_embeddings"]},
        "cluster": {"cross_node_bandwidth_bytes_per_s": 1e9, "nodes": [{"node_id": "node-0",
                    "devices": [{"device_id": "gpu-0", "device_profile_name": d["name"],
                    "hbm_capacity_bytes": d["memory_capacity_bytes"],
                    "hbm_bandwidth_bytes_per_s": d["memory_bandwidth_bytes_per_s"],
                    "peak_flops_by_dtype": {"bfloat16": d["peak_flops"]},
                    "nvlink_bandwidth_bytes_per_s": 1e9, "pcie_bandwidth_bytes_per_s": 1e9}]}]},
        "services": [{"service_id": "mix-0", "node_id": "node-0", "role": "Mix",
                      "token_budget": 4096, "chunk_size": 4096, "device_ids": ["gpu-0"],
                      "parallel_config": {"tp_size": 1, "pp_size": 1}}],
        "router": {"strategy": "round_robin"},
        "scheduler": {"recipe": "old_policy", "max_batch_size": 8, "max_tokens_per_batch": 4096,
                      "iteration_interval_s": 1e-6, "prefill_delay_min_tokens": 1},
        "predictor": {"calibration_params": scenario["senior_calibration_params"]},
        "memory": {"hbm_utilization": 0.9, "fixed_runtime_reserve_bytes": 2147483648},
        "kv_cache": {"block_size": 16, "max_blocks_per_worker": 65536, "enable_prefix_cache": False},
        "arrival": {"mode": "use_dataset", "dataset_path": str(out / "one_request.jsonl")},
        "workload": {"cache_dir": str(out / "workload_cache")},
        "output": {"output_dir": str(out / "senior_session")},
        "observability": {"retain_operator_breakdown": True},
    }
    config_file = out / "senior_config.yaml"
    config_file.write_text(yaml.safe_dump(raw, allow_unicode=True))
    config = ConfigLoader(config_file).load()
    snapshot = run_native_session(config).snapshot
    NativeOutputWriter(snapshot, out / "senior_session").write_all()
    # 【重点3】先验收完整输出，再比较。完整标记、文件SHA和明细计数缺一不可。
    session = out / "senior_session"
    manifest = read_json_object(session / "manifest.json")
    if not manifest["complete"]:
        raise ValueError("参照输出未完成")
    for artifact in manifest["files"]:
        content = (session / artifact["path"]).read_bytes()
        if len(content) != artifact["byte_size"] or hashlib.sha256(content).hexdigest() != artifact["checksum_sha256"]:
            raise ValueError("参照输出文件校验失败")
    detail_counts = {category: sum(1 for _ in iter_detail_records(session, category))
                     for category in {part["category"] for part in manifest["detail_parts"]}}
    read_json_object(session / "metrics.json")
    read_json_object(session / "trace.json")
    request_rows = read_jsonl_objects(session / "requests.jsonl")
    if len(request_rows) != 1 or not request_rows[0]["completed"] or request_rows[0]["actual_output_token_count"] != fixture["output_len"]:
        raise ValueError("请求完成状态或输出token数不符")
    own_run = simulate_events([Request(fixture["request_id"], fixture["arrival_time"] * 1000,
                                      fixture["prompt_len"], fixture["output_len"])],
                              Timing(2, 2), predictor=base_predictor)
    (out / "lab_session.json").write_text(json.dumps(asdict(own_run), indent=2))

    cases = [("Prefill", 1, n, "fit") for n in (128, 512, 2048)]
    cases += [("Prefill", 1, n, "validation") for n in (256, 1024)]
    cases += [("Decode", b, n, "fit") for b in (1, 4, 8) for n in (128, 512, 2048)]
    cases += [("Decode", 2, n, "validation") for n in (256, 1024)]
    rows = []
    for phase, batch_size, length, split in cases:
        context, count = (0, length) if phase == "Prefill" else (length, 1)
        items = tuple(PredictorItem(f"r-{i}", phase, count, context, True) for i in range(batch_size))
        shape = IterationShape(f"{phase}-B{batch_size}-L{length}", "gpu-0", phase.lower(), 0,
                              0, m["num_hidden_layers"], tuple({
                                  "request_ordinal": i, "phase": phase.lower(),
                                  "context_length": context, "range_start": context,
                                  "range_end": context + count,
                                  "prompt_end": length, "logits_token_count": 1,
                                  "suppress_kv_write": False,
                              } for i in range(batch_size)))
        reference = replay_shape(config, shape)
        estimate = base_predictor.estimate(items)
        rows.append({"shape_id": shape.iteration_id, "split": split, "items": [asdict(i) for i in items],
                     "reference_ms": reference.duration_s * 1000, "baseline_ms": estimate.duration_ms,
                     "basis_ms": [op.duration_ms for op in estimate.operators] + [1],
                     "reference_logical_operators": reference.logical_operator_count,
                     "reference_auxiliary_flops": reference.auxiliary_flops})

    # 【重点2】只用fit形状做参照拟合；留出的validation形状不参与调参。
    train = [row for row in rows if row["split"] == "fit"]
    x = np.array([row["basis_ms"] for row in train])
    y = np.array([row["reference_ms"] for row in train])
    lower = np.array([1e-9, 1e-9, 0.0])
    a, target = x / y[:, None], (y - x @ lower) / y
    choices = []
    for size in range(1, 4):
        for active in itertools.combinations(range(3), size):
            solution, _, rank, _ = np.linalg.lstsq(a[:, active], target, rcond=None)
            if rank != size or np.any(solution < 0):
                continue
            beta = lower.copy()
            beta[list(active)] += solution
            choices.append((float(np.sum(((x @ beta - y) / y) ** 2)), beta))
    if not choices:
        raise ValueError("没有可行的参照拟合参数")
    _, beta = min(choices, key=lambda pair: pair[0])
    factors = Calibration(*map(float, beta))
    fitted = RooflinePredictor(own_model, own_device, factors)
    for row in rows:
        estimate = fitted.estimate(tuple(PredictorItem(**item) for item in row["items"]))
        row["fitted_ms"] = estimate.duration_ms
        row["baseline_relative_error"] = row["baseline_ms"] / row["reference_ms"] - 1
        row["fitted_relative_error"] = row["fitted_ms"] / row["reference_ms"] - 1
    provenance = {
        "kind": "synthetic_simulator_reference_not_hardware_calibration", "scenario": scenario,
        "senior_git_commit": subprocess.check_output(["git", "-C", str(senior), "rev-parse", "HEAD"], text=True).strip(),
        "native_binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "native_binary_path": str(binary), "native_binary_matches_source_commit": "not_proven",
        "native_reported_source_commit": manifest["source_commit"],
        "output_audit": {"complete": True, "schema_version": manifest["schema_version"],
                         "checksums_verified": True, "detail_record_counts": detail_counts},
        "single_request_comparison": {
            "input_tokens": fixture["prompt_len"], "output_tokens": fixture["output_len"],
            "senior_token_times_ms": [t * 1000 for t in request_rows[0]["decode_token_completion_times_s"]],
            "lab_token_times_ms": list(own_run.batch_run.results[0].token_times_ms),
            "senior_scheduler_interval_ms": 0.001, "prefill_delay_min_tokens": 1,
        },
        "factors": asdict(factors), "rows": rows,
        "default_predictor_changed": False, "hardware_measured": False,
        "validation_max_abs_error": max(abs(r["fitted_relative_error"]) for r in rows if r["split"] == "validation"),
    }
    (out / "reference_comparison.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps({"termination": snapshot.termination_reason, "completed": snapshot.completed_request_count,
                      "factors": asdict(factors), "validation_max_abs_error": provenance["validation_max_abs_error"]}))


if __name__ == "__main__":
    main()
