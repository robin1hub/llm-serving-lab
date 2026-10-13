"""核对最小GPU试跑并汇总实际阶段；不进行拟合或发布校准profile。"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median


def summarize(run: Path) -> dict:
    """只接收完整smoke记录；保留共享条件与原始证据SHA。"""
    if not (run / "COMPLETE").is_file():
        raise ValueError("缺少COMPLETE，不能把启动尝试当完成测量")
    metadata = json.loads((run / "metadata.json").read_text())
    capacity = json.loads((run / "capacity.json").read_text())
    rows = [json.loads(line) for line in (run / "batches.jsonl").read_text().splitlines()]
    checks = [json.loads(line) for line in (run / "generation_checks.jsonl").read_text().splitlines()]
    if metadata["record_status"] != "measured" or capacity["record_status"] != "measured":
        raise ValueError("记录尚未完成")
    if len(checks) != 6 or len(rows) != 12 or any(row["split"] != "smoke" for row in rows):
        raise ValueError("本工具只验收两个形状、各一次预热两次采样的smoke")
    for check in checks:
        expected = 1 if check["target_phase"] == "Prefill" else 3
        if check["actual_output_tokens"] != [expected]:
            raise ValueError("请求输出长度与smoke目标不一致")
    for row in rows:
        if (row["record_status"] != "measured" or row["run_id"] != metadata["run_id"]
                or row["timing_scope"] != "worker_execute_wall"
                or not math.isfinite(row["duration_ms"]) or row["duration_ms"] <= 0
                or row["actual_batch_requests"] != 1 or len(row["items"]) != 1):
            raise ValueError("轮次状态、身份、计时边界或人数不符")
    summaries = []
    for phase, expected_count in (("Prefill", 2), ("Decode", 4)):
        # 【重点】Decode实验也先执行Prefill；按实际阶段筛选，不能把它算成Decode。
        selected = [row for row in rows if not row["is_warmup"]
                    and row["target_phase"] == phase and row["items"][0]["phase"] == phase]
        if len(selected) != expected_count:
            raise ValueError("有效执行轮次数不符合smoke计划")
        times = [row["duration_ms"] for row in selected]
        summaries.append({"phase": phase, "batch_requests": 1, "sample_count": len(times),
                          "new_tokens": sorted({row["items"][0]["new_tokens"] for row in selected}),
                          "context_tokens": sorted({row["items"][0]["context_tokens"] for row in selected}),
                          "median_ms": median(times), "min_ms": min(times), "max_ms": max(times)})
    artifacts = ("metadata.json", "capacity.json", "batches.jsonl", "generation_checks.jsonl",
                 "gpu_before.txt", "startup.log", "COMPLETE")
    return {"schema_version": 1, "kind": "hardware_smoke_trial_not_calibrated_profile",
            "run_id": metadata["run_id"], "complete": True, "record_status": "measured",
            "shared_gpu_trial": metadata["shared_gpu_trial"], "gpu_uuid": metadata["gpu_uuid"],
            "software": metadata["software"], "model_config_sha256": metadata["hashes"]["config.json"],
            "engine": metadata["engine"], "timing_scope": "worker_execute_wall",
            "capacity": capacity, "batches": summaries, "raw_rows": len(rows),
            "generation_checks": len(checks), "fit_performed": False,
            "raw_artifact_sha256": {name: hashlib.sha256((run / name).read_bytes()).hexdigest()
                                    for name in artifacts}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(result["batches"], ensure_ascii=False))


if __name__ == "__main__":
    main()
