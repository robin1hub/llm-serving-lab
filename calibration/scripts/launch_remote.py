"""服务器运行包装器：保留命令、日志与退出码，不覆盖旧结果。"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gpu-uuid", default="GPU-0f409808-9870-2765-3a7c-540b7d06c6c7")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-existing-memory-mib", type=int, default=64)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        raise ValueError("run-id必须是单个目录名")
    root = Path("/data/llm-serving-lab-calibration")
    out = root / "runs" / args.run_id
    if out.exists():
        raise ValueError("输出目录已存在")
    command = [str(root / "venv/bin/python"), str(root / "collect_batches.py"),
               "--model", "/data/HF_MODELS/Qwen3-8B", "--output", str(out),
               "--gpu-uuid", args.gpu_uuid,
               "--gpu-memory-utilization", str(args.gpu_memory_utilization),
               "--max-existing-memory-mib", str(args.max_existing_memory_mib)]
    if args.smoke:
        command.append("--smoke")
    # 所有缓存进入独立目录；卡号由采集器再次核对UUID及占用后决定。
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(root),
           "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "HF_HOME": str(root / "cache/hf"),
           "TORCH_HOME": str(root / "cache/torch"), "XDG_CACHE_HOME": str(root / "cache"),
           "VLLM_CACHE_ROOT": str(root / "cache/vllm"), "VLLM_CONFIG_ROOT": str(root / "config/vllm"),
           "HF_HUB_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1"}
    (root / "logs").mkdir(exist_ok=True)
    log = root / "logs" / f"{args.run_id}.log"
    with log.open("w", encoding="utf-8") as log_file:
        child = subprocess.Popen(command, cwd=root, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        (root / "logs" / f"{args.run_id}-launch.json").write_text(
            json.dumps({"pid": child.pid, "command": command}), encoding="utf-8")
        print(json.dumps({"pid": child.pid, "log": str(log)}), flush=True)
        code = child.wait()
    if out.exists():
        shutil.copyfile(log, out / "startup.log")
    (root / "logs" / f"{args.run_id}-exit.json").write_text(json.dumps({"exit": code}), encoding="utf-8")
    print(json.dumps({"exit": code, "log_tail": log.read_text(encoding="utf-8")[-2400:]}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
