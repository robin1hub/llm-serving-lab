"""检查独立环境的依赖；不加载模型、不分配GPU或测量推理耗时。"""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    if Path(sys.prefix).resolve() != root / "venv":
        raise ValueError("必须使用本项目venv，避免把其他环境误记成准备完成")
    # 【重点1】导入只核对包和动态库；CUDA_VISIBLE_DEVICES为空，禁止触碰已有GPU任务。
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    os.environ["HF_HOME"] = str(root / "cache/hf")
    os.environ["TORCH_HOME"] = str(root / "cache/torch")
    os.environ["XDG_CACHE_HOME"] = str(root / "cache")
    sys.dont_write_bytecode = True
    import torch
    import transformers
    import vllm

    versions = {name: importlib.metadata.version(name) for name in
                ("vllm", "torch", "transformers")}
    if versions != {"vllm": "0.11.0", "torch": "2.8.0", "transformers": "4.56.1"}:
        raise ValueError("主依赖版本与本轮固定版本不符")
    for module in (torch, vllm, transformers):
        if not Path(module.__file__).resolve().is_relative_to(root / "venv"):
            raise ValueError("导入路径落在独立环境之外，需要检查PYTHONPATH")

    check = subprocess.run([str(root / "tools/bin/uv"), "pip", "check", "--python", sys.executable],
                           capture_output=True, text=True, check=False)
    freeze = subprocess.run([str(root / "tools/bin/uv"), "pip", "freeze", "--python", sys.executable],
                            capture_output=True, text=True, check=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_name("requirements-resolved.txt").write_text(freeze.stdout)
    report = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "dependencies_ready" if check.returncode == 0 else "dependency_conflict",
        "project_root": str(root), "python": sys.version, "python_executable": sys.executable,
        "packages": versions,
        "module_locations": {"torch": torch.__file__, "vllm": vllm.__file__,
                             "transformers": transformers.__file__},
        "torch_cuda_build": torch.version.cuda, "dependency_check_exit": check.returncode,
        "dependency_check_output": check.stdout + check.stderr,
        "free_disk_bytes": shutil.disk_usage(root).free,
        "gpu_runtime_tested": False, "model_loaded": False, "calibration_measured": False,
        "notes": ["包导入与依赖检查通过不等于模型已经能运行；待空闲GPU再验证"],
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))
    if check.returncode:
        raise SystemExit(check.returncode)


if __name__ == "__main__":
    main()
