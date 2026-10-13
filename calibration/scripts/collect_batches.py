"""实际单卡采集入口；仅在独立vLLM环境运行，不属于模拟器运行时。"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone
from time import perf_counter
from batch_records import select_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-existing-memory-mib", type=int, default=64)
    parser.add_argument("--decode-output-tokens", type=int, default=32)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("显存使用比例必须在0与1之间")
    if args.output.exists():
        raise ValueError("使用新的输出目录，避免覆盖原始记录")
    args.output.mkdir(parents=True)
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    if importlib.metadata.version("vllm") != "0.11.0":
        raise ValueError("Worker插桩只核对了vLLM0.11.0，其他版本需要重新检查")
    gpu = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader"
    ], text=True)
    (args.output / "gpu_before.txt").write_text(gpu)
    physical_index = select_device(gpu, args.gpu_uuid, args.max_existing_memory_mib)
    # 0.11.0的NVML映射按整数解析CUDA_VISIBLE_DEVICES；先核对UUID再传数字卡号。
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_index)

    from vllm import LLM, SamplingParams

    settings = dict(model=str(args.model), dtype="bfloat16", tensor_parallel_size=1,
                    gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=4096, max_num_seqs=8,
                    max_num_batched_tokens=4096, block_size=16, enforce_eager=True,
                    enable_prefix_caching=False, enable_chunked_prefill=False,
                    worker_cls="worker_probe.TimedWorker", seed=42,
                    disable_log_stats=True)
    metadata = {
        "schema_version": 1, "record_status": "started", "run_id": args.output.name,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(), "gpu_uuid": args.gpu_uuid,
        "software": {name: importlib.metadata.version(name) for name in
                     ("vllm", "torch", "transformers")},
        "engine": settings, "timing_scope": "worker_execute_wall",
        "profiling_enabled": False, "cuda_graph_mode": "eager",
        "shared_gpu_trial": args.max_existing_memory_mib > 64,
        "max_existing_memory_mib": args.max_existing_memory_mib,
        "model_config": json.loads((args.model / "config.json").read_text()),
        "hashes": {name: hashlib.sha256((args.model / name).read_bytes()).hexdigest()
                   for name in ("config.json", "model.safetensors.index.json", "tokenizer.json")},
        "sampling": {"temperature": 0, "ignore_eos": True,
                     "decode_output_tokens": 3 if args.smoke else args.decode_output_tokens},
        "notes": ["初轮使用精确token IDs合成输入；非ShareGPT或HTTP测试",
                  "worker同步计时不包含完整engine_iteration；不声称生产服务校准完成",
                  "逐轮保留实际上下文；正式Decode生成32token，允许较大batch进入稳态"],
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    llm = LLM(**settings)
    capacity = llm.collective_rpc("calibration_capacity")[0]
    capacity.update(schema_version=1, record_status="measured", run_id=args.output.name,
                    capacity_source="TimedWorker reads initialized cache_config",
                    source_artifact="startup.log")
    (args.output / "capacity.json").write_text(json.dumps(capacity, indent=2))
    shapes = [("Prefill", 1, n, split) for split, lengths in
              (("fit", (128, 512, 2048)), ("validation", (256, 1024))) for n in lengths]
    shapes += [("Decode", b, n, split) for split, batches, lengths in
               (("fit", (1, 4, 8), (128, 512, 2048)), ("validation", (2,), (256, 1024)))
               for b in batches for n in lengths]
    if args.smoke:
        shapes = [("Prefill", 1, 128, "smoke"), ("Decode", 1, 128, "smoke")]
    with (args.output / "batches.jsonl").open("w") as rows_file, \
            (args.output / "generation_checks.jsonl").open("w") as checks:
        for phase, batch_size, length, split in shapes:
            shape_id = f"{phase}-B{batch_size}-L{length}"
            warmups, repeats = (1, 2) if args.smoke else (5, 20)
            for repeat in range(warmups + repeats):
                sample = {"run_id": args.output.name, "shape_id": shape_id,
                          "sample_id": f"{shape_id}-{repeat}", "split": split,
                          "is_warmup": repeat < warmups, "target_phase": phase}
                llm.collective_rpc("set_calibration_sample", args=(sample,))
                # 【重点3】精确token数直接输入，避免字符长度被误认成token长度。
                prompts = [{"prompt_token_ids": [1000 + i] * length} for i in range(batch_size)]
                output_len = 1 if phase == "Prefill" else (3 if args.smoke else args.decode_output_tokens)
                started = perf_counter()
                outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=output_len,
                                                               ignore_eos=True, detokenize=False),
                                       use_tqdm=False)
                generation_ms = (perf_counter() - started) * 1000
                assert len(outputs) == batch_size
                assert all(len(out.outputs[0].token_ids) == output_len for out in outputs)
                for row in llm.collective_rpc("drain_calibration_rows")[0]:
                    rows_file.write(json.dumps(row) + "\n")
                rows_file.flush()
                checks.write(json.dumps({**sample, "generation_ms": generation_ms,
                                         "timing_scope": "whole_generation_wall",
                                         "actual_output_tokens": [len(out.outputs[0].token_ids)
                                                                  for out in outputs]}) + "\n")
            print(f"Completed {shape_id}", flush=True)
    (args.output / "COMPLETE").write_text(datetime.now(timezone.utc).isoformat())
    metadata["record_status"] = "measured"
    metadata["model_loaded"] = True
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
