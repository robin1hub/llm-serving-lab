"""受控混合试跑：先让Decode就绪，再插入一个Prefill，按实际执行核验。"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess

from batch_records import match_controlled_mixed, select_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-existing-memory-mib", type=int, default=64)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("显存预算比例不合法")
    args.output.mkdir(parents=True, exist_ok=False)
    if importlib.metadata.version("vllm") != "0.11.0":
        raise ValueError("只核对vLLM0.11.0引擎接口")
    snapshot = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                                        "--format=csv,noheader"], text=True)
    (args.output / "gpu_before.txt").write_text(snapshot)
    index = select_device(snapshot, args.gpu_uuid, args.max_existing_memory_mib)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(index), VLLM_USE_V1="1", HF_HUB_OFFLINE="1", VLLM_NO_USAGE_STATS="1")
    from vllm import LLM, SamplingParams

    settings = dict(model=str(args.model), dtype="bfloat16", tensor_parallel_size=1,
                    gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=4096,
                    max_num_seqs=8, max_num_batched_tokens=4096, block_size=16, enforce_eager=True,
                    enable_prefix_caching=False, enable_chunked_prefill=True,
                    worker_cls="worker_probe.TimedWorker", seed=20261015, disable_log_stats=True)
    cases = [(d, p) for d in (1, 3, 7) for p in (128, 512, 1536)]
    warmups, repeats = 5, 20
    if args.smoke:
        cases, warmups, repeats = [(1, 128)], 1, 2
    metadata = {"schema_version": 1, "record_status": "started", "run_id": args.output.name,
                "captured_at_utc": datetime.now(timezone.utc).isoformat(), "gpu_uuid": args.gpu_uuid,
                "shared_gpu_trial": args.max_existing_memory_mib > 64,
                "max_existing_memory_mib": args.max_existing_memory_mib,
                "engine": settings, "timing_scope": "worker_execute_wall", "purpose": "mixed_validation_no_refit",
                "model_config": json.loads((args.model / "config.json").read_text()),
                "hashes": {name: hashlib.sha256((args.model / name).read_bytes()).hexdigest()
                           for name in ("config.json", "model.safetensors.index.json", "tokenizer.json")},
                "software": {name: importlib.metadata.version(name) for name in ("vllm", "torch", "transformers")},
                "matrix": {"cases": cases, "warmups": warmups, "repeats": repeats,
                           "resident_input_tokens": 512, "resident_output_tokens": 16, "prefill_output_tokens": 1}}
    meta_path = args.output / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    llm = LLM(**settings)
    engine = llm.llm_engine
    capacity = llm.collective_rpc("calibration_capacity")[0]
    metadata["effective_engine"] = capacity["effective_engine"]
    meta_path.write_text(json.dumps(metadata, indent=2))
    capacity.update(schema_version=1, record_status="measured", run_id=args.output.name,
                    capacity_source="TimedWorker reads initialized cache_config", source_artifact="startup.log")
    (args.output / "capacity.json").write_text(json.dumps(capacity, indent=2))
    with (args.output / "batches.jsonl").open("w") as rows_file, (args.output / "generation_checks.jsonl").open("w") as checks:
        for n_decode, prefill_tokens in cases:
            shape_id = f"Mixed-D{n_decode}-P{prefill_tokens}"
            for repeat in range(warmups + repeats):
                sample_id = f"{shape_id}-{repeat}"
                sample = {"run_id": args.output.name, "shape_id": shape_id, "sample_id": sample_id,
                          "split": "mixed_validation", "is_warmup": repeat < warmups, "target_phase": "Mixed"}
                llm.collective_rpc("set_calibration_sample", args=(sample,))
                residents = {f"{sample_id}-d{i}" for i in range(n_decode)}
                new_id = f"{sample_id}-p"
                for i, request_id in enumerate(sorted(residents)):
                    engine.add_request(request_id, {"prompt_token_ids": [3000 + i] * 512},
                                       SamplingParams(temperature=0, max_tokens=16, ignore_eos=True, detokenize=False))
                finished: dict[str, int] = {}

                def step() -> list[dict]:
                    for result in engine.step():
                        if result.finished:
                            finished[result.request_id] = len(result.outputs[0].token_ids)
                    return llm.collective_rpc("drain_calibration_rows")[0]

                # 【重点1】观察到全体成员已在同轮Decode，才投递新的Prefill；不改真实Scheduler。
                ready = False
                for _ in range(64):
                    records = step()
                    if any(len(row["items"]) == n_decode
                           and all(item["phase"] == "Decode" for item in row["items"]) for row in records):
                        ready = True
                        break
                    if not engine.has_unfinished_requests():
                        break
                if not ready or finished:
                    raise RuntimeError("Decode就绪前请求已结束，不能声称测到目标混合batch")
                engine.add_request(new_id, {"prompt_token_ids": [4000] * prefill_tokens},
                                   SamplingParams(temperature=0, max_tokens=1, ignore_eos=True, detokenize=False))
                for _ in range(128):
                    records = step()
                    if not engine.has_unfinished_requests():
                        break
                else:
                    raise RuntimeError("引擎未在有限轮次内完成")
                records = llm.collective_rpc("drain_calibration_rows")[0]
                # 【重点2】必须真实出现1P+nD，P是整段新输入，未把拆开的两轮相加冒充混合。
                matched = match_controlled_mixed(records, residents, new_id, prefill_tokens)
                if finished != {**{key: 16 for key in residents}, new_id: 1}:
                    raise RuntimeError(f"输出数不符合目标：{shape_id}, finished={finished}")
                for row in records:
                    rows_file.write(json.dumps(row) + "\n")
                rows_file.flush()
                checks.write(json.dumps({**sample, "matched_iteration": matched["iteration_index"],
                                         "actual_outputs": finished}) + "\n")
                checks.flush()
            print(f"Completed {shape_id}", flush=True)
    metadata["record_status"] = "measured"
    meta_path.write_text(json.dumps(metadata, indent=2))
    (args.output / "COMPLETE").write_text(datetime.now(timezone.utc).isoformat())


if __name__ == "__main__":
    main()
