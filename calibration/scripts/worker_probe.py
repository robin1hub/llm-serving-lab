"""vLLM 0.11.0 单卡同步计时；独立 Worker 子类，不修改安装包源码。"""

from time import perf_counter
from typing import Any

import torch
from vllm.v1.worker.gpu_worker import Worker
from batch_records import describe_work


class TimedWorker(Worker):
    def set_calibration_sample(self, metadata: dict) -> None:
        self.calibration_sample = metadata
        self.calibration_rows: list[dict] = []
        self.calibration_prompts: dict[str, int] = {}

    def calibration_capacity(self) -> dict:
        # 【重点1】启动后实际分配的块数，与纸面显存预算分开记录。
        config = self.vllm_config.cache_config
        return {
            "reported_gpu_blocks": config.num_gpu_blocks,
            "observed_block_tokens": config.block_size,
            "reported_kv_token_capacity": config.num_gpu_blocks * config.block_size,
            "reported_kv_budget_bytes": self.available_kv_cache_memory_bytes,
            "weight_vram_bytes": int(self.model_runner.model_memory_usage),
        }

    def drain_calibration_rows(self) -> list[dict]:
        return self.calibration_rows

    def execute_model(self, scheduler_output: Any) -> Any:
        if not hasattr(self, "calibration_sample"):
            return super().execute_model(scheduler_output)
        items = describe_work(scheduler_output, self.calibration_prompts)
        # 【重点2】计时只包围本轮Worker执行；不含队列、RPC、HTTP或记录开销。
        # 同步等待GPU，防止只测到CUDA异步提交；这是受同步扰动的eager基线。
        torch.cuda.synchronize()
        started = perf_counter()
        output = super().execute_model(scheduler_output)
        torch.cuda.synchronize()
        duration_ms = (perf_counter() - started) * 1000
        if items:
            self.calibration_rows.append({
                "schema_version": 1, "record_status": "measured",
                **self.calibration_sample, "iteration_index": len(self.calibration_rows),
                "timing_scope": "worker_execute_wall", "duration_ms": duration_ms,
                "measurement_method": "perf_counter with pre/post CUDA synchronize",
                "items": items, "actual_batch_requests": len(items),
                "actual_new_tokens": sum(item["new_tokens"] for item in items),
                "prefix_cache_hit_tokens": 0, "preemptions": 0,
                "includes": ["worker_preprocess", "model_forward", "sampling", "worker_output"],
                "excludes": ["engine_scheduler", "engine_output_update", "rpc", "http", "tokenization"],
            })
        return output
