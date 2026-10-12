"""验证数据准备语义；核心测试不依赖分词库和大型原始文件。"""

from collections.abc import Iterator
import json
from pathlib import Path
import tempfile
import unittest

from simulator.config import load_experiment
from simulator.dataset import prepare_requests, sha256_file


class FakeTokenizer:
    def lengths(self, prompt: str, answer: str) -> tuple[int, int]:
        return len(prompt.split()) + 2, len(answer.split())


def conversation(prompt: str = "hello world", answer: str = "one two") -> dict:
    return {"conversations": [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]}


class DatasetTests(unittest.TestCase):
    def prepare(self, rows: Iterator[object], **overrides):
        options = dict(count=2, max_input_tokens=10, max_output_tokens=10,
                       arrival_mode="fixed", requests_per_second=2.0, seed=42)
        options.update(overrides)
        return prepare_requests(rows, FakeTokenizer(), **options)

    def test_first_pair_filter_and_fixed_arrival(self) -> None:
        rows, info = self.prepare(iter([{}, conversation(answer=""), conversation(),
                                       conversation(prompt="x " * 20), conversation()]))
        self.assertEqual([r["arrival_ms"] for r in rows], [0, 500])
        self.assertEqual([r["input_tokens"] for r in rows], [4, 4])
        self.assertEqual(info["source_row_indices"], [2, 4])
        self.assertEqual(info["rejected_in_scanned_prefix"], {"invalid_first_pair": 2, "length_limit": 1})
        self.assertNotIn("conversations", rows[0])

    def test_poisson_repeatability_and_seed(self) -> None:
        def run(seed):
            return self.prepare(iter([conversation()] * 5), count=5, arrival_mode="poisson", seed=seed)[0]
        self.assertEqual(run(42), run(42))
        self.assertNotEqual(run(42), run(43))
        self.assertTrue(all(a["arrival_ms"] < b["arrival_ms"] for a, b in zip(run(42), run(42)[1:])))

    def test_rejects_insufficient_data_and_bad_parameters(self) -> None:
        with self.assertRaises(ValueError):
            self.prepare(iter([conversation()]))
        for options in ({"count": True}, {"requests_per_second": 0}, {"seed": True},
                        {"arrival_mode": "bad"}, {"requests_per_second": float("nan")}):
            with self.assertRaises(ValueError):
                self.prepare(iter([conversation()] * 2), **options)

    def test_manifest_is_carried_and_stale_metadata_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requests, _ = self.prepare(iter([conversation()] * 2))
            workload = root / "requests.json"
            workload.write_text(json.dumps(requests))
            manifest = root / "requests.manifest.json"
            manifest.write_text(json.dumps({"schema_version": 1, "requests_sha256": sha256_file(workload),
                                            "selected_count": 2}))
            config = root / "config.json"
            config.write_text(json.dumps({"workload": {"path": "requests.json"},
                                          "scheduler": {"policy": "prefill_first"},
                                          "timing": {"prefill_ms_per_input_token": 2, "decode_ms_per_step": 2}}))
            self.assertEqual(load_experiment(config).workload_metadata["selected_count"], 2)
            workload.write_text("[]")
            with self.assertRaises(ValueError):
                load_experiment(config)

    def test_streaming_reader_when_dependencies_available(self) -> None:
        try:
            import ijson
        except ImportError:
            self.skipTest("数据准备依赖未安装；基础模拟仍可单独运行")
        from simulator.dataset import iter_sharegpt
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            source.write_text(json.dumps([conversation(), conversation()]))
            self.assertEqual(len(list(iter_sharegpt(source))), 2)
            source.write_text("{}")
            with self.assertRaises(ValueError):
                list(iter_sharegpt(source))

    def test_real_local_tokenizer_when_available(self) -> None:
        path = Path(__file__).resolve().parents[1] / "assets/tokenizers/qwen2.5-7b-instruct"
        try:
            import transformers
        except ImportError:
            self.skipTest("未安装 transformers")
        if not (path / "tokenizer.json").exists():
            self.skipTest("未提供本地 tokenizer")
        from simulator.dataset import ChatTokenizer
        adapter = ChatTokenizer(path)
        prompt_length, output_length = adapter.lengths("你好，介绍一下batch。", "多个请求可以同轮执行。")
        self.assertGreater(prompt_length, len(adapter.tokenizer.encode("你好，介绍一下batch。", add_special_tokens=False)))
        self.assertEqual(output_length, len(adapter.tokenizer.encode("多个请求可以同轮执行。", add_special_tokens=False)))


if __name__ == "__main__":
    unittest.main()
