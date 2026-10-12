# LLM Serving 学习实验

通过小步实验学习请求、batch、事件与调度。当前只保留单卡事件模拟器，耗时仍为教学假设。

## 当前目录

```text
run_simulation.py       唯一运行入口
prepare_workload.py     离线数据准备入口
simulator/             models.py 数据结构、config.py 配置、engine.py 事件内核
                       metrics.py 结果汇总
configs/               实验配置
workloads/             请求数据
 tests/                当前实现的测试
 docs/chapters/         按章节阅读的讲解
 docs/HANDOFF.md        后续对话衔接
 docs/LEARNING_LOG.md   路线和历史记录
 output/               运行结果（不提交 Git）
```

## 运行

在 WSL2 中进入项目目录：

```bash
cd /mnt/c/Users/yh/Desktop/sim/llm-serving-lab
python3 run_simulation.py --config configs/basic.json
python3 run_simulation.py --config configs/decode_first.json --output output/decode.json
python3 -m unittest discover -s tests -q
```

Python 3.10+，基础模拟仅标准库。ShareGPT 数据准备使用本项目 .venv 的独立依赖，见第 5 章。默认结果 output/result.json，再次运行会覆盖同名文件。

## 按章节学习

1. [请求与排队](docs/chapters/01-requests.md)：Prefill、Decode、等待和延迟。
2. [Batch 与优先级](docs/chapters/02-batching.md)：组批、动态进出和等待取舍。
3. [事件队列](docs/chapters/03-events.md)：到达、唤醒、完成。
4. [配置实验](docs/chapters/04-configuration.md)：选择负载、策略和耗时。
5. [ShareGPT 负载](docs/chapters/05-sharegpt.md)：真实分词、长度过滤和到达计划。
6. [结果统计](docs/chapters/06-metrics.md)：分位数、吞吐口径与策略对照。

现在主要阅读第 5、6 章：两章已包含设计原因、函数分工、数据流和对应代码讲解。先看配置与结果，再读关键实现；事件基础可回看第 3 章。旧示例代码可通过 Git 历史查看，当前目录不再保留多套引擎。

当前可本地准备 ShareGPT 单轮负载；耗时尚未校准。后续计划：token 预算、分块与混合 batch、简单显存容量和 4090 校准。完整记录见 [学习日志](docs/LEARNING_LOG.md)，继续工作先读 [交接记录](docs/HANDOFF.md) 和 AGENTS.md。
