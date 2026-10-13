# LLM Serving 学习实验

通过小步实验学习请求、batch、事件与调度。当前只保留单卡事件模拟器，耗时仍为教学假设。

## 当前目录

```text
run_simulation.py       唯一运行入口
prepare_workload.py     离线数据准备入口
simulator/             models.py 数据结构、config.py 配置、engine.py 事件内核
                       metrics.py 结果汇总
                       scheduler.py 只读选批与人数上限
                       predictor.py 独立耗时预测与计算明细
                       kv_cache.py KV块身份、请求表、准入与生命周期
                       memory_budget.py 显存字节预算换算KV块数
                       recovery.py 受害者选择、暂停与重算恢复状态
calibration/           校准方案与空白记录模板（未实测）
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
7. [Scheduler 与选批](docs/chapters/07-scheduler.md)：候选计划、请求数上限和 Decode 轮转。
8. [Token 预算与分块](docs/chapters/08-token-budget.md)：流程图、状态表、数据结构和代码阅读顺序。
9. [Worker 与调度方案](docs/chapters/09-worker-profiles.md)：独立算法槽、角色引用、完成反馈与阶段交替。
10. [Predictor 耗时预测](docs/chapters/10-predictor.md)：工作量、上下文、算力/带宽、修正系数与手算。
11. [混合 batch](docs/chapters/11-mixed-batches.md)：剩余预算、同轮完成、成员阶段与耗时对照。
12. [KV块与容量准入](docs/chapters/12-kv-blocks.md)：物理块身份、请求表、预留/填充/就绪/释放与阻塞。
13. [4090显存预算](docs/chapters/13-memory-budget.md)：独立容量profile、权重/预留、字节换算与来源。

14. [抢占与重算恢复](docs/chapters/14-recompute-recovery.md)：独立策略、进展保护、恢复状态与无输出重算。

15. [单卡校准准备](docs/chapters/15-calibration-plan.md)：实验矩阵、测量边界、记录模板与独立验证。

现在主要阅读第 15 章，校准材料在 [calibration/](calibration/README.md)。目前按用户要求暂停实际校准，仅准备[服务器独立环境](calibration/server-environment.md)，尚未采集/拟合；第14章恢复例子保留。运行 `python3 run_simulation.py --config configs/recompute.json --output output/recompute.json` 查看暂停B、完成A、重算B的例子；默认仍为block，抢占需显式启用。第13章容量示例保留。运行 `python3 run_simulation.py --config configs/rtx4090_memory.json --output output/rtx4090_memory.json` 查看服务器4090＋Qwen3-8B容量示例。独立profile在configs/profiles中；1972块来自实读显存/模型元数据和90%使用比例、2GiB预留假设，尚未实际校准。原第12章人工块数实验保留。当前仍单GPU，共享/COW/传输未实现。当前契约见 [ARCHITECTURE](docs/ARCHITECTURE.md)。

当前支持 ShareGPT 单轮负载、人数/token 预算、Prefill 分块、阶段交替、混合 batch、KV分页准入与可配置抢占/重算，耗时尚未校准。后续计划：共享/COW、实际容量验证和4090耗时校准，多 worker 与 P→D 交接再单独实现。完整记录见 [学习日志](docs/LEARNING_LOG.md)，继续工作先读 [交接记录](docs/HANDOFF.md)、[架构契约](docs/ARCHITECTURE.md) 和 AGENTS.md。
