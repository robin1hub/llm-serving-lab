# LLM Serving 学习实验

通过小步实验学习请求、batch、事件与调度。当前支持单卡、多个独立Mix worker与固定1P1D交接；默认耗时为教学假设，限定范围的实测参数需显式选择。

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
                       worker.py 每张逻辑卡的不可变配置
                       handoff.py 固定1P1D的FIFO链路与KV交接
calibration/           校准方案、采集入口、参照与硬件实测摘要
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
python3 run_simulation.py --config configs/two_mix_workers.json --output output/two_mix_workers.json
python3 run_simulation.py --config configs/one_p_one_d.json --output output/one_p_one_d.json
python3 -m unittest discover -s tests -q
```

Python 3.10+，基础模拟仅标准库。ShareGPT 数据准备使用本项目 .venv 的独立依赖，见第 5 章。默认结果 output/result.json，再次运行会覆盖同名文件。

## 按章节学习

研究入口：[D侧高压力调度论文与实验线索](docs/READING.md)，区分准入、D路由、迁移和资源配置；运行实现仍按ROADMAP推进。

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
16. [师兄模拟器参照](docs/chapters/16-reference-comparison.md)：同形状预测、参照拟合与来源。
17. [完整硬件实测对照](docs/chapters/17-measured-comparison.md)：已有模型、实际batch匹配与原始预测误差。
18. [拟合与新形状验证](docs/chapters/18-calibration-validation.md)：冻结参数、半数新形状、适用范围。
19. [混合轮次验证](docs/chapters/19-mixed-validation.md)：已有219混合轮次、实际组成、验证限制。
20. [A6000受控混合验证](docs/chapters/20-a6000-mixed.md)：换硬件后的独立纯P/D拟合、1P+多D实验与验收。
21. [多worker事件模拟](docs/chapters/21-multi-workers.md)：两张Mix卡并行、独立状态、共享事件堆与固定归属。
22. [固定1P1D与传输](docs/chapters/22-fixed-pd.md)：首token、KV载荷、FIFO链路、目标就绪与D排队。

现在先读[第22章](docs/chapters/22-fixed-pd.md)：P在8ms产生首token，KV在8–13ms传输，D在15/17ms输出。路线第2阶段已完成，目标容量不足时的等待协议仍待第3阶段。第21章的独立Mix例子保留：A在卡0于8/10/12ms输出，B在卡1于4/6ms输出。详细路线见[ROADMAP](docs/ROADMAP.md)。

此前完成[第20章：A6000受控混合验证](docs/chapters/20-a6000-mixed.md)。直接使用A6000本地Qwen2.5-3B基础版，独立纯P/D拟合后冻结参数，9组1P+1/3/7D共180个目标轮的平均误差28.80%→5.64%，最大10.35%；未用混合数据调参。纯P/D的4组留出平均8.08%，P256仍高估21.40%。GPU已释放，没有拷贝/下载权重，环境、报告、代码讲解和交接均已保存。该结论只覆盖本次模型/硬件/同步eager执行轮，不代表请求级TTFT或所有混合组合。默认配置不变，当前契约见[ARCHITECTURE](docs/ARCHITECTURE.md)，历史4090结果见第18/19章。

当前支持ShareGPT单轮负载、人数/token预算、Prefill分块、阶段交替、混合batch、KV分页准入、Mix抢占/重算，以及独立多Mix与固定1P1D。PD暂不支持抢占或目标容量等待。4090/A6000校准仅覆盖对应报告范围，多卡请求级和链路精度未验证。完整历史见[学习日志](docs/LEARNING_LOG.md)，继续工作先读[交接](docs/HANDOFF.md)、[架构契约](docs/ARCHITECTURE.md)、[路线](docs/ROADMAP.md)和AGENTS.md。
