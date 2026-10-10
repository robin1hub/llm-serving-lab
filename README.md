# LLM Serving 学习实验

现在可用独立配置运行：`python3 run_simulation.py --config configs/basic.json`。配置、负载与结果结构的讲解见 [CONFIG_GUIDE.md](CONFIG_GUIDE.md)，事件机制讲解见 [EVENT_GUIDE.md](EVENT_GUIDE.md)。

最新一步是单卡事件驱动，运行 `python3 event_demo.py`；设计讲解见 [EVENT_GUIDE.md](EVENT_GUIDE.md)。原轮次版保留用于对照，以下学习步骤仍可单独运行。

目标是通过小段可读代码理解 serving 过程，并逐步探索 D 端压力下的调度。当前支持单请求、串行排队和单 GPU Decode batching，没有真实模型计算。

## 在 WSL2 运行

在任意位置克隆后进入仓库目录即可运行：

```bash
git clone https://github.com/robin1hub/llm-serving-lab.git
cd llm-serving-lab
python3 serial_demo.py
python3 -m unittest -v
```

仓库已公开，可以直接克隆。当前本机 WSL2 的运行路径如下：

```bash
cd /mnt/c/Users/yh/Desktop/sim/llm-serving-lab
python3 serving_lab.py
python3 serial_demo.py
python3 -m unittest -v
```

需要 Python 3.10+。只使用标准库，不需要安装依赖或使用相邻项目的虚拟环境。Windows 与 WSL2 在这里访问同一份文件。

## 第一步：先预测，再运行

默认请求 A 在 0ms 到达，输入 4 个 token，输出 3 个 token。

教学假设：GPU 初始空闲；Prefill 每个输入 token 耗时 2ms，最后产生首 token；每轮 Decode 耗时 2ms，产生一个后续 token。忽略分词、采样、网络和 CPU 开销。

所以首 token 在 8ms 产生，后两个在 10ms、12ms 产生。输出长度包含首 token，不能额外执行三轮 Decode。

## 如何阅读代码

1. `Request` 是输入，记录到达时间和长度，不包含真实 token 文本。
2. `Timing` 是教学耗时参数，与请求分开，以便之后换参数做实验。
3. `simulate()` 是本步骤的核心：等到请求到达且 GPU 可用，完成 Prefill，再循环执行剩余 Decode。`now_ms` 是虚拟时间，不是电脑的真实时钟。
4. `Event` 记录已发生的事；这里是日志记录，不是未来事件堆。
5. `Result` 保存 token 时间，并从时间记录计算指标；`main()` 只负责示例和打印。

`frozen=True` 防止执行函数修改输入或完成后的记录。局部列表在模拟过程中收集日志，返回时转为 tuple。类型标注帮助阅读；输入校验则在运行时拒绝无意义参数。

## 指标的含义

- TTFT：首 token 时间减到达时间。
- ITL：每两个相邻输出 token 的时间差。只有一个输出 token 时没有间隔，返回空元组，不记成 0ms。
- 端到端延迟：最后一个 token 时间减到达时间。

单请求在空闲 GPU 上没有排队，所以 TTFT 等于 Prefill 耗时。串行实验中，TTFT 包含等待。

## 自己改参数试一试

在 `main()` 中一次只改一个参数，先写下预测，再运行：

1. 把输出长度改为 1：还需要 Decode 吗？
2. 把到达时间改为 100ms：绝对完成时间与端到端延迟各如何变化？
3. 把输入长度改为 8：哪些 token 的时间会改变？

修改示例参数不会改变测试中的固定案例。

## 第二步：两个请求串行共享 GPU

运行 `python3 serial_demo.py`。规则是先到先服务（FCFS），同时到达保持输入顺序；一个请求从 Prefill 到生成结束独占 GPU，不抢占、不组 batch。

| 请求 | 到达 | 输入长度 | 输出长度 | 开始执行 | 输出 token 时刻 |
| --- | --- | --- | --- | --- | --- |
| A | 0ms | 4 | 3 | 0ms | 8、10、12ms |
| B | 3ms | 2 | 2 | 12ms | 16、18ms |

B 在 A 执行过程中已经到达，但只能等待。等待 12−3=9ms，Prefill 耗时 4ms，TTFT 为 16−3=13ms；端到端延迟为 18−3=15ms。

这一步的核心是 `start_ms = max(request.arrival_ms, gpu_available_ms)`。GPU 若先空闲，就等请求；请求若先到，就等 GPU。`simulate_serial()` 按到达时间排序，执行完一个请求后把最后 token 的时间作为新的 GPU 可用时间。

输入代表已知的实验 workload；这里根据串行规则直接计算时间，没有实时接入、显式队列或事件堆。日志中的到达时间仍是原始到达时间，不能把等待后的执行时间当成到达时间。`serial_demo.py` 只是把执行记录按时间合并打印，没有推进模拟状态。同一时刻按原记录顺序展示，不代表完整事件内核的优先级。

先读 `simulate()` 的开始时间计算，再读 `simulate_serial()`，最后看 `serial_demo.py`。尝试一次只修改一个参数：

1. B 的到达时间改为 20ms：GPU 中间空闲，B 的等待时间应为 0。
2. A 的输出长度改为 6：B 的输入不变，为什么 TTFT 会增大？
3. B 的到达时间改为 0ms：按照输入顺序，A 仍先执行。

理解这些结果后，下一步再实现 continuous batching：在轮次边界重新选请求，让多个请求参与同一轮 Decode。

## 第三步：Decode batching 与动态进出

运行 `python3 batch_demo.py`，使用与串行实验相同的 A/B 请求。

本步规则：轮次之间调度；若有已到达但未 Prefill 的请求，先选最早的一个执行完整 Prefill。否则所有已 Prefill 且未完成的请求组成一个 Decode batch，一轮各生成一个 token。Prefill 与 Decode 在同一张 GPU 上轮流执行，没有混合 batch。

教学假设：无容量限制；Decode 整轮固定 2ms，与成员数无关，不能用于推断真实 GPU 吞吐。持续到达的 Prefill 可能让 Decode 长时间等待，这是当前优先级的局限，第四步再比较其他规则。

| GPU 执行时间 | 阶段 | batch 成员 | 完成时发生什么 |
| --- | --- | --- | --- |
| 0–8ms | Prefill | A | A 首 token，加入活跃集合 |
| 8–12ms | Prefill | B | B 首 token，加入活跃集合；A 等待 |
| 12–14ms | Decode | A、B | 各生成一个 token；B 完成退出 |
| 14–16ms | Decode | A | A 完成退出 |

这里的并行是同一轮 batch 内多个请求各前进一步，不是多线程：批处理时间只加一次。每个请求自身的 token 仍然逐个产生。只有到达且完成 Prefill 的请求可以进入 Decode。

阅读 [batching_lab.py](batching_lab.py) 的 `simulate_batched()`：

1. `pending` 是尚未开始 Prefill 的请求，包含未来才到达的请求；选取时必须检查到达时间。
2. `active` 是已经产出首 token、还有输出未生成的请求。
3. `members = tuple(active)` 固定本轮成员；执行中到达的新请求不能插入。
4. 一次推进整轮时间，然后给每个成员记一个输出 token。
5. 轮末移除完成的请求，下轮重新选批；新请求在 Prefill 完成后加入活跃集合。

`_Progress` 是函数内部可变状态，`field(default_factory=list)` 为每个请求建立独立列表，防止共享记录。`BatchStep` 记录一轮 GPU 执行区间，最终 `BatchRun` 和请求结果仍不可变。事件日志仍按已知 workload 计算，未引入未来事件堆。输出长度只用于请求结束判断，不用于排序或优先级。

示例同时打印串行和 batching 指标：A 的 token 时间从 8/10/12ms 变成 8/14/16ms，ITL 从 2/2ms 变成 6/2ms；B 的首 token 从 16ms 提前到 12ms。改善新请求 TTFT 的同时，A 的生成间隔变长。

先尝试把 B 到达时间改为 9ms：它到达时 A 正在 8–10ms Decode，必须等到 10ms 才开始 Prefill。再尝试在 `requests` 中加入 C（13ms 到达，输入 1、输出 2），观察 C 怎样在后续轮次进入，B 怎样先退出。先手算再运行。

验证：`python3 -m unittest -v`，包含 batch 成员变化、中途到达、单 token 输出和单请求兼容性。

完整学习路线和阶段记录见 [LEARNING_LOG.md](LEARNING_LOG.md)，后续对话入口见 [HANDOFF.md](HANDOFF.md)，协作规范见 [AGENTS.md](AGENTS.md)。

## 第四步：Prefill / Decode 优先级

运行 `python3 priority_demo.py`。请求、耗时和组批逻辑保持相同，只改变轮次边界选择阶段的规则。`simulate_batched()` 默认仍是 `prefill_first`，已有示例保持原行为；传入 `policy="decode_first"` 使用严格 Decode 优先。

| 规则 | 8ms 的选择 | A token 时间 | B token 时间 | A 的 ITL | B 的 TTFT |
| --- | --- | --- | --- | --- | --- |
| Prefill 优先 | 先处理 B 输入 | 8/14/16ms | 12/14ms | 6/2ms | 9ms |
| 严格 Decode 优先 | 继续生成 A | 8/10/12ms | 16/18ms | 2/2ms | 13ms |

第一种缩短 B 的首 token 等待，但让 A 暂停生成；第二种保护 A 的生成间隔，但 B 等到 A 完成。当前严格 Decode 优先会排空活跃集合才接纳下一个请求，加上每轮只处理一个 Prefill，因此退化为串行、无法形成多请求 Decode batch。这是刻意比较的极端教学规则，不代表所有实际 decode-first 算法。

代码只需重点读 batching_lab.py 的【第四步重点】：`prefill_ready` 表示有已到达的新请求；`choose_prefill` 表示本轮是否选择处理它。`not active` 表示没有待生成的活跃请求。下面的 Prefill 和 Decode 执行代码均复用原实现。

本步先看结果并解释谁承担等待，不要求完整理解 Python 语法。用户理解后再进入第五步：Prefill 分块，逐步探索避免两种极端的方法。固定 Decode 耗时和无容量限制仍然是教学假设。

