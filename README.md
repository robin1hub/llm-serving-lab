# LLM Serving 学习实验

目标是通过小段可读代码理解 serving 过程，并逐步探索 D 端压力下的调度。当前支持单请求和单 GPU 的多请求串行执行，没有真实模型计算。

## 在 WSL2 运行

在任意位置克隆后进入仓库目录即可运行：

```bash
git clone https://github.com/robin1hub/llm-serving-lab.git
cd llm-serving-lab
python3 serial_demo.py
python3 -m unittest -v
```

私有仓库需要具有访问权限的 GitHub 登录。当前本机 WSL2 的运行路径如下：

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

完整学习路线和阶段记录见 [LEARNING_LOG.md](LEARNING_LOG.md)，后续对话入口见 [HANDOFF.md](HANDOFF.md)，协作规范见 [AGENTS.md](AGENTS.md)。
