# 项目交接记录

更新时间：2026-10-09（Asia/Hong_Kong）

## 用户目标与协作方式

用户希望理解 LLM serving，并最终探索 D 端压力大时的调度。模拟器是学习工具，不追求快速完成复杂系统。每步需要解释思路、实现与实验结果，保持编码规范，留下后续对话可接续的记录。

## 环境与边界

- GitHub 仓库计划：`robin1hub/llm-serving-lab`，默认私有，当前正在创建与首次推送。
- Windows 工作区：`C:/Users/yh/Desktop/sim/llm-serving-lab`。
- WSL2 路径：`/mnt/c/Users/yh/Desktop/sim/llm-serving-lab`。
- 相邻 `router-simulator` 是师兄的项目，供参考；本步骤不修改它或已有环境。
- 当前标准库实现，无依赖安装、无虚拟环境配置。

## 已实现

- `serving_lab.py`：不可变请求和耗时参数、单请求虚拟时间执行、事件日志、token 时间和指标。
- 默认输入：A，0ms 到达，4 个输入 token，3 个输出 token。
- 语义：完整 Prefill 产生首 token；之后每轮 Decode 产生一个 token。token 时间为 8、10、12ms。
- 第二步已增加 `simulate_serial()` 和 `serial_demo.py`：单 GPU，按到达顺序（同时间保持输入顺序）完整串行执行，不抢占。`simulate()` 增加关键字参数 `gpu_available_ms`，开始时间取到达与 GPU 可用时间的最大值。
- `Result` 增加 `start_ms` 和 `waiting_ms`。到达与开始 Prefill 分为两条事件。
- 默认 B 在 3ms 到达，输入 2、输出 2；12ms 开始，16/18ms 产出 token；等待 9ms、TTFT 13ms、总延迟 15ms。
- 参数为教学假设。没有显式事件队列、batch、KV 容量、网络或 P/D 分离。串行规则直接计算排队时间。
- `test_serving_lab.py`：首 token / Decode 轮数、单输出边界、到达时间平移及非法输入。

## 验证

已在 WSL2 Ubuntu / Python 3.10.6 验证：

- `python3 serving_lab.py`：正常退出，token 时间为 8、10、12ms；TTFT 8ms，ITL 2ms、2ms，端到端延迟 12ms。
- 第二步 `python3 serial_demo.py`：正常退出，A/B 的时间线与上述手算一致。
- 第二步 `python3 -m unittest -v`：8 个测试全部通过（含原 4 个）；覆盖排队、GPU 空闲间隙、到达排序和同时间稳定顺序、空 workload、重复 ID 拒绝。

## 下一步

用户已确认理解单请求和串行排队，下一步是 Decode batching 与动态进出。先约定 Prefill/Decode 的选批规则、轮次边界和教学 batch 耗时模型，再实现。完整八步路线及学习记录见 LEARNING_LOG.md。当前请求是建立 GitHub 仓库记录学习过程，尚未开始第三步代码。

后续对话可让助手先读本目录的 AGENTS.md、README.md、HANDOFF.md 和代码，检查实际现状后继续，不重复创建项目。
