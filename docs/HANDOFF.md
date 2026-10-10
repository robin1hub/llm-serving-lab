# 项目交接：当前状态

更新时间：2026-10-10（Asia/Hong_Kong）。公开仓库：https://github.com/robin1hub/llm-serving-lab 。

用户以学习为主，代码基础较弱；先讲机制、状态表和结果，再讲少量 Python，不要求完整读懂代码。代码需有中文原因注释和编号阅读重点。

## 当前结构与边界

- 唯一入口 run_simulation.py；simulator/models.py 保存不可变数据结构，config.py 严格加载 JSON，engine.py 保存事件循环。
- configs/ 选择 workload、P/D 策略和教学耗时；workloads/ 是人工 token 长度请求；结果保存到 output/。
- docs/chapters/ 按四章学习；docs/LEARNING_LOG.md 保留历史过程。
- 单 GPU、完整单请求 Prefill、所有活跃请求同轮 Decode。支持 Prefill 优先和严格 Decode 优先，后者当前退化为串行。
- heapq 按时间、事件优先级、序号排序。到达只排队，唤醒启动并注册完成，完成才产生 token。waiting 不包含未来请求，busy 防止重叠。
- 当前没有真实模型、显存容量、混合 batch、ShareGPT 或校准。未改师兄项目或环境。

## 整理目录

用户要求章节化且只保留当前使用的内容，旧单请求/串行/轮次引擎和 demo 已从工作目录移除，历史由 Git 保存。测试改为验证当前内核的手算结果和不变量，不依赖旧引擎。运行和测试命令见 README。

默认 Prefill 优先 A token 时间 8/14/16ms、B 为 12/14ms；严格 Decode 优先 A 为 8/10/12ms、B 为 16/18ms。配置驱动结果保持一致。

## 接下来

整理后 WSL2 验证：10 个当前测试通过，包含配置 CLI、手算策略时间线、到达处理中 GPU 忙碌、随机 workload 进度与执行互斥、输入校验和单输出边界。默认配置输出与整理前一致。旧阶段测试已移除，不与整理前 22 个测试数量直接比较。

先让用户熟悉章节、配置和当前代码位置，再讨论 batch 上限与资源机制的小步扩展；不要直接部署 vLLM 或迁移 C++。Windows 项目路径 C:/Users/yh/Desktop/sim/llm-serving-lab；WSL 路径 /mnt/c/Users/yh/Desktop/sim/llm-serving-lab。
