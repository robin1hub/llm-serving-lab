# 项目交接：当前状态

更新时间：2026-10-13（Asia/Hong_Kong）。公开仓库：https://github.com/robin1hub/llm-serving-lab 。

用户以学习为主，代码基础较弱；先讲机制、状态表和结果，再讲少量 Python，不要求完整读懂代码。代码需有中文原因注释和编号阅读重点。

## 当前结构与边界

- 唯一入口 run_simulation.py；simulator/models.py 保存不可变数据结构，config.py 严格加载 JSON，engine.py 保存事件循环。
- configs/ 选择 workload、P/D 策略和教学耗时；workloads/ 是人工 token 长度请求；结果保存到 output/。
- docs/chapters/ 按七章学习；第 5 章分词/负载，第 6 章统计，第 7 章独立 Scheduler；docs/LEARNING_LOG.md 保留历史过程。
- 单 GPU、完整单请求 Prefill、Decode 按可配置人数上限轮转选批。省略/null 上限兼容之前所有活跃请求同轮执行。支持 Prefill 优先和严格 Decode 优先，后者当前退化为串行。
- heapq 按时间、事件优先级、序号排序。到达只排队，唤醒启动并注册完成，完成才产生 token。waiting 不包含未来请求，busy 防止重叠。
- ShareGPT 单轮数据准备已接入，使用本地 Qwen2.5-7B-Instruct tokenizer；没有真实模型计算、显存容量、混合 batch 或校准。未改师兄项目或环境。

## 整理目录

用户要求章节化且只保留当前使用的内容，旧单请求/串行/轮次引擎和 demo 已从工作目录移除，历史由 Git 保存。测试改为验证当前内核的手算结果和不变量，不依赖旧引擎。运行和测试命令见 README。

默认 Prefill 优先 A token 时间 8/14/16ms、B 为 12/14ms；严格 Decode 优先 A 为 8/10/12ms、B 为 16/18ms。配置驱动结果保持一致。

## 接下来

最新阶段 2026-10-13：已提取 scheduler.py，冻结 ReadyTask/SchedulingSnapshot/BatchPlan 仅描述下一步工作，不暴露最终输出长度。decide 只读、返回计划或 None；engine 提交计划，完成时更新 token、剔除完成者、把已服务者轮转到尾部。max_batch_requests 为正整数/null/省略；省略保持原结果。输出 scheduler/plans 与候选 trace。无 KV capacity checker、事务失败、分块或混合 batch。

四请求示例 configs/batch_limit.json：Prefill A/B/C/D 后 Decode [A,B]、[C,D]、[A,B]、[C,D]，最后 16ms 完成。25 标准库测试中 23 通过、2 分词集成跳过；默认 ShareGPT 请求/steps/summary 与此前结果一致。第 7 章完整设计已同步；先让用户理解候选集合与本轮成员区别，随后才加 token 预算与分块。

2026-10-12 文档补全：按用户要求，第 5、6 章补入设计取舍、逐函数职责、数据结构、关键代码及测试映射。说明离线长度与模型生成的区别、token ID 未保存的边界、manifest 发布和核对、只读结果统计、间隔权重及未来未完成请求口径。仅文档改动，无新机制；当前让用户优先学习这两章，之后讨论资源限制。

最新阶段完成结果统计：simulator/metrics.py 只读 BatchRun，summary 写入结果并打印。请求级 TTFT/总延迟、按所有间隔的 ITL 输出 count/mean/P50/P95（最近秩）；吞吐窗口从最早到达到最后完成，含空闲。空样本 null。第 6 章有定义与手算。两份 ShareGPT 配置同负载对比：Prefill 优先平均 TTFT 7543.74ms、ITL 46.128ms、输出吞吐 241.232 token/s；严格 Decode 优先为 22425.36ms、2ms、196.807 token/s。仍未校准。

本次标准库环境测试 20 个中 18 通过、2 分词集成跳过；两策略均完成 100 请求、23977 输出 token。下一步先理解摘要和等待取舍，再加 batch 预算和简单 KV 容量。完整输出在 output/sharegpt.json 与 output/sharegpt_decode_first.json（不提交）。

最新阶段：用户同意先接 ShareGPT、直接使用分词、本地执行。prepare_workload.py + configs/prepare_sharegpt.json 流式读相邻项目 ShareGPT，取前 100 个有效单轮请求、输入 chat template、输出参考文本不含 EOS，过滤 4096/1024 上限而不截断。first_valid 不做随机抽样，seed 仅控制 poisson 到达。支持 fixed/poisson。数据写 workloads/generated/，不覆盖已有目录；manifest 记录源/分词校验和、模板和统计。运行侧核对 manifest 并保存到结果。

本地 tokenizer 仅从服务器复制三份 JSON 到 assets/tokenizers/qwen2.5-7b-instruct（不含模型权重）。项目 WSL2 .venv 用 /root/.local/bin/uv 安装 requirements-data.txt，默认 python 缺 ensurepip。未安装全局依赖。事件模拟仍可用系统 python3。完整讲解见第 5 章。

验证已完成：16 个测试通过（含本地真实 tokenizer 集成），准备 100 个有效样本扫描 141 条、首轮格式跳过 41 条；输入均值 340.15、参考输出均值 239.77。python3 run_simulation.py --config configs/sharegpt.json --summary --output output/sharegpt.json 完成 100 请求、9543 轮、23977 输出 token，长度逐条一致，最后完成 99394ms（教学假设）。生成目录已经存在，重复准备应换 output_dir，不覆盖已有数据。

先让用户查看准备参数、长度统计和真实长度负载产生的队列。之后再加 batch 预算和 KV 容量，最后对齐 4090 耗时；不自动部署 vLLM。原资料里的 10 个测试是整理阶段历史，当前测试数以最新验证记录为准。

整理后 WSL2 验证：10 个当前测试通过，包含配置 CLI、手算策略时间线、到达处理中 GPU 忙碌、随机 workload 进度与执行互斥、输入校验和单输出边界。默认配置输出与整理前一致。旧阶段测试已移除，不与整理前 22 个测试数量直接比较。

先让用户熟悉章节、配置和当前代码位置，再讨论 batch 上限与资源机制的小步扩展；不要直接部署 vLLM 或迁移 C++。Windows 项目路径 C:/Users/yh/Desktop/sim/llm-serving-lab；WSL 路径 /mnt/c/Users/yh/Desktop/sim/llm-serving-lab。
