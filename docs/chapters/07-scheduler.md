# 第 7 章：独立 Scheduler、BatchPlan 与人数上限

## 本步改变什么

过去在事件内核里直接选择整个 active 集合。现在拆成：

```text
就绪状态 → SchedulingSnapshot → Scheduler.decide → BatchPlan
                                                    ↓
                                   内核提交、占用 GPU、注册完成
                                                    ↓
                                      更新进度、退出、轮转
```

参考师兄的只读候选视图、执行计划与提交职责分离。师兄当前代码位于 csrc/src/scheduler/{facade,batching,capacity}.cpp，接口在 scheduler.hpp。我们只借鉴分工，不复刻其 C++ 框架、generation、回滚或 KV 容量策略。

本步只有请求数上限与 Decode 轮转，没有 token 预算、分块、混合 batch、KV 检查、抢占或截止时间优化。耗时仍是教学参数。

## 数据结构与职责

| 结构 | 保存什么 |
| --- | --- |
| SchedulerConfig | 阶段优先级、每轮请求数上限 |
| ReadyTask | 请求 ID、下一步阶段、本轮可以处理的 token 数 |
| SchedulingSnapshot | 按顺序排列的等待 Prefill、就绪 Decode 候选元组 |
| BatchPlan | 被选中的 ReadyTask 元组和选择原因 |
| BatchStep | 内核执行该计划的开始/完成时间、阶段和成员 |

同一计划暂时只能包含一个阶段。Prefill 的 token_count 为完整输入长度，完成后只产生首个输出 token；Decode 的 token_count 为 1。快照不暴露实际最终输出长度，Scheduler 不能根据已知答案决定谁先完成。

Snapshot 和 Plan 都是不可变数据。Scheduler.decide 不修改 waiting/active、不分配 KV、不推进时钟；相同快照和配置给出相同计划。空候选返回 None，表示 Idle。本步没有资源阻塞，所以尚不需要 Defer。

## 配置与兼容

```json
"scheduler": {
  "policy": "prefill_first",
  "max_batch_requests": 2
}
```

上限必须是正整数；0、负数、小数、字符串、布尔值均报错。省略或写 null 表示不限制本轮人数，保持旧配置行为。它不是活跃请求数上限或显存容量：active 可以有四个请求，本轮仅选两个。

Prefill 仍按到达顺序一次一个，任意正上限都允许它执行。Decode 按候选队列顺序取前 N 个。strict Decode 优先仍会排空 active 再开始新 Prefill，因此本步没有解决 Prefill 饥饿；轮转公平性只针对已就绪的 Decode 候选。

## 完成后轮转，为什么不在选批时轮转

假设 active 为 [A,B,C,D]、本轮选择 [A,B]。执行未完成前，不修改它们的生成进度，也不移动服务顺序。完成后先删除已结束请求，再把已服务且未结束的成员移到尾部：

```text
active [A,B,C,D] → 选择 [A,B] → 完成 → active [C,D,A,B]
active [C,D,A,B] → 选择 [C,D] → 完成 → active [A,B,C,D]
```

如果 A 在第一轮完成，下一轮队列应为 [C,D,B]，不能把已完成 A 放回去。rotate_after_completion 只返回新顺序，由内核采用；算法的候选决策本身不改变队列。

未选中请求继续等待，下轮还有资格；没有释放它的 KV（本版尚未建模 KV）。轮转避免固定队首长期占据限额，但长 Prefill 仍可能打断 Decode。

## 内核怎样执行计划

唤醒时，engine.py 建快照并调用 Scheduler.decide。计划确定后，内核才取走选中的 Prefill、设置开始时间、标记 GPU 忙、登记 BatchStep 和未来完成事件。Decode 未选中的候选保留在 active。

完成事件携带启动时那份不可变计划，不能在结束时重新选成员。内核按计划成员产生 token、结束请求，之后轮转。候选、启动、完成分别是不同阶段。

未来有容量规则时，在“计划提出”和“内核提交”之间插入容量检查。当前没有资源分配失败，因此未引入事务回滚，不能宣称已经保证 KV 准入。

## 可手算的四请求实验

运行：

```bash
python3 run_simulation.py --config configs/batch_limit.json --output output/batch_limit.json
```

四个请求同时到达，每个输入 1、输出 3，Prefill 每输入 token 2ms、Decode 整轮 2ms，上限 2。

| 时间 | 计划 |
| --- | --- |
| 0–2ms | Prefill A |
| 2–4ms | Prefill B |
| 4–6ms | Prefill C |
| 6–8ms | Prefill D |
| 8–10ms | Decode A、B |
| 10–12ms | Decode C、D |
| 12–14ms | Decode A、B，二者完成 |
| 14–16ms | Decode C、D，二者完成 |

A token 时间 2/10/14，B 为 4/10/14，C 为 6/12/16，D 为 8/12/16ms。后续 Decode 间隔为 4ms，即便每轮只耗时 2ms，因为请求要等待另一组执行。初次间隔还包含其余请求的 Prefill。

输出 plans 与 steps 按同一轮索引对应；plans 写工作量和原因，steps 写时间区间。trace 会打印候选集合、本轮选择和未全选的原因。输出 scheduler 保存实际人数上限和轮转规则。

练习：把上限改为 1 或 4，先预测执行顺序与 ITL，再看结果。变化发生在谁被选中，而不是算术上的 GPU 加速；固定 2ms batch 耗时未校准。

## 阅读与验证

先看 scheduler.py 的三个重点：选择阶段、截取本轮成员、完成后轮转。再看 engine.py 的“只读快照 → 候选计划”和“提交重点”，不用重读完整事件循环。

本次 WSL2 标准库测试：25 个中 23 通过、2 个分词集成因缺依赖跳过。覆盖只读决策、轮转与完成退出、上限 1/2/不限、非法配置、原优先级结果。默认 ShareGPT 的请求记录、执行区间、摘要与提取 Scheduler 前逐项一致，新增计划日志 9543 条。

下一步是 token 预算与分块 Prefill；容量、抢占和 TTFT/ITL 目标优化继续按小步引入。
