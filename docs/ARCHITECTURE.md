# 当前架构与代码契约

本文件是当前实现地图，供用户和后续 AI 快速定位。日期2026-10-13；历史阶段见 LEARNING_LOG，不以旧记录替代当前契约。每次机制变动同步更新本页、对应章节和 HANDOFF。

## 模块地图

```mermaid
flowchart LR
    SRC[ShareGPT原文本] --> PREP[prepare_workload.py + dataset.py]
    TOK[本地Qwen tokenizer] --> PREP
    PREP --> W[requests.json + manifest]
    CFG[实验JSON] --> C[config.load_experiment]
    W --> C
    C --> R[run_simulation.py]
    R --> E[engine.simulate_events]
    E --> S[scheduler.decide]
    S --> P[不可变BatchPlan]
    P --> E
    E --> PR[predictor.estimate]
    PR --> EST[不可变BatchEstimate]
    EST --> E
    E --> B[BatchRun / EventRun]
    B --> M[metrics.summarize]
    B --> O[结果JSON]
    M --> O
    M --> T[终端摘要]
```

| 文件 | 权责 | 不应做的事 |
| --- | --- | --- |
| simulator/models.py | 不可变Request/Timing/Event/Result/BatchStep/BatchRun及输入校验 | 不保存事件执行队列 |
| simulator/dataset.py | 单轮提取、分词长度、到达计划、文件摘要 | 不参与运行时调度 |
| prepare_workload.py | 校验准备配置、发布负载和manifest | 不加载模型权重或执行GPU |
| simulator/config.py | 校验实验JSON、读取标准负载、核对manifest | 不修改请求进度 |
| simulator/scheduler.py | SchedulerProfile、约束、快照、计划、只读选批、完成反馈、轮转纯函数 | 不分配KV、不生成token、不推进时间 |
| simulator/engine.py | 持有队列与进度、提交计划、处理事件与完成 | 不解析原始文本或拟合真实耗时 |
| simulator/predictor.py | 不可变工作描述、模型/硬件/修正参数、教学或roofline估算 | 不选批、不分配KV、不修改进度或时钟 |
| simulator/metrics.py | 从完成结果计算摘要 | 不改变事件/调度状态 |
| run_simulation.py | 连接加载、执行、统计、输出 | 不实现第二套调度机制 |

## 状态所有权与不可变边界

Request 记录输入参数；实际最终输出长度只供内核判断完成，不放入 Scheduler 快照。ReadyTask 的候选 token_count 是待处理工作量；BatchPlan 内的 token_count 是这轮实际执行工作量。

engine 的局部可变状态为 future 最小堆、waiting deque、active列表、prefill_done字典、token_times/records/starts字典、busy/wakeup_pending与时钟。没有全局运行状态，每次 simulate_events 独立。

每次运行创建独立 Scheduler，记录 last_completed_phase。decide 仍只读候选与阶段状态；完成事件通过 on_batch_completed 更新该实例的上轮阶段。配置方案不可变，可复用方案但不共享 Scheduler 的运行状态。

未来事件 _ScheduledEvent 按(time_ms, priority, sequence)排序，priority为arrival=0、complete=1、wakeup=2。事件优先级不是阶段优先级；同刻全部到达/完成发生后才选批。complete携带启动时的BatchPlan，不重新选人。

## 当前配置契约

推荐实验顶层为 workload/timing/scheduler_profiles/workers。方案含 prefill_algorithm、decode_algorithm、mix_algorithm 与必填 constraints 对象；支持 fcfs_chunked、round_robin 和三种 Mix 选择规则 prefill_first/decode_first/alternating。constraints 仅接受 max_batch_requests/max_tokens_per_batch/prefill_chunk_tokens，可选正整数或 null，省略/null 不限制。workers 按方案名引用，字段 id/role/scheduler_profile 必填。P 要求 Prefill 槽，D 要求 Decode 槽，Mix 要求三个槽。所有方案均校验；目前执行只接受一个 Mix worker，P/D 或多个 worker 明确报错。

旧 workload/scheduler/timing JSON 仍兼容，scheduler.policy 转换为 Mix 槽；新旧写法禁止混用，constraints 不允许 policy。Python 的 SchedulerConfig.policy 保留作兼容入口，提供 profile 后阶段选择以 profile.mix_algorithm 为准。未知字段/算法、错误引用均报错，不静默降级。第9章有完整配置与选批流程。

workload.path相对配置目录；CLI的--config和--output相对cwd。generated负载有同名manifest时验证SHA-256及数量。标准模型数据准备外的基础模拟只用标准库。

## 执行契约与必须保持的不变量

1. 单GPU执行区间不重叠；busy时到达只排队，不抢占。
2. Scheduler.decide不改变快照或队列，计划每轮同一阶段；Prefill一次一个请求，Decode轮转。
   Mix 槽仅在 P/D 都就绪时决定阶段；alternating 依据上轮完成阶段选择另一阶段，初始选 P。仅一阶段就绪时不空转，只有完成事件改变交替状态。
3. 每计划成员数不超过人数上限，token_count总和不超过token上限；Prefill还不超过chunk。
4. Prefill范围连续且恰好覆盖完整输入；prefill_done仅在完成事件累加。
5. 非最终Prefill块不生成输出、不进入active；最终块生成首token，单输出请求直接完成。
6. starts记录第一块开始，不能被后续块覆盖；waiting包含未開始或未完成输入，但不包含未来未到达请求。
7. Decode每成员一次产生一个输出，完成者退出；未完成已服务者移到轮转尾部。
8. 模拟结束后每个请求输出数量等于目标，token时间递增；不暴露目标长度给选批策略。
9. 时间只在取事件时前进；启动只安排未来complete。
10. 默认teaching且未设置新预算时保留原A/B与ShareGPT请求结果和执行区间；trace/plans/estimates可增加说明字段。选择roofline会改变事件时间，可能进一步改变下一轮就绪候选。

## 耗时与统计假设

耗时统一通过 Predictor.estimate(tuple[PredictorItem]) 返回 BatchEstimate。内核在提交前提供本轮新token、上下文、阶段、是否产出token。Prefill上下文是range_start；Decode是input_tokens+已产出输出数-1。预测不暴露未来输出目标。内核校验正数、有限耗时以及完成时间能推进；estimates与plans/steps同序保存。

默认teaching复用旧timing：Prefill按输入数线性，Decode整轮固定，无上下文/人数负担。可选roofline用dense/SwiGLU/GQA简化模型分矩阵组、Attention组；每组max计算时间与访存时间，再相加与batch开销。系数乘时间，权重每batch按读取一次估计，KV读写为粗略近似。未支持MoE/MLA、实际内核/缓存效率，也未校准，来源标记uncalibrated_roofline。

顶层predictor可省略（teaching）或显式配置kind。teaching要求timing且不允许model/device/calibration；roofline必须含这三个对象且禁止同时设置timing。模型维度和字节数是正整数，头数需整除；硬件算力/带宽与系数为有限正数，batch开销有限非负；未知字段报错。roofline没有显存容量准入；当前Scheduler不使用预测来选择成员。公式、单位、假设、手算见第10章。

TTFT包含全部输入块与期间等待。统计P50/P95用最近秩，ITL按所有间隔，吞吐窗口首到达至末完成含空闲，空样本null。

输出schema_version仍为1，新增 worker 与 resolved_scheduler_profile 为扩展，记录实际引用与解析后算法/约束；scheduler 扁平字段保留兼容。旧结果保持可读，但不能当作启用了新机制。plans[i]对应steps[i]。

第10章新增predictor配置与estimates明细。teaching的timing_source/timing维持原值；roofline来源uncalibrated_roofline、timing为null，不将内部兼容Timing默认值展示成启用参数。

## 明确尚未实现

KV容量/页分配、容量阻塞、抢占恢复、混合P/D batch、SLO优化、多worker/P→D交接/传输、真实模型执行、4090校准、失败/取消/截止终止。分块本身不会自动让Decode获得运行机会，需要启用交替等策略。

## 验证与交接入口

```bash
python3 -m unittest discover -s tests -q
python3 run_simulation.py --config configs/token_budget.json --output output/token_budget.json
```

test_profiles.py检查方案/角色绑定、交替状态、手算时间与旧配置兼容；test_token_budget.py检查范围、首token、限制与进度；test_scheduler.py检查只读选批与轮转；test_engine.py检查事件边界/互斥；test_config/test_dataset/test_metrics分别守住输入、分词和口径。依赖缺失的分词集成明确skip。

下一AI先确认git status，读取HANDOFF和本页，再看对应实现/测试；保留用户修改。当前实现到第10章，用户尚未确认理解此步，先讲解结果再选择下一机制，不自行扩为完整系统。test_predictor.py验证公式、上下文边界、兼容与导出。只在本教学项目工作，不修改相邻师兄项目或全局环境。资产和生成结果被Git忽略，本地存在不代表新机器能直接取得。
