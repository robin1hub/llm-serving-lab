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
    E --> RC[recovery 受害者选择与恢复状态]
    RC --> E
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
| simulator/memory_budget.py | 设备/模型/预留参数、KV字节及整块容量换算 | 不连接服务器、不部署模型、不管理块生命周期 |
| simulator/kv_cache.py | KVConfig/Block/Table/Reservation、单GPU池、只读准入、事务与冻结快照 | 不选原始候选、不计算token内容、不进行网络传输 |
| simulator/recovery.py | 独立策略配置、只读受害者选择、暂停/FIFO恢复与进展保护状态 | 不分配KV、不预测耗时、不决定最终输出长度 |
| simulator/predictor.py | 不可变工作描述、模型/硬件/修正参数、教学或roofline估算 | 不选批、不分配KV、不修改进度或时钟 |
| simulator/metrics.py | 从完成结果计算摘要 | 不改变事件/调度状态 |
| run_simulation.py | 连接加载、执行、统计、输出 | 不实现第二套调度机制 |

## 状态所有权与不可变边界

Request 记录输入参数；实际最终输出长度只供内核判断完成，不放入 Scheduler 快照。ReadyTask 的候选 token_count 是待处理工作量；BatchPlan 内的 token_count 是这轮实际执行工作量。

engine 的局部可变状态为 future 最小堆、waiting deque、active列表、prefill_done字典、token_times/records/starts字典、busy/wakeup_pending与时钟。没有全局运行状态，每次 simulate_events 独立。

每次运行创建独立 Scheduler，记录 last_completed_phase。decide 仍只读候选与阶段状态；完成事件通过 on_batch_completed 更新该实例的上轮阶段。配置方案不可变，可复用方案但不共享 Scheduler 的运行状态。

未来事件 _ScheduledEvent 按(time_ms, priority, sequence)排序，priority为arrival=0、complete=1、wakeup=2。事件优先级不是阶段优先级；同刻全部到达/完成发生后才选批。complete携带启动时的BatchPlan，不重新选人。

## 当前配置契约

推荐实验顶层为 workload/timing/scheduler_profiles/workers。方案含 prefill_algorithm、decode_algorithm、mix_algorithm 与必填 constraints 对象；支持 fcfs_chunked、round_robin 和四种 Mix 选择规则 prefill_first/decode_first/alternating/mixed_decode_first。constraints 仅接受 max_batch_requests/max_tokens_per_batch/prefill_chunk_tokens，可选正整数或 null，省略/null 不限制。workers 按方案名引用，字段 id/role/scheduler_profile 必填。P 要求 Prefill 槽，D 要求 Decode 槽，Mix 要求三个槽。所有方案均校验；目前执行只接受一个 Mix worker，P/D 或多个 worker 明确报错。

旧 workload/scheduler/timing JSON 仍兼容，scheduler.policy 转换为 Mix 槽；新旧写法禁止混用，constraints 不允许 policy。Python 的 SchedulerConfig.policy 保留作兼容入口，提供 profile 后阶段选择以 profile.mix_algorithm 为准。未知字段/算法、错误引用均报错，不静默降级。第9章有完整配置与选批流程。

workload.path相对配置目录；CLI的--config和--output相对cwd。generated负载有同名manifest时验证SHA-256及数量。标准模型数据准备外的基础模拟只用标准库。

## 执行契约与必须保持的不变量

1. 单GPU执行区间不重叠；busy时到达只排队，不抢占。
2. Scheduler.decide不改变快照或队列，计划可单阶段或Mixed；Prefill最多一个请求的输入块，Decode轮转。普通选批成员阶段仍为Prefill/Decode，批级Mixed标签不用于更新请求状态。
   Mix 槽仅在 P/D 都就绪时决定阶段；alternating 依据上轮完成阶段选择另一阶段，初始选 P。仅一阶段就绪时不空转，只有完成事件改变交替状态。mixed_decode_first先选Decode，再用剩余人数槽与token预算加入队首Prefill块；空间耗尽则仅Decode，不预留Prefill额度。
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

默认teaching复用旧timing：Prefill按输入数线性，Decode整轮固定，无上下文/人数负担；Mixed为Prefill部分加一次Decode时间，整批同一完成屏障。可选roofline用dense/SwiGLU/GQA简化模型分矩阵组、Attention组；每组max计算时间与访存时间，再相加与batch开销。系数乘时间，权重每batch按读取一次估计，KV读写为粗略近似。未支持MoE/MLA、实际内核/缓存效率，也未校准，来源标记uncalibrated_roofline。

顶层predictor可省略（teaching）或显式配置kind。teaching要求timing且不允许model/device/calibration；roofline必须含这三个对象且禁止同时设置timing。模型维度和字节数是正整数，头数需整除；硬件算力/带宽与系数为有限正数，batch开销有限非负；未知字段报错。roofline没有显存容量准入；当前Scheduler不使用预测来选择成员。公式、单位、假设、手算见第10章。

TTFT包含全部输入块与期间等待。统计P50/P95用最近秩，ITL按所有间隔，吞吐窗口首到达至末完成含空闲，空样本null。

输出schema_version仍为1，新增 worker 与 resolved_scheduler_profile 为扩展，记录实际引用与解析后算法/约束；scheduler 扁平字段保留兼容。旧结果保持可读，但不能当作启用了新机制。plans[i]对应steps[i]。

第10章新增predictor配置与estimates明细。teaching的timing_source/timing维持原值；roofline来源uncalibrated_roofline、timing为null，不将内部兼容Timing默认值展示成启用参数。

## 明确尚未实现

共享KV/COW/传输保护、实际显存预算验证、CPU交换、SLO优化、多worker/P→D交接/传输、真实模型执行、4090校准、失败/取消/截止终止。分块本身不会自动让Decode获得运行机会，需要启用交替等策略。

## 验证与交接入口

```bash
python3 -m unittest discover -s tests -q
python3 run_simulation.py --config configs/token_budget.json --output output/token_budget.json
```

test_profiles.py检查方案/角色绑定、交替状态、手算时间与旧配置兼容；test_token_budget.py检查范围、首token、限制与进度；test_scheduler.py检查只读选批与轮转；test_engine.py检查事件边界/互斥；test_config/test_dataset/test_metrics分别守住输入、分词和口径。依赖缺失的分词集成明确skip。

下一AI先确认git status，读取HANDOFF和本页，再看对应实现/测试；保留用户修改。运行机制实现到第14章，第15章是校准材料准备，用户尚未确认理解此步，先讲解结果再选择下一机制，不自行扩为完整系统。test_predictor.py验证公式、上下文边界、兼容与导出。只在本教学项目工作，不修改相邻师兄项目或全局环境。资产和生成结果被Git忽略，本地存在不代表新机器能直接取得。

## 第11章混合执行补充

提交遍历items，仅取走Prefill成员。完成按item.phase更新；中间输入块返回waiting、最终输入块才产首token。只轮转本轮Decode成员，新Prefill进入active不算已服务Decode。每轮仍一个完成事件，执行期间到达不得插入。roofline整体估算Mixed成员，权重/开销按整批一次；教学相加规则只作近似。test_mixed.py验证预算、完成/轮转、上下文、手算与到达边界。混合轮可减少次数，但不保证延迟/吞吐改善。详见第11章。

## 第12章：KV块契约

可选顶层kv_cache要求gpu_id/block_tokens，必须且只能另选total_blocks或memory_profile。前者直接配正整数块数，后者按独立profile换算。省略禁用，null不合法。一池一个GPU，解析后KVConfig仍是设备身份/正整数块数/块token数。

BlockPool独占活块、已提交请求块表、在途事务和ID。KVBlock保存设备/块ID、逻辑页序号、全局token有效范围、Reserved/Populating/Ready、request_refs和execution_owner；记录冻结且替换更新。块ID单调不复用，GPU+ID一起解释身份。当前每块仅一个请求引用；没有共享/COW/跨卡副本或传输接口。

admit只读删减候选，先留可增长Decode，剩余空间缩小Prefill。无成员时内核尝试过滤掉容量阻塞Decode的回退候选；不跳过FCFS Prefill队首。预测针对准入后计划。reserve先全量校验，创建新块并保护写尾块；begin_populating启动填充；complete才提交有效范围/请求表并解除执行保护；request完成后解除引用回收。rollback删除新块、还原旧块，已提交表不变。事务只能完成/回滚一次，单池当前最多一在途事务。

必须保持：活块数不超过total_blocks，空闲数不负；已提交表中的块存在/有请求引用，逻辑范围覆盖cached_tokens；未完成事务不增加cached_tokens；释放执行中的请求报错。Reserved/Populating映射在reservation.writes中，快照同时导出reservations。碎片仅统计Ready块的尾部空位，峰值包括曾预留块。

无准入计划时不制造无进展事件，等待已有未来事件；事件耗尽且有未完成请求则status=blocked。结果保留pending_progress与KV快照；requests/summary仅统计完成请求，summary_scope明确completed_requests_only；CLI保存JSON再返回2。成功返回0。历史每个KV动作保存完整冻结快照，旧配置禁用时kv_final=null/kv_records为空。schema_version仍1，扩展status/pending/kv字段。

KV分配与状态变化无独立耗时，未模拟张量/地址/外部空洞布局、共享/COW、未来增长预留；可选抢占恢复见第14章。第12章与test_kv_cache.py是阅读/验证入口。用户明确希望支持后续多卡多状态，不回退为只计总token的模型；扩展机制逐步实现。

## 第13章：独立显存预算profile

kv_cache.memory_profile路径相对实验配置目录；profile严格解析device（name/memory_capacity_bytes）、model（name/num_layers/num_kv_heads/head_dim/kv_bytes_per_element/weight_bytes）、memory（utilization/runtime_reserve_bytes；可选block_metadata_bytes/block_alignment_bytes）、可选provenance对象。比例在(0,1]，容量/模型正整数，预留/元数据非负，对齐正整数；预算不足或不足一整块在启动前报错。

memory_budget.resolve_budget计算floor容量×比例，再扣显式权重/运行预留；单卡dense/GQA每token KV=2×层×KV头×head_dim×元素字节，整页payload加元数据再对齐，KV预算除页大小取floor。MemoryBudget冻结且导出每步计算/配置/余量；provenance只作来源说明。启用roofline时检查KV结构/精度与容量profile一致；不代表模型耗时已校准。BlockPool不读取profile，只接收已解析KVConfig。

2026-10-13服务器只读：6张4090每卡24564MiB，使用单卡容量25757220864字节；现有Qwen3-8B为36层/8KV头/head_dim128/BF16，权重metadata.total_size=16381470720。当前例子比例0.9、运行预留2GiB、KV2字节、16token一块、元数据0/对齐1均可配置，其中预留及比例未实测。算得每token147456字节、每块2359296字节、KV预算4652544409字节、1972块/余12697字节。现有Qwen2.5-7B服务器目录未找到，明确选择Qwen3例子；没有更改原负载/tokenizer。

结果新增memory_budget/memory_provenance扩展字段，schema_version仍1。当前teaching耗时不变；容量示例不是4090性能benchmark。未测实际权重VRAM/运行时峰值、未做vLLM部署/耗时校准、多卡分片或MLA/滑窗模型。第13章和test_memory_budget.py验证手算、路径/互斥、服务器配置及结构冲突。

## 第14章：抢占与重算契约

顶层recovery含必填capacity_policy（block/preempt）、可选victim_policy（fewest_generated/newest，默认前者）、recovery_action（只支持recompute）。整个对象省略时block；preempt必须有KV。字段/算法严格校验。容量规则、受害者选择与恢复动作分别配置，不等同于复现师兄完整ksana/sglang策略。

正常准入及可增长Decode回退均失败、GPU空闲时，engine.try_preempt选择受益者与受害者。受害者必须在普通队列、有已提交KV且无执行保护；按当前单引用可释放块数检查是否足够让受益者处理至少1token。候选只有当前生成数和到达序号，无最终长度。释放请求引用后记录保存的cached_tokens并移出普通队列。focus受益者直到完成，不让其他请求重新抢空间；不能自我抢占。容量过小仍报告blocked。

RecoveryController独占FIFO暂停队列、restoring目标、focus和动作记录。普通驻留请求耗尽且无focus时开始FIFO恢复；恢复请求受保护到完成。Recompute是内核专用单成员批级/成员阶段，正常Scheduler快照不含此阶段，不能与P/D混批。范围从0连续重建至暂停前已提交缓存长度，受token/chunk/容量约束；同样reserve/populating/complete，经Predictor预测。无新输出且roofline不含LM head，teaching按每token Prefill成本；不更新原prefill_done、starts、token_times或Scheduler交替状态。恢复后根据是否已有输出回active或waiting。

cached_tokens与生成输出数分别保存；Decode当前最后一个输出尚未处理，不必加入重建。暂停的普通请求进度保持原值，KV表从0恢复，因此不得把prefill_done当恢复游标，游标来自BlockPool已提交cached_tokens。Recompute只构造历史长度/工作量，没有真实token内容或张量。

EventRun/JSON扩展recovery_records、recovery_pending（未恢复目标）和recovery_summary（抢占次数、重算token），schema_version仍1。逻辑输入/输出统计不重复计重算，batch步数包含重算。默认block无恢复记录；ShareGPT requests/steps/summary/estimates保持不变。第14章和test_recovery.py核对时序、历史输出/部分输入、预算、FIFO、失败及预测边界；这是保守进展保护，不是SLO/公平优化。

## 第15章：校准材料与运行时边界

calibration/plan.json是建议实验矩阵，templates/是metadata/capacity/batches/requests/comparison空白记录；不属于load_experiment配置、不被simulate_events加载，无新运行接口。record_status=template不表示实测；原始数据放忽略的output/calibration/<run_id>。时间ms、容量bytes，batch.items沿用PredictorItem；engine_iteration_wall与gpu_forward、request_end_to_end等边界不可混用。请求输出事件只有可靠逐token映射才能和模拟ITL直接比较。

当前仍uncalibrated_roofline/teaching_assumption，无自动采集、拟合/记录校验或校准profile发布能力。下一阶段容量报告先用显式total_blocks核对，耗时按fit/validation分形状独立验证，实际模型/版本和计时边界要先核对。首轮计划不覆盖Mixed/Recompute；详见第15章。

## 服务器环境准备补充

用户当前暂停实际校准，仅准备依赖。requirements-server.txt固定vLLM/torch/transformers；scripts/check_environment.py是独立服务器工具，与事件模拟器无依赖关系，禁用可见GPU、核对sys.prefix及包依赖、导出精确版本。environments/保存可复现依赖报告，dependencies_ready不等于已测真实推理。无采集/拟合/新Predictor来源；不得自动运行模型。环境说明见calibration/server-environment.md。

## 第16章：离线参照与真实采集边界

calibration/scripts是离线工具，不改变Scheduler/Engine/KV执行契约。run_senior_reference.py接收固定scenario，只读加载师兄已有native，先跑单请求DES并验收schema/SHA/数量，再replay固定纯P/D形状；使用相同PredictorItem，12fit/4validation。结果显式标synthetic_simulator_reference_not_hardware_calibration、hardware_measured=false及二进制来源；源HEAD与binary报告commit分开，当前不同。NumPy/YAML等来自师兄已有环境，事件模拟标准库依赖不变。

configs/reference_senior.json是显式可选参照系数，设备速度假设100TFLOP/s/1TB/s，原默认配置不变，Predictor来源仍uncalibrated_roofline。不能用TTFT拟合执行时间；完整DES示例另记调度周期及小Prefill等待条件。不覆盖Mixed/恢复/容量精度。

真实采集入口collect_batches.py/worker_probe.py只针对vLLM0.11.0单卡eager，Worker前后CUDA同步计时worker_execute_wall，不称engine_iteration或gpu_forward。batch_records从实际Scheduler输出提取上下文与工作量，拒绝抢占恢复和多token Decode；UUID核对后数字卡号映射及显存/利用率检查先于vLLM初始化。metadata.started与measured分开，当前启动未成功、无有效实测数据；独立环境依赖ready不代表GPU计时已验证。第15章初始“未实现采集器”描述为历史阶段，当前工具已准备。

## GPU0共享卡smoke补充

select_device新增显式max_existing_memory_mib阈值，默认64；共享试跑用3072且仍要求利用率0。collect_batches/launch_remote显式传显存使用比例，默认0.9不变，本次0.8，metadata保存shared_gpu_trial及阈值。授权只表示允许共享试跑，利用率0不构成独占保证。

TimedWorker在Qwen3-8B/vLLM0.11.0真实smoke已运行成功，读取实际1412块/16token和Worker计时。summarize_smoke只接受完整两形状试跑，过滤预热与Decode样本中的输入Prefill，提交摘要和证据SHA；不改运行时、Predictor来源或默认容量profile。完整拟合/验证仍未执行，不能用少量共享smoke发布可靠性能profile。

## 第17章：已有模型的完整实测对照

launch_remote支持显式--model，独立采集器不绑定Qwen3名称。Worker容量RPC返回effective_engine，metadata将请求/实际参数分开；V1实际chunk=True，即使请求False。旧王机smoke的请求参数不能当有效参数。核心事件模拟器仍无外部依赖或执行语义变化。

compare_measured.py读取完整matrix及显式load_experiment roofline配置，检查模型结构与实测一致、400生成输出数、16形状各20有效重复。纯目标阶段/完整人数筛选后，逐轮实际PredictorItem送estimate，按一次生成的轮次中位数、再20重复中位数对照；排除计数/上下文范围和原始SHA保存。16组等权误差不代表TTFT/ITL或逐token加权误差。compare脚本不使用其配置Scheduler生成batch，batch来自真实vLLM记录。

configs/measure_qwen25_3b.json只改模型结构、速度仍假设100TFLOP/s/1TB/s、系数1/1/开销0，不影响默认。杜机单4090/已有Qwen2.5-3B、50%预算完成硬件矩阵，平均绝对误差54.39%；fit_performed=False。有效chunk配置、eager/同步计时、其他主机任务等限制见第17章，不跨模型/硬件移植误差或系数。
