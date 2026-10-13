# 项目交接：当前状态

更新时间：2026-10-13（Asia/Hong_Kong）。公开仓库：https://github.com/robin1hub/llm-serving-lab 。

## 先读与协作要求

AGENTS.md → 本页 → ARCHITECTURE.md → 第16章16-reference-comparison.md（本地参照）/第15章（真实采集）→ 对应代码和测试。LEARNING_LOG保留历史，不能当作当前未实现列表。README是运行/章节索引。

用户首要目标是理解LLM serving与D端压力，代码基础较弱；要中文原因注释、编号阅读重点、流程图、结构表与手算。用户希望保留师兄块身份/状态等设计以扩展多卡多state，不能仅计总容量。小步实现，用户已同意进入校准准备，尚未实际测量，不自动启用共享/COW/传输或部署服务。只改本项目，不改师兄项目、服务器和全局依赖。

## 当前已实现

- 单卡到达/完成/唤醒事件，busy阻止重叠；候选快照/计划冻结，Scheduler只读选批。
- 独立scheduler_profiles算法槽与worker按名字引用；旧policy兼容，禁止混用。P/D方案可定义校验，但完整运行只支持一个Mix worker。
- Prefill FCFS分块、Decode轮转；人数/token/chunk限制正整数或null。Mix支持严格P优先、严格D优先、alternating和mixed_decode_first，后者先D、剩预算最多一个P块。
- 内核按成员阶段提交/完成，整批同一完成时刻；中间P块无输出，最终块产首token；只轮转Decode成员。
- 独立Predictor：默认teaching保持旧公式，Mixed为P线性耗时加一次D轮时间；可选未校准roofline，两组max计算/访存后相加与开销。模型/算力/带宽/系数可配，结果保存明细。Decode上下文为input+generated-1，无未来输出目标输入。
- 新增kv_cache.py：KVBlock身份/GPU/有效范围/状态/引用/执行所有者、RequestBlockTable、KVReservation、BlockPool与冻结快照。不是实际KV张量/地址。
- 可选kv_cache要求gpu_id/block_tokens，再独选人工total_blocks或独立memory_profile，后者按容量比例/权重/预留/KV结构/元数据/对齐换算。准入先保留可增长D再缩P；完全阻塞时尝试可增长D回退。预留→填充→就绪→完成请求释放，支持预留原子容量失败与事务回滚；执行中禁止释放。
- 默认block容量满无法增长时，等待已有未来事件；无事件则blocked，保存pending_progress/块表，不伪造完成。CLI保存结果再退出2；成功0。summary仅含完成请求，summary_scope标记口径。
- ShareGPT单轮离线准备、本地Qwen tokenizer、fixed/poisson到达、manifest SHA；基础模拟标准库。summary包含TTFT/ITL/latency最近秩分位数和吞吐；未校准。

## 当前验证与例子

最新72测试中70通过，2个分词集成因系统Python缺依赖跳过；新增3个采集记录/设备映射测试。禁用KV时默认ShareGPT requests/steps/summary/estimates逐项不变，本步结果output/sharegpt_recovery_regression.json与output/sharegpt_kv_regression.json对照。test_kv_cache核对身份/生命周期/历史冻结、保护、页边界/尾部浪费、原子失败与回滚、混合缩块、Decode回退、阻塞JSON和退出码、禁用/大容量兼容。

```bash
python3 -m unittest discover -s tests -q
python3 run_simulation.py --config configs/kv_blocks.json --output output/kv_blocks.json
python3 run_simulation.py --config configs/kv_blocked.json --output output/kv_blocked.json
```

例子A在0到达输入2输出3、B在1到达输入3输出2；2块每块2token、chunk2、混合算法。容量删减后轮次P A0–4、D A4–6/6–8释放A、P B8–12/12–14、D B14–16释放B。A输出4/6/8，B14/16ms；峰值2块，结束全空。改1块则A首token4ms后D跨页增长阻塞，B也不能入场，status=blocked、pending=A/B。

输出kv_records逐动作保存完整快照（块/表/在途reservation），kv_final保存最终状态；大负载结果会变大。块号不复用，但释放的容量可复用；Ready也可能是部分有效页。碎片仅统计Ready尾部空位，峰值包括曾预留块。

## 尚未实现

多worker/P→D交接、共享KV引用接口、写时复制、传输保护/跨卡副本、实际容量实测验证、外部碎片地址布局、未来增长预留、CPU交换、SLO准入、真实模型/4090测量和拟合。引用/GPU字段是实际结构，不等于这些机制已启用。当前一池最多一在途事务。

## 本地环境

Windows C:/Users/yh/Desktop/sim/llm-serving-lab；WSL /mnt/c/Users/yh/Desktop/sim/llm-serving-lab，Ubuntu、Python3.10+。只用标准库运行；数据依赖在项目.venv，由已有/root/.local/bin/uv安装（系统缺ensurepip），requirements-data.txt记录依赖。不安装全局包。

Qwen tokenizer三份JSON在assets/tokenizers/qwen2.5-7b-instruct；生成负载workloads/generated/sharegpt-100-fixed/requests.json已有100请求。原ShareGPT位于相邻router-simulator/datasets/sharegpt，仅只读。assets、.venv、workloads/generated、output均被Git忽略，新机器不能假定存在。prepare配置和历史学习细节在各章节。

## 服务器4090独立容量profile

configs/rtx4090_memory.json引用configs/profiles/rtx4090_qwen3_8b_bf16.json；来源只读服务器nvidia-smi与现有Qwen3-8B小JSON。6卡各24564MiB，但只模拟1卡。Qwen3是36层、8KV头、head_dim128、BF16，权重张量16381470720字节。此前Qwen2.5-7B目录当前不存在，不虚构同一模型配置。

90%使用比例、2GiB运行时预留、BF16 KV、每块16token、元数据0/对齐1是假设。容量换算1972块，144KiB/token、2.25MiB/块、KV预算4652544409字节。不是vLLM实测块数，权重张量大小仅估计VRAM；teaching耗时仍未校准。memory_budget.py严格解析并计算，结果memory_budget/provenance可复现，禁止profile和人工块数混用。若同时用roofline，KV结构/精度需一致。

新入口：python3 run_simulation.py --config configs/rtx4090_memory.json --output output/rtx4090_memory.json。69测试中67通过、2分词集成跳过；test_memory_budget覆盖公式/对齐/边界、相对路径、互斥、服务器值与模型冲突。配置文件运行时全本地，无需服务器。已有ShareGPT长度为Qwen2.5分词，之后校准Qwen3需对应分词器重准备。容量假设见第13章；不自动部署或继续完整校准。

## 最新：独立抢占选择与重算恢复

顶层recovery配置capacity_policy=block/preempt、victim_policy=fewest_generated/newest、recovery_action=recompute；默认block保持旧结果。preempt必须启用KV，swap明确拒绝。fewest_generated按已生成数升序、同数较晚到达优先；newest选较晚到达者，不使用最终输出目标。容量准入与Decode回退均失败后才选择受害者，只释放未执行、能实际释放足够页面让受益者前进一步的其他请求。

recovery.py保存独立选择纯函数、FIFO恢复队列、restoring与focus；engine实施释放/恢复。暂停保留原prefill_done、首开始时间和输出时间，保存已提交cached_tokens（含处理过的输出上下文）。受益者focus持续到完成；没有驻留工作且focus清空后，FIFO恢复一个请求并保护到完成。新Prefill不会插队抢回空间，避免教学例子的反复踢出，但不声称公平或SLO最优。

Recompute是独占恢复批，受token/chunk/KV预算约束；从0重建到保存的cached_tokens，经同一Predictor收费，不产token、不算LM head、不更新原prefill_done或交替策略状态。恢复完成再回普通Prefill/Decode。仍可能blocked，例如单个请求自己也放不下；禁止自我踢出。共享/COW、交换、跨卡恢复均未启用。

configs/recompute.json配2块×2token，A/B同时到达且输入2/输出3。8ms暂停B，A输出4/10/12ms；12–16ms重算B，B输出8/18/20ms。保留B首token8ms，下一间隔10ms；输入逻辑计数4、输出6，额外重算2token另列。输出recovery_records、recovery_pending和recovery_summary；blocked时暂停队列也保留。

验证69测试中67通过、2分词集成跳过；test_recovery覆盖选择、完整及部分Prefill恢复、含历史输出重建、FIFO/小预算、不可恢复阻塞、roofline无LM head及配置。运行python3 run_simulation.py --config configs/recompute.json --output output/recompute.json。阅读第14章流程图/状态图/结构表和编号标记；恢复实现保持不变；目前进入第15章校准材料准备，不自动实现交换。

## 第15章初始校准准备材料（未实测）

用户授权先做小型校准方案与数据记录格式。本步只新增calibration/plan.json、README和templates，章节15说明容量/执行轮/请求端三个口径及字段与代码映射；当时没有服务器操作、采集器、拟合器、新Predictor配置或真实计时数据；后续独立环境准备见本页末尾。模板record_status=template且观测null，不能导入为实测。复制至output/calibration/<run_id>/后采集，不提交原始数据。

建议单4090/Qwen3-8B BF16 TP1；先实际KV池容量，B1纯Prefill、B1/4/8纯Decode拟合，独立形状B2等验证，再20请求固定到达对照。首轮不含Mixed/Recompute校准。plan是建议矩阵，不是模拟器/服务配置；实际软件版本/模型/空闲GPU/参数尚未重新核对。Qwen2.5现有tokenizer不能冒充Qwen3。下一步只读核对环境、设计轮次计时并采一个最小形状，之后才扩矩阵/拟合；不能用HTTP TTFT或整次生成延迟拟合单轮Predictor。

本地核对JSON/JSONL解析、模板无测量值、plan拟合与验证形状分离及git diff --check。未改运行代码，不重复整套测试，基线仍69个测试67通过2跳过；用户尚未确认理解方案。

## 两台4090服务器检查（2026-10-13，只读）

详见calibration/server-survey.md。alpha-wang与alpha-du（WHU_4090_4同目标）均6×4090。本次wang空闲0–3，/data剩56G；du空闲4–5，剩67G；两机99%使用。wang的/data/HF_MODELS/Qwen3-8B五个权重分片均存在；du同目录仅小JSON/tokenizer，五分片缺失，先前读取模型元数据不能视为已有权重。du的verl环境包元数据有vLLM0.11.0，但未验证运行；wang有限范围未见现成环境。

目前倾向alpha-wang候选GPU0，直接复用权重，下一步再核对独立环境依赖/驱动及容量余量。无远端修改/安装/模型启动/下载；卡未预留，占用需重查。不要覆盖其他实验环境；compute-apps列表为空不能推断无任务。校准尚未开始。

## 用户最新要求：暂停实际校准，仅准备服务器环境

用户先授权开始校准，随后明确“先不校准，把环境准备好就行”。不得自动启动模型/测量。两机独立目录/data/llm-serving-lab-calibration，venv是新建Python3.10环境，tools内uv0.9.4；主依赖固定vLLM0.11.0/torch2.8.0/transformers4.56.1。杜机已通过导入与uv pip check，报告及159包清单在calibration/environments/alpha-du*；torch CUDA构建12.8。王机也已通过导入/版本/路径/159包依赖检查；官方源慢后改清华HTTPS镜像，结果在server-environment.md和environments/alpha-wang*。

杜机独立模型下载已按用户最新范围停止，只终止本任务download_model.py进程，partial保留；并非完整可运行模型。王机原/data/HF_MODELS/Qwen3-8B五分片仍齐全，后续优先王机，但原候选GPU已被其他任务占用，没有终止他人任务。没有vLLM服务、模型加载、容量实测、耗时采样或拟合；未经运行的计时草稿仅保留ignored output/calibration/drafts，不作为可用采集器发布。

新calibration/scripts/check_environment.py禁止可见GPU，只检查独立prefix、模块导入、包依赖并导出freeze。dependencies_ready只表示依赖通过，GPU运行/模型启动尚未验证。文档server-environment.md包含路径、命令、限制；只有用户再次要求恢复校准后才核对空闲GPU并启动，不自动推进。

最终环境状态：两机dependencies_ready，Python3.10.12，torch CUDA构建12.8；王机检查时剩约40.6GiB，杜机54.0GiB。激活脚本和检查报告/精确清单均保存；仅依赖准备，未测GPU运行或真实推理。

## 最新工作：真实启动暂缓，完成本地师兄参照（2026-10-13）

用户恢复王服务器校准后，两次smoke尝试均失败：wang-qwen3-8b-eager-smoke-20261013-01在vLLM0.11的CUDA UUID解析处失败；已改为先核对物理UUID，再传数字卡号。-02启动前发现GPU0占2441MiB，保护检查拒绝加载。没有模型加载、capacity.json、有效batch耗时或COMPLETE；metadata的started不能视作measured。用户回复“稍等一下吧”，随后授权改做本地师兄参照。不要自行启动服务器采样或终止他人任务。

已提交采集入口collect_batches.py、独立TimedWorker、batch_records.py、launch_remote.py。仅针对vLLM0.11.0：eager、BF16、TP1、禁用prefix/chunk，Worker前后CUDA同步包围execute_model，timing_scope=worker_execute_wall，含预处理/forward/sampling/output，不含Engine Scheduler/RPC/HTTP。不是完整engine_iteration；GPU插桩尚未通过真实运行。smoke输出3token，正式Decode输出32token，逐轮保存实际上下文/实际人数，不把整个生成耗时当batch。服务器已复制之前版本脚本；本地launch_remote包装器整理后尚未重传，恢复时先同步核对。环境导入ready不等于GPU计时ready。

本地参照脚本calibration/scripts/run_senior_reference.py用师兄现有Python3.13环境与native二进制只读运行；-S防自动editable构建、-B禁pyc、无可见GPU。所有缓存/日志/输出写本项目output。按师兄输出解读skill验收schema8、complete、文件SHA、明细数量：1请求输出2token，4事件/290算子。参照报告native源commit=3b374848…，源码HEAD=4b7a96be…，不是最新HEAD构建；result.json保存二进制SHA，两者不得混淆。

固定Qwen3-8B实际结构与共同假设100TFLOP/s/1TB/s，非4090有效性能。单请求输入4输出2；将师兄prefill_delay_min_tokens设1去掉默认小Prefill约50ms等待、轮次间隔1微秒，首输出师兄15.166638ms/我们原始15.139144ms。完整请求结果不用于拟合，16组replay_shape纯P/D工作量用于预测比较。

12组fit、4组validation；dense_factor=1.0121111392281201、attention_factor=0.9053187031593333、开销0。验证最大绝对相对误差3.74%→2.63%，部分Decode变差，fit短Prefill仍低估约6%。不保证全范围，不覆盖Mixed/Recompute/非零context Prefill/ShareGPT。新增可选configs/reference_senior.json，本地运行首输出15.3224ms、完成30.6432ms；默认配置与运行时实现不变，仍uncalibrated_roofline。条件/来源/16组对照已提交calibration/reference/{scenario,result}.json，完整输出忽略于output/calibration/senior-reference-04。

第16章有流程图、结构表、拟合公式、代码重点与结果。72测试中70通过、2分词集成缺依赖跳过，参照配置CLI成功。用户尚未确认理解本步。下一步由用户选择讨论差异，或空闲卡就绪后继续真实smoke；不能将参照系数包装为4090实测校准。

## 最新：王GPU0共享卡smoke成功（2026-10-13）

用户同意尝试占约2GB、利用率持续0的王机卡，撤销本任务继续等待空闲卡的限制。三次检查0–3各2441MiB/0%，选择GPU0，不清理/终止他人进程。新增显式--max-existing-memory-mib=3072、--gpu-memory-utilization=0.8；默认仍64MiB/0.9，非零利用率仍拒绝。metadata.shared_gpu_trial=true，不能宣称独占或无干扰。更新采集脚本已同步王服务器独立目录。

run_id=wang-qwen3-8b-eager-shared-smoke-20261013-03，成功加载Qwen3-8B、完成两形状smoke、退出0，已有COMPLETE/measured。实际权重VRAM16394201088 bytes，KV预算3332999987.2 bytes、1412块×16=22592token（80%预算，不能直接与原90%纸面1972块比较）。6次生成输出长度通过，12条Worker记录；剔除预热及Decode实验的输入Prefill，P有效2轮中位24.0518ms，D有效4轮中位18.3172ms、实际context128/129。是worker_execute_wall同步eager时间，不是TTFT或纯kernel。退出后GPU0恢复原2441MiB；无持续推理任务。

原始数据已取回ignored output/calibration/<run_id>，远端runs同目录保留；提交的calibration/measurements/wang-shared-smoke-20261013.json只有摘要/条件/原始证据SHA。新增summarize_smoke.py核对COMPLETE、measured、实际人数/阶段、输出数和预热排除，不拟合；5个采集专项测试通过。基础模拟器/默认Predictor/容量profile不变。

最新讲解calibration/wang-shared-smoke.md及第15章入口；第16章的参照实测限制只描述当时参照实验。当前已取得少量硬件smoke数据且TimedWorker真实运行通过，不能沿用“无真实数据/计时未验证”的旧状态。尚未完整fit/validation矩阵、共享干扰监测、请求级校准或发布硬件profile；样本少，系数仍未校准。下一步讨论是否扩展完整矩阵，用户尚未确认理解这轮结果。
