# 项目交接：当前状态

更新时间：2026-10-15（Asia/Hong_Kong）。公开仓库：https://github.com/robin1hub/llm-serving-lab 。

最新结果：A6000直接测试已完成。第20章和本页末尾是当前进度；下方旧阶段关于“尚未拟合/无新GPU结果/采集器仅草稿”的说明已被后续记录取代。

## 先读与协作要求

AGENTS.md → 本页 → ARCHITECTURE.md → 第20章20-a6000-mixed.md（A6000直接测试）/第19章（旧混合分析/停止拷贝）/第18章（拟合/验证）→ 对应代码和测试。LEARNING_LOG保留历史，不能当作当前未实现列表。README是运行/章节索引。

用户首要目标是理解LLM serving与D端压力，代码基础较弱；要中文原因注释、编号阅读重点、流程图、结构表与手算。用户希望保留师兄块身份/状态等设计以扩展多卡多state，不能仅计总容量。小步实现，用户已同意并完成最小实测及完整硬件矩阵，已完成限定范围拟合和新形状验证，不自动启用共享/COW/传输或部署服务。只改本项目，不改师兄项目、服务器和全局依赖。

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

此前72测试中70通过，2个分词集成因系统Python缺依赖跳过；新增3个采集记录/设备映射测试。禁用KV时默认ShareGPT requests/steps/summary/estimates逐项不变，本步结果output/sharegpt_recovery_regression.json与output/sharegpt_kv_regression.json对照。test_kv_cache核对身份/生命周期/历史冻结、保护、页边界/尾部浪费、原子失败与回滚、混合缩块、Decode回退、阻塞JSON和退出码、禁用/大容量兼容。

```bash
python3 -m unittest discover -s tests -q
python3 run_simulation.py --config configs/kv_blocks.json --output output/kv_blocks.json
python3 run_simulation.py --config configs/kv_blocked.json --output output/kv_blocked.json
```

例子A在0到达输入2输出3、B在1到达输入3输出2；2块每块2token、chunk2、混合算法。容量删减后轮次P A0–4、D A4–6/6–8释放A、P B8–12/12–14、D B14–16释放B。A输出4/6/8，B14/16ms；峰值2块，结束全空。改1块则A首token4ms后D跨页增长阻塞，B也不能入场，status=blocked、pending=A/B。

输出kv_records逐动作保存完整快照（块/表/在途reservation），kv_final保存最终状态；大负载结果会变大。块号不复用，但释放的容量可复用；Ready也可能是部分有效页。碎片仅统计Ready尾部空位，峰值包括曾预留块。

## 尚未实现

多worker/P→D交接、共享KV引用接口、写时复制、传输保护/跨卡副本、模拟容量公式与实测池大小的系统对齐、外部碎片地址布局、未来增长预留、CPU交换、SLO准入、跨模型/硬件泛化精度验证。已经有4090与A6000限定范围耗时拟合及实际KV池记录，不代表全部校准完成。引用/GPU字段不等于扩展机制已启用，当前一池最多一在途事务。

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

## 最新：换已有模型，杜机完整矩阵与原始误差完成（2026-10-14）

用户授权“跑一下看看实测和模拟误差”，随后提出A6000或杜机，最终要求用已有权重免下载。只读核对A6000：4×RTX A6000/49140MiB，卡1/2约6/4MiB且0%；已有完整Qwen3-8B、Qwen2.5-3B。杜机：GPU4为4MiB/0%，有完整Qwen2.5-3B-Instruct两分片，Qwen3-8B仍缺全部分片。按索引/safetensors头数据范围核对，不是全内容SHA。选择杜机GPU4+已有3B及此前独立vLLM0.11.0/torch2.8.0环境，没有权重/环境下载，A6000无修改/运行。

launch_remote新增--model，其他默认不变。du-qwen25-3b-eager-matrix-20261014-01以90%预算在Prefill2048遇OOM，无COMPLETE，不用于统计。第二次-02用50%预算成功，400次生成、9224条Worker记录、16形状每组5warmup/20实测；退出0/GPU4回4MiB，原始完整文件已取回ignored output/calibration/<run_id>，远端runs保留。vLLM权重VRAM6218668032bytes，KV6028470272bytes，10220块×16=163520token。其他卡/服务器仍忙，不承诺主机独占。

发现请求enable_chunked_prefill=false被V1改成true，新增Worker/metadata.effective_engine，实际prefix=False、chunk=True、利用率预算0.5。王机旧smoke日志也显示chunk=True，旧“禁用分块”表述已纠正，旧原始metadata.engine只是请求值，不改写旧原始数据。有效参数必须以实际读值/日志为准。

新configs/measure_qwen25_3b.json为未调参基线：3B结构36层/H2048/I11008/16头/2KV头/V151936，沿用假设100TFLOP/s/1TB/s与系数1/1/开销0。不是4090有效速度实测，也未套用师兄参照拟合系数。compare_measured.py检查完成、模型身份/输出数量/16形状20有效重复；逐轮按actual items预测，排除预热、其他/混合阶段、人数不完整轮次，按每次生成先取中位再按20次重复取中位，避免31轮D伪装31次独立重复。计时仍Worker同步eager，非TTFT。

16组等权平均绝对相对误差54.39%、最大64.14%。P128实测19.0245ms/预测7.1334（-62.50%）；P512 23.6084/28.8047（+22.01%）；P2048 87.0746/119.8388（+37.63%）；D B1/C128 16.1516/6.1770（-61.76%）。D实际context随生成到C+30，完整轮次人数/上下文均匹配。短P和D偏快、长P偏慢；固定开销/有效速度等只是待检验解释，不称已证明。

摘要已提交calibration/measurements/du-qwen25-3b-matrix-20261014.json，16组表/流程图/结构/统计和代码映射在第17章。fit_performed=False，未发布硬件校准profile，基础Engine/Predictor/default配置不变。6个采集专项测试通过，新基线CLI运行成功。用户尚未确认理解这轮结果；下一步讨论误差，再用原fit/validation划分拟合验证，不自动扩展Mixed/多卡或宣称请求级校准完成。

## 最新：同一实测由师兄Predictor预测（2026-10-14）

用户要求师兄预测Qwen2.5-3B这份数据。compare_senior_measured.py只读加载原有native二进制，官方replay_shape接口；统一3B结构/BF16/TP1/假设100TFLOP/s与1TB/s、师兄默认calibration_params={}，不拟合。SHA核对同一份杜机-02实测，复用compare_rows逐组断言实测中位数/重复/轮次/排除/context范围相同。6681有效轮次、541唯一工作量，request prompt_end来自原输入长度，增长context/range独立保留；只支持纯P/D，不运行DES或调度等待。

师兄平均绝对相对误差54.17%、最大64.04%，我们54.39%、64.14%。P128实测19.0245/我们7.1334/师兄7.8154ms；P2048实测87.0746/我们119.8388/师兄120.4810ms；D B1/C128实测16.1516/我们6.1770/师兄6.1816ms。证明本次两种解析配置都偏離实测，不代表师兄已校准配置或所有功能精度。不能称这些速度是实测有效速度，不能把离线执行估算冒充请求级TTFT/ITL。

第17章新增三方完整表/适配逻辑/复现；提交摘要calibration/measurements/senior-du-qwen25-3b-20261014.json，本地完整输出output/calibration/senior-du-qwen25-comparison-01。native SHA与第16章同一文件，旧build commit与当前HEAD不同，未重编译/修改师兄。8个采集专项测试通过，正式拟合尚未做；下一步仍待用户讨论。

## 预测公式详细讲解补充（2026-10-14）

用户询问预测时间的计算过程。已核对当前代码并实际打印D1/context128、P128、P2048的OperatorEstimate数字；第10章末尾新增工作量→算力/带宽→max→两组相加、权重显存读取/组批共享、attention位置对数及真实模型手算。D首轮6.176444ms、P128为7.133373ms、P2048为119.838831ms，与第17章的多轮汇总口径区别说明。强调速度是假设、固定差额不能直接当开销、factor不是已测GPU物理参数。仅讲解/文档，无新采样、拟合或公式变更；用户理解尚未确认。

## 最新：拟合并重采8旧/8新验证完成（2026-10-14）

用户授权校准再测试，要求一半新类型；明确选择“新长度和batch人数”，仍纯P/D。询问Mixed时已解释原记录的混合过渡被排除，专门Mixed验证尚未做；后续先沿用参数测试，误差大再补充模型/拟合，不默认另起一套参数。

fit_timing.py只读原du-qwen25-3b-eager-matrix-20261014-02的12组split=fit，原始SHA/基线Predictor身份检查，20重复等权基函数；标准库非负最小二乘有效集合、带主元消元。得到dense_factor=0.35379254965754375、attention_factor=4.3110183582864、batch_overhead_ms=14.168541970007391；profile configs/calibrated_qwen25_3b.json SHA=229f40d9ecad321b3631de1c65f0614f0dcb509937addf5e7027a3586c9e0dce，生成后冻结。原4组validation误差+2.57/-7.65/-4.68/-3.40%，未调参。训练来源报告measurements/du-qwen25-fit-20261014.json。

calibration/fresh_validation.json在新采集前固定16形状、8repeat/8novel（按形状数，不保证单请求数正好半数）。旧P128/256/512/2048，D B1/C128、B4/C512、B8/C2048、B2/C1024；新P192/384/768/1536，D B3/C384、B3/C1536、B6/C768、B6/C1536。seed20261014、token_base2000（旧1000），仍精确token合成输入，非ShareGPT/请求级时序。collector新增--matrix，默认原矩阵/seed保持；matrix_plan.py纯标准库校验，复制杜机独立目录。

杜机GPU4/3B/BF16/TP1/50%显存、vLLM0.11 eager Worker同步，实际chunk=True/prefix=False，与训练条件相同。du-qwen25-3b-eager-fresh-20261014-03完成400次生成/6874Worker记录、每形状5warmup/20有效重复、退出0/COMPLETE/GPU4恢复4MiB。validate_fresh.py核对冻结profile SHA、矩阵/模型/软件/有效运行参数/输出数，再复用compare_rows匹配实际工作量，测试不反向拟合。

同一新数据：校准前平均绝对误差46.27%→校准后5.95%，最大13.65%；旧8组48.30%→6.53%、最大11.63%；新8组44.24%→5.36%、最大13.65%。最大新P384高估，16形状均在此前20%教学阈值内。不要把原轮54.39%和新轮5.95%直接当同数据对照。新摘要measurements/du-qwen25-fresh-validation-20261014.json，完整原始数据/逐轮对照在ignored output/calibration对应run及du-qwen25-fresh-validation-01，远端保留。

仅验证Qwen2.5-3B/4090/当前边界/范围附近纯P/D。factor是补偿参数，不是单独测出的物理性能；新长度在原训练范围附近，不保证外推、其他模型/size/显卡/precision/CUDAGraph/框架/TP、Mixed/Recompute/异构上下文或TTFT。公式通用，但参数和精度声明限定profile条件。核心Engine/Predictor不改、默认配置不改；source字段仍固定uncalibrated_roofline，尚未做校准证明绑定的运行时元数据接口，真实来源以独立报告为准。

第18章完整流程/结构/参数/新数据/范围及命令；12个采集专项测试通过（参数恢复/边界、validation隔离、半数新形状、最大误差等），校准profile CLI成功加载；4token例子只是配置核对，未验证该范围。用户学习理解尚未确认，后续讨论Mixed测试与跨模型验证，不自动推进。

## 最新：已有混合轮次分析完成，用户停止权重拷贝（2026-10-15）

用户授权看混合batch效果，优先分析原du-qwen25-3b-eager-fresh-20261014-03中此前排除的混合过渡。219非warm Worker轮、139生成重复、7原投递案例，冻结原3B/4090纯P/D profile无重新拟合；平均绝对案例误差29.28%→21.77%、最大31.55%。典型1P1024+1D（19重复）实测49.51/预测40.98ms低估17.22%；3P512+1D（19重复）66.52/49.37低估25.78%；5P768+1D（19重复）158.32/108.37低估31.55%。7案例等权，案例内生成重复等权。此前纯P/D5.95%不能扩展混合；多数记录是多P混合，当前Scheduler最多1P块。其他1P多D档位多数1–2重复，不够验证所有目标。

analyze_mixed.py读取实际items一次estimate，非P时间+D时间相加；保留部分Prefill/LMHead标记、实际context。报告source_case只是投递实验标签，另按P/D人数与P总token分23组成档位；每轮仍用exact items，档位内上下文可能不同。摘要measurements/du-qwen25-existing-mixed-20261015.json，逐轮输出ignored output/calibration/du-qwen25-existing-mixed-01，模型/软件/有效参数/profile SHA核对。13专项测试通过，第19章说明统计/数据/范围，核心默认不改，无新拟合。

原计划补采1P128/512/1536+1/3/7D，先让D就绪再注入P。杜机mixed-smoke-20261015-01启动前候选GPU1已被他人占用保护退出；-02选GPU2但原/data/HF_MODELS/Qwen2.5-3B-Instruct/config.json不在，LLM初始化前失败，没有模型加载/有效数据/COMPLETE。模型目录现已不存在，/root/model/Qwen/Qwen3-0.6B也只有tokenizer不是可用模型，不能据目录名判断有权重。

用户要求替代模型且尽量不用下载。A6000的/data/HF_MODELS/Qwen2.5-3B两分片/配置仍齐全，是同结构基础版，不是旧Instruct checkpoint；Wang8B仍有但卡忙。曾开始scp -3把A6000已有模型拷到杜机独立models/Qwen2.5-3B-base-from-a6000，用户随后明确“别拷贝了”，已Ctrl-C终止本任务scp并核对本地匹配scp进程无残留。没有启动模型，不再传输/补采，不自动恢复。目标目录可能有partial，保留未删除，不能当完整权重。

受控collect_mixed.py尚未真实GPU验证，已移到ignored output/calibration/drafts/mixed/collect_mixed.py，未公开正式代码；本地/远端launch_remote已恢复正式不带--mixed版本，远端单独草稿可能仍存在。两份此前取消Wang试验留下的configs/measure_qwen3_8b.json与transfer_qwen3_8b_from_3b.json仍未提交草稿，不混入本轮提交。下一步只在用户重新授权/具备权重条件后考虑补采，当前无本任务运行中的采集/传输。用户理解进度尚未确认。

## 最新：A6000直接测试完成（2026-10-15）

用户随后明确“如果可以，直接测试a6000”，授权重新进行直接本地权重测试，不恢复跨机模型拷贝。alpha-a6000已有/data/HF_MODELS/Qwen2.5-3B基础版两分片完整；config SHA=27467e2fc311a7656c09a43ae9c3d408211e1f212bbddbc93c8596a06423826f，不是旧Instruct。GPU2 UUID GPU-208f7dc6-a184-3397-89f8-51d91e692d46，48GB，驱动570.124.06。新建独立/data/llm-serving-lab-calibration，Python3.10.12/vLLM0.11.0/torch2.8.0 CUDA12.8/transformers4.56.1，159包检查通过，记录calibration/environments/alpha-a6000.json及freeze。没有复用或修复别人旧Python3.12环境；仅安装项目依赖与小工具，没有模型权重下载/拷贝。

先mixed-smoke-20261015-01验证API，随后a6000-qwen25-3b-pure-20261015-02完成16组/400生成/9214 Worker记录。GPU50%预算、TP1BF16/eager/prefix=False、有效chunk=True、maxlen4096/maxseq8/tokenbudget4096/block16；KV31961块/511376token，预算18851833856字节、权重显存6218668032。未校准基线平均绝对误差49.56%。只用12fit拟合dense=0.7568788631690692、attention=3.060882675764624、overhead=11.729456370363053ms；独立profile configs/calibrated_a6000_qwen25_3b.json SHA=a471d55138c72911531508bcb783cd090ffe657b3f3bb55e40917be0c8005a0b，4090参数未改。4组留出平均8.08%，P256高估21.40%仍是局限，另外三组约3–5%；未回流留出数据调参。

参数冻结后，a6000-qwen25-3b-mixed-20261015-03完成9组/225实验/3743记录；1个P128/512/1536 × 1/3/7个D，每组5warm/20有效。collect_mixed先观察真实全体D就绪，再注入完整P，用原生engine.step/实际Worker计时；精确身份、P范围、单token D、最终输出16/1、恰好目标轮校验。正式analyze_mixed还完整验收矩阵，然后按generation_checks的matched_iteration只选180个目标轮。驻留准备阶段另外114个非warm混合轮必须排除；早期ignored analysis-01包含这些准备轮已作废，正式逐轮输出为a6000-qwen25-mixed-analysis-final，不据旧文件得出结论。

9组等权平均绝对误差28.80%→5.64%，最大10.35%，混合数据未拟合。P128+1D实测17.14/预测17.24ms；P512+3D36.62/34.58；P1536+7D97.33/87.26。实际D context514–515，按成员值预测。限定本次A6000/基础版3B/同步eager/当前组合，不能与4090旧多P混合21.77%直接归因对比，更不代表TTFT/ITL、其他模型、更长上下文、多P或中间chunk都验证。

摘要calibration/measurements/a6000-qwen25-{baseline,fit,controlled-mixed}-20261015.json，完整证据本地ignored output/calibration/<run_id>/及远端runs同名目录。两次正式采集退出0/COMPLETE，GPU2恢复4MiB/0%，无残留推理/模型传输。uv缓存保留在项目内，安装后/data余量约48GiB；不清理其他人的文件。collect_mixed正式公开，launcher新增--mixed；fit_timing只增加device-label/scope来源参数，原拟合公式/默认参数兼容；analyze_mixed补GPU身份/矩阵/目标轮验收。15个采集专项测试通过，配置加载及diff检查通过。第20章有流程/时序、数据结构、重点代码、全部9组表和复现入口。默认模拟器及核心公式不改。

当前用户步骤已完成，下一步先讨论结果：若继续可测更长D上下文或请求级混合调度对照，而不是自动启动更多测试/重新拟合。用户是否理解本次结果尚未确认。两份取消Wang试验的8B配置仍为原有未提交草稿，不混入提交。
