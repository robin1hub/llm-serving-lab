# 项目交接：当前状态

更新时间：2026-10-15（Asia/Hong_Kong）。公开仓库：https://github.com/robin1hub/llm-serving-lab 。

最新结果：路线第2阶段固定1P1D已实现，P产生首token后经FIFO链路交接KV到D；源保留、目标Ready、源释放边界已验证，但目标容量不足的等待/PD抢占尚未实现。第22章、ROADMAP与本页末尾为当前进度。第21章独立多Mix和第20章A6000校准保留；旧阶段未实现说明以最新记录为准。

最新研究交接：[D_HEAVY_RESEARCH_BRIEF.md](D_HEAVY_RESEARCH_BRIEF.md)已整理“D-heavy时哪些D迁到P、哪个P接收、迁多少”的自包含报告。师兄建议是**零阶优化**，不是零和；尚需确认泛指黑盒调参还是特指扰动梯度估计，以及离线/在线范围。报告含原始文献、已核对深度、指标、实现缺口、实验草案和下一session可复制任务。当前只授权整理报告；没有实现D→P或搜索器，也没有创建/联系其他session。

## 先读与协作要求

AGENTS.md → 本页 → ARCHITECTURE.md → ROADMAP.md → 第22章22-fixed-pd.md（第21章是多卡基础）→ 对应代码和测试。校准背景见第18–20章。LEARNING_LOG保留历史，不能当作当前未实现列表。

用户首要目标是理解LLM serving与D端压力，代码基础较弱；要中文原因注释、编号阅读重点、流程图、结构表与手算。用户希望保留师兄块身份/状态等设计以扩展多卡多state，不能仅计总容量。小步实现，用户已同意并完成最小实测及完整硬件矩阵，已完成限定范围拟合和新形状验证，不自动启用共享/COW/传输或部署服务。只改本项目，不改师兄项目、服务器和全局依赖。

## 当前已实现

- 共享到达/完成/唤醒事件堆，按worker_id分发；每卡busy阻止本卡重叠，不同卡可并行；候选快照/计划冻结，Scheduler只读选批。
- 独立scheduler_profiles算法槽与worker按名字引用；旧policy兼容，禁止混用。独立多Mix用显式固定routing；固定1P1D用pd配置且两端KV/recovery=block。P/D只需对应阶段算法槽，逐卡资源独立。
- Prefill FCFS分块、Decode轮转；人数/token/chunk限制正整数或null。Mix支持严格P优先、严格D优先、alternating和mixed_decode_first，后者先D、剩预算最多一个P块。
- 内核按成员阶段提交/完成，整批同一完成时刻；中间P块无输出，最终块产首token；只轮转Decode成员。
- 独立Predictor：默认teaching保持旧公式，Mixed为P线性耗时加一次D轮时间；可选未校准roofline，两组max计算/访存后相加与开销。模型/算力/带宽/系数可配，结果保存明细。Decode上下文为input+generated-1，无未来输出目标输入。
- 新增kv_cache.py：KVBlock身份/GPU/有效范围/状态/引用/执行所有者、RequestBlockTable、KVReservation、BlockPool与冻结快照。不是实际KV张量/地址。
- 可选kv_cache要求gpu_id/block_tokens，再独选人工total_blocks或独立memory_profile，后者按容量比例/权重/预留/KV结构/元数据/对齐换算。准入先保留可增长D再缩P；完全阻塞时尝试可增长D回退。预留→填充→就绪→完成请求释放，支持预留原子容量失败与事务回滚；执行中禁止释放。
- 默认block容量满无法增长时，等待已有未来事件；无事件则blocked，保存pending_progress/块表，不伪造完成。CLI保存结果再退出2；成功0。summary仅含完成请求，summary_scope标记口径。
- ShareGPT单轮离线准备、本地Qwen tokenizer、fixed/poisson到达、manifest SHA；基础模拟标准库。summary包含TTFT/ITL/latency最近秩分位数和吞吐；未校准。

## 当前验证与例子

最新106测试中104通过、2项分词依赖缺失跳过；13项新增PD测试覆盖KV交接/并行事务/链路/上下文/统计，原9项多卡及全部单卡测试继续通过。第1阶段基础/KV/重算旧CLI JSON对照保留在output/multi-worker-before和after；本轮新增四例回归见末尾。test_kv_cache继续验证生命周期/冻结、页边界、容量回滚和blocked导出。

```bash
python3 -m unittest discover -s tests -q
python3 run_simulation.py --config configs/kv_blocks.json --output output/kv_blocks.json
python3 run_simulation.py --config configs/kv_blocked.json --output output/kv_blocked.json
```

例子A在0到达输入2输出3、B在1到达输入3输出2；2块每块2token、chunk2、混合算法。容量删减后轮次P A0–4、D A4–6/6–8释放A、P B8–12/12–14、D B14–16释放B。A输出4/6/8，B14/16ms；峰值2块，结束全空。改1块则A首token4ms后D跨页增长阻塞，B也不能入场，status=blocked、pending=A/B。

输出kv_records逐动作保存完整快照（块/表/在途reservation），kv_final保存最终状态；大负载结果会变大。块号不复用，但释放的容量可复用；Ready也可能是部分有效页。碎片仅统计Ready尾部空位，峰值包括曾预留块。

## 尚未实现

目标容量不足的交接等待/PD抢占、动态路由、多P多D、共享KV引用、COW、通用传输pin、容量公式系统校准、外部碎片地址布局、未来增长预留、CPU交换、SLO准入、TP/PP及跨模型/硬件精度。已有固定1P1D与最小副本提交/源释放，不代表完整传输保护/恢复协议；池内允许不同请求并行写入，同一请求仍禁止重叠写。多卡请求级及链路未实测校准。

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

## 最新：多卡路线第1阶段完成（2026-10-15）

用户同意进入多卡调度学习，要求先把五阶段计划写文档，再实现第1阶段，并持续维护详细讲解。已先创建docs/ROADMAP.md：独立多Mix → 固定1P1D/传输 → 交接容量协议 → 可配置路由对比 → D压力调度。当前只完成第1阶段，用户理解尚未确认，不自动推进第2阶段。

新增simulator/worker.py的不可变WorkerConfig（worker_id/gpu_id/role、方案、timing/PredictorSettings、KV/recovery及容量来源）。engine.py将原闭包状态归入_WorkerRuntime：每卡waiting/active、busy/wakeup_pending、请求进度和输出记录、Scheduler/Predictor/BlockPool/RecoveryController独立。_ScheduledEvent增加worker_id；_drive_workers只有一个future/sequence，按time、arrival0/complete1/wakeup2、稳定序号排序，分发到目标卡。不同卡执行区间允许重叠，每卡仍独占执行。单卡simulate_events复用同一内核，返回原EventRun，保留旧trace/JSON语义。

load_experiment支持workers多条，字段id/gpu_id/role/scheduler_profile，多卡gpu_id必填且唯一、id唯一，当前完整运行只接受Mix。每卡可覆盖timing/predictor/kv_cache/recovery；公共耗时作为默认，子级来源整体替换父级互斥来源。多卡禁止顶层KV/recovery以免误用共享池，各卡KV.gpu_id需匹配，未设置KV即未启用容量准入。复用旧单卡解析校验并只读取一次负载；Experiment旧单卡字段保留为首卡兼容视图，多卡调用必须用workers与assignments。

多卡必填routing={policy:fixed, assignments:{请求ID:workerID}}，恰好覆盖所有请求，无未知worker；单卡省略则全部归唯一worker。不读未来输出长度做路由，不做负载均衡或迁移。simulate_workers返回ClusterRun/WorkerRun，steps为带worker_id/gpu_id的WorkerStep；全局启动顺序保证steps/plans/estimates同序。某卡容量阻塞只退出本卡dispatch，继续处理其他卡，堆耗尽后才汇总blocked/pending。

run_simulation.py单卡仍schema1且内容保持，多卡schema2：全局summary/requests/steps/plans/estimates/trace/routing，workers分别保存配置、KV/恢复、局部summary与busy_time_ms。全局吞吐窗口不是每卡时长之和，不能相加局部吞吐；blocked仅统计已完成请求并保存未完成进度、退出2。跨卡KV快照不合并。

新增configs/two_mix_workers.json与workloads/two_mix_requests.json。两卡均教学P每token2ms、D每轮2ms，各4块×2token；A到达0、输入4输出3归卡0，输出8/10/12ms；B到达0、输入2输出2归卡1，输出4/6ms。全局窗口12ms、2请求/5输出、166.67req/s；默认未校准逻辑例子，真实GPU和共享CPU/链路争用未模拟。运行python3 run_simulation.py --config configs/two_mix_workers.json --output output/two_mix_workers.json。只把B改给卡0时卡1保持空闲，卡0容量约束使B等待：A输出8/14/16，B12/18；文档解释原因。

验证：93项测试中91通过、2项分词依赖缺失跳过，新增tests/test_multi_worker.py九项，覆盖并发/单卡互斥、同刻全局事件顺序、独立KV与恢复/交替状态、不同速度、空负载/空闲卡、固定归属、错误配置、schema2与blocked CLI。改动前后basic/kv_blocks/recompute三个CLI完整JSON逐项一致，ignored output/multi-worker-before/*.json及output/multi-worker-after-*.json保留证据。所有真实采集器/校准profile和默认示例未更改。

学习入口第21章：结果手算、全局/每卡状态表、事件流程图、配置继承与拒绝规则、KV阻塞、输出schema、五处代码阅读顺序和两个练习。README/ARCHITECTURE/ROADMAP/LEARNING_LOG同步。核心阅读标记worker.py重点1、engine运行现场重点2/驱动器重点3、config重点4；不必一次读完dispatch的KV/恢复分支。两份原有8B草稿仍未提交，未修改师兄项目、服务器环境或全局依赖。

## 最新：固定1P1D与FIFO传输完成（2026-10-15）

用户确认继续第2阶段“来做一下试试”，当前步骤已完成，不自动进入目标容量等待。新增simulator/handoff.py：LinkConfig/PDConfig/TransferRecord与HandoffController，链接由bandwidth_bytes_per_s、latency_ms、kv_bytes_per_token配置。两卡一P一D、不同GPU、各自启用KV，PD恢复只允许block；pd与routing互斥，全部请求先到P。无pd的多worker仍只接受Mix，旧Python/JSON入口保留。

engine增加simulate_pd，仍用_drive_workers的一个堆。计算complete与transfer_complete为priority1，wakeup/transfer_start为2，同刻全部完成后才选批/开传。P最终Prefill产首token后：单输出直接完成；否则向控制器交接原请求信息/首开始/输入进度，D继续引用原token_times/records，P.departed排除该请求的完整结果，最终只在D汇总一份端到端结果。中间chunk不触发传输。P可继续处理其他输入，但源KV保留到交接完成。

控制器FIFO排队：开始拷入时检查D完整输入容量、reserve/begin_populating目标块；目标尚不可Decode。载荷为源cached_tokens（完整输入，不含刚生成首token）×每token字节数，不含页空位；耗时固定延迟+bytes/带宽×1000。完成事件先target.complete提交Ready/表，再source.release_request释放源，最后加入D.active；唤醒可能受源容量阻塞的P与D。D忙时新请求仍排队。target容量不足显式ValueError，不伪造完成/无条件接入，完整目标等待/重试/抢占协议留第3阶段。已进D后的增长仍可能blocked。

必要底层改动：BlockPool允许同池不同请求的多笔写入，以支持拷入B与计算A重叠；reserve在任何变更前检查所有在途writes的request_id，同一请求写入冲突拒绝。各事务独立commit/rollback，执行所有者字段复用为写入事务保护；尚无共享块/COW。PD禁抢占且P.departed不再参与候选，源保留由当前流程保证，不代表已实现通用transfer-pin。

LinkConfig校验单位/有限正耗时，PDConfig核对角色、资源、KV字节数/结构和两端roofline模型相同。教学示例1024bytes/token是假设；实际Qwen2.5-3B BF16需36864bytes/token，配置roofline/显存profile时必须一致。不模拟链路/计算共享HBM/PCIe/CPU争用，不声明真实双卡精度。

configs/one_p_one_d.json引用workloads/pd_request.json，A输入4输出3、P2ms/token、D2ms/轮；两卡各16块×2token，链路1024000bytes/s+1ms，每token载荷1024bytes。P0–8首token8，KV4096bytes在8–13传完，D13–15、15–17，输出8/15/17，ITL7/2。带宽减半输出8/19/21。两请求A4/3+B2/2时，P0–8/8–12，传输8–13/13–16（B等1ms），A输出8/15/17，B12/19，验证D计算与复制重叠。

多卡schema2增加execution_mode=fixed_1p1d、pd、transfers、requests.worker_path；worker_id是最终完成者，routing.assignments是初始P归属。各worker的produced_output_tokens记录实际生成（例子P1/D2），summary仍按本卡保存的完整完成请求统计，不能把P.summary=0误读为未执行。全局输入/输出只计一次。计算steps与transfers分开，CLI另打印KV区间；output/one_p_one_d.json保留完整结果。

验证106测试104通过2分词依赖跳过，新增tests/test_pd.py十三项：手算/首开始首token/context、目标Populating/Ready与源释放、带宽、FIFO/复制计算并行、同刻完成、源容量唤醒、单输出/空负载、chunk、目标不足拒绝、并行事务原子性/回滚、模型字节、CLI。旧basic/kv_blocks/recompute/two_mix_workers四例JSON与此前完整逐项一致，新输出在output/pd-regression。第22章状态/时序图、结构/代码阅读点、手算、多请求表与限制已同步README/ARCHITECTURE/ROADMAP。用户理解待确认，下一步先讲解本章，再讨论第3阶段目标容量等待。未访问服务器、未改师兄或全局依赖；两份8B草稿仍保留未提交。

## 文献筛选补充（2026-10-15）

用户强调最终目标为D侧高压力调度并要求找论文。已联网核对原文/arXiv/会议页，整理docs/READING.md：Mooncake过载准入、DistServe架构与资源、TetriInfer预测与D调度、Llumnix迁移、TaiChi流动Decode、STAR预测重调度、2026gate-and-route排队控制。版本与阅读深度明确标注，Mooncake过载章节采用2024v1第6节，STAR采用v2当前标题，后两篇目前仅摘要核对。建议先读Mooncake/TetriInfer，优先以1P1D的P准入实验衔接当前路线；尚未选定或实现算法，也没有改变第3阶段功能计划。在线策略不得读取真实最终输出长度，长度预测与batch耗时Predictor要分开。纯文档更新，测试基线仍106/104通过/2跳过，无新GPU实验。
