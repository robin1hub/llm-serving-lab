# 项目交接：当前状态

更新时间：2026-10-13（Asia/Hong_Kong）。公开仓库：https://github.com/robin1hub/llm-serving-lab 。

## 先读什么

AGENTS.md → 本页 → ARCHITECTURE.md → 当前章节11-mixed-batches.md，再看对应代码和测试。LEARNING_LOG.md是历史，不能把旧阶段的未实现列表当作当前状态。

## 用户目标与方式

用户希望理解LLM serving并最终研究D端压力。代码基础较弱，先讲机制、流程图、状态表和小例子，再读少量Python；中文原因注释和编号重点必需。不要求一次读完整文件，不自动推进后续机制。只改本项目，不改师兄项目、服务器或全局依赖。

## 当前已实现

- 独立Predictor接口：默认teaching保持旧公式；可选未校准roofline，模型/算力/带宽/两组系数/batch开销独立配置。矩阵组与Attention组分别max计算/访存时间，再求和。输出保存逐轮工作量、瓶颈和时间，与plans/steps对应。
- Prefill预测上下文为已完成输入，Decode为输入长度+输出已产生数量-1。只有最后输入块算LM Head。没有未来输出长度输入；Scheduler本步未用Predictor选批。
- roofline顶层禁止timing，必须含model/device/calibration；teaching要求timing。显存容量与CUDA核心数未纳入此耗时模型。虚拟小模型示例不是4090配置，尚未真实校准。

- 单卡事件：到达/完成/唤醒按时间与事件优先级排序，busy阻止重叠。
- 只读Scheduler、不可变快照和BatchPlan；Prefill一次一个，Decode按完成后轮转。
- scheduler_profiles 独立配置 P/D/Mix 算法槽与 constraints，workers 按名字引用。现有实验配置迁移为新写法，旧 scheduler.policy 仍兼容，两种写法不能混用。
- Prefill 为 fcfs_chunked，Decode 为 round_robin，Mix 可选 prefill_first/decode_first/alternating/mixed_decode_first。交替只在两阶段同时就绪时按上轮完成阶段切换，单阶段不空转；decide 不切换状态，完成事件才反馈。
- max_batch_requests、max_tokens_per_batch、prefill_chunk_tokens正整数/null或省略。null/省略不限制。可以定义P/D方案并校验能力，但完整请求只运行一个Mix worker，其他角色/多个worker明确拒绝，尚无交接。
- Prefill分块维护prefill_done，完成才累加；未完成放waiting队首，最终块才产生首token。首块开始时间保留，不被续块覆盖。
- ShareGPT单轮离线准备、本地Qwen tokenizer、fixed/poisson到达、first_valid前N有效筛选、长度过滤与manifest。原始文本/生成数据/分词文件不提交。
- JSON结果保存requests/steps/plans/trace、worker绑定、解析后方案、有效scheduler参数、manifest和summary。TTFT/总延迟按请求，ITL按间隔；最近秩P50/P95；吞吐窗口首到达至末完成。

## 最新验证与学习入口

```bash
python3 -m unittest discover -s tests -q
python3 run_simulation.py --config configs/token_budget.json --output output/token_budget.json
python3 run_simulation.py --config configs/worker_alternating.json --output output/worker_alternating.json
python3 run_simulation.py --config configs/predictor_roofline.json --output output/predictor_roofline.json
```

48个测试中46通过，2个分词集成因系统Python缺依赖跳过。新增test_predictor覆盖公式/两种瓶颈、系数、上下文与组批共享、事件时间线、配置和CLI导出。原basic/decode_first确定性时间线保持兼容；本步默认ShareGPT的requests/steps/summary/estimates逐项不变（output/sharegpt_mixed_regression.json）；默认ShareGPT的requests/steps/summary与引入Predictor前逐项一致，新增9543条预测记录，结果output/sharegpt_predictor_regression.json。

第10章虚拟例子：A输入3输出3、B输入1输出2，0到达，chunk2；小模型L1/H4/I8/头2/KV头1/V8，算力与带宽均1e6、开销0.1ms。5轮耗时0.724/0.516/0.484/0.900/0.548ms；A输出1.240/2.624/3.172，B1.724/2.624ms。第10章逐项手算；用户尚未确认理解Predictor，不自动推进校准。

分块例子A在0ms到达输入5输出3，B在3ms到达输入2输出2；token预算3、chunk2。A输入三块2+2+1，0–4/4–8/8–10ms；B在10–14ms Prefill；14–16ms Decode AB，16–18ms Decode A。A输出10/16/18ms，B14/16ms。结果output/token_budget.json，本地被Git忽略。

第9章新增对照：A输入1输出4、0ms到达，B输入6输出2、1ms到达；预算3、chunk2、人数2。prefill_first 的 A 输出2/16/18/20、B14/16ms；alternating 的 A 输出2/4/10/16、B20/22ms。A最大ITL从14降到6，B的TTFT从13增到19，总完成从20增到22ms。用户尚未确认理解此步，先读配置、手算表与完成反馈。

## 未实现与下一步

未实现KV容量/资源准入、抢占恢复、SLO优化、多worker/P→D交接/传输、真实模型或4090校准、失败/截止终止。严格优先级仍可能连续选择同一阶段；交替是可选算法，不代表已实现SLO优化。下一步确认用户理解后再讨论简单KV容量，不直接部署vLLM。

耗时假设：默认teaching仍是Prefill线性、Decode整轮固定2ms；roofline只是dense/GQA/SwiGLU两组粗略估计，忽略真实内核细节、缓存与复杂量化，系数手动配置，无拟合器。真实负载或理论硬件配置都不等于已校准速度。下一步先学习第11章，再讨论KV容量或真实测量。

## 本地环境

Windows：C:/Users/yh/Desktop/sim/llm-serving-lab；WSL：/mnt/c/Users/yh/Desktop/sim/llm-serving-lab。基础模拟标准库；数据准备用本项目.venv，requirements-data.txt由已有/root/.local/bin/uv安装（系统缺ensurepip）。Qwen tokenizer三份JSON在assets/tokenizers/qwen2.5-7b-instruct。ShareGPT准备目录已存在，重新准备换output_dir，不覆盖。

## 最新：混合batch（第11章）

用户授权实现，尚未确认理解。本轮最多一个Prefill块，先选Decode，再用剩余请求槽与token预算加入Prefill；无空间则纯Decode，持续Decode可导致Prefill饥饿。BatchPlan.phase可为Mixed，但items阶段仍是P/D，提交/完成按成员判断。整轮完成后统一更新，只轮转已服务Decode，新Prefill进入active排在这些成员前。

教学混合时间为Prefill线性时间加一次Decode轮时间；roofline整体估算成员、共享权重与一次开销。两者尚未校准，不保证混合更快。worker_mixed.json的A输出2/8/14/20、B20/22ms；predictor_mixed.json为A1.240/2.124/3.040、B2.124/3.040ms。结果均在output同名JSON；48测试中46通过、2分词集成跳过。test_mixed.py覆盖预算/槽不足、分块、轮转、同轮完成、预测手算与轮中到达。旧三策略继续兼容。
