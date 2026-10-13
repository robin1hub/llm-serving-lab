# 项目交接：当前状态

更新时间：2026-10-13（Asia/Hong_Kong）。公开仓库：https://github.com/robin1hub/llm-serving-lab 。

## 先读与协作要求

AGENTS.md → 本页 → ARCHITECTURE.md → 第13章13-memory-budget.md → 对应代码和测试。LEARNING_LOG保留历史，不能当作当前未实现列表。README是运行/章节索引。

用户首要目标是理解LLM serving与D端压力，代码基础较弱；要中文原因注释、编号阅读重点、流程图、结构表与手算。用户希望保留师兄块身份/状态等设计以扩展多卡多state，不能仅计总容量。小步实现，用户尚未确认理解第13章，不自动启用共享/COW/传输或部署服务。只改本项目，不改师兄项目、服务器和全局依赖。

## 当前已实现

- 单卡到达/完成/唤醒事件，busy阻止重叠；候选快照/计划冻结，Scheduler只读选批。
- 独立scheduler_profiles算法槽与worker按名字引用；旧policy兼容，禁止混用。P/D方案可定义校验，但完整运行只支持一个Mix worker。
- Prefill FCFS分块、Decode轮转；人数/token/chunk限制正整数或null。Mix支持严格P优先、严格D优先、alternating和mixed_decode_first，后者先D、剩预算最多一个P块。
- 内核按成员阶段提交/完成，整批同一完成时刻；中间P块无输出，最终块产首token；只轮转Decode成员。
- 独立Predictor：默认teaching保持旧公式，Mixed为P线性耗时加一次D轮时间；可选未校准roofline，两组max计算/访存后相加与开销。模型/算力/带宽/系数可配，结果保存明细。Decode上下文为input+generated-1，无未来输出目标输入。
- 新增kv_cache.py：KVBlock身份/GPU/有效范围/状态/引用/执行所有者、RequestBlockTable、KVReservation、BlockPool与冻结快照。不是实际KV张量/地址。
- 可选kv_cache要求gpu_id/block_tokens，再独选人工total_blocks或独立memory_profile，后者按容量比例/权重/预留/KV结构/元数据/对齐换算。准入先保留可增长D再缩P；完全阻塞时尝试可增长D回退。预留→填充→就绪→完成请求释放，支持预留原子容量失败与事务回滚；执行中禁止释放。
- 容量满无法增长时，等待已有未来事件；无事件则blocked，保存pending_progress/块表，不伪造完成。CLI保存结果再退出2；成功0。summary仅含完成请求，summary_scope标记口径。
- ShareGPT单轮离线准备、本地Qwen tokenizer、fixed/poisson到达、manifest SHA；基础模拟标准库。summary包含TTFT/ITL/latency最近秩分位数和吞吐；未校准。

## 当前验证与例子

61测试中59通过，2个分词集成因系统Python缺依赖跳过。禁用KV时默认ShareGPT requests/steps/summary/estimates逐项不变，结果output/sharegpt_kv_regression.json。test_kv_cache核对身份/生命周期/历史冻结、保护、页边界/尾部浪费、原子失败与回滚、混合缩块、Decode回退、阻塞JSON和退出码、禁用/大容量兼容。

```bash
python3 -m unittest discover -s tests -q
python3 run_simulation.py --config configs/kv_blocks.json --output output/kv_blocks.json
python3 run_simulation.py --config configs/kv_blocked.json --output output/kv_blocked.json
```

例子A在0到达输入2输出3、B在1到达输入3输出2；2块每块2token、chunk2、混合算法。容量删减后轮次P A0–4、D A4–6/6–8释放A、P B8–12/12–14、D B14–16释放B。A输出4/6/8，B14/16ms；峰值2块，结束全空。改1块则A首token4ms后D跨页增长阻塞，B也不能入场，status=blocked、pending=A/B。

输出kv_records逐动作保存完整快照（块/表/在途reservation），kv_final保存最终状态；大负载结果会变大。块号不复用，但释放的容量可复用；Ready也可能是部分有效页。碎片仅统计Ready尾部空位，峰值包括曾预留块。

## 尚未实现

多worker/P→D交接、共享KV引用接口、写时复制、传输保护/跨卡副本、实际容量实测验证、元数据/对齐、外部碎片地址布局、未来增长预留/抢占恢复、SLO准入、真实模型/4090测量和拟合。引用/GPU字段是实际结构，不等于这些机制已启用。当前一池最多一在途事务。

## 本地环境

Windows C:/Users/yh/Desktop/sim/llm-serving-lab；WSL /mnt/c/Users/yh/Desktop/sim/llm-serving-lab，Ubuntu、Python3.10+。只用标准库运行；数据依赖在项目.venv，由已有/root/.local/bin/uv安装（系统缺ensurepip），requirements-data.txt记录依赖。不安装全局包。

Qwen tokenizer三份JSON在assets/tokenizers/qwen2.5-7b-instruct；生成负载workloads/generated/sharegpt-100-fixed/requests.json已有100请求。原ShareGPT位于相邻router-simulator/datasets/sharegpt，仅只读。assets、.venv、workloads/generated、output均被Git忽略，新机器不能假定存在。prepare配置和历史学习细节在各章节。

## 最新：服务器4090独立容量profile

configs/rtx4090_memory.json引用configs/profiles/rtx4090_qwen3_8b_bf16.json；来源只读服务器nvidia-smi与现有Qwen3-8B小JSON。6卡各24564MiB，但只模拟1卡。Qwen3是36层、8KV头、head_dim128、BF16，权重张量16381470720字节。此前Qwen2.5-7B目录当前不存在，不虚构同一模型配置。

90%使用比例、2GiB运行时预留、BF16 KV、每块16token、元数据0/对齐1是假设。容量换算1972块，144KiB/token、2.25MiB/块、KV预算4652544409字节。不是vLLM实测块数，权重张量大小仅估计VRAM；teaching耗时仍未校准。memory_budget.py严格解析并计算，结果memory_budget/provenance可复现，禁止profile和人工块数混用。若同时用roofline，KV结构/精度需一致。

新入口：python3 run_simulation.py --config configs/rtx4090_memory.json --output output/rtx4090_memory.json。61测试中59通过、2分词集成跳过；test_memory_budget覆盖公式/对齐/边界、相对路径、互斥、服务器值与模型冲突。配置文件运行时全本地，无需服务器。已有ShareGPT长度为Qwen2.5分词，之后校准Qwen3需对应分词器重准备。下一步先看第13章与假设，不自动部署或继续完整校准。
