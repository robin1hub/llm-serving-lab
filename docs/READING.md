# D侧高压力调度：论文与实验入口

需要交给另一个session继续检索或设计实验时，先读[自包含研究报告](D_HEAVY_RESEARCH_BRIEF.md)：包含当前实现边界、D→P联合决策、零阶优化方法文献、实验草案和可复制任务。以下保留第一轮系统论文筛选，尚未实现研究策略。

检索日期：2026-10-15（Asia/Hong_Kong）。目标是研究D侧积压、KV容量和延迟约束下的调度。本页是初步筛选，不是完整综述或论文复现；检索使用论文原文、arXiv及会议官方页。模拟器仍处于第22章，以下实验均为后续建议，没有自动实现。

## 先分清问题

这是本项目的研究拆分，不是某一论文的结论：

- **全局过载**：所有D资源都处理不过来，需要准入、限流、排队或扩容；只换路由不增加总容量。
- **分布不均**：部分D很忙、部分有余量，可以改初始分配或迁移。
- **容量与时延限制不同**：KV还能放下，也可能已经无法满足生成延迟；同样人数的请求也可能有不同上下文/剩余工作量。
- **评价目标不同**：TPOT均值、单次ITL/TBT上限、P95间隔和整请求SLO达标不是同一指标，比较论文前先核对定义。

## 论文清单

| 论文与版本 | 已核对的切入点 | 对我们项目的意义 |
| --- | --- | --- |
| [Mooncake: Kimi’s KVCache-centric Architecture for LLM Serving，2024 v1](https://arxiv.org/html/2407.00079v1) | 第6节Overload-oriented Scheduling：预测未来D负载、提前拒绝，讨论简单拒绝策略的负载振荡 | 优先读过载部分；先做P准入基线，不必先实现多级缓存 |
| [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving，OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) | PD分离、按TTFT/TPOT目标配置两阶段资源与并行方式 | 作为固定PD架构及goodput目标的基础；不是单独解决D过载的完整答案 |
| [TetriInfer / Inference without Interference: Disaggregate LLM Inference for Mixed Downstream Workloads，2024 v1](https://arxiv.org/html/2401.11181v1) | §3.3.4/§3.4及§5.2.3：预测生成长度，做D间分配和D内存预留；收益受预测误差影响 | 多D后比较人数基线与工作量/容量基线，显式测试长度预测误差 |
| [Llumnix: Dynamic Scheduling for Large Language Model Serving，OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/sun-biao) | 运行中重调度与请求/KV状态迁移，处理负载不均和隔离 | 研究“入场分配后，还能否迁走”；不把当前项目后来的PD产品能力当作2024论文内容 |
| [TaiChi / Prefill-Decode Aggregation or Disaggregation? Unifying Both for Goodput-Optimized LLM Serving，2025 v1](https://arxiv.org/html/2508.01989v1) | §3.3 Flowing Decode Scheduling：P-heavy/D-heavy之间迁移Decode，利用延迟预算 | 研究D紧张时能否让P-heavy分担，但必须计入P干扰和迁移成本 |
| [STAR: Decode-Phase Rescheduling for LLM Inference，arXiv 2510.13668 v2，2026更新](https://arxiv.org/abs/2510.13668v2) | 当前摘要：持续预测剩余生成长度，联合当前/未来负载做Decode重调度 | 比较仅看当前占用与预测后续工作；模型隐状态预测器不在当前模拟器内 |
| [Large-Scale LLM Inference with Heterogeneous Workloads: Prefill-Decode Contention and Asymptotically Optimal Control，2026 v2](https://arxiv.org/abs/2602.02987v2) | 当前摘要：排队网络、状态相关服务率、gate-and-route联合控制P准入和D路由 | 后续理论线索；多GPU极限的保证不能直接当作两卡有限系统的保证 |

核对深度：Mooncake/TetriInfer/TaiChi已查看上述原文章节；DistServe/Llumnix核对官方会议页和摘要；STAR与2026控制论文目前只核对arXiv摘要/版本，没有精读算法、证明或验证其结果。

## 版本注意

Mooncake另有[FAST 2025正式版本](https://www.usenix.org/conference/fast25/presentation/qin)，标题为“Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot”。本页推荐过载调度的章节编号来自**2024 v1第6节**，不要把不同版本标题、章节和实验数字混用。

STAR的2510.13668当前v2标题为“STAR: Decode-Phase Rescheduling for LLM Inference”；早期检索可能显示“Adaptive Rescheduling in Prefill-Decode Disaggregated LLM Inference”。此处按已核对的2026-05-04 v2标识，不自行推断会议归属。

## 建议阅读顺序

1. DistServe先看架构图和优化目标，对应我们已做的1P1D。
2. Mooncake直接看v1第6节，围绕“什么时候允许新P进入、应预测哪个未来时刻的D负载”做笔记。
3. TetriInfer看D分配/预留与预测误差实验，再看TaiChi的D流出/流回逻辑。
4. 有迁移基础后读Llumnix/STAR；最后再读2026排队控制论文的建模和假设。

每篇先写清：观察哪些状态、预测什么、何时决策、有哪些动作、优化哪个指标。暂时不必逐行看实现代码。

## 可在模拟器中逐步验证的实验（我们的建议）

| 优先级 | 问题与基线 | 还缺的功能 |
| --- | --- | --- |
| 第一组 | 同一1P1D下比较不控制P、按当前D负载控制、按预计P完成时的D负载控制 | 先完成目标容量等待，再加入准入/拒绝与SLO统计 |
| 第二组 | 多D时比较轮询、最少请求数、KV占用、预计工作量路由 | 1P多D、独立路由接口 |
| 第三组 | D负载不均时，迁移收益是否超过KV复制与暂停成本 | 运行中D→D迁移与保护/恢复 |
| 第四组 | D全局紧张且P有余量时，P-heavy分担是否增加SLO达标请求 | P-heavy/D-heavy混合角色及干扰模型 |

第一组适合当前路线，尚未选定具体新算法。到达负载应同时包含稳定与突发、长短输出混合，报告SLO达标完成请求数/时间、TTFT、ITL分布、拒绝比例、等待与KV占用。指标口径由实验明确固定，不直接套用不同论文的数字。

**长度预测器与现有Predictor不同**：现有Predictor估计一批执行多久；这些调度工作还可能预测请求剩余生成多少token。在线策略不能读取trace中的真实最终输出长度。若做oracle对照，只作为标清身份的上界；可控误差的长度估计用于敏感性实验，不称为已训练出真实预测器。

当前任务只完成文献筛选与阅读规划，运行机制/实现路线不变。下一步可先讨论Mooncake过载章节，再决定如何设计实验。
