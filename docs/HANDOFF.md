# 项目交接：当前状态

更新时间：2026-10-13（Asia/Hong_Kong）。公开仓库：https://github.com/robin1hub/llm-serving-lab 。

## 先读什么

AGENTS.md → 本页 → ARCHITECTURE.md → 当前章节08-token-budget.md，再看对应代码和测试。LEARNING_LOG.md是历史，不能把旧阶段的未实现列表当作当前状态。

## 用户目标与方式

用户希望理解LLM serving并最终研究D端压力。代码基础较弱，先讲机制、流程图、状态表和小例子，再读少量Python；中文原因注释和编号重点必需。不要求一次读完整文件，不自动推进后续机制。只改本项目，不改师兄项目、服务器或全局依赖。

## 当前已实现

- 单卡事件：到达/完成/唤醒按时间与事件优先级排序，busy阻止重叠。
- 只读Scheduler、不可变快照和BatchPlan；Prefill一次一个，Decode按完成后轮转。
- scheduler.policy两种；max_batch_requests、max_tokens_per_batch、prefill_chunk_tokens正整数/null或省略。null/省略不限制，兼容旧配置。
- Prefill分块维护prefill_done，完成才累加；未完成放waiting队首，最终块才产生首token。首块开始时间保留，不被续块覆盖。
- ShareGPT单轮离线准备、本地Qwen tokenizer、fixed/poisson到达、first_valid前N有效筛选、长度过滤与manifest。原始文本/生成数据/分词文件不提交。
- JSON结果保存requests/steps/plans/trace、有效scheduler参数、manifest和summary。TTFT/总延迟按请求，ITL按间隔；最近秩P50/P95；吞吐窗口首到达至末完成。

## 最新验证与学习入口

```bash
python3 -m unittest discover -s tests -q
python3 run_simulation.py --config configs/token_budget.json --output output/token_budget.json
```

31个标准库测试中29通过，2个分词集成因系统Python缺依赖跳过。默认ShareGPT的requests/steps/summary与引入预算前一致。

分块例子A在0ms到达输入5输出3，B在3ms到达输入2输出2；token预算3、chunk2。A输入三块2+2+1，0–4/4–8/8–10ms；B在10–14ms Prefill；14–16ms Decode AB，16–18ms Decode A。A输出10/16/18ms，B14/16ms。结果output/token_budget.json，本地被Git忽略。

当前用户阅读第8章，先理解中间块没有输出、输入进度与输出数量不同、分块后还需策略才能让Decode获得机会。

## 未实现与下一步

未实现KV容量/资源准入、混合P/D batch、阶段交替、抢占恢复、SLO优化、多卡传输、真实模型或4090校准、失败/截止终止。当前优先级仍可能连续选择同一阶段，不能宣称分块已经改善ITL。下一步确认用户理解后再讨论混合或交替以及简单KV容量，不直接部署vLLM。

耗时假设：Prefill按本块输入数线性计算，Decode整轮固定2ms，无分块启动成本，也不体现上下文负担。真实负载不等于真实GPU速度。

## 本地环境

Windows：C:/Users/yh/Desktop/sim/llm-serving-lab；WSL：/mnt/c/Users/yh/Desktop/sim/llm-serving-lab。基础模拟标准库；数据准备用本项目.venv，requirements-data.txt由已有/root/.local/bin/uv安装（系统缺ensurepip）。Qwen tokenizer三份JSON在assets/tokenizers/qwen2.5-7b-instruct。ShareGPT准备目录已存在，重新准备换output_dir，不覆盖。