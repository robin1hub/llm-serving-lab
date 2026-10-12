# 第 6 章：结果统计

统计只读取已完成的 BatchRun，不改变调度。终端和结果 JSON 的 summary 来自同一模块 simulator/metrics.py。--summary 隐藏详细终端打印，完整请求和事件记录仍保存。

## 统计口径

- 完成请求数、输入 token 总数、实际输出 token 总数（包含首 token）。
- TTFT、端到端延迟：每请求贡献一个样本，输出 count、mean、p50、p95。
- ITL：汇总所有相邻 token 间隔，长输出请求贡献更多样本。单输出请求没有间隔，不填 0；这不同于先求每请求平均再平均。
- 分位数：最近秩法，排序后取 ceil(p×n) 项，从 1 开始编号。比如 [1,2,3,4] 的 P50 为 2，不做插值。第 5 章负载说明里的 median 是数学中位数，二者定义不同。
- 吞吐窗口：最早请求到达到最后请求完成，包括空闲、排队和排空，不包括首请求到达前时间。毫秒除以 1000 后再计算 req/s、token/s。
- 无样本的延迟或空负载吞吐写 null，不写 0 或无穷。

结果记录 percentile_method、itl_weighting 和 window.definition。吞吐包含全部窗口时间，不是稳态或 GPU 最大能力，耗时仍为教学假设。

## 同负载比较

```bash
python3 run_simulation.py --config configs/sharegpt.json --summary --output output/sharegpt.json
python3 run_simulation.py --config configs/sharegpt_decode_first.json --summary --output output/sharegpt_decode_first.json
```

两份配置仅策略不同，负载文件和耗时相同。先按第 5 章准备负载。

2026-10-12 本地结果：100 请求，输入 34015、输出 23977 token；每边 ITL 样本 23877 = 23977−100。

| 指标 | Prefill 优先 | 严格 Decode 优先 |
| --- | --- | --- |
| 平均 TTFT | 7543.74ms | 22425.36ms |
| TTFT P95 | 19842ms | 34882ms |
| 平均端到端延迟 | 18557.82ms | 22902.90ms |
| 端到端 P95 | 54854ms | 35144ms |
| 平均 ITL | 46.128ms | 2ms |
| ITL P95 | 2ms | 2ms |
| 统计窗口 | 99.394s | 121.830s |
| 请求吞吐 | 1.006 req/s | 0.821 req/s |
| 输出吞吐 | 241.232 token/s | 196.807 token/s |

Prefill 优先的 ITL 平均值大于 P95，是少量特别长的间隔拉高平均值。单看 P95 会遗漏更极端的停顿。严格 Decode 优先目前退化为串行，保护生成间隔却失去多请求组批机会。两策略等待分配不同，TTFT、ITL 和端到端尾延迟不必同时改善。

## 手算与代码

先读 simulator/metrics.py 的三个重点：请求样本、间隔样本、吞吐窗口。不需要阅读整个事件内核。

A/B 小例子：TTFT [8,9] 平均 8.5ms；ITL [6,2,2] 平均 10/3ms；窗口 16ms、输出 5 token，吞吐 312.5 token/s。这些数字与空负载、单输出、空闲间隙均有测试。

系统 Python 验证：20 个测试中 18 个通过，2 个分词集成测试因未安装数据依赖跳过；两策略真实 ShareGPT 模拟完整运行。项目 .venv 的真实分词在上一阶段已验证，本次不改变分词和事件行为。
