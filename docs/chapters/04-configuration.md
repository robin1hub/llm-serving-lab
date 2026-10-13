# 配置驱动实验：把“实验条件”和“执行机制”分开

本章保留初学阶段的 `scheduler.policy` 写法（仍兼容）。当前 configs 中的实验已迁移为独立 `scheduler_profiles` 与 `workers` 引用；配置方式见[第9章](09-worker-profiles.md)，耗时与路径语义保持相同。

以前想改 A 的长度需要改示例 Python，现在改负载文件即可。事件内核仍按原规则运行，本步没有新增显存或真实性能模型。

## 设计流程

```text
配置 JSON → 读取与校验 → Experiment（请求、策略、耗时）
                                 ↓
                           原事件模拟函数
                                 ↓
                      终端摘要 + 结果 JSON
```

simulator/config.py 只负责输入，simulator/engine.py 只负责执行，run_simulation.py 连接两者并保存输出。这样以后接 ShareGPT 或真实耗时表时，输入变化不必混入事件处理代码。

## 文件分工

- configs/basic.json：选择负载路径、Prefill 优先和教学耗时。
- configs/decode_first.json：同一负载和耗时，改成严格 Decode 优先。
- workloads/two_requests.json：A/B 到达时间、输入输出长度。当前是人工 token 长度数据，不是原始文本数据集。
- output/result.json：默认结果文件，运行会覆盖同名文件；output/ 不提交 Git。做对照时用 --output 指定不同文件。

## 如何运行

```bash
cd /mnt/c/Users/yh/Desktop/sim/llm-serving-lab
python3 run_simulation.py --config configs/basic.json --output output/prefill.json
python3 run_simulation.py --config configs/decode_first.json --output output/decode.json
python3 -m unittest discover -s tests -q
```

只有 --config 必填，--output 默认为 output/result.json。结果目录不存在时自动创建。CLI 的配置路径和输出路径相对当前工作目录；配置内 workload.path 则相对配置文件所在目录。basic.json 中 ../workloads/two_requests.json 表示从 configs/ 回到项目目录，再进入 workloads/。

## 配置内容

```json
{
  "workload": {"path": "../workloads/two_requests.json"},
  "scheduler": {"policy": "prefill_first"},
  "timing": {"prefill_ms_per_input_token": 2.0, "decode_ms_per_step": 2.0}
}
```

- workload.path 选择请求文件。
- scheduler.policy 仅接受 prefill_first、decode_first。
- prefill_ms_per_input_token 表示每个输入 token 的教学耗时。
- decode_ms_per_step 表示整个 Decode batch 一轮的教学耗时，与成员数无关。

每条负载记录包含 request_id、arrival_ms、input_tokens、output_tokens。ID 唯一；到达时间非负；长度必须是正整数。输出长度包含 Prefill 产生的首 token，只用于终止判断，不用于策略预测。空请求数组允许运行，结果为空。

JSON 不支持注释，说明放在本文件。字段全部必填，不接受未知字段、重复键、NaN 或 Infinity，防止拼写错误和覆盖值导致实验跑错。加载失败会打印原因并返回非零退出码，先校验完再启动模拟。

## 输出里看什么

requests 保存每个请求的原始参数、开始时间、等待时间、token 时间、TTFT、ITL、端到端延迟与日志。steps 保存 GPU 各轮区间和成员，trace 保存事件处理顺序。timing、policy 和路径记录实验来源；实际请求也写入结果，因此后来改负载文件不影响已有记录的解释。

timing_source 明确为 teaching_assumption，不能把结果当成 4090 实测。当前没有计算吞吐或分位数汇总，只有终端摘要和请求级数据。

## 先做一个小练习

复制 basic.json 成自己的配置，或者只改 workloads/two_requests.json 中 B 的 arrival_ms 为 9。先预测 B 何时开始，再运行查看 steps。要注意两份配置引用同一负载，改这个负载会同时影响它们；需要独立实验时复制负载并更新路径。

无需读懂全部加载代码。先看配置、负载和运行结果；第二遍看 simulator/config.py 的三个重点，最后看 run_simulation.py 的“读入 → 执行 → 保存”。

## 后续边界

显存配置需要先实现权重/KV 占用和资源不足时的处理；batch 上限需要改变选批逻辑；ShareGPT 需要真实 tokenizer 与到达计划；4090 校准需要实测耗时。当前不接受这些未实现字段。

整理前验证：22 个测试在 WSL2 通过；当前测试结果见交接记录，包括两策略 CLI 输出、从其他目录运行、未知/缺失字段、重复 ID、非法长度、缺文件和非法 JSON。
