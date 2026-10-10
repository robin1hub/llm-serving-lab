# 第 2 章：Batch 与优先级

当前 Prefill batch 仅一个请求，Decode batch 包含所有已 Prefill、尚未完成的请求。一轮各产生一个 token，时间只推进一次；完成的请求在轮末退出。进入活跃集合代表可被选择，不代表正在执行。

默认例子：0–8ms Prefill A，8–12ms Prefill B，12–14ms Decode [A,B]，14–16ms Decode [A]。B 首 token 提前，但 A 的首个 ITL 为 6ms。

严格 Decode 优先则在 8ms 继续 A，直到 12ms 完成，之后才处理 B。A 的 ITL 为 2ms，B 的 TTFT 为 13ms。当前单请求 Prefill 加严格排空 Decode 会退化为串行。

运行 configs/basic.json 与 configs/decode_first.json 比较。整轮 Decode 固定 2ms 是教学假设，不代表真实 GPU 性能。事件驱动的同样规则见下一章。
