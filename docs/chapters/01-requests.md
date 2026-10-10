# 第 1 章：请求与排队

Prefill 处理输入并产生首 token；之后每轮 Decode 产生一个后续 token。输入 4、输出 3，教学 Prefill 每输入 token 2ms、Decode 每轮 2ms，在空闲 GPU 上输出时刻为 8、10、12ms。

输出长度包括首 token，所以只需要两轮 Decode。TTFT 是首 token 减到达时间，ITL 是相邻输出时间差，总延迟是最后 token 减到达时间。

如果串行执行，B 在 3ms 到达但 A 在 12ms 才完成，B 等待 9ms；输入长度 2 的 Prefill 再花 4ms，B 的 TTFT 为 13ms。到达、开始、完成是不同时间。

现在的数据结构位于 simulator/models.py；旧阶段示例不再单独保留。下一章学习多个请求怎样共享一轮。
