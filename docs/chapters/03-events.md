# 单卡事件队列：先理解设计，再阅读代码

本步沿用单请求完整 Prefill、活跃请求组成 Decode batch 的规则，耗时仍为教学参数。先保持 Python：事件语义与语言无关，目前阅读和验证比执行性能更重要。没有部署 vLLM，也没有进行 4090 校准。

## 三种容器

| 名称 | 保存什么 | 回答什么问题 |
| --- | --- | --- |
| future | 未来的到达、完成、调度唤醒事件 | 下一件事什么时候发生？ |
| waiting | 已到达、还没开始 Prefill 的请求 | 哪些新请求可以被选择？ |
| active | 已完成 Prefill、尚未生成结束的请求 | 哪些请求可以参与 Decode？ |

与旧版 pending 不同，waiting 不包含未来请求：只有处理到达事件后才加入。future 里的未来请求不向调度器暴露。

## 三种事件

- arrival（到达）：加入 waiting；GPU 忙时只能等待，空闲时安排调度唤醒。
- wakeup（唤醒）：选择下一轮，设置 GPU 忙碌，注册未来完成通知。这里不产生 token，也不把时间推进到结束。
- complete（完成）：记录 token，加入或退出 active，释放 GPU，再唤醒调度。

busy 表示 GPU 是否已有计算在途，不等于程序正在实际计算。单卡只需要这个标记防止计算区间重叠。members 固定本轮成员，执行过程中不改变。

## 默认例子

0ms 到达 A，启动 Prefill 后把 8ms 完成通知放入 future。下一件事是 3ms 到达 B，所以时间先跳到 3ms。B 加入 waiting，但 GPU 忙碌。8ms 完成通知才让 A 产生首 token、释放 GPU；此时按 Prefill 优先选择 B。

最终轮次仍是 0–8 Prefill A、8–12 Prefill B、12–14 Decode [A,B]、14–16 Decode [A]。改变的是时间推进机制，结果不变。

## 最小循环

```text
取最早事件 → 时间跳到事件时刻 → 更新状态 → 必要时注册新事件
```

代码只有 `now_ms = event.time_ms` 推进时钟。开始计算时的 `end_ms` 是预测的完成时刻，用于注册事件；不是已经完成的事实。

同时间事件按到达、完成、唤醒顺序处理，再按注册序号排序，保证调度看见该时刻的全部到达和完成。事件优先级与 P/D 执行优先级不同。多个同刻到达只登记一次唤醒，避免重复启动。GPU 忙时不轮询，等完成通知再唤醒。

## 师兄实现与教学简化

参考相邻 router-simulator 的 docs/developer/event-kernel.md 和 csrc/include/router_simulator/queue_coordinator.hpp：同样将未来事件排序与请求、资源状态分开。师兄采用两级堆、提交顺序和 generation 失效机制；这里仅一个 heapq 最小堆，没有取消、传输或多线程。并未完整复刻其事件优先级和事务机制。

## 如何阅读与运行

先运行 `python3 run_simulation.py --config configs/basic.json`，重点看 0、3、8ms 的日志。再打开 simulator/engine.py：按【重点 1】容器、【重点 2】时钟、【重点 3】到达、【重点 4】完成、【重点 5】注册完成依次阅读。数据类、嵌套函数、结果封装可以第二遍再学。

验证用 `python3 -m unittest discover -s tests -q`。测试检查手算 token 时间、请求进度与资源互斥，覆盖两种策略、同刻到达、计算中到达、单输出和固定种子的 workload；GPU 执行区间不得重叠。日志措辞和内部容器实现不要求与旧版相同。

练习：B 在 3ms 到达时为什么不生成 token？把 B 到达时间改为 8ms，到达和完成同刻发生时，调度器应该看见哪些请求？
