# 校准材料入口

当前只有方案与空白记录模板，未测量、未拟合；不会被模拟器自动加载。

先看 [第15章](../docs/chapters/15-calibration-plan.md)，然后看 plan.json 和 templates/。

| 文件 | 用途 |
| --- | --- |
| plan.json | 建议实验矩阵、控制变量和拟合/验证划分 |
| templates/metadata.json | 每次运行的模型、设备、软件和有效配置 |
| templates/capacity.json | 每次启动的引擎实际容量及原始证据 |
| templates/batches.jsonl | 每行一个实际执行轮次的工作量和耗时 |
| templates/requests.jsonl | 每行一个请求的端侧时间、实际长度和流式事件 |
| templates/comparison.csv | 按形状对照预测与实测，当前只有表头 |

复制到 output/calibration/<run_id>/ 后填写。模板的 null 表示尚未记录，不是0；record_status=template不能作为实测数据。真实记录改为 measured，同时填原始证据。实际命令、启动日志、trace、客户端原始响应等也放在该目录，按metadata引用，原始文件不覆盖。实际数据不提交Git。

metadata的notes记录缺失字段原因，未知值保留null。JSONL每行一个完整JSON，不是一个大数组。采集器、计时插桩、拟合器与模板自动校验尚未实现；本步不更改模拟器行为，也不启动服务器任务。

服务器选择检查见 [server-survey.md](server-survey.md)：目前推荐alpha-wang复用完整Qwen3-8B，GPU0只是候选，尚未安装/启动。
