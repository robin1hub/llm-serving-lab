# 校准材料入口

最新：2026-10-15完成[A6000受控混合验证](../docs/chapters/20-a6000-mixed.md)。直接使用本地3B基础版，纯P/D 12组拟合、4组留出后冻结独立A6000 profile；再测9组1P+多个D/180目标轮，平均绝对误差28.80%→5.64%，最大10.35%，混合数据未参与拟合。正式采集器scripts/collect_mixed.py及launcher --mixed已通过GPU运行；分析仅选择验收的目标iteration，排除驻留准备阶段的其他混合轮。证据为measurements/a6000-qwen25-{baseline,fit,controlled-mixed}-20261015.json。GPU已释放，没有跨机权重拷贝；以下保留此前各阶段记录。

2026-10-15新增[第19章](../docs/chapters/19-mixed-validation.md)：旧实测219混合轮次、139生成重复、7原案例，纯P/D参数沿用后平均绝对误差21.77%。专门补采未运行成功；用户已要求停止模型拷贝。正式工具scripts/analyze_mixed.py，受控采集器仅ignored草稿，无新GPU结果。

最新见[第18章](../docs/chapters/18-calibration-validation.md)：已用旧12组fit求参数、旧4组验证、重新采8旧/8新形状测试，同一新数据平均误差46.27%→5.95%，最大13.65%。新结果未回流拟合。参数configs/calibrated_qwen25_3b.json，来源与统计measurements/du-qwen25-{fit,fresh-validation}-20261014.json；仅适用已测模型/硬件/运行边界，Mixed未验证。以下保留此前材料入口。

当前新增[第17章完整硬件对照](../docs/chapters/17-measured-comparison.md)：杜机已有Qwen2.5-3B、GPU4、16组矩阵/400次生成完成，基线误差54.39%，师兄默认解析预测同一实测误差54.17%，不拟合；三方表见第17章第6节。当前有方案、模板、独立环境及采集入口；[王服务器共享卡smoke](wang-shared-smoke.md)已成功，取得实际容量和少量P/D数据，尚未完成真实拟合。已完成师兄模拟器参照拟合；参照不是硬件校准，见[第16章](../docs/chapters/16-reference-comparison.md)及reference/result.json。可选configs/reference_senior.json仅在显式运行时使用。

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

metadata的notes记录缺失字段原因，未知值保留null。JSONL每行一个完整JSON，不是一个大数组。真实采集入口scripts/collect_batches.py与独立worker_probe.py已实现，最小GPU运行已验证，杜机完整矩阵已采集；标准库batch_records.py本地测试通过。run_senior_reference.py已完成模拟器参照拟合，不是硬件拟合器。模板自动完整验收尚未实现。

服务器选择检查见 [server-survey.md](server-survey.md)：优先alpha-wang复用完整Qwen3-8B；此前GPU0候选已被其他任务占用。两机独立依赖环境已准备，用户随后恢复王服务器实验，但启动失败且卡被占用，用户随后同意用占2441MiB但利用率0的卡试跑，GPU0以80%预算成功完成smoke。

当前改做本地模拟器参照，王GPU0共享卡最小采集已成功，杜机已完成完整矩阵，正式拟合尚未进行。环境路径、固定版本和无GPU检查说明见 [server-environment.md](server-environment.md)。本步不发布实测profile。
