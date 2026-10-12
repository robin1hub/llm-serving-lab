# 第 5 章：真实文本如何变成模拟负载

## 本步学什么

模拟器只需要长度和到达时间，不需要生成真实回答。先离线处理 ShareGPT，随后模拟可以反复复用同一负载：

```text
ShareGPT 文本 → 提取第一轮问答 → Qwen tokenizer → 长度 + 到达计划
                                                        ↓
                                             requests.json + manifest
                                                        ↓
                                                原事件模拟器
```

这是师兄原始 loader、离线 tokenizer 和 arrival plan 流程的简化版。事件内核没有改变。真实数据长度不等于真实计算耗时，结果仍标记 teaching_assumption。

## 为什么分成准备与执行两步

分词属于输入准备，事件内核只需要长度和到达时间。将二者分开有三个目的：换策略不重复分词、两策略使用完全相同的请求、基础模拟不依赖 transformers。真实 tokenizer 的导入发生在 ChatTokenizer 初始化内部，ijson 的导入发生在读取函数内部；仅运行模拟不会加载它们。

当前只保存 token 数，不保存 token ID。因为尚未模拟 prefix 匹配或模型内容，长度已经足够；以后要研究前缀复用时，才需要扩展数据结构保存真实 ID。

## 实现职责：从哪个函数看起

| 文件 / 函数 | 输入 | 输出与职责 |
| --- | --- | --- |
| prepare_workload.py / main | 准备配置路径 | 校验配置、组织分词、生成 manifest、发布目录 |
| dataset.py / iter_sharegpt | 原始 JSON 路径 | 流式逐条给出对象，不整表加载 |
| dataset.py / first_pair | 一条对象 | 第一组用户/助手文本，或 None 表示不支持 |
| dataset.py / ChatTokenizer.lengths | 用户文本、参考回答 | 输入/输出 token 数 |
| dataset.py / prepare_requests | 记录迭代器、分词器、过滤与到达参数 | 请求列表和准备统计 |
| config.py / load_experiment | 模拟配置 | 加载标准请求、检查 manifest 是否匹配 |

LengthTokenizer 是一个接口约定：只需要 lengths(prompt, answer)。真实实现使用 Qwen，单元测试用小型替身。这样过滤和到达逻辑可以独立测试，另有本地真实 tokenizer 集成测试检查模板开销。

prepare_requests 按“提取问答 → 分词 → 检查长度 → 生成到达时间 → 创建 Request”顺序处理。无效记录不会消耗一个到达间隔；只有接受的请求进入模拟负载。request_id 使用源文件行序号，例如 sharegpt-000006，因此跳过记录不会使后续 ID 改成另一条记录。

## 分词代码如何对应设计

```python
prompt_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=True,
    add_generation_prompt=True,
)
answer_ids = tokenizer.encode(answer, add_special_tokens=False)
input_tokens = len(prompt_ids)
output_tokens = len(answer_ids)
```

第一步让输入符合该模型的对话格式，第二步只计参考回答文本。len 数的是 token ID 的数量，不是字符、单词或字节。分词器不运行神经网络，不预测耗时，也不生成答案。

output_tokens 是离线规定的模拟终止长度；调度器不能据此优先挑真实短输出。原事件内核仅在判断该请求是否完成时使用它。

## 准备配置与模拟配置分开

- configs/prepare_sharegpt.json：原始文件、tokenizer 目录、请求数、长度限制、到达方式和输出目录。
- configs/sharegpt.json：选择准备好的 requests.json，然后指定原有策略和教学耗时。

所有准备路径相对准备配置所在目录。默认读取相邻师兄项目的 ShareGPT 原文件，不复制、不修改它。tokenizer 放 assets/tokenizers/qwen2.5-7b-instruct，只包括 tokenizer.json、tokenizer_config.json 和 config.json，不包括模型权重。

本机 tokenizer 文件来自已授权访问的服务器 /data/HF_MODELS/Qwen2.5-7B-Instruct。准备时强制只加载本地文件，不执行仓库自定义代码、不隐式下载；说明中记录这些文件的 SHA-256，不能仅靠 model_id 判断文件身份。

## 如何选请求

默认选前 100 个有效记录（first_valid），不是从全数据集均匀随机抽样，不能声称代表整个 ShareGPT。seed 只控制泊松到达，不改变样本选择。

每条记录只接受最前面连续的 human/gpt 问答，忽略后续轮次；不完整、空文本或角色不符合条件的记录跳过。系统消息开头的记录当前不支持。跳过原因与数量只统计已扫描的前缀，不是全数据集质量统计。

输入经过 Qwen 的 chat template，并添加生成提示；模板可能自动补系统提示。输出长度是参考回答文本的 token 数，不添加 EOS。真实模型可能生成不同长度；后续与服务对比需要固定或记录实际生成长度，不能假设参考回答必然等于模型输出。

默认输入不超过 4096 token、输出不超过 1024 token。超限样本跳过，不截断；这些是实验过滤条件，不是已实现的 GPU 容量限制。如果有效数量不够，明确报错，不悄悄少跑。

## 到达计划

- fixed：首请求 0ms 到达，之后间隔 1000 / requests_per_second 毫秒。
- poisson：首请求仍为 0ms，后续间隔来自指数分布；同参数和 seed 重复得到相同计划。

固定首请求为 0ms 是本实验约定。ShareGPT 本身没有提供本实验需要的服务到达时间。到达计划独立于 GPU 执行；改变服务速度不会改变请求何时到达。

## 本地环境与运行

基础模拟器仍只需标准库，只有数据准备需要额外依赖。依赖仅装在本项目 .venv。

本机 WSL2 已通过现有 uv 安装。重新建立环境可用：

```bash
uv venv --python python3 .venv
uv pip install --python .venv/bin/python -r requirements-data.txt
```

若系统支持 python3 -m venv，也可创建后用该环境的 pip 安装同一依赖文件。本机默认 Python 缺少 ensurepip，所以使用 uv，没有修改系统 Python 或师兄环境。未安装 PyTorch 的 tokenizer 提示是预期的，本步不需要模型执行库。

```bash
cd /mnt/c/Users/yh/Desktop/sim/llm-serving-lab
.venv/bin/python prepare_workload.py --config configs/prepare_sharegpt.json
python3 run_simulation.py --config configs/sharegpt.json --summary --output output/sharegpt.json
.venv/bin/python -m unittest discover -s tests -q
```

默认生成 workloads/generated/sharegpt-100-fixed/requests.json。输出目录已存在时拒绝覆盖；重复实验应复制准备配置并换 output_dir，随后让模拟配置引用新路径。可以修改 arrival.mode 为 poisson，同时选新输出目录。

--summary 只缩短终端打印，完整轮次、事件与 token 时间仍写入结果。当前没有大规模 trace 压缩，先用 100 个请求学习。

## 文件与可追溯性

requests.json 只包含稳定 request_id、arrival_ms、input_tokens、output_tokens，无原始文本。requests.manifest.json 保存源文件校验和、tokenizer 文件校验和、模板、依赖版本、实际选中的源行、准备参数、跳过统计和长度分布。

数据结构可以理解为：requests.json 是“执行用的题目”，manifest 是“题目如何制作的说明”。prepare_workload.py 先在同一父目录的临时目录写完两份文件，再 rename 发布目标目录，避免中途失败留下看似完整的负载；已存在的目标目录直接拒绝覆盖。

manifest 的 model_id 是配置中的说明字符串，不会远程解析模型版本。身份核对实际依靠本地 JSON 文件校验和及保存的模板。运行侧核对 requests_sha256 和 selected_count，不会再次读取原始 ShareGPT 或重新分词。

运行时发现同名 manifest 会核对请求文件校验和和数量，并把说明复制进结果。如果手工修改生成请求，旧 manifest 会失配；应重新准备而非继续沿用旧说明。已有小型人工负载没有 manifest 也可照常运行。

原始数据、assets/、workloads/generated/、.venv/、output/ 都不上传 GitHub。仓库保存工具、配置、测试和讲解；其他机器需要自行配置源数据和本地 tokenizer。

## 阅读重点与练习

先看准备配置和 manifest 的 length_statistics，不必读完数据解析代码。第二遍看 simulator/dataset.py 的三个重点：真实分词、流式读取、到达计划。prepare_workload.py 负责组织准备和发布，不处理 GPU 调度。

练习：1 req/s 改成 2 req/s，输入输出长度是否应该改变？将 fixed 改为 poisson，保持 seed 后两次准备的时间是否一致？这些变化会增加或减小排队，但不能证明 4090 的真实性能。

下一步再学习 batch 预算和 KV 容量，之后替换并校准耗时模型。

## 本次实际验证（2026-10-12）

从原文件前 141 条选到 100 个有效请求，41 条因第一组问答格式不符合规则被跳过。输入长度 30–3176 token，平均 340.15；参考输出 1–701 token，平均 239.77。

默认 1 req/s 固定到达的模拟完成全部 100 个请求，执行 9543 轮，累计输出 23977 token，最后一个完成事件为 99394ms。输出数逐请求核对正确，结果保留 manifest。以上时间来自教学耗时，不是 4090 的实测结果。

16 个测试在项目 WSL2 环境全部通过，包括真实 tokenizer 的模板开销、首轮过滤、泊松重复性、非法参数和失配 manifest；基础标准库测试仍可运行，不具备分词依赖或文件时集成测试会明确跳过。
