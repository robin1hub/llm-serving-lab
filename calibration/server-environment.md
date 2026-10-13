# 单卡校准服务器环境

用户已改为“先不校准，把环境准备好”。本步只准备独立依赖和检查记录，不启动vLLM服务、不加载模型、不测容量/耗时。王、杜两台的新环境均已通过导入、固定版本/导入路径和159包依赖检查。GPU运行、模型加载与校准均未执行。

## 隔离位置与版本

两台服务器统一目录：/data/llm-serving-lab-calibration/。

| 子目录 | 用途 |
| --- | --- |
| venv/ | 本项目Python环境，不复用系统site-packages |
| tools/ | 项目内uv安装工具 |
| logs/ | 安装日志、环境检查、精确依赖清单 |
| cache/ | 项目内缓存位置，后续运行应显式指定 |
| runs/ | 未来校准数据，当前无推理结果 |
| models/ | 杜服务器曾开始下载的独立模型副本；已停止，不能当作完整模型 |

版本约束参考[vLLM0.11.0官方依赖](https://github.com/vllm-project/vllm/blob/v0.11.0/requirements/cuda.txt)。固定vLLM0.11.0、torch2.8.0、transformers4.56.1；uv0.9.4。选择固定版本是为了后续核对内部每轮执行计时接口，不追求安装最新版。完整解析版本以后由requirements-resolved.txt记录，requirements-server.txt仅固定主要依赖。系统Python为3.10.12；驱动570.124.06报告CUDA12.8支持，最终torch CUDA构建版本以检查报告为准。

系统缺少ensurepip，所以用项目内uv创建/安装环境，未安装系统python3-venv。pip --target只用来把uv写入tools目录，后续uv pip安装到明确的venv，未执行全局pip升级或修改已有conda环境。此前杜服务器verl环境仅只读导入检查，确认版本存在；它不是本项目环境。

## 复现安装步骤

在尚未创建该项目目录的新服务器，使用下列命令。已准备的机器不需要重复运行；不要对已有其他项目环境执行这些命令。

```bash
mkdir -p /data/llm-serving-lab-calibration/tools
python3 -m pip --isolated install --no-cache-dir --index-url https://pypi.org/simple \
  --target /data/llm-serving-lab-calibration/tools uv==0.9.4
/data/llm-serving-lab-calibration/tools/bin/uv venv --python /usr/bin/python3 \
  /data/llm-serving-lab-calibration/venv
/data/llm-serving-lab-calibration/tools/bin/uv pip install --no-cache \
  --python /data/llm-serving-lab-calibration/venv/bin/python \
  --index-url https://pypi.org/simple \
  vllm==0.11.0 torch==2.8.0 transformers==4.56.1
```

本次安装未开wheel持久缓存，减少磁盘重复占用；下载过程仍有临时空间开销。模型不包含在Python环境中，王服务器仍使用/data/HF_MODELS/Qwen3-8B；杜服务器独立下载副本未完成且保留.partial，不清理他人模型/缓存。

## 激活和缓存隔离

两机均提供项目内activate-server.sh，内容与本仓库同名文件一致：

```bash
source /data/llm-serving-lab-calibration/activate-server.sh
which python
```

激活本身不加载模型。脚本将HF_HOME、TORCH_HOME、XDG_CACHE_HOME、VLLM_CACHE_ROOT、VLLM_CONFIG_ROOT指向本项目，避免使用共享环境缓存。未固定CUDA_VISIBLE_DEVICES，实际启动时仍要由新的实验明确选择空闲GPU UUID。

## 仅依赖检查

将scripts/check_environment.py复制到项目目录后，使用本项目Python执行：

```bash
PYTHONDONTWRITEBYTECODE=1 \
/data/llm-serving-lab-calibration/venv/bin/python \
  /data/llm-serving-lab-calibration/check_environment.py \
  --project-root /data/llm-serving-lab-calibration \
  --output /data/llm-serving-lab-calibration/logs/environment.json
```

检查脚本先核对sys.prefix，再禁用可见GPU、把缓存定向到项目目录，导入torch/vLLM/transformers，执行uv pip check并保存freeze。它记录GPU运行未验证、模型未加载、校准未测量；dependencies_ready只说明依赖层准备完成，不表示推理已经通过。GPU运行和模型启动留到用户要求恢复校准时再做。

后续启动前重新确认空闲GPU与磁盘，记录GPU UUID，不根据一次0%利用率判定显存空闲。王服务器已完整保存目标权重，但目前六卡均有占用；不终止现有任务。两机原本磁盘99%使用，余量随其他任务变化，不能把本次快照当长期保证。

本次杜机使用官方PyPI安装；王机官方源慢，切换https://pypi.tuna.tsinghua.edu.cn/simple镜像，保持HTTPS和固定版本。杜机导入和uv pip check已通过，共159包；报告见environments/alpha-du.json，精确版本见alpha-du-requirements.txt。无模型加载或GPU运行测试。

## 已完成的验证

| 服务器 | 环境状态 | 主依赖 | torch CUDA构建 | 检查时磁盘剩余 |
| --- | --- | --- | --- | --- |
| alpha-wang | dependencies_ready；159包通过 | vLLM0.11.0/torch2.8.0/transformers4.56.1 | 12.8 | 43557502976bytes，约40.6GiB |
| alpha-du | dependencies_ready；159包通过 | 同上 | 12.8 | 57961078784bytes，约54.0GiB |

environments/alpha-wang.json和alpha-du.json保存原始检查报告；对应*-requirements.txt保存各159包精确版本，下一次复现优先使用该清单。包位置均落在项目venv内，Python3.10.12，uv pip check退出0。无模型/GPU计算或校准数据，GPU可运行性仍需以后验证。

用户要求暂停校准；王机可复用原完整权重，杜机额外权重下载已停止且partial保留。没有推理服务或采集进程在后台继续运行。服务器独立目录保留，后续无需重新安装主依赖。磁盘余量只是报告时间的快照。

## 后续启动状态（2026-10-13）

用户随后恢复王服务器实验，两次smoke均未成功：第一次UUID解析兼容问题，已修正为核对UUID后传数字卡号；第二次候选卡占2441MiB，启动保护拒绝。没有加载模型或取得有效容量/耗时数据。用户要求稍等，当前没有运行中的采集器，未终止他人任务。

现改做本地师兄模拟器参照，见第16章与calibration/reference。采集脚本已保存scripts/collect_batches.py、worker_probe.py、batch_records.py和launch_remote.py，真实Worker计时仍未验证。之后恢复前需同步最新版脚本并复查空闲GPU。
