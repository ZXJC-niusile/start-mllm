# Chapter 0 环境准备与前置说明

> 如果你已有相应环境，可先核对所选路线的接口或硬件要求，再读 [前言](./前言.md)。
> 没有 GPU 可以先走第七章 API 路线；需要支持图像输入的可用接口、密钥与调用额度。

## 一、你需要准备什么

开始**概念阅读与 API 实践**，建议先具备：

1. 会写基础 Python（能写函数、能装 pip 包）
2. 能打开命令行（Windows 的 cmd / PowerShell，或 Linux/macOS 的 Terminal）

以下这些**有则更好，没有也能先读起来**：

- PyTorch 基础经验
- 对 Transformer、Token、Embedding 的初步认知
- 调通过 LLM 推理

如果暂时不熟悉，先按“概念优先、代码其次”读，后面逐步补齐。命令行、环境和路径不熟时，可先读 [基础篇 00](./foundation/00-环境命令行与Python项目基础.md)。

API 实践另需支持图像输入的可用接口、密钥与调用额度。**本地推理**建议理解张量的形状、设备与精度，并核对模型兼容性和硬件资源；**训练与微调是选学内容**，还需要 PyTorch 训练基础、数据与计算资源。

<a id="get-tutorial-code"></a>

## 二、获取教程代码

```bash
git clone https://github.com/datawhalechina/start-mllm.git
cd start-mllm
```

如果你暂时不会用 `git`，也可以在 GitHub 仓库页面点击 `Code` → `Download ZIP`，解压后进入 `start-mllm` 目录继续阅读和运行脚本。

后续章节中的脚本路径均以仓库根目录为基准。

## 三、Python 环境安装（最小路径）

### 1. 安装 Miniconda

访问 https://www.anaconda.com/docs/getting-started/miniconda/main 下载对应系统的安装包，按向导安装即可。若链接失效可直接搜索 "Miniconda download"。

### 2. 创建隔离环境

```bash
conda create -n start-mllm python=3.10 -y
conda activate start-mllm
```

### 3. 验证 Python

```bash
python --version   # 应显示 3.10.x
```

## 四、GPU 环境自检（本地推理需要）

如果你有一块 NVIDIA 显卡，且希望本地跑模型，按下面检查：

> 如果当前环境里还没有安装 `torch`，先完成下方“PyTorch GPU 版安装（如需）”，再回到这里做自检；否则会先报 `ModuleNotFoundError: No module named 'torch'`。

```bash
# 检查 CUDA 是否可用
python -c "import torch; print(torch.cuda.is_available())"
```

输出 `True` 表示当前 PyTorch 能访问 CUDA 设备。第七章本地示例的资源需求还受权重精度、图像尺寸、输入与输出长度、batch 和注意力实现影响；8GB 是否足够需在固定配置下实测。

只有当上一条命令输出 `True` 时，再继续执行下面这条，否则会直接报错：

```bash
# 仅在 CUDA 可用时执行：检查 GPU 型号与显存
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_properties(0).total_memory / 1024**3, 'GiB')"
```

如果输出 `False`，有三种可能：
1. 没有 NVIDIA 显卡 → 用第七章的 **API 方案**
2. 有显卡但没装驱动 → 先装驱动
3. 驱动装了但 PyTorch 是 CPU 版 → 重装 PyTorch GPU 版

### PyTorch GPU 版安装（如需）

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

cu121 表示 CUDA 12.1，仅为示例。实际安装时请到 https://pytorch.org/get-started/locally/ 根据你的 CUDA 版本选择对应的命令。

<a id="hardware-memory-budget"></a>

## 五、多模态硬件排雷与显存估算

很多同学在本地跑纯文本 LLM 时一切顺利，一换成多模态大模型就频繁 OOM（Out of Memory）。这一节帮你提前排雷，搞清楚显存到底被谁吃掉了。

### 为什么多模态模型特别容易 OOM

纯文本 LLM 的显存占用主要取决于模型参数量和序列长度。但多模态模型在此基础上额外增加了两个"大户"：

1. **视觉编码器参数**：ViT 本身也是几亿参数的模型，推理时需要加载到显存。
2. **视觉输入增加计算与缓存开销**：先区分图像 patch 数、视觉编码器的输出位置数，以及实际进入语言序列的视觉位置数。以固定 `14×14` patch、不补边、不重叠的切分为例，`448×448` 图片有 `(448/14)²=1024` 个 patch；这还不是语言模型的输入位置数。视觉编码器可能添加特殊 token，连接模块也可能合并或压缩特征。第三章的 [Q-Former 示例](./chapter3/第三章%20多模态生成架构.md)将 257 个视觉位置压缩成 32 个，再映射到语言模型输入。

若采用视觉与文本向量拼接的架构，应把**连接模块输出后实际拼入的位置数**计入语言序列。若通过语言模型内的交叉注意力读取视觉特征，则应分别计算语言自注意力缓存和视觉侧的计算、缓存开销。不能把 patch 数直接代入所有模型的 LLM KV Cache 公式。

### 显存估算公式（粗略）

```text
总显存 ≈ 全模型权重显存 + KV Cache 显存 + 运行时开销

其中：
  全模型权重显存 ≈ 总参数量 × 每权重元素字节数（FP16/BF16=2 字节）
  KV Cache 显存 ≈ 2 × 层数 × KV头数 × 每头维度 × 总序列长度 × batch_size × 每缓存元素字节数
  # 2 = Key + Value；总序列长度指语言自注意力实际缓存的位置数
  # 拼接式架构包括实际拼入的视觉位置、文本/特殊标记和已生成 token
  # GQA（分组查询注意力）中 KV头数可少于 Query头数，不能直接用 hidden_size 替代 KV头数 × 每头维度
  # 全模型权重已包含视觉编码器；运行时开销另含视觉/语言激活、临时工作区、框架开销与显存碎片
```

以上 KV 公式估算语言自注意力的标准稠密缓存张量；交叉注意力的视觉缓存需另计。量化权重/缓存还可能有额外元数据，静态预分配或滑动窗口缓存需按实际实现估算。

**举个例子**：Qwen2.5-VL-3B-Instruct
- [官方配置](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/37eefda6797047d24d5d08184aea3e097f3f8a50/config.json)为 36 层、`hidden_size=2048`、16 个 Query 头、2 个 KV 头，每头维度为 2048 ÷ 16 = 128。
- `batch_size=1`、总序列长度 1000、FP16 缓存时：2 × 36 × 2 × 128 × 1000 × 2 = **36,864,000 bytes ≈ 35.2 MiB**。这仅是 KV 张量，**不是总显存**。
- [官方权重索引](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/model.safetensors.index.json)的 `total_size=7,509,245,952` bytes，即 BF16 全模型权重约 **7.51 GB（6.99 GiB）**，已含视觉部分；不能把名称中的“3B”直接当作全模型精确参数量。

本节为**参考估算，尚待固定环境实测**。验证时请记录依赖版本、模型版本、图像尺寸、`max_pixels`、`max_new_tokens`、batch、dtype、注意力实现，以及峰值 `allocated` / `reserved` 显存。更大模型、更长序列和更高图像分辨率通常会增加资源需求。

### 国产与 Mac 硬件速览

| 硬件 | 多模态现状 | 建议 |
|---|---|---|
| **NVIDIA GPU** | 生态最完善，Transformers / vLLM 原生支持 | 首选方案 |
| **Mac M 系列（MPS）** | 可跑小模型（3B），速度较慢，部分算子兼容性不完美 | 学习体验可以，高频推理不推荐 |
| **昇腾（Ascend）** | 需要适配 MindSpore 或 CANN，部分开源模型已有支持 | 企业级部署可关注，个人学习门槛较高 |
| **纯 CPU** | 极慢，仅适合极小模型体验 | 不推荐，优先走 API 方案 |

显存不足时，可以先用下一节的 API 方案完成调用练习，再根据本地运行需求选择硬件。

## 六、没有 GPU 怎么办（API 方案）

通过 API 可以完成图文问答、评测、Demo 和接口驱动的工作流实践，无需本地 GPU。开始前需准备支持图像输入的可用接口、密钥与调用额度，并确认所选模型支持示例的请求格式。本地推理与微调示例有独立的资源要求，可以先跳过。

### 免费或低成本 API 选项（截至本文更新时）

> 最后核验日期：2026-05-20。额度、价格和模型可用性变化较快，以下内容仅作入门参考；请在实际使用前以各平台官方页面与最新文档为准。

| 平台 | 额度 | 特点 | 适用章节 |
|---|---|---|---|
| **DashScope（阿里云）** | 新用户百万 token 免费 | Qwen 系列官方接口，多模态支持好 | 第七、八、九章 |
| **SiliconFlow** | 注册送额度，价格较低 | 聚合多种开源 VLM，OpenAI 兼容 | 第七、八、九章 |
| **ModelScope** | 部分模型免费 | 国产模型生态，社区活跃 | 第七章 |
| **智谱 GLM-4V API** | 新用户免费额度 | 中文场景强，文档清晰 | 第七、八章 |

### API 方案的最小配置

以 DashScope 为例：

```bash
pip install -r docs/chapter7/code/requirements-api.txt   # 不含 torch，仅 50MB
```

.env 文件（或环境变量）：

```bash
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
MODEL_ID=qwen2.5-vl-3b-instruct  # 请以 DashScope 平台最新文档为准
```

然后直接运行第七章脚本即可，不需要任何 GPU。

## 七、本书代码依赖总览

| 章节 | 核心依赖 | 大小 | 说明 |
|---|---|---|---|
| 第四章数据校验 | 仅标准库 | 0 | 不依赖 PyTorch |
| 第五章评测 | openai, pillow | ~50MB | API 方式 |
| 第七章本地推理 | torch, transformers, pillow | ~3GB | 需 GPU |
| 第七章 API 推理 | openai, pillow | ~50MB | 无需 GPU |
| 第八章 Demo | gradio, openai | ~200MB | 无需 GPU |
| 第九章 Agent | openai, python-dotenv | ~50MB | 无需 GPU |

**建议**：先走 API 路线跑通第七~九章，建立手感后再决定是否本地部署。

## 八、常见问题

### Q：Mac（M 系列芯片）能跑吗？
A：能。MPS 后端支持大部分操作，但大模型推理速度通常不如 NVIDIA GPU。建议用 API 方案，或跑较小模型（如 Qwen2.5-VL-3B）。

### Q：Windows 和 Linux 命令有区别吗？
A：主要有两处：
- 设置环境变量：`$env:VAR="value"`（Windows PowerShell） vs `export VAR=value`（Linux/macOS）；若使用 Windows cmd 则写 `set VAR=value`
- 路径分隔符：`\`（Windows） vs `/`（Linux/macOS），但 Python 的 `pathlib` 已自动处理

### Q：需要多少钱？
A：API 调用是否收费取决于平台、模型和可用额度，首次调用前就应核对价格与预算。已有合适硬件时可以尝试本地小模型；微调和高频调用通常需要额外计算资源或费用。

### Q：我连命令行都不太熟，能跟上吗？
A：能。本教程主要会用到 `git clone`、`cd`、`conda create`、`pip install`、`python xxx.py`、以及设置环境变量这几类命令，基本都可以复制粘贴。如果你想花 10 分钟系统了解一下，推荐看 [菜鸟教程 - 命令行入门](https://www.runoob.com/w3cnote/linux-common-command-2.html)（Linux/macOS）或 [Microsoft 官方 PowerShell 入门](https://learn.microsoft.com/zh-cn/powershell/scripting/learn/ps101/01-getting-started)（Windows）。

## 九、本章小结

- **阅读与 API 路线**：Python 基础 + 能安装依赖、运行脚本；API 另需图文接口、密钥与额度
- **本地推理路线**：核对模型兼容性和硬件资源，按需补充 PyTorch 张量知识
- **训练与微调**：选学，另需训练基础、数据与计算资源
- **推荐顺序**：先 API 跑通 → 建立手感 → 再决定是否本地部署

首次学习从[基础篇](./foundation/README.md)开始，再进入[第一章](./chapter1/第一章%20多模态大模型概览.md)。已有基础时，可用 [LLM 基础速通](./llm-基础速通.md)复习。
