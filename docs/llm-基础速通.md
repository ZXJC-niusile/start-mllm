# LLM 基础速通

> **前置章节**：[Chapter 0 环境准备](./chapter0-环境准备.md)　|　**下一章**：[第一章 多模态大模型概览](./chapter1/第一章%20多模态大模型概览.md)

如果你只有 Python 基础，对 Transformer、Embedding、自回归生成这些概念还不熟悉，花 30 分钟读完本章就够了。我们只讲**后续章节会直接用到**的概念，不展开数学推导。

> **想更系统地补课？** 本教程属于 [Datawhale](https://github.com/datawhalechina) 开源社区，社区内有更完整的前置课程：[thorough-pytorch](https://github.com/datawhalechina/thorough-pytorch)、[base-llm：从 NLP 到 LLM](https://github.com/datawhalechina/base-llm)、[self-llm：开源大模型食用指南](https://github.com/datawhalechina/self-llm)。本章已覆盖后续所需的最小知识量，上述课程供按需深入。

## 一、Transformer：LLM 的底层架构

当前几乎所有大语言模型都基于 **Transformer** 架构。你不需要记住它的全部细节，但需要理解它的两个核心思想。

### 1.1 Self-Attention：让每个词关注其他词

传统模型处理文本是从左到右逐个读，像流水线一样。Transformer 的不同之处在于引入了 **Self-Attention（自注意力机制）**。

在原始 Transformer 的**编码器自注意力**中，每个 token 通常可以关注完整输入（忽略 padding 等被屏蔽的位置）；标准 ViT 的全局自注意力也类似。原始 Transformer 的**解码器自注意力则已经使用因果掩码**，不能看未来位置。

用一句话概括这个基础机制：

> 每个 token 都在问：“序列里哪些 token 和我最相关？”然后把相关 token 的信息加权混合到自己身上。

举个例子，处理“大语言模型很强大”时：

- 当处理“模型”这个 token 时，Self-Attention 会计算它和“大”、“语言”、“很”、“强大”各自的关联度
- “模型”和“语言”关联度高 → 后者的信息权重更大
- 最终“模型”的表示就融合了全局的上下文信息

**大语言模型的特殊之处：因果自注意力（Causal Self-Attention）**

现代生成式大语言模型通常采用 Decoder-Only 架构，延续了这种**因果自注意力（Causal Self-Attention）**约束。训练时即使整段文本一起输入，也不能让当前位置偷看后面的答案。

在训练和生成时：
- 每个 token **只能“看到”它前面的 token 以及它自己**，后面的 token 会被掩码（Mask）遮挡。
- 例如生成“大语言模型很强大”时，处理到“模型”这个 token，它只能和“大”、“语言”、“模型”计算关联度，而**看不到**未来的“很”、“强大”。
- 它把前置的相关信息加权混合，从而帮助预测下一个 token。

可以用三个位置检查这个规则（1 = 允许关注，0 = 屏蔽；这是可见性示意，不是某个库的 mask 参数格式）：

```text
            被关注的位置
            1  2  3
当前位置 1  1  0  0
当前位置 2  1  1  0
当前位置 3  1  1  1
```

小检查：位置 2 能看位置 3 吗？不能。位置 2 的表示用于预测位置 3 的 token，不能提前读取答案。这个约束在[原始 Transformer 论文 §3.1、§3.2.3](https://arxiv.org/html/1706.03762v7)中就已存在。

### 1.2 Decoder-Only 架构

现代 LLM（GPT、Qwen、Llama 等）采用 **Decoder-Only** 架构：只有解码器，没有编码器。你可以把它理解为：

```
输入文本 → Transformer 层 × N → 预测下一个 token
```

- 输入和输出都是 token 序列
- 模型逐层提取更高层的语义特征
- 最终在词表上输出下一个 token 的概率分布，再由解码策略选择 token

“Decoder-Only”这个词你会在论文和模型卡片里反复见到，记住它表示“只有解码器、靠预测下一个 token 来工作”就够了。

### 1.3 为什么叫“大”模型

LLM 的“大”指的是参数量。一个 7B（70 亿参数）的模型，就是有 70 亿个可学习的权重值。这些参数大部分分布在 Transformer 的注意力层和前馈网络里。参数越多，模型的表达能力越强，但推理时需要的算力和显存也越大。

> **和 Python 类比**：你可以把模型想象成一个超大的字典（dict），里面有几十亿个键值对。输入一个 token 序列，模型通过这些参数计算出下一个 token 的概率分布。

## 二、Token 与 Embedding

### 2.1 Token：模型处理文本的基本单位

模型不能直接处理字符串。文本在进入模型之前，需要先被切成一个个 **Token**。

```text
输入文本：  "今天天气真好"
分词结果：  ["今天", "天气", "真", "好"]
Token ID：  [2345,   1876,   932, 428]
```

- 英文通常按子词切分：`"unhappiness"` → `["un", "happiness"]`
- 中文通常按字或词切分：`"大语言模型"` → `["大", "语言", "模型"]`
- 每个 token 对应一个整数 ID，在模型的词表（Vocabulary）中唯一

词表大小通常是几万到十几万（如 Qwen2.5 的词表约 15 万 token）。分词器（Tokenizer）负责文本和 token ID 之间的转换，不同模型有不同的分词器。

### 2.2 Embedding：把 ID 变成向量

Token ID 是整数，不能直接输入神经网络。**Embedding（嵌入）** 就是把每个 token ID 映射成一个高维向量的过程。

```text
Token ID：     2345
               ↓ 查表
Embedding：    [0.12, -0.87, 0.33, ..., 0.56]   （2048 维向量）
```

你可以把 Embedding 层理解为一张查找表（Lookup Table）：词表里有 15 万个 token，每个 token 对应一个 2048 维的向量。输入一个 token ID，查表得到对应的向量。

为什么需要 Embedding？因为向量可以表示语义。在训练好的模型里：

- “猫”和“狗”的向量距离很近（都是动物）
- “猫”和“汽车”的向量距离很远（语义无关）
- “国王” - “男人” + “女人” ≈ “女王”（这是经典的 Word2Vec 向量运算类比，用于帮助你直觉理解 Embedding 的语义性质。现代 LLM 的 Embedding 机制有所不同，但核心思想——相近语义在向量空间中靠近——是一致的）

> **和 Python 类比**：Embedding 层就像一个 `dict`，key 是 token ID（整数），value 是一个固定长度的 list（向量）。查表操作就是 `embedding[token_id]`。

### 2.3 完整流程

```text
文本 → 分词器 → Token ID 序列 → Embedding 查表 → 向量序列 → 送入 Transformer
```

经过 Transformer 逐层处理后，最后一层的输出会再映射回词表大小的向量，得到每个 token 作为“下一个词”的概率。

## 三、自回归生成：LLM 怎么“说话”

LLM 生成文本的方式叫 **自回归（Autoregressive）**，核心就一句话：

> 每一步只预测一个 token，然后把它拼到输入末尾，继续预测下一个。

```text
输入：    "今天天气"
Step 1：  预测 → "真"     输入变为："今天天气真"
Step 2：  预测 → "好"     输入变为："今天天气真好"
Step 3：  预测 → "，"     输入变为："今天天气真好，"
Step 4：  预测 → [EOS]    结束
```

每一步，模型根据已有 token 输出下一个 token 的概率分布。**贪心解码（greedy）**取最高概率项；**采样（sampling）**按处理后的分布抽取，两者不应混为一谈。

### 3.1 KV Cache：为什么生成速度和显存是一对矛盾

你可能会想：每生成一个新 token，模型都要把前面所有 token 重新算一遍，岂不是很慢？确实如此——如果没有优化的话。

实际工程中，模型会把每一步计算过的 **Key** 和 **Value**（Self-Attention 里的两个中间量）缓存起来，下一步直接复用，避免重复计算。这个缓存就叫 **KV Cache**。

```text
生成第 1 个 token：计算全部输入的 Key/Value → 存入缓存
生成第 2 个 token：复用缓存 + 只算新 token 的 Key/Value → 追加到缓存
生成第 3 个 token：复用缓存 + 只算新 token 的 Key/Value → 追加到缓存
...
```

好处是生成速度大幅提升；代价是 **KV Cache 会随上下文长度线性膨胀**，吃掉大量显存。这就是为什么：

- 长上下文（如 128K token）推理时显存占用远超模型本身参数
- 多模态模型里一张高分辨率图片可能产生几千个视觉 token，进一步加剧显存压力
- 第六章讲 Serving 时，KV Cache 管理是决定并发量的核心问题

> **一句话记住**：KV Cache = 用空间换时间。它让自回归生成不必从头重算，但代价是显存随上下文长度线性增长。

### 3.2 生成参数

模型先输出词表上的分数（logits），经 softmax 得到概率；解码策略决定选哪个 token。下面讨论常见的采样参数。

**Temperature（温度）**：softmax 前用 logits 除以正数温度。小于 1 通常使分布更尖，大于 1 通常使分布更平；低温度仍是采样，不保证答案正确或每次完全一致。

**Top-p（核采样）**：按概率从高到低排列，保留累计概率**至少达到 p** 的最小候选集合，再在其中重新归一化并采样。它不保证恰好去掉 1−p 的概率质量，更不保证排除错误内容。

```text
示意分布：A = 0.60，B = 0.25，C = 0.15
贪心解码：选 A
top-p = 0.80：保留 A、B（累计 0.85），采样仍可能选 B
```

小检查：把 top-p 改为 0.90，会保留谁？答案是 A、B、C，因为前两项累计还不到 0.90。

做 OCR 或表格提取时，没有通用的温度/top-p 配方。先固定模型、提示词和一组小评测样本，再一次只改一个参数，比较实际错误：

- 本地 Transformers 可用 `do_sample=False, num_beams=1` 作为贪心解码起点；此时无需调采样温度或 top-p。
- 需要采样时再启用 `do_sample=True` 并设置正数温度。不要把某些 API 的 `temperature=0` 约定直接复制到本地采样代码。
- API 支持哪些参数、零温度代表什么，以服务商说明为准；跨硬件或服务版本也不应承诺完全一致的输出。

参数定义见 [Transformers 生成文档](https://huggingface.co/docs/transformers/v4.51.1/en/main_classes/text_generation)。

### 3.3 上下文长度

模型能处理的最大 token 数量叫 **上下文长度（Context Length）**，它限制了输入 + 输出的总 token 数。

```text
上下文长度 = 128K token
  ├── 输入：图像 token（约 1000~4000） + 文本 token（用户问题）
  └── 输出：模型回答的 token
  总和不能超过 128K
```

- 上下文越长，能处理的历史对话和参考资料越多
- 但上下文越长，推理速度越慢、显存占用越大
- 多图输入时，每张图可能占用 1000~4000 个 token，上下文消耗很快

> **和 Python 类比**：上下文长度就像一个固定大小的 list，你往里面放 token，放满了就不能再加。实际使用中需要控制输入长度，为输出留够空间。

### 3.4 Chat Template：模型怎么知道”谁在说话”

预训练阶段，模型只做续写——给它一段文字，它接着往后写。但对话场景里，模型需要区分“系统指令”“用户消息”“助手回复”三种角色。

**Chat Template（对话模板）** 就是一套固定格式，告诉模型这些角色边界在哪里。

#### 为什么它很重要

不同模型有不同的模板。以 Qwen 为例：

```text
<|im_start|>system
你是一个有用的助手。<|im_end|>
<|im_start|>user
这张图里有什么？<|im_end|>
<|im_start|>assistant
图中是一只橘猫...<|im_end|>
```

而 Llama 的模板长这样：

```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
你是一个有用的助手。<|eot_id|>
<|start_header_id|>user<|end_header_id|>
这张图里有什么？<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```

模板中那些 `<|im_start|>`、`<|eot_id|>` 等特殊标记，是模型在训练时学到的角色分隔符。如果推理时用了错误的模板，模型就不知道谁在说话，输出会乱掉。

#### 关键规则只有一条

> **训练和推理必须用同一套 Chat Template。**

这条规则简单但极其重要。第七章里你会看到，很多“模型回答空白或很奇怪”的问题，根源都是模板不对。

#### 实际使用中你不需要手动拼模板

Huggingface 的 `transformers` 库提供了 `apply_chat_template` 方法，会自动按模型的模板格式拼接消息：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

messages = [
    {"role": "system", "content": "你是一个有用的助手。"},
    {"role": "user", "content": "什么是多模态大模型？"},
]

# 自动拼接成模型需要的格式
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(text)
```

你只需要按 `{"role": ..., "content": ...}` 的标准格式构造消息列表，模板转换交给库处理。

> 如果你用 OpenAI 兼容接口（第七、八章的 API 方案），模板转换在服务端自动完成，你同样只需要关心消息列表的结构。

## 四、训练的核心概念

你不需要自己训练模型，但后续章节（第三、四章）会频繁提到这些概念，先建立基本认知会省很多力气。

### 4.1 Loss（损失）

**损失（Loss）** 是一个数字，衡量模型的预测和正确答案之间的差距。

```text
Loss 低 → 模型预测接近正确答案 → 学得好
Loss 高 → 模型预测偏离正确答案 → 学得差
```

在训练日志里你会看到：

```text
epoch 1/3  step 1200  loss 2.31
epoch 1/3  step 2400  loss 1.85
epoch 2/3  step 4800  loss 1.42
```

训练 Loss 逐步下降说明模型在优化当前训练目标，不等于实际任务效果一定提高。要注意：

- Loss 的绝对值因任务而异，重要的是**趋势**（是否在下降）
- 训练 Loss 下降而独立验证集的 Loss **持续上升**，可能提示过拟合。不要凭单个点下结论：先排查随机波动、数据划分、预处理和评估设置是否一致，并检查真实任务表现

### 4.2 训练过程的基本单位

```text
单卡、无梯度累积的例子：总训练数据 10000 条
  └── 每批 32 条，每批做前向 + 反向传播
        └── optimizer.step() 更新一次参数
              └── 完整遍历训练数据一遍 = 1 Epoch
```

| 概念 | 含义 | 例子 |
|---|---|---|
| **Epoch** | 模型完整看一遍所有训练数据 | 训练 3 个 Epoch = 所有数据看 3 遍 |
| **Batch Size** | 每次送入模型的样本数 | Batch Size = 32，每次用 32 条样本计算 |
| **Optimizer Step** | 优化器更新一次参数 | 上例保留末尾不足 32 条的一批时为 313 次；丢弃末批时为 312 次 |
| **Learning Rate（学习率）** | 参数更新的步幅 | 大 → 学得快但可能不稳定；小 → 学得稳但慢 |

**梯度累积**会改变这个对应关系：一个 micro-batch 是一次前向/反向处理的小批次；累积多个 micro-batch 的梯度后，才做一次 optimizer step。例如单卡每批 8 条、累积 4 批，完整一组更新的有效 batch size 为 32。末尾不足一组如何处理、日志中的 `step` 指哪一种，要看训练脚本；参见[梯度累积说明](https://huggingface.co/docs/transformers/grad_accumulation)。

训练日志里的 `lr 1e-4` 是当前学习率。`1e-4`、`2e-5` 等只是可能的配置值，不是预训练或微调的通用推荐；要结合模型、全量微调还是 LoRA、有效 batch size 和验证结果选择。

### 4.3 预训练与微调

大模型的训练通常分两大阶段：

**预训练（Pre-training）**：从零开始，在海量文本上学习语言能力

```text
训练数据：  互联网文本、书籍、代码...（TB 级）
训练目标：  预测下一个 token
训练成本：  需要大量 GPU 和时间（通常数周到数月）
产出：      基座模型（Base Model，即只经预训练、尚未指令微调的模型，详见[第四节“基座模型”](#44-基座模型)）
```

**指令微调（SFT, Supervised Fine-Tuning）**：在基座模型上，用任务数据进一步训练

```text
训练数据：  "指令-回答"格式的标注数据（规模随任务而变）
训练目标：  让模型学会执行具体任务
训练成本：  通常低于从零预训练，但取决于模型、数据、方法和硬件
产出：      指令模型（Chat Model / Instruct Model）
```

这个范式叫 **“预训练 → 微调”**，是当前 LLM 和 MLLM 的标准流程：

```text
预训练 → 基座模型（主要学习续写，未专门优化助手式指令遵循）
  ↓ 指令微调
指令模型（会听指令、会对话、会完成任务）
```

### 4.4 基座模型

**基座模型（Base Model）** 就是只经过预训练、还没经过指令微调的模型。

- 它主要学习续写，也可能通过合适的提示完成问答；例如 [GPT-3 论文](https://arxiv.org/abs/2005.14165)就在没有任务微调时评估了问答能力
- 给它“法国的首都是”，它可能续写“巴黎”，也可能继续展开；指令微调通常使回答格式和任务遵循更稳定
- 常见基座模型：Qwen2.5-7B、Llama3-8B、DeepSeek-V2-Lite

后续章节里“冻结基座模型”指保持所加载底座的参数不变，只训练指定的新模块（如 Connector、LoRA 适配器）。这里的“底座”也可能是已做过指令微调的 Instruct checkpoint，并非只能选择未指令微调的 Base 版本。

## 五、PyTorch 五分钟速查

如果你走 **API 路线**（第七章 OpenAI 兼容接口、第八章 Gradio Demo、第九章 Agent），完全不需要 PyTorch，可以跳过本节。

如果你想走 **本地推理路线**（第七章 Transformers 方式）或动手做 **LoRA 微调**（第四章），下面 5 个模式够你用了：

### 5.1 检查 GPU 是否可用

本节是 **PyTorch / Transformers 读码卡**，不另建一条下载 7B 模型的运行路线。5.1 和 5.5 只需已安装 PyTorch，不下载模型；完整本地推理请按[第七章](./chapter7/第七章%20动手跑通你的第一个%20VLM.md)的环境说明运行其中的 [transformers_chat.py](./chapter7/code/transformers_chat.py)。

```python
import torch
print(torch.cuda.is_available())        # True = 当前 PyTorch 可访问 CUDA 设备
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

这个检查不保证模型能装进显存，也不检测所有类型的加速设备。

### 5.2 加载预训练模型

看到 `from_pretrained(...)` 时，先看它属于哪个类：Transformers 的模型类加载配置和权重，`AutoTokenizer` 加载分词器，`AutoProcessor` 则可处理图文等输入。它们不是 PyTorch 自带的接口；模型类、版本和输入格式应以对应模型说明及第七章脚本为准。

### 5.3 把模型放到 GPU 上

读到 `model.to("cuda")`，表示尝试把整个模型移到 CUDA 设备，需要足够显存。不要把它当作任何机器都可复制的操作。

加载时的 `device_map="auto"` 借助 Accelerate 自动分配权重，通常先用 GPU，再用 CPU，必要时使用磁盘卸载；具体还受内存限额和卸载配置影响。能加载不等于速度快，也不保证后续推理不超内存。细节见 [Accelerate 大模型推理说明](https://huggingface.co/docs/accelerate/en/usage_guides/big_modeling)。

### 5.4 推理模式

读码时区分两件事：

- `model.eval()` 切换 Dropout、BatchNorm 等模块的评估行为，**不关闭梯度记录**。
- `torch.no_grad()` 关闭块内通常的梯度记录，减少不需要的计算图开销，**不替代 eval()**。

普通推理通常同时需要评估行为与关闭梯度记录；但不要只凭调用处没写 `no_grad()` 就断言有问题，封装方法也可能已处理。Transformers 的 `from_pretrained()` 默认把模型设为 eval；若此前调用了 `train()`，评估前应切回来。参见 [PyTorch 自动求导说明](https://docs.pytorch.org/docs/main/notes/autograd.html#evaluation-mode-nn-module-eval)和 [Transformers 加载说明](https://huggingface.co/docs/transformers/v4.51.1/en/main_classes/model)。

### 5.5 Tensor 基础

Tensor 是 PyTorch 的多维数组。先在 CPU 上观察形状、精度与设备，不必加载模型：

```python
import torch

x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
w = torch.ones(3, 4)
y = x @ w                          # [2, 3] @ [3, 4] → [2, 4]
print(x.shape, x.dtype, x.device)    # torch.Size([2, 3]) torch.float32 cpu
print(y.shape)                      # torch.Size([2, 4])
print(y)                            # 两行分别为 [6., 6., 6., 6.] 和 [15., 15., 15., 15.]
assert y.shape == (2, 4)
```

小练习：把 `w` 改成 `torch.ones(4, 3)`，运行前先判断能否相乘。答案：不能，矩阵乘法的内侧维度 3 和 4 不匹配。后续读 `input_ids`、`pixel_values`、`attention_mask` 时，也先检查 shape、dtype 和 device。

> **一句话总结**：PyTorch 提供张量、算子、神经网络模块和自动求导；Transformers 在其上提供模型加载、分词/预处理和 `generate()` 等接口。两层职责不同。

## 六、章节跳转

- 上一篇：[Chapter 0 环境准备](./chapter0-环境准备.md)
- 下一篇：[第一章 多模态大模型概览](./chapter1/第一章%20多模态大模型概览.md)
