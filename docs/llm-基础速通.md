# LLM 基础速通：从 token 到一次预测

> **前置章节**：[Chapter 0 环境准备](./chapter0-环境准备.md)　|　**下一章**：[第一章 多模态大模型概览](./chapter1/第一章%20多模态大模型概览.md)

本章沿着一条数据流学习：**文本 → token ID → 向量 → 因果注意力 → 词表分数 → 概率 → 选择或计算损失**。目标不是记住术语，而是能用一组小数字解释每一步输入什么、输出什么，以及错误会出现在哪里。

## 学习前检查与完成标准

需要会读 Python 列表、下标、循环和函数。还需要以下最小数学知识：

- `[3, 2]` 表示 3 行、每行 2 个数；`[3, 2] @ [2, 4] → [3, 4]`，相乘时内侧维度必须相等。
- 点积是对应元素相乘后求和，例如 `[1, 2] · [3, 4] = 11`；转置把行列交换。
- `exp(x)` 与自然对数 `ln(x)` 互为逆运算，如 `exp(ln(3)) = 3`。本章用 Python 的 `math.exp`、`math.log` 复核，不要求手算对数。

若这些还不熟，先到[基础篇总览](./foundation/README.md)补齐张量形状与矩阵乘法，优先读第 01 单元。希望系统理解框架与模型时，继续读 02～06 的数组、自动求导、神经网络、Tokenizer 和 Transformer 知识讲解；也可结合文末的 CPU Tensor 读码卡阅读。本章的必做脚本只依赖 Python 标准库，不需要 PyTorch、GPU、模型权重或 API 密钥。

完成后，你应能：

1. 解释为什么 token ID 的数值大小不表示语义距离。
2. 算出一个位置的注意力权重，解释本例为何在 softmax 前屏蔽未来位置。
3. 从 logits 算出概率，区分贪心、温度采样和 top-p。
4. 把输入与下一 token 标签对齐，算出一个位置的负对数损失。
5. 运行并解释“改变未来位置，过去的输出不变”的对照实验。

> 进一步补课可参考 [Datawhale](https://github.com/datawhalechina) 的 [thorough-pytorch](https://github.com/datawhalechina/thorough-pytorch)、[base-llm](https://github.com/datawhalechina/base-llm)、[self-llm](https://github.com/datawhalechina/self-llm)。它们用于扩展知识；下面的练习不以先读完这些课程为前提。

## 一、Token 与 Embedding：先把字符串变成可计算的表示

### 1.1 Token ID 是索引，不是词义的数值

Tokenizer 按固定规则把文本编码成 token ID。真实 tokenizer 可能使用子词、字节及特殊 token；一个汉字未必对应一个 token，一个 token 也未必是完整的词。不能靠肉眼分词来估算准确的上下文长度。

为便于核算，本章**人为定义**一个只有四项的词表，并用空格切分输入；它不是 Qwen 或 Llama 的分词结果：

| token | 我 | 爱 | 猫 | `<eos>` |
|---|---:|---:|---:|---:|
| ID | 0 | 1 | 2 | 3 |

```text
"我 爱 猫" → 按空格切分 → ["我", "爱", "猫"] → [0, 1, 2]
```

这里 `<eos>` 是我们约定的结束标记。ID 2 比 ID 1 大，只说明它在表中排得更后；不说明“猫比爱多一个单位”。换一种编号并同步调整相关参数表，模型的计算含义可以不变。

真实模型必须使用与 checkpoint 配套的 tokenizer。若想观察真实切分，可在后续已配置好的本地推理环境中查看 tokenizer 的 `encode`、`decode` 输出和特殊 token；本章不会用假造的切分冒充实际运行结果。

### 1.2 Embedding 是取行，不是把 ID 乘上一个常数

设词表大小 `Vocab = 4`，嵌入维度 `D = 2`，嵌入参数表 `E` 的形状就是 `[4, 2]`。为了手算，先人工设置：

```text
E = [[1, 0],    # ID 0：我
     [0, 1],    # ID 1：爱
     [1, 1],    # ID 2：猫
     [0, 0]]    # ID 3：<eos>

IDs = [0, 1, 2]        # [T]，T=3 个位置
X = E[IDs]            # 依次取第 0、1、2 行，得到 [T,D]=[3,2]
  = [[1,0], [0,1], [1,1]]
```

真实训练会更新嵌入表里的浮点参数。输入 ID 本身不是可求导的连续语义坐标；训练改变的是被选中行的参数，而不是把 ID 2 改成 2.1。

输入嵌入是 token 的初始表示；后续 Transformer 根据上下文更新它。同一 token 在不同句子中的**上下文表示**可以不同。向量中的某一维通常没有预先指定的“动物”“颜色”含义，也不能保证任意语义近的词在原始嵌入表中都很近。Word2Vec 的词向量类比是历史上有用的观察，不是现代 LLM 的普遍定律。

### 1.3 顺序信息从哪里来

仅查表不会告诉模型“猫爱我”与“我爱猫”中每个 token 的位置。真实 Transformer 还使用位置相关机制，例如原始 Transformer 的位置编码或许多现代模型的旋转位置编码。**因果掩码负责限制可见范围，位置机制负责提供位置信息，二者作用不同。**

下面只剖开一次注意力计算，暂时省略位置机制、多头、残差、归一化和前馈网络。人工参数也没有训练过，因此输出不应被解读为模型已学会中文。

## 二、因果注意力：把“关注”还原为可核算的加权和

### 2.1 Q、K、V 分别做什么

对同一个输入矩阵 `X`，用三个可学习的线性变换得到：

- **Query（Q）**：当前位置用于计算匹配分数的向量。
- **Key（K）**：各位置用于与 query 匹配的向量。
- **Value（V）**：按匹配权重实际汇总的信息。

Q、K 决定权重，权重再作用到 V；它们不是三份文字，也不保证直接对应人能解释的语义。注意下面的 `V` 是 value 矩阵，词表大小另写为 `Vocab`，避免同名混淆。

为了把数字算清楚，取单个头，`d_k = 1`、`d_v = 2`：

```text
Wq = [[0], [ln(3)]]       # [D,d_k]=[2,1]
Wk = [[0], [1]]           # [2,1]
Wv = [[1,0], [0,1]]       # [D,d_v]=[2,2]，单位矩阵

Q = X @ Wq = [[0], [ln(3)], [ln(3)]]   # [3,1]
K = X @ Wk = [[0], [1],     [1]]       # [3,1]
V = X @ Wv = [[1,0], [0,1], [1,1]]    # [3,2]
```

常见实现里每头 Q/K/V 的维度可能相同；本例特意让 key 与 value 维度不同，以简化手算。点积只要求 Q 与 K 的最后一维一致。

### 2.2 第二个位置如何得到权重

以第二个位置“爱”为例，它的 query 是 `[ln(3)]`。先与三个 key 点积，再除以 `sqrt(d_k)`；本例 `sqrt(1)=1`：

```text
未屏蔽的分数 = [ln(3)*0, ln(3)*1, ln(3)*1]
             = [0, ln(3), ln(3)]
```

但第二个位置要预测第三个 token，不能提前读取第三个位置的输入。“可见性表”是：

```text
                被关注的位置
                1  2  3
当前位置 1      1  0  0
当前位置 2      1  1  0
当前位置 3      1  1  1
```

1 表示可见，0 表示屏蔽。**这是示意表，不是所有库的布尔 mask 约定。** 在我们的加性 mask 中，可见位置加 0，未来位置加负无穷：

```text
第二行分数 + mask = [0, ln(3), -∞]
```

现在沿着这一行的**被关注位置轴**做 softmax。对任意一行分数 `s`，`softmax(s)[j] = exp(s[j]) / Σ exp(s[k])`；它把分数变成非负且和为 1 的权重：

```text
exp 分别为 [1, 3, 0]，总和为 4
注意力权重 = [1/4, 3/4, 0]
```

实现时通常先减去本行最大值再求 exp，以免溢出。所有项乘上同一个因子不改变归一化结果，因此 `softmax(s-max(s)) = softmax(s)`。

最后对 value 向量加权求和：

```text
h₂ = (1/4)*[1,0] + (3/4)*[0,1] + 0*[1,1]
   = [1/4, 3/4]
```

这就是一次“关注上下文”的具体含义。高注意力权重表示本次汇总中对应 value 的系数大，不能单凭它证明模型理解了句意或解释了完整决策。

### 2.3 整段一起算，为什么仍然不能偷看

把三行放在一起，形状是：

```text
Q @ Kᵀ           [3,1] @ [1,3] → [3,3]  每对位置的分数
A = softmax(QKᵀ / sqrt(d_k) + mask)      每一行归一化
H = A @ V        [3,3] @ [3,2] → [3,2]  每个位置的新表示

A = [[1,   0,   0],
     [1/4, 3/4, 0],
     [1/7, 3/7, 3/7]]
H = [[1,   0],
     [1/4, 3/4],
     [4/7, 6/7]]
```

训练时可以一次计算所有位置。只要因果约束正确，每个位置仍只能使用自己的前缀；“并行计算”不等于“看见未来”。改变第三个位置的输入，不应改变前两个位置的输出。第五节会实际测试这个性质。

原始 Transformer 的**编码器**自注意力通常可看完整输入（除去 padding 等被屏蔽位置）；它的**解码器**自注意力从一开始就有因果 mask。因果约束不是现代 LLM 生成时才临时添加的。原始论文是编码器—解码器结构，现代常见的 decoder-only LLM 使用因果自注意力层，不应把两种架构等同。参见 [Transformer 原论文 §3.1–3.2](https://arxiv.org/html/1706.03762v7)。

## 三、从 logits 到选择：下一 token 并不是注意力权重

### 3.1 两次 softmax，各自归一化的对象不同

注意力中的 `[3,3]` 权重是在**输入位置之间**分配信息。预测下一个 token 时，还需要把最后的隐藏表示映射成**词表上每个候选 token**的分数，即 logits。logits 可以为负，也不需要加起来等于 1。

真实模型先经过多层注意力、前馈网络等变换，再接词表预测头。本例省略这些层，直接把刚算出的 `H` 接到人工设置的预测头：

```text
W_vocab = [[0, 0,           0,           0],
           [0, 4*ln(2)/3,   4*ln(3)/3,   4*ln(4)/3]]   # [D,Vocab]=[2,4]
logits = H @ W_vocab                                  # [3,4]
```

这里是 **LM 词表预测头**，不是多头注意力内部的输出投影。为便于手算省略了 bias。

对第二个位置 `h₂=[1/4,3/4]`：

```text
logits₂ = [0, ln(2), ln(3), ln(4)]
exp(logits₂) = [1,2,3,4]
p₂ = [1/10,2/10,3/10,4/10] = [0.1,0.2,0.3,0.4]
候选：   我   爱   猫   <eos>
```

`p₂` 表示给定“我 爱”这个前缀时的下一 token 分布。它不是输入三处位置的注意力权重；两者的长度、归一化轴和用途都不同。

### 3.2 贪心、温度和 top-p

- **贪心解码**取最大分数的候选；上例选择 `<eos>`。它只最大化当前一步，不保证整段序列概率最高，更不保证内容正确。
- **采样**按分布抽取；上例即使“猫”不是最大项，也有 0.3 的概率被选中。
- **温度 T** 在 softmax 前用 logits 除以正数 `T`。`T=1` 得原分布；较低温度使分布更尖，但仍是采样。

上例取 `T=0.5`，相当于将 logits 乘 2：

```text
exp(logits/0.5) = [1²,2²,3²,4²] = [1,4,9,16]
新概率 = [1/30,4/30,9/30,16/30]
```

`<eos>` 概率从 0.4 增至约 0.5333，但不意味着它变成正确答案。

**Top-p** 先按概率从大到小排序，保留累计概率至少达到阈值的最小前缀，再在保留项中重新归一化。例如在原分布上设 `top-p=0.65`：

```text
排序：<eos>:0.4 → 猫:0.3 → 爱:0.2 → 我:0.1
0.4 不够；0.4+0.3=0.7 已够 → 保留 <eos>、猫
重新归一化：[0.4/0.7, 0.3/0.7] = [4/7,3/7]
```

保留概率质量为 0.7，而不是刚好 0.65。Top-p 不会辨别事实真假；如果最大概率的内容错了，它仍可能保留并选中。若同时使用温度与 top-p，要依据具体实现核对处理顺序。

本地 Transformers 可用 `do_sample=False, num_beams=1` 作为贪心起点；此时不需调采样温度和 top-p。采样时才设置正温度，不要把某些 API 的 `temperature=0` 约定直接复制到本地采样代码。没有通用 OCR 参数配方，应固定评测样本，一次只改一项。跨硬件或服务版本也不应承诺输出绝对一致。参见 [Transformers 生成参数](https://huggingface.co/docs/transformers/v4.51.1/en/main_classes/text_generation)和[核采样原论文](https://arxiv.org/abs/1904.09751)。

### 3.3 自回归、prefill 与 decode

正常的逐 token 生成把本次选择追加到输入，继续预测，直到结束标记或长度上限。本例若在“我 爱”后采样到“猫”，下一次就用“我 爱 猫”；若选到 `<eos>`，则按本例约定结束。实际模型的结束 token 与停止规则由模型和服务配置决定。

普通缓存推理常分为两阶段：

1. **Prefill**：对整个提示词做一次前向计算，建立各层的 K/V 缓存；最后一个提示位置的 logits 用于选出第一个新 token。
2. **Decode**：把刚选出的 token 输入模型，为它计算各层的新 Q/K/V；新的 query 与历史及当前 K/V 做注意力，追加新 K/V，再预测下一个 token。

因此，第一个生成 token 被选出时，它自己的 K/V 尚未必写入缓存；通常要到下一次前向处理这个 token 时才写入。缓存保存的是中间 K/V，不是答案文本，也不是模型重新训练后的参数。**使用缓存仍须读取历史 K/V 做注意力，并非每一步都是与上下文长度无关的常数工作量。** 参见 [Transformers 缓存计算说明](https://huggingface.co/docs/transformers/v4.51.1/en/cache_explanation)。

标准稠密全上下文 KV 缓存的容量随缓存序列长度线性增长；具体取决于层数、KV 头数（如 GQA）、每头维度、batch 与缓存精度。缓存是否超过权重显存没有统一答案；滑动窗口等缓存策略也有不同边界。公式见 [Chapter 0 显存估算](./chapter0-环境准备.md#五多模态硬件排雷与显存估算)。

## 四、从概率到损失：训练为什么必须对齐下一 token

### 4.1 同一段数据，输入和标签差一位

我们人为给训练序列加结束标记：`[我, 爱, 猫, <eos>]`。采用显式构造输入和目标的写法：

| 位置 | 当前输入 | 能使用的前缀 | 目标标签 |
|---|---|---|---|
| 1 | 我 | 我 | 爱 |
| 2 | 爱 | 我 爱 | 猫 |
| 3 | 猫 | 我 爱 猫 | `<eos>` |

```text
完整序列 = [0,1,2,3]
input_ids = 完整序列[:-1] = [0,1,2]
labels    = 完整序列[1:]  = [1,2,3]
```

第三个 token “猫”在训练输入中出现了，但因果 mask 保证第二个位置预测它时看不到它。若把每个位置的目标设成它自己，训练的就不再是这个下一 token 任务。

**读框架代码时不要重复 shift。** 上表是我们手工实现损失的接口；Transformers 的许多 `AutoModelForCausalLM` 实现允许 `labels=input_ids`，在模型内部再把 logits 与 labels 错开一位。必须查看具体模型的约定，而不是看到相同 labels 就判定错误，或在外面再错开一次。参考 [GPT-2 的 labels 参数说明](https://huggingface.co/docs/transformers/v4.51.1/en/model_doc/gpt2#transformers.GPT2LMHeadModel.forward)。

### 4.2 负对数损失如何惩罚错误

一个位置的目标 ID 是 `y`，模型给它的概率是 `p[y]`，则该位置的交叉熵损失是：

```text
loss = -ln(p[y])
```

第二个位置目标是“猫”（ID 2），上面已算出 `p₂[2]=0.3`：

```text
loss₂ = -ln(0.3) ≈ 1.203973
```

如果训练后目标概率升到 0.6，损失约为 0.510826；如果只有 0.01，损失约为 4.605170。**损失看的是目标的概率，不只是最终有没有选中目标。** 两个模型都贪心答错，也可能给目标不同概率，从而有不同损失。

本例三位置损失为 `[1.386294, 1.203973, 0.866154]`；按有效位置平均得到 `1.152140`。训练通常通过反向传播求参数的梯度，再由优化器更新参数，使训练目标下降。第五节只计算前向和损失，不实现反向传播或训练。

真实 batch 中要明确哪些位置计入平均：padding 往往不计；一些 SFT 设置只计算助手回答位置。**损失 mask 与因果 attention mask 不是一回事**：用户提示可以供模型读取，但不一定作为需要计分的输出。PyTorch 的 [CrossEntropyLoss](https://docs.pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html) 接收 logits 并包含相应的归一化/负对数计算，使用该接口时不要先手动 softmax 再当作 logits 传入。

### 4.3 训练日志能说明什么，不能说明什么

训练 loss 下降意味着当前训练目标被优化，不自动等于真实任务效果变好。只有在 tokenizer、数据、计分位置与归一化方式可比时，loss 数值才适合直接比较。训练 loss 下降、独立验证 loss 持续上升可能提示过拟合；先排查随机波动、数据划分、预处理和评估差异，不能凭一个点下结论。

- **Epoch**：遍历训练数据一遍。
- **Micro-batch**：一次前向/反向的小批数据。
- **Optimizer step**：实际更新一次参数。单卡每批 8 条，累积 4 批梯度后更新，完整一组的有效 batch size 为 32；不是每个 micro-batch 都更新。
- **学习率**：影响更新幅度的超参数。`1e-4`、`2e-5` 是可能的配置，不是预训练/LoRA 的通用处方。

例如 10000 条样本、单卡每批 32 条、无梯度累积：保留末尾不足 32 条的 batch 时每 epoch 更新 313 次，丢弃时为 312 次。累积后的末批如何处理以及日志的 `step` 含义，要看脚本。[梯度累积官方说明](https://huggingface.co/docs/transformers/grad_accumulation)可与第四章训练配置对照。

### 4.4 基座模型

预训练通常用大规模数据学习下一 token 预测等目标；指令微调（SFT）进一步用任务或对话示例优化输出行为。数据规模、耗时和成本依模型、训练方法及硬件而变，不能据“微调”二字承诺单卡几小时完成。

**Base model** 通常指尚未经过指令微调的预训练模型。它可能通过续写或提示完成问答，并非“不会回答问题”；[GPT-3 原论文](https://arxiv.org/abs/2005.14165)就评估了无任务微调时的问答。SFT 通常使格式与指令遵循更稳定，仍不能保证每次正确。后训练也可能包括偏好优化等步骤，预训练 → SFT 是入门主线而非所有模型的完整流程。

“冻结基座模型”在适配器训练语境中，是保持所加载底座的参数不变，只训练指定的新模块（例如 Connector、LoRA）。这个底座也可以是已经过指令微调的 Instruct checkpoint，不必是名称带 Base 的版本。

## 五、必做实验：用反例检查因果约束

完整脚本：[llm_causal_walkthrough.py](./foundation/code/llm_causal_walkthrough.py)。它实现本章的取嵌入、Q/K/V 投影、mask、softmax、词表预测头、shifted labels、损失和一次采样。只使用 Python 标准库，不会联网或下载模型。

从仓库根目录运行：

```bash
python docs/foundation/code/llm_causal_walkthrough.py
python docs/foundation/code/llm_causal_walkthrough.py --no-mask
```

若已进入 `docs`，路径相应改为 `foundation/code/llm_causal_walkthrough.py`。先预测结果，再运行，不要只看最后的 PASS。

### 5.1 正常运行应看到什么

关键输出如下（小数保留六位）：

```text
mode: CAUSAL
ids: [0, 1, 2] labels: [1, 2, 3]
shapes: X=(3,2) Q=K=(3,1) V=(3,2) A=(3,3) H=(3,2) logits=(3,4)
position 2 attention: [0.250000, 0.750000, 0.000000]
position 2 hidden: [0.250000, 0.750000]
position 2 probabilities: [0.100000, 0.200000, 0.300000, 0.400000]
per-position loss: [1.386294, 1.203973, 0.866154]
mean loss: 1.152140
future change leaves first 2 positions unchanged: True
```

脚本还打印 top-p 候选及一次固定随机种子的抽样，并检查数值、行归一化、标签对齐、因果不变性和 top-p。固定种子用于复核这个教学程序，不代表真实模型跨硬件结果都一致。

### 5.2 刻意移除 mask，会发生什么

`--no-mask` 后，第二个位置的权重变为 `[1/7,3/7,3/7]`，第三个位置已经参与计算。程序把第三个输入向量从 `[1,1]` 改成 `[9,-2]`，前两个位置的输出也随之改变：

```text
mode: UNMASKED (counterexample)
position 2 attention: [0.142857, 0.428571, 0.428571]
future change leaves first 2 positions unchanged: False
```

最后仍然显示 PASS，表示**程序成功检出了这个预期的错误反例**，不是说未屏蔽版本适合自回归训练。无 mask 会泄漏目标信息，训练时看到的条件与生成时不一致。不能因为某次 loss 更低就认为实现更好；本例未屏蔽的平均 loss 反而为 `1.211681`，泄漏并不必然使每组人工参数的 loss 降低。

### 5.3 三个动手改动

1. 在纸上重新算第二个位置 `T=0.5` 的概率。预测 `<eos>` 概率应约为 0.5333。
2. 把 `[0.60,0.25,0.15]` 的 top-p 阈值从 0.80 改到 0.90。先写候选 ID；应从 `[0,1]` 变成 `[0,1,2]`。
3. 临时把 `labels` 改成 `ids`。运行应在“shifted loss”相关断言处失败。解释失败：第二个位置错误地给“爱”打分，得到 `-ln(0.2)≈1.609438`，而非目标“猫”的 `1.203973`。完成后还原代码。

前两项可以从仓库根运行 `python` 进入交互环境，再粘贴以下完整代码核对（这里从本章脚本导入函数，不需要安装包）：

```python
import math
from docs.foundation.code.llm_causal_walkthrough import softmax, top_p

print(softmax([math.log(n) for n in (1, 2, 3, 4)], 0.5))
# 约 [0.033333, 0.133333, 0.300000, 0.533333]
print(top_p([0.60, 0.25, 0.15], 0.80)[0])  # [0, 1]
print(top_p([0.60, 0.25, 0.15], 0.90)[0])  # [0, 1, 2]
```

提交给自己的一份最小学习记录：两次运行的关键输出、第二个位置的手算过程，以及一句解释“为什么并行训练仍要 mask”。仅粘贴 PASS 不算完成。

## 六、真实模型读码卡：把小实验接回后续章节

### 6.1 上下文预算与多模态输入

对本教程讨论的 decoder-only 生成路径，输入与生成序列共同占用上下文预算，还须考虑服务单独的输入/输出限制。输入不只有用户文字：角色标记、历史对话、工具结果和部分架构中的视觉 token 都可能占位，要为输出留空间。

图像 token 数依模型、分辨率、切块及合并策略变化，不是每图固定 1000–4000。以 [Qwen2-VL 官方说明](https://qwenlm.github.io/blog/qwen2-vl/#developing-with-qwen2-vl)为例，可以通过 `min_pixels` / `max_pixels` 控制图像处理范围。视觉编码器输出通常是连续向量，不是从文本词表查出的整数 ID。

后续第三章会遇到接口：`视觉特征 [B,N,Dv] → Projector → [B,N,Dllm]`。这里 `B` 是 batch，`N` 是视觉 token 数。改变最后一维能让接口接上，但**维度相同不等于语义已经对齐**，仍需要合适的数据和训练。不是所有多模态架构都把图像按同一种方式拼进文本序列。

### 6.2 Chat Template：序列中的角色边界

消息列表 `[{"role":"user","content":"你好"}]` 需要按 checkpoint 配套的模板转换为 token 序列。模板可能插入角色、消息结束和助手回复起始标记；例如某些 Qwen 模板使用 `<|im_start|>` / `<|im_end|>`，Llama 3 系列使用不同的角色分隔标记。不要把一个系列的格式手工套给另一个系列。

读到 `tokenizer.apply_chat_template(...)` 或多模态 `processor.apply_chat_template(...)` 时，核对：

1. 模板是否与模型训练时使用的格式兼容？错误格式可能导致异常输出，不代表每次必然失败。
2. 是否准备开始新的助手回复？`add_generation_prompt=True` 在支持它的模板里追加回复起始标记；训练已有完整回答时通常不这样追加。
3. 是否重复加入了 BOS/EOS 等特殊 token？先 `tokenize=False` 再另行编码时，应按模型与模板约定处理，避免重复。

训练与推理应使用模型预期的同一套格式约定，但不是让二者每次调用参数都相同。OpenAI 兼容聊天接口通常由服务端处理模板；多模态输入仍须满足服务的消息结构与图像支持要求。完整可运行路径统一见[第七章](./chapter7/第七章%20动手跑通你的第一个%20VLM.md)，模板细节见 [Transformers 模板说明](https://huggingface.co/docs/transformers/v4.51.1/en/chat_templating)及[多模态模板说明](https://huggingface.co/docs/transformers/main/en/chat_templating_multimodal)。

### 6.3 PyTorch / Transformers 操作速查

如果你走 **API 路线**（第七章 OpenAI 兼容接口、第八章 Gradio Demo、第九章 Agent），完全不需要 PyTorch，可以跳过本节。

下面 5 个模式用于帮助你辨认 **本地推理代码**（第七章 Transformers 方式）中的常见操作。若要修改模型代码或尝试 **LoRA 微调**（第四章，选学），还需理解张量形状、设备与精度，并进一步学习自动求导和训练流程；这些速查示例不构成完整的训练入门。

#### A. 检查 GPU 是否可用

本节是 **PyTorch / Transformers 读码卡**，不另建一条下载 7B 模型的运行路线。A 和 E 只需已安装 PyTorch，不下载模型；完整本地推理请按[第七章](./chapter7/第七章%20动手跑通你的第一个%20VLM.md)的环境说明运行其中的 [transformers_chat.py](./chapter7/code/transformers_chat.py)。

```python
import torch
print(torch.cuda.is_available())        # True = 当前 PyTorch 可访问 CUDA 设备
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

这个检查不保证模型能装进显存，也不检测所有类型的加速设备。

#### B. 加载预训练模型

看到 `from_pretrained(...)` 时，先看它属于哪个类：Transformers 的模型类加载配置和权重，`AutoTokenizer` 加载分词器，`AutoProcessor` 则可处理图文等输入。它们不是 PyTorch 自带的接口；模型类、版本和输入格式应以对应模型说明及第七章脚本为准。

#### C. 把模型放到 GPU 上

读到 `model.to("cuda")`，表示尝试把整个模型移到 CUDA 设备，需要足够显存。不要把它当作任何机器都可复制的操作。

加载时的 `device_map="auto"` 借助 Accelerate 自动分配权重，通常先用 GPU，再用 CPU，必要时使用磁盘卸载；具体还受内存限额和卸载配置影响。能加载不等于速度快，也不保证后续推理不超内存。细节见 [Accelerate 大模型推理说明](https://huggingface.co/docs/accelerate/en/usage_guides/big_modeling)。

#### D. 推理模式

读码时区分两件事：

- `model.eval()` 切换 Dropout、BatchNorm 等模块的评估行为，**不关闭梯度记录**。
- `torch.no_grad()` 关闭块内通常的梯度记录，减少不需要的计算图开销，**不替代 eval()**。

普通推理通常同时需要评估行为与关闭梯度记录；但不要只凭调用处没写 `no_grad()` 就断言有问题，封装方法也可能已处理。Transformers 的 `from_pretrained()` 默认把模型设为 eval；若此前调用了 `train()`，评估前应切回来。参见 [PyTorch 自动求导说明](https://docs.pytorch.org/docs/main/notes/autograd.html#evaluation-mode-nn-module-eval)和 [Transformers 加载说明](https://huggingface.co/docs/transformers/v4.51.1/en/main_classes/model)。

#### E. Tensor 基础

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

## 七、自测与常见错因

先独立作答，再读答案；算错时回到对应中间量，不要只背结论。

| 问题 | 答案与错因 |
|---|---|
| 1. 把“猫”的 ID 从 2 改成 20，是否更接近其他动物？ | 否。ID 是索引；若词表只有 4 项，20 还会越界。语义不是 ID 数值大小。 |
| 2. 第二个位置为何能读“爱”本身，却不能读“猫”？ | 它以“我 爱”为已知前缀预测“猫”；允许对角线，屏蔽未来。把对角线也屏蔽并不是本章定义的任务。 |
| 3. softmax 后直接把未来权重置 0 是否等价？ | 不等价。未屏蔽第二行是 `[1/7,3/7,3/7]`，置零后前两项和只有 `4/7`；若再正确归一化才恢复 `[1/4,3/4,0]`。本章在 softmax 前屏蔽更直接。 |
| 4. 注意力 `[1/4,3/4,0]` 是否表示下一词“爱”的概率为 3/4？ | 否。它对应三个输入位置；词表概率是另一个长为 4 的向量，需经过词表预测头。 |
| 5. 第二个位置贪心输出错误，loss 是否只能为 1？ | 否。这里选 `<eos>`，目标“猫”的概率为 0.3，loss 为 `-ln(0.3)≈1.204`。损失不是 0/1 对错标记。 |
| 6. top-p=0.90 对 `[0.60,0.25,0.15]` 保留几项？ | 三项；前两项累计 0.85 尚未到阈值，不能把“丢弃最低项”当成固定规则。 |
| 7. 训练标签写成 `input_ids` 一定错吗？ | 不一定。框架可能内部 shift；手写损失则要按自己定义的输入/目标对齐，避免不 shift 或 shift 两次。 |
| 8. 有了 KV cache，历史 token 就不参与后续计算了吗？ | 仍参与。缓存省去历史 K/V 的重复计算，新的 query 仍要与可见 K/V 做注意力。 |
| 9. 训练 loss 下降，是否说明 OCR 更准确？ | 不一定。还要看独立验证与真实 OCR 错误；数据泄漏、计分范围不同也会使 loss 看似更好。 |

如果 2–5 题仍不确定，先重算本章第二个位置；如果 7 题不确定，重新画输入—标签表。全部能解释后，再进入多模态章节，把文本向量与视觉向量的接口对照起来。

## 原始论文与深入学习

- **Transformer**：Vaswani 等，*Attention Is All You Need*（2017，[论文](https://arxiv.org/abs/1706.03762)）。重点对照注意力、位置编码和解码器因果掩码。
- **Word2Vec 类比的来源**：Mikolov 等，*Efficient Estimation of Word Representations in Vector Space*（2013，[论文](https://arxiv.org/abs/1301.3781)）。静态词向量的类比现象不保证现代 LLM 的每个输入嵌入都有相同性质。
- **核采样**：Holtzman 等，*The Curious Case of Neural Text Degeneration*（2019 / ICLR 2020，[论文](https://arxiv.org/abs/1904.09751)）。论文解释 top-p 的动态候选集合，不为某组 OCR 温度参数提供通用正确性保证。
- **官方实现文档**：[Transformers KV Cache](https://huggingface.co/docs/transformers/main/kv_cache) 与 [多模态 Chat Template](https://huggingface.co/docs/transformers/main/en/chat_templating_multimodal)。用它们核对缓存策略和 API 行为；它们是软件文档，不是发明这些方法的论文。

## 八、章节跳转

- 上一篇：[Chapter 0 环境准备](./chapter0-环境准备.md)
- 下一篇：[第一章 多模态大模型概览](./chapter1/第一章%20多模态大模型概览.md)
