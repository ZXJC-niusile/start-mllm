# 第七章 动手跑通你的第一个 VLM


本章演示一次图文问答的完整调用：准备图片和问题、发送请求、读取并检查回答。

已有图文接口时可直接选择 API 路线；准备了兼容设备与环境时可选择本地路线。完成调用后，进入[第五章](../chapter5/第五章%20评测体系与工程选型.md)检查回答质量；需要分析服务成本时再读[第六章](../chapter6/第六章%20推理部署与%20Serving.md)。

本章提供两种方式：

1. 直接用 `Transformers` 调本地模型
2. 用 OpenAI 兼容接口调用多模态服务

## 一、代码位置

本章配套代码位于：

- `docs/chapter7/code/transformers_chat.py`
- `docs/chapter7/code/openai_compatible_client.py`
- `docs/chapter7/code/requirements.txt`

## 二、方式一：用 Transformers 本地推理

### 1. 安装依赖

本地路线需要 `torch` 与 `torchvision` 配套安装；`qwen-vl-utils` 在导入时就会使用 `torchvision`，即使只处理图片也不能省略。本章依赖范围为 `torch>=2.3,<2.7` 与 `torchvision>=0.18,<0.22`；请按 [PyTorch 官方历史版本安装表](https://pytorch.org/get-started/previous-versions/) 选择与系统、驱动适配的同一组 wheel（例如 `torch 2.6.0` 对应 `torchvision 0.21.0`），不要单独升级其中一个包。已有 GPU 环境也应先检查版本范围，避免安装时意外替换现有的 PyTorch。

```bash
python -m pip install -r docs/chapter7/code/requirements.txt
python -m pip check
python -c "import torch, torchvision; from qwen_vl_utils import process_vision_info; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

最后一条检查依赖导入与 CUDA 可见性，不下载模型；显存需求见第零章的估算方法。若导入失败，先解决缺包或 `torch` / `torchvision` 构建不匹配，再加载模型。

`MAX_NEW_TOKENS` 未设置或空白时使用 256；其他值必须为正整数，非法配置会在加载模型前报错。

### 2. 准备模型和图片

你需要准备：

- Qwen2.5-VL 系列模型或对应的本地权重目录（脚本固定使用 `Qwen2_5_VLForConditionalGeneration`，不支持直接替换为 Qwen2-VL、InternVL 或 LLaVA）
- 一张待测试图片（仓库未默认附带示例图；可先在 `docs/chapter5/code` 运行 `python create_placeholder_images.py`，再用路径 `docs/chapter5/code/images/sample_ui.png`——以下命令默认你在**仓库根目录**执行）

脚本默认使用环境变量：

- `MODEL_ID`：Qwen2.5-VL 模型名称或本地路径
- `IMAGE_PATH`：图片路径
- `QUESTION`：用户问题

推荐做法：把 `docs/chapter7/code/.env.example` 复制为同目录 `.env` 后直接改值，脚本会自动读取。若你暂时不想建 `.env`，也可以直接在终端里逐行写：

```powershell
$env:MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct"
$env:IMAGE_PATH="你的图片路径"
$env:QUESTION="你想问模型的问题"
```

### 3. 运行示例

```powershell
$env:MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct"
$env:IMAGE_PATH="docs/chapter5/code/images/sample_ui.png"
$env:QUESTION="请详细描述这张图，并指出其中最重要的视觉元素。"
python docs/chapter7/code/transformers_chat.py
```

Linux / macOS：

```bash
export MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct
export IMAGE_PATH=docs/chapter5/code/images/sample_ui.png
export QUESTION=请详细描述这张图，并指出其中最重要的视觉元素。
python docs/chapter7/code/transformers_chat.py
```

### 4. 脚本流程

本章默认模型的架构来源是 [Qwen2.5-VL 技术报告](https://arxiv.org/abs/2502.13923)；实际调用细节需对照 [官方模型卡](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) 与 [Transformers 多模态模板文档](https://huggingface.co/docs/transformers/main/en/chat_templating_multimodal)。

脚本包含四个步骤：

1. 加载模型和 processor
2. 构造图文消息
3. 让 processor 把消息变成模型输入
4. 调 `generate` 输出答案

建议你别只盯“命令成功”，而是重点看这三个点：

- 图像消息在代码里是如何表示的
- chat template 是怎么把图文消息拼接成输入的
- 生成后怎样把 prompt 部分裁掉，只保留模型新增输出

## 三、方式二：调用 OpenAI 兼容接口

很多时候你的模型并不直接由业务代码加载，而是由某个服务提供 OpenAI 兼容接口。这时候应用层只需要做两件事：

1. 把图片转成 URL 或 base64 data URL
2. 用 `chat.completions` 方式提交图文消息

### 1. 运行方式

若只走 API 路线，可跳过上面的本地模型依赖，单独安装：

```bash
python -m pip install -r docs/chapter7/code/requirements-api.txt
```

`openai_compatible_client.py` 与 `transformers_chat.py` 共用 `docs/learner_paths.py`：**不设置 `IMAGE_PATH` 时**，若已生成第五章占位图，会自动使用 `docs/chapter5/code/images/sample_ui.png`；也可显式指定任意路径（支持在仓库根目录或章节目录下运行）。

推荐先把 `docs/chapter7/code/.env.example` 复制为 `.env`，把接口地址、模型名和图片路径填进去，再直接运行脚本。若你想临时在终端里配置，最容易抄的写法是：

```powershell
$env:OPENAI_BASE_URL="你的真实接口"
$env:OPENAI_API_KEY="你的真实密钥"
$env:MODEL_ID="你的真实模型名"
$env:IMAGE_PATH="docs/chapter5/code/images/sample_ui.png"
$env:QUESTION="这张图传递了什么信息？"
python docs/chapter7/code/openai_compatible_client.py
```

Linux / macOS：

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY
export MODEL_ID=your-vlm
export IMAGE_PATH=docs/chapter5/code/images/sample_ui.png
export QUESTION=这张图传递了什么信息？
python docs/chapter7/code/openai_compatible_client.py
```

### 2. 为什么最好保留这一层

如果你后续要做：

- Web Demo
- API 服务
- Agent 工作流
- 多模型路由

那 OpenAI 兼容接口会比直接在业务层加载模型更灵活。

## 四、实践时最常见的报错

先按现象粗分一层，再进下面的细项（与第二章 Debug 顺序一致：先模型与模板，再协议，再资源）：

| 你看到的 | 优先怀疑 |
| --- | --- |
| 回答空白、极短、或明显复读模板 | `apply_chat_template`、消息结构、`processor` 是否吃到图 |
| HTTP 400/422、或服务端报缺图/格式错 | `image_url` / `data:` URL、字段名、多模态 schema |
| 显存溢出、进程被 kill | 分辨率、`max_new_tokens`、模型体量与量化 |

### 1. 模型能加载，但回答空白或很奇怪

优先检查：

- 是否用了正确的 chat template
- 图像是否真的传给了 processor
- 输入消息里的 `type` 字段是否符合模型要求

### 2. 接口调用成功，但服务端说图像格式不对

优先检查：

- 是不是把本地路径直接当 URL 发了
- data URL 是否包含 MIME 类型前缀
- 服务端是否真的支持多模态消息

### 3. 显存不够

可以从这些方向降低压力：

- 更换小参数模型
- 量化部署
- 降低输入图像分辨率
- 减少 `max_new_tokens`

## 五、基础实践：完成一次有图像证据的问答

选择一张自己能核对内容的真实图片，使用前面任一条路线完成调用。占位图只能检查请求链路，不适合这项内容判断。

1. 先写下一个有明确图像依据的问题，以及你从原图读出的答案。
2. 运行脚本，保存图片路径、问题、模型名、实际输出和调用配置。
3. 对照原图，标记回答中哪些内容有证据、哪些内容错误或无法确认。接口返回成功和答案正确要分别判断。

这份记录就是本节产出。遇到报错时按第四节定位；完成一次调用后，不需要再重复做一套同图提问练习。

## 六、受控比较：只改变问题的表达

沿用第五节的图片与问题，保持要询问的事实不变，把问题改写得更明确。例如，将“金额是多少？”改为“票据的总金额是多少？请不要把小计或税额当作总金额”。选择与你的图片相符的表述，不在提示词中泄露答案。

保持模型、图片、预处理和可设置的生成参数一致，只改变提示词。保留两组原始输出，并按同一标准检查目标事实是否正确、是否夹带无依据的内容。若要观察输出波动，可在各组内重复相同请求并记录每次结果；服务端实现也可能造成输出波动。

| 条件 | 提示词 | 实际输出 | 图像证据与错误 |
| --- | --- | --- | --- |
| 原始提问 | 填第五节的问题 | 填实际输出 | 对照原图填写 |
| 明确提问 | 填改写后的问题 | 填实际输出 | 使用相同判定标准 |

一张图上的改善只说明这个样例发生了变化，不能推出模型整体能力提高。更换自然图像与截图，是在观察不同样例上的表现；比较本地推理与 API 时，还可能同时改变权重、模板和服务设置，不能直接把差异归因于调用方式。需要扩大样本时，按[第五章](../chapter5/第五章%20评测体系与工程选型.md)建立固定评测集。

## 七、完成记录后的自查与可选扩展

直接检查第五、六节已有记录，不再重复调用：

- 能否沿代码指出图片、问题分别在哪一步进入请求或 processor？
- 本地路线中，`apply_chat_template` 起什么作用，为什么生成后要裁掉输入部分？走 API 路线时，能否解释消息中的文本与 `image_url` 字段？
- 两组比较改变了什么、保持了什么？结论是否超出了当前样本支持的范围？

希望练习代码扩展时，可增加输出保存逻辑，将 `image_path / question / answer` 写入 JSONL，并关联模型与调用配置。之后再扩展到固定图片列表；第五章已有批量评测脚本，可先阅读再决定是否自己实现。

## 八、本章小结

本章把图片、问题、消息模板和模型输出连接成一次可检查的调用，再用受控比较观察提示词的影响。读懂请求与输出，比只看到命令成功更接近独立排错。

接下来可用第五章检查回答质量，或进入第八章把调用封装成交互界面。不同示例的消息构造应逐项核对，不能因都能返回文本就假定请求完全一致。

## 九、原始论文与深入学习

- **模型论文**：*Qwen2.5-VL Technical Report*（2025，[原文](https://arxiv.org/abs/2502.13923)）。先读模型架构与动态分辨率，再对照本章 processor 的输入。
- **量化延伸论文**：[GPTQ](https://arxiv.org/abs/2210.17323)（2022）与 [AWQ](https://arxiv.org/abs/2306.00978)（2023）。分别了解后训练量化与激活感知权重量化；这是“量化部署”的进一步阅读，不表示本章脚本已实现它们，或任意 VLM 的视觉组件都能直接量化。
- **接口文档**：[Transformers 多模态模板](https://huggingface.co/docs/transformers/main/en/chat_templating_multimodal) 与 [OpenAI 图像输入指南](https://developers.openai.com/api/docs/guides/images-vision)。选择与你实际 SDK、服务端和端点一致的示例；逐项核对服务端支持的字段。

## 十、章节跳转

- 上一篇：[第六章 推理部署与 Serving](../chapter6/第六章%20推理部署与%20Serving.md)
- 下一篇：[第八章 构建一个图像问答 Demo](../chapter8/第八章%20构建一个图像问答%20Demo.md)
- 配套代码：[code 目录](https://github.com/datawhalechina/start-mllm/tree/main/docs/chapter7/code)（仓库路径 `docs/chapter7/code`）
