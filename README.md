<div align="center">
  <img src="./docs/images/banner.png" width="100%" alt="Start-MLLM Banner">
  <br>

  <h1>Start-MLLM（⚠️ Alpha内测版）</h1>
  <h3>从零开始学习多模态大模型</h3>
  <p><em>从概念、架构、训练到部署，系统搭建你的第一套 MLLM 知识体系</em></p>
</div>

<div align="center">
  <img src="./docs/images/readme-learning-roadmap.svg" width="860" alt="Start-MLLM 学习路线总览">
</div>

## 项目介绍

`Start-MLLM` 是面向具备 Python 基础的中文学习者的多模态大模型教程，主线为**图像与文本输入、文本输出的生成式视觉语言模型**。内容涵盖视觉表示、跨模态连接、数据与微调、评测和应用，配套提供推理、评测、Demo 与工具工作流示例。音频、视频及多模态生成仅作扩展介绍。

概念学习从基础篇开始；API 实践需要支持图像输入的接口、密钥和调用额度；本地推理与选学微调还需满足相应设备和环境要求。具体准备见[环境准备](./docs/chapter0-环境准备.md)。

## 在线阅读

[打开在线教程](https://datawhalechina.github.io/start-mllm/) · [前言：选择阅读路线](./docs/前言.md) · [基础篇总览](./docs/foundation/README.md)

## 目录

### 基础篇

| 章节名 | 简介 |
| ---- | ---- |
| [基础篇总览](./docs/foundation/README.md) | 为只有 Python 基础的学习者补齐最低工程和模型基础 |
| [00 环境、命令行与 Python 项目基础](./docs/foundation/00-环境命令行与Python项目基础.md) | 搞懂路径、终端、pip、conda、环境变量与常见报错 |
| [01 向量、矩阵与张量基础](./docs/foundation/01-向量矩阵与张量基础.md) | 手算点积与矩阵乘法，用 CPU 脚本检查视觉特征的形状与批量投影 |
| [02 NumPy 基础](./docs/foundation/02-NumPy基础.md) | 理解数组类型、索引、广播、归约与轴变换 |
| [03 PyTorch 张量与自动求导](./docs/foundation/03-PyTorch张量与自动求导.md) | 理解设备、计算图、链式法则与梯度累积 |
| [04 神经网络与学习过程](./docs/foundation/04-神经网络与学习过程.md) | 理解前向计算、损失、参数更新与泛化 |
| [05 Embedding 与 Tokenizer](./docs/foundation/05-Embedding与Tokenizer.md) | 理解文本编码、嵌入、序列长度与对话模板 |
| [06 Transformer 基础](./docs/foundation/06-Transformer基础.md) | 理解注意力、位置、残差与自回归生成 |
| [07 CV 基础与 ViT](./docs/foundation/07-CV基础与ViT.md) | 理解像素预处理、局部特征与视觉 token |
| [08 CLIP 与图文对齐](./docs/foundation/08-CLIP与图文对齐.md) | 理解双塔、对比目标与检索能力边界 |
| [09 多模态大模型入口](./docs/foundation/09-多模态大模型入口.md) | 串联视觉编码器、连接器与语言模型 |

### 入门导航

| 章节名 | 简介 |
| ---- | ---- |
| [前言](./docs/前言.md) | 系统学习、API 体验与已有 LLM 基础三条阅读路线 |
| [Chapter 0 环境准备](./docs/chapter0-环境准备.md) | 环境安装、GPU 自检、API 路线与命令行最小准备 |
| [LLM 基础速通](./docs/llm-基础速通.md) | 浏览 Transformer、Token、生成与训练的关键概念，按需补课 |
| [术语表](./docs/术语表.md) | 全书高频术语速查 |

### 理论篇（第 1～4 章）

| 章节名 | 简介 |
| ---- | ---- |
| [第一章 多模态大模型概览](./docs/chapter1/第一章%20多模态大模型概览.md) | 建立任务地图，理解多模态能力边界与常见误区 |
| [第二章 视觉编码器与跨模态对齐](./docs/chapter2/第二章%20视觉编码器与跨模态对齐.md) | 理解视觉 token、CLIP、对齐训练与输入表示 |
| [第三章 多模态生成架构](./docs/chapter3/第三章%20多模态生成架构.md) | 对照双塔匹配与生成模型的连接方式，理解 Projector 与训练范式 |
| [第四章 数据、训练与微调](./docs/chapter4/第四章%20数据、训练与微调.md) | 了解数据格式、训练流程、LoRA/QLoRA 与最小微调闭环 |

### 实战篇（第 5～10 章）

| 章节名 | 简介 |
| ---- | ---- |
| [第五章 评测体系与工程选型](./docs/chapter5/第五章%20评测体系与工程选型.md) | 建立多模态评测意识，跑通最小评测脚本 |
| [第六章 推理部署与 Serving](./docs/chapter6/第六章%20推理部署与%20Serving.md) | 理解部署链路、资源规划、性能观测与服务化思路 |
| [第七章 动手跑通你的第一个 VLM](./docs/chapter7/第七章%20动手跑通你的第一个%20VLM.md) | 使用 `Transformers` 或 OpenAI 兼容接口跑通图文问答 |
| [第八章 构建一个图像问答 Demo](./docs/chapter8/第八章%20构建一个图像问答%20Demo.md) | 基于 `Gradio` 构建最小可交互 Demo |
| [第九章 从单模态 Agent 到多模态 Agent](./docs/chapter9/第九章%20从单模态%20Agent%20到多模态%20Agent.md) | 把图像感知、工具调用与工作流连接起来 |
| [第十章 学习路线与开源项目实战建议](./docs/chapter10/第十章%20学习路线与开源项目实战建议.md) | 收束全书路线，帮助你继续扩展项目 |

### Extra-Chapter

| 章节名 | 简介 |
| ---- | ---- |
| [Extra01 OCR 与文档理解专题](./docs/Extra-Chapter/Extra01-OCR与文档理解专题.md) | 面向票据、合同、截图、版面理解等场景 |
| [Extra02 多图输入与比较专题](./docs/Extra-Chapter/Extra02-多图输入与比较专题.md) | 面向商品多图、版本比较、跨图归纳等任务 |
| [Extra03 长图处理与切块策略专题](./docs/Extra-Chapter/Extra03-长图处理与切块策略专题.md) | 面向长截图、长网页、长文档输入策略 |

## 贡献者名单

| 姓名 | 职责 | 简介 |
| :---- | :---- | :---- |
| [ZXJC-niusile](https://github.com/ZXJC-niusile) | 项目负责人 / 当前唯一贡献者 | 和大家一起学习多模态大模型，把复杂知识拆成能读懂、能跑通的教程 |

## 参与贡献

我们欢迎任何形式的参与，每一条 Issue 和 PR 都会让这份教程变得更好：

- **报告 Bug**：发现内容错误、表达不清、代码跑不通？直接提 [Issue](https://github.com/datawhalechina/start-mllm/issues)，请附上运行环境、复现步骤和报错；维护者会按影响范围逐步处理。
- **提出建议**：对章节结构、学习路径、内容深度有想法？欢迎在 Discussion 或 Issue 中交流。
- **完善内容**：修正示例、补充解释、优化脚手架？欢迎提 Pull Request。
- **贡献实战经验**：把你日常使用 MLLM 的踩坑记录、场景适配经验、评测心得整理成文档，贡献到 `Extra-Chapter`。像社区博客一样，让后来者少走弯路。


## Star History

[![Star History Chart](https://api.star-history.com/chart?repos=datawhalechina/start-mllm&type=date&legend=top-left&sealed_token=KU5qjLH1rvwlDW3BMK8q2a4_qcMFTxKsHT8X29Pe3EsnM0l_iwiTi5t4TSFvuqj3etqzPTR-3fncqgcM7a4Gv3z9ZZpxbbt2JPPvEds6t7-WzVq_1N_v0w)](https://www.star-history.com/?type=date&repos=datawhalechina%2Fstart-mllm)

## 关注我们

<div align=center>
<p>扫描下方二维码关注公众号：Datawhale</p>
<img src="https://raw.githubusercontent.com/datawhalechina/pumpkin-book/master/res/qrcode.jpeg" width = "180" height = "180">
</div>

## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />当前仓库中的教程文档与配套示例代码暂统一采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。
