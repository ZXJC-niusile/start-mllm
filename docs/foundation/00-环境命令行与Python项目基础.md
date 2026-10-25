# 00 环境、命令行与 Python 项目基础

## 0.1 为什么基础篇从这里开始

很多初学者在学习多模态大模型（MLLM）或者跑开源项目时，经常遇到的第一道大槛，并不是“模型看不懂”，而是：

- 按照教程复制了代码，却提示找不到文件（`FileNotFoundError`）。
- 刚刚 `pip install` 安装了包，运行代码却提示找不到模块（`ModuleNotFoundError`）。
- 看着教程里写的 `cd start-mllm` 满头大汗，不知道该往哪里输入。

**先把环境、路径、依赖这些“工程基建”弄清楚，后面跑脚本、跑 Demo、做评测才不会把所有的环境错误都误认为“模型坏了”。**

---

## 0.2 终端 / 命令行是什么

在日常使用电脑时，我们习惯用鼠标双击文件夹、点击图标来运行软件。而“终端（Terminal）”或者“命令行（Command Line）”则是用**纯打字**的方式给电脑下指令的地方。

不同的操作系统有自己不同的默认终端：
- **Windows**：通常使用 **PowerShell** 或传统的 cmd。
- **macOS / Linux**：通常使用 **Terminal**（终端），背后运行的往往是 bash 或 zsh。

不用对黑乎乎的窗口感到恐惧，把它当成一个只能听懂特定指令的“打字聊天框”就可以了。

---

## 0.3 当前目录是什么

我们在使用鼠标时，总是清楚自己打开了哪个文件夹（比如在“我的文档”或者“D 盘”里）。但在终端里，**电脑也是有“当前身处哪个文件夹”的概念的，这就叫“当前目录（Current Working Directory）”**。

- **绝对路径**：从最根部的盘符或者系统根目录开始的完整路径，比如 `C:\Users\data\start-mllm` 或 `/home/user/start-mllm`。
- **相对路径**：相对于你**当前所在的目录**出发的路径。比如 `./docs` 就是“当前目录下的 docs 文件夹”。

**为什么很多教程要求从仓库根目录运行命令？**
因为教程里的代码通常使用的是“相对路径”。如果你没在根目录运行代码，代码去找 `./docs/images/...` 的时候，就会从错误的起点出发，最后导致 `FileNotFoundError`（找不到文件）。

你可以用下面的命令来查看和切换当前目录：

**在 Windows PowerShell 中：**
```powershell
# 打印当前你到底在哪个目录下
pwd

# 列出当前目录下都有哪些文件和文件夹
dir

# 进入名为 start-mllm 的文件夹（类似于双击进入）
cd start-mllm
```

**在 Linux/macOS 中：**
```bash
# 打印当前工作目录 (Print Working Directory)
pwd

# 列出当前目录的内容 (List)
ls

# 进入名为 start-mllm 的文件夹
cd start-mllm
```

---

## 0.4 Python 命令到底在运行哪个 Python

当你学会了基础语法，你的电脑里可能已经不止有一个 Python 了。系统自带一个、你安装软件时附带一个、你可能又自己装了一个。

当你在终端里输入 `python` 时，电脑会去环境变量里找到排在最前面的那个 Python 解释器。为了防止“装错了地方”，你必须随时能确认**现在正在干活的是哪一个 Python**。

在终端中输入：
```bash
# 查看当前 Python 的版本
python --version

# 查看当前运行的 Python 的确切绝对路径
python -c "import sys; print(sys.executable)"
```

如果这个路径不是你心里预期的那个，那你后续安装的所有包都会装到别的地方去！

---

## 0.5 pip install 在做什么

`pip` 是 Python 的包管理器，`pip install` 的作用就是把第三方的代码包下载并安装到**当前的 Python 环境里**。

**最常见的误区**：你的系统里有两个环境 A 和 B。你在终端里用 `pip` 把包装到了 A 环境里，但是你的代码编辑器（比如 VSCode / PyCharm）或者当前终端使用的是 B 环境。于是代码一跑就报错：`ModuleNotFoundError`。

为了确保你把包准确地装进当前你正在使用的 Python 里，**强烈推荐不要直接使用 `pip install`，而是使用 `python -m pip install`**：

```bash
# 确保是当前 Python 对应的 pip
python -m pip --version

# 推荐的安装命令格式
python -m pip install openai
```
这样就能保证：你用 `python` 运行代码，包就老老实实装在这个 `python` 下面。

---

## 0.6 conda / venv 为什么有用

如果你以后要做十个不同的项目，有的项目要用 Python 3.8，有的要用 Python 3.10；有的要求包装 1.0 版本，有的要求 2.0 版本，混在一起一定会爆炸。

**环境隔离**（比如 conda 或 venv）就是为了给每个项目建一间“独立的小屋”。在这个小屋里，Python 的版本和装的包都是独立的，不会污染系统的其他环境。

这里给出 `conda` 的推荐使用命令（如果你暂时不用 conda，也可以先用已有的 Python 环境，但以后遇到依赖冲突时，建议回来补课）：

```bash
# 创建一个名为 start-mllm 的独立环境，并指定使用 Python 3.10
conda create -n start-mllm python=3.10 -y

# 激活进入这个环境（进入小屋）
conda activate start-mllm

# 验证一下，现在是不是在这个新的 Python 里了
python --version
```

---

## 0.7 requirements.txt 是什么

`requirements.txt` 是一个普通的文本文件，里面列出了这个项目（或者当前章节）需要用到的所有第三方包的名字和版本号。它是项目的“依赖清单”。

我们可以用一行命令让 `pip` 根据清单批量安装：
```bash
# -r 代表从文件中读取清单
python -m pip install -r docs/chapter7/code/requirements.txt
```

> **注意**：不要一看到项目就盲目地在所有目录乱执行 `pip install`。大型项目常常按模块拆分了不同的 requirements，先看当前章节需要什么再装。

---

## 0.8 .env 和环境变量是什么

在跑大模型或者实际开发中，我们经常需要配置 API 密钥（Key）、模型名称或者服务器地址。**把这些私密或容易变的配置直接写死在 Python 代码里是非常危险且不专业的**（万一你把代码传到了网上，别人就能偷用你的 Key 刷爆你的卡）。

所以，通常我们在当前目录下创建一个叫 `.env` 的隐藏文件（注意前面有个点），把配置写在里面：

```bash
OPENAI_BASE_URL=http://127.0.0.1:8000/v1
OPENAI_API_KEY=EMPTY
MODEL_ID=your-vlm
```
这只是配置说明，不是 Python 代码，也不要用 Python 去运行它。我们的 Python 代码会用特殊的工具库去读取这些环境变量。

如果你想在命令行临时给本次运行设置一个环境变量，你可以这样做：

**PowerShell 示例：**
```powershell
$env:MODEL_ID="your-vlm"
```

**Linux/macOS 示例：**
```bash
export MODEL_ID=your-vlm
```

---

## 0.9 最小 Python 路径练习

为了加深你对路径的理解，这里提供了一段测试代码。我们把它放在了 `docs/foundation/code/00_path_check.py` 中：

```python
from pathlib import Path

current_dir = Path.cwd()
print("当前运行目录:", current_dir)

print("\n当前目录下的文件和文件夹:")
for item in current_dir.iterdir():
    print("-", item.name)
```

**运行方式与解释：**

请在终端里执行：
```bash
python docs/foundation/code/00_path_check.py
```

- 如果你是从 **仓库根目录** 运行的，你会看到终端打印出的列表里包含 `README.md`、`docs` 等文件夹。
- 如果你是用 `cd` 进入了别的文件夹再运行这个脚本，你的“当前目录”就变了，输出的文件列表也会完全不同。
- 这就是为什么你在运行某些读取图片的脚本时，如果没有找对“当前工作目录”，就会遇到找不到图片的错误。

---

## 0.10 两个最常见错误：ModuleNotFoundError 和 FileNotFoundError

最后，我们来盘点一下每天都会见到的两位“老朋友”，并教你做最小排查。

### ModuleNotFoundError（找不到模块）
**常见原因：**
- 确实忘记安装这个包了。
- 安装到了环境 A，但你在环境 B 里运行代码。
- 忘记激活 conda 环境（没有执行 `conda activate xxx`）。
- 你的 Python 文件名和包名重名了（比如你写了个 `openai.py`，代码就会错误地去导入你自己，而不是真正的工具包）。

**排查命令：**
```bash
# 看看包究竟有没有装，装在了哪里
python -m pip show openai

# 看看现在到底是在用哪个 Python 跑
python -c "import sys; print(sys.executable)"
```

### FileNotFoundError（找不到文件或目录）
**常见原因：**
- 你在终端里的“当前目录”不对。
- 相对路径拼写错误。
- 目标文件还没被生成出来。
- Windows (反斜杠 `\`) 和 Linux (正斜杠 `/`) 的路径分隔符混乱。

**排查方法：**
不要盯着代码看，立刻在终端里用命令查看一下周围：
```bash
# Linux/macOS 看这里
pwd
ls

# Windows PowerShell 看这里
pwd
dir
```
看看你想读取的文件，是不是真的躺在你以为的地方。

---

## 0.11 本章练习

请务必动手完成以下练习：

**必做：**
1. 打开终端，使用 `cd` 命令进入 `start-mllm` 仓库根目录。
2. 运行 `python --version` 查看版本。
3. 运行 `python -c "import sys; print(sys.executable)"` 查看 Python 具体路径。
4. 运行 `python docs/foundation/code/00_path_check.py`。
5. 观察输出列表，确认是否能看到 `README.md` 和 `docs`。

**进阶：**
1. 故意用 `cd docs` 进入 docs 目录，然后再次尝试运行刚刚那个脚本（注意此时你的运行命令应该怎么写？）。
2. 观察输出的变化。
3. 试着用自己的话解释，为什么你在子目录运行代码时，原来写着相对路径 `../images/xxx.png` 的代码有可能会找不到文件。

**挑战：**
1. 创建一个新的 conda 虚拟环境，名叫 `mllm-test`。
2. 激活这个环境。
3. 在新环境中运行 `python --version` 和 `python -m pip --version`。
4. 用自己的话写下：这个独立的小屋和系统的 Python 有什么根本区别。

---

## 0.12 本章小结

恭喜你！学完这一章，你已经具备了初级但极其重要的工程调试能力。你现在真正带走的武器是：
- 会随时确认**自己在哪个目录**。
- 会随时确认**自己正在用哪个 Python**。
- 知道该如何安全、正确地**安装依赖**。
- 知道 `requirements.txt` 和 `.env` 分别有什么作用。
- 在遇到模块报错和文件报错时，不再慌乱，而是会用几个命令做初步排查。

后续基础篇仍在建设中。现在可以进入 [Chapter 0 环境准备](../chapter0-环境准备.md) 配置所选路线的环境，再按需阅读 [LLM 基础速通](../llm-基础速通.md)；想先获得一次运行体验，也可以在准备好图文接口、密钥与额度后，按 [第七章](../chapter7/第七章%20动手跑通你的第一个%20VLM.md) 的 API 路线继续。
