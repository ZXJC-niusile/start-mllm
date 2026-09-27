"""
测试当前运行目录的最小脚本。
用于帮助初学者理解“当前工作目录”的概念以及如何排查 FileNotFoundError。
"""

from pathlib import Path

# 获取当前工作目录（即你在哪里运行这个脚本）
current_dir = Path.cwd()
print("当前运行目录:", current_dir)

print("\n当前目录下的文件和文件夹:")
# 遍历当前目录下的所有文件和文件夹，打印它们的名字
for item in current_dir.iterdir():
    print("-", item.name)
