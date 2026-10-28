# 开发检查

在仓库根目录运行，使用 Python 3.10 或更高版本：

```bash
python scripts/run_tests.py
python scripts/check_docs.py
```

测试入口会查找 `docs` 下所有 `test_*.py` 所在目录，分别执行 unittest 并汇总退出码。测试使用标准库与 SDK/模型桩，不安装训练依赖、不下载模型、不调用 API；测试失败或未发现测试文件时返回非零退出码。Windows 无符号链接权限时，相关子用例会明确跳过。

文档检查覆盖 Git 跟踪的 Markdown 文件，检查本地目标路径、显式片段 ID 与代码围栏。跨页片段链接应指向显式 HTML `id`，以兼容 GitHub 和 Docsify。它不检查外部网址可用性或浏览器渲染。

上述命令可在 Windows 与 Linux 中执行。真实 GPU 训练、模型兼容性和在线 API 行为需要单独验证。
