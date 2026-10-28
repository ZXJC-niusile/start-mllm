占位图由脚本生成，不默认提交到 Git。

在 `docs/chapter4/code` 下执行：

```bash
python create_placeholder_images.py
```

再运行 `python prepare_dataset_manifest.py`，含图样本的 `image_exists` 应为 `true`；纯文本样本 `sample-007` 没有图片，该字段为 `false`。
