"""
最小可跑的多模态 LoRA 微调脚本
================================
目标：在 Colab T4 (16GB) 或本地 24GB 卡上，对 Qwen2.5-VL-3B-Instruct
      做 100 条数据的 caption 微调，建立"数据 → 训练 → 验证"的闭环手感。

依赖安装（与 requirements-api.txt 不冲突，需要 torch）：
    pip install torch transformers accelerate peft datasets pillow

最低版本要求：transformers >= 4.45（Qwen2.5-VL 模型支持）

运行前准备：
    1. 准备自己的图文数据，格式同 sample_multimodal_sft.jsonl
    2. 修改下面 DATA_PATH 指向你的 JSONL
    3. 运行：python lora_finetune_minimal.py
"""

import json
import os
import warnings
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    __version__ as transformers_version,
)

_TRANSFORMERS_MIN = "4.45"
_TRANSFORMERS_VER_TUPLE = tuple(int(x) for x in _TRANSFORMERS_MIN.split("."))
_TRANSFORMERS_CUR_TUPLE = tuple(int(x) for x in transformers_version.split(".")[:2])
if _TRANSFORMERS_CUR_TUPLE < _TRANSFORMERS_VER_TUPLE:
    warnings.warn(
        f"当前 transformers 版本为 {transformers_version}，建议升级到 >= {_TRANSFORMERS_MIN} 以确保 Qwen2.5-VL 兼容。"
        f"运行：pip install --upgrade transformers",
        stacklevel=1,
    )


# ======================== 用户配置区 ========================
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DATA_PATH = Path(__file__).with_name("sample_multimodal_sft.jsonl")
OUTPUT_DIR = Path("./lora_output")
MAX_SAMPLES = 100          # 先用 100 条建立手感
NUM_EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM = 4             # 等效 batch_size = 1 * 4 = 4
LR = 2e-4
MAX_SEQ_LEN = 512
LORA_R = 16                # LoRA rank：越大表达能力越强，显存越高
LORA_ALPHA = 32            # 常见取值为 rank 或 2 * rank；越大对原模型影响越强
LORA_DROPOUT = 0.05
# Qwen2.5-VL 的微调目标模块（含注意力与 FFN 层），不同模型需查官方文档
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# ==========================================================


def load_jsonl(path: Path) -> list[dict]:
    """加载 JSONL，取前 MAX_SAMPLES 条。"""
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if len(samples) >= MAX_SAMPLES:
                break
    return samples


def format_for_qwen25vl(sample: dict) -> dict | None:
    """保留角色与图像占位；图片相对 DATA_PATH 解析，纯文本样本不打开图片。"""
    messages = []
    image_count = 0
    for index, message in enumerate(sample.get("messages", [])):
        role = message["role"]
        if role not in ("system", "user", "assistant") or (role == "system" and index != 0):
            raise ValueError("仅支持开头的 system 和交替的 user/assistant 消息")
        parts = []
        for part in message["content"]:
            if part["type"] == "text":
                parts.append({"type": "text", "text": part["text"]})
            elif part["type"] == "image" and role == "user":
                path = Path(part["image"])
                if not path.is_absolute():
                    path = DATA_PATH.parent / path
                if not path.is_file():
                    return None
                with Image.open(path) as image:
                    parts.append({"type": "image", "image": image.convert("RGB")})
                image_count += 1
            else:
                raise ValueError("本示例只支持文本回复与 user 消息中的图片")
        if role == "assistant" and not any(p["type"] == "text" and p["text"].strip() for p in parts):
            raise ValueError("assistant 回复不能为空")
        messages.append({"role": role, "content": parts})
    turns = [m["role"] for m in messages if m["role"] != "system"]
    if not turns or len(turns) % 2 or turns != ["user", "assistant"] * (len(turns) // 2):
        raise ValueError("训练样本必须包含完整、交替的 user/assistant 轮次")
    if sample.get("image") and not image_count:
        raise ValueError("顶层 image 不会自动插入占位；请在 user.content 中声明图片")
    return {"messages": messages}


def _images_in(messages):
    return [part["image"] for message in messages for part in message["content"]
            if part["type"] == "image"]


def _encode_prefix(processor, messages, *, generation_prompt=False):
    """边界也走多模态 processor，计入展开后的视觉 token。"""
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=generation_prompt,
    )
    return processor(
        text=[text], images=_images_in(messages) or None,
        add_special_tokens=False, truncation=False,
    )["input_ids"][0]


def collate_fn(batch, processor):
    """只监督每轮 assistant；边界无法对齐或过长时停止，避免静默错训。"""
    texts = [processor.apply_chat_template(
        item["messages"], tokenize=False, add_generation_prompt=False,
    ) for item in batch]
    images = [image for item in batch for image in _images_in(item["messages"])]
    # 不截断视觉 token；Trainer 负责把 CPU batch 移到训练设备。
    inputs = processor(
        text=texts, images=images or None, return_tensors="pt", padding=True,
        add_special_tokens=False, truncation=False,
    )
    labels = inputs["input_ids"].clone()
    for i, item in enumerate(batch):
        positions = [j for j, active in enumerate(inputs["attention_mask"][i].tolist()) if active]
        token_ids = [inputs["input_ids"][i, j].item() for j in positions]
        if len(token_ids) > MAX_SEQ_LEN:
            raise ValueError(f"样本含 {len(token_ids)} tokens，超过 MAX_SEQ_LEN={MAX_SEQ_LEN}；"
                             "请缩短文本、降低图片分辨率或调整预算，不能直接截断视觉 token")
        labels[i, :] = -100
        supervised = 0
        messages = item["messages"]
        for turn, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            prompt_ids = _encode_prefix(processor, messages[:turn], generation_prompt=True)
            through_ids = _encode_prefix(processor, messages[:turn + 1])
            start, end = len(prompt_ids), len(through_ids)
            if not (start < end and token_ids[:start] == prompt_ids and token_ids[:end] == through_ids):
                raise ValueError("chat template/tokenizer 前缀不一致，无法安全定位 assistant 标签")
            for j in positions[start:end]:
                labels[i, j] = inputs["input_ids"][i, j]
                supervised += 1
        if not supervised:
            raise ValueError("样本没有可监督的 assistant token")
    inputs["labels"] = labels
    return inputs


class SimpleMultimodalDataset(torch.utils.data.Dataset):
    """极简 Dataset，返回格式化后的单条样本。"""
    def __init__(self, samples: list[dict]):
        formatted = [format_for_qwen25vl(s) for s in samples]
        # 过滤掉图片缺失的样本
        before_count = len(formatted)
        self.samples = [s for s in formatted if s is not None]
        skipped = before_count - len(self.samples)
        if skipped > 0:
            print(f"[警告] 跳过 {skipped} 条图片缺失的样本（共 {before_count} 条）")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def main():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 加载模型和 processor
    print(f"\nLoading model: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Model loaded. Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M")

    # 2. 配置 LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. 加载数据
    print(f"\nLoading data from: {DATA_PATH}")
    raw_samples = load_jsonl(DATA_PATH)
    print(f"Total samples: {len(raw_samples)}")
    dataset = SimpleMultimodalDataset(raw_samples)
    if not len(dataset):
        raise ValueError("没有有效训练样本；请先检查 JSONL 与图片路径")

    # 4. 训练参数
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
        report_to="none",  # 不连 wandb，简化首次体验
    )

    # 5. 自定义 data collator（简化版）
    def data_collator(batch):
        return collate_fn(batch, processor)

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # 7. 开始训练
    print("\n========== 开始训练 ==========")
    trainer.train()

    # 8. 保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR / "lora_adapter")
    processor.save_pretrained(OUTPUT_DIR / "lora_adapter")
    print(f"\nLoRA adapter saved to: {OUTPUT_DIR / 'lora_adapter'}")

    # 9. 显存占用总结
    if torch.cuda.is_available():
        max_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak GPU memory: {max_mem:.2f} GB")
        print(f"\n若显存不足，可尝试：")
        print(f"  - 减小 LORA_R（当前 {LORA_R}）→ 8")
        print(f"  - 缩短文本或降低输入图片分辨率（token 上限 {MAX_SEQ_LEN}，不自动截断）")
        print(f"  - 启用 QLoRA（把模型加载改为 4-bit）")

    print("\n========== 训练完成 ==========")
    print("下一步建议：")
    print("  1. 用 merge_and_unload() 合并 LoRA 权重，或直接加载 adapter 推理")
    print("  2. 用训练样本检查是否学会拟合；另留未参与训练的图片评估泛化")
    print("  3. 把评测 JSONL（第五章格式）跑一遍，量化收益")


if __name__ == "__main__":
    main()
