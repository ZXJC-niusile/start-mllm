import argparse
import json
import sys
from pathlib import Path
from typing import Any


def validate_record(record: dict[str, Any], *, dataset_dir: Path) -> dict[str, Any]:
    raw = record.get("image", "")
    image_path = Path(raw) if Path(raw).is_absolute() else (dataset_dir / raw)
    messages = record.get("messages", [])
    user_turns = sum(1 for item in messages if item.get("role") == "user")
    assistant_turns = sum(1 for item in messages if item.get("role") == "assistant")
    has_empty_text = False
    content_images = []

    # 训练读取 content 中的图片，而非仅用于索引的顶层 image。
    for i, message in enumerate(messages):
        for j, content in enumerate(message.get("content", [])):
            if content.get("type") == "image":
                content_path = Path(content["image"])
                if not content_path.is_absolute():
                    content_path = dataset_dir / content_path
                content_images.append({
                    "field": f"messages[{i}].content[{j}].image",
                    "image": str(content_path),
                    "image_exists": content_path.is_file(),
                })

    for message in messages:
        for content in message.get("content", []):
            if content.get("type") == "text" and not str(content.get("text", "")).strip():
                has_empty_text = True

    return {
        "id": record.get("id", ""),
        "image": str(image_path),
        "image_exists": image_path.is_file(),
        "content_images": content_images,
        "message_count": len(messages),
        "user_turns": user_turns,
        "assistant_turns": assistant_turns,
        "has_empty_text": has_empty_text,
    }


def validate_sft_format(record: dict[str, Any], line_num: int, *, dataset_dir: Path | None = None) -> list[str]:
    """
    校验本仓 LoRA 示例支持的 SFT 结构与图片索引约定。
    返回错误列表，空列表表示通过。
    """
    errors: list[str] = []
    if not isinstance(record, dict):
        return [f"[{line_num}] 每条样本必须是 JSON 对象"]
    if not isinstance(record.get("image", ""), str):
        errors.append(f"[{line_num}] 'image' 必须是路径字符串（纯文本样本可为空）")

    if "images" in record:
        errors.append(f"[{line_num}] 不支持顶层 images；请在 user.content 中声明每张图片")

    # 1. 必须有 id
    if not record.get("id"):
        errors.append(f"[{line_num}] 缺少 'id' 字段")

    # 2. 必须有 messages
    messages = record.get("messages")
    if not messages:
        errors.append(f"[{line_num}] 缺少 'messages' 字段或为空")
        return errors
    if not isinstance(messages, list):
        errors.append(f"[{line_num}] 'messages' 必须是列表")
        return errors

    # 3. messages 结构校验
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append(f"[{line_num}] messages[{i}] 必须是字典")
            continue

        role = msg.get("role")
        if role not in ("user", "assistant", "system"):
            errors.append(f"[{line_num}] messages[{i}].role 非法: '{role}'，必须为 user/assistant/system")

        if role == "system" and i != 0:
            errors.append(f"[{line_num}] system 消息只允许出现在开头")

        content = msg.get("content")
        if not content:
            errors.append(f"[{line_num}] messages[{i}] 缺少 'content'")
            continue
        if not isinstance(content, list):
            errors.append(f"[{line_num}] messages[{i}].content 必须是列表")
            continue

        for j, part in enumerate(content):
            if not isinstance(part, dict):
                errors.append(f"[{line_num}] messages[{i}].content[{j}] 必须是字典")
                continue

            part_type = part.get("type")
            if part_type not in ("text", "image"):
                errors.append(
                    f"[{line_num}] messages[{i}].content[{j}].type 非法: '{part_type}'，必须为 text/image"
                )

            if part_type == "text":
                text_val = part.get("text")
                if text_val is None:
                    errors.append(f"[{line_num}] messages[{i}].content[{j}] text 类型缺少 'text' 值")
                elif not isinstance(text_val, str):
                    errors.append(f"[{line_num}] messages[{i}].content[{j}] text 值必须是字符串")
                elif not text_val.strip():
                    errors.append(f"[{line_num}] messages[{i}].content[{j}] text 值为空")

            if part_type == "image":
                if role != "user":
                    errors.append(f"[{line_num}] messages[{i}].content[{j}] 图片只允许出现在 user 消息中")
                img_val = part.get("image")
                if not isinstance(img_val, str) or not img_val.strip():
                    errors.append(f"[{line_num}] messages[{i}].content[{j}] image 类型需要非空路径字符串")

    # 4. 轮次校验：必须以 user 开始，user/assistant 交替（允许 system 在开头）
    indexed = [(i, m) for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") != "system"]
    non_system = [m for _, m in indexed]
    if non_system:
        if non_system[0].get("role") != "user":
            errors.append(f"[{line_num}] 非 system 消息必须以 user 开始")
        for i in range(1, len(non_system)):
            expected = "assistant" if non_system[i - 1].get("role") == "user" else "user"
            if non_system[i].get("role") != expected:
                errors.append(
                    f"[{line_num}] 轮次不交替: messages[{indexed[i][0]}] 为 {non_system[i].get('role')}，"
                    f"期望 {expected}"
                )

    if not non_system or non_system[-1].get("role") != "assistant":
        errors.append(f"[{line_num}] SFT 样本必须以 assistant 回复结束")

    # 5. user 消息中至少包含一个 image 或 text
    for i, msg in enumerate(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            content = [part for part in content if isinstance(part, dict)]
            has_image = any(c.get("type") == "image" for c in content)
            has_text = any(c.get("type") == "text" and str(c.get("text", "")).strip() for c in content)
            if not has_image and not has_text:
                errors.append(f"[{line_num}] messages[{i}] (user) 既无 image 也无有效 text")

    # 6. assistant 消息中至少包含一个 text
    for i, msg in enumerate(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            content = [part for part in content if isinstance(part, dict)]
            has_text = any(c.get("type") == "text" and str(c.get("text", "")).strip() for c in content)
            if not has_text:
                errors.append(f"[{line_num}] messages[{i}] (assistant) 缺少有效 text")

    top = record.get("image", "")
    if isinstance(top, str) and top:
        base = dataset_dir or Path.cwd()
        top_path = (base / top).resolve()
        declared = []
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict) or msg.get("role") != "user" or not isinstance(msg.get("content"), list):
                continue
            for j, part in enumerate(msg["content"]):
                if isinstance(part, dict) and part.get("type") == "image" and isinstance(part.get("image"), str):
                    declared.append((f"messages[{i}].content[{j}].image", (base / part["image"]).resolve()))
        if top_path not in [path for _, path in declared]:
            fields = ", ".join(f"{field}={path}" for field, path in declared) or "无图片声明"
            errors.append(f"[{line_num}] 顶层 image 必须对应 user.content 中的图片；当前 {fields}")

    return errors


def _try_utf8_stdio() -> None:
    """Use UTF-8 for Chinese diagnostics when stdout/stderr are redirected."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (OSError, ValueError, AttributeError):
                pass


def main() -> None:
    _try_utf8_stdio()
    parser = argparse.ArgumentParser(
        description="校验多模态 SFT 风格 JSONL（图片是否存在、轮次、SFT 格式合规性等）。"
    )
    parser.add_argument(
        "--dataset",
        "-d",
        default="sample_multimodal_sft.jsonl",
        help="JSONL 路径（相对当前工作目录或绝对路径），默认 sample_multimodal_sft.jsonl",
    )
    parser.add_argument(
        "--check-sft-format",
        action="store_true",
        help="启用严格的 SFT 格式校验（role/content 结构、类型合法性、轮次交替等）",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = (Path.cwd() / dataset_path).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataset_dir = dataset_path.parent
    summaries: list[dict[str, Any]] = []
    sft_errors: list[str] = []
    line_num = 0

    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line_num += 1
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                if not args.check_sft_format:
                    raise
                sft_errors.append(f"[{line_num}] JSON 解析失败: {exc.msg}")
                continue
            if args.check_sft_format:
                errors = validate_sft_format(record, line_num, dataset_dir=dataset_dir)
                if errors:
                    sft_errors.extend(errors)
                    continue  # 结构未通过时，不能交给假设字段有效的基础统计函数。
            summary = validate_record(record, dataset_dir=dataset_dir)
            summaries.append(summary)
            if args.check_sft_format:
                if record.get("image") and not summary["image_exists"]:
                    sft_errors.append(f"[{line_num}] 顶层 image 不存在或不是文件: {summary['image']}")
                for image in summary["content_images"]:
                    if not image["image_exists"]:
                        sft_errors.append(
                            f"[{line_num}] {image['field']} 图片不存在或不是文件: {image['image']}"
                        )

    # 基础校验结果
    print(json.dumps(summaries, ensure_ascii=False, indent=2))

    # SFT 格式校验报告
    if args.check_sft_format:
        print("\n" + "=" * 50)
        if sft_errors:
            print(f"SFT 格式校验：发现 {len(sft_errors)} 个问题")
            for err in sft_errors:
                print(f"  ❌ {err}")
            print("\n修复建议：")
            print("  1. 每条样本必须有 'id' 字段")
            print("  2. messages 必须是列表，每个元素是字典")
            print("  3. role 必须是 user/assistant/system 之一")
            print("  4. content 必须是列表，元素 type 必须是 text 或 image")
            print("  5. user/assistant 必须交替出现，以 user 开始")
            print("  6. user 消息至少包含一个 image 或 text")
            print("  7. assistant 消息至少包含一个 text，且样本以 assistant 结束")
            raise SystemExit(1)
        else:
            print("SFT 格式校验：✅ 全部通过（共 {} 条）".format(len(summaries)))


if __name__ == "__main__":
    main()
