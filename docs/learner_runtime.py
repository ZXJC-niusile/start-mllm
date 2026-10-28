"""示例共用的响应和配置检查，不依赖 SDK。"""
import os


def first_message(response):
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("Model returned no choices.")
    message = getattr(choices[0], "message", None)
    if message is None:
        raise ValueError("Model returned no message.")
    return message


def positive_env_int(name, default, *, maximum=None):
    """未设置或空白使用默认值；其他值必须为范围内的正整数。"""
    raw = os.getenv(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        raise ValueError(f"{name} must be a positive integer.") from None
    if value <= 0 or (maximum is not None and value > maximum):
        limit = f" between 1 and {maximum}" if maximum is not None else " greater than 0"
        raise ValueError(f"{name} must be an integer{limit}.")
    return value
