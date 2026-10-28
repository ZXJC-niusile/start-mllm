"""图像 API 请求共用的文件编码。"""
import base64
import mimetypes
from pathlib import Path


def to_data_url(image_path: str) -> str:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type is None:
        mime_type = "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"
