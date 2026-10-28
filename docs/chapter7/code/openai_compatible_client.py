import os
import sys
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

_DOCS = Path(__file__).resolve().parents[2]
if str(_DOCS) not in sys.path:
    sys.path.insert(0, str(_DOCS))
from learner_paths import resolve_image_path
from learner_io import to_data_url

load_dotenv()


BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")
MODEL_ID = os.getenv("MODEL_ID", "your-vlm")
IMAGE_PATH = os.getenv("IMAGE_PATH", "").strip()
QUESTION = os.getenv("QUESTION", "请详细描述这张图，并说明你判断的依据。")




def main() -> int:
    try:
        try:
            max_tokens = int(os.getenv("MAX_TOKENS", "256"))
        except ValueError:
            raise ValueError("MAX_TOKENS must be a positive integer.") from None
        if max_tokens <= 0:
            raise ValueError("MAX_TOKENS must be a positive integer.")

        client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
        image_file = str(resolve_image_path(IMAGE_PATH or None, script_file=__file__))
        data_url = to_data_url(image_file)

        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": QUESTION},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            max_tokens=max_tokens,
        )

        if not response.choices:
            raise ValueError("Model returned no choices.")
        answer = response.choices[0].message.content
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Model returned no text answer.")
        print(answer)
        return 0
    except Exception as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
