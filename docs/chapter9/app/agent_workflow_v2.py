"""
多模态 Agent 骨架 v2：基于 OpenAI Tool-Calling 的 JSON Schema 路由
===============================================================
与 agent_workflow_demo.py 的区别：
  - v1（demo 版）用关键字匹配（if "报错" in summary）路由工具，教学和原型可用
  - v2（本文件）用 OpenAI 标准的 tool-calling JSON schema，演示一轮工具调用协议（仍是教学骨架）

依赖：
    pip install openai python-dotenv

运行：
    $env:OPENAI_BASE_URL="你的真实接口"
    $env:OPENAI_API_KEY="你的真实密钥"
    $env:MODEL_ID="你的真实模型名"
    $env:IMAGE_PATH="docs/chapter5/code/images/sample_ui.png"
    $env:QUESTION="请分析这张截图并给出下一步建议。"
    python agent_workflow_v2.py
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

_DOCS = Path(__file__).resolve().parents[2]
if str(_DOCS) not in sys.path:
    sys.path.insert(0, str(_DOCS))
from learner_paths import resolve_image_path
from learner_io import to_data_url


load_dotenv()




# ========== Tool Schema 定义 ==========
# 这是 OpenAI 兼容接口的标准函数调用格式，参数仍需在本地校验，工具实现仍为占位
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "knowledge_lookup",
            "description": "当截图或描述中包含报错、异常、错误码时，调用此工具查询知识库或 FAQ",
            "parameters": {
                "type": "object",
                "properties": {
                    "error_keywords": {
                        "type": "string",
                        "description": "提取到的关键错误信息或错误码",
                    },
                },
                "required": ["error_keywords"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "structured_extract",
            "description": "当需要提取结构化字段（如金额、日期、表格内容）时调用此工具",
            "parameters": {
                "type": "object",
                "properties": {
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "需要提取的字段列表，如 ['金额', '日期', '商户名']",
                    },
                },
                "required": ["fields"],
            },
        },
    },
]


class MultimodalAgentV2:
    def __init__(self) -> None:
        self.client = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
            api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
        )
        self.model_id = os.getenv("MODEL_ID", "your-vlm")

    def perceive_and_plan(self, image_path: str, question: str) -> dict:
        """
        第一步：感知 + 规划。
        把图片和问题发给模型，让模型自己决定是否需要调用工具。
        如果模型决定调用工具，会返回 tool_calls；否则直接返回文本回答。
        """
        data_url = to_data_url(image_path)

        # 系统提示：告诉模型它的角色和可用工具
        system_prompt = (
            "你是一个多模态助手。请先分析用户提供的图片，然后根据用户问题决定下一步：\n"
            "1. 如果图片包含报错、异常、错误码，请调用 knowledge_lookup 工具\n"
            "2. 如果需要提取金额、日期、表格等结构化信息，请调用 structured_extract 工具\n"
            "3. 否则直接回答用户问题。工具目前只是占位实现；若返回 not_implemented 或 error，"
            "请明确说明未完成检索或抽取，不要把占位结果当成事实。"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]

        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",  # 让模型自己决定是否调用工具
            max_tokens=512,
        )

        message = response.choices[0].message
        result = {
            "content": message.content,
            "tool_calls": [],
            "messages": messages + [message.model_dump(exclude_none=True)],
        }

        # 检查模型是否请求了工具调用
        if message.tool_calls:
            for tc in message.tool_calls:
                result["tool_calls"].append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,  # 保留原始 JSON，执行前再校验
                })

        return result

    def execute_tools(self, tool_calls: list[dict]) -> list[dict]:
        """
        第二步：执行工具。
        根据模型决定的 tool_calls，执行对应的本地函数。
        """
        outputs = []
        for tc in tool_calls:
            name = tc["name"]
            args = tc["arguments"]
            tool_id = tc["id"]

            try:
                if name not in {"knowledge_lookup", "structured_extract"}:
                    raise ValueError(f"Unknown tool: {name}")
                args = json.loads(args)
                if not isinstance(args, dict):
                    raise ValueError("Tool arguments must be a JSON object.")
                if name == "knowledge_lookup":
                    keywords = args.get("error_keywords")
                    if not isinstance(keywords, str) or not keywords.strip():
                        raise ValueError("error_keywords must be a non-empty string.")
                    output = self.knowledge_lookup(keywords)
                else:
                    fields = args.get("fields")
                    if (not isinstance(fields, list) or not fields
                            or any(not isinstance(f, str) or not f.strip() for f in fields)):
                        raise ValueError("fields must be a non-empty list of non-empty strings.")
                    output = self.structured_extract(fields)
            except Exception as exc:
                # 一条调用失败也要返回对应结果，不能破坏其余调用的消息配对。
                output = json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False)

            outputs.append({
                "tool_call_id": tool_id,
                "name": name,
                "output": output,
            })

        return outputs

    def knowledge_lookup(self, error_keywords: str) -> str:
        """占位实现：实际项目中应接入知识库或向量检索。"""
        return json.dumps({
            "status": "not_implemented",
            "error_keywords": error_keywords,
            "message": "知识检索占位工具：尚未连接知识库，未执行检索。",
        }, ensure_ascii=False)

    def structured_extract(self, fields: list[str]) -> str:
        """占位实现：没有接收图片或 OCR 文本，不能产生真实字段值。"""
        return json.dumps({
            "status": "not_implemented",
            "requested_fields": fields,
            "values": {field: None for field in fields},
            "message": "结构化提取占位工具：尚未连接 OCR 或解析器，未执行抽取。",
        }, ensure_ascii=False)

    def summarize(self, original_question: str, perception: dict, tool_outputs: list[dict]) -> str:
        """
        第三步：汇总。
        把感知结果和工具执行结果发给模型，生成最终回答。
        """
        # 延续原始图文上下文，并保留 assistant.tool_calls，才能匹配 tool_call_id。
        messages = list(perception["messages"])

        # 添加工具结果（用 tool 角色）
        for out in tool_outputs:
            messages.append({
                "role": "tool",
                "content": out["output"],
                "tool_call_id": out["tool_call_id"],
            })

        # 请求最终总结
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=512,
        )

        answer = response.choices[0].message.content
        if not answer or not answer.strip():
            raise ValueError("Model returned no final answer.")
        return answer

    def run(self, image_path: str, question: str) -> dict[str, object]:
        if not question.strip():
            return {"status": "error", "error": "Question is empty."}

        # Step 1: 感知 + 规划
        perception = self.perceive_and_plan(image_path, question)

        # Step 2: 执行工具（如果有）
        tool_outputs = []
        if perception["tool_calls"]:
            tool_outputs = self.execute_tools(perception["tool_calls"])

        # Step 3: 汇总生成最终回答
        if perception["tool_calls"]:
            final_answer = self.summarize(question, perception, tool_outputs)
        else:
            final_answer = perception["content"]
            if not final_answer or not final_answer.strip():
                raise ValueError("Model returned no answer or tool calls.")

        return {
            "status": "ok",
            "question": question,
            "perception": perception["content"],
            "tool_calls": perception["tool_calls"],
            "tool_outputs": tool_outputs,
            "final_answer": final_answer,
        }


def main() -> int:
    raw_path = os.getenv("IMAGE_PATH", "").strip()
    question = os.getenv("QUESTION", "请分析这张图片，并给出下一步建议。")

    try:
        image_path = str(resolve_image_path(raw_path or None, script_file=__file__))
        agent = MultimodalAgentV2()
        result = agent.run(image_path, question)
    except Exception as exc:
        result = {
            "status": "error",
            "error": str(exc),
            "question": question,
        }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
